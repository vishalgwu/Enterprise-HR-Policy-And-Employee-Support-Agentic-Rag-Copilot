"""The tool registry: what the agent may do, and what it costs to let it.

A graph calls the same three things in the same order every time, so the
wiring *is* the policy. An agent that chooses its own tools has no wiring, so
every property the wiring used to guarantee has to be declared somewhere else.
That somewhere is `ToolSpec`: how long a call may take, whether a failure may
be retried, whether a retry is safe at all, who is allowed to ask, and whether
a person has to say yes first. The registry enforces them so a handler cannot
forget to.

**This module is agent-layer and must not import `app/services/*`.** Every
service arrives as a duck-typed injected argument, and each is used through a
small interface rather than a type:

    idempotency   `.fingerprint(*parts)`, `.claim(key, action=)` returning an
                  object with `.is_new` / `.is_replay` / `.result`,
                  `.complete(key, result)`, `.release(key)`
    approvals     `.request(action=, reason=, question=, payload=,
                  conversation_id=, requested_by=)` returning an id or None
    authorize     a callable `(principal, action) -> bool`
    budget        `.allows(kind)`, `.charge_tool()`, `.stop_reason`
    tracer        `app.core.tracing.Tracer`, or `null_tracer()`

The parameter models and the pure built-in handlers live here too, so the
module is testable with no services at all.

**A tool's error text is ours, never a provider's.** A `ToolResult.error` is
rendered into the next prompt and returned on the HTTP response, and this
process holds API keys; a provider exception carries URLs, request ids and
sometimes the offending payload. Handlers signal an expected failure by raising
`ToolError` with text they wrote; anything else is logged with its traceback and
reported as a type name.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings, get_logger, get_settings
from app.core.tracing import Tracer, null_tracer

log = get_logger("tools")

__all__ = [
    "CalcArgs",
    "EscalateArgs",
    "NoteArgs",
    "SearchArgs",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "calculate",
    "register_builtins",
    "safe_eval",
]

# How much of a tool's output the model is shown. An observation is rendered
# into every later prompt in the loop, so an unbounded one makes each step more
# expensive than the last -- the same reason `normalise_history` caps turns.
OBSERVATION_LIMIT = 4000


class ToolError(Exception):
    """A handler signalling a failure it expected and has words for.

    Distinct from every other exception because its message is safe to show:
    it was written here, in this repository, for a reader. Anything else is
    logged and reported as a type name.
    """


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool, and every property the registry needs to run it safely.

    `parameters` is a pydantic model and its JSON schema is the contract: it is
    what the model is shown and what the arguments are validated against, so
    the description the model reads and the validation it is held to cannot
    drift apart.

    `retries` is honoured only when `idempotent` -- retrying a call that may
    have already taken effect is how one request becomes two. `side_effecting`
    is the stronger statement: it additionally means the call is fingerprinted
    into the idempotency ledger and never run in parallel with its siblings.
    """

    name: str
    description: str
    parameters: type[BaseModel]
    handler: Callable[..., Any]
    requires: str = "tool.run"
    timeout_seconds: float = 20.0
    retries: int = 0
    idempotent: bool = True
    side_effecting: bool = False
    needs_approval: bool = False
    cost_hint: str = "free"  # "free" | "cheap" | "network" | "llm"

    def schema(self) -> dict[str, Any]:
        """The shape a function-calling API wants."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }

    def render(self) -> str:
        """One compact line for `{tools}` in `REACT_PROMPT`.

        Compact on purpose: the catalogue is re-rendered into the prompt on
        every step of the loop, so a full JSON schema per tool is paid for once
        per turn. The flags are included because a model told in advance that a
        tool needs approval stops choosing it for things that do not.
        """
        schema = self.parameters.model_json_schema()
        required = set(schema.get("required") or ())
        arguments = [
            f"{name}: {spec.get('type', 'any')}"
            if name in required
            else f"{name}?: {spec.get('type', 'any')}"
            for name, spec in (schema.get("properties") or {}).items()
        ]
        flags = []
        if self.needs_approval:
            flags.append("needs human approval")
        if self.side_effecting:
            flags.append("side-effecting")
        suffix = f" [{'; '.join(flags)}]" if flags else ""
        return f"- {self.name}({', '.join(arguments)}): {self.description}{suffix}"


@dataclass(slots=True)
class ToolResult:
    """What one `call` produced, including how it failed.

    The flags are separate booleans rather than a status enum because they are
    not alternatives to the caller: `denied` is something the agent may route
    around, `needs_approval` ends the run, and `replayed` is a success that did
    no work. A single status field would make each of those a string comparison
    somebody can get wrong.
    """

    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    replayed: bool = False
    needs_approval: bool = False
    approval_id: int | None = None
    denied: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "attempts": self.attempts,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.error:
            data["error"] = self.error
        if self.replayed:
            data["replayed"] = True
        if self.needs_approval:
            data["needs_approval"] = True
            data["approval_id"] = self.approval_id
        if self.denied:
            data["denied"] = True
        return data

    def observation(self) -> str:
        """What the model sees next turn.

        Every branch says what to do next, not only what happened. A model told
        "failed" tries the same thing again; a model told "failed, try another
        tool" changes tack.
        """
        if self.needs_approval:
            return (
                f"{self.name} cannot run without a person approving it "
                f"(request {self.approval_id}). Stop here and say so plainly."
            )
        if self.denied:
            return (
                f"{self.name} is not available to this caller. "
                "Answer from another source or say you cannot."
            )
        if not self.ok:
            return f"{self.name} failed: {self.error}"
        return _render_value(self.value)


def _render_value(value: Any) -> str:
    """Flatten a handler's return into prompt text.

    A dict carrying `text` is a search result that has already rendered its own
    evidence (see `app/agent/react.py` for the shape), and using that rather
    than the whole dict keeps a page of chunk metadata out of the prompt.
    """
    if value is None:
        return "(the tool returned nothing)"
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bool, int, float)):
        text = str(value)
    elif isinstance(value, dict) and isinstance(value.get("text"), str):
        text = value["text"]
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if len(text) > OBSERVATION_LIMIT:
        text = text[:OBSERVATION_LIMIT] + " ...(truncated)"
    return text or "(the tool returned nothing)"


def _argument_error(name: str, exc: ValidationError) -> str:
    """Name the offending fields and *only* the fields.

    `str(exc)` echoes the input value back -- pydantic prints
    `input_value=...` -- and the input here is whatever a language model put in
    `action_input` on behalf of an employee who may have pasted a key, an id or
    a salary into the chat box. This message is rendered into the next prompt
    and returned on the HTTP response, so it is built from `exc.errors()` field
    locations and error *types*, both of which are schema facts rather than
    data. The same failure, with the same cause, is recorded in `Settings`:
    a model-level pydantic validator there once rendered every API key into an
    error message.
    """
    parts = []
    for error in exc.errors():
        location = ".".join(str(piece) for piece in error.get("loc", ())) or "(root)"
        parts.append(f"{location} ({error.get('type', 'invalid')})")
    fields = "; ".join(parts) or "unknown"
    return f"{name} was called with invalid arguments: {fields}"


class ToolRegistry:
    """Every tool the agent may call, and the rules it is held to.

    One registry per request is unnecessary -- the specs are immutable -- but
    the *services* differ per request (a budget, a tracer, a principal), so the
    registry takes them at construction and a caller that needs a different
    budget builds a different registry. That is the same injection the rest of
    this codebase uses, carried into the tool layer.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        budget: Any = None,
        idempotency: Any = None,
        approvals: Any = None,
        authorize: Callable[[Any, str], bool] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._tracer = tracer or null_tracer()
        self._budget = budget
        self._idempotency = idempotency
        self._approvals = approvals
        # None means allow: a registry built with no authoriser is a CLI or a
        # test, and making those pass a permit-everything callable would be
        # noise with a chance of being got wrong.
        self._authorize = authorize
        self._tools: dict[str, ToolSpec] = {}
        self._counters: dict[str, int] = {}
        self._counter_lock = threading.Lock()

    # --- Registration --------------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        """Add a tool. Refuses a duplicate name.

        Raising here is deliberate and is the one place in this module that
        does: registration happens at construction, not on a request, and a
        second tool silently shadowing the first would mean the catalogue the
        model reads and the handler that runs describe different things.
        """
        if spec.name in self._tools:
            raise ValueError(f"a tool named {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: type[BaseModel],
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form, for a handler defined beside its registration."""

        def decorate(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    **kwargs,
                )
            )
            return handler

        return decorate

    # --- Reading -------------------------------------------------------------
    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def _permitted(self, principal: Any) -> list[ToolSpec]:
        return [
            spec
            for spec in self._tools.values()
            if self._authorize is None or self._authorised(principal, spec)
        ]

    def names(self, principal: Any = None) -> list[str]:
        return [spec.name for spec in self._permitted(principal)]

    def catalogue(self, principal: Any = None) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._permitted(principal)]

    def render_catalogue(self, principal: Any = None) -> str:
        """The catalogue as prompt text, filtered to what this caller may use.

        Filtered rather than listed in full because a prompt naming a tool the
        registry would refuse is a model being invited to fail: it spends a step
        choosing it, gets a refusal, and has learned nothing it could not have
        been told up front.
        """
        permitted = self._permitted(principal)
        if not permitted:
            return "(no tools are available to you)"
        return "\n".join(spec.render() for spec in permitted)

    def stats(self) -> dict[str, Any]:
        with self._counter_lock:
            counters = dict(self._counters)
        return {"registered": len(self._tools), **counters}

    # --- Calling -------------------------------------------------------------
    def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        principal: Any = None,
        conversation_id: str | None = None,
        question: str = "",
    ) -> ToolResult:
        """Run one tool, in this order:

        1. **Unknown tool** -- answer with the names that *are* available. A
           model that hallucinated a tool name recovers from being told the real
           ones; it cannot recover from a stack trace.
        2. **Budget** -- if the tool-call ceiling is spent, refuse with the
           budget's own `stop_reason`. Nothing is called and nothing is charged.
        3. **Authorization** -- `authorize(principal, spec.requires)`. A refusal
           names the action and says nothing about what the tool would have
           done, because the refusal itself is returned to the caller.
        4. **Argument validation** -- against `spec.parameters`. The message
           names fields only; see `_argument_error`.
        5. **Approval** -- a tool needing a person records a request and returns
           without running. A store that answers `None` has not recorded it, so
           that refuses too: fail closed.
        6. **Idempotency** -- a side-effecting tool claims a fingerprint of its
           validated arguments. A replay returns the first call's result; a
           claim already in flight refuses rather than racing it.
        7. **Run**, with a timeout, retrying only an idempotent tool.
        8. Back off exponentially with jitter between attempts.
        9. Charge the budget once per `call`, not once per attempt.
        10. Complete the claim on success; release it on a terminal failure.
        11. Record a span either way.

        The order is the point. Validation before approval means a malformed
        request is not queued for a human; approval before the ledger means an
        unapproved call never reserves a fingerprint it will not use; the charge
        after both means a refusal is free.
        """
        started = time.perf_counter()
        with self._tracer.span(f"tool.{name}", kind="tool") as span:
            result = self._dispatch(
                name, arguments, principal, conversation_id, question
            )
            result.duration_ms = (time.perf_counter() - started) * 1000
            span.set(attempts=result.attempts, ok=result.ok)
            if result.replayed:
                span.set(replayed=True)
            if not result.ok and not result.denied and not result.needs_approval:
                self._tracer.degraded(
                    f"tool.{name}", reason=result.error or "failed", kind="tool"
                )
            return result

    def _dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        principal: Any,
        conversation_id: str | None,
        question: str,
    ) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            self._bump("unknown")
            available = ", ".join(self.names(principal)) or "none"
            return ToolResult(
                name=name,
                ok=False,
                error=f"no such tool: {name!r}. Available tools: {available}",
            )

        if self._budget is not None and not self._budget.allows("tool_calls"):
            self._bump("refused")
            reason = (
                getattr(self._budget, "stop_reason", None)
                or "the tool-call budget for this question is spent"
            )
            return ToolResult(name=name, ok=False, error=reason)

        if self._authorize is not None and not self._authorised(principal, spec):
            self._bump("denied")
            return ToolResult(
                name=name,
                ok=False,
                denied=True,
                error=f"not authorised for {spec.requires}",
            )

        try:
            validated = spec.parameters.model_validate(arguments or {})
        except ValidationError as exc:
            self._bump("invalid")
            return ToolResult(name=name, ok=False, error=_argument_error(name, exc))
        payload = validated.model_dump()

        if spec.needs_approval:
            return self._request_approval(
                spec, payload, principal, conversation_id, question
            )

        claim = None
        if spec.side_effecting and self._idempotency is not None:
            claim = self._claim(spec, payload)
            if claim is None:
                self._bump("refused")
                return ToolResult(
                    name=name,
                    ok=False,
                    error=(
                        f"{name} could not be reserved against repeat execution, "
                        "so it was not run"
                    ),
                )
            if claim.is_replay:
                self._charge()
                self._bump("replayed")
                return ToolResult(
                    name=name, ok=True, value=claim.result, replayed=True
                )
            if not claim.is_new:
                self._bump("in_flight")
                return ToolResult(
                    name=name,
                    ok=False,
                    error=(
                        f"{name} is already running with these arguments; "
                        "a second one was not started"
                    ),
                )

        # Once per call, deliberately, and not once per attempt: a retry is the
        # registry's own decision about an unreliable provider, and billing the
        # agent for it would make a flaky Tavily look like an expensive agent
        # and shorten the run that needed the budget most.
        self._charge()
        return self._run(spec, payload, claim)

    def _authorised(self, principal: Any, spec: ToolSpec) -> bool:
        if self._authorize is None:
            return True
        try:
            return bool(self._authorize(principal, spec.requires))
        except Exception:  # noqa: BLE001 - an authoriser that fails denies
            log.exception("[tool.%s] the authoriser failed; refusing", spec.name)
            return False

    def _request_approval(
        self,
        spec: ToolSpec,
        payload: dict[str, Any],
        principal: Any,
        conversation_id: str | None,
        question: str,
    ) -> ToolResult:
        """Queue the call for a person and return without running it."""
        if self._approvals is None:
            self._bump("refused")
            return ToolResult(
                name=spec.name,
                ok=False,
                error=(
                    f"{spec.name} needs human approval and no approval queue is "
                    "configured on this deployment"
                ),
            )
        try:
            recorded = self._approvals.request(
                action=spec.name,
                reason=spec.description,
                question=question,
                payload=payload,
                conversation_id=conversation_id,
                requested_by=str(principal or "employee"),
            )
        except Exception:  # noqa: BLE001
            log.exception("[tool.%s] recording an approval request failed", spec.name)
            recorded = None
        # The store may answer with the queued row or with its id; both mean
        # "recorded", and only `None` means it is not in the queue.
        approval_id = getattr(recorded, "id", recorded)
        if approval_id is None:
            # Not recorded means no person will ever see it, so running anyway
            # would be running unapproved. Fail closed.
            self._bump("refused")
            return ToolResult(
                name=spec.name,
                ok=False,
                error=(
                    f"{spec.name} needs human approval and the request could not "
                    "be recorded, so it was not run"
                ),
            )
        self._bump("approval")
        return ToolResult(
            name=spec.name,
            ok=False,
            needs_approval=True,
            approval_id=int(approval_id),
            error=f"{spec.name} is waiting for human approval",
        )

    def _claim(self, spec: ToolSpec, payload: dict[str, Any]) -> Any | None:
        try:
            key = self._idempotency.fingerprint(spec.name, payload)
            return self._idempotency.claim(key, action=spec.name)
        except Exception:  # noqa: BLE001 - an unusable ledger refuses the call
            log.exception("[tool.%s] claiming an idempotency key failed", spec.name)
            return None

    def _run(
        self, spec: ToolSpec, payload: dict[str, Any], claim: Any | None
    ) -> ToolResult:
        timeout = self._timeout(spec)
        allowed = 1 + (self._retries(spec) if spec.idempotent else 0)
        error: str | None = None
        timed_out = False

        for attempt in range(allowed):
            try:
                value = self._invoke(spec, payload, timeout)
            except TimeoutError:
                timed_out = True
                error = f"{spec.name} did not answer within {timeout:g}s"
                log.warning("[tool.%s] timed out after %.1fs", spec.name, timeout)
            except ToolError as exc:
                timed_out = False
                # Safe to surface: a handler raising ToolError wrote this text.
                error = str(exc)
                log.info("[tool.%s] expected failure: %s", spec.name, error)
            except Exception as exc:  # noqa: BLE001 - never a provider's words
                timed_out = False
                log.exception("[tool.%s] failed", spec.name)
                error = f"{spec.name} failed with {type(exc).__name__}"
            else:
                if claim is not None:
                    self._complete(claim, value)
                self._bump("ok")
                return ToolResult(
                    name=spec.name, ok=True, value=value, attempts=attempt + 1
                )
            if attempt + 1 < allowed:
                self._bump("retry")
                time.sleep(self._backoff(attempt))

        if claim is not None and not timed_out:
            self._release(claim)
        elif claim is not None:
            # A timeout bounds our *wait*, not the work -- the worker thread is
            # still running and may yet perform the side effect. Releasing the
            # claim here would let a retry duplicate it, so the claim is left in
            # flight and the ledger's stall window decides when it is safe to
            # re-win.
            log.warning(
                "[tool.%s] timed out with a claim held; leaving it in flight so "
                "a retry cannot duplicate the side effect",
                spec.name,
            )
        self._bump("failed")
        return ToolResult(name=spec.name, ok=False, error=error, attempts=allowed)

    def _invoke(self, spec: ToolSpec, payload: dict[str, Any], timeout: float) -> Any:
        """Call the handler, bounding how long we wait for it.

        A future's `result(timeout=...)` leaves the worker thread running --
        Python cannot kill a thread, and there is no safe general way to
        interrupt one. So this bounds *the caller's wait*, not the work: the
        handler may still be mid-request when we have given up on it. That is
        why `tool_parallelism` is deliberately small (a timed-out call keeps its
        thread until it finishes), and why a handler is still expected to pass
        its own timeout to its HTTP client rather than relying on this.

        `shutdown(wait=False)` for the same reason: waiting would re-join the
        very thread the timeout exists to stop waiting for.
        """
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"tool-{spec.name}"
        )
        try:
            return executor.submit(lambda: spec.handler(**payload)).result(
                timeout=timeout
            )
        finally:
            executor.shutdown(wait=False)

    # --- Parallel fan-out ----------------------------------------------------
    def call_parallel(
        self,
        calls: Sequence[tuple[str, dict[str, Any] | None]],
        *,
        principal: Any = None,
        max_workers: int | None = None,
    ) -> list[ToolResult]:
        """Run several tools at once, answering in the order they were given.

        **In call order, not completion order.** The results are rendered into
        the next prompt as observations, and a model reading them out of order
        attributes the wrong answer to the wrong question -- which for an HR
        answer means a correct sentence about the wrong policy. Completion order
        is also nondeterministic, so the bug would appear in roughly one run in
        three and never in a test.

        **A side-effecting tool is not parallelised.** It runs on the calling
        thread, before the fan-out. Two side effects racing on the same
        idempotency key is exactly the case the ledger cannot fix: both claims
        are decided before either completes, so one wins and the other refuses
        the employee's request for no reason the employee can see.

        One call raising does not lose the others: a worker that escapes its own
        error handling still answers with a failed `ToolResult`.
        """
        requested = [
            (index, name, arguments)
            for index, (name, arguments) in enumerate(calls)
        ]
        results: dict[int, ToolResult] = {}
        fan_out: list[tuple[int, str, dict[str, Any] | None]] = []

        for index, name, arguments in requested:
            spec = self.get(name)
            if spec is not None and spec.side_effecting:
                self._tracer.event("tool.sequential", kind="tool", tool=name)
                results[index] = self.call(name, arguments, principal=principal)
            else:
                fan_out.append((index, name, arguments))

        if fan_out:
            workers = max(
                1,
                min(
                    len(fan_out),
                    int(max_workers or getattr(self._settings, "tool_parallelism", 4)),
                ),
            )
            # Read on this thread and passed in, because `tracer.depth` is
            # per-thread and a worker reading it would find zero.
            parent_depth = self._tracer.depth
            executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="tool-fanout"
            )
            try:
                futures = {
                    executor.submit(
                        self._worker, parent_depth, name, arguments, principal
                    ): index
                    for index, name, arguments in fan_out
                }
                for future, index in futures.items():
                    results[index] = future.result()
            finally:
                executor.shutdown(wait=False)

        return [results[index] for index, _, _ in requested]

    def _worker(
        self,
        parent_depth: int,
        name: str,
        arguments: dict[str, Any] | None,
        principal: Any,
    ) -> ToolResult:
        self._tracer.adopt_depth(parent_depth)
        try:
            return self.call(name, arguments, principal=principal)
        except Exception as exc:  # noqa: BLE001 - one call must not lose the rest
            log.exception("[tool.%s] escaped its own error handling", name)
            return ToolResult(
                name=name, ok=False, error=f"{name} failed with {type(exc).__name__}"
            )

    # --- Small helpers -------------------------------------------------------
    def _timeout(self, spec: ToolSpec) -> float:
        """The spec's timeout, clamped by the deployment's ceiling.

        `TOOL_TIMEOUT_SECONDS` is a ceiling rather than a default so a tool
        that declares a generous one of its own still cannot hang a request on
        a deployment that will not wait that long.
        """
        ceiling = float(getattr(self._settings, "tool_timeout_seconds", 20.0))
        return min(float(spec.timeout_seconds), ceiling)

    def _retries(self, spec: ToolSpec) -> int:
        ceiling = int(getattr(self._settings, "tool_retries", 2))
        return max(0, min(int(spec.retries), ceiling))

    def _backoff(self, attempt: int) -> float:
        """Exponential, with jitter.

        The jitter is not decoration. Several calls that failed together --
        which is what a rate limit or a brief outage produces -- would otherwise
        retry at the same instant and reproduce the burst that caused the
        failure, so the second attempt fails for the same reason as the first.
        """
        base = float(getattr(self._settings, "tool_backoff_seconds", 0.4))
        delay = base * (2**attempt)
        return delay + random.random() * delay * 0.25

    def _charge(self) -> None:
        if self._budget is None:
            return
        try:
            self._budget.charge_tool()
        except Exception:  # noqa: BLE001 - accounting must not cost an answer
            log.exception("Charging the tool budget failed")

    def _complete(self, claim: Any, value: Any) -> None:
        try:
            self._idempotency.complete(claim.key, value)
        except Exception:  # noqa: BLE001
            log.exception("Completing an idempotency claim failed")

    def _release(self, claim: Any) -> None:
        try:
            self._idempotency.release(claim.key)
        except Exception:  # noqa: BLE001
            log.exception("Releasing an idempotency claim failed")

    def _bump(self, counter: str) -> None:
        with self._counter_lock:
            self._counters[counter] = self._counters.get(counter, 0) + 1


# --- Parameter models --------------------------------------------------------
# Bounds on every field, because these validate what a language model wrote.
# `min_length` catches the empty string a model emits when it has decided to
# call a tool and has nothing to put in it; the maxima are a cost control -- a
# 50k-character "query" is a prompt being smuggled into a retrieval.


class SearchArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=500,
        description="What to look for, as a self-contained question or phrase.",
    )
    k: int = Field(
        default=4, ge=1, le=10, description="How many passages to return."
    )


class NoteArgs(BaseModel):
    text: str = Field(
        min_length=1, max_length=2000, description="The note to keep for later."
    )
    kind: str = Field(
        default="note", description="plan | finding | question | note."
    )


class EscalateArgs(BaseModel):
    reason: str = Field(
        min_length=3,
        max_length=500,
        description="Why a person has to decide this one, in one sentence.",
    )


class CalcArgs(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Arithmetic only: numbers, + - * / // % ** and parentheses. "
            "No names, no functions."
        ),
    )


# --- The one place a model's output is executed ------------------------------

# The operators an HR calculation needs and nothing else.
_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# `a ** b` with a large b is a denial of service in pure arithmetic: no name is
# resolved and no function is called, yet `9 ** 9 ** 9` occupies a core for
# longer than anybody will wait. Both ceilings are needed -- the exponent bound
# stops one big power, the digit bound stops a nest of small ones compounding.
MAX_EXPONENT = 64
MAX_RESULT_DIGITS = 100


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic, and refuse everything else.

    HR questions contain arithmetic -- "I have 12 days left and take 5, what is
    my carry-over?" -- and a language model doing arithmetic in prose is the
    classic wrong-number failure this product must not have. So the model is
    given a calculator instead of being trusted to count.

    **An AST walker rather than `eval` with a restricted globals dict.** That
    second approach has been escaped repeatedly through dunder traversal:
    `().__class__.__bases__[0].__subclasses__()` reaches the entire class
    hierarchy from a literal with no names in scope at all, and from there to
    something that opens a file or a socket. Walking the tree and refusing every
    node type that is not a number or an arithmetic operator removes the
    traversal rather than trying to block its destinations, and there is no list
    of forbidden names to keep up to date.

    Raises `ToolError` for anything refused, so the registry shows the model
    words it can act on rather than a type name.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            "that is not an arithmetic expression; use numbers and + - * / "
            "() only"
        ) from exc
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        # Numbers only. A string constant is refused because `"a" * 10**9` is
        # arithmetic by node type and a memory exhaustion by effect.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numbers are allowed in a calculation")
        return node.value

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("that operator is not allowed in a calculation")
        return handler(_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("that operator is not allowed in a calculation")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        try:
            return handler(left, right)
        except ZeroDivisionError as exc:
            raise ToolError("that calculation divides by zero") from exc
        except OverflowError as exc:
            raise ToolError("that calculation is too large to compute") from exc

    # Everything else, by node type rather than by name: `ast.Name` (there are
    # no variables to read), `ast.Call` (no function is callable from here),
    # `ast.Attribute` (attribute access is the dunder-traversal escape),
    # `ast.Subscript`, comprehensions, lambdas, walrus, f-strings. Refusing by
    # type means a new Python syntax feature arrives refused rather than
    # arriving allowed.
    raise ToolError(
        f"{type(node).__name__} is not allowed in a calculation; arithmetic only"
    )


def _check_power(base: Any, exponent: Any) -> None:
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError(f"exponents above {MAX_EXPONENT} are not allowed")
    if base in (0, 1, -1) or exponent == 0:
        return
    if abs(exponent) * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise ToolError("that calculation would produce an unreasonably large number")


def calculate(expression: str) -> dict[str, Any]:
    """The calculator handler. Pure: no services, no network, no clock."""
    value = safe_eval(expression)
    rendered = f"{value:g}"
    return {
        "expression": expression,
        "result": value,
        "text": f"{expression} = {rendered}",
    }


CALCULATE = ToolSpec(
    name="calculate",
    description=(
        "Work out an arithmetic expression exactly. Use this for any number "
        "the employee asks you to compute rather than doing it in your head."
    ),
    parameters=CalcArgs,
    handler=calculate,
    timeout_seconds=2.0,
    retries=0,
    cost_hint="free",
)


def register_builtins(registry: ToolRegistry) -> ToolRegistry:
    """Register the tools that need nothing injected. Returns the registry.

    Only the calculator, which is the whole point: everything else an HR
    copilot does touches a corpus, a search provider or a database, and those
    are built at the services layer where their clients live.
    """
    registry.register(CALCULATE)
    return registry
