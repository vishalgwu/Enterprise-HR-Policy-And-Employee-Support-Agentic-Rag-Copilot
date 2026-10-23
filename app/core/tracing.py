"""In-process tracing: what ran, how long it took, and what it decided.

This is the local half of the observability split this project already made.
LangSmith answers "where did the time and tokens go, and what broke?" for a
development machine with a key; it is off by default, redacts payloads when on,
and needs a third party. Neither of those is a thing a public deployment can
lean on, and an interviewer clicking through the console cannot be asked to open
somebody else's LangSmith project.

So: a span tree built in the request itself, returned on the API response, and
rendered in the UI. No dependency, no network, no key.

**Payload redaction is the default here too, and for the same reason.** A span's
attributes are serialised onto an HTTP response and written to the audit log, so
an attribute holding the employee's question or a retrieved policy chunk is that
text leaving the span and entering somewhere it was not accounted for. Keys in
`PAYLOAD_KEYS` are replaced with their length when `redact_payloads` is on,
which keeps the shape of what happened ("the grader saw 2.1k characters") while
dropping the content. Pass `redact_payloads=False` only for a local debug run.

**Thread safety is not optional.** `ToolRegistry.call_parallel` runs tool calls
on a threadpool and every one of them traces, so the span list is mutated from
several threads at once. Depth is per-thread (a `threading.local`) and the list
is behind a lock; a parallel sibling therefore records at the depth of the
thread that spawned it, which is what makes a parallel fan-out read as a fan-out
rather than as a chain.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PAYLOAD_KEYS",
    "Span",
    "Tracer",
    "null_tracer",
]

# Attribute keys whose values are content rather than metadata: the employee's
# question, a retrieved chunk, a model's reply. Replaced with a character count
# when the tracer is redacting. Listed in one place so a new caller cannot
# invent a synonym that escapes redaction -- if you need a new one, add it here
# rather than passing the text under a key nobody is checking.
PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "args",
        "context",
        "evidence",
        "input",
        "observation",
        "output",
        "payload",
        "prompt",
        "query",
        "question",
        "reply",
        "result",
        "text",
        "thought",
        "value",
    }
)

# Span kinds. Not an enum: the set is open on purpose -- a new stage should be
# able to name itself without editing this module -- but these are the ones the
# UI knows how to colour, and `Tracer.summary` groups by whatever it is given.
KIND_AGENT = "agent"
KIND_CACHE = "cache"
KIND_GUARD = "guard"
KIND_LLM = "llm"
KIND_RETRIEVAL = "retrieval"
KIND_ROUTER = "router"
KIND_STEP = "step"
KIND_TOOL = "tool"


@dataclass(slots=True)
class Span:
    """One timed unit of work.

    `offset_ms` is measured from the tracer's own start rather than from the
    epoch, which is what lets two runs be compared and keeps the numbers small
    enough to read. `depth` is the nesting level at the moment the span opened,
    so the flat list reconstructs a tree without parent pointers -- and without
    a parent pointer there is nothing to get wrong when a span is dropped.
    """

    name: str
    kind: str = KIND_STEP
    depth: int = 0
    offset_ms: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # "ok" | "error" | "degraded"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe shape, with the empty fields left out.

        Omitting empties is not cosmetic: a span list is returned on every chat
        response and written to the audit row, and `"error": null` on forty
        spans is forty keys of nothing.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "depth": self.depth,
            "offset_ms": round(self.offset_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
        }
        if self.error:
            data["error"] = self.error
        if self.attributes:
            data["attributes"] = self.attributes
        return data

    def set(self, **attributes: Any) -> Span:
        """Add attributes after the span opened. Returns self, so it chains.

        Values are *not* redacted here: `Tracer.span` hands out a span whose
        `attributes` dict has already been filtered, and anything added later
        goes through `Tracer._clean` via the closure in `span()`. A span built
        by hand (in a test) is the caller's own business.
        """
        self.attributes.update(attributes)
        return self


class Tracer:
    """Collects spans for one run.

    One tracer per request, not one per process. It holds no clients and starts
    its clock in `__init__`, so building one is free and a run that forgets to
    close a span still produces a readable list.

    `enabled=False` makes every method a no-op that allocates nothing beyond the
    span object itself, so a hot path does not have to branch on whether tracing
    is on.
    """

    def __init__(
        self,
        enabled: bool = True,
        redact_payloads: bool = True,
        run_id: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.redact_payloads = redact_payloads
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._started = time.perf_counter()
        self._spans: list[Span] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    # --- Depth, per thread ---------------------------------------------------
    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @_depth.setter
    def _depth(self, value: int) -> None:
        self._local.depth = max(0, value)

    def adopt_depth(self, depth: int) -> None:
        """Set this thread's starting depth.

        Called by a worker thread so a parallel tool call nests under the span
        that fanned out, rather than appearing at the top level. Without it a
        parallel fan-out reads as four unrelated root spans.
        """
        self._depth = depth

    @property
    def depth(self) -> int:
        """The current nesting level on this thread, for a fan-out to pass on."""
        return self._depth

    # --- Recording -----------------------------------------------------------
    def _clean(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Drop payload content, keeping its size.

        A value that is not a string is kept as-is: an int, a bool or a short
        list under a payload key is a count or a decision, not prose, and
        replacing `{"chunks": 4}` with a character count would be a loss.
        """
        if not self.redact_payloads:
            return dict(attributes)
        cleaned: dict[str, Any] = {}
        for key, value in attributes.items():
            if key in PAYLOAD_KEYS and isinstance(value, str):
                cleaned[f"{key}_chars"] = len(value)
            else:
                cleaned[key] = value
        return cleaned

    def _record(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)

    @contextmanager
    def span(
        self, name: str, kind: str = KIND_STEP, **attributes: Any
    ) -> Iterator[Span]:
        """Time a block of work, recording it even when it raises.

        The span is appended on *entry*, not on exit, so a run that dies
        half-way still shows where it got to -- which is the run whose trace is
        worth having. The duration is filled in afterwards by mutating the
        object already in the list.

        A raised exception is recorded as `status="error"` with the exception
        type and a truncated message, and then re-raised: this decides nothing
        about error handling, it only observes it. The graph's own `guard()`
        remains what turns a failure into a degraded answer.
        """
        if not self.enabled:
            yield Span(name=name, kind=kind)
            return

        span = Span(
            name=name,
            kind=kind,
            depth=self._depth,
            offset_ms=(time.perf_counter() - self._started) * 1000,
            attributes=self._clean(attributes),
        )
        self._record(span)
        self._depth = span.depth + 1
        started = time.perf_counter()
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 - observed, then re-raised
            span.status = "error"
            # The type plus a truncated message: a provider error carries URLs,
            # request ids and sometimes the payload, and this span is serialised
            # onto an HTTP response.
            span.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - started) * 1000
            span.attributes = self._clean(span.attributes)
            self._depth = span.depth

    def event(self, name: str, kind: str = KIND_STEP, **attributes: Any) -> Span:
        """Record something that happened without taking measurable time.

        A cache hit, a routing decision, a guardrail verdict: facts worth a line
        in the trace that are not a block of work to be timed.
        """
        span = Span(
            name=name,
            kind=kind,
            depth=self._depth,
            offset_ms=(time.perf_counter() - self._started) * 1000,
            attributes=self._clean(attributes),
        )
        if self.enabled:
            self._record(span)
        return span

    def degraded(self, name: str, reason: str, kind: str = KIND_STEP) -> Span:
        """Record a call that failed and was absorbed by a fallback.

        Distinct from `event` because an outage and a genuine absence of
        evidence produce the same answer, and only the trace tells them apart --
        the same reason `guard()` writes into the graph's trace.
        """
        span = self.event(name, kind=kind, reason=reason)
        span.status = "degraded"
        return span

    # --- Reading -------------------------------------------------------------
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def to_list(self) -> list[dict[str, Any]]:
        """The span tree, flat and JSON-safe, in the order things started."""
        return [span.to_dict() for span in self.spans()]

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def summary(self) -> dict[str, Any]:
        """Totals an operator actually asks for, grouped by kind.

        `self_ms` per kind rather than total duration, because nested spans
        double-count: an `agent` span wrapping four `llm` spans would otherwise
        report the LLM time twice and make the percentages sum past 100. The
        self time of a span is its duration minus the duration of the spans
        directly inside it, which is what "where did the time go" means.
        """
        spans = self.spans()
        by_kind: dict[str, dict[str, Any]] = {}
        for index, span in enumerate(spans):
            child_ms = 0.0
            for other in spans[index + 1 :]:
                if other.depth <= span.depth:
                    break
                if other.depth == span.depth + 1:
                    child_ms += other.duration_ms
            bucket = by_kind.setdefault(span.kind, {"count": 0, "self_ms": 0.0})
            bucket["count"] += 1
            bucket["self_ms"] += max(0.0, span.duration_ms - child_ms)

        for bucket in by_kind.values():
            bucket["self_ms"] = round(bucket["self_ms"], 2)

        return {
            "run_id": self.run_id,
            "spans": len(spans),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "errors": sum(1 for s in spans if s.status == "error"),
            "degraded": sum(1 for s in spans if s.status == "degraded"),
            "by_kind": by_kind,
            "redacted": self.redact_payloads,
        }


def null_tracer() -> Tracer:
    """A tracer that records nothing, for a caller that has no use for one.

    Returned rather than `None` so every call site can write `tracer.span(...)`
    unconditionally instead of guarding each one.
    """
    return Tracer(enabled=False)
