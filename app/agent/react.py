"""A minimal ReAct loop: think, act, observe, repeat until the answer is ready.

The graph in `app/agent/graph.py` decides the sequence once, at build time, and
every question walks the same eleven nodes. That is the right shape for the
questions it was built for and the wrong one for a question that needs two
lookups and a sum: the graph has no edge for "retrieve, then work out the
arithmetic, then retrieve again". This loop has no fixed sequence at all -- the
model picks one tool per turn from `ToolRegistry.render_catalogue` and sees what
it returned before picking the next.

What that buys is the question the graph cannot answer. What it costs is every
guarantee the wiring used to provide, which is why this module is mostly
guarantees:

- a ceiling on the turns, from `Budget`, with a composed partial answer rather
  than an exception when it is reached,
- interception of a repeated identical tool call, because a model that repeats
  itself will keep repeating itself until the budget stops it,
- `source_used` derived from the tools that actually succeeded rather than from
  what the model says it used,
- and the same no-raise contract every graph node has: a failure produces
  `GENERATION_FAILED`, never a traceback.

**It never answers from the model's own knowledge by accident.** Every path
that cannot get a decision out of the model either goes and looks something up
or composes from observations already gathered. An HR answer invented from
training data is the single worst output this product can produce, and it is
indistinguishable from a good one to the person reading it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.agent.budgets import Budget
from app.agent.decisions import failed_generation
from app.agent.prompts import (
    BUDGET_STOPPED_NOTE,
    ESCALATED_ANSWER,
    GENERATION_FAILED,
    REACT_FORCED_ANSWER_PROMPT,
    REACT_PROMPT,
)
from app.agent.schemas import ReActDecision, normalise_history
from app.agent.tools import ToolRegistry, ToolResult
from app.core.config import Settings, get_logger, get_settings
from app.core.tracing import Tracer, null_tracer
from app.rag.clients import get_llm, is_rate_limit, message_text

log = get_logger("react")

__all__ = ["ReActAgent", "ReActResult", "ReActStep"]

# The tool names that establish provenance. They are the contract between this
# loop and the services layer that builds the tools: a renamed tool silently
# stops contributing to `source_used`, which would make an answer grounded in
# the private KB report itself as `direct`. Change both together.
KB_TOOLS = frozenset({"kb_search", "policy_search", "kb_lookup"})
WEB_TOOLS = frozenset({"web_search", "public_search"})
EVIDENCE_TOOLS = KB_TOOLS | WEB_TOOLS

# What the safe default reaches for on the first step. Resolved against the
# registry rather than assumed, so a deployment that registers a differently
# named search tool still gets a lookup instead of nothing.
DEFAULT_SEARCH_TOOL = "kb_search"

# Recovery accepts these spellings for each field. A model that has already
# failed to call the bound tool is not going to be precise about key names.
_DECISION_ALIASES: dict[str, tuple[str, ...]] = {
    "thought": ("thought", "reasoning"),
    "action": ("action", "tool", "tool_name"),
    "action_input": ("action_input", "tool_input", "arguments", "args", "input"),
    "final_answer": ("final_answer", "answer", "final"),
}


@dataclass(slots=True)
class ReActStep:
    """One turn of the loop, as the model will read it back next turn.

    `ok` is about the *tool*, not the step: a search that came back empty is a
    step that worked and an observation worth having.
    """

    index: int
    thought: str
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    ok: bool = True

    def render(self) -> str:
        """The transcript format, which is also what the prompt asks for."""
        lines = [f"Thought: {self.thought or '(none)'}"]
        if self.action:
            arguments = json.dumps(self.action_input or {}, default=str)
            lines.append(f"Action: {self.action}({arguments})")
            lines.append(f"Observation: {self.observation or '(nothing)'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"index": self.index, "thought": self.thought}
        if self.action:
            data["action"] = self.action
            data["action_input"] = self.action_input
            data["observation"] = self.observation
            data["ok"] = self.ok
        return data


@dataclass(slots=True)
class ReActResult:
    """Everything one run produced, in the shape the API layer answers with.

    `stopped_on_budget` is beside the answer rather than inferable from it
    because a partial answer that does not say it is partial is
    indistinguishable from a complete one that happened to be short -- and for
    an entitlement question that difference is the whole thing.
    """

    question: str
    answer: str
    steps: list[ReActStep]
    source_used: str
    citations: list[dict[str, str]]
    stopped_on_budget: bool
    budget: dict[str, Any]
    trace: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "source_used": self.source_used,
            "citations": self.citations,
            "stopped_on_budget": self.stopped_on_budget,
            "budget": self.budget,
            "trace": self.trace,
        }


class ReActAgent:
    """Drives `REACT_PROMPT` against a `ToolRegistry` until it has an answer.

    Every dependency is injected with a `None` default, like `build_nodes`:
    that is how the whole of this is tested offline, including the paths a real
    question only reaches when something goes wrong twice in a row.
    """

    def __init__(
        self,
        llm: Any = None,
        registry: ToolRegistry | None = None,
        *,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        budget: Any = None,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or get_llm(settings=self._settings)
        self._tracer = tracer or null_tracer()
        self._budget = budget or Budget.from_settings(self._settings)
        self._registry = registry or ToolRegistry(
            settings=self._settings, tracer=self._tracer, budget=self._budget
        )
        self._max_steps = int(
            max_steps or getattr(self._settings, "max_agent_steps", 8)
        )
        self._verbose = verbose
        # Built once, like the graph's router and grader: binding the structured
        # output per turn would construct a runnable per step for nothing.
        self._decider = REACT_PROMPT | self._llm.with_structured_output(ReActDecision)

    # --- Public --------------------------------------------------------------
    def run(
        self,
        question: str,
        *,
        history: Any = None,
        principal: Any = None,
        conversation_id: str | None = None,
    ) -> ReActResult:
        """Answer one question. Never raises.

        The same contract every graph node has, and for the same reason: an
        employee asking about parental leave gets a sentence saying the model
        was unreachable, not a 500.
        """
        self._budget.start()
        try:
            return self._run(question, history, principal, conversation_id)
        except Exception:  # noqa: BLE001 - the loop must never abort a request
            log.exception("The ReAct loop failed; reporting an error")
            return ReActResult(
                question=question,
                answer=GENERATION_FAILED,
                steps=[],
                source_used="error",
                citations=[],
                stopped_on_budget=False,
                budget=self._budget.report(),
                trace=["ReAct: the loop failed"],
            )

    # --- The loop ------------------------------------------------------------
    def _run(
        self,
        question: str,
        history: Any,
        principal: Any,
        conversation_id: str | None,
    ) -> ReActResult:
        turns = normalise_history(history)
        catalogue = self._registry.render_catalogue(principal)
        trace: list[str] = []
        steps: list[ReActStep] = []
        citations: list[dict[str, str]] = []
        # (action, canonical arguments) -> the observation it produced.
        seen: dict[tuple[str, str], str] = {}
        succeeded: list[str] = []
        attempted: list[str] = []
        answer = ""
        escalated = False
        stopped = False

        with self._tracer.span("react", kind="agent", question=question):
            for index in range(self._max_steps):
                if not self._budget.allows("steps"):
                    stopped = True
                    trace += self._note(
                        "ReAct", self._budget.stop_reason or "out of budget"
                    )
                    break
                self._budget.charge_step()

                with self._tracer.span(f"react.step.{index}", kind="step"):
                    decision = self._decide(
                        question, catalogue, self._transcript(turns, steps), index
                    )

                step = ReActStep(index=index, thought=(decision.thought or "").strip())
                steps.append(step)
                action = (decision.action or "").strip()
                final = (decision.final_answer or "").strip()

                # Both set: prefer the action. An agent that chose to look
                # something up has told you its answer is not ready, and the
                # answer it wrote alongside was written before the evidence
                # arrived.
                if action:
                    step.action = action
                    step.action_input = dict(decision.action_input or {})
                    key = (action, _canonical_args(step.action_input))
                    if key in seen:
                        # A loop budget stops this eventually, but it stops it
                        # by spending the budget -- and the model is far more
                        # likely to change tack when told the call already ran
                        # than when silently re-served the same observation.
                        step.observation = (
                            "That exact call has already run. It returned: "
                            f"{seen[key]}\nChange the arguments or use a "
                            "different tool."
                        )
                        step.ok = False
                        trace += self._note(
                            "ReAct", f"{action} repeated with the same arguments"
                        )
                        continue

                    result = self._registry.call(
                        action,
                        step.action_input,
                        principal=principal,
                        conversation_id=conversation_id,
                        question=question,
                    )
                    attempted.append(action)
                    step.observation = result.observation()
                    step.ok = result.ok
                    seen[key] = step.observation
                    trace += self._note("Tool", f"{action}: {_outcome(result)}")
                    if result.ok:
                        succeeded.append(action)
                        citations = _merge_citations(citations, result.value)

                    # The asymmetry is deliberate. `needs_approval` ends the
                    # run: a person now owns the request, and letting the model
                    # carry on would have it looking for a way round a control
                    # that exists to stop it. `denied` is only an observation,
                    # because this caller not being allowed one tool says
                    # nothing about the others -- an employee refused the ticket
                    # tool can still be told the policy.
                    if result.needs_approval:
                        escalated = True
                        answer = ESCALATED_ANSWER
                        break
                    continue

                if final:
                    answer = final
                    trace += self._note("ReAct", "answered")
                    break

                # Neither an action nor an answer: the model has nothing to add,
                # so compose from what is already on the transcript rather than
                # spending another turn asking again.
                trace += self._note("ReAct", "no action and no answer; composing")
                break
            else:
                stopped = True
                trace += self._note(
                    "ReAct", f"reached the {self._max_steps}-step ceiling"
                )

            if not answer:
                composed = self._compose(question, self._transcript(turns, steps))
                if composed:
                    answer = composed
                    if stopped:
                        answer += BUDGET_STOPPED_NOTE
                else:
                    answer = GENERATION_FAILED
                    trace += self._note("ReAct", "composing the answer failed")

        source_used = _source_used(escalated, answer, succeeded, attempted)
        trace += self._note("ReAct", f"source {source_used}")
        return ReActResult(
            question=question,
            answer=answer,
            steps=steps,
            source_used=source_used,
            citations=citations,
            stopped_on_budget=stopped,
            budget=self._budget.report(),
            trace=trace,
        )

    # --- One decision --------------------------------------------------------
    def _decide(
        self, question: str, catalogue: str, transcript: str, index: int
    ) -> ReActDecision:
        """The structured call, with the recovery `app/agent/decisions.py` sets.

        Same three stages as `route_question` and `grade_evidence`: try the
        structured call, recover the model's raw text from a Groq
        `tool_use_failed` body, and only then fall back to a safe default.
        """
        try:
            decision = self._decider.invoke(
                {
                    "tools": catalogue,
                    "question": question,
                    "transcript": transcript or "(nothing yet)",
                    "budget": self._budget_line(),
                }
            )
            self._budget.charge_llm()
            return decision
        except Exception as exc:  # noqa: BLE001 - a decision must not abort a run
            # Charged either way: the round trip happened and the budget would
            # otherwise undercount exactly the run that is failing repeatedly.
            self._budget.charge_llm()
            recovered = _parse_decision(failed_generation(exc))
            if recovered is not None:
                log.debug("[ReAct] recovered a decision from a failed structured call")
                return recovered
            if is_rate_limit(exc):
                log.error(
                    "[ReAct] RATE LIMITED; falling back to a lookup rather than "
                    "an answer -- this is a quota failure, not a judgement (%s)",
                    str(exc)[:160],
                )
            else:
                log.warning(
                    "[ReAct] the step decision failed (%s); falling back",
                    type(exc).__name__,
                )
            self._tracer.degraded("react.decide", reason="unreadable", kind="llm")
            return self._safe_default(question, index)

    def _safe_default(self, question: str, index: int) -> ReActDecision:
        """What to do when the model's decision cannot be read at all.

        **Not "answer from memory".** A model we could not parse has told us
        nothing, and filling the gap from training data is the one failure an HR
        copilot must not have. So on the first step the default is to go and
        look -- the same reasoning that makes `route_question` default to `kb`
        rather than `direct` -- and after that it is to compose from the
        observations already gathered, which is evidence somebody retrieved.
        """
        if index == 0:
            search = self._search_tool()
            if search is not None:
                return ReActDecision(
                    thought="the decision was unreadable; searching policy instead",
                    action=search,
                    action_input={"query": question},
                )
        return ReActDecision(
            thought="composing from what has been gathered so far",
        )

    def _search_tool(self) -> str | None:
        if self._registry.get(DEFAULT_SEARCH_TOOL) is not None:
            return DEFAULT_SEARCH_TOOL
        for name in self._registry.names():
            spec = self._registry.get(name)
            if spec is None:
                continue
            properties = spec.parameters.model_json_schema().get("properties") or {}
            if "query" in properties:
                return name
        return None

    # --- Composing -----------------------------------------------------------
    def _compose(self, question: str, transcript: str) -> str:
        """The best answer the gathered evidence supports, and what is missing."""
        try:
            text = message_text(
                (REACT_FORCED_ANSWER_PROMPT | self._llm).invoke(
                    {
                        "question": question,
                        "transcript": transcript or "(nothing was gathered)",
                    }
                )
            )
            self._budget.charge_llm()
            return text.strip()
        except Exception:  # noqa: BLE001
            log.exception("Composing the partial answer failed")
            self._tracer.degraded("react.compose", reason="unreachable", kind="llm")
            return ""

    # --- Small helpers -------------------------------------------------------
    def _transcript(self, turns: list[dict[str, str]], steps: list[ReActStep]) -> str:
        parts: list[str] = []
        if turns:
            rendered = "\n".join(
                f"{'Employee' if turn['role'] == 'user' else 'Assistant'}: "
                f"{turn['content']}"
                for turn in turns
            )
            parts.append(f"Earlier in this conversation:\n{rendered}")
        parts.extend(step.render() for step in steps)
        return "\n\n".join(parts)

    def _budget_line(self) -> str:
        """`budget.remaining()` as one short human string for the prompt.

        Told how little is left, a model stops planning a third search. Told
        nothing, it plans one and is cut off mid-answer.
        """
        left = self._budget.remaining()
        return (
            f"{left['steps']:.0f} steps, {left['tool_calls']:.0f} tool calls, "
            f"{left['seconds']:.0f}s"
        )

    def _note(self, tag: str, message: Any) -> list[str]:
        """Log one step and return it for `trace`, as `build_nodes.note` does."""
        if self._verbose:
            log.info("[%s] %s", tag, message)
        return [f"{tag}: {message}"]


# --- Module-level helpers ----------------------------------------------------


def _canonical_args(arguments: dict[str, Any]) -> str:
    """A stable rendering, so reordered keys are still the same call."""
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(sorted(arguments))


def _outcome(result: ToolResult) -> str:
    if result.needs_approval:
        return "needs human approval"
    if result.denied:
        return "refused"
    if result.replayed:
        return "replayed a previous result"
    return "ok" if result.ok else f"failed ({result.error})"


def _parse_decision(text: str | None) -> ReActDecision | None:
    """Read a decision out of the raw text a failed structured call carried.

    JSON only, and deliberately so. A `failed_generation` body often holds the
    model's prose instead of a tool call, and treating that prose as the final
    answer would turn every structured-output failure into an answer written
    from the model's own knowledge -- which is the exact outcome
    `_safe_default` exists to avoid. Tolerant about *shape* (a JSON object
    anywhere in the text, with several spellings per field) and strict about
    kind.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    fields: dict[str, Any] = {}
    for target, aliases in _DECISION_ALIASES.items():
        for alias in aliases:
            if alias in data:
                fields[target] = data[alias]
                break
    if not isinstance(fields.get("action_input"), dict):
        fields["action_input"] = {}
    for key in ("thought", "action", "final_answer"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = str(fields[key])
    try:
        return ReActDecision(**fields)
    except ValidationError:
        return None


def _documents_to_citations(chunks: Any) -> list[dict[str, str]]:
    if not isinstance(chunks, list):
        return []
    found: list[dict[str, str]] = []
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = chunk if isinstance(chunk, dict) else None
        if not isinstance(metadata, dict) or "source" not in metadata:
            continue
        source = str(metadata.get("source", "unknown"))
        found.append(
            {
                "source": source,
                "title": str(metadata.get("title", source)),
                "department": str(metadata.get("department", "")),
            }
        )
    return found


def _citations_from(value: Any) -> list[dict[str, str]]:
    """The shape a search tool is expected to return.

        {"chunks": [Document, ...],
         "citations": [{"source": ..., "title": ...}, ...],
         "text": "<evidence rendered for the model>"}

    `text` is what the model reads, `citations` is what the UI shows, `chunks`
    is the evidence itself. All three are optional and the reader is tolerant on
    purpose: the tools that produce this live at the services layer, which this
    module may not import, so the only thing holding the two ends together is
    this docstring. A tool returning a bare list of Documents, or a dict with
    only `text`, degrades to no citations rather than to an exception.
    """
    if isinstance(value, dict):
        cites = value.get("citations")
        if isinstance(cites, list):
            return [
                {str(k): str(v) for k, v in cite.items()}
                for cite in cites
                if isinstance(cite, dict)
            ]
        return _documents_to_citations(value.get("chunks"))
    if isinstance(value, list):
        return _documents_to_citations(value)
    return []


def _merge_citations(
    existing: list[dict[str, str]], value: Any
) -> list[dict[str, str]]:
    """Accumulate, deduplicated on `source`.

    Accumulating is right here and wrong in the graph, which replaces them: a
    graph rewrite re-retrieves and the old chunks stop supporting the answer,
    whereas every ReAct observation stays on the transcript the answer is
    composed from.
    """
    found = _citations_from(value)
    if not found:
        return existing
    merged = list(existing)
    seen = {cite.get("source") for cite in merged}
    for cite in found:
        source = cite.get("source")
        if source in seen:
            continue
        seen.add(source)
        merged.append(cite)
    return merged


def _source_used(
    escalated: bool, answer: str, succeeded: list[str], attempted: list[str]
) -> str:
    """Provenance from what ran, not from what the model claims.

    A model asked to cite its sources will cheerfully write "according to the
    company handbook" over an answer composed from a web article, so the field
    the audit log and the UI show is derived from the tools that actually
    returned something. The private KB outranks the web for the same reason the
    prompt says to search it first: company policy is the authority here.
    """
    if escalated:
        return "escalated"
    if answer == GENERATION_FAILED:
        return "error"
    if any(name in KB_TOOLS for name in succeeded):
        return "private_kb"
    if any(name in WEB_TOOLS for name in succeeded):
        return "web_search"
    if any(name in EVIDENCE_TOOLS for name in attempted):
        # Evidence was sought and none came back. "direct" would describe this
        # as an answer that never needed a lookup, which is the opposite.
        return "insufficient_evidence"
    return "direct"
