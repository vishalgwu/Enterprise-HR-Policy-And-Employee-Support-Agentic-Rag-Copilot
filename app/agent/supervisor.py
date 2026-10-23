"""The supervisor-worker team: one question, several specialists, one answer.

    START -> plan -> supervise
      supervise -> policy | benefits | payroll | workplace | web | general
      supervise -> escalate                              (a person must decide)
      supervise -> compose                               (done, or out of room)
      <any worker> -> supervise                          (handoff back)
      compose  -> END
      escalate -> END

**Why a compiled graph and not a while-loop.** A hand-rolled loop would be
shorter. It would also be invisible: this project's whole shape is "the graph is
the architecture" -- `docs/agent-graph.md` is generated from the compiled edges,
the console lights nodes from the trace, and
`test_the_compiled_graph_matches_the_documented_design` is what stops the
picture drifting from the code. A loop gets none of that, and a multi-agent
system is exactly the one whose control flow a reader most needs drawn.

**Why the rules live in the branch and not in the prompt.** Three things must
hold on every run: the web specialist is unreachable until a private one has
reported, no specialist is consulted twice, and the loop stops at
`max_handoffs`. `SUPERVISOR_PROMPT` asks for all three. A prompt instruction is
a suggestion -- it holds on most questions and fails on the one where the model
is confident -- and the failures here are not cosmetic: web-before-private
answers an HR question from a public article when policy existed, a repeated
specialist burns a handoff to re-read the same documents, and an unbounded loop
is a model deciding, each time, that one more opinion would help. So
`delegate_after_supervise` enforces them in code, and the prompt's version is
there only to make the model's first guess usually right.

**What the workers are.** Six specialists, mapped onto the documents that
actually exist in `data/private_kb/` rather than onto a generic HR org chart --
see `WORKER_DOMAINS`. A worker retrieves from its own slice, reports what its
documents say, and says plainly when they do not cover the question. **A worker
that found nothing is a worker that did its job**: the one failure the
supervisor cannot detect is a specialist guessing to look useful, which is why
`covered` is a flag on the worker's own report and not a grade the supervisor
awards.

Node bodies are closures over the clients, like `app.agent.nodes`, and degrade
the same way: `guard()` turns a failed retrieval into zero chunks and a failed
generation into an uncovered finding. None of them raises -- a worker that
throws would cost the answer every other worker had already found.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from operator import add
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agent.decisions import coerce, failed_generation
from app.agent.prompts import (
    BUDGET_STOPPED_NOTE,
    COMPOSE_PROMPT,
    ESCALATED_ANSWER,
    ESCALATION_PROMPT,
    GENERATION_FAILED,
    INSUFFICIENT_ANSWER,
    SPLIT_PROMPT,
    SUPERVISOR_PROMPT,
    WORKER_PROMPT,
)
from app.agent.schemas import DelegationDecision, SplitDecision, normalise_history
from app.core.config import Settings, get_logger, get_settings
from app.core.tracing import Tracer, null_tracer
from app.rag.clients import (
    get_embeddings,
    get_llm,
    get_web_search,
    is_rate_limit,
    message_text,
    web_result_urls,
    web_results_to_text,
)
from app.rag.loaders import cite_sources, format_documents
from app.rag.retrieval import get_kb_retriever
from app.services.state_db import now_iso

log = get_logger("supervisor")

__all__ = [
    "Finding",
    "PRIVATE_WORKERS",
    "SUPERVISOR_BRANCHES",
    "SUPERVISOR_EDGES",
    "SUPERVISOR_NODE_NAMES",
    "SupervisorState",
    "SupervisorTeam",
    "WORKER_DOMAINS",
    "build_supervisor_graph",
    "build_supervisor_nodes",
    "initial_supervisor_state",
]


# --- The specialisms ---------------------------------------------------------
# Derived from the eleven files in `data/private_kb/`, not from an org chart.
# The `description` strings must stay word-for-word what `SUPERVISOR_PROMPT`
# lists, because that prompt is what the model routes on and this table is what
# the code routes on; two descriptions of one specialist is how a question gets
# delegated to a worker whose documents do not cover it.
#
# `doc_hints` are the real file names. They are what makes a specialism
# checkable: adding a document to the knowledge base without adding it here
# means no specialist claims it, and `general` is the only one that would find
# it.
#
# `departments` is the metadata filter a *foldered* deployment uses -- a
# subdirectory under `data/private_kb/` becomes the `department` on every chunk
# in it. The shipped corpus is flat, so every chunk is `general` and an
# unconditional per-worker filter would return zero chunks for five of the six
# specialists. See `_retriever_for`: the filter is applied only where the corpus
# is actually filed that way.
WORKER_DOMAINS: dict[str, dict[str, Any]] = {
    "policy": {
        "label": "Policy",
        "departments": ["policy"],
        "doc_hints": [
            "leave-and-time-off.md",
            "attendance-and-working-hours.md",
            "remote-and-hybrid-work.md",
        ],
        "description": (
            "leave, time off, attendance, working hours, remote and hybrid work"
        ),
    },
    "benefits": {
        "label": "Benefits",
        "departments": ["benefits"],
        "doc_hints": ["benefits-and-insurance.md"],
        "description": "insurance, medical cover, allowances, wellbeing",
    },
    "payroll": {
        "label": "Payroll",
        "departments": ["payroll"],
        "doc_hints": [
            "payroll-and-compensation.md",
            "expense-reimbursement.md",
        ],
        "description": "pay, salary, payslips, tax, expenses, reimbursement",
    },
    "workplace": {
        "label": "Workplace",
        "departments": ["workplace"],
        "doc_hints": [
            "code-of-conduct-and-grievance.md",
            "employee-records-and-data-privacy.md",
            "hr-forms-and-requests.md",
        ],
        "description": (
            "conduct, grievance, data privacy, employee records, forms and requests"
        ),
    },
    "web": {
        "label": "Public web",
        "departments": [],
        "doc_hints": [],
        "description": (
            "public, non-company information, and only after the private "
            "specialists have come back empty"
        ),
    },
    # The catch-all, and deliberately unfiltered: it is the only specialist that
    # can see the whole corpus. `onboarding-and-offboarding.md` is listed here
    # rather than under policy because joining and leaving cut across every
    # other specialism -- payroll registration, benefit elections, equipment,
    # notice -- so filing it under one of them would hide it from the other
    # three.
    "general": {
        "label": "General",
        "departments": [],
        "doc_hints": [
            "onboarding-and-offboarding.md",
            "hr-copilot-how-it-works.md",
        ],
        "description": "anything that does not obviously belong to one of the above",
    },
}

# Specialists that read the private knowledge base. `web` is a worker but not a
# private one, and `escalate` is a destination rather than a worker at all.
PRIVATE_WORKERS: tuple[str, ...] = (
    "policy",
    "benefits",
    "payroll",
    "workplace",
    "general",
)

WORKER_NAMES: tuple[str, ...] = tuple(WORKER_DOMAINS)

# What a `DelegationDecision.worker` may say, for the recovery path in
# `_delegate`. Same shape as `ROUTE_VALUES` / `GRADE_VALUES` in
# `app.agent.decisions`.
WORKER_VALUES: tuple[str, ...] = (*WORKER_NAMES, "escalate")

SUPERVISOR_NODE_NAMES: tuple[str, ...] = (
    "plan",
    "supervise",
    *WORKER_NAMES,
    "compose",
    "escalate",
)

TERMINAL_NODES: tuple[str, ...] = ("compose", "escalate")

# Unconditional edges: the plan opens the run, and every worker hands back.
SUPERVISOR_EDGES: tuple[tuple[str, str], ...] = (
    ("plan", "supervise"),
    *tuple((worker, "supervise") for worker in WORKER_NAMES),
)

# The one conditional branch, as {source: (condition, {returned name: target})}.
# Explicit rather than inferred, for the reason `app.agent.graph` gives: it is
# what the rendered diagram draws, and an unmapped return fails at compile time
# instead of on whichever question reaches it first.
SUPERVISOR_BRANCHES: dict[str, tuple[str, dict[str, str]]] = {
    "supervise": (
        "delegate_after_supervise",
        {
            **{worker: worker for worker in WORKER_NAMES},
            "compose": "compose",
            "escalate": "escalate",
        },
    ),
}

# A question this short with no joining word is one question. Splitting it would
# cost a model call to be told so, which taxes the common case to serve the rare
# one -- the same reasoning `contextualize` already uses in `app.agent.nodes`,
# where no history means no LLM call.
SPLIT_MIN_WORDS = 8
SPLIT_HINTS: tuple[str, ...] = (
    " and ",
    " also ",
    " plus ",
    " as well as ",
    " versus ",
    " vs ",
    ";",
)

# A worker saying its documents do not cover the question, in its own words.
# **This is not grading.** It is reading the sentence `WORKER_PROMPT` asked for,
# so an honest "my documents do not cover this" is recorded as `covered=False`
# rather than merged into the answer as if it were a finding. A phrasing this
# misses costs the composer one paragraph of scaffolding to drop, which is the
# cheap direction for this check to fail in.
NOT_COVERED_MARKERS: tuple[str, ...] = (
    "do not cover",
    "does not cover",
    "don't cover",
    "doesn't cover",
    "not covered",
    "no information",
    "nothing in my documents",
    "my documents do not",
)


@dataclass(slots=True)
class Finding:
    """What one specialist reported back.

    `covered` is the load-bearing field. The supervisor composes from the
    findings that claim coverage and says which part of the question nobody
    covered, so a worker that reports prose *and* `covered=False` would have its
    hedging merged into the answer as fact.
    """

    worker: str
    text: str
    citations: list[dict[str, str]] = field(default_factory=list)
    chunks: int = 0
    covered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "text": self.text,
            "citations": list(self.citations),
            "chunks": self.chunks,
            "covered": self.covered,
        }


class SupervisorState(TypedDict):
    """State the supervisor and its workers share.

    **Four keys accumulate, and each uses the `add` reducer for the reason
    `AgentState.trace` documents**: a plain TypedDict key is *replaced* by
    whatever a node returns, so a bare `list` would leave only the last node's
    entry -- and on this graph that is catastrophic rather than untidy, because
    the whole point of the run is the several workers' several reports. Four
    workers would compose from one finding.

    All four are seeded by `initial_supervisor_state`, because a reducer has
    nothing to fold into on a missing key.

    `consulted` deliberately does *not* accumulate: a worker returns the whole
    new list, so there is one writer and no chance of a duplicate entry making
    `len(consulted)` disagree with `handoff_count`. `handoff_count` is a plain
    int for the same reason -- a reducer on it would sum every node's view of
    the total.
    """

    question: str
    standalone_question: str
    history: list[dict[str, str]]

    trace: Annotated[list[str], add]
    worker_path: Annotated[list[str], add]
    findings: Annotated[list[dict[str, Any]], add]
    handoffs: Annotated[list[dict[str, Any]], add]

    consulted: list[str]
    handoff_count: int

    citations: NotRequired[list[dict[str, str]]]
    subquestions: NotRequired[list[str]]
    next_worker: NotRequired[str]
    # Why the supervisor chose that worker, in its own words. Written by
    # `supervise` and read by the worker node, which is what puts a reason on
    # the handoff record rather than leaving it to be reconstructed from the
    # trace.
    delegation_reason: NotRequired[str]
    escalated: NotRequired[bool]
    escalation_reason: NotRequired[str]
    answer: NotRequired[str]
    source_used: NotRequired[str]


def initial_supervisor_state(
    question: str, history: Sequence[Mapping[str, Any]] | None = None
) -> SupervisorState:
    """Seed a state so no caller can forget a key a reducer folds into.

    `standalone_question` is seeded equal to `question`: this graph has no
    contextualise node, because it is reached either directly (single turn) or
    from a caller that has already resolved the follow-up and passes the
    resolved wording in. `history` is carried for the audit row and for that
    caller's benefit; no prompt on this graph renders it.
    """
    return SupervisorState(
        question=question,
        standalone_question=question,
        history=normalise_history(history),
        trace=[],
        worker_path=[],
        findings=[],
        handoffs=[],
        consulted=[],
        handoff_count=0,
    )


# --- Helpers used by both the nodes and the branch ---------------------------


def _asked(state: SupervisorState) -> str:
    return state.get("standalone_question") or state["question"]


def looks_multi_part(question: str) -> bool:
    """A cheap guess at whether a question is worth a split call.

    Deliberately lexical. The alternative is spending a `SplitDecision` call on
    every question to be told that a seven-word one is one question, and the
    overwhelming majority of HR questions are seven-word ones.
    """
    words = question.split()
    if len(words) < SPLIT_MIN_WORDS:
        return False
    if question.count("?") > 1:
        return True
    lowered = f" {question.lower()} "
    return any(hint in lowered for hint in SPLIT_HINTS)


def _private_reported(state: SupervisorState) -> bool:
    return any(worker in PRIVATE_WORKERS for worker in state.get("consulted") or [])


def _first_unconsulted_private(state: SupervisorState) -> str | None:
    consulted = set(state.get("consulted") or [])
    for worker in PRIVATE_WORKERS:
        if worker not in consulted:
            return worker
    return None


def _render_findings(findings: Sequence[Mapping[str, Any]]) -> str:
    """The findings as the supervisor and the composer both see them."""
    if not findings:
        return "(nothing yet)"
    lines: list[str] = []
    for finding in findings:
        worker = finding.get("worker", "?")
        if not finding.get("covered"):
            lines.append(
                f"[{worker}] reported that its documents do not cover this."
            )
            continue
        sources = ", ".join(
            str(c.get("source", "")) for c in finding.get("citations") or []
        )
        text = str(finding.get("text", "")).strip()
        lines.append(f"[{worker}] {text}")
        if sources:
            lines.append(f"    sources: {sources}")
    return "\n".join(lines)


def _dedupe_citations(
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """One entry per source document, in the order the workers found them.

    Two specialists routinely cite the same file -- the knowledge base overlaps
    on purpose -- and an answer listing `leave-and-time-off.md` twice reads as
    two sources agreeing when it is one source counted twice.
    """
    seen: dict[str, dict[str, str]] = {}
    for finding in findings:
        for citation in finding.get("citations") or []:
            source = str(citation.get("source", "")).strip()
            if source and source not in seen:
                seen[source] = {str(k): str(v) for k, v in citation.items()}
    return list(seen.values())


def _foldered_departments(settings: Settings) -> frozenset[str]:
    """Department names the knowledge base is actually filed under.

    One `iterdir` of `kb_dir`, because `app.rag.loaders` derives a chunk's
    `department` from its subdirectory and files in the root get `general`. The
    shipped corpus is flat, so this is empty and no worker filters; a deployment
    that files benefits under `data/private_kb/benefits/` gets the filter
    without any other change. Guessing instead would mean five of six
    specialists retrieving nothing at all, which looks exactly like an empty
    knowledge base.
    """
    try:
        return frozenset(
            path.name.lower() for path in Path(settings.kb_dir).iterdir() if path.is_dir()
        )
    except OSError:
        return frozenset()


def _delegate(
    question: str,
    consulted: Sequence[str],
    findings: Sequence[Mapping[str, Any]],
    delegator: Any,
    *,
    first_hop: bool,
) -> tuple[str, str, bool]:
    """One delegation decision, surviving a structured-output failure.

    Same recovery shape as `route_question` and `grade_evidence` in
    `app.agent.decisions`: try the structured call, recover the model's raw text
    from a Groq `tool_use_failed` body, then fall back to a safe default.

    **The default is `general` on the first hop and "done" on every hop after
    it.** Defaulting to another handoff on an unreadable decision is how a
    broken model spends the entire budget: every hop fails the same way, every
    failure says "delegate", and the run consults six specialists before
    composing from whatever the first one happened to find. One hop to the
    unfiltered specialist is worth a call; the second is a loop.
    """
    try:
        decision = delegator.invoke(
            {
                "question": question,
                "consulted": ", ".join(consulted) if consulted else "none yet",
                "findings": _render_findings(findings),
            }
        )
        return decision.worker, decision.reason, bool(decision.done)
    except Exception as exc:  # noqa: BLE001 - no client error may crash the graph
        recovered = coerce(failed_generation(exc), WORKER_VALUES)
        if recovered:
            log.debug("[Supervisor] recovered %r from a failed structured call", recovered)
            return recovered, "recovered from a malformed decision", False
        if is_rate_limit(exc):
            log.error(
                "[Supervisor] RATE LIMITED; this is a quota failure, not a "
                "judgement (%s)",
                str(exc)[:160],
            )
        else:
            log.warning(
                "[Supervisor] delegation decision failed (%s); using the safe default",
                type(exc).__name__,
            )
        if first_hop:
            return "general", "the delegation decision was unreadable", False
        return "general", "the delegation decision was unreadable", True


# --- Nodes -------------------------------------------------------------------


def build_supervisor_nodes(
    llm: Any = None,
    retriever: Any = None,
    web_search: Any = None,
    *,
    settings: Settings | None = None,
    tracer: Tracer | None = None,
    budget: Any = None,
    scratchpad: Any = None,
    approvals: Any = None,
    conversation_id: str | None = None,
    verbose: bool = False,
) -> dict[str, Callable]:
    """Construct every node once, closed over the shared clients and the run.

    `tracer`, `budget`, `scratchpad` and `conversation_id` belong to one
    request, which is why `SupervisorTeam` rebuilds the graph per question
    rather than per process -- see its docstring. The clients do not, so they
    are resolved once and reused.

    `scratchpad` and `approvals` are duck-typed -- anything with
    `.append(conversation_id, text, author=, kind=)` and anything with
    `.request(action=, question=, reason=, payload=, risk=, requested_by=,
    conversation_id=)`. That is what keeps this module out of `app.services`,
    which it may not import.
    """
    settings = settings or get_settings()
    tracer = tracer if tracer is not None else null_tracer()
    llm = llm or get_llm(settings=settings)

    max_handoffs = settings.max_handoffs
    foldered = _foldered_departments(settings) if retriever is None else frozenset()
    _retrievers: dict[str, Any] = {}
    # Resolved on first use rather than here, like the retrievers: the web
    # specialist is reached on a minority of questions, and a deployment with
    # no Tavily key should degrade on the one question that wanted it rather
    # than fail to build a graph for all the others.
    _search: list[Any] = [web_search]

    delegator = SUPERVISOR_PROMPT | llm.with_structured_output(DelegationDecision)
    splitter = SPLIT_PROMPT | llm.with_structured_output(SplitDecision)

    # --- Shared plumbing -----------------------------------------------------
    def note(tag: str, message: Any) -> list[str]:
        """Log one step and return it for the node to put in `trace`.

        As in `app.agent.nodes`: `verbose` silences the console line only, so
        the console and the trace cannot drift apart.
        """
        if verbose:
            log.info("[%s] %s", tag, message)
        return [f"{tag}: {message}"]

    def guard(
        tag: str,
        action: Callable[[], Any],
        fallback: Any,
        steps: list[str] | None = None,
    ) -> Any:
        """Run a network call; on failure degrade to `fallback` and say so.

        A worker that raised would abort the run and cost the answer every
        other worker had already found, which on this graph is a strictly worse
        failure than it is on the single-agent one.
        """
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - a node must never abort the graph
            if is_rate_limit(exc):
                log.error(
                    "[%s] RATE LIMITED by the provider - this is a quota "
                    "failure, not a knowledge gap (%s)",
                    tag,
                    str(exc)[:160],
                )
                detail = "rate limited by the provider; results are degraded"
            else:
                log.warning(
                    "[%s] failed (%s); continuing with fallback",
                    tag,
                    type(exc).__name__,
                )
                detail = f"failed ({type(exc).__name__}); continuing with fallback"
            if steps is not None:
                steps.append(f"{tag}: {detail}")
            tracer.degraded(tag, reason=detail, kind="tool")
            return fallback

    def pad(kind: str, text: str, author: str) -> None:
        """Write one scratchpad note, if this run has a pad. Never raises."""
        if scratchpad is None or not conversation_id or not text.strip():
            return
        try:
            scratchpad.append(conversation_id, text, author=author, kind=kind)
        except Exception:  # noqa: BLE001 - a lost note must not cost the answer
            log.exception("Writing a scratchpad note failed")

    def _record_escalation(
        state: SupervisorState, reason: str, handover: str
    ) -> list[str]:
        """File the escalation in the approval queue. Returns trace steps.

        Degrades rather than raising, like every other node here, but the trace
        distinguishes the two outcomes: an employee told that a person will pick
        their question up, when nothing recorded it, is the worst failure this
        node has -- so "escalation was not recorded" has to be visible to
        whoever reads the run afterwards, not inferable only from the queue's
        emptiness.
        """
        if approvals is None:
            return []
        try:
            recorded = approvals.request(
                action="supervisor_escalation",
                question=state.get("question", ""),
                reason=reason,
                payload={
                    "handover": handover,
                    "workers_consulted": list(state.get("consulted") or []),
                    "findings": len(state.get("findings") or []),
                },
                risk="high",
                requested_by="supervisor",
                conversation_id=conversation_id,
            )
        except Exception:  # noqa: BLE001 - a node must never abort the graph
            log.exception("Recording the escalation failed")
            tracer.degraded("supervisor.escalate", reason="queue write raised")
            return ["Escalate: the approval queue could not be written to"]

        if recorded is None:
            tracer.degraded("supervisor.escalate", reason="queue declined the write")
            return ["Escalate: the approval queue did not record this"]
        return [f"Escalate: queued as approval #{getattr(recorded, 'id', recorded)}"]

    def retriever_for(worker: str) -> Any:
        """The retriever one specialist reads through.

        An injected retriever serves every worker -- that is the test seam, and
        also what a deployment with one gated retriever gets. Otherwise each
        private specialist gets its own, narrowed by `department` only where the
        corpus is filed that way (see `_foldered_departments`).
        """
        if retriever is not None:
            return retriever
        if worker not in _retrievers:
            wanted = [
                dept
                for dept in WORKER_DOMAINS[worker]["departments"]
                if dept in foldered
            ]
            _retrievers[worker] = get_kb_retriever(
                embedding=get_embeddings(),
                department=wanted[0] if wanted else None,
                settings=settings,
            )
        return _retrievers[worker]

    def search_tool() -> Any:
        if _search[0] is None:
            _search[0] = get_web_search(settings=settings)
        return _search[0]

    def forced_stop(state: SupervisorState) -> str | None:
        """Why this run must compose now, or None.

        One implementation, read by three callers that each need the same
        answer: `supervise` to skip a pointless model call, the branch to force
        the edge, and `compose` to say in the trace why it was reached. **Never
        let the loop run on the model's judgement alone** -- `done` is a model
        output and this is not.
        """
        if int(state.get("handoff_count", 0)) >= max_handoffs:
            return f"the handoff limit ({max_handoffs}) was reached"
        if budget is not None and not budget.allows("steps"):
            return budget.stop_reason or "the step budget was reached"
        return None

    # --- 0. Plan -------------------------------------------------------------
    def plan(state: SupervisorState) -> dict[str, Any]:
        """Decide once whether this question is really several.

        **Gated on a cheap heuristic.** Paying a round trip to be told a
        seven-word question is one question taxes the common case to serve the
        rare one -- the same trade `contextualize` makes in `app.agent.nodes`,
        where no history means no LLM call.

        Splitting does not fan out retrieval. It tells the *supervisor* that two
        policy areas are in play, which is what turns one handoff into two; a
        single retrieval over a two-area question fills its top-k with whichever
        area the corpus covers better, and that is the failure
        `SplitDecision` exists to name.
        """
        question = _asked(state)
        if not looks_multi_part(question):
            pad("plan", f"Single-area question: {question}", "supervisor")
            return {"trace": note("Planner", "one policy area; no split call")}

        steps: list[str] = []
        with tracer.span("supervisor.plan", kind="llm", question=question):
            decision = guard(
                "Planner",
                lambda: splitter.invoke({"question": question}),
                None,
                steps,
            )
        if budget is not None:
            budget.charge_llm()

        subquestions: list[str] = []
        if decision is not None and getattr(decision, "needs_split", False):
            subquestions = [
                str(part).strip()
                for part in (getattr(decision, "subquestions", None) or [])
                if str(part).strip()
            ]
        if subquestions:
            pad(
                "plan",
                "Question spans several areas:\n"
                + "\n".join(f"- {part}" for part in subquestions),
                "supervisor",
            )
            steps += note("Planner", f"{len(subquestions)} sub-question(s)")
            return {"subquestions": subquestions, "trace": steps}

        pad("plan", f"Single-area question: {question}", "supervisor")
        return {"trace": steps + note("Planner", "one policy area")}

    # --- 1. Supervise --------------------------------------------------------
    def supervise(state: SupervisorState) -> dict[str, Any]:
        """Choose the next specialist, or say the team is done.

        Writes what the model decided; the branch is what enforces the rules
        over it. The two are separated on purpose: this node's trace entry
        should say what the supervisor *wanted*, and the trace entry for what
        actually happened belongs to the node that happened.
        """
        if budget is not None:
            budget.charge_step()

        stop = forced_stop(state)
        if stop is not None:
            # Deliberately no model call: the decision is already made, and
            # asking anyway is a round trip spent on an answer nobody will read.
            return {
                "next_worker": "compose",
                "trace": note("Supervisor", f"composing because {stop}"),
            }

        consulted = list(state.get("consulted") or [])
        question = _asked(state)
        subquestions = state.get("subquestions") or []
        if subquestions:
            question = question + "\n\nParts to cover:\n" + "\n".join(
                f"- {part}" for part in subquestions
            )

        with tracer.span("supervisor.delegate", kind="router", consulted=len(consulted)):
            worker, reason, done = _delegate(
                question,
                consulted,
                state.get("findings") or [],
                delegator,
                first_hop=not consulted,
            )
        if budget is not None:
            budget.charge_llm()

        # An escalation request outranks `done`: a supervisor that has decided a
        # person must see this has told you its answer is not the product.
        if worker == "escalate":
            target = "escalate"
        elif done:
            target = "compose"
        else:
            target = worker

        return {
            "next_worker": target,
            "delegation_reason": reason,
            "escalation_reason": reason if target == "escalate" else "",
            "trace": note("Supervisor", f"{target}" + (f" ({reason})" if reason else "")),
        }

    def delegate_after_supervise(state: SupervisorState) -> str:
        """The only branch, and where the three hard rules live.

        Each of them is also in `SUPERVISOR_PROMPT`, and that is not where they
        are enforced. A prompt instruction holds on most questions and fails on
        the one where the model is confident; these three failures are a public
        article answering a question policy covered, a handoff spent re-reading
        documents already read, and a loop with no ceiling.
        """
        stop = forced_stop(state)
        if stop is not None:
            return "compose"

        target = state.get("next_worker") or "general"
        if target in ("compose", "escalate"):
            return target
        if target not in WORKER_DOMAINS:
            # The schema's Literal should make this unreachable; composing from
            # what we have beats routing to a node that does not exist.
            return "compose"

        consulted = state.get("consulted") or []
        if target in consulted:
            # Re-reading the same documents cannot produce a new finding, and
            # the handoff it costs is one the question's other half needed.
            return "compose"

        if target == "web" and not _private_reported(state):
            # Private first is a product requirement, not a preference: a
            # general article that contradicts company policy is wrong here, and
            # an employee cannot tell which they were given. The rule is "private
            # first", not "never web", so an early request for the web becomes
            # the private attempt it should have been.
            fallback = _first_unconsulted_private(state)
            return fallback or "compose"

        return target

    # --- 2. The specialists --------------------------------------------------
    def _report(state: SupervisorState, worker: str) -> dict[str, Any]:
        """One specialist's turn: retrieve its slice, report, hand back."""
        question = _asked(state)
        steps: list[str] = []
        citations: list[dict[str, str]] = []

        with tracer.span(f"worker.{worker}", kind="agent", query=question):
            if worker == "web":
                raw = guard(
                    f"Worker/{worker}",
                    lambda: search_tool().invoke({"query": question}),
                    None,
                    steps,
                )
                if budget is not None:
                    budget.charge_tool()
                context = web_results_to_text(raw)
                chunks = len(web_result_urls(raw))
                citations = [
                    {
                        "source": url,
                        "title": url,
                        "department": "public-web",
                        "doc_type": "web",
                    }
                    for url in web_result_urls(raw)
                ]
            else:
                docs = guard(
                    f"Worker/{worker}",
                    lambda: retriever_for(worker).invoke(question),
                    [],
                    steps,
                )
                if budget is not None:
                    budget.charge_tool()
                context = format_documents(docs)
                chunks = len(docs)
                citations = cite_sources(docs)

            if not context.strip():
                # A successful worker, not a failed one. No prose, because prose
                # about having nothing is what the composer would merge as a
                # finding.
                finding = Finding(worker=worker, text="", chunks=0, covered=False)
                steps += note(f"Worker/{worker}", "no evidence; covered=False")
            else:
                answer = guard(
                    f"Worker/{worker}",
                    lambda: message_text(
                        (WORKER_PROMPT | llm).invoke(
                            {
                                "worker": worker,
                                "question": question,
                                "context": context,
                            }
                        )
                    ),
                    "",
                    steps,
                ).strip()
                if budget is not None:
                    budget.charge_llm()
                lowered = answer.lower()
                covered = bool(answer) and not any(
                    marker in lowered for marker in NOT_COVERED_MARKERS
                )
                finding = Finding(
                    worker=worker,
                    text=answer if covered else "",
                    citations=citations if covered else [],
                    chunks=chunks,
                    covered=covered,
                )
                steps += note(
                    f"Worker/{worker}", f"{chunks} chunk(s); covered={covered}"
                )

        pad(
            "finding",
            finding.text or f"My documents do not cover: {question}",
            worker,
        )
        reason = state.get("delegation_reason") or ""
        handoff = f"supervise -> {worker}"
        pad("handoff", f"{handoff}: {reason}" if reason else handoff, "supervisor")

        return {
            "consulted": [*(state.get("consulted") or []), worker],
            "handoff_count": int(state.get("handoff_count", 0)) + 1,
            "handoffs": [
                {"from": "supervise", "to": worker, "reason": reason, "at": now_iso()}
            ],
            "worker_path": [worker],
            "findings": [finding.to_dict()],
            "trace": steps,
        }

    def _worker_node(worker: str) -> Callable[[SupervisorState], dict[str, Any]]:
        def run(state: SupervisorState) -> dict[str, Any]:
            return _report(state, worker)

        run.__name__ = f"worker_{worker}"
        return run

    # --- 3. Compose ----------------------------------------------------------
    def compose(state: SupervisorState) -> dict[str, Any]:
        """Merge the findings into the one reply the employee reads."""
        findings = list(state.get("findings") or [])
        covered = [f for f in findings if f.get("covered")]
        citations = _dedupe_citations(covered)
        steps: list[str] = []

        stop = forced_stop(state)
        if stop is not None:
            steps += note("Compose", f"forced: {stop}")

        if not covered:
            steps += note("Compose", "no specialist covered this question")
            return {
                "answer": INSUFFICIENT_ANSWER,
                "source_used": "insufficient_evidence",
                "citations": citations,
                "trace": steps,
            }

        with tracer.span("supervisor.compose", kind="llm", findings=len(covered)):
            answer = guard(
                "Compose",
                lambda: message_text(
                    (COMPOSE_PROMPT | llm).invoke(
                        {
                            "question": _asked(state),
                            "findings": _render_findings(covered),
                        }
                    )
                ),
                "",
                steps,
            ).strip()
        if budget is not None:
            budget.charge_llm()

        if not answer:
            return {
                "answer": GENERATION_FAILED,
                "source_used": "error",
                "citations": citations,
                "trace": steps + note("Compose", "produced no answer"),
            }

        # Private outranks the web when both reported: an answer resting on
        # company policy is a different kind of answer, and the provenance the
        # UI shows must be the stronger of the two rather than the last one in.
        workers = {f.get("worker") for f in covered}
        if workers & set(PRIVATE_WORKERS):
            source_used = "private_kb"
        elif "web" in workers:
            source_used = "web_search"
        else:
            source_used = "insufficient_evidence"

        if stop is not None:
            answer += BUDGET_STOPPED_NOTE
        return {
            "answer": answer,
            "source_used": source_used,
            "citations": citations,
            "trace": steps + note("Compose", f"answered from {source_used}"),
        }

    # --- 4. Escalate ---------------------------------------------------------
    def escalate(state: SupervisorState) -> dict[str, Any]:
        """Hand the question to a person, and do not answer it.

        The note is for the person who picks this up, so it is written to the
        scratchpad rather than returned: the employee gets `ESCALATED_ANSWER`
        and nothing else. **This node must not invent an answer** -- the whole
        reason it exists is that the documents do not contain the judgement the
        question needs, and a composed reply here would be exactly the guess the
        escalation was supposed to prevent.
        """
        reason = (
            state.get("escalation_reason")
            or state.get("delegation_reason")
            or "a person must decide this one"
        )
        steps: list[str] = []
        with tracer.span("supervisor.escalate", kind="llm"):
            handover = guard(
                "Escalate",
                lambda: message_text(
                    (ESCALATION_PROMPT | llm).invoke(
                        {
                            "question": _asked(state),
                            "reason": reason,
                            "findings": _render_findings(state.get("findings") or []),
                        }
                    )
                ),
                "",
                steps,
            ).strip()
        if budget is not None:
            budget.charge_llm()

        pad(
            "handoff",
            handover or f"Escalated without a note: {reason}",
            "supervisor",
        )

        # The queue, not just the scratchpad. An escalation nobody can see is
        # not an escalation: the scratchpad is scoped to one conversation and
        # nothing lists conversation ids, so without this row the employee is
        # told a person will pick it up and no person ever learns of it.
        #
        # It is a *record*, not a gate -- unlike the `raise_ticket` tool, this
        # does not wait for a decision, because escalating already *is* asking
        # for a human and the answer has been given. A failed write therefore
        # degrades rather than blocking: the trace says so, and the scratchpad
        # note above still holds the handover.
        steps += _record_escalation(state, reason, handover)

        return {
            "escalated": True,
            "escalation_reason": reason,
            "answer": ESCALATED_ANSWER,
            "source_used": "escalated",
            "citations": _dedupe_citations(state.get("findings") or []),
            "trace": steps + note("Escalate", reason),
        }

    nodes: dict[str, Callable] = {
        "plan": plan,
        "supervise": supervise,
        "compose": compose,
        "escalate": escalate,
        "delegate_after_supervise": delegate_after_supervise,
    }
    for worker in WORKER_NAMES:
        nodes[worker] = _worker_node(worker)
    return nodes


# --- Wiring ------------------------------------------------------------------


def build_supervisor_graph(
    llm: Any = None,
    retriever: Any = None,
    web_search: Any = None,
    *,
    settings: Settings | None = None,
    tracer: Tracer | None = None,
    budget: Any = None,
    scratchpad: Any = None,
    approvals: Any = None,
    conversation_id: str | None = None,
    verbose: bool = False,
) -> Any:
    """Wire and compile the supervisor graph. Every client is injectable."""
    nodes = build_supervisor_nodes(
        llm,
        retriever,
        web_search,
        settings=settings,
        tracer=tracer,
        budget=budget,
        scratchpad=scratchpad,
        approvals=approvals,
        conversation_id=conversation_id,
        verbose=verbose,
    )

    builder = StateGraph(SupervisorState)
    for name in SUPERVISOR_NODE_NAMES:
        builder.add_node(name, nodes[name])

    builder.add_edge(START, "plan")
    for source, target in SUPERVISOR_EDGES:
        builder.add_edge(source, target)
    for source, (condition, path_map) in SUPERVISOR_BRANCHES.items():
        unknown = set(path_map.values()) - set(SUPERVISOR_NODE_NAMES)
        if unknown:
            raise ValueError(f"Branch {source!r} targets unknown node(s): {unknown}")
        builder.add_conditional_edges(source, nodes[condition], path_map)
    for terminal in TERMINAL_NODES:
        builder.add_edge(terminal, END)

    return builder.compile()


class SupervisorTeam:
    """The facade: one question in, one composed answer plus the path out.

    **The graph is built per question, not per process.** It closes over the
    tracer, the budget, the scratchpad and the conversation id, all of which
    belong to one request -- a graph held on the instance would trace one
    employee's question into another's span list. The *clients* do not belong to
    one request, so they are resolved once on first use and reused: building
    them is what costs (an embedding model load, a provider handshake), and
    compiling a `StateGraph` over clients that already exist is not.

    `graph=` is a test hook, for a caller that has built its own. A graph passed
    in closes over whatever it was built with, so the per-request tracer and
    budget `ask` is handed cannot reach it.
    """

    def __init__(
        self,
        llm: Any = None,
        retriever: Any = None,
        web_search: Any = None,
        *,
        settings: Settings | None = None,
        scratchpad: Any = None,
        approvals: Any = None,
        verbose: bool = False,
        graph: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scratchpad = scratchpad
        self.approvals = approvals
        self.verbose = verbose
        self._llm = llm
        self._retriever = retriever
        self._web_search = web_search
        self._graph = graph

    def ask(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        tracer: Tracer | None = None,
        budget: Any = None,
        conversation_id: str | None = None,
        on_step: Callable[[list[str], list[str]], None] | None = None,
    ) -> dict[str, Any]:
        """Run one question through the team.

        Raises ValueError on an empty question rather than spending a planning
        call to discover there was nothing to answer -- the same guard
        `app.agent.graph.ask` has.

        `on_step(new_trace_lines, worker_path)` is called after each step when
        given, so a caller can show progress. It matters more here than on the
        single graph: a supervisor run is a planning call, a delegation call per
        hop, a generation per worker and a composition, which is twenty seconds
        of silence on a real question. Without it the console sat on one word
        for the whole run -- and the thing being demonstrated is precisely the
        sequence of workers it was not showing.

        Duck-typed as a plain callable rather than taking the `EventStream`
        itself, because this module is agent-layer and `app/services/streaming
        .py` is not something it may import.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        tracer = tracer if tracer is not None else null_tracer()

        graph = self._graph or build_supervisor_graph(
            self._llm,
            self._retriever,
            self._web_search,
            settings=self.settings,
            tracer=tracer,
            budget=budget,
            scratchpad=self.scratchpad,
            approvals=self.approvals,
            conversation_id=conversation_id,
            verbose=self.verbose,
        )

        if conversation_id:
            self._note_question(conversation_id, question)

        start = initial_supervisor_state(question, history)
        # Two nodes per handoff plus the plan, the last supervise turn and the
        # composition. Generous, because tripping LangGraph's own ceiling
        # raises where every other limit here degrades.
        config = {"recursion_limit": 2 * self.settings.max_handoffs + 10}

        with tracer.span("supervisor", kind="agent", question=question):
            if on_step is None:
                state = graph.invoke(start, config)
            else:
                state = self._stream(graph, start, config, on_step)

        return {
            "question": state["question"],
            "answer": state.get("answer", INSUFFICIENT_ANSWER),
            "source_used": state.get("source_used", "insufficient_evidence"),
            "escalated": bool(state.get("escalated", False)),
            "escalation_reason": state.get("escalation_reason") or None,
            "subquestions": state.get("subquestions") or [],
            "worker_path": state.get("worker_path") or [],
            "consulted": state.get("consulted") or [],
            "findings": state.get("findings") or [],
            "handoffs": state.get("handoffs") or [],
            "handoff_count": int(state.get("handoff_count", 0)),
            "citations": state.get("citations") or [],
            "trace": state.get("trace") or [],
            "budget": budget.report() if budget is not None else None,
        }

    @staticmethod
    def _stream(
        graph: Any,
        start: SupervisorState,
        config: dict[str, Any],
        on_step: Callable[[list[str], list[str]], None],
    ) -> dict[str, Any]:
        """Run the graph, reporting each step, and return the final state.

        `stream_mode="values"` yields the whole state after every step, so the
        reducers have already been applied and the last value is the final
        state. That is what keeps this from having to merge updates itself and
        from getting `trace`'s `add` reducer wrong.

        Falls back to `invoke` if streaming is unavailable or raises: a
        LangGraph version without `.stream`, or a graph a test injected that
        only implements `invoke`. Progress is a courtesy and the answer is not,
        so the answer must not depend on which.
        """
        seen = 0
        state: dict[str, Any] = dict(start)
        try:
            for value in graph.stream(start, config, stream_mode="values"):
                state = value
                steps = list(value.get("trace") or [])
                if len(steps) > seen:
                    on_step(steps[seen:], list(value.get("worker_path") or []))
                    seen = len(steps)
        except Exception:  # noqa: BLE001 - fall back rather than lose the answer
            log.exception("Streaming the supervisor failed; running it plainly")
            return graph.invoke(start, config)
        return state

    def _note_question(self, conversation_id: str, question: str) -> None:
        if self.scratchpad is None:
            return
        try:
            self.scratchpad.append(
                conversation_id, question, author="employee", kind="user"
            )
        except Exception:  # noqa: BLE001 - a lost note must not cost the answer
            log.exception("Writing the question to the scratchpad failed")
