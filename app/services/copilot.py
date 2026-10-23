"""The copilot facade: one question in, one serialisable answer out.

This is the seam every caller uses -- the CLI demos today, the `/chat` endpoint
next -- so that neither has to know about `AgentState`, LangGraph or which
namespace the KB lives in. `AnswerResult` is deliberately a Pydantic model
rather than a dict: it is the API response contract, and it carries the decision
trace the product brief asks the UI to show.

The graph is built lazily and once. Building it constructs the router, grader,
retriever and web-search client; doing that per request would reload the
embedding model and re-establish every client on every message.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.graph import ask as run_graph
from app.agent.graph import build_graph
from app.agent.schemas import AgentState
from app.core.config import Settings, get_logger, get_settings
from app.rag.loaders import cite_sources

log = get_logger("copilot")


class SourceRef(BaseModel):
    """One cited document."""

    source: str
    title: str = ""
    department: str = "general"
    doc_type: str = ""
    # Set when a re-ranker scored it, so the UI can show *why* this chunk was
    # among the four the answer rests on. Optional because the dense path
    # without re-ranking has no such number and inventing one would be worse
    # than its absence.
    score: float | None = None


class AnswerResult(BaseModel):
    """A finished answer plus the trace of how the agent reached it.

    This is the API response contract, the CLI's input and what the audit log
    reads. It has grown a lot: everything added below is a fact some surface
    needs and none of them is derivable from the answer text.

    The grouping is deliberate, and the order is roughly "what was asked", "what
    was decided", "what it cost", "who did it" and "what to distrust". A reader
    scanning one of these in `/api/audit` should be able to answer, in order:
    what did it say, where did that come from, was anything altered on the way,
    which model wrote it, what did it cost, and did it finish.
    """

    question: str
    answer: str
    route: str | None = None
    source_used: str | None = None
    retry_count: int = 0
    rewritten_query: str | None = Field(
        default=None,
        description="Set only when a rewrite actually changed the query.",
    )
    resolved_question: str | None = Field(
        default=None,
        description=(
            "Set only when a follow-up was resolved against the conversation, "
            "so the UI can show what the agent actually went looking for."
        ),
    )
    kb_grade: str | None = None
    web_grade: str | None = None
    kb_chunks: int = 0
    sources: list[SourceRef] = Field(default_factory=list)
    web_urls: list[str] = Field(default_factory=list)
    trace: list[str] = Field(
        default_factory=list,
        description="One entry per node, in the order the graph ran them.",
    )

    # --- How it was answered -------------------------------------------------
    answer_mode: str = Field(
        default="graph",
        description=(
            "graph | react | supervisor | cached | extractive | blocked | "
            "escalated. Which orchestration produced this, which is not the "
            "same question as which source the evidence came from."
        ),
    )
    retrieval_mode: str | None = Field(
        default=None, description="dense | hybrid | lexical, as resolved at run time."
    )

    # --- What was altered on the way ----------------------------------------
    redacted_question: str | None = Field(
        default=None,
        description=(
            "The question as it left this process, when it differed from what "
            "was typed. Present only when redaction actually changed something."
        ),
    )
    pii_redacted: list[str] = Field(
        default_factory=list, description="Kinds of personal data removed."
    )
    guard_input: dict[str, Any] = Field(default_factory=dict)
    guard_output: dict[str, Any] = Field(default_factory=dict)
    blocked: bool = Field(
        default=False,
        description="True when a guard refused the question outright.",
    )

    # --- Which model, and what it cost --------------------------------------
    model: str | None = None
    provider: str | None = None
    model_tier: str | None = None
    complexity: str | None = None
    routing_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = Field(
        default=0.0,
        description=(
            "A projection from a token estimate and a published list price, "
            "not an invoice. See app/services/routing.py."
        ),
    )
    byok: bool = Field(
        default=False, description="Answered on a caller-supplied provider key."
    )
    cached: bool = False
    cache_similarity: float | None = None

    # --- Multi-agent ---------------------------------------------------------
    worker_path: list[str] = Field(
        default_factory=list,
        description="The specialists consulted, in order.",
    )
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    escalated: bool = False
    approval_id: int | None = None

    # --- What it spent and whether it finished ------------------------------
    steps: int = 0
    tool_calls: int = 0
    stop_reason: str | None = Field(
        default=None,
        description=(
            "Set when a budget ended the run early. Its presence is what makes "
            "a partial answer distinguishable from a short complete one."
        ),
    )
    budget: dict[str, Any] = Field(default_factory=dict)
    degraded: list[str] = Field(
        default_factory=list,
        description=(
            "Calls that failed and were absorbed by a fallback. An outage and a "
            "genuine absence of evidence produce the same answer; only this "
            "tells them apart."
        ),
    )

    # --- Observability -------------------------------------------------------
    spans: list[dict[str, Any]] = Field(default_factory=list)
    span_summary: dict[str, Any] = Field(default_factory=dict)
    scratchpad: list[dict[str, Any]] = Field(default_factory=list)

    # --- Request identity ----------------------------------------------------
    latency_ms: int | None = None
    conversation_id: str | None = None
    subject: str | None = Field(
        default=None, description="A hashed, non-identifying caller id."
    )

    @property
    def grounded(self) -> bool:
        """True when the answer rests on retrieved evidence."""
        return self.source_used in ("private_kb", "web_search")

    @property
    def audited_question(self) -> str:
        """What the audit log should keep.

        The redacted form when redaction changed something, otherwise the
        original. One property rather than a branch at the call site, because
        the rule has to be the same for the audit row, the scratchpad and the
        cache key, and three copies of it is two chances to keep the wrong one.
        """
        return self.redacted_question or self.question

    @property
    def partial(self) -> bool:
        """True when a budget stopped the run before it was finished."""
        return bool(self.stop_reason)


def to_answer(state: AgentState | dict[str, Any]) -> AnswerResult:
    """Project the graph's final state onto the API contract."""
    kb_docs = state.get("kb_docs") or []
    question = state.get("question", "")
    current_query = state.get("current_query")
    # `retrieve_kb` writes citations beside the chunks. Recomputing from
    # kb_docs is the fallback for a state that never went through that node --
    # a partial state in a test, or one built before citations were threaded.
    citations = state.get("citations")
    if citations is None:
        citations = cite_sources(kb_docs)
    # Reported only when it actually differs, for the same reason
    # `rewritten_query` is: a field that always echoes the question tells the UI
    # nothing, and a field that is set only when something happened is a signal.
    standalone = state.get("standalone_question")
    baseline = standalone or question
    trace = list(state.get("trace") or [])
    redacted = state.get("redacted_question")
    return AnswerResult(
        question=question,
        answer=(state.get("answer") or "").strip(),
        route=state.get("route"),
        source_used=state.get("source_used"),
        retry_count=state.get("retry_count", 0),
        # Present only when redaction changed something: a field that always
        # echoes the question tells the UI nothing, the same reasoning
        # `rewritten_query` and `resolved_question` already follow.
        redacted_question=redacted if redacted and redacted != question else None,
        guard_input=dict(state.get("guard_input") or {}),
        guard_output=dict(state.get("guard_output") or {}),
        pii_redacted=list((state.get("guard_input") or {}).get("redacted") or []),
        model=state.get("model_used"),
        model_tier=state.get("model_tier"),
        routing_reason=state.get("model_reason"),
        complexity=state.get("complexity"),
        cached=bool(state.get("cache_hit")),
        cache_similarity=state.get("cache_similarity"),
        # A degraded call writes "<tag>: failed (...)" or "... rate limited ..."
        # into the trace via `guard()` in `app/agent/nodes.py`. Derived here
        # rather than threaded through state because the trace is already the
        # single record of it and a second channel is a second thing to keep in
        # step.
        degraded=[
            step
            for step in trace
            if "continuing with fallback" in step or "rate limited" in step
        ],
        # Against the standalone question, not the original. `contextualize`
        # already moved current_query once for a follow-up, and comparing with
        # the employee's verbatim wording would report every follow-up as a
        # rewrite that never happened.
        rewritten_query=current_query if current_query != baseline else None,
        resolved_question=standalone if standalone and standalone != question else None,
        kb_grade=state.get("kb_grade"),
        web_grade=state.get("web_grade"),
        kb_chunks=len(kb_docs),
        sources=[SourceRef(**ref) for ref in citations],
        web_urls=state.get("web_urls") or [],
        trace=trace,
    )


class Copilot:
    """The seam every caller uses. One question in, one `AnswerResult` out.

    Dependencies are injectable for the same reason the nodes' are: a test
    builds a Copilot over fakes and never touches the network.

    **It is a facade over `app/services/pipeline.py`, not the pipeline itself.**
    Guardrails, the semantic cache, model routing, the choice of orchestration
    and the cost accounting all live there, because this module is also the
    place `AnswerResult` is defined -- the API contract -- and a contract
    sharing a file with three hundred lines of orchestration is a contract
    nobody reads. What stays here is the surface: construct with whatever
    clients you have, call `ask`, get the contract back.

    The signature is unchanged from before the pipeline existed, and the
    `graph` property with it. That matters: `scripts/demo.py` and the API tests
    both build a Copilot over an injected graph of fakes, and the point of
    adding ten stages around the graph was not to make the graph harder to
    exercise on its own.
    """

    def __init__(
        self,
        graph: Any = None,
        verbose: bool = False,
        settings: Settings | None = None,
        *,
        pipeline: Any = None,
        **build_kwargs: Any,
    ) -> None:
        self._graph = graph
        self._verbose = verbose
        self._settings = settings or get_settings()
        self._build_kwargs = build_kwargs
        self._pipeline = pipeline

    @property
    def graph(self) -> Any:
        """The compiled graph, built on first use.

        Still here, and still built once, for a caller that wants the graph
        rather than the pipeline -- `scripts/demo.py` prints the state, and the
        tests assert against it directly.
        """
        if self._graph is None:
            log.info("Building the agent graph")
            self._graph = build_graph(
                verbose=self._verbose, settings=self._settings, **self._build_kwargs
            )
        return self._graph

    @property
    def pipeline(self) -> Any:
        """The answer pipeline, built on first use.

        Imported inside the property rather than at module scope, and that is
        not laziness: `app/services/pipeline.py` imports `AnswerResult` from
        *this* module, so a module-scope import here is a cycle. The contract
        has to be importable without the orchestration -- `app/services/audit.py`
        imports it and has no business pulling in the guards, the cache and the
        router to write a row.
        """
        if self._pipeline is None:
            from app.services.pipeline import AnswerPipeline

            self._pipeline = AnswerPipeline(
                self.settings,
                graph=self._graph,
                verbose=self._verbose,
                **self._build_kwargs,
            )
        return self._pipeline

    @property
    def settings(self) -> Settings:
        return self._settings

    def ask(
        self,
        question: str,
        history: Any = None,
        *,
        conversation_id: str | None = None,
        principal: Any = None,
        mode: str | None = None,
        byok: bool = False,
        stream: Any = None,
    ) -> AnswerResult:
        """Answer one question. Raises ValueError on an empty question.

        `history` is the conversation so far, oldest first, as
        `{"role": ..., "content": ...}` turns. The Copilot itself is stateless
        -- the caller owns the conversation and hands back as much of it as it
        wants resolved against. That is deliberate: one process serves every
        employee, so holding conversations here would mean holding one person's
        HR questions in memory on behalf of a request from someone else.

        `mode` forces one orchestration (`graph`, `react`, `supervisor`,
        `extractive`) for this question only, overriding `AGENT_MODE`. It is
        what the console's mode switch sends, and it is per-question rather
        than per-deployment because the interesting comparison is the same
        question answered three ways.

        `stream` is an `EventStream` the pipeline emits stages into. The answer
        is identical with and without it.
        """
        return self.pipeline.answer(
            question,
            history=history,
            conversation_id=conversation_id,
            principal=principal,
            mode=mode,
            byok=byok,
            stream=stream,
        )

    def ask_graph_only(self, question: str, history: Any = None) -> AnswerResult:
        """Run the bare graph, with none of the stages around it.

        Kept because it is the baseline every claim about the pipeline is
        measured against: "the cache saved 40% of calls" and "routing did not
        hurt grounding" are both differences from *this*, and a codebase that
        cannot run the thing it is comparing to has numbers nobody can check.
        `scripts/demo.py --graph-only` uses it.
        """
        state = run_graph(
            question, graph=self.graph, verbose=self._verbose, history=history
        )
        return to_answer(state)

    def warmup(self) -> None:
        """Build the pipeline ahead of the first request, at API startup.

        Touches the retriever, which is what loads the embedding model, so this
        is the call that moves a ~5-second first-question penalty to startup.
        Deliberately not called by `create_app`: a container that must answer
        `/health` before it is marked ready should not spend that time before
        it can report anything at all.
        """
        _ = self.pipeline.retriever


# There is deliberately no module-level `get_copilot()` singleton here. It
# existed, unused, and it was a second answer to a question this project has
# already answered: the application's Copilot is built in `create_app` and read
# off `app.state` by `app/api/deps.py`, and a CLI constructs its own
# `Copilot(...)`. A process-wide global alongside those is not a shortcut, it is
# a third wiring path that takes the process-wide Settings rather than the ones
# its caller was given -- which is exactly the coupling the injection everywhere
# else in this codebase exists to avoid.


# --- Console rendering -------------------------------------------------------
# Returns a string rather than printing: library code logs and returns, only an
# entry point prints.

RULE = "=" * 86
THIN = "-" * 86


def render_answer(
    result: AnswerResult, show_sources: bool = True, show_trace: bool = True
) -> str:
    """Format an answer and its provenance for a terminal."""
    lines = [
        RULE,
        f"QUESTION      {result.question}",
    ]
    if result.rewritten_query:
        lines.append(f"REWRITTEN AS  {result.rewritten_query}")
    lines += [
        THIN,
        f"ROUTE         {result.route or '-'}",
        f"SOURCE USED   {result.source_used or '-'}",
        f"RETRIES       {result.retry_count}",
        f"KB EVIDENCE   {result.kb_chunks} chunk(s), graded {result.kb_grade or '-'}",
        f"WEB EVIDENCE  {len(result.web_urls)} url(s), graded {result.web_grade or '-'}",
    ]
    if show_sources and result.sources:
        lines.append(f"KB SOURCES    {', '.join(s.source for s in result.sources)}")
    if show_trace and result.trace:
        lines.append(THIN)
        lines.append("TRACE")
        lines += [f"  {n}. {step}" for n, step in enumerate(result.trace, 1)]
    lines += [THIN, "ANSWER", result.answer, RULE]
    return "\n".join(lines)
