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


class AnswerResult(BaseModel):
    """A finished answer plus the trace of how the agent reached it."""

    question: str
    answer: str
    route: str | None = None
    source_used: str | None = None
    retry_count: int = 0
    rewritten_query: str | None = Field(
        default=None,
        description="Set only when a rewrite actually changed the query.",
    )
    kb_grade: str | None = None
    web_grade: str | None = None
    kb_chunks: int = 0
    sources: list[SourceRef] = Field(default_factory=list)
    web_urls: list[str] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True when the answer rests on retrieved evidence."""
        return self.source_used in ("private_kb", "web_search")


def to_answer(state: AgentState | dict[str, Any]) -> AnswerResult:
    """Project the graph's final state onto the API contract."""
    kb_docs = state.get("kb_docs") or []
    question = state.get("question", "")
    current_query = state.get("current_query")
    return AnswerResult(
        question=question,
        answer=(state.get("answer") or "").strip(),
        route=state.get("route"),
        source_used=state.get("source_used"),
        retry_count=state.get("retry_count", 0),
        rewritten_query=current_query if current_query != question else None,
        kb_grade=state.get("kb_grade"),
        web_grade=state.get("web_grade"),
        kb_chunks=len(kb_docs),
        sources=[SourceRef(**ref) for ref in cite_sources(kb_docs)],
        web_urls=state.get("web_urls") or [],
    )


class Copilot:
    """Holds one compiled graph and answers questions with it.

    Dependencies are injectable for the same reason the nodes' are: a test
    builds a Copilot over fakes and never touches the network.
    """

    def __init__(
        self,
        graph: Any = None,
        verbose: bool = False,
        settings: Settings | None = None,
        **build_kwargs: Any,
    ) -> None:
        self._graph = graph
        self._verbose = verbose
        self._settings = settings or get_settings()
        self._build_kwargs = build_kwargs

    @property
    def graph(self) -> Any:
        """The compiled graph, built on first use."""
        if self._graph is None:
            log.info("Building the agent graph")
            self._graph = build_graph(
                verbose=self._verbose, settings=self._settings, **self._build_kwargs
            )
        return self._graph

    def ask(self, question: str) -> AnswerResult:
        """Answer one question. Raises ValueError on an empty question."""
        state = run_graph(question, graph=self.graph, verbose=self._verbose)
        return to_answer(state)

    def warmup(self) -> None:
        """Build the graph ahead of the first request, at API startup."""
        _ = self.graph


_copilot: Copilot | None = None


def get_copilot() -> Copilot:
    """Process-wide Copilot, for use as a FastAPI dependency."""
    global _copilot
    if _copilot is None:
        _copilot = Copilot()
    return _copilot


# --- Console rendering -------------------------------------------------------
# Returns a string rather than printing: library code logs and returns, only an
# entry point prints.

RULE = "=" * 86
THIN = "-" * 86


def render_answer(result: AnswerResult, show_sources: bool = True) -> str:
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
    lines += [THIN, "ANSWER", result.answer, RULE]
    return "\n".join(lines)
