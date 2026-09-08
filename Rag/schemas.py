"""Structured decision types for the agent graph.

The graph branches on these decisions, so they have to be machine-readable: a
sentence of prose is unusable as a routing edge. Each decision is a Pydantic
model bound to the LLM with `with_structured_output`, which constrains the model
to the schema rather than parsing free text after the fact.

Do not switch these to `method="json_mode"`. It is unconstrained JSON and this
model fills the enum field with prose -- verified: a route query came back as
`{"route": "Employees accrue 1.5 days of PTO per month..."}`. The default
(function calling) enforces the enum.
"""

if __package__ in (None, ""):  # allow `python Rag/schemas.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from Rag.clients import get_llm, is_rate_limit
from Rag.config import configure_logging, configure_stdout, get_logger
from Rag.loaders import format_documents

log = get_logger("schemas")

# format_documents lives in loaders but is re-exported here: callers building a
# grading prompt want the decision types and the evidence renderer together.
__all__ = [
    "AgentState",
    "EvidenceGrade",
    "Grade",
    "MAX_RETRIES",
    "Route",
    "RouteDecision",
    "SourceUsed",
    "format_documents",
    "get_evidence_grader",
    "get_router",
    "grade_evidence",
    "initial_state",
    "route_question",
]

Route = Literal["kb", "direct"]
Grade = Literal["good", "weak"]
# Provenance of the finished answer, kept separate from `route`: the router's
# choice and what actually produced the answer are different facts, and a run
# that routes to "kb" can still end up answering from the web.
SourceUsed = Literal[
    "private_kb", "web_search", "direct", "insufficient_evidence", "error"
]

# One rewrite-and-retry before escalating to web search. Higher values mostly
# buy latency: if a rewrite does not fix retrieval, a second rarely does.
MAX_RETRIES = 1


class RouteDecision(BaseModel):
    """Whether a message needs a policy lookup at all."""

    route: Route = Field(
        description=(
            "kb if answering requires internal HR policy documents. "
            "direct for greetings, thanks, small talk, or anything answerable "
            "without looking up policy."
        )
    )


class EvidenceGrade(BaseModel):
    """Whether retrieved evidence can actually answer the question."""

    grade: Grade = Field(
        description=(
            "good if the evidence contains the specific facts needed to answer "
            "the question. weak if it is off-topic, generic, or missing the "
            "specific detail asked for."
        )
    )


class AgentState(TypedDict):
    """State threaded through the graph.

    `question` is the employee's original wording and never changes;
    `current_query` is what retrieval actually runs and is what a rewrite
    replaces, so the original survives for the final answer and for logging.

    Keys are NotRequired because nodes return partial updates -- a fresh state
    only needs what `initial_state` seeds.
    """

    question: str
    current_query: str
    retry_count: int
    route: NotRequired[Route]
    kb_docs: NotRequired[List[Document]]
    web_results: NotRequired[str]
    kb_grade: NotRequired[Grade]
    web_grade: NotRequired[Grade]
    answer: NotRequired[str]
    source_used: NotRequired[SourceUsed]


def initial_state(question: str) -> AgentState:
    """Seed a state so callers cannot forget current_query or retry_count."""
    return AgentState(question=question, current_query=question, retry_count=0)


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route messages for an internal HR assistant.\n"
            "Answer 'direct' for messages that need no evidence lookup: "
            "greetings, thanks, farewells, small talk, and questions about who "
            "or what this assistant is and what it can help with.\n"
            "Answer 'kb' for every message seeking a fact -- company policy, "
            "benefits, leave, pay, expenses, conduct, and equally any question "
            "about the outside world. Downstream stages check whether the "
            "knowledge base covers it and fall back to web search when it does "
            "not, so routing to 'kb' is never wasted.\n"
            "Never route an external factual question to 'direct': that makes "
            "the assistant answer from memory with no evidence behind it, which "
            "is the one thing it must not do.",
        ),
        ("human", "{question}"),
    ]
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether the supplied evidence can answer the question.\n"
            "Answer 'good' only if the evidence states the specific facts the "
            "question asks for.\n"
            "Answer 'weak' if the evidence is off-topic, only generally "
            "related, or omits the specific detail requested.\n"
            "Judge the evidence as given. Do not use outside knowledge, and do "
            "not credit evidence for being merely on the same broad subject.",
        ),
        ("human", "Question:\n{question}\n\nEvidence:\n{evidence}"),
    ]
)


def get_router(llm=None):
    """Runnable mapping {"question": str} to a RouteDecision."""
    return ROUTER_PROMPT | (llm or get_llm()).with_structured_output(RouteDecision)


def get_evidence_grader(llm=None):
    """Runnable mapping {"question": str, "evidence": str} to an EvidenceGrade."""
    return GRADER_PROMPT | (llm or get_llm()).with_structured_output(EvidenceGrade)


def _failed_generation(exc) -> str | None:
    """Recover the model's raw text from a Groq tool_use_failed error.

    Groq returns 400 `tool_use_failed` when the model answers in plain text
    instead of calling the tool the structured output binds. It echoes what the
    model actually said in `error.failed_generation`, and that text is usually
    the correct value -- an observed failure carried exactly "direct".
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            generated = error.get("failed_generation")
            if isinstance(generated, str):
                return generated
    return None


def _coerce(text: str | None, allowed: tuple[str, ...]) -> str | None:
    """Match loose model text against an enum: 'direct', '"direct"', {"route": "direct"}."""
    if not text:
        return None
    lowered = text.strip().strip("\"' \n\t.").lower()
    for value in allowed:
        if lowered == value:
            return value
    # Fall back to containment so a JSON blob or a short sentence still resolves,
    # but only when exactly one candidate appears -- otherwise it is a guess.
    hits = [v for v in allowed if v in lowered]
    return hits[0] if len(hits) == 1 else None


def route_question(question: str, router=None) -> Route:
    """Route one message, surviving a structured-output failure.

    Falls back to 'kb' rather than 'direct': routing an unclassifiable message
    to retrieval costs a lookup, whereas defaulting to 'direct' would make the
    assistant answer from memory with no evidence behind it.
    """
    try:
        return (router or get_router()).invoke({"question": question}).route
    except Exception as exc:  # noqa: BLE001 - any client error must not crash the graph
        recovered = _coerce(_failed_generation(exc), ("kb", "direct"))
        if recovered:
            return recovered  # type: ignore[return-value]
        if is_rate_limit(exc):
            log.error("[Router] RATE LIMITED; defaulting to kb (%s)", str(exc)[:160])
        else:
            log.warning("[Router] structured output failed (%s); defaulting to kb",
                        type(exc).__name__)
        return "kb"


def grade_evidence(question: str, evidence: str, grader=None) -> Grade:
    """Grade evidence. Empty evidence is 'weak' without spending a call.

    Falls back to 'weak' on a structured-output failure, so an unreadable grade
    sends the graph to its fallback path instead of letting an ungraded answer
    through.
    """
    if not evidence.strip():
        return "weak"
    try:
        return (grader or get_evidence_grader()).invoke(
            {"question": question, "evidence": evidence}
        ).grade
    except Exception as exc:  # noqa: BLE001
        recovered = _coerce(_failed_generation(exc), ("good", "weak"))
        if recovered:
            return recovered  # type: ignore[return-value]
        if is_rate_limit(exc):
            log.error("[Grader] RATE LIMITED; grading weak by default, which is a "
                      "quota failure and not a judgement (%s)", str(exc)[:160])
        else:
            log.warning("[Grader] structured output failed (%s); defaulting to weak",
                        type(exc).__name__)
        return "weak"


def main():
    configure_stdout()
    configure_logging()
    llm = get_llm()
    router = get_router(llm)
    grader = get_evidence_grader(llm)

    print("--- ROUTER ---")
    for q in [
        "hi there",
        "thanks, that helps!",
        "How many PTO days do I get per year?",
        "Can I work remotely from another country?",
    ]:
        print(f"  {route_question(q, router):7} <- {q!r}")

    print("\n--- GRADER ---")
    pto_evidence = (
        "[1] source=leave-and-time-off.md\n"
        "Full-time employees accrue 20 days of paid time off per calendar year."
    )
    expense_evidence = (
        "[1] source=expense-reimbursement.md\n"
        "Expenses must be submitted in Expensify within 60 days."
    )
    question = "How many PTO days do I get per year?"
    print(f"  {grade_evidence(question, pto_evidence, grader):5} <- matching evidence")
    print(f"  {grade_evidence(question, expense_evidence, grader):5} <- unrelated evidence")
    print(f"  {grade_evidence(question, '', grader):5} <- empty evidence (no LLM call)")

    print(f"\n--- STATE ---\n  {initial_state('How many PTO days do I get?')}")


if __name__ == "__main__":
    main()
