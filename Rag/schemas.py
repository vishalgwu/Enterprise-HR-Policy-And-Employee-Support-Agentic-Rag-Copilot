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

from Rag.clients import get_llm
from Rag.config import configure_stdout
from Rag.loaders import format_documents

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
SourceUsed = Literal["private_kb", "web_search", "direct", "insufficient_evidence"]

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


def route_question(question: str, router=None) -> Route:
    return (router or get_router()).invoke({"question": question}).route


def grade_evidence(question: str, evidence: str, grader=None) -> Grade:
    """Grade evidence. Empty evidence is 'weak' without spending a call."""
    if not evidence.strip():
        return "weak"
    return (grader or get_evidence_grader()).invoke(
        {"question": question, "evidence": evidence}
    ).grade


def main():
    configure_stdout()
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
