"""Structured decision types and the state threaded through the graph.

The graph branches on these decisions, so they have to be machine-readable: a
sentence of prose is unusable as a routing edge. Each decision is a Pydantic
model bound to the LLM with `with_structured_output`, which constrains the model
to the schema instead of parsing free text after the fact.

Do not switch these to `method="json_mode"`. It is unconstrained JSON and this
model fills the enum field with prose -- verified: a route query came back as
`{"route": "Employees accrue 1.5 days of PTO per month..."}`, which is exactly
the failure structured output exists to prevent. The default (function calling)
enforces the enum and is also the fastest of the working methods; `json_schema`
works too.
"""

from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

Route = Literal["kb", "direct"]
Grade = Literal["good", "weak"]

# Provenance of the finished answer, kept separate from `route`: the router's
# choice and what actually produced the answer are different facts, and a run
# that routes to "kb" can still end up answering from the web.
SourceUsed = Literal[
    "private_kb", "web_search", "direct", "insufficient_evidence", "error"
]


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
    replaces. Keeping them separate is deliberate: the original survives for the
    final answer, the citations and the logs. Collapsing them loses what was
    actually asked.

    Keys are NotRequired because nodes return partial updates -- a fresh state
    only needs what `initial_state` seeds.
    """

    question: str
    current_query: str
    retry_count: int
    route: NotRequired[Route]
    kb_docs: NotRequired[List[Document]]
    web_results: NotRequired[str]
    web_urls: NotRequired[List[str]]
    kb_grade: NotRequired[Grade]
    web_grade: NotRequired[Grade]
    answer: NotRequired[str]
    source_used: NotRequired[SourceUsed]


def initial_state(question: str) -> AgentState:
    """Seed a state so callers cannot forget current_query or retry_count."""
    return AgentState(question=question, current_query=question, retry_count=0)
