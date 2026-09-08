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

from collections.abc import Mapping, Sequence
from operator import add
from typing import Annotated, Any, Literal, NotRequired

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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

    Most keys are NotRequired because nodes return partial updates -- a fresh
    state only needs what `initial_state` seeds.

    **`trace` is the one key that accumulates rather than overwrites.** A plain
    TypedDict key is *replaced* by whatever a node returns, so a bare
    `list[str]` would leave only the last node's entries and the rewrite loop
    would erase its own history -- exactly the run whose path is worth seeing.
    The `add` reducer in the `Annotated` metadata is what makes each node's
    return append instead, so nodes return only their own new steps. It is a
    required key, seeded empty by `initial_state`, because a reducer has nothing
    to fold into on a key that may be absent.

    **`citations` deliberately does not accumulate.** It is derived from
    `kb_docs`, and a rewrite loops back through `retrieve_kb` and *replaces*
    those chunks; accumulating would cite documents that lost their evidence on
    the retry. Overwriting alongside `kb_docs` keeps the two describing the same
    retrieval.

    **Three question-shaped keys, each with a different job.** They look
    redundant and are not:

    - `question` is what the employee typed, verbatim. It never changes, and it
      is what the audit log and the UI show.
    - `standalone_question` is `question` resolved against `history`, so
      "what about contractors?" becomes a question that means something on its
      own. Set once by `contextualize`; it is what the router, the graders and
      generation all see. Equal to `question` when there is no history, which is
      why a single-turn run behaves exactly as it did before.
    - `current_query` is what retrieval actually runs. It starts as
      `standalone_question` and is what a rewrite replaces, over and over.

    Collapsing any pair loses something real: without the first the audit log
    records words nobody typed, without the second a follow-up retrieves
    nothing, without the third a rewrite would overwrite the question itself.
    """

    question: str
    current_query: str
    retry_count: int
    trace: Annotated[list[str], add]
    # Prior turns, oldest first, as {"role": "user"|"assistant", "content": str}.
    # Required and seeded empty: `contextualize` branches on whether it is empty
    # and must never meet a missing key.
    history: list[dict[str, str]]
    standalone_question: NotRequired[str]
    route: NotRequired[Route]
    kb_docs: NotRequired[list[Document]]
    citations: NotRequired[list[dict[str, str]]]
    web_results: NotRequired[str]
    web_urls: NotRequired[list[str]]
    kb_grade: NotRequired[Grade]
    web_grade: NotRequired[Grade]
    answer: NotRequired[str]
    source_used: NotRequired[SourceUsed]


def normalise_history(
    history: Sequence[Mapping[str, Any]] | None, limit: int = 8
) -> list[dict[str, str]]:
    """Keep the last `limit` well-formed turns, oldest first.

    History arrives over HTTP from a browser, so it is untrusted input in both
    the security sense and the shape sense: rows missing a role, a role nobody
    recognises, `content` that is not a string. Anything unusable is dropped
    rather than allowed to reach a prompt.

    The cap is a cost control, not tidiness. Every turn is rendered into the
    contextualise prompt on *every* follow-up, so an unbounded history means a
    conversation that gets more expensive the longer it runs. Eight turns is
    four exchanges, which is well past where a pronoun still refers to
    something.
    """
    if not history:
        return []
    clean: list[dict[str, str]] = []
    for turn in history:
        role = str(turn.get("role", "")).strip().lower()
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        clean.append({"role": role, "content": content.strip()})
    return clean[-limit:]


def initial_state(
    question: str, history: Sequence[Mapping[str, Any]] | None = None
) -> AgentState:
    """Seed a state so callers cannot forget current_query, retry_count, the
    trace the reducer folds into, or the history `contextualize` branches on."""
    return AgentState(
        question=question,
        current_query=question,
        retry_count=0,
        trace=[],
        history=normalise_history(history),
    )
