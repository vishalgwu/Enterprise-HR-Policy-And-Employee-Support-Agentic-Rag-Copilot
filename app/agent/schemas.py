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


# --- Agentic decisions -------------------------------------------------------
# The decisions the tool-choosing and multi-agent orchestrations make. Same
# rule as the two above: a branch cannot be driven by a sentence of prose, so
# each is a Pydantic model bound with `with_structured_output`, and each has a
# safe default in `app.agent.decisions` for the call that comes back malformed.

# The specialists the supervisor may delegate to. Literal rather than str so a
# hallucinated worker name is rejected by the schema instead of routing to a
# node that does not exist. `general` is the catch-all, and `escalate` is the
# only one that ends in a human rather than an answer.
Worker = Literal[
    "policy", "benefits", "payroll", "workplace", "web", "general", "escalate"
]

Complexity = Literal["simple", "standard", "complex"]


class ReActDecision(BaseModel):
    """One turn of a ReAct loop: think, then either act or answer.

    `action` and `final_answer` are mutually exclusive in intent but not in the
    schema, because a model that fills both is a thing that happens and
    rejecting it would cost the turn. `app.agent.react` resolves the conflict in
    favour of the action -- an agent that has decided to look something up has
    told you its answer is not ready.
    """

    thought: str = Field(
        default="",
        description=(
            "One sentence on what you now know and what you need next. Not the "
            "answer, and not a restatement of the question."
        ),
    )
    action: str = Field(
        default="",
        description=(
            "The exact name of one tool to call, or empty if you can answer now."
        ),
    )
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for that tool, matching its parameter schema.",
    )
    final_answer: str = Field(
        default="",
        description=(
            "The answer for the employee, or empty if a tool call is needed "
            "first. Cite the source documents you used."
        ),
    )


class DelegationDecision(BaseModel):
    """The supervisor's next move: a worker to hand to, or compose now."""

    worker: Worker = Field(
        description=(
            "Which specialist should look at this next. 'escalate' only when a "
            "human must decide. Pick the one whose documents hold the answer."
        )
    )
    reason: str = Field(
        default="",
        description="One clause on why this worker and not another. For the trace.",
    )
    done: bool = Field(
        default=False,
        description=(
            "True when the findings already answer the question and no further "
            "delegation is needed."
        ),
    )


class SplitDecision(BaseModel):
    """Whether one question is really several, and along which seams.

    The question behind "when to split": a question that spans two policy areas
    is answered badly by one retrieval over both, because the top-k fills with
    whichever area is better represented in the corpus. Splitting it costs an
    extra retrieval and buys an answer that covers both halves.
    """

    needs_split: bool = Field(
        description=(
            "True only when the question asks about two or more separate policy "
            "areas that different documents cover."
        )
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description=(
            "The question as two to three self-contained questions, each "
            "answerable from one policy area. Empty when needs_split is false."
        ),
    )
    reason: str = Field(default="", description="One clause, for the trace.")


class ComplexityDecision(BaseModel):
    """How much model a question needs. Drives routing, not retrieval."""

    complexity: Complexity = Field(
        description=(
            "simple for a greeting or a single-fact lookup; standard for an "
            "ordinary policy question; complex for one spanning several "
            "policies, needing arithmetic, or asking for a comparison."
        )
    )


class RelevanceScores(BaseModel):
    """The LLM re-ranker's verdict: one score per candidate, in order.

    A list rather than a mapping because a model asked for a dict of document
    ids reliably invents ids; asked for N numbers in order, it returns N numbers
    in order. `app.rag.rerank` pads or truncates to the candidate count rather
    than trusting the length.
    """

    scores: list[int] = Field(
        default_factory=list,
        description=(
            "0-10 for each numbered passage, in the same order, where 10 means "
            "it directly states the fact the question asks for and 0 means it "
            "is unrelated. One number per passage."
        ),
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

    # --- Added by the production layers around the graph ---------------------
    # All NotRequired, and all written by exactly one place, so a graph run
    # without the surrounding pipeline (every existing test) behaves as before.
    #
    # `redacted_question` is what left the process when `REDACT_PII` is on, and
    # is what the audit log stores by default. It exists beside `question`
    # rather than replacing it because the employee's own wording is what the UI
    # shows back to them, and showing someone their own question with their own
    # phone number starred out is confusing rather than careful.
    redacted_question: NotRequired[str]
    # Guardrail verdicts as `GuardVerdict.to_dict()`. Carried in state rather
    # than returned separately so one audit row holds the question, the answer
    # and the reason either of them was altered.
    guard_input: NotRequired[dict[str, Any]]
    guard_output: NotRequired[dict[str, Any]]
    # Which model actually answered, and why that one. Set by the router in
    # `app.services.routing`; reported on every answer because "which model
    # answered?" is the first question about a bad one.
    model_used: NotRequired[str]
    model_tier: NotRequired[str]
    model_reason: NotRequired[str]
    complexity: NotRequired[Complexity]
    # Whether the answer came from the semantic cache rather than from a run.
    # A cache hit is a materially different thing to audit: nothing was
    # retrieved and nothing was graded on this request.
    cache_hit: NotRequired[bool]
    cache_similarity: NotRequired[float]


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
