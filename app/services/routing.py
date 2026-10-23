"""Model routing and cost projection: the cheapest real lever, and its risk.

Every question in this system currently pays for the same model. "Hi" pays for
it, "thanks" pays for it, and so does a two-area prorated-accrual question that
genuinely needs the better one. Routing is the largest cost saving available
here that does not involve answering fewer questions -- and it is also the one
most likely to quietly damage quality, because a cheap model on a hard question
produces a plausible answer, not an error. Nothing in the logs says "this one
should have gone to the big model".

So three things hold it in place:

* **The classifier's uncertain answer is the middle tier, never the cheap one.**
  A classifier whose default is "fast" is a classifier that silently degrades
  exactly the questions it failed to understand. `classify_complexity` returns
  `"standard"` unless a signal fires.
* **The decision is reported on every answer.** `RoutingDecision.to_dict()` goes
  into the trace and the audit row, with the reason in words, so "why did that
  answer come out thin?" has an answer that does not require re-running it.
* **It switches off.** `MODEL_ROUTING_ENABLED=0` pins every question to
  `active_llm_model`, which is the unrouted baseline quality has to be measured
  against. A saving nobody measured against a baseline is not a saving.

The classifier is a heuristic and runs offline. Asking a model which model to
use costs a model call, which is most of what routing was trying to save on a
short question, and `COMPLEXITY_PROMPT` exists in `app/agent/prompts.py` for the
deployment that would rather pay it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings, get_logger, get_settings

log = get_logger("routing")

__all__ = [
    "MODEL_CATALOG",
    "ModelRouter",
    "ModelSpec",
    "RoutingDecision",
    "Tier",
    "classify_complexity",
    "estimate_tokens",
    "llm_for",
    "model_spec",
    "project_cost",
]

Tier = Literal["fast", "balanced", "deep"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model, its tier, and what a million tokens through it costs."""

    name: str
    provider: str  # "groq" | "openai"
    tier: Tier
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    context_window: int
    note: str = ""


# ---------------------------------------------------------------------------
# PRICES ARE A PROJECTION, NOT AN INVOICE.
#
# These are published list prices, they move, and nothing here reconciles them
# against anything. They exist to drive two things that tolerate being roughly
# right: the `cost` ceiling in `app.agent.budgets`, which only has to stop a
# runaway, and a number on a dashboard, which only has to show that one model
# costs twenty times another. They are multiplied by `estimate_tokens`, which is
# itself +/-20%, so the product is an order-of-magnitude figure with decimal
# places -- read it as a ceiling.
#
# Two specific reasons not to trust it as a bill:
#
#   * Groq's free tier bills nothing at all. It caps tokens per *day* instead
#     (observed: `429 ... tokens per day (TPD): Limit 200000, Used 199771`), so
#     on the default provider this projection is a token-burn estimate wearing a
#     dollar sign, and the thing that actually stops a run is the quota.
#   * Agentic endpoints like `groq/compound` are billed through whichever
#     underlying models they chose, which is not knowable from here.
#
# The number to trust is the provider's own billing page. Where a current price
# is not something to be confident about, the `note` field says so rather than
# the figure implying a precision it does not have.
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, ModelSpec] = {
    "openai/gpt-oss-20b": ModelSpec(
        name="openai/gpt-oss-20b",
        provider="groq",
        tier="fast",
        input_usd_per_mtok=0.10,
        output_usd_per_mtok=0.50,
        context_window=131_072,
        note=(
            "This repo's GROQ_MODEL default. Price approximate -- the free tier "
            "bills nothing and caps tokens per day, so this figure is only used "
            "to make the projection comparable across tiers."
        ),
    ),
    "openai/gpt-oss-120b": ModelSpec(
        name="openai/gpt-oss-120b",
        provider="groq",
        tier="deep",
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.75,
        context_window=131_072,
        note=(
            "The escape hatch when the 20b daily quota is spent: limits are "
            "per-model. Price approximate, same caveat as the 20b."
        ),
    ),
    "groq/compound": ModelSpec(
        name="groq/compound",
        provider="groq",
        tier="deep",
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.75,
        context_window=131_072,
        note=(
            "An agentic endpoint that calls its own tools, so it is billed "
            "through whichever underlying models it used. Not confident of any "
            "single per-token price; these are the 120b's as a stand-in."
        ),
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini",
        provider="openai",
        tier="balanced",
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
        context_window=128_000,
        note="This repo's OPENAI_MODEL default. Published list price.",
    ),
    "gpt-4o": ModelSpec(
        name="gpt-4o",
        provider="openai",
        tier="deep",
        input_usd_per_mtok=2.50,
        output_usd_per_mtok=10.00,
        context_window=128_000,
        note=(
            "Roughly 17x gpt-4o-mini on input. Listed so the ladder has a top "
            "rung with a price that makes routing obviously worth doing."
        ),
    ),
    "text-embedding-3-small": ModelSpec(
        name="text-embedding-3-small",
        provider="openai",
        tier="fast",
        input_usd_per_mtok=0.02,
        output_usd_per_mtok=0.0,
        context_window=8_191,
        note=(
            "1536-d. Embeddings have no completion, hence an output price of "
            "zero. Here because the semantic cache and ingest both spend on it "
            "and a cost dashboard that omits them is wrong about the total."
        ),
    ),
    "text-embedding-3-large": ModelSpec(
        name="text-embedding-3-large",
        provider="openai",
        tier="balanced",
        input_usd_per_mtok=0.13,
        output_usd_per_mtok=0.0,
        context_window=8_191,
        note="3072-d, so it needs a different Pinecone index again.",
    ),
}


def model_spec(name: str) -> ModelSpec | None:
    """The catalog entry for a model id, or None if it is not listed."""
    return MODEL_CATALOG.get(name)


# Four characters per token is the honest approximation. It is about +/-20% on
# English prose, and worse on code, tables and the markdown pipe tables the
# generator emits -- all of which tokenize denser than words do.
#
# A real tokenizer is a dependency (`tiktoken`, and a different one per
# provider) plus a download, to improve a number that drives a budget ceiling
# and a dashboard. Both tolerate 20%: a run stopped at 115k estimated tokens
# instead of 120k actual ones stopped for the right reason.
CHARS_PER_TOKEN = 4

# What a completion is assumed to cost when projecting before the call. An HR
# answer from this system runs two to five short paragraphs; 400 tokens is the
# middle of that. It is an assumption, not a measurement, and it only moves the
# output half of a projection.
ASSUMED_COMPLETION_TOKENS = 400


def estimate_tokens(text: str) -> int:
    """Characters over four, rounded up. See `CHARS_PER_TOKEN` for the caveats."""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _tokens_for_chars(chars: int) -> int:
    """The same approximation for a caller that has a length, not a string."""
    if chars <= 0:
        return 0
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def project_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Projected US dollars for one call. Zero for a model not in the catalog.

    An unlisted model is logged at WARNING rather than priced silently. Zero is
    the right return -- inventing a price for a model nobody put in the table
    would put a made-up number in a budget ceiling -- but a cost dashboard
    reading 0.00 because the model id was renamed looks identical to one reading
    0.00 because the deployment is on Groq's free tier, and those want very
    different responses.
    """
    spec = model_spec(model)
    if spec is None:
        log.warning("No price for model %r; projecting 0.0 for this call", model)
        return 0.0
    return (
        max(0, input_tokens) * spec.input_usd_per_mtok
        + max(0, output_tokens) * spec.output_usd_per_mtok
    ) / 1_000_000


@dataclass(slots=True)
class RoutingDecision:
    """Which model answered, and the words explaining why.

    `reason` is prose because it is read by a person in the trace and the audit
    row, and `complexity` is kept beside it because the same complexity can map
    to different models across deployments -- recording only the model id would
    make two deployments' audit logs incomparable.
    """

    model: str
    tier: Tier
    provider: str
    reason: str
    complexity: str
    estimated_input_tokens: int = 0
    projected_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "provider": self.provider,
            "reason": self.reason,
            "complexity": self.complexity,
            "estimated_input_tokens": self.estimated_input_tokens,
            "projected_usd": round(self.projected_usd, 6),
        }


# --- The heuristic -----------------------------------------------------------

# A greeting, a thanks, or a question about the assistant itself. The last group
# belongs here rather than looking like a policy question for the same reason
# the router prompt keeps it on `direct`: "who are you?" retrieves nothing, and
# an earlier version of that prompt sent it to the KB and answered from an
# unrelated web page.
GREETING = re.compile(
    r"^(hi|hey|hello|yo|good (morning|afternoon|evening)|thanks|thank you|"
    r"thanks a lot|cheers|bye|goodbye|who are you|what are you|"
    r"what can you do)"
    # An optional form of address. "hello there" is a greeting and was
    # classified `standard` without this, which spends the middle model on the
    # word "hello". The list is closed rather than `\w+` on purpose: an open
    # trailing word would make "hi, what is my notice period" a greeting.
    r"(\s+(there|team|everyone|folks|all))?"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# Arithmetic and comparison, split into two strengths. These are the questions
# where a cheap model does not fail loudly: it produces a number, in the right
# format, that is wrong.
#
# **The split exists because one list was measurably wrong.** Everything below
# was originally one tuple, any match promoting to "complex" -- and "how many"
# is how the single most ordinary HR question there is begins. Measured against
# the console's own suggested questions, "How many days of paid time off do
# full-time employees get?" and "What is the notice period if I resign?" both
# came out complex, which routed two plain lookups to the deep model and, on
# `AGENT_MODE=auto`, to the supervisor -- four model calls for a question one
# retrieval answers. Meanwhile the genuinely hard one in the same set ("I'm a
# contractor going on parental leave mid-notice, what am I owed?") came out
# standard. The classifier was not slightly miscalibrated; it was inverted on
# exactly the cases that matter.
ARITHMETIC_STRONG = (
    "difference between",
    "versus",
    " vs ",
    " vs.",
    "compared to",
    "prorated",
    "pro rata",
    "pro-rata",
    "calculate",
    "work out",
    "add up",
)

# Interrogatives and conditionals. Each one is ordinary on its own and
# suggestive in company: "how much" is a lookup, "how much ... if I ..." is a
# calculation. One of these promotes only alongside a second signal.
ARITHMETIC_WEAK = (
    "how much",
    "how many",
    "how long",
    "if i",
    "more than",
    "less than",
    "carry over",
    "left over",
    "remaining",
    "total",
)

# Populations the knowledge base is *not* written for. This is the strongest
# single-area signal there is, and the reason is in the corpus rather than in
# the grammar: policy documents here say "employees", and whether a statement
# about employees covers a contractor is precisely the judgement the LLM grader
# is recorded as finding borderline (`CLAUDE.md` has the 8/8-then-flipped
# measurement). A question that names one of these needs the better model
# whatever else it looks like.
#
# "full-time" is deliberately absent: it is the population the corpus *is*
# written for, so naming it makes a question more ordinary, not less.
POPULATION = (
    "contractor",
    "contract staff",
    "consultant",
    "freelance",
    "intern",
    "part-time",
    "part time",
    "fixed-term",
    "fixed term",
    "probation",
    "secondment",
    "agency worker",
)

# The policy areas the knowledge base is organised around. Two distinct areas in
# one question is the strongest "complex" signal there is, and the one that also
# triggers splitting: the KB keeps "notice period" and "PTO payout on
# termination" in different documents, so a question touching both needs both
# retrieved and reconciled rather than one of them answered confidently.
POLICY_AREAS: dict[str, tuple[str, ...]] = {
    "leave": (
        "leave", "pto", "holiday", "vacation", "sick", "sabbatical",
        "maternity", "paternity", "parental", "time off", "carry over",
    ),
    # "accrual" and "accrue" are deliberately *not* here, though they look like
    # payroll words. In this corpus accrual is a leave concept -- "full-time
    # employees accrue 20 days of paid time off" -- so filing it under pay made
    # every ordinary PTO question report two policy areas and route to the deep
    # model. Measured on "Do full-time employees accrue paid time off?", which
    # came out complex for no reason a reader of the question would recognise.
    "pay": (
        "pay", "payroll", "salary", "wage", "bonus", "payslip", "increment",
        "reimburse", "expense", "form 16", "tax",
    ),
    "benefits": (
        "benefit", "insurance", "medical", "health", "pension", "provident",
        "wellness", "gratuity", "stipend",
    ),
    "conduct": (
        "conduct", "harassment", "grievance", "discipline", "code of conduct",
        "whistleblow", "conflict of interest", "ethic",
    ),
    "working-time": (
        "core hours", "working hours", "remote", "hybrid", "wfh", "shift",
        "overtime", "attendance",
    ),
    "joining-and-exit": (
        "onboarding", "offboarding", "notice period", "resign", "termination",
        "probation", "exit", "handover", "last working day", "ijp",
    ),
}

# Boundaries matter for a handful of these: "pay" is inside "paternity" and
# "company", and an area match that fires on a substring would report two areas
# for every question about parental leave.
_AREA_PATTERNS: dict[str, re.Pattern[str]] = {
    area: re.compile(
        "|".join(rf"\b{re.escape(term)}" for term in terms), re.IGNORECASE
    )
    for area, terms in POLICY_AREAS.items()
}

# A question long enough, or with enough clauses, that it is really several
# questions. 40 words is roughly three lines in the console's input.
LONG_WORDS = 40
MANY_CLAUSES = 4
_CLAUSE = re.compile(r"[,;]|\band\b|\bor\b|\bbut\b|\balso\b", re.IGNORECASE)

# How much question has to accompany a single weak signal for it to count.
# Below this, "how many days of leave do I get?" is what it looks like: a
# lookup with an interrogative on the front.
WEAK_WORDS = 18
WEAK_CLAUSES = 3

# How much history counts toward the area signal. Only the last exchange: a
# follow-up's real subject is split between what was typed and what it refers
# to ("what about contractors?"), but a topic from six turns ago is not the
# current question's second area, and counting it would mark every long
# conversation complex.
HISTORY_TURNS_CONSIDERED = 2


def _areas(text: str) -> set[str]:
    return {area for area, pattern in _AREA_PATTERNS.items() if pattern.search(text)}


def classify_complexity(
    question: str, history: Sequence[Mapping[str, Any]] | None = None
) -> tuple[str, str]:
    """A complexity band and a short clause saying why, with no model call.

    Returns one of `app.agent.schemas.Complexity` -- "simple", "standard" or
    "complex" -- plus a human reason for the trace.

    **The default is "standard", and that is the whole safety property.** Every
    branch below has to earn its answer; a question that matches nothing is not
    understood, and the only defensible thing to do with a question you have not
    understood is to spend the middle amount on it. A classifier that guessed
    "simple" here would be cheapest precisely on the questions it failed to
    read.
    """
    text = question.strip()
    if not text:
        return "standard", "empty question; defaulting to the middle tier"

    if GREETING.match(text):
        return "simple", "a greeting or a question about the assistant itself"

    lowered = text.casefold()
    areas = _areas(lowered)
    for turn in list(history or [])[-HISTORY_TURNS_CONSIDERED:]:
        content = turn.get("content")
        if isinstance(content, str):
            areas |= _areas(content.casefold())
    if len(areas) > 1:
        named = ", ".join(sorted(areas)[:3])
        return "complex", f"touches more than one policy area ({named})"

    strong = next(
        (phrase.strip() for phrase in ARITHMETIC_STRONG if phrase in lowered), None
    )
    if strong is not None:
        return "complex", f"asks for a comparison or a calculation ({strong!r})"

    population = next(
        (phrase for phrase in POPULATION if phrase in lowered), None
    )
    if population is not None and areas:
        return (
            "complex",
            f"asks about {population!r}, a population the policy documents do "
            f"not describe directly",
        )

    words = len(lowered.split())
    clauses = 1 + len(_CLAUSE.findall(lowered))
    if words >= LONG_WORDS or clauses >= MANY_CLAUSES:
        return "complex", f"{words} words in {clauses} clauses"

    # A weak signal is ordinary alone and suggestive in company. Two of them, or
    # one inside a question with some size to it, is a calculation rather than a
    # lookup: "how much leave" is a lookup, "how much leave do I carry over if I
    # started in March" is not.
    weak = [phrase.strip() for phrase in ARITHMETIC_WEAK if phrase in lowered]
    if weak and (len(weak) > 1 or words >= WEAK_WORDS or clauses >= WEAK_CLAUSES):
        return (
            "complex",
            f"asks for a figure under a condition ({', '.join(repr(w) for w in weak[:2])})",
        )

    return "standard", "no complexity signal; the middle tier is the safe default"


# --- The router --------------------------------------------------------------

_TIER_FOR_COMPLEXITY: dict[str, Tier] = {
    "simple": "fast",
    "standard": "balanced",
    "complex": "deep",
}

# How much projected cost must still fit for the ladder to be worth climbing.
# Below this, the run finishes on the cheap model rather than not finishing: an
# employee with a partial answer from a small model is better served than one
# whose question stopped because the budget ran out mid-ladder.
COST_HEADROOM_USD = 0.01


class ModelRouter:
    """Picks a model per question, and says why in words.

    `llm` is accepted and held, not used: `choose` is a heuristic with no model
    call, so there is nothing for a client to do here yet. It is in the
    signature because the LLM classifier this would otherwise grow -- using
    `COMPLEXITY_PROMPT`, the way `app.agent.decisions` does its structured
    calls -- should arrive as an injected client rather than as a signature
    change at every call site.
    """

    def __init__(self, settings: Settings | None = None, llm: Any = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm

    def _model_for(self, tier: Tier) -> str:
        """The configured id for a tier, falling back to the middle rung.

        An empty `FAST_MODEL` or `DEEP_MODEL` is documented in `Settings` as "no
        separate model for this tier", so it collapses the ladder onto
        `active_llm_model` rather than failing on an id the configured key
        cannot reach.
        """
        configured = {
            "fast": self.settings.fast_model,
            "balanced": self.settings.active_llm_model,
            "deep": self.settings.deep_model,
        }[tier]
        return configured.strip() or self.settings.active_llm_model

    def choose(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        context_chars: int = 0,
        tracer: Any = None,
        budget: Any = None,
    ) -> RoutingDecision:
        """Which model should answer this question.

        `context_chars` is the retrieved evidence that will go into the prompt.
        It is passed as a length rather than as text because that is all the
        projection needs, and because a span attribute holding policy chunks is
        content leaving the deployment.
        """
        # Classified even when routing is off: it costs nothing, it is the
        # number the unrouted baseline has to be compared against, and a
        # deployment measuring whether routing hurt quality needs both halves
        # recorded under the same name.
        complexity, reason = classify_complexity(question, history)

        if not self.settings.model_routing_enabled:
            tier: Tier = "balanced"
            model = self.settings.active_llm_model
            reason = "routing disabled"
        elif budget is not None and not budget.allows("cost", COST_HEADROOM_USD):
            tier = "fast"
            model = self._model_for("fast")
            reason = "cost budget nearly spent; finishing on the fast model"
        else:
            tier = _TIER_FOR_COMPLEXITY.get(complexity, "balanced")
            model = self._model_for(tier)

        input_tokens = estimate_tokens(question) + _tokens_for_chars(context_chars)
        decision = RoutingDecision(
            model=model,
            tier=tier,
            provider=self.settings.llm_provider,
            reason=reason,
            complexity=complexity,
            estimated_input_tokens=input_tokens,
            projected_usd=project_cost(model, input_tokens, ASSUMED_COMPLETION_TOKENS),
        )
        if tracer is not None:
            tracer.event(
                "route.model",
                kind="router",
                model=decision.model,
                tier=decision.tier,
                complexity=decision.complexity,
                projected_usd=decision.to_dict()["projected_usd"],
            )
        return decision

    def tiers(self) -> dict[str, str]:
        """Tier to model id, for `/health` and the console.

        Every tier resolves to a non-empty id, because `_model_for` collapses an
        unconfigured one onto `active_llm_model`. Two tiers showing the same id
        is the honest picture of a deployment that set neither end of the ladder.
        """
        return {tier: self._model_for(tier) for tier in ("fast", "balanced", "deep")}


def llm_for(
    decision: RoutingDecision,
    settings: Settings | None = None,
    llm_factory: Any = None,
) -> Any:
    """A chat client for the model the router chose.

    `llm_factory` is injectable so a test never constructs a real client, which
    is the same reason every other seam in this codebase takes its dependency as
    an argument. The default is `app.rag.clients.get_llm`, which stays the only
    place a provider is branched on -- this function picks a model id, not a
    provider SDK.
    """
    if llm_factory is None:
        from app.rag.clients import get_llm

        llm_factory = get_llm
    return llm_factory(model=decision.model, settings=settings)
