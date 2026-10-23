"""Model routing: the classifier, the tier ladder and the cost projection.

Offline throughout -- `classify_complexity` makes no model call by design, and
`llm_for` takes an injectable factory so nothing here constructs a provider
client.

The test worth reading twice is
`test_an_ambiguous_one_liner_defaults_to_standard_rather_than_fast`. A
classifier whose uncertain answer is "cheap model" is a classifier that
silently degrades exactly the questions it failed to understand, and the only
visible symptom is a thin answer nobody flags.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from conftest import settings_for

from app.agent.budgets import Budget
from app.core.tracing import Tracer
from app.services.routing import (
    MODEL_CATALOG,
    ModelRouter,
    RoutingDecision,
    classify_complexity,
    estimate_tokens,
    llm_for,
    model_spec,
    project_cost,
)

FAST = "openai/gpt-oss-20b"
BALANCED = "groq/compound"
DEEP = "openai/gpt-oss-120b"


def router_for(**overrides: Any) -> ModelRouter:
    """A router over a three-rung ladder, so a tier mapping is observable."""
    base: dict[str, Any] = dict(FAST_MODEL=FAST, GROQ_MODEL=BALANCED, DEEP_MODEL=DEEP)
    base.update(overrides)
    return ModelRouter(settings=settings_for(**base))


class RecordingHandler(logging.Handler):
    """Captures this module's own log records.

    caplog attaches to the root logger, and `configure_logging()` sets
    `propagate = False` on the package logger -- so an entry point having run
    anywhere in the process would make a caplog assertion pass or fail by test
    order. Attaching here cannot.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# --- The classifier ----------------------------------------------------------


@pytest.mark.parametrize(
    "question", ["hi", "Hello!", "thanks", "Thank you.", "who are you?"]
)
def test_a_greeting_classifies_simple(question):
    complexity, reason = classify_complexity(question)
    assert complexity == "simple"
    assert reason


def test_an_ordinary_policy_question_classifies_standard():
    complexity, reason = classify_complexity(
        "What is the notice period for a resignation?"
    )
    assert complexity == "standard"
    assert reason


def test_a_two_area_arithmetic_question_classifies_complex():
    """Two policy areas in one question is the strongest signal there is.

    The KB keeps the notice period and the PTO payout on termination in
    different documents, so this needs both retrieved and reconciled rather than
    one of them answered confidently.
    """
    complexity, reason = classify_complexity(
        "If I resign, how much of my accrued PTO is paid out and "
        "what notice period applies?"
    )
    assert complexity == "complex"
    assert "policy area" in reason


def test_an_arithmetic_question_inside_one_area_is_still_complex():
    complexity, reason = classify_complexity(
        "How much leave do I carry over if I started in March?"
    )
    assert complexity == "complex"
    assert reason


def test_an_ambiguous_one_liner_defaults_to_standard_rather_than_fast():
    """The conservative default, pinned.

    Nothing in this question matches a signal, so the classifier has not
    understood it. A classifier that guessed "simple" here would be cheapest
    precisely on the questions it could not read, and the symptom would be a
    thin answer rather than an error.
    """
    complexity, reason = classify_complexity("Can you look at the thing from Tuesday?")
    assert complexity == "standard"
    assert "default" in reason


def test_an_empty_question_also_defaults_to_standard():
    assert classify_complexity("   ")[0] == "standard"


def test_a_long_multi_clause_question_classifies_complex():
    question = " ".join(["please explain the policy, and also the exception"] * 6)
    complexity, _ = classify_complexity(question)
    assert complexity == "complex"


def test_a_second_policy_area_from_the_recent_history_counts():
    """A follow-up's real subject is split between what was typed and what it
    refers to, so the last exchange is read for areas too."""
    history = [{"role": "user", "content": "What notice period do I serve?"}]
    complexity, reason = classify_complexity("Does that affect my pension?", history)
    assert complexity == "complex"
    assert "policy area" in reason


# --- Tokens and cost ---------------------------------------------------------


def test_estimate_tokens_is_monotonic_and_roughly_chars_over_four():
    assert estimate_tokens("") == 0
    lengths = [estimate_tokens("x" * n) for n in range(0, 400, 13)]
    assert lengths == sorted(lengths)
    assert estimate_tokens("x" * 400) == 100


def test_project_cost_is_linear_in_tokens():
    one = project_cost("gpt-4o-mini", 1_000, 1_000)
    two = project_cost("gpt-4o-mini", 2_000, 2_000)
    assert one == pytest.approx(0.00075)
    assert two == pytest.approx(2 * one)
    assert project_cost("gpt-4o-mini", 0, 0) == 0.0


def test_a_dearer_model_projects_dearer():
    cheap = project_cost("gpt-4o-mini", 10_000, 400)
    dear = project_cost("gpt-4o", 10_000, 400)
    assert dear > cheap * 10


def test_an_unknown_model_is_reported_rather_than_silently_priced():
    """Zero is the right number; silence is not.

    Inventing a price for a model nobody listed would put a made-up figure in a
    budget ceiling. But a dashboard reading 0.00 because an id was renamed looks
    identical to one reading 0.00 because the deployment is on Groq's free tier,
    and those want very different responses -- so the miss is logged.
    """
    from app.services import routing

    handler = RecordingHandler()
    routing.log.addHandler(handler)
    try:
        assert project_cost("acme/no-such-model", 1_000, 1_000) == 0.0
    finally:
        routing.log.removeHandler(handler)

    assert any("no-such-model" in message for message in handler.messages)


def test_every_model_this_repo_names_is_in_the_catalog():
    for name in (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound",
        "gpt-4o-mini",
        "gpt-4o",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ):
        spec = model_spec(name)
        assert spec is not None
        assert spec.note, f"{name} has no note explaining it"
        assert spec.context_window > 0


def test_an_unlisted_model_has_no_spec():
    assert model_spec("acme/no-such-model") is None
    assert len(MODEL_CATALOG) >= 7


# --- The router --------------------------------------------------------------


def test_choose_maps_each_complexity_to_its_configured_tier_model():
    router = router_for()

    assert router.choose("hi").model == FAST
    assert router.choose("What is the notice period?").model == BALANCED
    assert (
        router.choose(
            "If I resign, how much accrued PTO is paid out and what notice "
            "period applies?"
        ).model
        == DEEP
    )


def test_an_empty_tier_id_falls_back_to_the_active_model():
    """Empty is documented in Settings as "no separate model for this tier"."""
    router = router_for(FAST_MODEL="", DEEP_MODEL="")
    simple = router.choose("hi")
    assert simple.tier == "fast"
    assert simple.model == BALANCED
    assert set(router.tiers().values()) == {BALANCED}


def test_routing_disabled_pins_one_model_and_says_why():
    router = router_for(MODEL_ROUTING_ENABLED=False)
    decision = router.choose("hi")

    assert decision.model == BALANCED
    assert decision.tier == "balanced"
    assert decision.reason == "routing disabled"
    # Still classified, because the unrouted baseline has to be comparable.
    assert decision.complexity == "simple"


def test_a_nearly_spent_cost_budget_forces_the_fast_tier():
    router = router_for()
    budget = Budget(max_cost_usd=0.01, cost_usd=0.0095).start()
    hard = (
        "If I resign, how much accrued PTO is paid out and what notice period "
        "applies?"
    )

    assert router.choose(hard).tier == "deep"  # control: no budget, no override

    decision = router.choose(hard, budget=budget)
    assert decision.tier == "fast"
    assert decision.model == FAST
    assert "budget" in decision.reason
    # A run out of money finishes on the cheap model rather than not at all.
    assert decision.complexity == "complex"


def test_a_budget_with_headroom_does_not_override_the_tier():
    router = router_for()
    budget = Budget(max_cost_usd=1.0).start()
    assert router.choose("hi", budget=budget).tier == "fast"
    assert router.choose("What is the notice period?", budget=budget).model == BALANCED


def test_a_decision_carries_its_token_estimate_and_projection():
    router = router_for()
    decision = router.choose("What is the notice period?", context_chars=4_000)

    assert decision.estimated_input_tokens >= 1_000
    assert decision.projected_usd > 0.0
    assert decision.to_dict()["model"] == BALANCED
    assert decision.to_dict()["projected_usd"] == pytest.approx(
        decision.projected_usd, abs=1e-6
    )


def test_the_routing_decision_is_recorded_on_the_tracer():
    router = router_for()
    tracer = Tracer()
    router.choose("hi", tracer=tracer)

    span = tracer.to_list()[0]
    assert span["name"] == "route.model"
    assert span["kind"] == "router"
    assert span["attributes"]["model"] == FAST
    assert span["attributes"]["complexity"] == "simple"


def test_tiers_never_returns_an_empty_model_id():
    for router in (router_for(), router_for(FAST_MODEL="", DEEP_MODEL="")):
        tiers = router.tiers()
        assert set(tiers) == {"fast", "balanced", "deep"}
        assert all(model for model in tiers.values())


def test_llm_for_builds_the_chosen_model_through_the_injected_factory():
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "a client"

    decision = RoutingDecision(
        model=DEEP, tier="deep", provider="groq", reason="test", complexity="complex"
    )
    settings = settings_for()

    assert llm_for(decision, settings=settings, llm_factory=factory) == "a client"
    assert calls == [{"model": DEEP, "settings": settings}]


# --- The labelled complexity set ---------------------------------------------
# One table rather than thirteen tests, because the value is the *ratio*: a
# classifier is not right or wrong on a question, it is right on a proportion of
# them, and a change that fixes one case by breaking two is a change the
# individual tests would report as progress.
#
# Measured when the strong/weak split was introduced. Before it, this table
# scored 10/13 and -- much worse than the count suggests -- it was wrong in both
# directions at once: "How many days of paid time off do full-time employees
# get?" came out complex while "I'm a contractor going on parental leave
# mid-notice" came out standard. Routing two plain lookups to the deep model is
# money; routing the hard one to the middle is the answer.

LABELLED_COMPLEXITY: tuple[tuple[str, str], ...] = (
    ("simple", "hello there"),
    ("simple", "thanks"),
    ("simple", "who are you"),
    ("standard", "How many days of paid time off do full-time employees get?"),
    ("standard", "What is the notice period if I resign?"),
    ("standard", "When are approved expenses paid out?"),
    ("standard", "Can I work fully remotely from another city?"),
    ("standard", "What are the core hours?"),
    ("standard", "Who approves a travel expense?"),
    ("complex", "I'm a contractor going on parental leave mid-notice, what am I owed?"),
    ("complex", "How is my notice period paid, and does my insurance continue?"),
    ("complex", "How much leave do I carry over if I started in March?"),
    ("complex", "What is the difference between sick leave and casual leave?"),
    (
        "complex",
        "If I resign during probation, how much notice do I owe and is my bonus "
        "prorated?",
    ),
)


def test_the_labelled_complexity_set_is_classified_correctly():
    wrong = [
        (question, expected, classify_complexity(question)[0])
        for expected, question in LABELLED_COMPLEXITY
        if classify_complexity(question)[0] != expected
    ]
    assert not wrong, f"{len(wrong)} of {len(LABELLED_COMPLEXITY)} misclassified: {wrong}"


def test_an_ordinary_lookup_with_an_interrogative_is_not_complex():
    """The regression this table exists for.

    "How many" begins the single most ordinary HR question there is. Treating
    it as arithmetic sent a one-retrieval lookup to the deep model, and on
    `AGENT_MODE=auto` to the supervisor as well -- four model calls instead of
    one, on the question asked most often.
    """
    complexity, _ = classify_complexity("How many days of annual leave do I get?")
    assert complexity == "standard"


def test_a_weak_signal_promotes_in_company_rather_than_alone():
    alone, _ = classify_complexity("How long is the standard lunch break?")
    together, reason = classify_complexity(
        "How long do I have to claim, and how much can I claim in total?"
    )
    assert alone != "complex"
    assert together == "complex"
    assert reason


def test_a_population_the_documents_do_not_describe_is_complex():
    """The corpus says "employees"; whether that covers a contractor is the
    judgement the grader is recorded as finding borderline."""
    complexity, reason = classify_complexity(
        "Do contractors accrue paid time off?"
    )
    assert complexity == "complex"
    assert "contractor" in reason


def test_naming_the_population_the_documents_do_describe_is_not_complex():
    complexity, _ = classify_complexity("Do full-time employees accrue paid time off?")
    assert complexity == "standard"
