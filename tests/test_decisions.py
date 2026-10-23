"""Routing and grading, and the recovery both depend on.

The failure these cover is real and observed: Groq returns 400 `tool_use_failed`
when the model answers in plain text instead of calling the bound tool, and the
router is on the path of every question.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM, settings_for

from app.agent.decisions import (
    GRADE_VALUES,
    ROUTE_VALUES,
    coerce,
    failed_generation,
    get_evidence_grader,
    grade_evidence,
    route_question,
)
from app.agent.prompts import (
    GRADER_EXAMPLES,
    GRADER_PROMPT,
    GRADER_PROMPT_FEWSHOT,
)
from app.agent.schemas import RouteDecision


class _ToolUseFailed(Exception):
    """Mimics the Groq error, which carries the model's real text in `body`."""

    def __init__(self, generated: str) -> None:
        super().__init__("Tool choice is required, but model did not call a tool")
        self.body = {"error": {"failed_generation": generated}}


class _Boom:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("provider down")


class _Recoverable:
    def __init__(self, generated: str) -> None:
        self.generated = generated

    def invoke(self, *args, **kwargs):
        raise _ToolUseFailed(self.generated)


class _RateLimited(Exception):
    status_code = 429


class _Throttled:
    def invoke(self, *args, **kwargs):
        raise _RateLimited("Rate limit reached for tokens per day (TPD)")


# --- coerce ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("direct", "direct"),
        ('"direct"', "direct"),
        ("  direct.\n", "direct"),
        ("DIRECT", "direct"),
        ('{"route": "direct"}', "direct"),
        ("kb", "kb"),
        ("", None),
        (None, None),
        ("something else entirely", None),
    ],
)
def test_coerce_resolves_loose_model_text(text, expected):
    assert coerce(text, ROUTE_VALUES) == expected


def test_coerce_refuses_to_guess_between_two_candidates():
    assert coerce("either good or weak", GRADE_VALUES) is None


def test_failed_generation_reads_the_groq_error_body():
    assert failed_generation(_ToolUseFailed("direct")) == "direct"


def test_failed_generation_of_an_ordinary_error_is_none():
    assert failed_generation(RuntimeError("nope")) is None


# --- Router ------------------------------------------------------------------


def test_router_recovers_the_value_from_a_failed_structured_call():
    """The observed failure carried exactly "direct"; throwing it away is a bug."""
    assert route_question("hello", _Recoverable("direct")) == "direct"


def test_router_falls_back_to_kb_not_direct():
    """Defaulting to direct would answer from memory with no evidence."""
    assert route_question("How many PTO days?", _Boom()) == "kb"


def test_router_survives_a_rate_limit():
    assert route_question("How many PTO days?", _Throttled()) == "kb"


# --- Grader ------------------------------------------------------------------


def test_empty_evidence_is_weak_without_calling_the_model():
    assert grade_evidence("q", "   ", _Boom()) == "weak"


def test_grader_recovers_the_value_from_a_failed_structured_call():
    assert grade_evidence("q", "evidence", _Recoverable("good")) == "good"


def test_grader_falls_back_to_weak():
    """Distrust rather than pass ungraded evidence."""
    assert grade_evidence("q", "evidence", _Boom()) == "weak"


def test_grader_survives_a_rate_limit():
    assert grade_evidence("q", "evidence", _Throttled()) == "weak"


def test_few_shot_grading_is_off_by_default_and_the_switch_actually_swaps_the_prompt():
    """The switch has to reach the grader, not only /health.

    `FEWSHOT_GRADING` existed as a setting, `GRADER_PROMPT_FEWSHOT` existed as a
    template, and nothing connected them -- so setting the flag changed what
    `/api/config` reported and nothing about how evidence was graded. A switch
    that silently does nothing is worse than no switch: it makes an A/B
    measurement that was never actually run look like it was.
    """
    plain = get_evidence_grader(FakeLLM(), settings=settings_for())
    fewshot = get_evidence_grader(FakeLLM(), settings=settings_for(FEWSHOT_GRADING=True))

    assert plain.first is GRADER_PROMPT
    assert fewshot.first is GRADER_PROMPT_FEWSHOT


def test_the_few_shot_examples_are_the_four_corners_of_the_decision():
    """Four, not ten: the grader runs once or twice per answer, so every
    example is tokens on every graded question."""
    grades = [grade for _question, _evidence, grade in GRADER_EXAMPLES]
    assert grades == ["good", "weak", "weak", "good"]
    assert len(GRADER_EXAMPLES) == 4

    rendered = GRADER_PROMPT_FEWSHOT.messages[0].prompt.template
    for question, _evidence, _grade in GRADER_EXAMPLES:
        assert question in rendered


# --- A degraded decision has to be visible in the trace ----------------------


class _RateLimited(Exception):
    """A provider refusing for quota, in the shape `is_rate_limit` detects."""

    status_code = 429

    def __str__(self) -> str:
        return "Error code: 429 - tokens per day (TPD): Limit 200000, Used 199771"


class _Exploding:
    def invoke(self, *_args, **_kwargs):
        raise _RateLimited()


class _FakeRouter:
    """A router that simply works, for the negative case below."""

    def __init__(self, route: str) -> None:
        self.route = route

    def invoke(self, *_args, **_kwargs):
        return RouteDecision(route=self.route)


def test_a_rate_limited_router_records_that_it_was_a_quota_failure():
    """The safe default is `kb`, which is what a working router often chooses.

    So "the router said kb" and "the router was rate limited and defaulted to
    kb" are the same line in a trace that does not say which -- and the second
    one means every grade after it is degraded too. It was logged at ERROR on
    the server and recorded nowhere a reader of the answer could see.
    """
    steps: list[str] = []
    route = route_question("How many PTO days?", _Exploding(), steps)

    assert route == "kb"
    assert steps, "a degraded decision must reach the trace"
    assert "rate limited" in steps[0]
    assert "not a judgement" in steps[0]


def test_a_rate_limited_grader_records_it_too():
    steps: list[str] = []
    grade = grade_evidence("q", "some evidence", _Exploding(), steps)

    assert grade == "weak"
    assert any("rate limited" in step for step in steps)


def test_an_ordinary_structured_output_failure_is_recorded_differently():
    """A malformed reply and an exhausted quota both default to the safe value
    and want different things done about them."""

    class _Malformed:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("the model said something unparseable")

    steps: list[str] = []
    assert grade_evidence("q", "evidence", _Malformed(), steps) == "weak"
    assert "structured output failed" in steps[0]
    assert "rate limited" not in steps[0]


def test_a_successful_decision_adds_nothing_to_the_trace():
    """Only a *degraded* call writes here; the node logs the decision itself."""
    steps: list[str] = []
    route_question("hello", _FakeRouter("direct"), steps)
    assert steps == []
