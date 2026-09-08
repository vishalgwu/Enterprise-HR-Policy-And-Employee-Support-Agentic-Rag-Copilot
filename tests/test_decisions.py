"""Routing and grading, and the recovery both depend on.

The failure these cover is real and observed: Groq returns 400 `tool_use_failed`
when the model answers in plain text instead of calling the bound tool, and the
router is on the path of every question.
"""

from __future__ import annotations

import pytest

from app.agent.decisions import (
    GRADE_VALUES,
    ROUTE_VALUES,
    coerce,
    failed_generation,
    grade_evidence,
    route_question,
)


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
