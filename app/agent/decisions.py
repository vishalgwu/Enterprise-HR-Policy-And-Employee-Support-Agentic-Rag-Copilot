"""Routing and grading, with the recovery both of them need.

Structured output fails intermittently and must never crash the graph. Groq
returns `400 tool_use_failed` -- "Tool choice is required, but model did not call
a tool" -- when the model answers in plain text instead of invoking the bound
tool. Observed live on the router, whose call is on the path of *every*
question.

The error body carries `error.failed_generation`, which holds what the model
actually said and is usually the correct value (an observed failure carried
exactly `"direct"`). So: try the structured call, recover the raw text, and only
then fall back to a safe default -- the router to `kb` (seek evidence rather
than answer from memory) and the grader to `weak` (distrust rather than pass
ungraded evidence).

Graph nodes must call `route_question` / `grade_evidence` from this module, not
`router.invoke` / `grader.invoke` directly.
"""

from __future__ import annotations

from typing import Any, Tuple

from app.agent.prompts import GRADER_PROMPT, ROUTER_PROMPT
from app.agent.schemas import EvidenceGrade, Grade, Route, RouteDecision
from app.core.config import get_logger
from app.rag.clients import get_llm, is_rate_limit

log = get_logger("decisions")

ROUTE_VALUES: Tuple[str, ...] = ("kb", "direct")
GRADE_VALUES: Tuple[str, ...] = ("good", "weak")


def get_router(llm: Any = None):
    """Runnable mapping {"question": str} to a RouteDecision."""
    return ROUTER_PROMPT | (llm or get_llm()).with_structured_output(RouteDecision)


def get_evidence_grader(llm: Any = None):
    """Runnable mapping {"question": str, "evidence": str} to an EvidenceGrade."""
    return GRADER_PROMPT | (llm or get_llm()).with_structured_output(EvidenceGrade)


def failed_generation(exc: BaseException) -> str | None:
    """Recover the model's raw text from a Groq tool_use_failed error."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            generated = error.get("failed_generation")
            if isinstance(generated, str):
                return generated
    return None


def coerce(text: str | None, allowed: Tuple[str, ...]) -> str | None:
    """Match loose model text against an enum.

    Handles 'direct', '"direct"', and {"route": "direct"}. Falls back to
    containment so a JSON blob or a short sentence still resolves, but only when
    exactly one candidate appears -- with two it would be a guess, and guessing
    is what the safe default is for.
    """
    if not text:
        return None
    lowered = text.strip().strip("\"' \n\t.").lower()
    for value in allowed:
        if lowered == value:
            return value
    hits = [v for v in allowed if v in lowered]
    return hits[0] if len(hits) == 1 else None


def _recover(exc: BaseException, allowed: Tuple[str, ...], tag: str, default: str) -> str:
    recovered = coerce(failed_generation(exc), allowed)
    if recovered:
        log.debug("[%s] recovered %r from a failed structured call", tag, recovered)
        return recovered
    if is_rate_limit(exc):
        # Error level, deliberately: every safe default here is "weak", so an
        # exhausted quota otherwise looks exactly like a knowledge gap. If a
        # whole test run suddenly grades everything weak, check the quota before
        # touching a prompt.
        log.error(
            "[%s] RATE LIMITED; defaulting to %r -- this is a quota failure, "
            "not a judgement (%s)",
            tag,
            default,
            str(exc)[:160],
        )
    else:
        log.warning(
            "[%s] structured output failed (%s); defaulting to %r",
            tag,
            type(exc).__name__,
            default,
        )
    return default


def route_question(question: str, router: Any = None) -> Route:
    """Route one message, surviving a structured-output failure.

    Falls back to 'kb' rather than 'direct': routing an unclassifiable message
    to retrieval costs a lookup, whereas defaulting to 'direct' would make the
    assistant answer from memory with no evidence behind it.
    """
    try:
        return (router or get_router()).invoke({"question": question}).route
    except Exception as exc:  # noqa: BLE001 - no client error may crash the graph
        return _recover(exc, ROUTE_VALUES, "Router", "kb")  # type: ignore[return-value]


def grade_evidence(question: str, evidence: str, grader: Any = None) -> Grade:
    """Grade evidence. Empty evidence is 'weak' without spending a call.

    Falls back to 'weak' on a structured-output failure, so an unreadable grade
    sends the graph to its fallback path instead of letting an ungraded answer
    through.
    """
    if not evidence.strip():
        return "weak"
    try:
        return (
            (grader or get_evidence_grader())
            .invoke({"question": question, "evidence": evidence})
            .grade
        )
    except Exception as exc:  # noqa: BLE001
        return _recover(exc, GRADE_VALUES, "Grader", "weak")  # type: ignore[return-value]
