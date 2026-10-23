"""What may go into the agent, and what may come back out.

Two checks with the same shape and opposite jobs. `InputGuard` runs on the
employee's question before anything is embedded, retrieved or sent to a
provider; `OutputGuard` runs on the finished answer before it reaches a browser
or the audit log. Both return a `GuardVerdict` rather than raising, because a
guard that raises turns a judgement about content into a 500.

**Injection is flagged and neutralised, not refused, and that is the decision
worth arguing.** The obvious design is to refuse anything containing "ignore
previous instructions". Three reasons not to:

- The prompts here are already retrieval-grounded: the generation prompt is
  given numbered context and told to answer from it, and the graders judge
  evidence rather than follow it. An instruction smuggled into a question is
  arriving at a model that has been told what its source of truth is.
- "Ignore what I said before" is a thing a confused employee genuinely types,
  mid-conversation, about their own earlier question. Refusing it costs a real
  person their answer for looking like an attacker.
- A refusal is a free oracle. It tells whoever is probing exactly which
  phrasing tripped the check, which is the one piece of information an attacker
  wants and the one thing a flag-and-continue never gives them.

So an injection attempt is recorded at `severity="high"`, written to the trace
and the audit row where a human can review it, and the instruction-like span is
replaced with `[instruction removed]` so the model never reads it. Refusal is
reserved for the case where that leaves nothing behind -- a message that is
*only* an attack has no question in it to answer, so there is nothing to cost
anyone.

**`"off_topic"` is in the category vocabulary and this module never produces
it, deliberately.** A keyword list that refuses "what is the capital of France?"
makes the product feel broken, and it is not even the better detector: the
similarity gate scores an out-of-domain question below 0.15 and the router
sends a greeting straight to a direct answer, both of which handle the general
case better than any list of words could. Refusal here is for requests that are
unsafe *and* out of scope -- self-harm instructions, violence, fraud,
impersonation, a legal opinion -- which is a short list precisely because it has
to be defensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_logger
from app.core.tracing import Tracer, null_tracer
from app.services import pii

log = get_logger("guardrails")

__all__ = [
    "INJECTION_PATTERNS",
    "REFUSAL_TOPICS",
    "GuardVerdict",
    "InputGuard",
    "OutputGuard",
]

# What replaces a neutralised instruction. Visible rather than silent: the model
# reads it as a hole in the text instead of a sentence it should act on, and an
# admin reading the audit row can see where the question was interfered with.
INSTRUCTION_PLACEHOLDER = "[instruction removed]"

# Stands in for an answer that redaction emptied completely. Never "": a caller
# handed an empty string renders a blank bubble and has no way to know whether
# the model said nothing or the guard took it all.
REDACTED_ANSWER = (
    "I had to remove the whole of that answer, because all of it was personal "
    "data. Ask me about the policy itself and I can quote that instead."
)

# A hard ceiling for callers that do not have one. `ChatRequest` in
# `app/api/routes.py` already caps a question at 3000 characters and answers a
# longer one with a 422, so this is not the HTTP limit and must not duplicate
# that error -- it is the limit for the CLI, a tool call and an ingestion
# worker, none of which go through pydantic validation.
MAX_QUESTION_CHARS = 8000

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CATEGORY_RANK = {"ok": 0, "pii": 1, "too_long": 2, "injection": 3, "unsafe": 4,
                  "secret": 5, "empty": 6}


# --- Patterns ---------------------------------------------------------------
# Instruction-shaped spans. Each one is bounded rather than running to the end
# of the string: neutralising a span removes it, and a greedy pattern would take
# the employee's actual question with it.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:ignore|disregard)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+)?"
        r"(?:previous|prior|preceding|above|earlier|system|initial)\s+"
        r"(?:instructions?|prompts?|rules?|messages?|directions?|context)",
        r"\bforget\s+(?:everything|all)(?:\s+(?:you|above|before|i)\b[\w\s]{0,24})?",
        r"\b(?:reveal|show|print|repeat|output|disclose|tell\s+me)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:full\s+|exact\s+|original\s+|entire\s+)?"
        r"(?:system\s+)?(?:prompt|instructions?|rules?)",
        r"\bwhat\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|you\s+must\s+now)"
        r"[\w\s'-]{0,40}",
        r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you\s+are)[\w\s'-]{0,32}",
        r"\bact\s+as\s+(?:if\s+)?you\s+(?:are|were)[\w\s'-]{0,32}",
        r"\b(?:developer|debug)\s+mode\b",
        r"\b(?:jailbreak|do\s+anything\s+now)\b",
        r"\bnew\s+instructions?\s*:",
        r"\bsystem\s*:\s*you\s+are\b",
        r"\b(?:override|bypass|ignore)\s+(?:your\s+|the\s+|all\s+)?"
        r"(?:instructions?|rules?|guardrails?|safety|restrictions?|filters?)",
        r"\bdo\s+not\s+follow\s+(?:your|the|any)\s+(?:previous\s+)?"
        r"(?:instructions?|rules?|guidelines?)",
        r"\b(?:repeat|print)\s+(?:the\s+)?(?:text|words|everything)\s+above",
        r"<\s*/?\s*(?:\|?im_(?:start|end)\|?|system|assistant)\s*\|?\s*>",
        r"\[\s*(?:system|/?inst)\s*\]",
    )
)

# Requests that are both unsafe and outside what an HR policy lookup should be
# answering. Short on purpose -- every entry has to be one a reader would agree
# with, because the cost of a wrong entry is a real employee told no.
#
# Note what is *not* here: a statement of distress. "I am being harassed and it
# is making me want to hurt myself" is not a request for instructions, and
# refusing it would be the worst possible answer. That is a case for the
# `escalate` path, which puts it in front of a person.
REFUSAL_TOPICS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # self-harm, scoped to a request for method rather than an admission
        r"\bhow\s+(?:to|do\s+i|can\s+i)\s+(?:kill|harm|hurt|injure)\s+myself\b",
        r"\bhow\s+to\s+commit\s+suicide\b",
        r"\b(?:best|easiest|quickest|painless)\s+way\s+to\s+(?:kill\s+myself|die)\b",
        # violence
        r"\bhow\s+to\s+(?:make|build|construct)\s+(?:a\s+|an\s+)?"
        r"(?:bomb|explosive|weapon|gun)\b",
        r"\bhow\s+(?:to|do\s+i)\s+(?:kill|poison|stab|shoot|attack)\s+"
        r"(?:a|an|my|someone|somebody|him|her|them)\b",
        # fraud and other illegality
        r"\b(?:launder|embezzle)\s+(?:the\s+|company\s+)?(?:money|funds|payroll)\b",
        r"\bhow\s+(?:to|do\s+i)\s+(?:hack|break)\s+into\b",
        r"\bforge\s+(?:a\s+|an\s+|my\s+)?"
        r"(?:signature|payslip|document|certificate|letter|timesheet)\b",
        # impersonation
        r"\b(?:impersonate|pose\s+as)\s+(?:an?\s+|the\s+|my\s+)?"
        r"(?:employee|colleague|manager|hr|ceo|director)\b",
        # a legal opinion, which this is not qualified to give and which an
        # employee could reasonably act on
        r"\blegal\s+(?:advice|opinion)\b",
        r"\bas\s+my\s+(?:lawyer|attorney|solicitor)\b",
        r"\brepresent\s+me\s+in\s+court\b",
    )
)

# Numbers an answer should only be asserting off the back of evidence: a money
# amount, a percentage, or a count of days, weeks or months.
_POLICY_NUMBER = re.compile(
    r"(?:[$£€₹]\s?\d|\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s+"
    r"(?:days?|weeks?|months?|years?|hours?)\b)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class GuardVerdict:
    """What a guard decided, and the text the caller should actually use.

    `text` is always the sanitised text, never a refusal message: a caller that
    is refused answers with the canned refusal in `app/agent/prompts.py` rather
    than with anything from here, so one sentence is not written twice.
    """

    allowed: bool
    text: str
    category: str = "ok"
    reasons: list[str] = field(default_factory=list)
    redacted: list[str] = field(default_factory=list)
    severity: str = "none"
    # The text as it arrived. Kept so `modified` can be a property instead of a
    # second return value, and left out of `to_dict` on purpose -- the verdict
    # is serialised onto the HTTP response and into the audit row, which is
    # exactly where the unredacted original must not go.
    original: str = ""

    @property
    def modified(self) -> bool:
        """Whether the guard rewrote the text it was given."""
        return self.text != self.original

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "category": self.category,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "redacted": list(self.redacted),
            "modified": self.modified,
        }


class _Guard:
    """Shared construction. Both guards resolve the same three switches.

    `settings` is optional and never fetched from the process when absent: a
    guard built with nothing is on, redacting, and coupled to no configuration
    at all, which is what lets a test and a CLI construct one in a line.
    """

    _enabled_setting = "input_guard_enabled"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        redact_pii: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.settings = settings
        if enabled is None:
            enabled = (
                getattr(settings, self._enabled_setting) if settings else True
            )
        if redact_pii is None:
            redact_pii = settings.redact_pii if settings else True
        self.enabled = bool(enabled)
        self.redact_pii = bool(redact_pii)

    @staticmethod
    def _worse(current: str, candidate: str) -> str:
        """The more severe of two severities."""
        if _SEVERITY_RANK[candidate] > _SEVERITY_RANK[current]:
            return candidate
        return current

    @staticmethod
    def _label(current: str, candidate: str) -> str:
        """The more alarming of two categories, so one line names the worst."""
        if _CATEGORY_RANK[candidate] > _CATEGORY_RANK[current]:
            return candidate
        return current


class InputGuard(_Guard):
    """The check an employee's question passes before anything acts on it."""

    _enabled_setting = "input_guard_enabled"

    def check(self, question: str, tracer: Tracer | None = None) -> GuardVerdict:
        """Sanitise a question, and say whether it may be answered at all."""
        tracer = tracer or null_tracer()
        if not self.enabled:
            return GuardVerdict(allowed=True, text=question, original=question)

        verdict = GuardVerdict(allowed=True, text=question, original=question)

        if not question or not question.strip():
            verdict.allowed = False
            verdict.category = "empty"
            verdict.severity = "low"
            verdict.reasons.append("The question was empty.")
            return self._trace(verdict, tracer)

        if len(question) > MAX_QUESTION_CHARS:
            verdict.allowed = False
            verdict.category = "too_long"
            verdict.severity = "low"
            verdict.reasons.append(
                f"The question was longer than {MAX_QUESTION_CHARS} characters."
            )
            return self._trace(verdict, tracer)

        text = question
        text, injections = _neutralise(text)
        if injections:
            verdict.severity = self._worse(verdict.severity, "high")
            verdict.category = self._label(verdict.category, "injection")
            verdict.reasons.append(
                f"Removed {injections} instruction-like "
                f"{'span' if injections == 1 else 'spans'} from the question."
            )
            if not _has_substance(text):
                # Nothing left to answer, so refusing costs no real question.
                verdict.allowed = False
                verdict.text = text
                verdict.reasons.append(
                    "The message was an instruction and nothing else."
                )
                return self._trace(verdict, tracer)

        # Always, regardless of `redact_pii`. A pasted key is not personal data
        # the deployment may choose to keep: it is a live credential belonging
        # to someone else, and the audit log is designed to still have it in a
        # year.
        text, keys = pii.redact(text, kinds=["api_key"])
        if keys:
            verdict.severity = self._worse(verdict.severity, "high")
            verdict.category = self._label(verdict.category, "secret")
            verdict.redacted.extend(keys)
            verdict.reasons.append(
                "That looked like an API key, so it was removed before anything "
                "saw it. If you meant to use your own key, put it in the Key "
                "panel rather than the chat box."
            )

        if self.redact_pii:
            text, kinds = pii.redact(
                text, kinds=[k for k in pii.PII_PATTERNS if k != "api_key"]
            )
            if kinds:
                verdict.severity = self._worse(verdict.severity, "low")
                verdict.category = self._label(verdict.category, "pii")
                verdict.redacted.extend(kinds)
                verdict.reasons.append(
                    "Personal details were removed before the question was sent "
                    "anywhere: " + ", ".join(kinds) + "."
                )

        verdict.text = text

        topic = _refused_topic(question)
        if topic is not None:
            verdict.allowed = False
            verdict.category = "unsafe"
            verdict.severity = "high"
            verdict.reasons.append("Out of scope and not safe to answer.")

        return self._trace(verdict, tracer)

    @staticmethod
    def _trace(verdict: GuardVerdict, tracer: Tracer) -> GuardVerdict:
        tracer.event(
            "guard.input",
            kind="guard",
            category=verdict.category,
            severity=verdict.severity,
            allowed=verdict.allowed,
            redacted=list(verdict.redacted),
        )
        return verdict


class OutputGuard(_Guard):
    """The check an answer passes on its way to a browser and the audit log."""

    _enabled_setting = "output_guard_enabled"

    def check(
        self,
        answer: str,
        tracer: Tracer | None = None,
        grounded: bool = True,
    ) -> GuardVerdict:
        """Sanitise an answer, and say whether it may be shown.

        `grounded` is the run's own verdict on whether the answer rested on
        retrieved evidence. It is a parameter rather than something inferred
        here because only the caller knows it.
        """
        tracer = tracer or null_tracer()
        if not self.enabled:
            return GuardVerdict(allowed=True, text=answer, original=answer)

        verdict = GuardVerdict(allowed=True, text=answer, original=answer)
        text = answer or ""

        # A key in an answer is not redacted and shown: it means something in
        # the context or the prompt chain is leaking a credential, and an answer
        # with a hole where the key was still tells the reader there is one to
        # find. Blocked outright, and the run is logged for someone to look at.
        if pii.find_pii(text, kinds=["api_key"]):
            log.error("Blocked an answer that contained something key-shaped")
            verdict.allowed = False
            verdict.category = "secret"
            verdict.severity = "high"
            verdict.text = pii.redact(text, kinds=["api_key"])[0]
            verdict.redacted.append("api_key")
            verdict.reasons.append("The answer contained a credential.")
            return self._trace(verdict, tracer)

        if self.redact_pii:
            # The realistic case is not invention: it is a model repeating an
            # employee id or a phone number out of a retrieved document.
            text, kinds = pii.redact(text)
            if kinds:
                verdict.severity = self._worse(verdict.severity, "low")
                verdict.category = self._label(verdict.category, "pii")
                verdict.redacted.extend(kinds)
                verdict.reasons.append(
                    "Personal details were removed from the answer: "
                    + ", ".join(kinds)
                    + "."
                )

        if not grounded and _POLICY_NUMBER.search(text):
            # Flagged, not blocked. An answer that states "20 days" with no
            # evidence behind it may still be right, and in an HR product the
            # judgement about whether to trust it belongs to the human reading
            # the trace rather than to a regex. What the flag buys is a
            # searchable audit log: this exact failure -- a confident number
            # with nothing under it -- is the one worth being able to query for
            # after the fact, and it is invisible unless it is recorded.
            verdict.severity = self._worse(verdict.severity, "medium")
            verdict.category = self._label(verdict.category, "unsafe")
            verdict.reasons.append(
                "The answer states a figure but the run had no evidence to "
                "ground it."
            )

        # Only when redaction is what emptied it. A genuinely short answer --
        # "Yes, 20 days." -- must not be replaced by this.
        if verdict.redacted and not _has_substance(text, minimum=1):
            verdict.text = REDACTED_ANSWER
            verdict.severity = self._worse(verdict.severity, "medium")
            verdict.reasons.append("Redaction left nothing of the answer.")
        else:
            verdict.text = text

        return self._trace(verdict, tracer)

    @staticmethod
    def _trace(verdict: GuardVerdict, tracer: Tracer) -> GuardVerdict:
        tracer.event(
            "guard.output",
            kind="guard",
            category=verdict.category,
            severity=verdict.severity,
            allowed=verdict.allowed,
            redacted=list(verdict.redacted),
        )
        return verdict


# --- Helpers ----------------------------------------------------------------
def _neutralise(text: str) -> tuple[str, int]:
    """Replace every instruction-shaped span, and say how many there were.

    Applied pattern by pattern over the already-rewritten text, so two
    overlapping phrasings of the same attempt collapse into one placeholder
    rather than nesting.
    """
    removed = 0
    for pattern in INJECTION_PATTERNS:
        text, count = pattern.subn(INSTRUCTION_PLACEHOLDER, text)
        removed += count
    return text, removed


def _has_substance(text: str, minimum: int = 3) -> bool:
    """Whether anything but placeholders and punctuation is left.

    Three words is the threshold for a question, not one: "and", "then" and
    "please" survive the removal of the instruction around them, and a message
    whose entire remainder is a conjunction had no question in it. An answer is
    judged at `minimum=1` instead -- "Contact [REDACTED_EMAIL]" is a poor
    answer but it is still an answer, and replacing it wholesale would hide
    more than it saved.
    """
    stripped = text.replace(INSTRUCTION_PLACEHOLDER, " ")
    stripped = re.sub(r"\[REDACTED_[A-Z_]+\]", " ", stripped)
    return len(re.findall(r"[A-Za-z0-9]+", stripped)) >= minimum


def _refused_topic(text: str) -> str | None:
    """The first refusal pattern this text matches, or None."""
    for pattern in REFUSAL_TOPICS:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return None
