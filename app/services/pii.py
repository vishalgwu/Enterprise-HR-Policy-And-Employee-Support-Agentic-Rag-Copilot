"""Finding and removing personal data from text, before it travels.

This product invites an employee to describe their own situation, so a question
carrying a salary figure, a phone number, a national id or a medical detail is
the *normal* case rather than the exception. "My aadhaar is 4321 8765 2109, can
I claim relocation?" is a perfectly ordinary thing to type into a chat box
labelled "ask HR".

Three places that text would otherwise end up, none of which anyone chose:

- **a provider's servers**, because every question is sent to a model;
- **the vector index**, because a question is embedded to retrieve against it,
  and a cache keyed on the question keeps the text as well;
- **the audit log**, which is deliberately long-lived and admin-readable --
  the one store in this system whose whole purpose is to still be there in a
  year.

So redaction happens once, at the edge, before any of the three. `InputGuard`
in `app/services/guardrails.py` is the caller; this module only finds and
replaces.

**What regex PII detection is, and what it is not.** It catches *structured
identifiers* reliably: a thing with a shape -- an email address, a card number,
an aadhaar number, an API key -- is exactly what a pattern is good at. It
catches free text about a medical condition, a divorce, a disciplinary matter or
someone else's name **not at all**. There is no pattern for "I need leave
because my father has dementia", and pretending otherwise would be worse than
having nothing, because it invites the assumption that a redacted question is a
safe question.

That gap is why the other two controls exist rather than being redundant with
this one: the **output guard** catches a model echoing sensitive detail back out
of retrieved context, and the **escalate** path hands a question to a human
instead of answering it when the situation is one a policy lookup should not be
deciding. This module is the cheap, certain layer underneath them, not the
whole of it.

**False positives are the expensive failure here, not misses.** The redacted
question is what the model is actually asked, so a pattern that eats "20 days"
or "Form 16" out of an HR question corrupts the question and produces a worse
answer than leaving the text alone would have. That is the reason the phone
pattern insists on separators or a country code, the card pattern insists on
Luhn, and `passport` and `date_of_birth` are keyed on a nearby label rather than
on a bare token shape: in an HR corpus, "HR2024001" and "2024-04-01" are a
policy reference and an effective date, not a passport and a birthday.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

# No logger here on purpose. This module is called on every question and
# finding personal data is the expected case, not an event -- a log line per
# match would be a log of where the PII was, which is most of the way to a log
# of the PII itself. The guard that calls this is what records a verdict, and
# it records counts and kinds rather than values.

__all__ = [
    "PII_PATTERNS",
    "PiiKind",
    "PiiMatch",
    "find_pii",
    "has_pii",
    "luhn",
    "redact",
    "summarise",
]

PiiKind = Literal[
    "email",
    "phone",
    "ssn",
    "aadhaar",
    "pan",
    "credit_card",
    "iban",
    "ip",
    "passport",
    "date_of_birth",
    "employee_id",
    "bank_account",
    "api_key",
]


@dataclass(frozen=True, slots=True)
class PiiMatch:
    """One span of text identified as personal data."""

    kind: PiiKind
    start: int
    end: int
    text: str

    @property
    def placeholder(self) -> str:
        """What replaces the span. Names the kind, so the question still reads.

        "What is the payout on [REDACTED_BANK_ACCOUNT]?" tells a model, and an
        investigator reading the audit row, what was taken out. A bare
        "[REDACTED]" everywhere loses the shape of the question along with the
        value.
        """
        return f"[REDACTED_{self.kind.upper()}]"


# --- Patterns ---------------------------------------------------------------
# Insertion order is load-bearing: it is the last tie-break when two patterns
# match the *same* span, so the more specific kind goes first. A 16-digit card
# number and a generic long account number are the same sixteen characters, and
# "credit_card" is the more useful label for the thing that just got redacted.

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Separators or a country code are mandatory. An HR corpus is full of bare
# numbers -- "20 days", "Form 16", "section 4.2", "10:00 to 16:00" -- and a
# pattern loose enough to catch "12345" as a phone number redacts all of them.
# The one bare shape allowed is a ten-digit Indian mobile, which has a
# distinctive leading digit and no other plausible reading in this corpus.
_PHONE = re.compile(
    r"""
    (?<![\w.+-])
    (?:
        \+\d{1,3}[\s.-]?\d{2,5}[\s.-]?\d{3,5}(?:[\s.-]?\d{2,5})?   # +91 98765 43210
      | \(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}                          # (555) 123-4567
      | \d{3}[\s.-]\d{3}[\s.-]\d{4}                                # 555-123-4567
      | [6-9]\d{9}                                                 # 9876543210
    )
    # A word character or a hyphen after the number means this is part of a
    # longer token, not a phone number. A full stop does *not*, unless a digit
    # follows it: "call me on +44 7700 900123." is how a person actually writes
    # one, and the plain `(?![\w.-])` this replaced refused every phone number
    # at the end of a sentence -- which is most of them. `(?!\.\d)` keeps the
    # original intent, which was to refuse a phone-shaped run inside a longer
    # dotted number.
    (?![\w-])(?!\.\d)
    """,
    re.VERBOSE,
)

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Aadhaar is twelve digits, conventionally in groups of four, and never starts
# with 0 or 1.
_AADHAAR = re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")

# PAN is five letters, four digits, one letter, and always uppercase. Matched
# case-sensitively on purpose: case-insensitively it starts eating ordinary
# words that happen to sit next to a number.
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

_CREDIT_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30}\b")

_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Label-anchored, unlike the rest. A bare letter-plus-digits token is
# indistinguishable from an internal reference -- "HR2024001" is a policy
# number -- so the label is what makes it a passport number, and the label is
# redacted with it rather than left behind pointing at a hole.
_PASSPORT = re.compile(
    r"\bpassports?\s*(?:no\.?|number|#)?\s*:?\s*[A-Z]{1,2}\d{6,8}\b",
    re.IGNORECASE,
)

# Also label-anchored, and for a sharper version of the same reason: every
# policy document in the knowledge base contains dates, and a bare date is an
# effective date far more often than it is a birthday.
_DATE_OF_BIRTH = re.compile(
    r"""
    \b(?:d\.?o\.?b\.?|date\s+of\s+birth|born(?:\s+on)?)
    (?:\s+(?:is|was))?\s*:?\s*
    (?:
        \d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}
      | \d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}
      | [A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_EMPLOYEE_ID = re.compile(
    r"""
    \b(?:
        EMP[-_ ]?\d{3,8}
      | employee\s+(?:id|no\.?|number|code)\s*:?\s*\#?\s*\d{3,8}
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_BANK_ACCOUNT = re.compile(r"(?<![\d-])\d{9,18}(?![\d-])")

# The shapes this repo actually handles, and the reason is specific. An
# employee pasting a stack trace, or a visitor pasting their own provider key
# into the chat box instead of the key panel, must not have it written into the
# audit log -- which is the one store designed to still hold it in a year, and
# which `/api/audit` is admin-gated to protect. A leaked key in there is a
# credential this deployment never asked for and cannot rotate.
_API_KEY = re.compile(
    r"""
    (?:
        \b(?:sk|pk|rk)-(?:proj-)?[A-Za-z0-9_-]{16,}     # OpenAI
      | \b(?:gsk|pcsk)_[A-Za-z0-9_-]{16,}               # Groq, Pinecone
      | \btvly-[A-Za-z0-9_-]{16,}                       # Tavily
      | \b(?:lsv2_(?:pt|sk)_|xoxb-|ghp_|github_pat_|AKIA)[A-Za-z0-9_-]{10,}
    )
    """,
    re.VERBOSE,
)

PII_PATTERNS: dict[PiiKind, re.Pattern[str]] = {
    "api_key": _API_KEY,
    "email": _EMAIL,
    "iban": _IBAN,
    "pan": _PAN,
    "ssn": _SSN,
    "credit_card": _CREDIT_CARD,
    "aadhaar": _AADHAAR,
    "passport": _PASSPORT,
    "date_of_birth": _DATE_OF_BIRTH,
    "employee_id": _EMPLOYEE_ID,
    "ip": _IP,
    "phone": _PHONE,
    "bank_account": _BANK_ACCOUNT,
}

_KIND_ORDER = {kind: index for index, kind in enumerate(PII_PATTERNS)}


# --- Validation -------------------------------------------------------------
def luhn(digits: str) -> bool:
    """The card checksum, ignoring spaces and hyphens.

    Without it, a 13-to-16-digit pattern redacts order numbers, employee ids,
    invoice references and plain amounts. A redactor that eats ordinary numbers
    out of HR questions is one people switch off, and a switched-off redactor
    protects nothing -- so the cost of a false positive here is the whole
    control, not one bad answer.
    """
    stripped = [int(char) for char in digits if char.isdigit()]
    if len(stripped) < 2:
        return False
    total = 0
    for position, digit in enumerate(reversed(stripped)):
        if position % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# --- Finding ----------------------------------------------------------------
def find_pii(text: str, kinds: Collection[PiiKind] | None = None) -> list[PiiMatch]:
    """Every span of personal data in `text`, sorted by position.

    `kinds=None` means all of them. Overlaps are resolved before returning:
    see `_resolve`.
    """
    if not text:
        return []
    wanted = PII_PATTERNS.keys() if kinds is None else [
        kind for kind in PII_PATTERNS if kind in kinds
    ]

    found: list[PiiMatch] = []
    for kind in wanted:
        for match in PII_PATTERNS[kind].finditer(text):
            if kind == "credit_card" and not luhn(match.group(0)):
                continue
            found.append(
                PiiMatch(
                    kind=kind,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                )
            )
    return _resolve(found)


def _resolve(matches: list[PiiMatch]) -> list[PiiMatch]:
    """Keep the longest of any overlapping group, then the earliest.

    Patterns genuinely overlap, and picking wrong leaves digits behind. A card
    number written "4111 1111 1111 1111" also contains an aadhaar-shaped
    twelve-digit prefix; redacting the prefix leaves the last group of four
    sitting in the text, which is both a disclosure and a lie about having
    redacted it. Longest-first makes the span that covers the whole identifier
    win.

    Insertion order in `PII_PATTERNS` is the final tie-break, for the case the
    other two cannot settle: an unseparated sixteen-digit card and a generic
    long account number are the *same* span, so one of them has to be declared
    the more specific reading, and it is written down rather than left to
    whichever pattern happened to run first.
    """
    ranked = sorted(
        matches, key=lambda m: (-(m.end - m.start), m.start, _KIND_ORDER[m.kind])
    )
    kept: list[PiiMatch] = []
    for candidate in ranked:
        if any(
            candidate.start < other.end and other.start < candidate.end
            for other in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda m: m.start)


def has_pii(text: str) -> bool:
    """Whether there is anything to redact. Cheaper to read than `find_pii`."""
    return bool(find_pii(text))


# --- Redacting --------------------------------------------------------------
def redact(
    text: str, kinds: Collection[PiiKind] | None = None
) -> tuple[str, list[PiiKind]]:
    """Replace every match with its placeholder.

    Returns the rewritten text and the kinds removed, in the order they first
    appeared -- the second half is what goes in the trace and the audit row, so
    "this question had a phone number and a card in it" is answerable without
    keeping the question.

    Rebuilt right-to-left, because replacing a span changes every offset after
    it and the matches were found against the original string.
    """
    matches = find_pii(text, kinds)
    if not matches:
        return text, []

    rebuilt = text
    for match in reversed(matches):
        rebuilt = rebuilt[: match.start] + match.placeholder + rebuilt[match.end :]

    kinds_found: list[PiiKind] = []
    for match in matches:
        if match.kind not in kinds_found:
            kinds_found.append(match.kind)
    return rebuilt, kinds_found


def summarise(matches: Sequence[PiiMatch]) -> dict[str, int]:
    """How many of each kind, for a trace attribute and a metrics counter.

    Counts rather than values, so this is safe to put somewhere the values are
    not -- which is the whole point of having it.
    """
    counts: dict[str, int] = {}
    for match in matches:
        counts[match.kind] = counts.get(match.kind, 0) + 1
    return counts
