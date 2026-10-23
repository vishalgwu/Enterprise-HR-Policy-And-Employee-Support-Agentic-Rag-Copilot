"""What the redactor finds, and -- just as important -- what it leaves alone."""

from __future__ import annotations

import pytest

from app.services.pii import (
    PII_PATTERNS,
    find_pii,
    has_pii,
    luhn,
    redact,
    summarise,
)

# One realistic example per kind. A missing entry here means a pattern shipped
# with nothing exercising it.
EXAMPLES = {
    "email": "write to priya.sharma@example.com about it",
    "phone": "my mobile is +91 98765 43210",
    "ssn": "the form wants 123-45-6789",
    "aadhaar": "aadhaar 4321 8765 2109",
    "pan": "PAN ABCDE1234F",
    "credit_card": "the card on file is 4111 1111 1111 1111",
    "iban": "salary goes to DE89370400440532013000",
    "ip": "the kiosk at 10.20.30.40",
    "passport": "Passport No: K1234567",
    "date_of_birth": "DOB: 1990-04-12",
    "employee_id": "raised by EMP-100234",
    "bank_account": "account 123456789012",
    "api_key": "it printed gsk_abcdefghij1234567890ABCDEFGHIJ",
}

# Text from the knowledge base and from ordinary HR questions. Nothing here is
# personal data, and a redactor that eats any of it corrupts the question the
# model is asked -- which is a worse outcome than a miss.
INNOCENT = [
    "Full-time employees accrue 20 days of paid time off.",
    "Form 16 is issued in June.",
    "Core hours are 10:00 to 16:00.",
    "See section 4.2 of the handbook.",
    "The notice period is 30 days, or 90 days for senior staff.",
    "Expenses are paid within 15 days of approval.",
    "This policy is effective 2024-04-01 and supersedes HR2024001.",
    "I get 1.5 days of PTO per month.",
]


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_every_pattern_matches_a_realistic_example(kind):
    kinds = {match.kind for match in find_pii(EXAMPLES[kind])}
    assert kind in kinds


def test_every_kind_in_the_pattern_table_has_an_example():
    assert set(PII_PATTERNS) == set(EXAMPLES)


@pytest.mark.parametrize("text", INNOCENT)
def test_ordinary_policy_text_is_left_completely_alone(text):
    assert find_pii(text) == []
    assert redact(text) == (text, [])


def test_a_sixteen_digit_number_that_fails_luhn_is_not_a_card():
    # An order number, an invoice reference or an amount. Treating it as a card
    # is the false positive that gets a redactor switched off.
    assert luhn("1234567890123456") is False
    kinds = {match.kind for match in find_pii("order 1234567890123456 was late")}
    assert "credit_card" not in kinds


def test_a_card_number_beats_the_generic_long_number_pattern_on_the_same_span():
    # The same sixteen characters match both `credit_card` and `bank_account`.
    # Only one match may survive, and it has to be the one covering every
    # digit: resolving to the wrong reading is how digits get left behind.
    matches = find_pii("card 4111111111111111 on file")
    assert [m.kind for m in matches] == ["credit_card"]
    assert matches[0].text == "4111111111111111"
    assert "1111" not in redact("card 4111111111111111 on file")[0]


def test_an_overlapping_group_resolves_to_the_longest_match():
    # A spaced card number contains an aadhaar-shaped twelve-digit prefix.
    # Longest-first is what stops the last group of four surviving redaction.
    text = "charge 4111 1111 1111 1111 please"
    matches = find_pii(text)
    assert len(matches) == 1
    assert matches[0].kind == "credit_card"
    assert matches[0].text == "4111 1111 1111 1111"


def test_redaction_keeps_the_surrounding_text_intact_and_in_order():
    text = (
        "Hi, I am priya.sharma@example.com, my mobile is +91 98765 43210 "
        "and my id is EMP-100234 -- can you check my leave balance?"
    )
    cleaned, kinds = redact(text)
    assert cleaned == (
        "Hi, I am [REDACTED_EMAIL], my mobile is [REDACTED_PHONE] "
        "and my id is [REDACTED_EMPLOYEE_ID] -- can you check my leave balance?"
    )
    assert kinds == ["email", "phone", "employee_id"]


def test_redaction_can_be_narrowed_to_one_kind():
    text = "priya.sharma@example.com, 10.20.30.40"
    cleaned, kinds = redact(text, kinds=["email"])
    assert kinds == ["email"]
    assert "10.20.30.40" in cleaned
    assert "priya" not in cleaned


def test_summarise_counts_by_kind():
    matches = find_pii("a@x.com and b@y.com and 10.20.30.40")
    assert summarise(matches) == {"email": 2, "ip": 1}
    assert summarise([]) == {}


@pytest.mark.parametrize(
    "key",
    [
        "sk-proj-abcdefghij1234567890ABCDEF",
        "gsk_abcdefghij1234567890ABCDEFGHIJ",
        "pcsk_abcdefghij1234567890ABCDEFGH",
        "tvly-abcdefghij1234567890ABCDEF",
    ],
)
def test_each_provider_key_shape_this_repo_handles_is_caught(key):
    # The four prefixes that actually appear in this project's `.env`. A
    # visitor pasting one into the chat box must not have it written into the
    # audit log, which is the store designed to still hold it in a year.
    cleaned, kinds = redact(f"my call fails with {key} -- why?")
    assert kinds == ["api_key"]
    assert key not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned


def test_has_pii_answers_the_cheap_question():
    assert has_pii("call me on 555-123-4567") is True
    assert has_pii("how much leave do I get?") is False


def test_matches_come_back_sorted_by_position():
    text = "10.20.30.40 then a@b.com then EMP-100234"
    starts = [match.start for match in find_pii(text)]
    assert starts == sorted(starts)


def test_the_placeholder_names_the_kind_it_replaced():
    match = find_pii("a@b.com")[0]
    assert match.placeholder == "[REDACTED_EMAIL]"


def test_luhn_ignores_separators_and_refuses_the_too_short():
    assert luhn("4111 1111 1111 1111") is True
    assert luhn("4") is False
    assert luhn("") is False
