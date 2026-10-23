"""The shared scratchpad: what gets written, what gets rendered, what gets lost.

Every test builds its own store under `tmp_path`, so none of them touches the
configured `state.db` and the suite stays offline and order-independent.
"""

from __future__ import annotations

import pytest

from app.services.scratchpad import Scratchpad
from app.services.state_db import StateStore

CONV = "c1b2a3d4-0000-4000-8000-000000000001"
OTHER = "c1b2a3d4-0000-4000-8000-000000000002"


@pytest.fixture
def pad(tmp_path):
    return Scratchpad(StateStore(tmp_path / "state.db"))


def test_an_appended_note_round_trips(pad):
    note = pad.append(CONV, "PTO accrues at 1.67 days a month.", author="policy",
                      kind="finding")

    assert note is not None
    assert note.author == "policy"
    assert note.kind == "finding"
    read = pad.notes(CONV)
    assert [n.text for n in read] == ["PTO accrues at 1.67 days a month."]
    assert read[0].id == note.id


def test_notes_come_back_oldest_first(pad):
    for text in ("first", "second", "third"):
        pad.append(CONV, text)

    assert [n.text for n in pad.notes(CONV)] == ["first", "second", "third"]


def test_a_tight_limit_keeps_the_recent_notes_not_the_first_ones(pad):
    """`limit` selects newest-first and the result is reversed, so a small
    window shows the end of the pad rather than its beginning."""
    for text in ("first", "second", "third"):
        pad.append(CONV, text)

    assert [n.text for n in pad.notes(CONV, limit=2)] == ["second", "third"]


def test_notes_can_be_filtered_by_kind(pad):
    pad.append(CONV, "the plan", kind="plan")
    pad.append(CONV, "a finding", kind="finding")

    assert [n.kind for n in pad.notes(CONV, kinds=["finding"])] == ["finding"]


def test_a_long_note_is_truncated_rather_than_refused(tmp_path):
    pad = Scratchpad(StateStore(tmp_path / "state.db"), max_chars=50)

    note = pad.append(CONV, "x" * 400)

    assert note is not None
    assert len(note.text) == 50
    assert note.text.endswith("…")


def test_render_respects_max_chars_and_still_includes_a_pinned_note(pad):
    pinned = pad.append(CONV, "The employee is a contractor.", kind="decision")
    assert pinned is not None
    pad.pin(pinned.id)
    for index in range(12):
        pad.append(CONV, f"filler note number {index} " + "y" * 60, kind="finding")

    rendered = pad.render(CONV, max_chars=300)

    assert "The employee is a contractor." in rendered
    assert len(rendered) <= 300 + 120  # the pinned line is admitted before the budget
    # The budget is applied newest-first, so the last filler note survives and
    # the first does not.
    assert "filler note number 11" in rendered
    assert "filler note number 0 " not in rendered


def test_render_reads_oldest_first(pad):
    pad.append(CONV, "earlier")
    pad.append(CONV, "later")

    rendered = pad.render(CONV)

    assert rendered.index("earlier") < rendered.index("later")


def test_render_is_empty_for_a_conversation_with_no_pad(pad):
    assert pad.render(CONV) == ""


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "../../etc/passwd", "a/b", "a b", "-leading-dash", "x" * 65, None],
)
def test_an_invalid_conversation_id_is_rejected(pad, bad):
    """Rejected, not sanitised. A sanitised id silently merges two
    conversations, which means one employee's notes in another's prompt."""
    assert pad.append(bad, "a note") is None
    assert pad.notes(bad) == []


def test_clear_removes_only_that_conversation(pad):
    pad.append(CONV, "mine")
    pad.append(OTHER, "theirs")

    removed = pad.clear(CONV)

    assert removed == 1
    assert pad.notes(CONV) == []
    assert [n.text for n in pad.notes(OTHER)] == ["theirs"]


def test_pin_reports_whether_a_row_changed(pad):
    note = pad.append(CONV, "worth keeping")
    assert note is not None

    assert pad.pin(note.id) is True
    assert pad.notes(CONV)[0].pinned is True
    assert pad.pin(note.id + 9999) is False


def test_a_write_failure_returns_none_without_raising(tmp_path):
    """An unwritable path degrades the pad; it does not fail the answer.

    The database path is put *inside* a regular file, so creating the parent
    directory cannot succeed.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied", encoding="utf-8")
    pad = Scratchpad(StateStore(blocker / "nested" / "state.db"))

    assert pad.append(CONV, "a note") is None
    assert pad.notes(CONV) == []
    assert pad.render(CONV) == ""


def test_an_unknown_kind_is_recorded_as_a_plain_note(pad):
    note = pad.append(CONV, "something", kind="speculation")  # type: ignore[arg-type]

    assert note is not None
    assert note.kind == "note"


def test_a_full_pad_stops_accepting_notes(tmp_path):
    pad = Scratchpad(StateStore(tmp_path / "state.db"), max_notes=3)
    for index in range(3):
        assert pad.append(CONV, f"note {index}") is not None

    assert pad.append(CONV, "one too many") is None
    assert len(pad.notes(CONV)) == 3


def test_recent_conversations_groups_by_conversation(pad):
    pad.append(CONV, "one")
    pad.append(CONV, "two")
    pad.append(OTHER, "only one here")

    recent = pad.recent_conversations()

    # Most recently written first.
    assert [row["conversation_id"] for row in recent] == [OTHER, CONV]
    assert {row["conversation_id"]: row["notes"] for row in recent} == {
        OTHER: 1,
        CONV: 2,
    }


def test_stats_counts_notes_conversations_and_kinds(pad):
    pad.append(CONV, "the plan", kind="plan")
    note = pad.append(CONV, "a finding", kind="finding")
    assert note is not None
    pad.pin(note.id)
    pad.append(OTHER, "elsewhere")

    stats = pad.stats()

    assert stats["notes"] == 3
    assert stats["conversations"] == 2
    assert stats["pinned"] == 1
    assert stats["by_kind"]["plan"] == 1
