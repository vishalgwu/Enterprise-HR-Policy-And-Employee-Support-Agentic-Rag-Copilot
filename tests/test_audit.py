"""The audit store: what gets written, what survives, and what must not raise."""

from __future__ import annotations

import json
import sqlite3

from app.services.audit import AuditStore
from app.services.copilot import AnswerResult, SourceRef


def answer(**overrides) -> AnswerResult:
    base = dict(
        question="How many PTO days do I get?",
        answer="Twenty days.",
        route="kb",
        source_used="private_kb",
        retry_count=0,
        kb_grade="good",
        kb_chunks=1,
        sources=[SourceRef(source="leave-and-time-off.md", title="Leave")],
        web_urls=[],
        trace=["Router: kb", "KB Grader: good"],
    )
    base.update(overrides)
    return AnswerResult(**base)


def store(tmp_path) -> AuditStore:
    return AuditStore(tmp_path / "audit.db")


# --- Writing and reading -----------------------------------------------------


def test_an_answer_round_trips_with_its_lists_intact(tmp_path):
    audit = store(tmp_path)
    entry_id = audit.record(answer(), latency_ms=1234)

    entry = audit.get(entry_id)
    assert entry["question"] == "How many PTO days do I get?"
    assert entry["source_used"] == "private_kb"
    assert entry["latency_ms"] == 1234
    assert entry["grounded"] is True
    # The JSON columns come back as lists, not as the strings they are stored as.
    assert entry["trace"] == ["Router: kb", "KB Grader: good"]
    assert entry["sources"][0]["source"] == "leave-and-time-off.md"
    assert entry["web_urls"] == []


def test_the_schema_is_created_on_first_use_not_at_construction(tmp_path):
    """`app.main` builds an application at import; constructing a store there
    must not write a database file."""
    path = tmp_path / "nested" / "audit.db"
    audit = AuditStore(path)
    assert not path.exists()

    audit.record(answer())
    assert path.exists()


def test_a_missing_entry_is_none_not_an_error(tmp_path):
    assert store(tmp_path).get(999) is None


def test_entries_come_back_newest_first(tmp_path):
    audit = store(tmp_path)
    for n in range(3):
        audit.record(answer(question=f"question {n}"))

    assert [e["question"] for e in audit.recent()] == [
        "question 2",
        "question 1",
        "question 0",
    ]


def test_entries_can_be_narrowed_to_one_source(tmp_path):
    audit = store(tmp_path)
    audit.record(answer())
    audit.record(answer(source_used="web_search"))
    audit.record(answer(source_used="direct"))

    assert audit.count() == 3
    assert audit.count(source_used="web_search") == 1
    assert [e["source_used"] for e in audit.recent(source_used="web_search")] == [
        "web_search"
    ]


def test_paging_uses_the_id_not_the_timestamp(tmp_path):
    """`asked_at` has one-second resolution, so three questions asked in the
    same second would otherwise page in an arbitrary order."""
    audit = store(tmp_path)
    for n in range(5):
        audit.record(answer(question=f"q{n}"))

    page_one = audit.recent(limit=2, offset=0)
    page_two = audit.recent(limit=2, offset=2)
    assert [e["question"] for e in page_one] == ["q4", "q3"]
    assert [e["question"] for e in page_two] == ["q2", "q1"]


# --- Stats -------------------------------------------------------------------


def test_stats_report_how_answers_were_resolved(tmp_path):
    audit = store(tmp_path)
    audit.record(answer(), latency_ms=100)
    audit.record(answer(source_used="web_search"), latency_ms=300)
    audit.record(answer(source_used="insufficient_evidence"), latency_ms=200)

    stats = audit.stats()
    assert stats["total"] == 3
    assert stats["by_source"]["private_kb"] == 1
    # Only the KB and web answers rest on evidence.
    assert stats["grounded"] == 2
    assert stats["grounded_rate"] == round(2 / 3, 3)
    assert stats["average_latency_ms"] == 200


def test_stats_on_an_empty_log_do_not_divide_by_zero(tmp_path):
    stats = store(tmp_path).stats()
    assert stats == {
        "total": 0,
        "by_source": {},
        "grounded": 0,
        "grounded_rate": 0.0,
        "rewritten": 0,
        "average_latency_ms": None,
    }


def test_a_rewritten_question_is_counted(tmp_path):
    audit = store(tmp_path)
    audit.record(answer())
    audit.record(answer(retry_count=1, rewritten_query="annual PTO allowance"))
    assert audit.stats()["rewritten"] == 1


# --- Degradation -------------------------------------------------------------


def test_a_failed_write_returns_none_rather_than_costing_the_answer(tmp_path):
    """A full disk or a read-only mount degrades the audit trail. Degrading the
    audit trail is much cheaper than turning a good answer into a 500."""
    unwritable = tmp_path / "audit.db"
    unwritable.mkdir()  # a directory where the database file should be

    assert AuditStore(unwritable).record(answer()) is None


def test_an_unreadable_json_column_degrades_to_an_empty_list(tmp_path):
    """One corrupt row must not fail the request for the entries around it."""
    audit = store(tmp_path)
    entry_id = audit.record(answer())

    with sqlite3.connect(tmp_path / "audit.db") as connection:
        connection.execute(
            "UPDATE chat_audit SET trace = ? WHERE id = ?", ("not json", entry_id)
        )
        connection.commit()

    entry = audit.get(entry_id)
    assert entry["trace"] == []
    # The rest of the row still reads.
    assert entry["question"] == "How many PTO days do I get?"


def test_the_stored_lists_really_are_json(tmp_path):
    """Pinned so a change to the write path cannot start storing a repr."""
    audit = store(tmp_path)
    audit.record(answer())
    with sqlite3.connect(tmp_path / "audit.db") as connection:
        raw = connection.execute("SELECT trace FROM chat_audit").fetchone()[0]
    assert json.loads(raw) == ["Router: kb", "KB Grader: good"]
