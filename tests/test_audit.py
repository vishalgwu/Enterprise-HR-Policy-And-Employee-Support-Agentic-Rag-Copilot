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
    """Every metric has to survive a log with nothing in it.

    Asserted as "no value is truthy" rather than against a literal dict: the
    metrics endpoint keeps gaining aggregates, and a literal would make adding
    one a two-file change for no safety. The key set *is* pinned below, so a new
    aggregate still has to be added here deliberately -- and the thing that
    would actually break, a rate dividing by a zero total, is a ZeroDivisionError
    this call would raise before any assertion ran.
    """
    stats = store(tmp_path).stats()

    assert stats["total"] == 0
    for name, value in stats.items():
        assert not value, f"{name} should be empty, zero or None on an empty log"

    assert set(stats) == {
        "total",
        "by_source",
        "grounded",
        "grounded_rate",
        "rewritten",
        "average_latency_ms",
        "by_mode",
        "by_model",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_samples",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "cost_per_answer_usd",
        "cached",
        "cache_hit_rate",
        "byok",
        "escalated",
        "escalation_rate",
        "tool_calls",
        "agent_steps",
        "guardrail_flagged",
        "pii_redacted_answers",
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


# --- Which form of the question is kept --------------------------------------


def _redacted_answer():
    """An answer whose question was altered on the way out of the process."""
    return AnswerResult(
        question="Call me on 555-123-4567 about my leave",
        redacted_question="Call me on [REDACTED_PHONE] about my leave",
        answer="Twenty days.",
        source_used="private_kb",
        pii_redacted=["phone"],
    )


def test_the_audit_log_stores_the_redacted_question_by_default(tmp_path):
    """The compliance record is long-lived and readable by an administrator.

    The one thing worse than an investigator missing a detail is this file
    being the place somebody's phone number ended up.
    """
    audit = store(tmp_path)
    audit.record(_redacted_answer())
    entry = audit.recent()[0]

    assert "555-123-4567" not in entry["question"]
    assert "[REDACTED_PHONE]" in entry["question"]
    assert entry["pii_redacted"] == ["phone"]


def test_the_verbatim_question_can_be_kept_deliberately(tmp_path):
    """`AUDIT_STORES_REDACTED=0` is the opposite decision, for a deployment
    whose investigators need exactly what was typed.

    It had no effect at all until this was wired: the setting was reported on
    `/api/config` and read nowhere, so a deployment that turned it off kept
    storing the redacted form and had no way to notice.
    """
    audit = AuditStore(tmp_path / "audit.db", store_redacted=False)
    audit.record(_redacted_answer())
    entry = audit.recent()[0]

    assert entry["question"] == "Call me on 555-123-4567 about my leave"


def test_a_question_that_was_never_redacted_is_stored_either_way(tmp_path):
    """`redacted_question` is only set when redaction changed something, so
    both settings have to fall back to the original rather than to None."""
    plain = AnswerResult(question="What are the core hours?", answer="10 to 4.")

    for store_redacted in (True, False):
        audit = AuditStore(
            tmp_path / f"audit-{store_redacted}.db", store_redacted=store_redacted
        )
        audit.record(plain)
        assert audit.recent()[0]["question"] == "What are the core hours?"
