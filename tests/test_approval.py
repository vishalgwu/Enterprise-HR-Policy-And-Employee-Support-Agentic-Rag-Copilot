"""The approval queue: landing pending, being decided once, and expiring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.approval import ApprovalStore
from app.services.state_db import StateStore, now_iso


def store_at(tmp_path, **kwargs) -> ApprovalStore:
    return ApprovalStore(StateStore(tmp_path / "state.db"), **kwargs)


def test_a_request_lands_pending(tmp_path):
    store = store_at(tmp_path)
    request = store.request(
        "ticket.create",
        question="Can someone look at my payslip?",
        reason="The agent could not answer from policy.",
        payload={"queue": "payroll"},
        risk="low",
        requested_by="abc123",
        conversation_id="conv-1",
    )

    assert request is not None
    assert request.status == "pending"
    assert request.is_pending is True
    assert request.decided_at is None
    assert request.payload == {"queue": "payroll"}

    stored = store.get(request.id)
    assert stored is not None
    assert stored.action == "ticket.create"
    assert stored.risk == "low"
    assert stored.requested_by == "abc123"
    assert stored.conversation_id == "conv-1"
    assert stored.payload == {"queue": "payroll"}


def test_approving_records_who_decided_and_when(tmp_path):
    store = store_at(tmp_path)
    request = store.request("ticket.create", question="q")
    assert request is not None

    decided = store.decide(request.id, "approved", decided_by="hr", note="looks fine")
    assert decided is not None
    assert decided.status == "approved"
    assert decided.decided_by == "hr"
    assert decided.note == "looks fine"
    assert decided.decided_at is not None
    assert decided.is_pending is False


def test_deciding_twice_returns_the_first_decision_unchanged(tmp_path):
    # Two HR staff a second apart must not produce last-write-wins on something
    # that may already have had a side effect.
    store = store_at(tmp_path)
    request = store.request("ticket.create", question="q")
    assert request is not None

    first = store.decide(request.id, "approved", decided_by="hr-one")
    second = store.decide(request.id, "rejected", decided_by="hr-two", note="no")

    assert first is not None and second is not None
    assert second.status == "approved"
    assert second.decided_by == "hr-one"
    assert second.note is None
    assert second.decided_at == first.decided_at
    assert store.get(request.id).status == "approved"


def test_an_invalid_decision_raises_value_error(tmp_path):
    store = store_at(tmp_path)
    request = store.request("ticket.create")
    assert request is not None
    with pytest.raises(ValueError):
        store.decide(request.id, "maybe", decided_by="hr")
    assert store.get(request.id).status == "pending"


def test_deciding_something_that_does_not_exist_returns_none(tmp_path):
    assert store_at(tmp_path).decide(999, "approved", decided_by="hr") is None


def test_pending_excludes_decided_rows_and_is_oldest_first(tmp_path):
    store = store_at(tmp_path)
    first = store.request("ticket.create", question="one")
    second = store.request("kb.write", question="two")
    third = store.request("escalate", question="three")
    assert first and second and third

    store.decide(second.id, "rejected", decided_by="hr")

    assert [row.id for row in store.pending()] == [first.id, third.id]
    assert [row.id for row in store.recent()] == [third.id, second.id, first.id]
    assert [row.id for row in store.recent(status="rejected")] == [second.id]


def test_expire_stale_only_touches_rows_past_the_ttl(tmp_path):
    state = StateStore(tmp_path / "state.db")
    store = ApprovalStore(state, ttl_hours=72)

    fresh = store.request("ticket.create", question="fresh")
    assert fresh is not None
    old_created = (datetime.now(UTC) - timedelta(hours=100)).isoformat(
        timespec="seconds"
    )
    old_id = state.execute(
        "INSERT INTO approvals (created_at, status, action) VALUES (?,?,?)",
        (old_created, "pending", "ticket.create"),
    )
    assert old_id is not None

    assert store.expire_stale() == 1
    assert store.get(old_id).status == "expired"
    assert store.get(fresh.id).status == "pending"
    # An expiry is not a rejection: nobody declined it, nobody looked.
    assert store.get(old_id).decided_by is None
    assert store.expire_stale() == 0


def test_a_write_failure_returns_none_without_raising(tmp_path):
    # The parent of the database path is a file, so the store cannot even make
    # its directory. A degraded queue beats a 500 -- but a None here means the
    # approval was not recorded, so the caller must refuse.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    store = ApprovalStore(StateStore(blocker / "state.db"))

    assert store.request("ticket.create", question="q") is None
    assert store.pending() == []
    assert store.get(1) is None


def test_unreadable_payload_json_degrades_to_an_empty_dict(tmp_path):
    state = StateStore(tmp_path / "state.db")
    store = ApprovalStore(state)
    row_id = state.execute(
        "INSERT INTO approvals (created_at, status, action, payload) "
        "VALUES (?,?,?,?)",
        (now_iso(), "pending", "ticket.create", "{not json"),
    )
    assert row_id is not None

    stored = store.get(row_id)
    assert stored is not None
    assert stored.payload == {}
    # One bad row must not hide the rows around it.
    assert [row.id for row in store.pending()] == [row_id]


def test_a_payload_that_is_not_an_object_degrades_too(tmp_path):
    state = StateStore(tmp_path / "state.db")
    store = ApprovalStore(state)
    row_id = state.execute(
        "INSERT INTO approvals (created_at, status, action, payload) "
        "VALUES (?,?,?,?)",
        (now_iso(), "pending", "ticket.create", "[1, 2, 3]"),
    )
    assert row_id is not None
    assert store.get(row_id).payload == {}


def test_an_unserialisable_payload_still_queues_the_request(tmp_path):
    store = store_at(tmp_path)
    request = store.request("ticket.create", payload={"when": object()})
    assert request is not None
    assert store.get(request.id).payload == {}


def test_stats_add_up(tmp_path):
    state = StateStore(tmp_path / "state.db")
    store = ApprovalStore(state, ttl_hours=1)

    approved = store.request("ticket.create", question="a")
    rejected = store.request("ticket.create", question="b")
    store.request("ticket.create", question="c")
    assert approved and rejected

    store.decide(approved.id, "approved", decided_by="hr")
    store.decide(rejected.id, "rejected", decided_by="hr")
    state.execute(
        "INSERT INTO approvals (created_at, status, action) VALUES (?,?,?)",
        (
            (datetime.now(UTC) - timedelta(hours=5)).isoformat(timespec="seconds"),
            "pending",
            "ticket.create",
        ),
    )

    stats = store.stats()
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 2
    assert stats["expired"] == 0
    assert stats["total"] == 4
    assert stats["oldest_pending_age_hours"] >= 4.9

    store.expire_stale()
    after = store.stats()
    assert after["expired"] == 1
    assert after["pending"] == 1
    assert after["total"] == 4


def test_stats_on_an_empty_queue_report_nothing_waiting(tmp_path):
    stats = store_at(tmp_path).stats()
    assert stats["total"] == 0
    assert stats["oldest_pending_age_hours"] is None


def test_the_store_needs_a_store_or_a_path():
    with pytest.raises(ValueError):
        ApprovalStore()


def test_a_db_path_is_enough_to_build_one(tmp_path):
    store = ApprovalStore(db_path=tmp_path / "state.db")
    request = store.request("ticket.create", question="q")
    assert request is not None
    assert store.get(request.id).question == "q"


def test_a_request_serialises_for_the_api(tmp_path):
    store = store_at(tmp_path)
    request = store.request("kb.write", question="q", payload={"file": "pto.md"})
    assert request is not None
    data = request.to_dict()
    assert data["status"] == "pending"
    assert data["is_pending"] is True
    assert data["payload"] == {"file": "pto.md"}
    assert data["action"] == "kb.write"
