"""The human in the loop: a queue of things the agent stopped to ask about.

Some actions should not happen because a language model decided they should.
Raising a ticket in someone's name, writing to a system of record, mailing a
manager: the agent can propose one, but a person approves it. A proposal lands
here as `pending` and the run stops; a decision later moves it to `approved` or
`rejected`, with who decided and when.

The `approvals` table lives in `app/services/state_db.py`, with the rest of the
agent's working state and deliberately not in the audit log -- a dropped
approval row costs a decision that can be made again, while the audit log is
the compliance record a deployment must not lose. This module is the reader and
writer; it does not define the table.

**A failed write returns `None`, and the caller must fail closed on it.** That
follows the contract `StateStore` and `AuditStore.record` both hold -- a full
disk degrades a feature rather than turning a working answer into a 500 -- but
the consequence here is the opposite way round from everywhere else in this
codebase. A `None` from `request()` means **the approval was not recorded**, so
there is no queue entry for anyone to decide and no record that the agent ever
asked. The tool must therefore refuse to proceed. The tempting reading of
`None` is "carry on without it", and carrying on is precisely the thing the
approval existed to prevent.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_logger
from app.services.state_db import StateStore, now_iso

log = get_logger("approval")

__all__ = ["ApprovalRequest", "ApprovalStore", "Risk", "Status"]

Status = Literal["pending", "approved", "rejected", "expired"]
Risk = Literal["low", "medium", "high"]

# The only two things a human may decide. Anything else is a programming error
# in the caller rather than a bad input from a user, so `decide` raises on it
# instead of returning None and letting a typo look like a write failure.
DECISIONS: tuple[str, ...] = ("approved", "rejected")


@dataclass(slots=True)
class ApprovalRequest:
    """One row of the queue."""

    id: int
    created_at: str
    status: Status
    action: str
    reason: str
    question: str
    payload: dict[str, Any]
    risk: Risk
    requested_by: str
    conversation_id: str | None = None
    decided_at: str | None = None
    decided_by: str | None = None
    note: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "question": self.question,
            "payload": dict(self.payload),
            "risk": self.risk,
            "requested_by": self.requested_by,
            "conversation_id": self.conversation_id,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "note": self.note,
            "is_pending": self.is_pending,
        }


class ApprovalStore:
    """Reads and writes the `approvals` table.

    Takes a `StateStore` so a test can point it at `tmp_path`, or a `db_path`
    for a caller that has no store to hand. No work happens in `__init__`: the
    schema is created on the store's first connection, which is what keeps
    building an application at import from writing a database file.
    """

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        db_path: Path | str | None = None,
        ttl_hours: int = 72,
    ) -> None:
        if store is None:
            if db_path is None:
                raise ValueError("ApprovalStore needs either a store or a db_path.")
            store = StateStore(db_path)
        self.store = store
        self.ttl_hours = ttl_hours

    # --- Writing -------------------------------------------------------------
    def request(
        self,
        action: str,
        *,
        question: str = "",
        reason: str = "",
        payload: dict[str, Any] | None = None,
        risk: Risk = "medium",
        requested_by: str = "employee",
        conversation_id: str | None = None,
    ) -> ApprovalRequest | None:
        """Queue one proposal. Returns it, or None if it could not be written.

        Never raises. **A `None` means the request is not in the queue**, so the
        caller must refuse the action rather than continue -- see the module
        docstring.
        """
        created_at = now_iso()
        body = _encode(payload)
        row_id = self.store.execute(
            """
            INSERT INTO approvals (
                created_at, status, action, reason, question, payload,
                conversation_id, requested_by, risk
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                created_at,
                "pending",
                action,
                reason,
                question,
                body,
                conversation_id,
                requested_by,
                risk,
            ),
        )
        if row_id is None:
            # `StateStore.execute` has already logged the reason with its
            # traceback. This line is what tells a reader of the logs which
            # feature degraded.
            log.error("Approval for %r was not queued; the caller must refuse", action)
            return None
        return ApprovalRequest(
            id=row_id,
            created_at=created_at,
            status="pending",
            action=action,
            reason=reason,
            question=question,
            payload=dict(payload or {}),
            risk=risk,
            requested_by=requested_by,
            conversation_id=conversation_id,
        )

    def decide(
        self,
        request_id: int,
        decision: str,
        *,
        decided_by: str,
        note: str | None = None,
    ) -> ApprovalRequest | None:
        """Approve or reject a pending request. Single-shot and idempotent.

        Deciding a request that already has a decision returns the existing row
        **unchanged** and logs the attempt at warning. Not last-write-wins:
        an approval may already have had a side effect by the time the second
        click arrives, and two HR staff pressing approve and reject a second
        apart would otherwise leave the queue claiming the thing was rejected
        while it had in fact been done. The first decision is the decision.

        The guard is in the UPDATE itself -- `WHERE id = ? AND status =
        'pending'` -- rather than a read followed by a write, so two callers
        racing inside the same second still produce one winner.
        """
        if decision not in DECISIONS:
            raise ValueError(
                f"A decision must be one of {DECISIONS}, not {decision!r}."
            )
        try:
            with self.store.connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE approvals
                       SET status = ?, decided_at = ?, decided_by = ?, note = ?
                     WHERE id = ? AND status = 'pending'
                    """,
                    (decision, now_iso(), decided_by, note, request_id),
                )
                connection.commit()
                changed = cursor.rowcount
                row = connection.execute(
                    "SELECT * FROM approvals WHERE id = ?", (request_id,)
                ).fetchone()
        except Exception:  # noqa: BLE001 - a degraded queue beats a 500
            log.exception("Could not decide approval %s", request_id)
            return None

        if row is None:
            log.warning("Approval %s does not exist", request_id)
            return None
        if not changed:
            log.warning(
                "Approval %s is already %s; left the first decision alone",
                request_id,
                row["status"],
            )
        return _to_request(row)

    def expire_stale(self) -> int:
        """Move pending requests older than `ttl_hours` to `expired`.

        An approval queue with no expiry becomes a list nobody reads: a month of
        undecided rows at the top is indistinguishable from a broken feature,
        and the one request that mattered is below them.

        `expired` is its own status and must not be read as `rejected`. Nobody
        declined this -- nobody looked. The two have different follow-ups: a
        rejection is an answer to give the employee, an expiry is a queue that
        is not being worked.
        """
        cutoff = (
            datetime.now(UTC) - timedelta(hours=self.ttl_hours)
        ).isoformat(timespec="seconds")
        try:
            with self.store.connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE approvals
                       SET status = 'expired', decided_at = ?
                     WHERE status = 'pending' AND created_at < ?
                    """,
                    (now_iso(), cutoff),
                )
                connection.commit()
                expired = int(cursor.rowcount or 0)
        except Exception:  # noqa: BLE001
            log.exception("Could not expire stale approvals")
            return 0
        if expired:
            log.info("Expired %s approval request(s) past %sh", expired, self.ttl_hours)
        return expired

    # --- Reading -------------------------------------------------------------
    def get(self, request_id: int) -> ApprovalRequest | None:
        rows = self.store.query("SELECT * FROM approvals WHERE id = ?", (request_id,))
        return _to_request(rows[0]) if rows else None

    def pending(self, limit: int = 50) -> list[ApprovalRequest]:
        """The undecided queue, oldest first -- it is worked in order."""
        rows = self.store.query(
            "SELECT * FROM approvals WHERE status = 'pending' "
            "ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        return [_to_request(row) for row in rows]

    def recent(
        self, limit: int = 50, status: Status | None = None
    ) -> list[ApprovalRequest]:
        """The newest rows first, optionally narrowed to one status.

        Ordered by id rather than `created_at`: the timestamp has one-second
        resolution, so two requests in the same second would come back in an
        arbitrary order.
        """
        sql = "SELECT * FROM approvals"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [_to_request(row) for row in self.store.query(sql, params)]

    def stats(self) -> dict[str, Any]:
        """Counts by status, plus how long the oldest pending row has waited.

        The age is the number worth watching: a queue with four pending rows is
        healthy, and a queue whose oldest pending row is three days old is one
        nobody is looking at.
        """
        counts = {status: 0 for status in ("pending", "approved", "rejected", "expired")}
        for row in self.store.query(
            "SELECT status, COUNT(*) AS n FROM approvals GROUP BY status"
        ):
            counts[str(row["status"])] = int(row["n"])
        oldest = self.store.scalar(
            "SELECT MIN(created_at) FROM approvals WHERE status = 'pending'",
            default=None,
        )
        return {
            **counts,
            "total": sum(counts.values()),
            "oldest_pending_age_hours": _age_hours(oldest),
        }


# --- Helpers ----------------------------------------------------------------
def _encode(payload: dict[str, Any] | None) -> str:
    """The payload as JSON, degrading to `{}` rather than failing the write.

    A payload carrying something unserialisable is a bug in the caller, but it
    is not a reason to lose the approval request itself -- the action, the
    reason and the question are what a human decides from.
    """
    try:
        return json.dumps(payload or {})
    except (TypeError, ValueError):
        log.warning("An approval payload was not serialisable; stored {} instead")
        return "{}"


def _to_request(row: sqlite3.Row) -> ApprovalRequest:
    """One row as an `ApprovalRequest`.

    An unreadable `payload` decodes to `{}` with a warning rather than failing
    the whole listing, the same choice `_row_to_dict` in `app/services/audit.py`
    makes: one bad row must not hide the rows around it from the person whose
    job is to work through them.
    """
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        log.warning("Approval %s has unreadable payload JSON", row["id"])
        payload = {}
    if not isinstance(payload, dict):
        log.warning("Approval %s has a payload that is not an object", row["id"])
        payload = {}
    return ApprovalRequest(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        status=row["status"],
        action=str(row["action"]),
        reason=str(row["reason"] or ""),
        question=str(row["question"] or ""),
        payload=payload,
        risk=row["risk"] or "medium",
        requested_by=str(row["requested_by"] or "employee"),
        conversation_id=row["conversation_id"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
        note=row["note"],
    )


def _age_hours(timestamp: str | None) -> float | None:
    """How long ago an ISO timestamp was, in hours, or None if unreadable."""
    if not timestamp:
        return None
    try:
        then = datetime.fromisoformat(timestamp)
    except ValueError:
        log.warning("An approval row has an unreadable created_at")
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return round((datetime.now(UTC) - then).total_seconds() / 3600, 2)
