"""The ledger that stops a retry happening twice.

A retried tool call is the whole point of retries, and it is also how an HR
ticket gets raised twice. The two are the same mechanism seen from different
ends: the registry cannot tell a request that failed from a request whose
*reply* failed, so retrying is the only way to survive a flaky provider and is
also the only way to duplicate a side effect.

The distinction between a tool that may be retried freely and one that may not
is a property of the tool, declared once on its `ToolSpec` (`idempotent`,
`side_effecting`) rather than decided at each call site. This module is what
makes the second kind safe to retry at all: a fingerprint of the request is
claimed *before* the side effect runs, and a second call carrying the same
fingerprint is handed the first call's result instead of performing a second
side effect.

**In-memory mode is per process, and it is not a distributed lock.** With no
store and no path, the ledger is a dict behind a lock -- correct for one
process, worth nothing across two. Vercel runs each function instance
separately, so two concurrent requests can land in two processes that share no
memory and each win the same claim. Idempotency across instances needs the
SQLite path on a shared volume, or an external store (Redis, a row in the
provider's own database with a unique constraint). In-memory is the right
default for a single container and the wrong one for anything that scales
horizontally; nothing here pretends otherwise.

**Everything swallows and logs, with one exception.** A read or a write that
fails degrades the ledger, which is the contract `app/services/state_db.py`
sets out. But a *failure to claim* is reported as such -- `claim()` returns an
`in_flight` claim -- because the caller's correct response to "I could not
reserve this" is to refuse, not to proceed. Fail closed: a duplicated HR ticket
is worse than a refused one.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.config import get_logger
from app.services.state_db import StateStore

log = get_logger("idempotency")

__all__ = ["Claim", "IdempotencyStore", "fingerprint"]

STATE_IN_FLIGHT = "in_flight"
STATE_DONE = "done"

WON = "won"
REPLAYED = "replayed"


def _canonical(value: Any) -> Any:
    """How `fingerprint` renders a value json cannot.

    Sets are sorted rather than repr'd: a set's iteration order is not stable
    across processes, so its repr would fingerprint the same request two
    different ways and the ledger would stop matching anything.
    """
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def fingerprint(*parts: Any) -> str:
    """A stable 32-hex-character digest of a request.

    Stability across processes is the requirement, not uniqueness in the
    abstract: the claim is written by one request and matched by another, which
    on a container restart is a different interpreter. So sorted keys, a
    canonical separator set, and nothing that varies per process -- no `hash()`
    (PYTHONHASHSEED randomises it for strings), no `id()`, no timestamp.
    """
    rendered = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=_canonical
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class Claim:
    """The answer to "may I perform this side effect?".

    Three states rather than a bool, because "somebody else is doing it right
    now" is neither a yes nor a safe no: replaying a result we do not have would
    invent one, and running anyway is the duplicate this module exists to
    prevent. The caller refuses and the employee is told to try again.
    """

    key: str
    state: str  # "won" | "replayed" | "in_flight"
    result: Any = None

    @property
    def is_new(self) -> bool:
        """We won the claim; do the work, then `complete()` it."""
        return self.state == WON

    @property
    def is_replay(self) -> bool:
        """Somebody already did the work; use `.result` and do not repeat it."""
        return self.state == REPLAYED


def _as_datetime(stamp: Any) -> datetime | None:
    if isinstance(stamp, datetime):
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _passed(stamp: Any, now: datetime) -> bool:
    """Whether a deadline is behind us. An unreadable stamp counts as passed.

    A row whose `expires_at` cannot be parsed is a row no later reader will ever
    be able to expire, so treating it as live would leave it blocking its key
    for good.
    """
    moment = _as_datetime(stamp)
    return moment is None or moment <= now


def _older_than(stamp: Any, seconds: float, now: datetime) -> bool:
    moment = _as_datetime(stamp)
    if moment is None:
        return True
    return (now - moment).total_seconds() > seconds


def _dumps(result: Any) -> str:
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(result))


def _loads(payload: Any) -> Any:
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return payload


class IdempotencyStore:
    """SQLite-backed when given a path, in-memory when not.

    Uses the `idempotency` table defined in `app/services/state_db.py`; the
    schema lives there so an added column is added in the place a reader looks.

    `ttl_seconds` is how long a completed result stays replayable -- a day, long
    enough that a retry an hour later still finds it and short enough that the
    table does not grow without bound. `stall_seconds` is something else
    entirely: see `claim()`.
    """

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        db_path: Path | str | None = None,
        ttl_seconds: int = 86_400,
        stall_seconds: float = 60.0,
    ) -> None:
        if store is None and db_path is not None:
            store = StateStore(db_path)
        self._store = store
        self.ttl_seconds = int(ttl_seconds)
        self.stall_seconds = float(stall_seconds)
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(*parts: Any) -> str:
        """The module function, reachable through an injected store.

        `app/agent/tools.py` is agent-layer and may not import this module, so
        the registry asks the store it was handed for the key rather than
        keeping a second copy of the canonicalisation. Two copies is two
        chances for one of them to be changed alone, and the failure that
        produces is a ledger that silently stops matching.
        """
        return fingerprint(*parts)

    @property
    def backend(self) -> str:
        return "sqlite" if self._store is not None else "memory"

    # --- Claiming ------------------------------------------------------------
    def claim(self, key: str, action: str = "") -> Claim:
        """Reserve `key` before performing its side effect.

        **Atomic, and that is the whole value of it.** `INSERT ... ON CONFLICT
        DO NOTHING` followed by a read, inside one `BEGIN IMMEDIATE`, rather
        than a SELECT and then an INSERT: two threads running the latter both
        see no row, both insert, and both perform the side effect -- which is
        precisely the duplicate this ledger exists to prevent, reintroduced by
        the check that was supposed to prevent it. `rowcount` on the insert is
        what says whether *we* created the row.

        **A stalled claim may be re-won.** An `in_flight` row older than
        `stall_seconds` is treated as abandoned and deleted, with a warning. A
        process killed mid-call otherwise leaves a key nobody will ever
        complete or release, and since the row survives for the full TTL that
        converts one crash into a day of refused retries. Sixty seconds is
        comfortably past any tool timeout the registry allows, so a live call is
        never stolen from.
        """
        if self._store is None:
            return self._claim_memory(key, action)

        now = datetime.now(UTC)
        try:
            with self._store.connect() as connection:
                # Autocommit, so the explicit BEGIN IMMEDIATE below is the only
                # transaction: sqlite3's implicit one is DEFERRED, which takes
                # its write lock late and can lose the upgrade to a concurrent
                # writer.
                connection.isolation_level = None
                connection.execute("BEGIN IMMEDIATE")
                try:
                    self._clear_stale(connection, key, now)
                    cursor = connection.execute(
                        "INSERT INTO idempotency "
                        "(key, created_at, expires_at, state, action) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(key) DO NOTHING",
                        (
                            key,
                            now.isoformat(timespec="seconds"),
                            (
                                now + timedelta(seconds=self.ttl_seconds)
                            ).isoformat(timespec="seconds"),
                            STATE_IN_FLIGHT,
                            action,
                        ),
                    )
                    won = cursor.rowcount == 1
                    row = connection.execute(
                        "SELECT state, result FROM idempotency WHERE key = ?", (key,)
                    ).fetchone()
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
        except Exception:  # noqa: BLE001 - reported as a refusal, see below
            # The one thing this module does not swallow into a success. A
            # caller that cannot reserve must refuse, so the failure is handed
            # back as `in_flight` rather than as a won claim.
            log.exception("Idempotency claim failed for %s", key[:16])
            return Claim(key=key, state=STATE_IN_FLIGHT)

        if won:
            return Claim(key=key, state=WON)
        if row is not None and row["state"] == STATE_DONE:
            return Claim(key=key, state=REPLAYED, result=_loads(row["result"]))
        return Claim(key=key, state=STATE_IN_FLIGHT)

    def _clear_stale(self, connection: Any, key: str, now: datetime) -> None:
        """Delete the row holding `key` if it is expired or abandoned."""
        row = connection.execute(
            "SELECT created_at, expires_at, state FROM idempotency WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return
        if _passed(row["expires_at"], now):
            connection.execute("DELETE FROM idempotency WHERE key = ?", (key,))
            return
        if row["state"] == STATE_IN_FLIGHT and _older_than(
            row["created_at"], self.stall_seconds, now
        ):
            log.warning(
                "Idempotency claim %s has been in flight since %s (over %.0fs); "
                "treating it as abandoned so a retry is not refused for the "
                "whole TTL",
                key[:16],
                row["created_at"],
                self.stall_seconds,
            )
            connection.execute("DELETE FROM idempotency WHERE key = ?", (key,))

    def _claim_memory(self, key: str, action: str) -> Claim:
        """The same semantics in one process, behind a lock."""
        now = datetime.now(UTC)
        with self._lock:
            row = self._memory.get(key)
            if row is not None:
                expired = _passed(row["expires_at"], now)
                stalled = row["state"] == STATE_IN_FLIGHT and _older_than(
                    row["created_at"], self.stall_seconds, now
                )
                if stalled and not expired:
                    log.warning(
                        "Idempotency claim %s abandoned in flight; re-winnable",
                        key[:16],
                    )
                if expired or stalled:
                    self._memory.pop(key, None)
                    row = None
            if row is None:
                self._memory[key] = {
                    "created_at": now,
                    "expires_at": now + timedelta(seconds=self.ttl_seconds),
                    "state": STATE_IN_FLIGHT,
                    "action": action,
                    "result": None,
                }
                return Claim(key=key, state=WON)
            if row["state"] == STATE_DONE:
                return Claim(key=key, state=REPLAYED, result=row["result"])
            return Claim(key=key, state=STATE_IN_FLIGHT)

    # --- Finishing -----------------------------------------------------------
    def complete(self, key: str, result: Any) -> None:
        """Record what the side effect produced, so a retry replays it."""
        if self._store is None:
            with self._lock:
                row = self._memory.get(key)
                if row is None:
                    # Released or expired under us; there is nothing to flip and
                    # inventing a row would make a result replayable that no
                    # claim is holding.
                    return
                row["state"] = STATE_DONE
                # Round-tripped so both backends hand back the same shape: the
                # SQLite one necessarily goes through json, and a caller that
                # got a tuple from one and a list from the other would have two
                # behaviours to test.
                row["result"] = _loads(_dumps(result))
            return
        self._store.execute(
            "UPDATE idempotency SET state = ?, result = ? WHERE key = ?",
            (STATE_DONE, _dumps(result), key),
        )

    def release(self, key: str) -> None:
        """The work failed; drop the claim so a retry may try again.

        Deletes rather than marking failed. A failed attempt left in the table
        would have to be distinguished from a live one by every later reader,
        and the only thing anybody wants to know is whether the side effect
        happened -- which it did not.
        """
        if self._store is None:
            with self._lock:
                self._memory.pop(key, None)
            return
        self._store.execute("DELETE FROM idempotency WHERE key = ?", (key,))

    # --- Reading -------------------------------------------------------------
    def lookup(self, key: str) -> Any | None:
        """The stored result for a completed claim, or None."""
        now = datetime.now(UTC)
        if self._store is None:
            with self._lock:
                row = self._memory.get(key)
                if row is None or row["state"] != STATE_DONE:
                    return None
                return None if _passed(row["expires_at"], now) else row["result"]
        rows = self._store.query(
            "SELECT result, expires_at FROM idempotency "
            "WHERE key = ? AND state = ?",
            (key, STATE_DONE),
        )
        if not rows:
            return None
        if _passed(rows[0]["expires_at"], now):
            return None
        return _loads(rows[0]["result"])

    def purge_expired(self) -> int:
        """Drop every claim past its TTL. Returns how many went."""
        now = datetime.now(UTC)
        if self._store is None:
            with self._lock:
                stale = [
                    key
                    for key, row in self._memory.items()
                    if _passed(row["expires_at"], now)
                ]
                for key in stale:
                    del self._memory[key]
                return len(stale)
        try:
            with self._store.connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM idempotency WHERE expires_at <= ?",
                    (now.isoformat(timespec="seconds"),),
                )
                connection.commit()
                return int(cursor.rowcount or 0)
        except Exception:  # noqa: BLE001 - a degraded purge beats a 500
            log.exception("Purging expired idempotency claims failed")
            return 0

    def stats(self) -> dict[str, Any]:
        """Counts by state, plus which backend is answering.

        `backend` is on here deliberately: "memory" is the single fact an
        operator needs to know before trusting the ledger behind more than one
        instance.
        """
        if self._store is None:
            with self._lock:
                states = [row["state"] for row in self._memory.values()]
            return {
                "backend": self.backend,
                "entries": len(states),
                "in_flight": states.count(STATE_IN_FLIGHT),
                "done": states.count(STATE_DONE),
                "ttl_seconds": self.ttl_seconds,
            }
        rows = self._store.query(
            "SELECT state, COUNT(*) AS n FROM idempotency GROUP BY state"
        )
        counts = {str(row["state"]): int(row["n"]) for row in rows}
        return {
            "backend": self.backend,
            "entries": sum(counts.values()),
            "in_flight": counts.get(STATE_IN_FLIGHT, 0),
            "done": counts.get(STATE_DONE, 0),
            "ttl_seconds": self.ttl_seconds,
        }
