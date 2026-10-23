"""One SQLite file for the agent's own state, and the schema that defines it.

Four features need to remember something between requests: the scratchpad an
agent writes its working notes into, the approval queue a human decides from,
the long-term facts a returning employee should not have to repeat, and the
idempotency ledger that stops a retried side effect happening twice. All four
are small, all four are per-deployment, and all four are read and written by the
same process.

They share one file for a reason that is not tidiness. Each of them separately
would need the same WAL-mode dance, the same create-on-first-use, the same
additive migration and the same "never raise into the request" discipline that
`app/services/audit.py` documents at length -- and four copies of that is four
chances for one of them to be the one that fails silently. The schema lives here
in one place, applied once, so an added column is added in the place a reader
looks.

**Separate from `audit.db`, deliberately.** The audit log is the compliance
record: every employee question and the answer they were given, admin-gated,
mounted on a volume, and the one file a deployment must not lose. This is
working state -- a dropped scratchpad note costs an agent its notes, a dropped
approval row costs a decision that can be made again. Keeping them in one file
would mean the backup policy for the compliance record had to also cover a table
that churns on every request, and that the audit log's size told you nothing
about how many questions were asked.

**Nothing here raises into a request.** Every writer wraps its own call the way
`AuditStore.record` does: a full disk degrades a feature, and degrading a
feature is cheaper than turning a working answer into a 500. Readers return
empty rather than erroring, because a scratchpad that cannot be read is an agent
with no notes, which is how it starts every conversation anyway.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_logger

log = get_logger("state")

__all__ = ["StateStore", "now_iso"]


def now_iso() -> str:
    """UTC, to the second. One function so four tables sort together."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# The whole schema, in one script. `IF NOT EXISTS` throughout, so this is safe
# to run on every first connection of a process.
SCHEMA = """
-- Working notes an agent writes while it reasons, scoped to a conversation.
-- `author` is the agent or worker that wrote it; `kind` separates a plan from
-- a finding from a question so the UI can style them and a prompt can include
-- only what it needs.
CREATE TABLE IF NOT EXISTS scratchpad (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    conversation_id TEXT    NOT NULL,
    author          TEXT    NOT NULL DEFAULT 'agent',
    kind            TEXT    NOT NULL DEFAULT 'note',
    text            TEXT    NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0
);

-- The human-in-the-loop queue. A side-effecting tool call that needs a person
-- to say yes lands here as `pending` and the agent stops; a decision later
-- moves it to `approved` or `rejected`, with who decided and why.
CREATE TABLE IF NOT EXISTS approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    decided_at      TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    action          TEXT    NOT NULL,
    reason          TEXT    NOT NULL DEFAULT '',
    question        TEXT    NOT NULL DEFAULT '',
    payload         TEXT    NOT NULL DEFAULT '{}',
    conversation_id TEXT,
    requested_by    TEXT    NOT NULL DEFAULT 'employee',
    decided_by      TEXT,
    note            TEXT,
    risk            TEXT    NOT NULL DEFAULT 'medium'
);

-- Long-term facts about a person, so a returning employee is not asked their
-- department every time. Keyed on (subject, key) so a later value replaces an
-- earlier one rather than accumulating contradictions.
CREATE TABLE IF NOT EXISTS facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    subject         TEXT    NOT NULL,
    key             TEXT    NOT NULL,
    value           TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'conversation',
    confidence      REAL    NOT NULL DEFAULT 0.6,
    UNIQUE (subject, key)
);

-- The idempotency ledger: a fingerprint of a side-effecting request, the
-- result it produced, and when the claim expires. `state` distinguishes a claim
-- that is still running from one that finished, which is what makes a retry
-- arriving mid-flight wait rather than duplicate.
CREATE TABLE IF NOT EXISTS idempotency (
    key             TEXT    PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    expires_at      TEXT    NOT NULL,
    state           TEXT    NOT NULL DEFAULT 'in_flight',
    action          TEXT    NOT NULL DEFAULT '',
    result          TEXT
);
"""

# Applied after the migration, because an index cannot be built on a column that
# does not exist yet -- the same ordering, and the same reason, as
# `app/services/audit.py`.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_scratchpad_conv
    ON scratchpad(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_approvals_status
    ON approvals(status, id DESC);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_idempotency_expiry ON idempotency(expires_at);
"""

# Columns added after a table first shipped, as {table: {column: SQL type}}.
# `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
# a database written by an earlier version keeps its old shape and every INSERT
# naming a new column fails with "no such column" -- which the writers swallow,
# so the symptom is a feature that silently stops working. Anything added to
# SCHEMA belongs here too.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "scratchpad": {"pinned": "INTEGER NOT NULL DEFAULT 0"},
    "approvals": {"risk": "TEXT NOT NULL DEFAULT 'medium'"},
}


def _migrate(connection: sqlite3.Connection) -> None:
    """Add any column this version writes that an older database lacks."""
    for table, columns in ADDED_COLUMNS.items():
        present = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not present:  # the table itself is new; SCHEMA just created it
            continue
        for column, sql_type in columns.items():
            if column not in present:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                )
                log.info("State database migrated: %s gained %s", table, column)


class StateStore:
    """Connection management for the agent's state file.

    The schema is created on first *use*, not in `__init__`, for the same reason
    `AuditStore` does it: `app.main` builds an application at import, and a
    constructor that touched the filesystem would make importing the module
    write a database file.

    A connection per call, like `AuditStore`: FastAPI runs `def` endpoints in a
    threadpool so handlers do not share a thread, and one long-lived
    `sqlite3` connection would need `check_same_thread=False` plus its own
    locking. Opening costs microseconds against an LLM round trip that costs
    seconds.
    """

    def __init__(self, db_path: Path | str, timeout: float = 5.0) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._ready = False
        self._ready_lock = threading.Lock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, applying the schema the first time through."""
        with self._ready_lock:
            first = not self._ready
            if first:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        try:
            if first:
                # WAL lets a reader run while a writer holds the file, which is
                # what lets /api/approvals list while /api/chat is appending a
                # scratchpad note.
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(SCHEMA)
                _migrate(connection)
                connection.executescript(INDEXES)
                connection.commit()
                with self._ready_lock:
                    self._ready = True
            yield connection
        finally:
            connection.close()

    # --- Convenience ---------------------------------------------------------
    def execute(self, sql: str, params: Any = ()) -> int | None:
        """Run one statement and commit. Returns the new rowid, or None on error.

        Swallowing is the contract: see the module docstring. The reason is
        logged at ERROR so a feature that has silently stopped working is
        visible in the logs rather than only in its own absence.
        """
        try:
            with self.connect() as connection:
                cursor = connection.execute(sql, params)
                connection.commit()
                return int(cursor.lastrowid or 0) or None
        except Exception:  # noqa: BLE001 - a degraded feature beats a 500
            log.exception("State write failed: %s", sql.split("\n", 1)[0][:80])
            return None

    def query(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        """Run one SELECT. Returns [] on error, for the same reason."""
        try:
            with self.connect() as connection:
                return list(connection.execute(sql, params).fetchall())
        except Exception:  # noqa: BLE001
            log.exception("State read failed: %s", sql.split("\n", 1)[0][:80])
            return []

    def scalar(self, sql: str, params: Any = (), default: Any = 0) -> Any:
        rows = self.query(sql, params)
        if not rows:
            return default
        value = rows[0][0]
        return default if value is None else value
