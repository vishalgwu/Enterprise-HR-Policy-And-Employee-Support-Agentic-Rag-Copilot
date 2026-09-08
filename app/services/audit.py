"""The audit log: what was asked, what was answered, and how it was decided.

An HR copilot answers questions about pay, leave and conduct, so "what did it
tell someone about the notice period in March?" is a question the business will
eventually have to answer. That needs a record written at answer time, because
the graph is not deterministic -- `temperature=0` is not determinism on Groq,
and re-running the question later can produce a different route and a different
answer.

SQLite because the target architecture mounts a volume for exactly this, and
because one file with no server is the right size for a per-deployment log.
`audit_db_path` is a setting rather than a fixed path so that volume can live
anywhere.

**Recording must never cost the employee their answer.** `record()` catches
everything and returns None: a full disk, a locked database or a read-only mount
degrades the audit trail, and degrading the audit trail is much cheaper than
turning a working answer into a 500. The failure is logged at ERROR, so a
silently unwritten log is visible in the logs rather than only in its own
absence.

**A connection per call, deliberately.** FastAPI runs `def` endpoints in a
threadpool, so handlers do not share a thread; one long-lived `sqlite3`
connection would need `check_same_thread=False` and its own locking. Opening per
call costs microseconds against an LLM round trip that costs seconds, and WAL
mode lets a reader run while a writer holds the file.

This module imports `AnswerResult` from `copilot` and never the other way round.
Both are service-layer, so the layering table does not order them; the one-way
direction is what keeps them from becoming a cycle.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_logger, get_settings
from app.services.copilot import AnswerResult

log = get_logger("audit")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at        TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    route           TEXT,
    source_used     TEXT,
    grounded        INTEGER NOT NULL DEFAULT 0,
    kb_grade        TEXT,
    web_grade       TEXT,
    kb_chunks       INTEGER NOT NULL DEFAULT 0,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    rewritten_query TEXT,
    sources         TEXT    NOT NULL DEFAULT '[]',
    web_urls        TEXT    NOT NULL DEFAULT '[]',
    trace           TEXT    NOT NULL DEFAULT '[]',
    latency_ms      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chat_audit_asked_at ON chat_audit(asked_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_audit_source ON chat_audit(source_used);
"""

# Columns holding a JSON array. Listed once so reading and writing cannot
# disagree about which fields need decoding.
JSON_COLUMNS = ("sources", "web_urls", "trace")


class AuditStore:
    """A SQLite-backed record of every answered question.

    The schema is created on first use rather than in `__init__`, because
    `app.main` builds an application at import and a constructor that touches
    the filesystem would make importing the module write a database file.
    """

    def __init__(self, db_path: Path | str, timeout: float = 5.0) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        self._ready = False

    # --- Connection ----------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, creating the schema the first time through."""
        if not self._ready:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        try:
            if not self._ready:
                # WAL lets /audit read while /chat is writing. It is a property
                # of the database file, so setting it once is enough -- but it
                # is cheap and idempotent, and this runs only until _ready.
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(SCHEMA)
                connection.commit()
                self._ready = True
            yield connection
        finally:
            connection.close()

    # --- Writing -------------------------------------------------------------
    def record(
        self, result: AnswerResult, latency_ms: int | None = None
    ) -> int | None:
        """Store one answered question. Returns its id, or None if it failed.

        Never raises. An audit write that fails must not turn a good answer into
        an error response, so the caller gets None and the reason goes to the
        log at ERROR.
        """
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO chat_audit (
                        asked_at, question, answer, route, source_used, grounded,
                        kb_grade, web_grade, kb_chunks, retry_count,
                        rewritten_query, sources, web_urls, trace, latency_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        result.question,
                        result.answer,
                        result.route,
                        result.source_used,
                        int(result.grounded),
                        result.kb_grade,
                        result.web_grade,
                        result.kb_chunks,
                        result.retry_count,
                        result.rewritten_query,
                        json.dumps([s.model_dump() for s in result.sources]),
                        json.dumps(result.web_urls),
                        json.dumps(result.trace),
                        latency_ms,
                    ),
                )
                connection.commit()
                return int(cursor.lastrowid or 0) or None
        except Exception:  # noqa: BLE001 - the answer matters more than the log
            log.exception("Could not write the audit record; the answer still stands")
            return None

    # --- Reading -------------------------------------------------------------
    def recent(
        self, limit: int = 50, offset: int = 0, source_used: str | None = None
    ) -> list[dict[str, Any]]:
        """The newest entries first, optionally narrowed to one source.

        Ordered by id rather than `asked_at`: the timestamp has one-second
        resolution, so two questions in the same second would come back in an
        arbitrary order, and the id is monotonic by construction.
        """
        query = "SELECT * FROM chat_audit"
        params: list[Any] = []
        if source_used:
            query += " WHERE source_used = ?"
            params.append(source_used)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get(self, entry_id: int) -> dict[str, Any] | None:
        """One entry by id, or None when it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_audit WHERE id = ?", (entry_id,)
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def count(self, source_used: str | None = None) -> int:
        """How many entries there are, for paging the list endpoint."""
        query = "SELECT COUNT(*) FROM chat_audit"
        params: list[Any] = []
        if source_used:
            query += " WHERE source_used = ?"
            params.append(source_used)
        with self._connect() as connection:
            return int(connection.execute(query, params).fetchone()[0])

    def stats(self) -> dict[str, Any]:
        """Aggregate health of the answers, not of the process.

        `grounded_rate` is the number worth watching: it is the share of answers
        that rested on retrieved evidence rather than ending in an admission or
        an error, and a fall in it is how a broken ingest, an exhausted token
        quota or a drifting threshold shows up from outside.
        """
        with self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM chat_audit").fetchone()[0]
            )
            by_source = {
                str(row["source_used"] or "unknown"): int(row["n"])
                for row in connection.execute(
                    "SELECT source_used, COUNT(*) AS n FROM chat_audit "
                    "GROUP BY source_used ORDER BY n DESC"
                )
            }
            grounded = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_audit WHERE grounded = 1"
                ).fetchone()[0]
            )
            latency = connection.execute(
                "SELECT AVG(latency_ms) FROM chat_audit WHERE latency_ms IS NOT NULL"
            ).fetchone()[0]
            rewrites = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_audit WHERE retry_count > 0"
                ).fetchone()[0]
            )

        return {
            "total": total,
            "by_source": by_source,
            "grounded": grounded,
            "grounded_rate": round(grounded / total, 3) if total else 0.0,
            "rewritten": rewrites,
            "average_latency_ms": int(latency) if latency is not None else None,
        }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Decode one row, turning the JSON columns back into lists.

    A row written before a schema change -- or by hand -- may hold something
    that is not JSON. That is a reason to return an empty list for that field,
    not to fail the whole request for the entries around it.
    """
    entry = dict(row)
    for column in JSON_COLUMNS:
        try:
            entry[column] = json.loads(entry.get(column) or "[]")
        except (TypeError, ValueError):
            log.warning("Audit row %s has unreadable %s", entry.get("id"), column)
            entry[column] = []
    entry["grounded"] = bool(entry.get("grounded"))
    return entry


def get_audit_store(settings: Settings | None = None) -> AuditStore:
    """An AuditStore over the configured database path."""
    settings = settings or get_settings()
    return AuditStore(settings.audit_db_path)
