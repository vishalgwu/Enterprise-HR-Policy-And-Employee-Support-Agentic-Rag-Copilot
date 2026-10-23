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

from app.core.config import get_logger
from app.services.copilot import AnswerResult

log = get_logger("audit")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at        TEXT    NOT NULL,
    conversation_id TEXT,
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
    resolved_question TEXT,
    sources         TEXT    NOT NULL DEFAULT '[]',
    web_urls        TEXT    NOT NULL DEFAULT '[]',
    trace           TEXT    NOT NULL DEFAULT '[]',
    latency_ms      INTEGER,

    -- Added with the production layers. Split between columns and JSON on one
    -- rule: a field the metrics endpoint groups or sums is a column, and a
    -- field only ever read back whole for one row is JSON. Grouping on a JSON
    -- field means loading every row into Python to answer "cost by model",
    -- which stops being viable at exactly the point the dashboard gets
    -- interesting.
    answer_mode     TEXT,
    model           TEXT,
    provider        TEXT,
    model_tier      TEXT,
    complexity      TEXT,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0,
    cached          INTEGER NOT NULL DEFAULT 0,
    byok            INTEGER NOT NULL DEFAULT 0,
    escalated       INTEGER NOT NULL DEFAULT 0,
    steps           INTEGER NOT NULL DEFAULT 0,
    tool_calls      INTEGER NOT NULL DEFAULT 0,
    stop_reason     TEXT,
    -- A hashed, non-identifying caller id. See `fingerprint_subject` in
    -- `app/services/authorization.py` for why it is a salted hash and not an
    -- address: this row is long-lived and admin-readable, and an IP in it is
    -- personal data the product has no reason to keep.
    subject         TEXT,
    worker_path     TEXT    NOT NULL DEFAULT '[]',
    handoffs        TEXT    NOT NULL DEFAULT '[]',
    spans           TEXT    NOT NULL DEFAULT '[]',
    budget          TEXT    NOT NULL DEFAULT '{}',
    guard_flags     TEXT    NOT NULL DEFAULT '{}',
    pii_redacted    TEXT    NOT NULL DEFAULT '[]'
);
"""

# Separate from SCHEMA, and applied *after* the migration, because an index can
# only be built on a column that exists. On a database written before
# `conversation_id` did, running these together fails on the index and takes the
# whole schema step with it -- which `record()` then swallows, so the audit log
# simply stops. The migration test pins the order.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chat_audit_asked_at ON chat_audit(asked_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_audit_source ON chat_audit(source_used);
CREATE INDEX IF NOT EXISTS idx_chat_audit_conv ON chat_audit(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_audit_model ON chat_audit(model);
CREATE INDEX IF NOT EXISTS idx_chat_audit_mode ON chat_audit(answer_mode);
"""

# Columns holding a JSON array. Listed once so reading and writing cannot
# disagree about which fields need decoding.
JSON_COLUMNS = (
    "sources",
    "web_urls",
    "trace",
    "worker_path",
    "handoffs",
    "spans",
    "pii_redacted",
)
# Columns holding a JSON *object*. Separate from the list above because the
# empty value differs, and decoding an object to `[]` on a bad row would make
# `entry["budget"]["steps"]` a TypeError in the UI rather than a missing key.
JSON_OBJECT_COLUMNS = ("budget", "guard_flags")

# Columns added after the first release, as {name: SQL type}. `CREATE TABLE IF
# NOT EXISTS` does nothing to a table that already exists, so a database written
# by an earlier version keeps its old shape and every INSERT naming a new column
# fails with "no such column". `record()` catches everything, so the symptom
# would be an audit log that silently stopped growing -- the one failure mode
# this store exists to prevent. Anything added to SCHEMA belongs here too.
ADDED_COLUMNS = {
    "conversation_id": "TEXT",
    "resolved_question": "TEXT",
    "answer_mode": "TEXT",
    "model": "TEXT",
    "provider": "TEXT",
    "model_tier": "TEXT",
    "complexity": "TEXT",
    "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",
    "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
    "cost_usd": "REAL NOT NULL DEFAULT 0",
    "cached": "INTEGER NOT NULL DEFAULT 0",
    "byok": "INTEGER NOT NULL DEFAULT 0",
    "escalated": "INTEGER NOT NULL DEFAULT 0",
    "steps": "INTEGER NOT NULL DEFAULT 0",
    "tool_calls": "INTEGER NOT NULL DEFAULT 0",
    "stop_reason": "TEXT",
    "subject": "TEXT",
    "worker_path": "TEXT NOT NULL DEFAULT '[]'",
    "handoffs": "TEXT NOT NULL DEFAULT '[]'",
    "spans": "TEXT NOT NULL DEFAULT '[]'",
    "budget": "TEXT NOT NULL DEFAULT '{}'",
    "guard_flags": "TEXT NOT NULL DEFAULT '{}'",
    "pii_redacted": "TEXT NOT NULL DEFAULT '[]'",
}


def _migrate(connection: sqlite3.Connection) -> None:
    """Add any column this version writes that an older database lacks.

    Only ever additive: SQLite can add a nullable column to a populated table
    cheaply, and existing rows get NULL, which is the honest value for "this
    predates the column".
    """
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(chat_audit)")}
    for column, sql_type in ADDED_COLUMNS.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE chat_audit ADD COLUMN {column} {sql_type}")
            log.info("Audit database migrated: added column %s", column)


class AuditStore:
    """A SQLite-backed record of every answered question.

    The schema is created on first use rather than in `__init__`, because
    `app.main` builds an application at import and a constructor that touches
    the filesystem would make importing the module write a database file.
    """

    def __init__(
        self,
        db_path: Path | str,
        timeout: float = 5.0,
        store_spans: bool = True,
        store_redacted: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        # Whether this log keeps the redacted question or the employee's
        # verbatim wording. Redacted by default: this record is long-lived and
        # readable by an administrator, and the one thing worse than an
        # investigator missing a detail is the compliance record being the
        # place somebody's phone number ended up.
        #
        # `AUDIT_STORES_REDACTED=0` is the deliberate opposite, for a
        # deployment whose investigators need exactly what was typed. It is a
        # separate decision from `REDACT_PII`, which governs what leaves the
        # process -- turning redaction off entirely makes this setting moot,
        # because then there is no redacted form to prefer.
        self.store_redacted = store_redacted
        # Spans are the largest thing on a row -- forty entries of a few
        # hundred bytes each -- and they are the one field nothing aggregates.
        # Switchable off for a deployment whose audit volume is large enough
        # that the compliance record should not also be a performance archive.
        self.store_spans = store_spans
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
                _migrate(connection)
                connection.executescript(INDEXES)
                connection.commit()
                self._ready = True
            yield connection
        finally:
            connection.close()

    # --- Writing -------------------------------------------------------------
    def record(
        self,
        result: AnswerResult,
        latency_ms: int | None = None,
        conversation_id: str | None = None,
        subject: str | None = None,
    ) -> int | None:
        """Store one answered question. Returns its id, or None if it failed.

        Never raises. An audit write that fails must not turn a good answer into
        an error response, so the caller gets None and the reason goes to the
        log at ERROR.

        **The question stored is `result.audited_question` unless this store was
        built with `store_redacted=False`.** When `REDACT_PII` is on the two
        differ: the employee's verbatim wording is what the UI shows back to
        them, and the redacted form is what belongs in a record that is
        long-lived and readable by an administrator. `AnswerResult.audited_
        question` decides which string is which, so that rule lives in one
        place; this decides whether to use it.
        """
        getter = result.model_dump() if hasattr(result, "model_dump") else {}

        def field(name: str, default: Any = None) -> Any:
            """Read a field that an older `AnswerResult` may not have.

            Tolerant on purpose: `AuditStore` is constructed with whatever
            result object the caller has, and a test building a minimal one
            must not have to fill in twenty production fields to assert that a
            row was written.
            """
            value = getattr(result, name, None)
            if value is None:
                value = getter.get(name, default)
            return default if value is None else value

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO chat_audit (
                        asked_at, conversation_id, question, answer, route,
                        source_used, grounded, kb_grade, web_grade, kb_chunks,
                        retry_count, rewritten_query, resolved_question,
                        sources, web_urls, trace, latency_ms,
                        answer_mode, model, provider, model_tier, complexity,
                        prompt_tokens, completion_tokens, cost_usd, cached,
                        byok, escalated, steps, tool_calls, stop_reason,
                        subject, worker_path, handoffs, spans, budget,
                        guard_flags, pii_redacted
                    ) VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    (
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        conversation_id or field("conversation_id"),
                        (
                            field("audited_question")
                            if self.store_redacted
                            else result.question
                        )
                        or result.question,
                        result.answer,
                        result.route,
                        result.source_used,
                        int(result.grounded),
                        result.kb_grade,
                        result.web_grade,
                        result.kb_chunks,
                        result.retry_count,
                        result.rewritten_query,
                        result.resolved_question,
                        json.dumps([s.model_dump() for s in result.sources]),
                        json.dumps(result.web_urls),
                        json.dumps(result.trace),
                        latency_ms if latency_ms is not None else field("latency_ms"),
                        field("answer_mode", "graph"),
                        field("model"),
                        field("provider"),
                        field("model_tier"),
                        field("complexity"),
                        int(field("prompt_tokens", 0)),
                        int(field("completion_tokens", 0)),
                        float(field("cost_usd", 0.0)),
                        int(bool(field("cached", False))),
                        int(bool(field("byok", False))),
                        int(bool(field("escalated", False))),
                        int(field("steps", 0)),
                        int(field("tool_calls", 0)),
                        field("stop_reason"),
                        subject or field("subject"),
                        json.dumps(field("worker_path", [])),
                        json.dumps(field("handoffs", [])),
                        json.dumps(field("spans", []) if self.store_spans else []),
                        json.dumps(field("budget", {})),
                        json.dumps(_guard_flags(result)),
                        json.dumps(field("pii_redacted", [])),
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
            by_mode = {
                str(row["answer_mode"] or "graph"): int(row["n"])
                for row in connection.execute(
                    "SELECT answer_mode, COUNT(*) AS n FROM chat_audit "
                    "GROUP BY answer_mode ORDER BY n DESC"
                )
            }
            by_model = {
                str(row["model"] or "unreported"): {
                    "answers": int(row["n"]),
                    "tokens": int(row["tok"] or 0),
                    "cost_usd": round(float(row["cost"] or 0.0), 6),
                }
                for row in connection.execute(
                    "SELECT model, COUNT(*) AS n, "
                    "       SUM(prompt_tokens + completion_tokens) AS tok, "
                    "       SUM(cost_usd) AS cost "
                    "FROM chat_audit GROUP BY model ORDER BY n DESC"
                )
            }
            totals = connection.execute(
                "SELECT SUM(prompt_tokens) AS pin, SUM(completion_tokens) AS pout, "
                "       SUM(cost_usd) AS cost, SUM(cached) AS cached, "
                "       SUM(byok) AS byok, SUM(escalated) AS escalated, "
                "       SUM(tool_calls) AS tools, SUM(steps) AS steps "
                "FROM chat_audit"
            ).fetchone()
            # Bounded, and that bound is the point: percentiles are computed in
            # Python, so an unbounded SELECT would load the whole log into
            # memory to answer a dashboard request. The newest thousand is also
            # the right window -- a p95 over a deployment's entire history is a
            # number that stops moving when it matters most.
            latencies = sorted(
                float(row[0])
                for row in connection.execute(
                    "SELECT latency_ms FROM chat_audit "
                    "WHERE latency_ms IS NOT NULL ORDER BY id DESC LIMIT 1000"
                )
                if row[0] is not None
            )
            guarded = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_audit "
                    "WHERE guard_flags IS NOT NULL AND guard_flags NOT IN ('', '{}')"
                ).fetchone()[0]
            )
            redacted = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_audit "
                    "WHERE pii_redacted IS NOT NULL "
                    "AND pii_redacted NOT IN ('', '[]')"
                ).fetchone()[0]
            )

        cached = int(totals["cached"] or 0) if totals is not None else 0
        return {
            "total": total,
            "by_source": by_source,
            "grounded": grounded,
            "grounded_rate": round(grounded / total, 3) if total else 0.0,
            "rewritten": rewrites,
            "average_latency_ms": int(latency) if latency is not None else None,
            # --- Added with the production layers ---------------------------
            "by_mode": by_mode,
            "by_model": by_model,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_p99_ms": _percentile(latencies, 0.99),
            "latency_samples": len(latencies),
            "prompt_tokens": int((totals["pin"] if totals else 0) or 0),
            "completion_tokens": int((totals["pout"] if totals else 0) or 0),
            "cost_usd": round(float((totals["cost"] if totals else 0.0) or 0.0), 6),
            "cost_per_answer_usd": (
                round(float(totals["cost"] or 0.0) / total, 6)
                if total and totals is not None
                else 0.0
            ),
            "cached": cached,
            "cache_hit_rate": round(cached / total, 3) if total else 0.0,
            "byok": int((totals["byok"] if totals else 0) or 0),
            "escalated": int((totals["escalated"] if totals else 0) or 0),
            "escalation_rate": (
                round(float(totals["escalated"] or 0) / total, 3)
                if total and totals is not None
                else 0.0
            ),
            "tool_calls": int((totals["tools"] if totals else 0) or 0),
            "agent_steps": int((totals["steps"] if totals else 0) or 0),
            "guardrail_flagged": guarded,
            "pii_redacted_answers": redacted,
        }


def _guard_flags(result: Any) -> dict[str, Any]:
    """The guardrail verdicts, reduced to what is worth searching for.

    Categories and severities only -- never the text either guard saw. The
    question and the answer are already columns on this row, and a verdict
    carrying a second copy of the question would mean redaction had to be got
    right in two places.
    """
    flags: dict[str, Any] = {}
    for side in ("guard_input", "guard_output"):
        verdict = getattr(result, side, None) or {}
        if not isinstance(verdict, dict) or not verdict:
            continue
        reduced = {
            "category": verdict.get("category"),
            "severity": verdict.get("severity"),
            "allowed": verdict.get("allowed", True),
            "modified": verdict.get("modified", False),
            "reasons": list(verdict.get("reasons") or [])[:5],
        }
        if reduced["category"] not in (None, "ok") or reduced["modified"]:
            flags[side.removeprefix("guard_")] = reduced
    return flags


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile over an already sorted list.

    Computed in Python rather than in SQL because SQLite has no percentile
    function without an extension, and because the alternative -- averaging --
    is the statistic that hides the problem: a p95 of nine seconds and a mean of
    two tell completely different stories about the same deployment, and only
    one of them is what a user experienced.
    """
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Decode one row, turning the JSON columns back into lists.

    A row written before a schema change -- or by hand -- may hold something
    that is not JSON. That is a reason to return an empty list for that field,
    not to fail the whole request for the entries around it.
    """
    entry = dict(row)
    for column in JSON_COLUMNS:
        if column not in entry:
            continue
        try:
            entry[column] = json.loads(entry.get(column) or "[]")
        except (TypeError, ValueError):
            log.warning("Audit row %s has unreadable %s", entry.get("id"), column)
            entry[column] = []
    for column in JSON_OBJECT_COLUMNS:
        if column not in entry:
            continue
        try:
            entry[column] = json.loads(entry.get(column) or "{}")
        except (TypeError, ValueError):
            log.warning("Audit row %s has unreadable %s", entry.get("id"), column)
            entry[column] = {}
    for column in ("grounded", "cached", "byok", "escalated"):
        if column in entry:
            entry[column] = bool(entry.get(column))
    return entry
