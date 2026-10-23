"""The shared working surface a supervisor and its workers reason over.

A single-agent graph keeps its working state in `AgentState` and that is
enough, because one node's output is the next node's input: the wiring itself
carries the information forward. A supervisor and four workers are a different
shape. A worker needs to see what another worker already established without
the supervisor having to copy it into a prompt, and a human reading the trace
afterwards needs that same view -- otherwise "why did the payroll specialist
say that?" is answerable only by re-running the question.

The scratchpad is that surface: append-only, scoped to a conversation, readable
by every participant. It is also the one place in this system where the
*reasoning* is durable rather than reconstructed. The span trace says what ran
and the audit log says what was decided; neither holds the sentence a worker
wrote for the next worker to read. That is why the UI shows it.

It lives in the `scratchpad` table in `app.services.state_db`, which defines
the schema; nothing here redefines it. Every writer keeps that module's
contract -- a failure degrades the feature and is logged, it never reaches the
request.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_logger, get_settings
from app.services.state_db import StateStore, now_iso

log = get_logger("scratchpad")

__all__ = ["Note", "NoteKind", "Scratchpad"]

# What a note is for. `kind` is what lets a prompt include only the notes it
# needs (a worker wants findings, not the plan) and the UI style them
# differently, which is the whole reason it is a column rather than a prefix
# inside `text`.
#
#   plan      the supervisor's intent for this question
#   finding   what one worker established, and from which documents
#   question  something a worker needs answered before it can report
#   decision  a choice made, with its reason
#   handoff   who was handed what, and why
#   note      anything else an agent wrote
#   user      a line the employee contributed, kept beside the agents' own
NoteKind = Literal[
    "plan", "finding", "question", "decision", "handoff", "note", "user"
]

_KINDS: frozenset[str] = frozenset(
    ("plan", "finding", "question", "decision", "handoff", "note", "user")
)

# A conversation id arrives over HTTP from a browser -- it is the client's own
# `crypto.randomUUID()` -- so it is untrusted text that ends up in a SQL
# parameter, in a log line and in a UI label. Matched rather than escaped, and
# **rejected rather than sanitised**: sanitising `a/b` and `a:b` to `ab` merges
# two conversations into one, which means one employee's notes rendered into
# another employee's prompt. A rejected id costs this conversation its
# scratchpad; a silently merged one costs the wrong person their privacy.
CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# How a rendered note reads in a prompt. Short, because it is repeated once per
# note on every model call that includes the pad.
_LINE = "- [{kind}] {author}: {text}"


@dataclass(slots=True)
class Note:
    """One line somebody wrote on the pad."""

    id: int
    created_at: str
    conversation_id: str
    author: str
    kind: NoteKind
    text: str
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "conversation_id": self.conversation_id,
            "author": self.author,
            "kind": self.kind,
            "text": self.text,
            "pinned": self.pinned,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Note:
        return cls(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            conversation_id=str(row["conversation_id"]),
            author=str(row["author"]),
            kind=str(row["kind"]),  # type: ignore[arg-type]
            text=str(row["text"]),
            pinned=bool(row["pinned"]),
        )

    def render(self) -> str:
        return _LINE.format(kind=self.kind, author=self.author, text=self.text)


class Scratchpad:
    """Append-only working notes, one pad per conversation.

    `store` is injected so a test points it at `tmp_path` and so the
    application can share one `StateStore` across the four features that use
    the file. Passing neither falls back to the configured path, which is what
    a CLI wants.
    """

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        db_path: Path | str | None = None,
        max_notes: int = 200,
        max_chars: int = 2000,
    ) -> None:
        if store is None:
            store = StateStore(db_path or get_settings().state_db_path)
        self.store = store
        self.max_notes = max_notes
        self.max_chars = max_chars

    # --- Writing -------------------------------------------------------------
    def append(
        self,
        conversation_id: str,
        text: str,
        *,
        author: str = "agent",
        kind: NoteKind = "note",
        pinned: bool = False,
    ) -> Note | None:
        """Add one note. Returns it, or None when nothing was written.

        A long note is **truncated, not refused**: a worker that wrote three
        paragraphs of finding should lose the tail, not the finding. Refusing
        would drop the one sentence the next worker needed, and the writer has
        no way to know it happened.
        """
        conversation = self._valid_conversation(conversation_id)
        if conversation is None:
            return None
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > self.max_chars:
            text = text[: self.max_chars - 1].rstrip() + "…"
        if kind not in _KINDS:
            # A kind nobody styles is a cosmetic problem; dropping the note is
            # not. The caller is often a model-driven path, so this is its
            # normal failure rather than a bug.
            log.debug("Unknown note kind %r; recording it as a plain note", kind)
            kind = "note"

        # A conversation that has written 200 notes is a loop, not a
        # conversation, and the 201st note is almost certainly the loop's.
        # Bounding writes rather than pruning old ones keeps the table what it
        # says it is: a record.
        if self._count(conversation) >= self.max_notes:
            log.warning(
                "Scratchpad for %s is full (%d notes); dropping the new note",
                conversation,
                self.max_notes,
            )
            return None

        created_at = now_iso()
        note_id = self.store.execute(
            "INSERT INTO scratchpad "
            "(created_at, conversation_id, author, kind, text, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, conversation, author, kind, text, int(bool(pinned))),
        )
        if note_id is None:
            return None
        return Note(
            id=note_id,
            created_at=created_at,
            conversation_id=conversation,
            author=author,
            kind=kind,
            text=text,
            pinned=bool(pinned),
        )

    def pin(self, note_id: int, pinned: bool = True) -> bool:
        """Mark a note as always-included. True when a row actually changed."""
        changed = self._write(
            "UPDATE scratchpad SET pinned = ? WHERE id = ?",
            (int(bool(pinned)), int(note_id)),
        )
        return changed > 0

    def clear(self, conversation_id: str) -> int:
        """Delete one conversation's notes. Returns how many went.

        Scoped to the conversation and nothing else: this is what an employee
        clearing their own thread calls, and it must not be reachable as "clear
        the pad".
        """
        conversation = self._valid_conversation(conversation_id)
        if conversation is None:
            return 0
        return self._write(
            "DELETE FROM scratchpad WHERE conversation_id = ?", (conversation,)
        )

    # --- Reading -------------------------------------------------------------
    def notes(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        kinds: list[str] | tuple[str, ...] | None = None,
    ) -> list[Note]:
        """The last `limit` notes for one conversation, oldest first.

        Selected newest-first so `limit` keeps the *recent* notes, then
        reversed: a prompt and a UI both read a transcript forwards.
        """
        conversation = self._valid_conversation(conversation_id)
        if conversation is None:
            return []
        sql = "SELECT * FROM scratchpad WHERE conversation_id = ?"
        params: list[Any] = [conversation]
        wanted = [k for k in (kinds or ()) if k in _KINDS]
        if wanted:
            sql += f" AND kind IN ({', '.join('?' * len(wanted))})"
            params += wanted
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.store.query(sql, tuple(params))
        return [Note.from_row(row) for row in reversed(rows)]

    def render(
        self,
        conversation_id: str,
        *,
        limit: int = 40,
        kinds: list[str] | tuple[str, ...] | None = None,
        max_chars: int = 4000,
    ) -> str:
        """The pad as prompt text, oldest first, within a character budget.

        **The budget is on the renderer, not on the table.** The table is the
        record: a note that was written stays written, because the audit view
        and the UI both read it and a record that drops its own middle is worse
        than no record. The prompt is the cost -- every note in it is paid for
        on every model call that includes the pad, and a pad that grows all
        conversation makes the last turn the most expensive one. So the ceiling
        lives where the money is spent.

        Built newest-first and reversed at the end, so the notes that survive a
        tight budget are the recent ones. Pinned notes are included before the
        budget applies at all: pinning is the one way a caller says "this must
        reach the model", and a budget that could drop it would make pinning a
        suggestion.
        """
        notes = self.notes(conversation_id, limit=limit, kinds=kinds)
        if not notes:
            return ""

        pinned = [n for n in notes if n.pinned]
        rest = [n for n in notes if not n.pinned]

        lines = [n.render() for n in pinned]
        spent = sum(len(line) + 1 for line in lines)
        kept: list[str] = []
        for note in reversed(rest):
            line = note.render()
            if spent + len(line) + 1 > max_chars and kept:
                break
            kept.append(line)
            spent += len(line) + 1
        lines += list(reversed(kept))
        return "\n".join(lines)

    def recent_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Conversations that have a pad, most recently written first."""
        rows = self.store.query(
            "SELECT conversation_id, COUNT(*) AS notes, "
            "MIN(created_at) AS first_at, MAX(created_at) AS last_at, "
            "MAX(id) AS last_id "
            "FROM scratchpad GROUP BY conversation_id "
            "ORDER BY last_id DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [
            {
                "conversation_id": str(row["conversation_id"]),
                "notes": int(row["notes"]),
                "first_at": str(row["first_at"]),
                "last_at": str(row["last_at"]),
            }
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        """Counts for `/health` and the console."""
        by_kind = {
            str(row["kind"]): int(row["n"])
            for row in self.store.query(
                "SELECT kind, COUNT(*) AS n FROM scratchpad GROUP BY kind"
            )
        }
        return {
            "notes": self.store.scalar("SELECT COUNT(*) FROM scratchpad", default=0),
            "conversations": self.store.scalar(
                "SELECT COUNT(DISTINCT conversation_id) FROM scratchpad", default=0
            ),
            "pinned": self.store.scalar(
                "SELECT COUNT(*) FROM scratchpad WHERE pinned = 1", default=0
            ),
            "by_kind": by_kind,
        }

    # --- Internals -----------------------------------------------------------
    def _valid_conversation(self, conversation_id: Any) -> str | None:
        candidate = str(conversation_id or "").strip()
        if not CONVERSATION_ID_PATTERN.match(candidate):
            log.debug("Rejected a scratchpad conversation id of %d character(s)",
                      len(candidate))
            return None
        return candidate

    def _count(self, conversation_id: str) -> int:
        return int(
            self.store.scalar(
                "SELECT COUNT(*) FROM scratchpad WHERE conversation_id = ?",
                (conversation_id,),
                default=0,
            )
        )

    def _write(self, sql: str, params: Any) -> int:
        """Run one UPDATE or DELETE and return the rows it touched.

        `StateStore.execute` reports a new rowid, which says nothing about an
        UPDATE -- SQLite leaves `lastrowid` at whatever the last INSERT set, so
        a `pin()` built on it would report success for a note id that does not
        exist. Same swallow-and-log contract as the store itself.
        """
        try:
            with self.store.connect() as connection:
                cursor = connection.execute(sql, params)
                connection.commit()
                return int(cursor.rowcount or 0)
        except Exception:  # noqa: BLE001 - a degraded pad beats a failed answer
            log.exception("Scratchpad write failed: %s", sql.split(" ", 2)[0])
            return 0
