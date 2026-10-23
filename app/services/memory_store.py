"""Long-term facts: the handful of things a returning employee should not have
to repeat.

Short-term memory is the conversation, and this product deliberately does not
hold it -- the browser owns the thread and posts it back with each turn, for
the reasons `CLAUDE.md` sets out at length (`/api/chat` is unauthenticated, so
there is no identity a server-side conversation could be scoped to).

Long-term memory is a different proposition, and a genuinely useful one here.
An employee who said once that they are a contractor in the Bangalore office
should not be asked again, because the answer to half the questions in this
knowledge base turns on exactly that: notice periods, PTO payout, pay dates,
whether core hours apply. Asking twice is not merely rude, it produces a
*hedged* answer where a precise one was available.

It is also the most privacy-sensitive thing in the product, so four
constraints, each of which is a refusal of something easier:

- **Keyed on a hashed subject, never a name.** The file holds pseudonyms. See
  `subject_key` for what that does and does not buy.
- **A closed set of keys.** `MEMORY_KEYS` and nothing else. Free text turns
  this into a profile store, and a profile store is a thing that needs a
  privacy policy.
- **A confidence on every fact**, because these are inferred from
  conversation, not entered on a form, and a prompt should be able to tell the
  difference.
- **`forget()` as a first-class operation**, not an afterthought bolted on
  when somebody asks for it.

Backed by the `facts` table in `app.services.state_db`, which defines the
schema; nothing here redefines it.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_logger, get_settings
from app.services.state_db import StateStore, now_iso

log = get_logger("memory")

__all__ = ["Fact", "FactMemory", "MEMORY_KEYS", "subject_key"]

# A closed set. Free-text keys turn this into a profile store, and a profile
# store is a thing that needs a privacy policy.
#
# Each one is here because some document in `data/private_kb/` answers
# differently depending on it: employment type (salaried against hourly and
# contract staff), department and location (local law on PTO payout), work mode
# (the three-day hybrid minimum), tenure band (notice periods), and the
# language the reply should be written in.
MEMORY_KEYS: frozenset[str] = frozenset(
    {
        "employment_type",
        "department",
        "location",
        "work_mode",
        "tenure_band",
        "preferred_language",
    }
)

# Below this, a fact is rendered with a hedge. These are inferred from what
# somebody said in passing, and "you told me you are a contractor" is a worse
# failure than asking.
CONFIDENT = 0.5

MAX_VALUE_CHARS = 120


def subject_key(subject: str) -> str:
    """A stable pseudonym for one person, used as the storage key.

    **This is pseudonymisation, not anonymisation, and the difference matters.**
    The digest keeps a name or an employee id out of the database file, out of
    `stats()`, and out of every log line this module writes -- which is what
    stops a working-state file becoming a staff directory. It does not defeat
    anyone holding both this file and a list of employees: the input space is
    small, so the same person always hashes to the same value and that value
    can be looked up by hashing every candidate. Treat the table as sensitive
    regardless; the hash is defence in depth, not a reason to relax anything
    else.

    Case- and whitespace-insensitive so "A.Patel" and "a.patel " are one
    person rather than two half-remembered ones.
    """
    normalised = str(subject or "").strip().lower()
    if not normalised:
        return ""
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class Fact:
    """One remembered thing. `subject` is the pseudonym, never the name."""

    subject: str
    key: str
    value: str
    source: str
    confidence: float
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Fact:
        return cls(
            subject=str(row["subject"]),
            key=str(row["key"]),
            value=str(row["value"]),
            source=str(row["source"]),
            confidence=float(row["confidence"]),
            updated_at=str(row["updated_at"]),
        )

    def render(self) -> str:
        label = self.key.replace("_", " ")
        if self.confidence < CONFIDENT:
            return f"{label}: {self.value} (unconfirmed)"
        return f"{label}: {self.value}"


class FactMemory:
    """Reads and writes the `facts` table, for one deployment.

    `store` is injected for the same reasons as `Scratchpad`: a test points it
    at `tmp_path`, and the application shares one `StateStore` across the
    features that live in the file.
    """

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        db_path: Path | str | None = None,
        max_per_subject: int = 20,
    ) -> None:
        if store is None:
            store = StateStore(db_path or get_settings().state_db_path)
        self.store = store
        self.max_per_subject = max_per_subject

    # --- Writing -------------------------------------------------------------
    def remember(
        self,
        subject: str,
        key: str,
        value: str,
        *,
        source: str = "conversation",
        confidence: float = 0.6,
    ) -> Fact | None:
        """Record one fact, replacing any earlier value for the same key.

        **Replaces rather than appends.** Two contradictory values for
        `employment_type` are worse than one possibly-stale value, because the
        prompt that renders both teaches the model to hedge every answer it
        gives afterwards -- and a hedged answer about notice periods is the
        thing this memory existed to avoid. One value can be wrong and then
        corrected; two values are permanently unusable. The *higher* confidence
        of the two survives, so a fact the employee stated outright is not
        downgraded by a later guess.

        An unknown key returns None and logs at debug rather than raising. The
        caller is a language model extracting facts from prose, and an
        unexpected key is its ordinary failure mode, not a bug in this module.
        """
        pseudonym = subject_key(subject)
        if not pseudonym:
            return None
        key = str(key or "").strip().lower()
        if key not in MEMORY_KEYS:
            log.debug("Ignoring a fact with the unknown key %r", key)
            return None
        value = str(value or "").strip()[:MAX_VALUE_CHARS]
        if not value:
            return None
        confidence = min(1.0, max(0.0, float(confidence)))

        # The closed key set already bounds a subject to six rows; this is the
        # backstop for the day somebody adds a seventh key and a loop that
        # writes a new one per turn. Only a *new* key can grow the row count --
        # a replacement cannot, which is the point of replacing.
        known = self.store.scalar(
            "SELECT COUNT(*) FROM facts WHERE subject = ? AND key = ?",
            (pseudonym, key),
            default=0,
        )
        if not known and self._count(pseudonym) >= self.max_per_subject:
            log.warning("Fact store for one subject is full; dropping %r", key)
            return None

        stamp = now_iso()
        # `_write` rather than `StateStore.execute`: an `ON CONFLICT DO UPDATE`
        # inserts no row, so `lastrowid` stays 0 and `execute` reports None --
        # a successful replacement would be indistinguishable from a failure,
        # and replacement is the common case here.
        wrote = self._write(
            "INSERT INTO facts "
            "(created_at, updated_at, subject, key, value, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(subject, key) DO UPDATE SET "
            "  value = excluded.value, "
            "  source = excluded.source, "
            "  updated_at = excluded.updated_at, "
            "  confidence = MAX(facts.confidence, excluded.confidence)",
            (stamp, stamp, pseudonym, key, value, source, confidence),
        )
        if not wrote:
            return None
        # Read back rather than returning what was asked for: the stored
        # confidence is the MAX above, which is not necessarily this call's.
        rows = self.store.query(
            "SELECT * FROM facts WHERE subject = ? AND key = ?", (pseudonym, key)
        )
        return Fact.from_row(rows[0]) if rows else None

    def forget(self, subject: str, key: str | None = None) -> int:
        """Delete one key, or everything known about a subject. Returns rows gone.

        A first-class operation, with its own endpoint and its own test, rather
        than a DELETE somebody writes by hand when it is first asked for.
        """
        pseudonym = subject_key(subject)
        if not pseudonym:
            return 0
        if key is None:
            return self._write("DELETE FROM facts WHERE subject = ?", (pseudonym,))
        return self._write(
            "DELETE FROM facts WHERE subject = ? AND key = ?",
            (pseudonym, str(key).strip().lower()),
        )

    # --- Reading -------------------------------------------------------------
    def recall(
        self, subject: str, keys: list[str] | tuple[str, ...] | None = None
    ) -> list[Fact]:
        """What is known about one person, by key. Empty for an unknown subject.

        `keys=None` means every key. A `keys` list that survives filtering to
        nothing -- because the caller named keys that are not in `MEMORY_KEYS`
        -- returns nothing rather than falling back to every key: asking for one
        unknown key and being handed the whole profile is the surprising
        direction for this to fail in.
        """
        pseudonym = subject_key(subject)
        if not pseudonym:
            return []
        sql = "SELECT * FROM facts WHERE subject = ?"
        params: list[Any] = [pseudonym]
        wanted = [str(k).strip().lower() for k in (keys or ()) if k]
        wanted = [k for k in wanted if k in MEMORY_KEYS]
        if keys is not None and not wanted:
            return []
        if wanted:
            sql += f" AND key IN ({', '.join('?' * len(wanted))})"
            params += wanted
        sql += " ORDER BY key"
        return [Fact.from_row(row) for row in self.store.query(sql, tuple(params))]

    def render(self, subject: str) -> str:
        """The facts as one prompt line, or "" when nothing is known.

        Empty rather than a placeholder for an unknown subject, so a prompt can
        concatenate it unconditionally instead of every caller branching on
        whether this person has been met before.
        """
        facts = self.recall(subject)
        if not facts:
            return ""
        return "Known about this person: " + "; ".join(f.render() for f in facts) + "."

    def stats(self) -> dict[str, Any]:
        """Counts only. No subjects and no values -- this is for `/health`."""
        by_key = {
            str(row["key"]): int(row["n"])
            for row in self.store.query(
                "SELECT key, COUNT(*) AS n FROM facts GROUP BY key"
            )
        }
        return {
            "facts": self.store.scalar("SELECT COUNT(*) FROM facts", default=0),
            "subjects": self.store.scalar(
                "SELECT COUNT(DISTINCT subject) FROM facts", default=0
            ),
            "keys": sorted(MEMORY_KEYS),
            "by_key": by_key,
        }

    # --- Internals -----------------------------------------------------------
    def _count(self, pseudonym: str) -> int:
        return int(
            self.store.scalar(
                "SELECT COUNT(*) FROM facts WHERE subject = ?", (pseudonym,), default=0
            )
        )

    def _write(self, sql: str, params: Any) -> int:
        """One write, reporting the rows it touched. See `Scratchpad._write`."""
        try:
            with self.store.connect() as connection:
                cursor = connection.execute(sql, params)
                connection.commit()
                return int(cursor.rowcount or 0)
        except Exception:  # noqa: BLE001 - a degraded memory beats a failed answer
            log.exception("Fact write failed: %s", sql.split(" ", 2)[0])
            return 0
