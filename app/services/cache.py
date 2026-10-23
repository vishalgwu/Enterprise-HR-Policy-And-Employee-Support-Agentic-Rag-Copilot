"""A semantic answer cache, and the reasons it is deliberately timid.

Caching by *meaning* is the most dangerous cache in this product. Two questions
can be arbitrarily close in embedding space and have different correct answers:
"how much PTO do I get?" and "how much PTO do contractors get?" differ by one
word and by the entire answer, and the KB has a separate policy for each. A
cache that serves the first to the second is not a slow answer or a missing
answer -- it is a confidently wrong HR answer, with a citation attached, which
is the one failure this project spends most of its design avoiding.

So every knob is set against the cache rather than for it:

* **The threshold is high.** `settings.cache_similarity` is 0.95 raw cosine.
  Measured on the labelled set, paraphrase pairs sat at 0.94-0.99 and the
  closest *different* question pair at 0.91, so 0.95 buys the reloads and the
  retries and declines the near misses. A cache that hits more often is not a
  better cache here.
* **The TTL is short.** Fifteen minutes. Policy changes, and a stale HR answer
  is a wrong answer that looks exactly like a right one -- same prose, same
  citation, superseded content. Fifteen minutes covers the burst one
  announcement causes without outliving an edit to the policy it is about.
* **The whole answer object is stored, citations included.** Serving cached text
  beside freshly computed citations would attach today's sources to yesterday's
  words, which is worse than either being stale on its own: the citation is what
  an employee checks the answer against.

**Two layers, and the exact one comes first.** A normalised-text match catches
the common case -- the same question asked twice, a page reload, a retry -- and
costs nothing. Only a miss there is worth an embedding.

**It works with no embedding at all.** `embedding=None` degrades to exact
matching, which is the mode a deployment with no embedding provider runs in, and
it still catches reloads. `stats()["mode"]` says which mode it is in so a
deployment cannot quietly be in the other one.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_logger

log = get_logger("cache")

__all__ = [
    "CACHEABLE_SOURCES",
    "CacheEntry",
    "CacheHit",
    "SemanticCache",
    "cosine_similarity",
    "normalise_question",
]

# Which answers are worth keeping, as an allowlist of `AgentState.source_used`
# (see `app.agent.schemas.SourceUsed`). The two that are missing are the point:
#
#   error                 caching a failure for fifteen minutes turns one
#                         outage into fifteen minutes of outage, and the
#                         retry that would have worked never reaches a
#                         provider.
#   insufficient_evidence caching "I could not find that" means the ingest
#                         that fixes it changes nothing for a quarter of an
#                         hour, to the one employee most likely to ask again.
#
# An allowlist rather than a denylist so a payload with no `source_used` at all
# -- a partial state, a hand-built dict -- is declined rather than stored on the
# assumption that it was fine.
CACHEABLE_SOURCES = {"private_kb", "web_search", "direct"}

_WHITESPACE = re.compile(r"\s+")
# Trailing punctuation only. "PTO?" and "PTO" are the same question; "don't"
# and "dont" are not something to guess at, so nothing internal is touched.
_TRAILING = "?!.,;:"


def normalise_question(question: str) -> str:
    """The exact-match key: casefolded, whitespace collapsed, punctuation off."""
    collapsed = _WHITESPACE.sub(" ", question.strip())
    return collapsed.casefold().rstrip(_TRAILING).strip()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Raw cosine, in pure Python.

    No numpy: the only vectors this ever sees are a handful of 384- or 1536-d
    questions per request, and a C dependency for that is a dependency to
    install, pin and ship in an image for no measurable gain.

    A zero vector has no direction, so there is no angle to report -- that is 0
    rather than a division by zero, and it is reachable: an embedding provider
    returning a zero vector for an empty or unparseable question is a thing that
    happens, and it must read as "no similarity", never as a match.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_sq += a * a
        right_sq += b * b
    if left_sq <= 0.0 or right_sq <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_sq) * math.sqrt(right_sq))


@dataclass(slots=True)
class CacheEntry:
    """One cached answer, and the question that produced it.

    `question` is kept verbatim rather than only as a key: a semantic hit means
    the employee asked something *different* from what was answered, and both
    the console and anyone debugging a wrong cached answer need to see which
    question's answer they were actually served.
    """

    key: str
    question: str
    payload: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: float = 0.0
    hits: int = 0


@dataclass(slots=True)
class CacheHit:
    """A hit, and enough about it to report and to distrust.

    `similarity` and `kind` are returned rather than kept internal because they
    belong on the answer: `/api/chat` reports `cache_hit` and
    `cache_similarity`, and "this came from a 0.96 match on a different
    question" is materially different information to hand a reader than "this
    was cached".
    """

    entry: CacheEntry
    similarity: float
    kind: str  # "exact" | "semantic"


class SemanticCache:
    """An LRU of answers, keyed by text and then by meaning.

    Dependencies are injected for the same reason they are everywhere else
    here: `embedding` is any object with `.embed_query(str) -> list[float]`, so
    a test uses a counting fake and never loads a 90 MB model.

    `settings`, when given, supplies all four knobs. The keyword defaults are
    for a caller that has no `Settings` -- a script, a test -- and match the
    field defaults in `app.core.config`.
    """

    def __init__(
        self,
        embedding: Any = None,
        *,
        threshold: float = 0.95,
        ttl_seconds: int = 900,
        max_entries: int = 512,
        enabled: bool = True,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            threshold = settings.cache_similarity
            ttl_seconds = settings.cache_ttl_seconds
            max_entries = settings.cache_max_entries
            enabled = settings.cache_enabled
        self.embedding = embedding
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max(1, max_entries)
        self.enabled = enabled

        # Ordered least-recently-hit first, which is what makes eviction a
        # `popitem(last=False)` rather than a scan for the oldest timestamp.
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        # FastAPI serves `def` handlers from a threadpool, so two employees
        # genuinely do reach this at the same moment. Without the lock the
        # OrderedDict is mutated from both and the eviction count is fiction.
        self._lock = threading.Lock()

        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._evictions = 0
        self._embedding_failures = 0

    # --- Reading -------------------------------------------------------------
    def get(self, question: str, tracer: Any = None) -> CacheHit | None:
        """An answer for this question, or None.

        Exact first, and only then semantic. Paying an embedding round trip to
        discover that you are holding the identical string is the cache charging
        rent -- it is the commonest hit and the cheapest one to find.
        """
        if not self.enabled:
            return None

        exact = self._take_exact(normalise_question(question))
        if exact is not None:
            return self._note_hit(
                CacheHit(entry=exact, similarity=1.0, kind="exact"), tracer
            )

        if self.embedding is None:
            return self._note_miss(tracer, reason="no embedding provider")

        vector = self._embed(question)
        if vector is None:
            # Degraded, not failed: the exact layer has already had its go, so
            # this is a miss and the question gets answered properly.
            return self._note_miss(tracer, reason="embedding unavailable")

        nearest, score = self._take_nearest(vector)
        if nearest is None:
            return self._note_miss(tracer, reason=f"nearest {score:.3f}")
        return self._note_hit(
            CacheHit(entry=nearest, similarity=score, kind="semantic"), tracer
        )

    # --- Writing -------------------------------------------------------------
    def put(
        self, question: str, payload: dict[str, Any], tracer: Any = None
    ) -> CacheEntry | None:
        """Store an answer, unless it is one that must not be stored.

        Returns the entry, or None when the payload was declined -- see
        `CACHEABLE_SOURCES`. A declined put is recorded in the trace rather than
        dropped silently, because "why did that not cache?" is otherwise a
        question only the source answers.
        """
        if not self.enabled:
            return None

        source = payload.get("source_used")
        if source not in CACHEABLE_SOURCES:
            _event(tracer, "cache.store", stored=False, source_used=str(source))
            return None

        vector = self._embed(question) if self.embedding is not None else None
        key = normalise_question(question)
        entry = CacheEntry(
            key=key,
            question=question,
            payload=payload,
            embedding=vector,
            created_at=time.monotonic(),
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
        _event(tracer, "cache.store", stored=True, source_used=str(source))
        return entry

    def invalidate(self) -> int:
        """Drop everything, returning how many entries went.

        Whole-cache only, and on purpose: an ingest or an upload can change the
        answer to a question that mentions none of the words in the file, so
        there is no honest way to invalidate selectively. Fifteen minutes of TTL
        plus "clear it when the corpus changes" is the pair that holds.
        """
        with self._lock:
            dropped = len(self._entries)
            self._entries.clear()
        return dropped

    # --- Reporting -----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Counters for `/health`, the console and a decision about the knobs.

        `mode` is here because the two modes have very different hit rates and a
        deployment that silently lost its embedding provider would otherwise
        look like one whose cache simply stopped helping.
        """
        with self._lock:
            entries = len(self._entries)
            hits, misses = self._hits, self._misses
            lookups = hits + misses
            return {
                "enabled": self.enabled,
                "mode": self._mode(),
                "entries": entries,
                "max_entries": self.max_entries,
                "hits": hits,
                "misses": misses,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
                "evictions": self._evictions,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl_seconds,
                "embedding_failures": self._embedding_failures,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # --- Internals -----------------------------------------------------------
    def _mode(self) -> str:
        if not self.enabled:
            return "disabled"
        return "semantic" if self.embedding is not None else "exact"

    def _take_exact(self, key: str) -> CacheEntry | None:
        """The entry for this exact key, counted as a hit if there is one."""
        with self._lock:
            self._drop_expired()
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            self._exact_hits += 1
            return entry

    def _take_nearest(self, vector: list[float]) -> tuple[CacheEntry | None, float]:
        """The closest entry above the threshold, and the best score either way.

        The score is returned even on a miss: "nearest was 0.91" is the number
        that tells you whether the threshold is doing its job or merely sitting
        above everything.
        """
        with self._lock:
            best: CacheEntry | None = None
            best_score = 0.0
            for candidate in self._entries.values():
                if candidate.embedding is None:
                    continue
                score = cosine_similarity(vector, candidate.embedding)
                if score > best_score:
                    best, best_score = candidate, score
            if best is None or best_score < self.threshold:
                return None, best_score
            self._entries.move_to_end(best.key)
            best.hits += 1
            self._hits += 1
            self._semantic_hits += 1
            return best, best_score

    def _drop_expired(self) -> None:
        """Lazily, on read. Called with the lock held.

        No background thread: a sweeper would need a lifecycle, and the only
        entry whose staleness matters is one somebody is about to be served.
        """
        if self.ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self.ttl_seconds
        stale = [k for k, e in self._entries.items() if e.created_at < cutoff]
        for key in stale:
            del self._entries[key]
            self._evictions += 1

    def _embed(self, question: str) -> list[float] | None:
        """Embed a question, or None if the provider would not.

        An embedding provider is a network call or a model on disk, and neither
        is allowed to cost an answer: the cache is an optimisation, so its
        failure mode is "no cache", never "no reply". Counted in `stats()` so a
        provider that is failing every call is visible as a number rather than
        as a hit rate that quietly went to zero.
        """
        try:
            vector = self.embedding.embed_query(question)
        except Exception:  # noqa: BLE001 - absorbed on purpose, see docstring
            with self._lock:
                self._embedding_failures += 1
            log.warning(
                "Cache embedding failed; falling back to exact matching",
                exc_info=True,
            )
            return None
        return [float(x) for x in vector] if vector else None

    def _note_hit(self, hit: CacheHit, tracer: Any) -> CacheHit:
        _event(
            tracer,
            "cache.hit",
            similarity=round(hit.similarity, 4),
            kind_of_hit=hit.kind,
        )
        return hit

    def _note_miss(self, tracer: Any, reason: str) -> None:
        with self._lock:
            self._misses += 1
        _event(tracer, "cache.miss", reason=reason)
        return None


def _event(tracer: Any, name: str, **attributes: Any) -> None:
    """Record on a tracer if there is one. Duck-typed, so a fake works."""
    if tracer is None:
        return
    tracer.event(name, kind="cache", **attributes)
