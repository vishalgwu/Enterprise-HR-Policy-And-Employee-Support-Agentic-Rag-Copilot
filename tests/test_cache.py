"""The semantic answer cache, including the case it must refuse.

The test that matters most here is
`test_a_close_but_different_question_misses_at_the_default_threshold`: the
PTO/contractor pair is the reason the threshold is 0.95 rather than something
that would hit more often. Everything else in this file is bookkeeping around
that one risk.

All of it runs offline. `CountingEmbedding` hands back a fixed vector per
question and counts its calls, which is what lets the exact layer be tested for
*not* spending an embedding -- the property that makes the cache cheap on its
commonest hit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from conftest import settings_for

from app.core.tracing import Tracer
from app.services.cache import (
    CACHEABLE_SOURCES,
    SemanticCache,
    cosine_similarity,
    normalise_question,
)

# Vectors chosen so the cosines land either side of the 0.95 gate, and so the
# near-miss pair sits at 0.91 -- the figure `cache_similarity` was measured
# against in `app/core/config.py`.
PTO_MINE = "How much PTO do I get?"
PTO_PARAPHRASE = "How many paid time off days do I receive?"  # 0.98, a hit
PTO_LOOSE = "Tell me about the leave policy"  # 0.86, a miss
PTO_CONTRACTORS = "How much PTO do contractors get?"  # 0.91, a miss on purpose
UNRELATED = "What is the capital of France?"

VECTORS: dict[str, list[float]] = {
    PTO_MINE: [1.0, 0.0],
    PTO_PARAPHRASE: [1.0, 0.2],
    PTO_LOOSE: [1.0, 0.6],
    PTO_CONTRACTORS: [1.0, 0.456],
    UNRELATED: [0.0, 1.0],
}


class CountingEmbedding:
    """An embedding client that returns scripted vectors and counts calls."""

    def __init__(self, vectors: dict[str, list[float]] | None = None,
                 fail: bool = False) -> None:
        self.vectors = dict(vectors if vectors is not None else VECTORS)
        self.fail = fail
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("simulated embedding outage")
        return self.vectors.get(text, [0.0, 0.0, 1.0])


def answer(source_used: str = "private_kb", **extra: Any) -> dict[str, Any]:
    """An `AnswerResult`-shaped payload, as `/api/chat` would cache one."""
    payload: dict[str, Any] = {
        "question": PTO_MINE,
        "answer": "Full-time employees accrue 20 days of paid time off a year.",
        "source_used": source_used,
        "sources": [
            {"source": "leave-and-time-off.md", "title": "Leave And Time Off"}
        ],
        "trace": ["Router: kb", "KB Retriever: 3 chunk(s)"],
    }
    payload.update(extra)
    return payload


def cache_for(**kwargs: Any) -> SemanticCache:
    kwargs.setdefault("embedding", CountingEmbedding())
    return SemanticCache(**kwargs)


# --- Normalising and cosine --------------------------------------------------


def test_normalising_folds_case_whitespace_and_trailing_punctuation():
    assert normalise_question("  How  much   PTO?? ") == "how much pto"


def test_a_zero_vector_has_no_similarity_rather_than_a_division_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


# --- The exact layer ---------------------------------------------------------


def test_the_same_question_twice_is_an_exact_hit_that_costs_no_embedding():
    """The commonest hit -- a reload, a retry -- must not pay for a vector.

    Paying an embedding round trip to discover you are holding the identical
    string is the cache charging rent.
    """
    embedding = CountingEmbedding()
    cache = SemanticCache(embedding)
    cache.put(PTO_MINE, answer())
    embedding.calls.clear()

    hit = cache.get(PTO_MINE)

    assert hit is not None
    assert hit.kind == "exact"
    assert hit.similarity == 1.0
    assert embedding.calls == []


def test_punctuation_and_case_do_not_defeat_the_exact_layer():
    cache = cache_for()
    cache.put(PTO_MINE, answer())
    hit = cache.get("how much pto do i get")
    assert hit is not None
    assert hit.kind == "exact"


# --- The semantic layer ------------------------------------------------------


def test_a_paraphrase_above_the_threshold_is_a_semantic_hit():
    cache = cache_for()
    cache.put(PTO_MINE, answer())

    hit = cache.get(PTO_PARAPHRASE)

    assert hit is not None
    assert hit.kind == "semantic"
    assert hit.similarity > 0.95
    assert hit.entry.question == PTO_MINE


def test_a_paraphrase_below_the_threshold_is_a_miss():
    cache = cache_for()
    cache.put(PTO_MINE, answer())
    assert cache.get(PTO_LOOSE) is None


def test_a_close_but_different_question_misses_at_the_default_threshold():
    """The whole reason the threshold is 0.95.

    "How much PTO do I get?" and "How much PTO do contractors get?" differ by
    one word and by the entire answer -- the KB has a separate policy for
    contractors. Their embeddings sit at 0.91, so a cache tuned to hit more
    often would serve the employee answer to the contractor question, with the
    employee policy cited underneath it.
    """
    cache = cache_for()
    cache.put(PTO_MINE, answer())

    assert cache.get(PTO_CONTRACTORS) is None
    assert cache.stats()["misses"] == 1


def test_an_unrelated_question_does_not_find_anything():
    cache = cache_for()
    cache.put(PTO_MINE, answer())
    assert cache.get(UNRELATED) is None


# --- TTL and eviction --------------------------------------------------------


def test_an_expired_entry_is_a_miss_and_is_dropped():
    cache = SemanticCache(embedding=None, ttl_seconds=900)
    entry = cache.put(PTO_MINE, answer())
    assert entry is not None

    entry.created_at -= cache.ttl_seconds + 1

    assert cache.get(PTO_MINE) is None
    assert len(cache) == 0
    assert cache.stats()["evictions"] == 1


def test_a_full_cache_evicts_the_least_recently_hit_entry():
    cache = SemanticCache(embedding=None, max_entries=2)
    cache.put("first question", answer())
    cache.put("second question", answer())

    # Touching the first makes the second the least recently hit, so it is the
    # one the third put displaces.
    assert cache.get("first question") is not None
    cache.put("third question", answer())

    assert len(cache) == 2
    assert cache.get("second question") is None
    assert cache.get("first question") is not None
    assert cache.get("third question") is not None
    assert cache.stats()["evictions"] == 1


# --- Degraded modes ----------------------------------------------------------


def test_a_cache_with_no_embedding_still_serves_exact_hits():
    """Not a crippled mode: it is the mode a keyless deployment runs in."""
    cache = SemanticCache(embedding=None)
    cache.put(PTO_MINE, answer())

    assert cache.stats()["mode"] == "exact"
    assert cache.get(PTO_MINE) is not None
    assert cache.get(PTO_PARAPHRASE) is None


def test_a_cache_with_an_embedding_reports_semantic_mode():
    assert cache_for().stats()["mode"] == "semantic"


def test_an_embedding_that_raises_degrades_to_exact_matching():
    embedding = CountingEmbedding(fail=True)
    cache = SemanticCache(embedding)

    assert cache.put(PTO_MINE, answer()) is not None
    assert cache.get(PTO_MINE) is not None  # the exact layer never needed it
    assert cache.get(PTO_PARAPHRASE) is None  # and the semantic one cannot run

    stats = cache.stats()
    assert stats["embedding_failures"] == 2
    assert stats["exact_hits"] == 1


# --- What must not be cached -------------------------------------------------


@pytest.mark.parametrize("source", ["insufficient_evidence", "error"])
def test_an_unusable_answer_is_not_cached(source):
    """Caching a failure turns one outage into fifteen minutes of outage.

    And caching "I could not find that" means the ingest which fixes it changes
    nothing for a quarter of an hour, for the person most likely to ask again.
    """
    cache = cache_for()
    assert cache.put(PTO_MINE, answer(source_used=source)) is None
    assert len(cache) == 0
    assert cache.get(PTO_MINE) is None


def test_a_payload_with_no_source_at_all_is_declined():
    cache = cache_for()
    payload = answer()
    payload.pop("source_used")
    assert cache.put(PTO_MINE, payload) is None


def test_every_cacheable_source_is_actually_stored():
    for source in sorted(CACHEABLE_SOURCES):
        cache = cache_for()
        assert cache.put(PTO_MINE, answer(source_used=source)) is not None


# --- What comes back ---------------------------------------------------------


def test_the_cached_payload_comes_back_whole_including_its_citations():
    """Citations travel with the text, never recomputed beside it.

    Serving cached prose next to freshly computed citations would attach today's
    sources to yesterday's words, and the citation is the thing an employee
    checks the answer against.
    """
    cache = cache_for()
    stored = answer()
    cache.put(PTO_MINE, stored)

    hit = cache.get(PTO_MINE)

    assert hit is not None
    assert hit.entry.payload == stored
    assert hit.entry.payload["sources"][0]["source"] == "leave-and-time-off.md"
    assert hit.entry.payload["trace"] == stored["trace"]


def test_a_disabled_cache_never_stores_and_never_hits():
    cache = SemanticCache(CountingEmbedding(), enabled=False)
    assert cache.put(PTO_MINE, answer()) is None
    assert cache.get(PTO_MINE) is None
    assert len(cache) == 0
    assert cache.stats()["mode"] == "disabled"


def test_settings_supply_the_knobs_when_they_are_given():
    settings = settings_for(
        CACHE_SIMILARITY=0.8, CACHE_TTL_SECONDS=60, CACHE_MAX_ENTRIES=4
    )
    cache = SemanticCache(CountingEmbedding(), settings=settings)

    assert cache.threshold == 0.8
    assert cache.ttl_seconds == 60
    assert cache.max_entries == 4


def test_invalidating_drops_everything_and_says_how_much():
    cache = SemanticCache(embedding=None)
    cache.put("one", answer())
    cache.put("two", answer())
    assert cache.invalidate() == 2
    assert len(cache) == 0
    assert cache.invalidate() == 0


# --- Tracing -----------------------------------------------------------------


def test_a_hit_and_a_miss_are_both_recorded_on_the_tracer():
    cache = cache_for()
    tracer = Tracer()
    cache.put(PTO_MINE, answer(), tracer=tracer)
    cache.get(PTO_MINE, tracer=tracer)
    cache.get(UNRELATED, tracer=tracer)

    names = [span["name"] for span in tracer.to_list()]
    assert names == ["cache.store", "cache.hit", "cache.miss"]
    assert all(span["kind"] == "cache" for span in tracer.to_list())


def test_a_refused_store_is_visible_in_the_trace():
    """Otherwise "why did that not cache?" is a question only the source answers."""
    cache = cache_for()
    tracer = Tracer()
    cache.put(PTO_MINE, answer(source_used="error"), tracer=tracer)

    span = tracer.to_list()[0]
    assert span["name"] == "cache.store"
    assert span["attributes"]["stored"] is False


# --- Concurrency -------------------------------------------------------------


def test_concurrent_puts_neither_raise_nor_overrun_the_ceiling():
    """FastAPI serves `def` handlers from a threadpool, so this happens."""
    cache = SemanticCache(embedding=None, max_entries=32)

    def store(index: int) -> None:
        cache.put(f"question number {index}", answer())
        cache.get(f"question number {index // 2}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store, range(400)))

    assert len(cache) <= 32
    stats = cache.stats()
    assert stats["entries"] == len(cache)
    assert stats["hits"] + stats["misses"] == 400
