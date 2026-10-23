"""Hybrid retrieval: two rankings, fused by rank, then re-ordered.

Dense and sparse retrieval fail differently, and that -- not a general belief
that more is better -- is the argument for running both. An embedding finds the
chunk that paraphrases the question and misses the one that spells out the form
number; BM25 does the opposite, because it has no notion of paraphrase at all.
Where they agree, the agreement is the strongest signal available here.

**Fusion is by rank, not by score.** A cosine similarity of 0.34 and a BM25
score of 7.8 are not the same kind of number, and no amount of normalisation
makes them comparable *across* queries: min-max over one result set rescales by
whatever the best hit happened to score, so the same chunk receives a different
fused score depending on what else came back with it. Reciprocal rank fusion
throws the magnitudes away and keeps the one thing both lists mean the same way
-- the order. It is also why there is nothing here to re-tune when the
embedding model changes.

This module also owns `build_retriever`, the one retriever factory the rest of
the application calls. Everything about which retrieval a deployment can
actually serve is decided there and nowhere else.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from app.core.config import Settings, get_logger, get_settings
from app.core.tracing import KIND_RETRIEVAL, null_tracer
from app.rag.lexical import BM25Index, LexicalRetriever, build_kb_index
from app.rag.loaders import chunk_id
from app.rag.rerank import Ranked, documents_of, rerank
from app.rag.retrieval import get_kb_retriever

log = get_logger("hybrid")

__all__ = [
    "HybridRetriever",
    "build_retriever",
    "document_key",
    "reciprocal_rank_fusion",
]


def document_key(document: Document) -> str:
    """Identity for de-duplicating one chunk across two rankings.

    `chunk_id` is the right answer whenever the metadata is there, because it is
    the id the vector store itself keys on -- so the dense and the sparse side
    agree about what "the same chunk" is without comparing any text.

    It needs `start_index`, though, and not every Document has one. A
    Tavily-derived document, a hand-built one in a test, and anything that did
    not come out of `split_documents` carry no offset, and `chunk_id` would then
    hash every chunk of one file to the same value -- quietly fusing a whole
    document into one entry and losing the rest. The fallback hashes the source
    plus the first 200 characters instead: enough to tell two chunks of one file
    apart, and stable between the two retrievers because both are handing back
    the same stored text.
    """
    metadata = document.metadata or {}
    if metadata.get("source") and metadata.get("start_index") is not None:
        return chunk_id(document)
    basis = f"{metadata.get('source', '')}:{document.page_content[:200]}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[tuple[Document, float]]:
    """Fuse ranked lists into one, by position.

    Each list contributes `weight / (k + rank)` for every document in it, with
    `rank` counted from 1. `k` damps the top of each list: at k=60 the gap
    between rank 1 and rank 2 is small, so a chunk that *both* retrievers
    returned outranks one that only a single retriever returned, even when the
    loner was first. That is the property the whole thing exists for, and it
    falls out of the sum rather than being special-cased -- there is no "both
    agreed" bonus to get wrong.

    `weights` is per ranking, in the order given, defaulting to 1.0 each.
    `HybridRetriever` passes `[dense_weight, 1 - dense_weight]`, so the dense
    list's weight is the setting and the sparse list gets the remainder.

    The first `Document` object seen for a key is the one returned. The two
    sides can carry subtly different metadata for the same chunk -- Pinecone
    returns what was upserted, the lexical index returns what is on disk right
    now -- and picking one deterministically beats merging two dicts whose
    disagreement nobody has looked at.
    """
    chosen = list(weights) if weights is not None else []
    chosen += [1.0] * (len(rankings) - len(chosen))
    chosen = chosen[: len(rankings)]

    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    arrival: dict[str, int] = {}

    for ranking, weight in zip(rankings, chosen, strict=True):
        for rank, document in enumerate(ranking, 1):
            key = document_key(document)
            if key not in first_seen:
                first_seen[key] = document
                arrival[key] = len(arrival)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    order = sorted(scores, key=lambda key: (-scores[key], arrival[key]))
    fused = [(first_seen[key], scores[key]) for key in order]
    return fused[:limit] if limit else fused


class HybridRetriever:
    """Dense and sparse retrieval, fused and re-ranked, behind `invoke`.

    Degrades at every step rather than raising. One side failing is a worse
    ranking; both sides failing is zero chunks, and zero chunks is a state the
    graph already knows what to do with -- an empty retrieval is what sends a
    question to the web fallback. An exception escaping here would instead abort
    the whole run, which is the one outcome that costs an employee the answer
    the other source could have given.
    """

    def __init__(
        self,
        dense: Any = None,
        sparse: Any = None,
        *,
        k: int = 4,
        candidates: int = 12,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
        rerank_strategy: str = "lexical",
        llm: Any = None,
        tracer: Any = None,
        budget: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.k = k
        # Never fewer candidates than the answer needs: a re-ranker can only
        # reorder what it is handed, so a candidate list cut to k leaves it
        # nothing to do.
        self.candidates = max(candidates, k)
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.rerank_strategy = rerank_strategy
        self.llm = llm
        self.tracer = tracer or null_tracer()
        self.budget = budget
        self.settings = settings

    @property
    def mode(self) -> str:
        """What this instance can actually do, as opposed to what it is called."""
        if self.dense is not None and self.sparse is not None:
            return "hybrid"
        if self.dense is not None:
            return "dense"
        if self.sparse is not None:
            return "lexical"
        return "empty"

    def _side(self, name: str, retriever: Any, query: str) -> list[Document]:
        if retriever is None:
            return []
        try:
            return list(retriever.invoke(query) or [])
        except Exception as exc:  # noqa: BLE001 - one side failing is not a failed search
            log.exception("[Hybrid] the %s retriever failed; continuing without it", name)
            # The recorded reason is the exception's *type* and nothing else. A
            # provider error carries URLs, request ids and sometimes the
            # offending payload, and this span is serialised onto the HTTP
            # response and written to the audit row.
            self.tracer.degraded(name, reason=type(exc).__name__, kind=KIND_RETRIEVAL)
            return []

    def search(self, query: str) -> list[Ranked]:
        """The same work as `invoke`, with the scores and reasons kept."""
        with self.tracer.span(
            "retrieve.hybrid", kind=KIND_RETRIEVAL, mode=self.mode, k=self.k
        ) as span:
            dense = self._side("retrieve.dense", self.dense, query)
            sparse = self._side("retrieve.lexical", self.sparse, query)
            span.set(dense=len(dense), lexical=len(sparse))
            if not dense and not sparse:
                return []

            fused = reciprocal_rank_fusion(
                [dense, sparse],
                k=self.rrf_k,
                weights=[self.dense_weight, 1 - self.dense_weight],
                limit=self.candidates,
            )
            ranked = rerank(
                query,
                [document for document, _ in fused],
                k=self.k,
                strategy=self.rerank_strategy,
                llm=self.llm,
                tracer=self.tracer,
                budget=self.budget,
                settings=self.settings,
            )
            span.set(candidates=len(fused), chunks=len(ranked))
            return ranked

    def invoke(self, query: str, *args: object, **kwargs: object) -> list[Document]:
        """The retriever contract. Returns [] rather than raising, always."""
        try:
            return documents_of(self.search(query))
        except Exception:  # noqa: BLE001 - see the class docstring
            log.exception("[Hybrid] the fused search failed; answering with no chunks")
            return []


# --- Choosing a retriever for this deployment --------------------------------


def _dense_retriever(settings: Settings, embedding: Any, k: int) -> Any | None:
    try:
        return get_kb_retriever(k=k, embedding=embedding, settings=settings)
    except Exception:  # noqa: BLE001 - a missing key is a reduced shape, not a fault
        log.exception("[Retrieval] the dense retriever could not be built")
        return None


def _lexical_retriever(settings: Settings, k: int) -> LexicalRetriever | None:
    try:
        return LexicalRetriever(build_kb_index(settings=settings), k=k)
    except Exception:  # noqa: BLE001
        log.exception("[Retrieval] the BM25 index could not be built")
        return None


def build_retriever(
    settings: Settings | None = None,
    embedding: Any = None,
    dense: Any = None,
    sparse: Any = None,
    llm: Any = None,
    tracer: Any = None,
    budget: Any = None,
    k: int | None = None,
) -> Any:
    """The one entry point the rest of the application calls for a retriever.

    Reads `settings.effective_retrieval_mode`, which has already resolved
    `auto` against what this deployment has keys for, and returns a retriever
    whose `invoke(query)` gives Documents:

        dense     `get_kb_retriever(...)`, exactly the behaviour this project
                  shipped with.
        lexical   BM25 over the knowledge base on disk. No keys, no network.
        hybrid    both, fused and re-ranked.

    **It degrades instead of propagating.** A deployment that loses Pinecone --
    an expired key, an index deleted, a region outage -- should give worse
    answers, not no answers, and BM25 over the same eleven documents needs
    nothing from the network. The alternative is an application that boots,
    reports healthy, and fails on the first question.
    """
    settings = settings or get_settings()
    k = k or settings.default_k
    mode = settings.effective_retrieval_mode

    if mode == "lexical":
        if sparse is not None:
            return sparse
        # An unreadable knowledge base becomes an empty index rather than an
        # exception: zero chunks routes to the web fallback, which is a worse
        # answer and still an answer.
        return _lexical_retriever(settings, k) or LexicalRetriever(
            BM25Index.from_documents([]), k=k
        )

    built_dense = dense if dense is not None else _dense_retriever(settings, embedding, k)
    if built_dense is None:
        log.warning("[Retrieval] no dense retriever; falling back to lexical retrieval")
        return _lexical_retriever(settings, k) or LexicalRetriever(
            BM25Index.from_documents([]), k=k
        )

    if mode == "dense":
        return built_dense

    built_sparse = sparse if sparse is not None else _lexical_retriever(settings, k)
    if built_sparse is None:
        log.warning("[Retrieval] no BM25 index; falling back to dense retrieval alone")
        return built_dense

    return HybridRetriever(
        dense=built_dense,
        sparse=built_sparse,
        k=k,
        candidates=settings.rerank_candidates,
        dense_weight=settings.hybrid_dense_weight,
        rrf_k=settings.rrf_k,
        rerank_strategy=settings.rerank_strategy,
        llm=llm,
        tracer=tracer,
        budget=budget,
        settings=settings,
    )
