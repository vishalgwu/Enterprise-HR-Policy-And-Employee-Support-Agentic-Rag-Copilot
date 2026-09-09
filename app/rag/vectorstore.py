"""Pinecone index management and retrieval.

Every operation is parameterised by *both* index and namespace, so the same code
serves the private HR corpus, the scraped public article and a throwaway index
in a test. Nothing here hardcodes which corpus it is working on.

Namespace discipline is the thing to get right: a namespace typo fails silently,
because the upsert succeeds and every later retrieval simply returns [].
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.core.config import Settings, get_logger, get_settings
from app.rag.clients import get_embeddings
from app.rag.loaders import chunk_id

log = get_logger("vectorstore")

# Sentinel for `get_retriever(score_threshold=...)` meaning "use the configured
# gate". Not None, because None is a meaningful value there -- it disables the
# gate entirely -- so it cannot double as "unset". Defined here rather than in
# `retrieval.py` because this is the module that *interprets* it; a caller that
# spells the bare -1.0 is relying on a number, not on a contract.
USE_CONFIGURED_GATE = -1.0


@lru_cache(maxsize=1)
def get_pinecone_client():
    from pinecone import Pinecone

    return Pinecone(api_key=get_settings().require("pinecone_api_key"))


def _index_name(index_name: str | None, settings: Settings | None = None) -> str:
    return index_name or (settings or get_settings()).pinecone_index


def _attr(obj: Any, name: str) -> Any:
    """Pinecone v7 returns objects or dicts depending on the call path."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _index_names(pc: Any) -> set[str]:
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return set(indexes.names())
    return {ix["name"] if isinstance(ix, dict) else ix.name for ix in indexes}


def _index_ready(description: Any) -> bool:
    status = _attr(description, "status")
    if status is None:
        return False
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def _wait_until_index_ready(pc: Any, index_name: str, settings: Settings) -> None:
    deadline = time.monotonic() + settings.index_ready_timeout
    while time.monotonic() < deadline:
        if _index_ready(pc.describe_index(index_name)):
            return
        time.sleep(settings.index_ready_poll_interval)
    raise TimeoutError(
        f"Pinecone index {index_name!r} was not ready after "
        f"{settings.index_ready_timeout}s."
    )


def ensure_index(
    pc: Any = None,
    index_name: str | None = None,
    dimension: int | None = None,
    settings: Settings | None = None,
):
    """Create the serverless index if absent, then verify its dimension.

    The dimension check catches the trap where the embedding model changed but
    the index did not: Pinecone cannot resize an index, so the mismatch would
    otherwise surface much later as an opaque upsert error.
    """
    from pinecone import ServerlessSpec

    settings = settings or get_settings()
    pc = pc or get_pinecone_client()
    index_name = _index_name(index_name, settings)
    dimension = dimension or settings.embedding_dim

    # `embedding_dim` is 0 when EMBEDDING_DIM is unset and the active model is
    # not in KNOWN_EMBEDDING_DIMS. `validate_embedding()` says so clearly, but
    # only entry points call it -- an /ingest request reaches here directly, and
    # creating a 0-d index is not recoverable, because Pinecone cannot resize
    # one. Refuse at the last point that can still name the cause.
    if dimension <= 0:
        raise ValueError(
            f"Cannot create or verify index {index_name!r}: no embedding "
            f"dimension is known for {settings.active_embedding_model!r}. Set "
            f"EMBEDDING_DIM explicitly, or add the model to "
            f"KNOWN_EMBEDDING_DIMS."
        )

    if index_name not in _index_names(pc):
        log.info("Creating Pinecone index %r (%d-d, %s)", index_name, dimension,
                 settings.pinecone_metric)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )
    _wait_until_index_ready(pc, index_name, settings)

    actual = _attr(pc.describe_index(index_name), "dimension")
    if actual is not None and int(actual) != int(dimension):
        raise ValueError(
            f"Index {index_name!r} is {actual}-d but {dimension}-d vectors were "
            f"expected. Pinecone cannot resize an index -- point PINECONE_INDEX "
            f"at a new name."
        )
    return pc


def namespace_counts(
    pc: Any = None, index_name: str | None = None, settings: Settings | None = None
) -> dict[str, int]:
    """Vector count per namespace, for health checks and ingest reporting."""
    settings = settings or get_settings()
    pc = pc or get_pinecone_client()
    stats = pc.Index(_index_name(index_name, settings)).describe_index_stats()
    return {
        name: (info or {}).get("vector_count", 0)
        for name, info in (stats.get("namespaces") or {}).items()
    }


def namespace_count(
    namespace: str,
    pc: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> int:
    """Vector count for one namespace, or 0 when it holds nothing yet.

    One `describe_index_stats` call either way, so this reads the shared
    unpacking rather than repeating it -- Pinecone reports the counts for every
    namespace in a single response, and two spellings of that response shape are
    two things to fix when the SDK changes one of them.
    """
    return namespace_counts(pc, index_name, settings).get(namespace, 0)


def wait_for_vectors(
    namespace: str,
    expected: int,
    pc: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> int:
    """Block until upserted vectors are actually queryable.

    Without this a retrieval issued straight after ingest returns an empty list,
    because Pinecone serverless is eventually consistent. Do not drop it when
    refactoring ingest.
    """
    settings = settings or get_settings()
    pc = pc or get_pinecone_client()
    deadline = time.monotonic() + settings.freshness_timeout
    count = 0
    while time.monotonic() < deadline:
        count = namespace_count(namespace, pc, index_name, settings)
        if count >= expected:
            return count
        time.sleep(settings.freshness_poll_interval)
    raise TimeoutError(
        f"Namespace {namespace!r} held {count} vectors after "
        f"{settings.freshness_timeout}s, expected at least {expected}."
    )


def prune_orphans(
    namespace: str,
    keep_ids: Iterable[str],
    pc: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> int:
    """Delete vectors in a namespace that no longer have a source chunk.

    Ingest reconciles rather than only upserting. Upserting alone never removes
    anything, so a deleted or renamed policy file would leave vectors behind
    that keep answering questions -- for an HR copilot, that means serving
    withdrawn policy.
    """
    settings = settings or get_settings()
    pc = pc or get_pinecone_client()
    index = pc.Index(_index_name(index_name, settings))

    existing: set[str] = set()
    for page in index.list(namespace=namespace):
        existing.update(page)

    stale = existing - set(keep_ids)
    if stale:
        index.delete(ids=list(stale), namespace=namespace)
    return len(stale)


def delete_namespace(
    namespace: str,
    pc: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> bool:
    """Drop an entire namespace. Used to clean up after a namespace rename.

    Returns False when the namespace does not exist, so calling it twice is not
    an error.
    """
    settings = settings or get_settings()
    pc = pc or get_pinecone_client()
    index = pc.Index(_index_name(index_name, settings))
    if namespace not in namespace_counts(pc, index_name, settings):
        return False
    index.delete(delete_all=True, namespace=namespace)
    log.info("Deleted namespace %r", namespace)
    return True


def get_vectorstore(
    namespace: str,
    embedding: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
):
    """Vector store bound to one namespace of one index."""
    from langchain_pinecone import PineconeVectorStore

    settings = settings or get_settings()
    return PineconeVectorStore(
        index_name=_index_name(index_name, settings),
        embedding=embedding or get_embeddings(),
        namespace=namespace,
    )


def upsert_documents(
    chunks: Sequence[Document],
    namespace: str,
    embedding: Any = None,
    prune: bool = True,
    pc: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
):
    """Embed chunks into a namespace, wait for them, then drop orphans.

    Idempotent: ids derive from origin, source and offset, so re-running
    overwrites rather than duplicating.
    """
    from langchain_pinecone import PineconeVectorStore

    if not chunks:
        raise ValueError("No chunks to upsert.")

    settings = settings or get_settings()
    index_name = _index_name(index_name, settings)
    embedding = embedding or get_embeddings()
    ids = [chunk_id(chunk) for chunk in chunks]
    pc = ensure_index(pc, index_name, settings=settings)

    store = PineconeVectorStore.from_documents(
        documents=list(chunks),
        embedding=embedding,
        index_name=index_name,
        namespace=namespace,
        ids=ids,
    )
    wait_for_vectors(namespace, len(set(ids)), pc, index_name, settings)
    if prune:
        pruned = prune_orphans(namespace, set(ids), pc, index_name, settings)
        if pruned:
            log.info("Pruned %d orphaned vector(s) with no source on disk", pruned)
    return store


def to_relevance_scale(raw_cosine: float) -> float:
    """Convert raw cosine to LangChain's 0..1 relevance scale.

    `search_type="similarity_score_threshold"` compares against (c + 1) / 2, not
    against raw cosine. Passing a raw cosine straight through filters nothing:
    0.15 would be read as raw cosine -0.70.
    """
    return (raw_cosine + 1) / 2


def get_retriever(
    namespace: str,
    k: int | None = None,
    score_threshold: float | None = USE_CONFIGURED_GATE,
    embedding: Any = None,
    metadata_filter: dict[str, Any] | None = None,
    index_name: str | None = None,
    settings: Settings | None = None,
):
    """Retriever over one namespace, optionally filtered by metadata.

    `score_threshold` is a *raw cosine* value. `USE_CONFIGURED_GATE` means "use
    the configured gate"; None disables the gate and always returns k chunks.

    `metadata_filter` is a Pinecone filter expression, e.g.
    `{"department": {"$eq": "payroll"}}` or `{"doc_type": {"$in": ["pdf"]}}`.
    """
    settings = settings or get_settings()
    k = k or settings.default_k
    if score_threshold == USE_CONFIGURED_GATE:
        score_threshold = settings.relevance_threshold

    vectorstore = get_vectorstore(namespace, embedding, index_name, settings)
    search_kwargs: dict[str, Any] = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    if score_threshold is None:
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    search_kwargs["score_threshold"] = to_relevance_scale(score_threshold)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs=search_kwargs
    )


def similarity_with_scores(
    question: str,
    namespace: str,
    k: int | None = None,
    embedding: Any = None,
    metadata_filter: dict[str, Any] | None = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> list[tuple[Document, float]]:
    """Top-k chunks with raw cosine scores, ungated.

    The scores reported here are on the same scale `score_threshold` expects,
    which is what makes the gate tunable from observed numbers.
    """
    settings = settings or get_settings()
    store = get_vectorstore(namespace, embedding, index_name, settings)
    return store.similarity_search_with_score(
        question, k=k or settings.default_k, filter=metadata_filter
    )
