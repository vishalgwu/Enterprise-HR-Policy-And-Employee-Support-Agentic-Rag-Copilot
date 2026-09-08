"""Pinecone index management and retrieval, parameterised by namespace.

Every operation takes a namespace so the same code serves the public web corpus
and the private KB. Nothing here hardcodes which corpus it is working on.
"""

import time
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from Rag import config
from Rag.clients import get_embeddings
from Rag.config import get_logger
from Rag.loaders import chunk_id

log = get_logger("vectorstore")


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=config.PINECONE_API_KEY)


def _index_names(pc: Pinecone) -> set[str]:
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return set(indexes.names())
    return {ix["name"] if isinstance(ix, dict) else ix.name for ix in indexes}


def _attr(obj, name):
    """Pinecone v7 returns objects or dicts depending on the call path."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _index_ready(description) -> bool:
    status = _attr(description, "status")
    if status is None:
        return False
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def _wait_until_index_ready(pc: Pinecone) -> None:
    deadline = time.monotonic() + config.INDEX_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _index_ready(pc.describe_index(config.PINECONE_INDEX_NAME)):
            return
        time.sleep(config.INDEX_READY_POLL_INTERVAL)
    raise TimeoutError(
        f"Pinecone index {config.PINECONE_INDEX_NAME!r} was not ready after "
        f"{config.INDEX_READY_TIMEOUT}s."
    )


def ensure_index(pc: Pinecone | None = None, dimension: int | None = None) -> Pinecone:
    """Create the serverless index if absent, then verify its dimension.

    The dimension check catches the trap where the embedding model changed but
    the index did not: Pinecone cannot resize an index, so the mismatch would
    otherwise surface as an opaque upsert error.
    """
    pc = pc or get_pinecone_client()
    dimension = dimension or config.EMBEDDING_DIM

    if config.PINECONE_INDEX_NAME not in _index_names(pc):
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    _wait_until_index_ready(pc)

    actual = _attr(pc.describe_index(config.PINECONE_INDEX_NAME), "dimension")
    if actual is not None and int(actual) != int(dimension):
        raise ValueError(
            f"Index {config.PINECONE_INDEX_NAME!r} is {actual}-d but {dimension}-d "
            f"vectors were expected. Pinecone cannot resize an index -- point "
            f"PINECONE_INDEX at a new name."
        )
    return pc


def namespace_count(namespace: str, pc: Pinecone | None = None) -> int:
    pc = pc or get_pinecone_client()
    stats = pc.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
    return ((stats.get("namespaces") or {}).get(namespace) or {}).get("vector_count", 0)


def wait_for_vectors(namespace: str, expected: int, pc: Pinecone | None = None) -> int:
    """Block until upserted vectors are actually queryable.

    Without this a retrieval issued straight after ingest returns an empty list,
    because Pinecone serverless is eventually consistent.
    """
    pc = pc or get_pinecone_client()
    deadline = time.monotonic() + config.FRESHNESS_TIMEOUT
    count = 0
    while time.monotonic() < deadline:
        count = namespace_count(namespace, pc)
        if count >= expected:
            return count
        time.sleep(config.FRESHNESS_POLL_INTERVAL)
    raise TimeoutError(
        f"Namespace {namespace!r} held {count} vectors after "
        f"{config.FRESHNESS_TIMEOUT}s, expected at least {expected}."
    )


def prune_orphans(namespace: str, keep_ids: Iterable[str],
                  pc: Pinecone | None = None) -> int:
    """Delete vectors in a namespace that no longer have a source chunk.

    Upserting alone never removes anything, so a deleted or renamed source file
    would leave vectors behind that still answer questions -- for an HR KB, that
    means serving withdrawn policy.
    """
    pc = pc or get_pinecone_client()
    index = pc.Index(config.PINECONE_INDEX_NAME)

    existing: set[str] = set()
    for page in index.list(namespace=namespace):
        existing.update(page)

    stale = existing - set(keep_ids)
    if stale:
        index.delete(ids=list(stale), namespace=namespace)
    return len(stale)


def get_vectorstore(namespace: str, embedding=None) -> PineconeVectorStore:
    """Vector store bound to one namespace."""
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embedding or get_embeddings(),
        namespace=namespace,
    )


def upsert_documents(
    chunks: Sequence[Document],
    namespace: str,
    embedding=None,
    prune: bool = True,
    pc: Pinecone | None = None,
) -> PineconeVectorStore:
    """Embed chunks into a namespace, wait for them, and drop orphans.

    Idempotent: ids derive from source and offset, so re-running overwrites
    rather than duplicating.
    """
    if not chunks:
        raise ValueError("No chunks to upsert.")

    embedding = embedding or get_embeddings()
    ids = [chunk_id(chunk) for chunk in chunks]
    pc = ensure_index(pc)

    store = PineconeVectorStore.from_documents(
        documents=list(chunks),
        embedding=embedding,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=namespace,
        ids=ids,
    )
    wait_for_vectors(namespace, len(chunks), pc)
    if prune:
        pruned = prune_orphans(namespace, set(ids), pc)
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
    k: int = config.DEFAULT_K,
    score_threshold: float | None = config.RELEVANCE_THRESHOLD,
    embedding=None,
):
    """Retriever over one namespace.

    `score_threshold` is a raw cosine value; pass None to disable the gate and
    always return k chunks.
    """
    vectorstore = get_vectorstore(namespace, embedding)
    if score_threshold is None:
        return vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": to_relevance_scale(score_threshold)},
    )


def similarity_with_scores(
    question: str,
    namespace: str,
    k: int = config.DEFAULT_K,
    embedding=None,
) -> List[Tuple[Document, float]]:
    """Top-k chunks with raw cosine scores, ungated."""
    return get_vectorstore(namespace, embedding).similarity_search_with_score(question, k=k)
