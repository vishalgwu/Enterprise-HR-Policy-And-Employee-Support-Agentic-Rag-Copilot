"""Corpus-level retrieval: the private HR knowledge base and the public corpus.

Thin, deliberately. Everything here is `app.rag.vectorstore` with a namespace
already chosen, so there is exactly one place that decides which namespace each
corpus lives in and no caller has to remember.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.rag.vectorstore import get_retriever, similarity_with_scores

# Sentinel meaning "use the configured gate". None is a meaningful value -- it
# disables the gate -- so it cannot double as the default.
USE_CONFIGURED_GATE = -1.0


def kb_namespace(settings: Settings | None = None) -> str:
    return (settings or get_settings()).private_namespace


def public_namespace(settings: Settings | None = None) -> str:
    return (settings or get_settings()).public_namespace


def get_kb_retriever(
    k: int | None = None,
    score_threshold: float | None = USE_CONFIGURED_GATE,
    embedding: Any = None,
    department: str | None = None,
    doc_type: str | None = None,
    settings: Settings | None = None,
):
    """Retriever over the private HR knowledge base only.

    `department` and `doc_type` narrow retrieval to part of the corpus -- the
    metadata filtering the product brief calls for. They compose with the
    similarity gate rather than replacing it.
    """
    settings = settings or get_settings()
    return get_retriever(
        kb_namespace(settings),
        k=k,
        score_threshold=score_threshold,
        embedding=embedding,
        metadata_filter=build_filter(department=department, doc_type=doc_type),
        settings=settings,
    )


def get_public_retriever(
    k: int | None = None,
    score_threshold: float | None = None,
    embedding: Any = None,
    settings: Settings | None = None,
):
    """Retriever over the scraped public corpus. Ungated by default."""
    settings = settings or get_settings()
    return get_retriever(
        public_namespace(settings),
        k=k,
        score_threshold=score_threshold,
        embedding=embedding,
        settings=settings,
    )


def build_filter(
    department: str | None = None, doc_type: str | None = None
) -> Dict[str, Any] | None:
    """Compose a Pinecone metadata filter, or None when nothing is constrained."""
    clauses: Dict[str, Any] = {}
    if department:
        clauses["department"] = {"$eq": department}
    if doc_type:
        clauses["doc_type"] = {"$eq": doc_type}
    return clauses or None


def retrieve_with_scores(
    question: str,
    k: int | None = None,
    embedding: Any = None,
    department: str | None = None,
    doc_type: str | None = None,
    settings: Settings | None = None,
) -> List[Tuple[Document, float]]:
    """Top-k private-KB chunks with raw cosine scores, ungated.

    The diagnostic behind the gate: these are the numbers `relevance_threshold`
    is set from.
    """
    settings = settings or get_settings()
    return similarity_with_scores(
        question,
        kb_namespace(settings),
        k=k,
        embedding=embedding,
        metadata_filter=build_filter(department=department, doc_type=doc_type),
        settings=settings,
    )


def retrieve_relevant(
    question: str,
    k: int | None = None,
    score_threshold: float | None = None,
    embedding: Any = None,
    settings: Settings | None = None,
) -> List[Tuple[Document, float]]:
    """Private-KB chunks above the raw-cosine gate, as (document, score) pairs.

    An empty list means nothing in the KB is close enough to the question -- the
    signal for the agent to rewrite the query or fall back to web search.
    """
    settings = settings or get_settings()
    threshold = (
        settings.relevance_threshold if score_threshold is None else score_threshold
    )
    return [
        (doc, score)
        for doc, score in retrieve_with_scores(
            question, k=k, embedding=embedding, settings=settings
        )
        if score >= threshold
    ]
