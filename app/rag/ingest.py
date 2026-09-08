"""The ingest pipeline: documents in, reconciled namespace out.

One implementation serves every corpus. A directory of HR policy, a single
uploaded PDF and a scraped web page differ only in how their Documents are
produced, so they share load -> chunk -> embed -> upsert -> reconcile.

Ingest *reconciles*; it does not only upsert. Orphan pruning is why: a policy
file that was deleted or renamed leaves vectors behind that keep answering
questions, and an HR copilot quoting withdrawn policy is worse than one that
says it does not know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document

from app.core.config import Settings, get_logger, get_settings
from app.rag.clients import get_embeddings
from app.rag.loaders import (
    load_directory,
    load_file,
    load_web_documents,
    split_documents,
)
from app.rag.vectorstore import namespace_count, upsert_documents

log = get_logger("ingest")


@dataclass
class IngestReport:
    """What one ingest actually did. Returned rather than printed.

    Library code logs and returns; only an entry point prints. This is the shape
    the `/ingest` endpoint and the CLI both render.
    """

    namespace: str
    documents: int = 0
    chunks: int = 0
    vectors_in_namespace: int = 0
    sources: List[str] = field(default_factory=list)
    departments: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.documents} document(s) -> {self.chunks} chunk(s); "
            f"namespace {self.namespace!r} now holds "
            f"{self.vectors_in_namespace} vector(s)"
        )


def _describe(documents: Sequence[Document]) -> tuple[List[str], Dict[str, int]]:
    sources: List[str] = []
    departments: Dict[str, int] = {}
    for document in documents:
        source = str((document.metadata or {}).get("source", "unknown"))
        if source not in sources:
            sources.append(source)
        department = str((document.metadata or {}).get("department", "general"))
        departments[department] = departments.get(department, 0) + 1
    return sources, departments


def ingest_documents(
    documents: Sequence[Document],
    namespace: str,
    embedding: Any = None,
    prune: bool = True,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Chunk, embed and upsert documents into one namespace.

    `prune=False` is for adding an uploaded document to a namespace that other
    documents already occupy: pruning there would delete everything the upload
    did not contain. Pruning belongs to a full corpus rebuild.
    """
    settings = settings or get_settings()
    if not documents:
        raise ValueError("No documents to ingest.")

    embedding = embedding or get_embeddings()
    chunks = split_documents(documents, settings=settings)
    if not chunks:
        raise RuntimeError("Documents produced no chunks; check chunk_size.")

    upsert_documents(
        chunks,
        namespace=namespace,
        embedding=embedding,
        prune=prune,
        index_name=index_name,
        settings=settings,
    )

    sources, departments = _describe(documents)
    report = IngestReport(
        namespace=namespace,
        documents=len(documents),
        chunks=len(chunks),
        vectors_in_namespace=namespace_count(
            namespace, index_name=index_name, settings=settings
        ),
        sources=sources,
        departments=departments,
    )
    log.info("Ingested into %r: %s", namespace, report.summary())
    return report


def ingest_directory(
    directory: Path | str,
    namespace: str,
    origin: str | None = None,
    embedding: Any = None,
    prune: bool = True,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Rebuild a namespace from every supported document under a directory.

    Prunes by default: the directory is the whole truth for this namespace, so
    anything in the namespace without a file on disk is stale.
    """
    documents = load_directory(directory, origin=origin or namespace)
    return ingest_documents(
        documents,
        namespace=namespace,
        embedding=embedding,
        prune=prune,
        index_name=index_name,
        settings=settings,
    )


def ingest_upload(
    path: Path | str,
    namespace: str,
    department: str = "general",
    origin: str | None = None,
    embedding: Any = None,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Add one uploaded file to a namespace without disturbing what is there.

    Never prunes -- see `ingest_documents`.
    """
    document = load_file(path, origin=origin or namespace, department=department)
    if document is None:
        raise ValueError(f"No extractable text in {Path(path).name}.")
    return ingest_documents(
        [document],
        namespace=namespace,
        embedding=embedding,
        prune=False,
        index_name=index_name,
        settings=settings,
    )


def ingest_web_page(
    url: str,
    namespace: str,
    content_class: str | None = None,
    title: str = "",
    embedding: Any = None,
    prune: bool = True,
    index_name: str | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Scrape one page into a namespace."""
    settings = settings or get_settings()
    documents = load_web_documents(
        url, content_class=content_class, title=title, settings=settings
    )
    return ingest_documents(
        documents,
        namespace=namespace,
        embedding=embedding,
        prune=prune,
        index_name=index_name,
        settings=settings,
    )
