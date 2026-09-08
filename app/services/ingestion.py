"""The ingestion pipeline as one service-layer door: load -> chunk -> index.

Every step here already exists one layer down, in `app.rag.loaders` (reading
markdown, text, PDF and DOCX; chunking; stable chunk ids) and `app.rag.ingest`
(embed, upsert, reconcile). This module is a facade over those, not a second
implementation of them: the API layer and the scripts get one import to
remember, and there is still exactly one place where a PDF is parsed.

That matters more than it sounds. Two loaders drift -- one learns to read DOCX
tables and the other does not, one records `origin` in the chunk id and the
other does not -- and the symptom is a retrieval that silently returns the wrong
half of the corpus. So add capability in `app.rag.loaders` and expose it here.

The shape of the pipeline:

    load_file / load_directory   bytes on disk  -> Document with metadata
    chunk_documents              Document       -> chunks with start_index
    ingest_upload / ingest_directory            -> embedded, upserted, pruned

`load_file` returns a list rather than a single Document, so callers can treat
one file and one directory the same way, and so an empty file is an empty list
rather than a None to unpack.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings, get_logger
from app.rag.ingest import (
    IngestReport,
    ingest_directory,
    ingest_documents,
    ingest_upload,
)
from app.rag.loaders import (
    DEFAULT_DEPARTMENT,
    SUPPORTED_SUFFIXES,
    chunk_id,
    is_supported,
    read_document_text,
    split_documents,
)
from app.rag.loaders import load_directory as _load_directory
from app.rag.loaders import load_file as _load_file

log = get_logger("ingestion")

# The extensions an upload may carry. Derived from the loader's table rather
# than restated, so the allow-list and the reader registry cannot disagree.
SUPPORTED = frozenset(SUPPORTED_SUFFIXES)

__all__ = [
    "DEFAULT_DEPARTMENT",
    "SUPPORTED",
    "IngestReport",
    "chunk_documents",
    "chunk_id",
    "ingest_directory",
    "ingest_documents",
    "ingest_upload",
    "is_supported",
    "load_directory",
    "load_file",
    "read_document_text",
]


def load_file(
    path: Path | str,
    origin: str = "upload",
    department: str = DEFAULT_DEPARTMENT,
) -> list[Document]:
    """Read one supported file into a list of one Document -- or none.

    `origin` is not optional decoration: it is part of the chunk id, and it is
    what keeps `benefits.pdf` uploaded by two departments from colliding on one
    vector. It defaults to "upload" because that is the caller this signature
    exists for; corpus ingests pass their namespace.

    A file with no extractable text yields `[]` rather than raising, so one
    blank document cannot fail a whole batch. An unsupported extension does
    raise -- a silently skipped upload looks to the user like a successful one.
    """
    document = _load_file(path, origin=origin, department=department)
    return [document] if document is not None else []


def load_directory(
    directory: Path | str,
    origin: str,
    recursive: bool = True,
) -> list[Document]:
    """Load every supported file under a directory.

    A subdirectory name becomes the document's `department`, so
    `data/private_kb/payroll/bonus.md` is filterable without a manifest.
    """
    return _load_directory(directory, origin=origin, recursive=recursive)


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Split documents into chunks, each carrying its offset in the original.

    Separators are chosen per document from its `doc_type`, so a directory
    holding markdown and PDF is one call: markdown splits on headings first,
    prose on paragraphs. `add_start_index` is what makes `chunk_id` stable, so
    re-ingesting an edited file updates chunks in place rather than leaving the
    old text behind as a second vector.

    Size and overlap default to `Settings.chunk_size` / `chunk_overlap`. Pass
    `settings` to say *which* Settings: every other function in this codebase
    threads it, and a facade that silently reaches for the process-wide one
    cannot serve a FastAPI route that had Settings injected.
    """
    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        settings=settings,
    )
