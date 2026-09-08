"""Document loading, chunking and evidence rendering.

Format-agnostic by design: the private KB, an HR upload and the scraped public
article differ only in where bytes come from and which separators split them
best. One loader handles markdown, plain text, PDF and DOCX, because the product
brief lets authorised HR staff upload all four.

Heavy readers (`pypdf`, `python-docx`) are imported inside their functions so
that ingesting markdown never pays to import a PDF parser.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import (
    MARKDOWN_SEPARATORS,
    PROSE_SEPARATORS,
    Settings,
    get_logger,
    get_settings,
)

log = get_logger("loaders")

# Extension -> the `doc_type` recorded in metadata. Also the allow-list for
# uploads: anything not here is rejected rather than silently skipped.
SUPPORTED_SUFFIXES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
}

DEFAULT_DEPARTMENT = "general"

# Which separators suit which format. Markdown gets heading-first splitting;
# everything else is prose.
_SEPARATORS_BY_TYPE: dict[str, list[str]] = {
    "markdown": MARKDOWN_SEPARATORS,
}


# --- Identity ----------------------------------------------------------------


def chunk_id(chunk: Document) -> str:
    """Stable id so re-ingesting overwrites instead of duplicating.

    Keyed on origin, source and byte offset rather than on content, so an edited
    chunk updates in place instead of leaving its previous version behind as a
    second vector.

    `origin` is part of the basis because `source` is only a file name: without
    it, `benefits.pdf` uploaded by two departments would collide on one id and
    the second ingest would silently overwrite the first.
    """
    metadata = chunk.metadata or {}
    basis = ":".join(
        str(metadata.get(key, ""))
        for key in ("origin", "source", "start_index")
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


# --- File readers ------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    blocks = [p.text.strip() for p in document.paragraphs]
    # Tables carry real policy content (benefit tiers, expense caps) and are not
    # in `paragraphs`, so they would be dropped silently without this.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(block for block in blocks if block)


_READERS: dict[str, Callable[[Path], str]] = {
    "markdown": _read_text,
    "text": _read_text,
    "pdf": _read_pdf,
    "docx": _read_docx,
}


def is_supported(path: Path | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def read_document_text(path: Path | str) -> str:
    """Extract plain text from one supported file."""
    path = Path(path)
    doc_type = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if doc_type is None:
        raise ValueError(
            f"Unsupported file type {path.suffix!r} for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )
    return _READERS[doc_type](path)


# --- Document construction ---------------------------------------------------


def _title_from(text: str, fallback: str) -> str:
    """First markdown heading, else the first non-empty line, else the stem."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        return line.lstrip("#").strip() or fallback
    return fallback


def load_file(
    path: Path | str,
    origin: str,
    department: str = DEFAULT_DEPARTMENT,
) -> Document | None:
    """Read one file into a Document with retrieval metadata.

    Returns None for an empty file rather than raising, so one blank document in
    a directory cannot fail a whole ingest.

    The metadata keys are the ones retrieval filters on -- `department` and
    `doc_type` are what the product brief calls for -- plus `source` and `title`
    for citing the answer.
    """
    path = Path(path)
    text = read_document_text(path).strip()
    if not text:
        log.warning("Skipping %s: no extractable text", path.name)
        return None

    return Document(
        page_content=text,
        metadata={
            "source": path.name,
            "title": _title_from(text, path.stem),
            "doc_id": path.stem,
            "doc_type": SUPPORTED_SUFFIXES[path.suffix.lower()],
            "department": department or DEFAULT_DEPARTMENT,
            "origin": origin,
            "ingested_on": date.today().isoformat(),
        },
    )


def load_directory(
    directory: Path | str,
    origin: str,
    recursive: bool = True,
) -> list[Document]:
    """Load every supported file under a directory.

    A subdirectory name becomes the document's `department`, so
    `data/private_kb/payroll/bonus.md` is filterable without a manifest and
    files sitting directly in the root stay `general`. Unsupported files are
    skipped with a log line rather than failing the run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    paths = sorted(directory.rglob("*") if recursive else directory.glob("*"))
    documents: list[Document] = []
    for path in paths:
        if not path.is_file():
            continue
        if not is_supported(path):
            log.debug("Skipping unsupported file %s", path.name)
            continue
        relative = path.relative_to(directory).parts
        department = relative[0] if len(relative) > 1 else DEFAULT_DEPARTMENT
        document = load_file(path, origin=origin, department=department)
        if document is not None:
            documents.append(document)

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {directory}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )
    return documents


def load_web_documents(
    url: str,
    content_class: str | None = None,
    title: str = "",
    timeout: int = 30,
    settings: Settings | None = None,
) -> list[Document]:
    """Load one web page, optionally scoped to a single content container.

    Scoping matters: the unscoped HR Acuity article page is ~18.5k characters of
    which roughly a fifth is navigation, related-post cards and footer.
    """
    from bs4 import SoupStrainer
    from langchain_community.document_loaders import WebBaseLoader

    settings = settings or get_settings()
    bs_kwargs = (
        {"parse_only": SoupStrainer(class_=class_matcher(content_class))}
        if content_class
        else {}
    )
    loader = WebBaseLoader(
        web_paths=[url],
        header_template={"User-Agent": settings.user_agent},
        requests_kwargs={"timeout": timeout},
        bs_kwargs=bs_kwargs,
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    documents = [d for d in loader.load() if d.page_content.strip()]
    if not documents:
        raise RuntimeError(
            f"No text extracted from {url}. The page layout may have changed; "
            f"check content_class={content_class!r}."
        )
    for document in documents:
        document.metadata.update(
            {
                "source": url,
                "doc_type": "web",
                "department": DEFAULT_DEPARTMENT,
                "origin": "public-web",
            }
        )
        document.metadata.setdefault("title", title or url)
    return documents


def class_matcher(class_name: str) -> Callable[[object], bool]:
    """SoupStrainer matcher for one CSS class.

    bs4 compares `class_` against the entire class attribute, so the plain
    string "entry-content" silently fails to match
    `class="entry-content section-padding ..."` and yields an empty document.
    A callable that tokenises the attribute is the only thing that works.
    """

    def matches(value: object) -> bool:
        return bool(value) and class_name in str(value).split()

    return matches


# --- Chunking ----------------------------------------------------------------


def separators_for(doc_type: str | None) -> list[str]:
    """Best separators for a format, defaulting to prose."""
    return _SEPARATORS_BY_TYPE.get(doc_type or "", PROSE_SEPARATORS)


def split_documents(
    documents: Iterable[Document],
    separators: Sequence[str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Chunk documents, recording each chunk's offset for stable ids.

    With `separators=None` each document is split by its own `doc_type`, so a
    mixed directory of markdown and PDF is one call rather than one call per
    format. Pass separators explicitly to force a single strategy.
    """
    settings = settings or get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitters: dict[tuple, RecursiveCharacterTextSplitter] = {}

    def splitter_for(chosen: Sequence[str]) -> RecursiveCharacterTextSplitter:
        key = tuple(chosen)
        if key not in splitters:
            splitters[key] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                separators=list(chosen),
            )
        return splitters[key]

    chunks: list[Document] = []
    for document in documents:
        chosen = separators or separators_for(document.metadata.get("doc_type"))
        chunks.extend(splitter_for(chosen).split_documents([document]))
    return chunks


# --- Rendering ---------------------------------------------------------------


def format_documents(docs: Iterable[Document]) -> str:
    """Render chunks as numbered, source-labelled evidence for a prompt.

    Deliberately unchanged from the format the grader prompt was calibrated
    against. Adding department or title here would alter what the grader sees on
    every question, which is not a free change.
    """
    docs = list(docs)
    if not docs:
        return ""
    return "\n\n".join(
        f"[{i}] source={doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def cite_sources(docs: Iterable[Document]) -> list[dict[str, str]]:
    """Deduplicated source descriptors, for the `sources` field of an answer."""
    seen: dict[str, dict[str, str]] = {}
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "unknown"))
        if source not in seen:
            seen[source] = {
                "source": source,
                "title": str(metadata.get("title", source)),
                "department": str(metadata.get("department", DEFAULT_DEPARTMENT)),
                "doc_type": str(metadata.get("doc_type", "")),
            }
    return list(seen.values())
