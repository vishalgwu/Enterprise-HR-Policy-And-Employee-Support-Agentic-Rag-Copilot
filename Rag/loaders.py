"""Document loading, chunking and rendering.

Shared by both corpora: the scraped public article and the private markdown KB
differ only in where the text comes from and which separators split it best.
"""

import hashlib
from pathlib import Path
from typing import Callable, Iterable, List

from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Rag import config


def chunk_id(chunk: Document) -> str:
    """Stable id so re-ingesting overwrites instead of duplicating.

    Keyed on source plus offset rather than content, so an edited chunk updates
    in place instead of leaving its previous version behind as a second vector.
    """
    basis = f"{chunk.metadata.get('source', '')}:{chunk.metadata.get('start_index', '')}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def class_matcher(class_name: str) -> Callable[[object], bool]:
    """SoupStrainer matcher for one CSS class.

    bs4 compares `class_` against the entire class attribute, so the plain
    string "entry-content" silently fails to match
    `class="entry-content section-padding ..."` and yields an empty document.
    """

    def matches(value) -> bool:
        return bool(value) and class_name in str(value).split()

    return matches


def load_web_documents(
    url: str = config.HR_POLICY_URL,
    content_class: str | None = config.HR_POLICY_CONTENT_CLASS,
    title: str = "HR Policies and Procedures",
    timeout: int = 30,
) -> List[Document]:
    """Load one web page, optionally scoped to a single content container."""
    bs_kwargs = (
        {"parse_only": SoupStrainer(class_=class_matcher(content_class))}
        if content_class
        else {}
    )
    loader = WebBaseLoader(
        web_paths=[url],
        header_template={"User-Agent": config.USER_AGENT},
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
        document.metadata["source"] = url
        document.metadata.setdefault("title", title)
        document.metadata.setdefault("origin", "public-web")
    return documents


def load_markdown_documents(
    directory: Path | None = None,
    origin: str = "private-kb",
) -> List[Document]:
    """Read every markdown file in a directory as one Document each."""
    directory = Path(directory) if directory else config.KB_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0]
        heading = first_line.lstrip("#").strip() if first_line.startswith("#") else ""
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "title": heading or path.stem,
                    "doc_id": path.stem,
                    "origin": origin,
                },
            )
        )

    if not documents:
        raise RuntimeError(f"No markdown documents found in {directory}")
    return documents


def split_documents(
    documents: Iterable[Document],
    separators: List[str] | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Chunk documents, recording each chunk's offset for stable ids."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=separators or config.PROSE_SEPARATORS,
    )
    return splitter.split_documents(list(documents))


def format_documents(docs: Iterable[Document]) -> str:
    """Render chunks as numbered, source-labelled evidence for a prompt."""
    docs = list(docs)
    if not docs:
        return ""
    return "\n\n".join(
        f"[{i}] source={doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )
