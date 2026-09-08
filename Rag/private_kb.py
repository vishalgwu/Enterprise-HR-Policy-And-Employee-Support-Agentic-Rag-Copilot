"""Private (internal) HR knowledge base: ingest + retrieval.

The scraped public article lives in the default Pinecone namespace; the private
KB lives in its own, so internal policy is never mixed with public content in a
retrieval result. The mechanics are shared with the public corpus -- see
`Rag.vectorstore` -- this module only supplies the corpus and its defaults.
"""

if __package__ in (None, ""):  # allow `python Rag/private_kb.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from Rag import config
from Rag.clients import get_embeddings
from Rag.loaders import chunk_id, load_markdown_documents, split_documents
from Rag.vectorstore import (
    get_retriever,
    get_vectorstore as _get_vectorstore,
    namespace_count,
    prune_orphans,
    similarity_with_scores,
    to_relevance_scale,
    upsert_documents,
    wait_for_vectors,
)

PRIVATE_NAMESPACE = config.PRIVATE_NAMESPACE
KB_DIR = config.KB_DIR
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
RELEVANCE_THRESHOLD = config.RELEVANCE_THRESHOLD
FRESHNESS_TIMEOUT = config.FRESHNESS_TIMEOUT
FRESHNESS_POLL_INTERVAL = config.FRESHNESS_POLL_INTERVAL

_chunk_id = chunk_id
_to_relevance_scale = to_relevance_scale


def load_private_documents(kb_dir: Path | None = None) -> List[Document]:
    """Read every markdown file in the private KB directory."""
    return load_markdown_documents(kb_dir or config.KB_DIR, origin="private-kb")


def split_private_documents(documents) -> List[Document]:
    """Chunk on markdown headings so each chunk starts at its section title."""
    return split_documents(documents, separators=config.MARKDOWN_SEPARATORS)


def get_vectorstore(embedding=None) -> PineconeVectorStore:
    """Vector store bound to the private namespace."""
    return _get_vectorstore(PRIVATE_NAMESPACE, embedding)


def build_private_kb(embedding=None, kb_dir: Path | None = None) -> PineconeVectorStore:
    """Chunk the private documents and upsert them into the private namespace."""
    chunks = split_private_documents(load_private_documents(kb_dir))
    return upsert_documents(chunks, namespace=PRIVATE_NAMESPACE, embedding=embedding)


def get_kb_retriever(k: int = config.DEFAULT_K,
                     score_threshold: float | None = RELEVANCE_THRESHOLD,
                     embedding=None):
    """Retriever over the private KB only.

    `score_threshold` is a raw cosine value; pass None to disable the gate and
    always return k chunks.
    """
    return get_retriever(PRIVATE_NAMESPACE, k=k, score_threshold=score_threshold,
                         embedding=embedding)


def retrieve_with_scores(question: str, k: int = config.DEFAULT_K,
                         embedding=None) -> List[Tuple[Document, float]]:
    """Top-k chunks with raw cosine scores, ungated."""
    return similarity_with_scores(question, PRIVATE_NAMESPACE, k=k, embedding=embedding)


def retrieve_relevant(question: str, k: int = config.DEFAULT_K,
                      score_threshold: float = RELEVANCE_THRESHOLD,
                      embedding=None) -> List[Tuple[Document, float]]:
    """Top-k chunks above the raw-cosine gate, as (document, score) pairs.

    An empty list means nothing in the private KB is close enough to the
    question -- the signal for the agent to rewrite the query or fall back to
    web search.
    """
    return [
        (doc, score)
        for doc, score in retrieve_with_scores(question, k=k, embedding=embedding)
        if score >= score_threshold
    ]


def _namespace_count(pc: Pinecone | None = None) -> int:
    return namespace_count(PRIVATE_NAMESPACE, pc)


def _wait_for_vectors(pc: Pinecone, expected: int) -> int:
    return wait_for_vectors(PRIVATE_NAMESPACE, expected, pc)


def _prune_orphans(pc: Pinecone, keep_ids) -> int:
    return prune_orphans(PRIVATE_NAMESPACE, keep_ids, pc)


def main():
    config.configure_stdout()
    config.configure_logging()
    embeddings = get_embeddings()

    documents = load_private_documents()
    chunks = split_private_documents(documents)
    print(f"Private KB: {len(documents)} document(s) -> {len(chunks)} chunk(s)")

    build_private_kb(embedding=embeddings)
    print(f"Namespace {PRIVATE_NAMESPACE!r} holds {_namespace_count()} vector(s)")

    retriever = get_kb_retriever(k=4, embedding=embeddings)

    test_question = "What happens if retrieved documents are not relevant in Agentic RAG?"
    kb_docs = retriever.invoke(test_question)

    for i, doc in enumerate(kb_docs, 1):
        print(f"\n--- KB RESULT {i} ---")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:700])

    print("\n--- SCORES (ungated, raw cosine) ---")
    for doc, score in retrieve_with_scores(test_question, k=4, embedding=embeddings):
        keep = "keep" if score >= RELEVANCE_THRESHOLD else "drop"
        print(f"{score:.4f}  {keep}  {doc.metadata.get('source')}")

    # An out-of-domain question must come back empty so the agent knows to fall
    # back rather than answering from whatever happened to rank highest.
    off_topic = "What is the capital of France?"
    print(f"\n--- OUT OF DOMAIN: {off_topic!r} ---")
    print("gated retriever returned:", len(retriever.invoke(off_topic)), "doc(s)")
    for doc, score in retrieve_with_scores(off_topic, k=2, embedding=embeddings):
        print(f"  ungated would have returned {score:.4f}  {doc.metadata.get('source')}")


if __name__ == "__main__":
    main()
