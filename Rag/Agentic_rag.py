"""Public web corpus: ingest the HR Acuity policy article into Pinecone.

The reusable pieces live in sibling modules -- `config`, `clients`, `loaders`,
`vectorstore` -- so this file is only the public-web entry point plus the names
older code imports from here.
"""

if __package__ in (None, ""):  # allow `python Rag/Agentic_rag.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List

from langchain_core.documents import Document
from pinecone import Pinecone

from Rag import config
from Rag.clients import embedding_dimension, get_embeddings, get_llm, get_web_search
from Rag.loaders import chunk_id, class_matcher, load_web_documents, split_documents
from Rag.vectorstore import (
    ensure_index,
    get_retriever,
    namespace_count,
    similarity_with_scores,
    upsert_documents,
)

# Re-exported so existing imports keep working.
configure_stdout = config.configure_stdout
GROQ_API_KEY = config.GROQ_API_KEY
TAVILY_API_KEY = config.TAVILY_API_KEY
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_CLOUD = config.PINECONE_CLOUD
PINECONE_REGION = config.PINECONE_REGION
PUBLIC_NAMESPACE = config.PUBLIC_NAMESPACE
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM
GROQ_MODEL = config.GROQ_MODEL
GROQ_TEMPERATURE = config.GROQ_TEMPERATURE
WEB_SEARCH_MAX_RESULTS = config.WEB_SEARCH_MAX_RESULTS
HR_POLICY_URL = config.HR_POLICY_URL
CONTENT_CLASS = config.HR_POLICY_CONTENT_CLASS
USER_AGENT = config.USER_AGENT
INDEX_READY_TIMEOUT = config.INDEX_READY_TIMEOUT
INDEX_READY_POLL_INTERVAL = config.INDEX_READY_POLL_INTERVAL

_chunk_id = chunk_id
_is_content_class = class_matcher(config.HR_POLICY_CONTENT_CLASS)


def load_hr_policy_documents() -> List[Document]:
    """Load the HR Acuity policy article, scoped to the article body."""
    return load_web_documents(
        url=config.HR_POLICY_URL,
        content_class=config.HR_POLICY_CONTENT_CLASS,
    )


def split_hr_policy_documents(documents) -> List[Document]:
    return split_documents(documents, separators=config.PROSE_SEPARATORS)


def ensure_pinecone_index(pc: Pinecone | None = None) -> Pinecone:
    """Create the 384-d serverless index if it does not already exist."""
    return ensure_index(pc)


def upsert_hr_policy_to_pinecone(chunks, embedding=None, pc: Pinecone | None = None):
    """Embed the public article into the default namespace."""
    return upsert_documents(
        chunks,
        namespace=config.PUBLIC_NAMESPACE,
        embedding=embedding,
        pc=pc,
    )


def get_public_retriever(k: int = config.DEFAULT_K, score_threshold=None, embedding=None):
    """Retriever over the scraped public corpus."""
    return get_retriever(
        config.PUBLIC_NAMESPACE, k=k, score_threshold=score_threshold, embedding=embedding
    )


def main():
    config.configure_stdout()

    documents = load_hr_policy_documents()
    chunks = split_hr_policy_documents(documents)
    print(f"Loaded {len(documents)} document(s) from {config.HR_POLICY_URL}")
    print(f"Split into {len(chunks)} chunk(s) for RAG")
    print(f"Preview: {documents[0].page_content[:300].replace(chr(10), ' ')}...")

    embeddings = get_embeddings()
    dim = embedding_dimension(embeddings)
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {dim}")
    if dim != config.EMBEDDING_DIM:
        raise ValueError(
            f"{config.EMBEDDING_MODEL} returned {dim}-d vectors but EMBEDDING_DIM "
            f"is {config.EMBEDDING_DIM}. Update it and use a fresh PINECONE_INDEX."
        )

    upsert_hr_policy_to_pinecone(chunks, embedding=embeddings)
    print(
        f"Pinecone index '{config.PINECONE_INDEX_NAME}' is ready "
        f"({config.EMBEDDING_DIM}-d, {config.EMBEDDING_MODEL}); "
        f"public namespace holds {namespace_count(config.PUBLIC_NAMESPACE)} vector(s)"
    )

    match = similarity_with_scores(
        "What HR policies should a company consider?",
        config.PUBLIC_NAMESPACE,
        k=1,
        embedding=embeddings,
    )
    if match:
        doc, score = match[0]
        print(f"Sample retrieval ({score:.4f}): {doc.page_content[:200].replace(chr(10), ' ')}...")

    print("\n--- GROQ LLM ---")
    reply = get_llm().invoke("Explain Agentic RAG in one sentence.")
    print(f"Model: {reply.response_metadata.get('model_name', config.GROQ_MODEL)}")
    print(f"Tokens: {reply.usage_metadata}")
    print(f"Reply: {reply.content.strip()}")

    print("\n--- TAVILY WEB SEARCH ---")
    hits = get_web_search(max_results=3).invoke(
        {"query": "What is agentic RAG document grading?"}
    )
    print(f"Answer: {str(hits.get('answer'))[:240]}")
    for hit in hits.get("results", [])[:3]:
        print(f"  - {hit.get('title', '')[:66]} | {hit.get('url', '')}")


if __name__ == "__main__":
    main()
