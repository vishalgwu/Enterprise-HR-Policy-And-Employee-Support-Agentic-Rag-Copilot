"""Model and tool clients: embeddings, chat LLM, web search.

Every client is a factory, never a module-level singleton. Constructing one at
import time -- and especially calling `.invoke()` there -- would make importing
this module cost an API call.
"""

# Imported first: it exports GROQ_API_KEY / TAVILY_API_KEY / PINECONE_API_KEY
# into the environment, which the clients below read when constructed.
from Rag import config

from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch


@lru_cache(maxsize=2)
def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings, cached per model name.

    Caching matters: constructing this loads a ~90 MB model from disk. Without
    the cache every retrieval call that omits an explicit `embedding` would
    reload it, which is ruinous once a request handler retrieves per turn.

    Vectors are normalized, so the cosine metric on the Pinecone index is
    equivalent to a dot product.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embedding=None) -> int:
    """Actual output width of the embedding model, for validating the index."""
    return len((embedding or get_embeddings()).embed_query("dimension probe"))


def get_llm(model: str | None = None,
            temperature: float = config.GROQ_TEMPERATURE) -> ChatGroq:
    """Groq chat model, for routing, grading and answer generation.

    Left uncached: callers legitimately want different temperatures for grading
    (0, deterministic) and for writing prose, and constructing one is cheap.
    """
    return ChatGroq(model=model or config.GROQ_MODEL, temperature=temperature)


def get_web_search(max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> TavilySearch:
    """Tavily search: the fallback when the private KB has nothing relevant.

    `.invoke({"query": ...})` returns a dict with `answer` and `results` keys,
    not a list of Documents -- callers have to adapt it before it can be mixed
    with retriever output.
    """
    return TavilySearch(
        max_results=max_results,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )
