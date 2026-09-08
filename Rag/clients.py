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


def is_rate_limit(exc: BaseException) -> bool:
    """True when a provider rejected the call for quota, not for content.

    Worth distinguishing: the graph's safe defaults turn any failure into
    "evidence is weak", so a exhausted quota looks exactly like a knowledge gap
    unless it is named. Groq's free tier caps tokens per *day*, so this does not
    clear in a few seconds.
    """
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "error code: 429" in text


def web_results_to_text(result) -> str:
    """Flatten a Tavily response into text a grader and a prompt can consume.

    Tavily returns a dict (`answer`, `results`, ...) rather than Documents, and
    the shape varies with package version, so this tolerates a bare string too.
    """
    if not isinstance(result, dict):
        return str(result or "")

    lines = []
    answer = result.get("answer")
    if answer:
        lines.append(f"Tavily answer: {answer}")
    for item in result.get("results") or []:
        lines.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )
    return "\n\n".join(lines) if lines else ""


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
