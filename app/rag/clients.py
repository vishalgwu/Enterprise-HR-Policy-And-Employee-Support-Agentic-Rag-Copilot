"""Model and tool clients: embeddings, chat LLM, web search.

Every client is a factory, never a module-level singleton. Constructing one at
import time -- and especially calling `.invoke()` there -- would make importing
this module cost an API call.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Imported first: importing it exports GROQ_API_KEY / TAVILY_API_KEY /
# PINECONE_API_KEY into the environment, which the clients below read when they
# are constructed.
from app.core.config import Settings, get_settings


@lru_cache(maxsize=2)
def get_embeddings(model_name: str | None = None):
    """Local sentence-transformers embeddings, cached per model name.

    The cache is load-bearing: constructing this loads a ~90 MB model from disk.
    Without it every retrieval that omits an explicit `embedding` would reload
    the model, which is ruinous once a request handler retrieves per turn.

    Vectors are normalised, so the cosine metric on the Pinecone index is
    equivalent to a dot product.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name or get_settings().embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embedding: Any = None) -> int:
    """Actual output width of the embedding model, for validating the index."""
    return len((embedding or get_embeddings()).embed_query("dimension probe"))


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
):
    """Groq chat model, for routing, grading, rewriting and generation.

    Left uncached: callers legitimately want different temperatures for grading
    (0, deterministic) and for writing prose, and constructing one is cheap.
    """
    from langchain_groq import ChatGroq

    settings = settings or get_settings()
    settings.require("groq_api_key")
    return ChatGroq(
        model=model or settings.groq_model,
        temperature=settings.groq_temperature if temperature is None else temperature,
    )


def get_web_search(max_results: int | None = None, settings: Settings | None = None):
    """Tavily search: the fallback when the private KB has nothing relevant.

    `.invoke({"query": ...})` returns a dict with `answer` and `results` keys,
    not a list of Documents -- callers have to adapt it with
    `web_results_to_text` before it can be mixed with retriever output.
    """
    from langchain_tavily import TavilySearch

    settings = settings or get_settings()
    settings.require("tavily_api_key")
    return TavilySearch(
        max_results=max_results or settings.web_search_max_results,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )


def is_rate_limit(exc: BaseException) -> bool:
    """True when a provider rejected the call for quota, not for content.

    Worth distinguishing: every safe default in the graph turns a failure into
    "evidence is weak", so an exhausted quota looks exactly like a knowledge gap
    unless it is named. Groq's free tier caps tokens per *day*, so this does not
    clear in a few seconds -- observed:
    `429 ... tokens per day (TPD): Limit 200000, Used 199771`.
    """
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "error code: 429" in text


def web_results_to_text(result: Any) -> str:
    """Flatten a Tavily response into text a grader and a prompt can consume.

    Tavily returns a dict (`query`, `answer`, `results`, `images`, ...) rather
    than Documents, and the shape varies with package version, so this tolerates
    a bare string too.
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


def web_result_urls(result: Any, limit: int = 5) -> list[str]:
    """URLs behind a Tavily response, for citing sources in an API reply."""
    if not isinstance(result, dict):
        return []
    urls = [item.get("url") for item in (result.get("results") or [])]
    return [u for u in urls if u][:limit]
