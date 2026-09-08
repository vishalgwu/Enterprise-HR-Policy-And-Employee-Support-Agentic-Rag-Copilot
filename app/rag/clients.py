"""Model and tool clients: embeddings, chat LLM, web search.

Every client is a factory, never a module-level singleton. Constructing one at
import time -- and especially calling `.invoke()` there -- would make importing
this module cost an API call.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

# Imported first: importing it exports GROQ_API_KEY / TAVILY_API_KEY /
# PINECONE_API_KEY into the environment, which the clients below read when they
# are constructed.
from app.core.config import Settings, get_settings


@lru_cache(maxsize=4)
def _build_embeddings(provider: str, model_name: str, api_key: str):
    """Cached per (provider, model). See `get_embeddings` for why."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        # text-embedding-3-* are already unit-normalised, so cosine on the
        # Pinecone index is still equivalent to a dot product.
        return OpenAIEmbeddings(model=model_name, api_key=api_key)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(model_name: str | None = None, settings: Settings | None = None):
    """Embeddings for the configured provider.

    `huggingface` (default) runs `all-MiniLM-L6-v2` locally: no per-query cost
    and no employee question leaving the machine at embed time. `openai` calls
    the API instead, which is the only option on a host that cannot ship torch --
    a Vercel function, for one.

    The cache is load-bearing for the local provider: constructing it loads a
    ~90 MB model from disk, so without it every retrieval that omits an explicit
    `embedding` would reload the model. That is ruinous once a request handler
    retrieves per turn.

    Switching provider changes the vector width, which the Pinecone index is
    created with and cannot change. `Settings` rejects a contradictory
    `EMBEDDING_DIM` at construction; a new model still needs a new index name.
    """
    settings = settings or get_settings()
    provider = settings.embedding_provider
    api_key = settings.require("openai_api_key") if provider == "openai" else ""
    return _build_embeddings(
        provider, model_name or settings.active_embedding_model, api_key
    )


def embedding_dimension(embedding: Any = None) -> int:
    """Actual output width of the embedding model, for validating the index."""
    return len((embedding or get_embeddings()).embed_query("dimension probe"))


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
    provider: str | None = None,
):
    """Chat model for routing, grading, rewriting and generation.

    Groq by default; `LLM_PROVIDER=openai` switches to ChatOpenAI. Both satisfy
    the same contract the agent depends on -- `.with_structured_output` binding
    a Pydantic model via function calling -- so no node changes when the
    provider does.

    Left uncached: callers legitimately want different temperatures for grading
    (0, deterministic) and for writing prose, and constructing one is cheap.
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.openai_model,
            temperature=temperature,
            api_key=settings.require("openai_api_key"),
        )

    from langchain_groq import ChatGroq

    settings.require("groq_api_key")
    return ChatGroq(model=model or settings.groq_model, temperature=temperature)


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

    Covers both providers: `openai.RateLimitError` and Groq's rate-limit error
    share the class name, and both carry status 429.
    """
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return (
        "rate limit" in text
        or "error code: 429" in text
        or "insufficient_quota" in text  # OpenAI: billing exhausted, not throttled
    )


# `openai/gpt-oss-*` emits OpenAI's file-citation tokens -- 【2†L1-L4】,
# 【4:2†source】 -- because it was trained to cite retrieved documents that way.
# Nothing in this system numbers documents like that, so they reference nothing
# at all; observed live on a web-fallback answer about paid holidays, which came
# back with four of them. They matter more here than they would elsewhere: in an
# HR answer they read as authoritative citations to sources that do not exist,
# next to real ones this product does supply. The leading whitespace is part of
# the match so removing a token mid-sentence does not leave a double space.
#
# Bounded rather than greedy on purpose -- an unmatched opening bracket in
# legitimate text must not swallow the rest of the answer.
CITATION_ARTIFACT = re.compile(r"[ 	]*【[^】]{0,120}】")


def strip_citation_artifacts(text: str) -> str:
    """Remove a provider's dangling citation markers from a reply."""
    return CITATION_ARTIFACT.sub("", text)


def message_text(message: Any) -> str:
    """Flatten a chat model's reply to plain text, and clean provider noise.

    `.content` is a plain string for Groq and OpenAI today, but the message
    interface allows a list of content blocks and providers do return one. The
    graph must not depend on which: `"".strip()` on a list raises
    AttributeError, and a node that raises aborts the entire run. This is the
    same adapter role `web_results_to_text` plays for Tavily -- one place that
    absorbs a provider's response shape so nothing downstream has to.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Content blocks are dicts like {"type": "text", "text": "..."}; keep
        # only the text parts, since an image block has nothing to contribute.
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = "" if content is None else str(content)
    # Cleaned here, at the one funnel every reply passes through, so the API
    # response, the CLI and the audit record all hold the same clean text.
    return strip_citation_artifacts(text)


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
