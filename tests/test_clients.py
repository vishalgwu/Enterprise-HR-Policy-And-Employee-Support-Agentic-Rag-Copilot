"""The pure adapters in the client layer.

Everything here is the seam between a provider's response shape and the rest of
the system, so all of it runs offline -- no Groq, no Tavily, no key.

These matter more than their size suggests. `message_text` and
`web_results_to_text` are the only places a provider's shape is allowed to
leak in, and `is_rate_limit` is what keeps an exhausted quota from being read as
a knowledge gap: every safe default in the graph is "weak", so without it a dead
quota and a genuinely uncovered question look identical in the logs.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.rag.clients import (
    is_rate_limit,
    message_text,
    web_result_urls,
    web_results_to_text,
)

# --- Chat replies ------------------------------------------------------------


def test_a_plain_string_reply_passes_through():
    assert message_text(AIMessage(content="Twenty days.")) == "Twenty days."


def test_content_blocks_are_flattened_to_their_text():
    """The message interface allows a list, and providers do return one.

    This is the case that used to abort a run: `.content.strip()` on a list
    raises AttributeError, and a node that raises kills the whole graph.
    """
    message = AIMessage(
        content=[
            {"type": "text", "text": "Twenty days"},
            {"type": "text", "text": ", carried over to 31 March."},
        ]
    )
    assert message_text(message) == "Twenty days, carried over to 31 March."


def test_a_block_with_no_text_contributes_nothing():
    message = AIMessage(content=[{"type": "image_url", "image_url": "x"},
                                 {"type": "text", "text": "See above."}])
    assert message_text(message) == "See above."


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_reply_is_an_empty_string_not_an_error(empty):
    assert message_text(AIMessage(content=empty or "")) == ""
    assert message_text(empty) == ""


def test_something_that_is_not_a_message_is_still_coerced():
    """A node must get a string back whatever the provider handed it."""
    assert message_text(42) == "42"


# --- Rate limiting -----------------------------------------------------------


class _StatusError(Exception):
    status_code = 429


class RateLimitError(Exception):
    """Both openai and groq raise a class with exactly this name."""


@pytest.mark.parametrize(
    "exc",
    [
        RateLimitError("slow down"),
        _StatusError("429"),
        RuntimeError("Error code: 429 - tokens per day (TPD): Limit 200000"),
        RuntimeError("Rate limit reached for model"),
        RuntimeError("insufficient_quota: billing hard limit reached"),
    ],
)
def test_quota_failures_are_recognised(exc):
    assert is_rate_limit(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("tool_use_failed: model did not call a tool"),
        ValueError("PINECONE_API is not set"),
        TimeoutError("index was not ready"),
    ],
)
def test_ordinary_failures_are_not_mistaken_for_quota(exc):
    assert not is_rate_limit(exc)


# --- Tavily responses --------------------------------------------------------

TAVILY_RESPONSE = {
    "query": "uk statutory holiday",
    "answer": "The UK statutory minimum is 28 days.",
    "results": [
        {"title": "Holiday entitlement", "url": "https://gov.uk/holiday",
         "content": "Almost all workers get 5.6 weeks."},
        {"title": "Calculating leave", "url": "https://gov.uk/calculate",
         "content": "Use the holiday calculator."},
    ],
}


def test_a_tavily_response_becomes_gradeable_text():
    text = web_results_to_text(TAVILY_RESPONSE)
    assert "The UK statutory minimum is 28 days." in text
    assert "https://gov.uk/holiday" in text
    assert "Almost all workers get 5.6 weeks." in text


def test_an_empty_search_yields_empty_text_so_the_grader_is_skipped():
    assert web_results_to_text({"query": "x", "results": []}) == ""
    assert web_results_to_text(None) == ""


def test_a_bare_string_response_is_tolerated():
    """The shape has changed across langchain-tavily versions."""
    assert web_results_to_text("just a string") == "just a string"


def test_urls_are_extracted_in_order_and_capped():
    assert web_result_urls(TAVILY_RESPONSE) == [
        "https://gov.uk/holiday",
        "https://gov.uk/calculate",
    ]
    assert web_result_urls(TAVILY_RESPONSE, limit=1) == ["https://gov.uk/holiday"]


def test_results_without_a_url_are_dropped_rather_than_cited_as_blank():
    response = {"results": [{"title": "no url"}, {"url": "https://ok"}]}
    assert web_result_urls(response) == ["https://ok"]


def test_urls_from_a_failed_search_are_empty_not_an_error():
    assert web_result_urls(None) == []
