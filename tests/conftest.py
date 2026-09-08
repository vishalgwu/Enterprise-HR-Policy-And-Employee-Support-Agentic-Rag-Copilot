"""Fakes for every client the agent talks to.

The whole suite runs offline: no Groq, no Pinecone, no Tavily, no embedding
model download. That is only possible because `build_nodes`, `build_graph` and
`Copilot` take their dependencies as arguments -- the fakes below are the reason
that injection exists.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.schemas import EvidenceGrade, RouteDecision  # noqa: E402
from app.core.config import Settings  # noqa: E402


class FakeLLM(Runnable):
    """A chat model that returns scripted decisions and a fixed reply.

    `grades` is consumed in order, so one instance can grade the KB weak and
    then the web good -- which is exactly the sequence the fallback path needs.
    The last value repeats once the list runs out.
    """

    def __init__(
        self,
        reply: str = "A generated answer.",
        route: str = "kb",
        grades: Sequence[str] = ("good",),
        rewrite: str | None = None,
        fail: bool = False,
    ) -> None:
        self.reply = reply
        self.route = route
        self.grades = list(grades)
        self.rewrite = rewrite
        self.fail = fail
        self.calls: list[str] = []

    # --- Runnable ---------------------------------------------------------
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append("invoke")
        if self.fail:
            raise RuntimeError("simulated provider outage")
        text = str(input)
        if self.rewrite is not None and "Rewrite" in text:
            return AIMessage(content=self.rewrite)
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def decide(_input: Any) -> Any:
            self.calls.append(schema.__name__)
            if self.fail:
                raise RuntimeError("simulated provider outage")
            if schema is RouteDecision:
                return RouteDecision(route=self.route)
            grade = self.grades.pop(0) if len(self.grades) > 1 else self.grades[0]
            return EvidenceGrade(grade=grade)

        return RunnableLambda(decide)


class FakeRetriever:
    """A retriever that returns scripted chunks and records its queries.

    `results` may be a list (returned every time) or a list of lists (one per
    call), which is how a retry that finally succeeds is exercised.
    """

    def __init__(self, results: Any = None, fail: bool = False) -> None:
        self.results = results if results is not None else []
        self.fail = fail
        self.queries: list[str] = []

    def invoke(self, query: str, *args: Any, **kwargs: Any) -> list[Document]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("simulated Pinecone outage")
        if self.results and isinstance(self.results[0], list):
            index = min(len(self.queries) - 1, len(self.results) - 1)
            return self.results[index]
        return self.results


class FakeWebSearch:
    """A Tavily stand-in returning the dict shape the real client returns."""

    def __init__(self, answer: str = "", results: Any = None, fail: bool = False) -> None:
        self.answer = answer
        self.results = results if results is not None else []
        self.fail = fail
        self.queries: list[str] = []

    def invoke(self, payload: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(payload.get("query", ""))
        if self.fail:
            raise RuntimeError("simulated Tavily outage")
        return {"query": payload.get("query"), "answer": self.answer,
                "results": self.results}


def kb_doc(
    content: str = "Full-time employees accrue 20 days of paid time off.",
    source: str = "leave-and-time-off.md",
    department: str = "general",
    doc_type: str = "markdown",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "title": source.replace("-", " ").replace(".md", "").title(),
            "department": department,
            "doc_type": doc_type,
            "origin": "hr-docs",
        },
    )


TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_HIDE_INPUTS",
    "LANGSMITH_HIDE_OUTPUTS",
)

PROVIDER_VARS = ("OPENAI_API_KEY",)


@pytest.fixture(autouse=True)
def clean_process_env(monkeypatch):
    """Give every test a process environment free of this project's exports.

    Two things depend on it.

    Offline guarantee: `app.core.config` exports tracing into `os.environ` at
    import, so on a machine with LangSmith enabled every graph test would upload
    its run — turning a unit test into a network call and shipping fake HR
    content to a third party.

    Isolation: `OPENAI_API_KEY` is the one setting whose export name equals its
    input alias, so a test that exports it configures every `Settings` built
    afterwards — `_env_file=None` does not help, because environment outranks
    the file. Clearing at setup rather than teardown is deliberate:
    `monkeypatch.delenv(..., raising=False)` records nothing when the variable
    is absent, so it has nothing to undo for one the test then creates.
    """
    for name in TRACING_VARS + PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Settings that never reach a real service."""
    return Settings(
        GROQ_API="test-groq",
        TAVILY_API="test-tavily",
        PINECONE_API="test-pinecone",
        PINECONE_INDEX="test-index",
        PRIVATE_NAMESPACE="test-hr-docs",
        PUBLIC_NAMESPACE="test-public",
    )


@pytest.fixture
def kb_docs() -> list[Document]:
    return [kb_doc()]
