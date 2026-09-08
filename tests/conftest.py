"""Fakes for every client the agent talks to.

The whole suite runs offline: no Groq, no Pinecone, no Tavily, no embedding
model download. That is only possible because `build_nodes`, `build_graph` and
`Copilot` take their dependencies as arguments -- the fakes below are the reason
that injection exists.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
        self.calls: List[str] = []

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
        self.queries: List[str] = []

    def invoke(self, query: str, *args: Any, **kwargs: Any) -> List[Document]:
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
        self.queries: List[str] = []

    def invoke(self, payload: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
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
def kb_docs() -> List[Document]:
    return [kb_doc()]
