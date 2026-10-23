"""The ReAct loop, including every path a real question only reaches twice."""

from __future__ import annotations

from typing import Any

from conftest import settings_for
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.agent.budgets import Budget
from app.agent.prompts import (
    BUDGET_STOPPED_NOTE,
    ESCALATED_ANSWER,
    GENERATION_FAILED,
)
from app.agent.react import ReActAgent
from app.agent.schemas import ReActDecision
from app.agent.tools import SearchArgs, ToolRegistry, ToolSpec


class ToolUseFailed(RuntimeError):
    """Groq's `400 tool_use_failed`, with the body the recovery reads.

    Shaped exactly like the real one: `error.failed_generation` holds what the
    model said instead of calling the bound tool.
    """

    def __init__(self, generated: str) -> None:
        super().__init__("400 tool_use_failed")
        self.body = {"error": {"failed_generation": generated}}


class ScriptedLLM(Runnable):
    """Returns scripted `ReActDecision`s, and a fixed reply for composition.

    `errors` is consumed before `decisions`, so a run can be scripted to fail
    its first structured call and succeed on the next -- which is the only way
    to exercise the recovery deterministically. The last decision repeats once
    the list runs out, like `FakeLLM.grades`.
    """

    def __init__(
        self,
        decisions: Any = (),
        reply: str = "Composed from the evidence gathered.",
        errors: Any = (),
        raise_always: bool = False,
    ) -> None:
        self.decisions = list(decisions)
        self.reply = reply
        self.errors = list(errors)
        self.raise_always = raise_always
        self.calls: list[str] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append("invoke")
        if self.raise_always:
            raise RuntimeError("simulated provider outage")
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def decide(_input: Any) -> Any:
            self.calls.append(schema.__name__)
            if self.raise_always:
                raise RuntimeError("simulated provider outage")
            if self.errors:
                raise self.errors.pop(0)
            if not self.decisions:
                return ReActDecision(final_answer="Nothing further.")
            if len(self.decisions) > 1:
                return self.decisions.pop(0)
            return self.decisions[0]

        return RunnableLambda(decide)


class TicketArgs(BaseModel):
    summary: str = Field(min_length=1, max_length=200)


EVIDENCE = {
    "chunks": [],
    "citations": [
        {"source": "leave-and-time-off.md", "title": "Leave And Time Off"}
    ],
    "text": "[1] source=leave-and-time-off.md\nEmployees accrue 20 days of PTO.",
}


def kb_search_spec(handler: Any = None, **kwargs: Any) -> ToolSpec:
    def default(query: str, k: int = 4) -> dict[str, Any]:
        return EVIDENCE

    fields: dict[str, Any] = {
        "name": "kb_search",
        "description": "Search the private HR knowledge base.",
        "parameters": SearchArgs,
        "handler": handler or default,
    }
    fields.update(kwargs)
    return ToolSpec(**fields)


def build(
    decisions: Any = (),
    *,
    tools: Any = (),
    budget: Any = None,
    reply: str = "Composed from the evidence gathered.",
    errors: Any = (),
    raise_always: bool = False,
    authorize: Any = None,
    approvals: Any = None,
    max_steps: int | None = None,
) -> tuple[ReActAgent, ScriptedLLM, ToolRegistry]:
    settings = settings_for(TOOL_BACKOFF_SECONDS=0.0)
    # One budget, shared by the agent and the registry, as a request is wired:
    # a registry with its own would make "the run cost 12 calls" two opinions.
    budget = budget or Budget.from_settings(settings)
    registry = ToolRegistry(
        settings=settings,
        authorize=authorize,
        approvals=approvals,
        budget=budget,
    )
    for spec in tools:
        registry.register(spec)
    llm = ScriptedLLM(
        decisions, reply=reply, errors=errors, raise_always=raise_always
    )
    agent = ReActAgent(
        llm=llm,
        registry=registry,
        settings=settings,
        budget=budget,
        max_steps=max_steps,
    )
    return agent, llm, registry


# --- The ordinary paths ------------------------------------------------------


def test_a_question_the_model_can_answer_outright_takes_one_step():
    agent, llm, _ = build(
        [ReActDecision(thought="a greeting", final_answer="Hello -- ask me anything.")]
    )

    result = agent.run("hi")

    assert result.answer == "Hello -- ask me anything."
    assert len(result.steps) == 1
    assert result.source_used == "direct"
    assert result.stopped_on_budget is False
    assert llm.calls.count("ReActDecision") == 1


def test_a_run_that_searches_then_answers_takes_two_steps():
    calls: list[str] = []

    def search(query: str, k: int = 4) -> dict[str, Any]:
        calls.append(query)
        return EVIDENCE

    agent, _, _ = build(
        [
            ReActDecision(
                thought="I need the leave policy",
                action="kb_search",
                action_input={"query": "pto accrual"},
            ),
            ReActDecision(final_answer="20 days, per leave-and-time-off.md."),
        ],
        tools=[kb_search_spec(search)],
    )

    result = agent.run("how much PTO do I get?")

    assert calls == ["pto accrual"]
    assert len(result.steps) == 2
    assert result.steps[0].action == "kb_search"
    assert "source=leave-and-time-off.md" in result.steps[0].observation
    assert result.answer == "20 days, per leave-and-time-off.md."


def test_an_action_wins_when_the_model_fills_both_fields():
    agent, _, _ = build(
        [
            ReActDecision(
                action="kb_search",
                action_input={"query": "pto"},
                final_answer="Probably 20 days.",
            ),
            ReActDecision(final_answer="20 days, per leave-and-time-off.md."),
        ],
        tools=[kb_search_spec()],
    )

    result = agent.run("how much PTO?")

    assert result.steps[0].action == "kb_search"
    assert result.answer == "20 days, per leave-and-time-off.md."


# --- Repetition --------------------------------------------------------------


def test_a_repeated_identical_tool_call_is_intercepted_and_not_run_again():
    calls: list[str] = []

    def search(query: str, k: int = 4) -> dict[str, Any]:
        calls.append(query)
        return EVIDENCE

    repeat = ReActDecision(
        thought="let me search again",
        action="kb_search",
        action_input={"query": "pto accrual"},
    )
    agent, _, _ = build(
        [repeat, repeat, ReActDecision(final_answer="20 days.")],
        tools=[kb_search_spec(search)],
    )

    result = agent.run("how much PTO?")

    assert calls == ["pto accrual"]
    assert len(result.steps) == 3
    assert "already run" in result.steps[1].observation
    assert result.steps[1].ok is False
    assert result.answer == "20 days."


# --- Budget ------------------------------------------------------------------


def test_a_one_step_budget_forces_a_composed_answer_and_says_it_is_partial():
    budget = Budget(max_steps=1)
    agent, _, _ = build(
        [
            ReActDecision(
                thought="searching", action="kb_search", action_input={"query": "pto"}
            )
        ],
        tools=[kb_search_spec()],
        budget=budget,
        reply="PTO accrues at 1.67 days a month.",
    )

    result = agent.run("how much PTO do I get?")

    assert result.stopped_on_budget is True
    assert result.answer.startswith("PTO accrues at 1.67 days a month.")
    assert result.answer.endswith(BUDGET_STOPPED_NOTE)
    assert result.budget["steps"] == 1


def test_a_run_that_never_decides_to_answer_stops_at_the_step_ceiling():
    agent, _, _ = build(
        [
            ReActDecision(
                action="kb_search", action_input={"query": "pto"}
            )
        ],
        tools=[kb_search_spec()],
        max_steps=2,
    )

    result = agent.run("how much PTO?")

    assert len(result.steps) == 2
    assert result.stopped_on_budget is True
    assert BUDGET_STOPPED_NOTE in result.answer


# --- Structured-output recovery ----------------------------------------------


def test_a_malformed_structured_reply_is_recovered_rather_than_crashing():
    calls: list[str] = []

    def search(query: str, k: int = 4) -> dict[str, Any]:
        calls.append(query)
        return EVIDENCE

    agent, _, _ = build(
        [ReActDecision(final_answer="20 days, per leave-and-time-off.md.")],
        tools=[kb_search_spec(search)],
        errors=[
            ToolUseFailed(
                'Here you go: {"action": "kb_search", '
                '"action_input": {"query": "pto accrual"}}'
            )
        ],
    )

    result = agent.run("how much PTO?")

    assert calls == ["pto accrual"]
    assert result.answer == "20 days, per leave-and-time-off.md."


def test_an_unreadable_first_decision_falls_back_to_a_lookup_not_to_memory():
    calls: list[str] = []

    def search(query: str, k: int = 4) -> dict[str, Any]:
        calls.append(query)
        return EVIDENCE

    agent, _, _ = build(
        [ReActDecision(final_answer="20 days.")],
        tools=[kb_search_spec(search)],
        errors=[RuntimeError("unparseable")],
    )

    result = agent.run("how much PTO do I get?")

    assert calls == ["how much PTO do I get?"]
    assert result.source_used == "private_kb"


# --- Approval and refusal ----------------------------------------------------


def test_a_tool_that_needs_approval_ends_the_run_as_escalated():
    calls: list[int] = []

    class Approvals:
        def request(self, **kwargs: Any) -> int:
            return 3

    agent, _, _ = build(
        [
            ReActDecision(
                action="raise_ticket", action_input={"summary": "pay query"}
            ),
            ReActDecision(final_answer="should never be reached"),
        ],
        tools=[
            ToolSpec(
                name="raise_ticket",
                description="Raise an HR ticket.",
                parameters=TicketArgs,
                handler=lambda summary: calls.append(1),
                needs_approval=True,
                side_effecting=True,
            )
        ],
        approvals=Approvals(),
    )

    result = agent.run("my pay is wrong")

    assert calls == []
    assert result.answer == ESCALATED_ANSWER
    assert result.source_used == "escalated"
    assert len(result.steps) == 1


def test_a_denied_tool_becomes_an_observation_and_the_loop_continues():
    calls: list[int] = []

    agent, _, _ = build(
        [
            ReActDecision(
                action="raise_ticket", action_input={"summary": "pay query"}
            ),
            ReActDecision(final_answer="I cannot raise that for you."),
        ],
        tools=[
            kb_search_spec(),
            ToolSpec(
                name="raise_ticket",
                description="Raise an HR ticket in Workday.",
                parameters=TicketArgs,
                handler=lambda summary: calls.append(1),
                requires="ticket.raise",
            ),
        ],
        authorize=lambda principal, action: action != "ticket.raise",
    )

    result = agent.run("raise a ticket for me")

    assert calls == []
    assert len(result.steps) == 2
    assert "not available" in result.steps[0].observation
    assert "Workday" not in result.steps[0].observation
    assert result.answer == "I cannot raise that for you."


# --- Citations and provenance ------------------------------------------------


def test_citations_are_collected_from_a_search_result():
    agent, _, _ = build(
        [
            ReActDecision(action="kb_search", action_input={"query": "pto"}),
            ReActDecision(final_answer="20 days."),
        ],
        tools=[kb_search_spec()],
    )

    result = agent.run("how much PTO?")

    assert [cite["source"] for cite in result.citations] == [
        "leave-and-time-off.md"
    ]


def test_citations_are_read_from_bare_documents_too():
    def search(query: str, k: int = 4) -> dict[str, Any]:
        return {"chunks": [{"source": "payroll.md", "title": "Payroll"}]}

    agent, _, _ = build(
        [
            ReActDecision(action="kb_search", action_input={"query": "pay"}),
            ReActDecision(final_answer="On the 25th."),
        ],
        tools=[kb_search_spec(search)],
    )

    assert agent.run("when am I paid?").citations[0]["source"] == "payroll.md"


def test_source_used_reflects_the_tools_that_ran_not_the_model_s_claim():
    agent, _, _ = build(
        [
            ReActDecision(action="kb_search", action_input={"query": "pto"}),
            ReActDecision(
                final_answer="According to a public web article, 20 days."
            ),
        ],
        tools=[kb_search_spec()],
    )

    assert agent.run("how much PTO?").source_used == "private_kb"


def test_a_search_that_returns_nothing_reports_insufficient_evidence():
    def search(query: str, k: int = 4) -> dict[str, Any]:
        raise RuntimeError("simulated Pinecone outage")

    agent, _, _ = build(
        [
            ReActDecision(action="kb_search", action_input={"query": "pto"}),
            ReActDecision(final_answer="I could not find that."),
        ],
        tools=[kb_search_spec(search)],
    )

    result = agent.run("how much PTO?")

    assert result.steps[0].ok is False
    assert result.source_used == "insufficient_evidence"


# --- Total failure -----------------------------------------------------------


def test_a_model_that_always_raises_reports_an_error_rather_than_raising():
    agent, _, _ = build(raise_always=True)

    result = agent.run("how much PTO do I get?")

    assert result.answer == GENERATION_FAILED
    assert result.source_used == "error"
    assert result.stopped_on_budget is False


def test_the_result_serialises_for_the_api():
    agent, _, _ = build(
        [
            ReActDecision(
                thought="searching", action="kb_search", action_input={"query": "pto"}
            ),
            ReActDecision(final_answer="20 days."),
        ],
        tools=[kb_search_spec()],
    )

    data = agent.run("how much PTO?").to_dict()

    assert data["source_used"] == "private_kb"
    assert data["steps"][0]["action"] == "kb_search"
    assert data["budget"]["tool_calls"] == 1
    assert isinstance(data["trace"], list)
