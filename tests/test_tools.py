"""The tool registry's guarantees, each tested as the failure it prevents."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from conftest import settings_for
from pydantic import BaseModel, Field

from app.agent.budgets import Budget
from app.agent.tools import (
    CalcArgs,
    SearchArgs,
    ToolError,
    ToolRegistry,
    ToolSpec,
    register_builtins,
    safe_eval,
)
from app.core.tracing import Tracer
from app.services.idempotency import IdempotencyStore, fingerprint


class EchoArgs(BaseModel):
    text: str = Field(default="hello", max_length=100)


class TicketArgs(BaseModel):
    summary: str = Field(min_length=1, max_length=200)


class FakeApprovals:
    """An approval queue that records what it was asked and answers with an id.

    `approval_id=None` is the interesting setting: it is what a real store
    returns when the write failed, and the registry has to refuse rather than
    run.
    """

    def __init__(self, approval_id: int | None = 7) -> None:
        self.approval_id = approval_id
        self.requests: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> int | None:
        self.requests.append(kwargs)
        return self.approval_id


def fast_settings(**overrides: Any) -> Any:
    """Settings with no backoff, so a retry test does not sleep for a second."""
    base = {"TOOL_BACKOFF_SECONDS": 0.0}
    base.update(overrides)
    return settings_for(**base)


def registry(**kwargs: Any) -> ToolRegistry:
    kwargs.setdefault("settings", fast_settings())
    return ToolRegistry(**kwargs)


def spec(handler: Any, name: str = "echo", **kwargs: Any) -> ToolSpec:
    fields = {
        "name": name,
        "description": "An echo tool.",
        "parameters": EchoArgs,
        "handler": handler,
    }
    fields.update(kwargs)
    return ToolSpec(**fields)  # type: ignore[arg-type]


# --- Unknown tools and arguments --------------------------------------------


def test_an_unknown_tool_name_answers_with_the_names_that_do_exist():
    reg = registry()
    reg.register(spec(lambda text: text, name="kb_search"))

    result = reg.call("raise_ticket", {})

    assert result.ok is False
    assert "no such tool" in result.error
    assert "kb_search" in result.error


def test_invalid_arguments_name_the_field_and_never_echo_its_value():
    secret = "gsk_live_" + "A" * 600
    reg = registry()
    reg.register(
        spec(lambda query, k: query, name="kb_search", parameters=SearchArgs)
    )

    result = reg.call("kb_search", {"query": secret})

    assert result.ok is False
    assert "query" in result.error
    assert secret not in result.error
    assert "gsk_live_" not in result.error


# --- Timeouts and retries ----------------------------------------------------


def test_a_tool_over_its_timeout_fails_without_blocking_the_caller():
    def slow(text: str) -> str:
        time.sleep(0.5)
        return text

    reg = registry()
    reg.register(spec(slow, name="slow", timeout_seconds=0.05))

    started = time.perf_counter()
    result = reg.call("slow", {})
    waited = time.perf_counter() - started

    assert result.ok is False
    assert "did not answer" in result.error
    assert waited < 0.4


def test_a_flaky_idempotent_tool_succeeds_on_a_retry_and_reports_its_attempts():
    attempts: list[int] = []

    def flaky(text: str) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("401 from https://api.groq.com key=gsk_live_SECRET")
        return "third time"

    reg = registry()
    reg.register(spec(flaky, name="flaky", retries=2, idempotent=True))

    result = reg.call("flaky", {})

    assert result.ok is True
    assert result.value == "third time"
    assert result.attempts == 3


def test_a_non_idempotent_tool_is_not_retried():
    attempts: list[int] = []

    def flaky(text: str) -> str:
        attempts.append(1)
        raise RuntimeError("boom")

    reg = registry()
    reg.register(spec(flaky, name="flaky", retries=2, idempotent=False))

    result = reg.call("flaky", {})

    assert result.ok is False
    assert result.attempts == 1
    assert len(attempts) == 1


def test_a_provider_error_never_reaches_the_tool_result():
    def leaky(text: str) -> str:
        raise RuntimeError("401 from https://api.groq.com key=gsk_live_SECRET")

    reg = registry()
    reg.register(spec(leaky, name="leaky"))

    result = reg.call("leaky", {})

    assert "gsk_live_SECRET" not in result.error
    assert "api.groq.com" not in result.error
    assert "RuntimeError" in result.error


def test_a_tool_error_is_shown_because_we_wrote_it():
    def refuses(text: str) -> str:
        raise ToolError("no policy document covers contractors")

    reg = registry()
    reg.register(spec(refuses, name="refuses"))

    assert "contractors" in reg.call("refuses", {}).error


# --- Budget ------------------------------------------------------------------


def test_the_budget_is_charged_once_however_many_attempts_it_took():
    attempts: list[int] = []

    def flaky(text: str) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    budget = Budget()
    reg = registry(budget=budget)
    reg.register(spec(flaky, name="flaky", retries=2))

    reg.call("flaky", {})

    assert len(attempts) == 3
    assert budget.tool_calls == 1


def test_a_spent_budget_refuses_without_calling_the_handler():
    calls: list[int] = []
    budget = Budget(max_tool_calls=1).charge_tool()
    reg = registry(budget=budget)
    reg.register(spec(lambda text: calls.append(1), name="echo"))

    result = reg.call("echo", {})

    assert result.ok is False
    assert calls == []
    assert "budget" in result.error


# --- Authorization -----------------------------------------------------------


def test_an_unauthorised_call_does_not_run_and_does_not_say_what_it_would_do():
    calls: list[int] = []
    reg = registry(authorize=lambda principal, action: action != "ticket.raise")
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a Workday ticket containing the employee's salary.",
            parameters=TicketArgs,
            handler=lambda summary: calls.append(1),
            requires="ticket.raise",
        )
    )

    result = reg.call("raise_ticket", {"summary": "pay query"})

    assert result.denied is True
    assert calls == []
    assert "ticket.raise" in result.error
    assert "Workday" not in result.error
    assert "salary" not in result.error


def test_an_unauthorised_tool_is_left_out_of_the_catalogue():
    reg = registry(authorize=lambda principal, action: action != "ticket.raise")
    reg.register(spec(lambda text: text, name="echo"))
    reg.register(
        spec(
            lambda text: text,
            name="raise_ticket",
            requires="ticket.raise",
            parameters=EchoArgs,
        )
    )

    assert reg.names() == ["echo"]
    assert "raise_ticket" not in reg.render_catalogue()


# --- Approval ----------------------------------------------------------------


def test_a_tool_needing_approval_records_a_request_and_does_not_run():
    calls: list[int] = []
    approvals = FakeApprovals(approval_id=11)
    reg = registry(approvals=approvals)
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=lambda summary: calls.append(1),
            needs_approval=True,
            side_effecting=True,
        )
    )

    result = reg.call(
        "raise_ticket", {"summary": "pay query"}, question="when am I paid?"
    )

    assert calls == []
    assert result.needs_approval is True
    assert result.approval_id == 11
    assert approvals.requests[0]["action"] == "raise_ticket"
    assert approvals.requests[0]["payload"] == {"summary": "pay query"}


def test_an_approval_store_that_records_nothing_refuses_rather_than_running():
    calls: list[int] = []
    reg = registry(approvals=FakeApprovals(approval_id=None))
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=lambda summary: calls.append(1),
            needs_approval=True,
        )
    )

    result = reg.call("raise_ticket", {"summary": "pay query"})

    assert calls == []
    assert result.ok is False
    assert result.needs_approval is False
    assert "could not be recorded" in result.error


# --- Idempotency -------------------------------------------------------------


def test_a_side_effecting_tool_called_twice_runs_once_and_replays():
    calls: list[int] = []

    def raise_ticket(summary: str) -> dict[str, str]:
        calls.append(1)
        return {"ticket": f"HR-{len(calls)}"}

    reg = registry(idempotency=IdempotencyStore())
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=raise_ticket,
            side_effecting=True,
        )
    )

    first = reg.call("raise_ticket", {"summary": "pay query"})
    second = reg.call("raise_ticket", {"summary": "pay query"})

    assert len(calls) == 1
    assert first.replayed is False
    assert second.replayed is True
    assert second.ok is True
    assert second.value == first.value == {"ticket": "HR-1"}


def test_a_claim_already_in_flight_refuses_rather_than_racing_it():
    calls: list[int] = []
    store = IdempotencyStore()
    store.claim(fingerprint("raise_ticket", {"summary": "pay query"}))
    reg = registry(idempotency=store)
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=lambda summary: calls.append(1),
            side_effecting=True,
        )
    )

    result = reg.call("raise_ticket", {"summary": "pay query"})

    assert calls == []
    assert result.ok is False
    assert "already running" in result.error


def test_different_arguments_are_a_different_claim():
    calls: list[str] = []
    reg = registry(idempotency=IdempotencyStore())
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=lambda summary: calls.append(summary) or {"ok": summary},
            side_effecting=True,
        )
    )

    reg.call("raise_ticket", {"summary": "pay"})
    reg.call("raise_ticket", {"summary": "leave"})

    assert calls == ["pay", "leave"]


# --- The ledger itself -------------------------------------------------------
# In-memory above, on disk here: the SQLite path is the one a deployment runs,
# and the two are only interchangeable if both are exercised.


def test_a_fingerprint_does_not_depend_on_the_order_the_keys_were_written():
    assert fingerprint("kb_search", {"k": 4, "query": "pto"}) == fingerprint(
        "kb_search", {"query": "pto", "k": 4}
    )


def test_the_sqlite_ledger_replays_a_completed_claim(tmp_path):
    store = IdempotencyStore(db_path=tmp_path / "state.db")
    key = fingerprint("raise_ticket", {"summary": "pay"})

    assert store.claim(key, action="raise_ticket").is_new is True
    store.complete(key, {"ticket": "HR-1"})
    replay = store.claim(key)

    assert replay.is_replay is True
    assert replay.result == {"ticket": "HR-1"}
    assert store.lookup(key) == {"ticket": "HR-1"}
    assert store.stats()["backend"] == "sqlite"


def test_a_released_claim_can_be_won_again(tmp_path):
    store = IdempotencyStore(db_path=tmp_path / "state.db")
    key = fingerprint("raise_ticket", {"summary": "pay"})

    store.claim(key)
    store.release(key)

    assert store.claim(key).is_new is True


def test_an_abandoned_claim_is_re_won_rather_than_blocking_for_the_whole_ttl(
    tmp_path,
):
    store = IdempotencyStore(db_path=tmp_path / "state.db", stall_seconds=0.0)
    key = fingerprint("raise_ticket", {"summary": "pay"})

    store.claim(key)

    assert store.claim(key).is_new is True


def test_only_one_of_two_concurrent_claims_wins(tmp_path):
    store = IdempotencyStore(db_path=tmp_path / "state.db")
    key = fingerprint("raise_ticket", {"summary": "pay"})
    outcomes: list[str] = []
    ready = threading.Barrier(4)

    def contend() -> None:
        ready.wait()
        outcomes.append(store.claim(key).state)

    threads = [threading.Thread(target=contend) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("won") == 1
    assert len(outcomes) == 4


def test_an_expired_claim_is_purged(tmp_path):
    store = IdempotencyStore(db_path=tmp_path / "state.db", ttl_seconds=-1)
    store.claim(fingerprint("raise_ticket", {"summary": "pay"}))

    assert store.purge_expired() == 1
    assert store.stats()["entries"] == 0


# --- Parallel fan-out --------------------------------------------------------


def test_call_parallel_answers_in_call_order_not_completion_order():
    def make(delay: float) -> Any:
        def handler(text: str) -> str:
            time.sleep(delay)
            return text

        return handler

    reg = registry()
    for name, delay in (("slowest", 0.15), ("middle", 0.07), ("quickest", 0.0)):
        reg.register(spec(make(delay), name=name))

    results = reg.call_parallel(
        [("slowest", {}), ("middle", {}), ("quickest", {})]
    )

    assert [result.name for result in results] == ["slowest", "middle", "quickest"]
    assert all(result.ok for result in results)


def test_one_failing_call_does_not_lose_the_others():
    def boom(text: str) -> str:
        raise ValueError("boom")

    reg = registry()
    reg.register(spec(boom, name="boom"))
    reg.register(spec(lambda text: "fine", name="fine"))

    results = reg.call_parallel([("boom", {}), ("fine", {}), ("missing", {})])

    assert results[0].ok is False
    assert results[1].ok is True
    assert results[2].ok is False
    assert "no such tool" in results[2].error


def test_call_parallel_refuses_to_parallelise_a_side_effecting_tool():
    window: list[tuple[str, float, float]] = []
    lock = threading.Lock()

    def make(name: str) -> Any:
        def handler(summary: str) -> str:
            entered = time.perf_counter()
            time.sleep(0.1)
            with lock:
                window.append((name, entered, time.perf_counter()))
            return name

        return handler

    tracer = Tracer(enabled=True)
    reg = registry(tracer=tracer)
    for name in ("first", "second"):
        reg.register(
            ToolSpec(
                name=name,
                description="A side effect.",
                parameters=TicketArgs,
                handler=make(name),
                side_effecting=True,
            )
        )

    reg.call_parallel([("first", {"summary": "a"}), ("second", {"summary": "b"})])

    assert len(window) == 2
    first, second = sorted(window, key=lambda row: row[1])
    assert second[1] >= first[2]
    sequential = [s for s in tracer.to_list() if s["name"] == "tool.sequential"]
    assert len(sequential) == 2


# --- Tracing -----------------------------------------------------------------


def test_a_call_records_a_tool_span_with_its_outcome():
    tracer = Tracer(enabled=True)
    reg = register_builtins(registry(tracer=tracer))

    reg.call("calculate", {"expression": "12 - 5"})

    spans = [s for s in tracer.to_list() if s["name"] == "tool.calculate"]
    assert len(spans) == 1
    assert spans[0]["kind"] == "tool"
    assert spans[0]["attributes"]["ok"] is True
    assert spans[0]["attributes"]["attempts"] == 1


def test_a_swallowed_failure_is_recorded_as_degraded():
    tracer = Tracer(enabled=True)
    reg = registry(tracer=tracer)
    reg.register(spec(lambda text: 1 / 0, name="broken"))

    reg.call("broken", {})

    assert any(span["status"] == "degraded" for span in tracer.to_list())


# --- The catalogue -----------------------------------------------------------


def test_a_duplicate_registration_is_refused():
    reg = registry()
    reg.register(spec(lambda text: text, name="echo"))

    with pytest.raises(ValueError, match="already registered"):
        reg.register(spec(lambda text: text, name="echo"))


def test_the_rendered_catalogue_flags_a_tool_that_needs_approval():
    reg = registry()
    reg.register(
        ToolSpec(
            name="raise_ticket",
            description="Raise a ticket.",
            parameters=TicketArgs,
            handler=lambda summary: summary,
            needs_approval=True,
            side_effecting=True,
        )
    )

    line = reg.render_catalogue()

    assert "raise_ticket(summary: string)" in line
    assert "needs human approval" in line


def test_the_decorator_form_registers_the_handler():
    reg = registry()

    @reg.tool("echo", "Say it back.", EchoArgs)
    def echo(text: str) -> str:
        return text

    assert reg.call("echo", {"text": "hi"}).value == "hi"


# --- safe_eval ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12 - 5", 7.0),
        ("(2 + 3) * 4", 20.0),
        ("7 / 2", 3.5),
        ("20 - 12 % 5", 18.0),
        ("-4 + 10", 6.0),
        ("2 ** 10", 1024.0),
        ("17 // 5", 3.0),
    ],
)
def test_safe_eval_computes_the_arithmetic_an_hr_question_contains(
    expression, expected
):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "days_left + 1",
        "len([1, 2])",
        "().__class__.__bases__",
        "__import__('os').listdir('.')",
        "2 ** 10000",
        "(10 ** 60) ** 60",
        "[x for x in range(3)]",
        "'a' * 1000000",
        "1 / 0",
        "lambda: 1",
    ],
)
def test_safe_eval_refuses_everything_that_is_not_arithmetic(expression):
    with pytest.raises(ToolError):
        safe_eval(expression)


def test_the_calculator_tool_renders_its_working():
    reg = register_builtins(registry())

    result = reg.call("calculate", {"expression": "12 - 5"})

    assert result.ok is True
    assert result.value["result"] == 7.0
    assert "12 - 5 = 7" in result.observation()


def test_the_calculator_refuses_an_expression_longer_than_its_field_allows():
    reg = register_builtins(registry())

    result = reg.call("calculate", {"expression": "1 + " * 60 + "1"})

    assert result.ok is False
    assert "expression" in result.error


def test_the_calculator_parameter_model_rejects_an_empty_expression():
    with pytest.raises(ToolError):
        safe_eval("")
    assert CalcArgs.model_json_schema()["properties"]["expression"]["minLength"] == 1


# --- Counters ----------------------------------------------------------------


def test_stats_count_what_the_registry_did():
    reg = register_builtins(registry())
    reg.call("calculate", {"expression": "1 + 1"})
    reg.call("nope", {})

    stats = reg.stats()

    assert stats["registered"] == 1
    assert stats["ok"] == 1
    assert stats["unknown"] == 1
