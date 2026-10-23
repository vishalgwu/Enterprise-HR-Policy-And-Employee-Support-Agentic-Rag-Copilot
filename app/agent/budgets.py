"""Loop budgets: the thing that stops an agent costing more than the answer.

A graph with a bounded rewrite loop needs one counter. An agent that chooses its
own tools needs a real budget, because the failure mode changes shape: a graph
that loops forever is a bug you can see in the wiring, while an agent that loops
forever is a model deciding, each time, that one more search would help. Nothing
in the wiring is wrong. The only thing that stops it is a ceiling.

Six ceilings, because they fail differently and an agent can hit any one of
them first:

    steps        how many times the agent gets to decide. Bounds the loop.
    tool_calls   how much work it gets to do. Bounds fan-out and retries.
    llm_calls    how many model round trips. Bounds a self-chat.
    tokens       how much context it gets to burn. Bounds prompt growth.
    seconds      wall clock. Bounds a provider that is merely slow.
    cost         projected dollars. Bounds the one an operator is asked about.

**A budget is a stopping condition, not an error.** The agent asks `allows()`
before it spends and composes a best-effort answer when the answer is no; it
does not catch `BudgetExceeded` on the main path. That distinction is the whole
design: an employee asking about parental leave should get the partial answer the
agent had assembled, labelled as partial, rather than a 500 that says
"max_steps". `charge()` raising exists for the inner call -- a tool, a worker --
that genuinely has nowhere to degrade to and must unwind to a caller that does.

**It is one object, shared and mutated.** The supervisor, every worker and every
tool charge the *same* budget, which is what makes "the run cost 12 calls" a
fact rather than six separate opinions. It is therefore thread-safe: parallel
tool calls charge concurrently, and a budget that undercounts under concurrency
is a budget that does not bound anything.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Budget",
    "BudgetExceeded",
    "BudgetKind",
    "DEFAULT_BUDGET",
]

BudgetKind = Literal["steps", "tool_calls", "llm_calls", "tokens", "seconds", "cost"]

# Human wording per limit, kept beside the field names so a message and a
# counter cannot disagree about what ran out.
_LIMIT_WORDING: dict[str, str] = {
    "steps": "reasoning steps",
    "tool_calls": "tool calls",
    "llm_calls": "model calls",
    "tokens": "tokens",
    "seconds": "seconds of wall clock",
    "cost": "US dollars",
}


class BudgetExceeded(RuntimeError):
    """Raised by `charge()` when a limit is already spent.

    Carries the three facts a caller needs to say something useful: which
    ceiling, what it was, and what had been used. A handler that only has the
    message has to parse prose to decide whether to degrade or to give up.
    """

    def __init__(self, kind: str, limit: float, used: float) -> None:
        self.kind = kind
        self.limit = limit
        self.used = used
        noun = _LIMIT_WORDING.get(kind, kind)
        super().__init__(
            f"Budget exhausted: {used:g} of {limit:g} {noun} used. "
            f"The run must stop and answer with what it has."
        )


@dataclass
class Budget:
    """What one run is allowed to spend, and what it has spent.

    Mutable and shared, deliberately -- see the module docstring. Construct one
    per request (`Budget.from_settings(settings)`), hand the same object to
    every component, and read `report()` at the end.

    `max_cost_usd` is a *projection*, not a bill. It is charged from a token
    count multiplied by a published per-million price, which is the only cost
    number available before a provider's invoice arrives a month later. It is
    right to about the accuracy of the token estimate, which is good enough to
    stop a runaway and not good enough to reconcile accounts; `app.services.
    routing` says so where the prices live.
    """

    max_steps: int = 8
    max_tool_calls: int = 12
    max_llm_calls: int = 16
    max_tokens: int = 120_000
    max_seconds: float = 45.0
    max_cost_usd: float = 0.25

    steps: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    # Set on the first charge or by `start()`, never in __init__: a budget built
    # at import or held on a settings object would start its clock then, and
    # every later run would look as if it had already spent an hour.
    started_at: float | None = None
    # Why the run stopped, in the agent's own words, for the trace and the UI.
    stop_reason: str | None = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # --- Lifecycle -----------------------------------------------------------
    def start(self) -> Budget:
        """Begin the clock. Idempotent, and returns self so it chains.

        Idempotent because several components legitimately call it -- the
        supervisor, then the first worker -- and the second call must not reset
        the deadline to a later one, which would make the wall-clock ceiling
        unbounded in exactly the run that needed it.
        """
        with self._lock:
            if self.started_at is None:
                self.started_at = time.monotonic()
        return self

    @classmethod
    def from_settings(cls, settings: Any) -> Budget:
        """A budget from configuration, for one request.

        Takes `Any` rather than `Settings` so this module keeps its place in the
        layering: budgets are agent-layer and must not import the whole config
        module to read six numbers off a duck-typed object.
        """
        return cls(
            max_steps=int(getattr(settings, "max_agent_steps", 8)),
            max_tool_calls=int(getattr(settings, "max_tool_calls", 12)),
            max_llm_calls=int(getattr(settings, "max_llm_calls", 16)),
            max_tokens=int(getattr(settings, "max_agent_tokens", 120_000)),
            max_seconds=float(getattr(settings, "max_agent_seconds", 45.0)),
            max_cost_usd=float(getattr(settings, "max_agent_cost_usd", 0.25)),
        )

    # --- Querying ------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def _used(self, kind: str) -> float:
        if kind == "seconds":
            return self.elapsed_seconds
        if kind == "cost":
            return self.cost_usd
        return float(getattr(self, kind, 0))

    def _limit(self, kind: str) -> float:
        if kind == "cost":
            return self.max_cost_usd
        return float(getattr(self, f"max_{kind}", 0))

    def exceeded(self) -> str | None:
        """The first ceiling that is spent, or None.

        Order is deliberate: wall clock first, because a run that has already
        taken 45 seconds should say so rather than report a step count that
        happens to also be full. After that, cheapest signal to most expensive.
        """
        for kind in ("seconds", "cost", "tokens", "llm_calls", "tool_calls", "steps"):
            limit = self._limit(kind)
            if limit > 0 and self._used(kind) >= limit:
                return kind
        return None

    def allows(self, kind: BudgetKind = "steps", amount: float = 1) -> bool:
        """Whether `amount` more of `kind` fits. The graceful-degradation call.

        This is what an agent loop asks. It records `stop_reason` as a side
        effect when the answer is no, so the caller does not have to work out
        what to tell the user -- and so a run that stopped early always carries
        its reason into the trace and the audit row.
        """
        limit = self._limit(kind)
        if limit <= 0:
            return True
        with self._lock:
            if self._used(kind) + amount > limit:
                if self.stop_reason is None:
                    noun = _LIMIT_WORDING.get(kind, kind)
                    self.stop_reason = (
                        f"stopped early: the {noun} budget "
                        f"({limit:g}) was reached"
                    )
                return False
        # A different ceiling may already be spent even when this one fits.
        spent = self.exceeded()
        if spent is not None:
            if self.stop_reason is None:
                noun = _LIMIT_WORDING.get(spent, spent)
                self.stop_reason = (
                    f"stopped early: the {noun} budget "
                    f"({self._limit(spent):g}) was reached"
                )
            return False
        return True

    # --- Spending ------------------------------------------------------------
    def charge(
        self,
        kind: BudgetKind = "steps",
        amount: float = 1,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> Budget:
        """Spend from the budget, raising when a ceiling is already gone.

        Checked *before* adding, so the limit is a ceiling rather than a
        suggestion the last call gets to exceed. `tokens` and `cost_usd` are
        charged alongside whatever `kind` is, because a model call spends three
        things at once and charging them in three calls is three chances to
        forget one.
        """
        self.start()
        with self._lock:
            limit = self._limit(kind)
            if limit > 0 and self._used(kind) >= limit:
                raise BudgetExceeded(kind, limit, self._used(kind))
            if kind in ("steps", "tool_calls", "llm_calls"):
                setattr(self, kind, int(getattr(self, kind)) + int(amount))
            elif kind == "tokens":
                self.tokens += int(amount)
            elif kind == "cost":
                self.cost_usd += float(amount)
            if kind != "tokens":
                self.tokens += int(tokens)
            if kind != "cost":
                self.cost_usd += float(cost_usd)
        return self

    def charge_llm(
        self, tokens: int = 0, cost_usd: float = 0.0, calls: int = 1
    ) -> Budget:
        """One model round trip, with what it burned. Never raises.

        Deliberately forgiving: the call has already happened by the time its
        token count is known, so refusing to record it would make the budget
        lie about a cost that was genuinely incurred. The *next* `allows()` is
        what stops the run.
        """
        self.start()
        with self._lock:
            self.llm_calls += calls
            self.tokens += int(tokens)
            self.cost_usd += float(cost_usd)
        return self

    def charge_tool(self, calls: int = 1) -> Budget:
        """One tool invocation. Never raises, for the reason above."""
        self.start()
        with self._lock:
            self.tool_calls += calls
        return self

    def charge_step(self) -> Budget:
        """One agent decision. Never raises, for the reason above."""
        self.start()
        with self._lock:
            self.steps += 1
        return self

    # --- Reporting -----------------------------------------------------------
    def remaining(self) -> dict[str, float]:
        """What is left of each ceiling, floored at zero."""
        return {
            kind: max(0.0, self._limit(kind) - self._used(kind))
            for kind in ("steps", "tool_calls", "llm_calls", "tokens", "seconds", "cost")
        }

    def report(self) -> dict[str, Any]:
        """What the run spent, for the API response and the audit row.

        `exhausted` and `stop_reason` are the two fields worth reading: together
        they say whether the answer is complete or is the best the agent could
        assemble before it ran out, which is a materially different thing to
        hand an employee and must not be inferable only from a short answer.
        """
        spent = self.exceeded()
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "max_seconds": self.max_seconds,
            "cost_usd": round(self.cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "exhausted": spent,
            "stop_reason": self.stop_reason,
        }


# A module-level default for a caller that wants the shape without the numbers
# -- a docs example, a `--help` string. **Do not charge against it.** It is one
# object, so charging would accumulate across every run in the process and the
# first request after a while would find the budget already spent. Construct one
# per request with `Budget.from_settings(settings)`.
DEFAULT_BUDGET = Budget()
