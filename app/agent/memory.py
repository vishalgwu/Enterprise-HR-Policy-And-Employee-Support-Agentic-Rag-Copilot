"""The short-term window: the conversation as the agent sees it.

Agent-layer, so it imports nothing from `app.services`. The long-term fact
store is a *service* and arrives injected and duck-typed -- anything with
`.render(subject)` and `.recall(subject)` will do, which is what keeps
`app.services.memory_store` on the other side of the layering boundary and lets
a test pass an object with two methods.

Validation is not repeated here. `normalise_history` in `app.agent.schemas`
already drops the malformed turns a browser can send -- an unrecognised role (a
`system` turn is the obvious thing to try), a missing or non-string `content` --
and it is the same function `initial_state` uses, so the graph and this window
agree about what a turn is. A second validator would be a second opinion.

**`max_chars` is the real control; `max_turns` is the cheap one.** Eight short
turns and eight 3000-character turns cost very different amounts, and the whole
history is rendered into the contextualise prompt on *every* follow-up -- so an
unbounded window makes a conversation more expensive the longer it runs, and
the turn where somebody pasted a policy document into the chat is the turn that
makes every later one expensive. Counting turns is free and catches the ordinary
case; counting characters is what actually bounds the bill.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from app.agent.prompts import HISTORY_SUMMARY_PROMPT
from app.agent.schemas import normalise_history
from app.core.config import get_logger
from app.core.tracing import Tracer, null_tracer
from app.rag.clients import message_text

log = get_logger("memory")

__all__ = ["ConversationMemory", "compact", "remembered"]

# How many turns at the end of the window are never summarised. Two exchanges:
# far enough back for "it" and "that" to still resolve, close enough that
# compaction leaves room for three more before anything is evicted.
KEEP_RECENT = 4

_SUMMARY_PREFIX = "Earlier in this conversation: "


@dataclass(slots=True)
class ConversationMemory:
    """The windowed conversation, with a token-aware budget.

    Immutable in use: `add` returns a new window rather than mutating this one,
    so a node holding a memory cannot have it changed underneath by a sibling.
    """

    turns: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 8
    max_chars: int = 6000

    def __post_init__(self) -> None:
        self.turns = self._trimmed(self.turns)

    @classmethod
    def from_history(
        cls,
        history: Sequence[Mapping[str, Any]] | None,
        *,
        max_turns: int = 8,
        max_chars: int = 6000,
    ) -> ConversationMemory:
        """Build a window from whatever the browser posted.

        `normalise_history` does the dropping and the turn cap; the character
        cap is applied here because it depends on this window's own budget.
        """
        return cls(
            turns=normalise_history(history, limit=max_turns),
            max_turns=max_turns,
            max_chars=max_chars,
        )

    # --- Shape ---------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.turns

    @property
    def chars(self) -> int:
        return sum(len(turn["content"]) for turn in self.turns)

    def add(self, role: str, content: str) -> ConversationMemory:
        """Append one turn and re-apply the window. Returns the new window."""
        turn = normalise_history([{"role": role, "content": content}], limit=1)
        if not turn:
            return self
        return replace(self, turns=self._trimmed(self.turns + turn))

    def as_history(self) -> list[dict[str, str]]:
        """The turns, as the shape `/api/chat` and `initial_state` take."""
        return [dict(turn) for turn in self.turns]

    def render(self) -> str:
        """The window as a transcript, oldest first. "" when empty.

        Exactly the wording `contextualize` builds in `app.agent.nodes`, so a
        prompt calibrated against one reads the same text from the other.
        """
        return "\n".join(
            f"{'Employee' if turn['role'] == 'user' else 'Assistant'}: "
            f"{turn['content']}"
            for turn in self.turns
        )

    def summary_needed(self) -> bool:
        """Whether the next turn will start throwing information away.

        True when the window is at capacity: one more turn now evicts the
        oldest one outright, and summarising it first keeps something of it.
        """
        return len(self.turns) >= self.max_turns or self.chars > self.max_chars

    # --- Internals -----------------------------------------------------------
    def _trimmed(self, turns: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop the oldest turns until the window fits both ceilings.

        Never drops the last turn, however long it is: a window holding nothing
        is worse than a window holding one oversized thing, and the renderer is
        where an oversized single turn gets dealt with.
        """
        kept = list(turns[-self.max_turns :]) if self.max_turns > 0 else list(turns)
        while len(kept) > 1 and sum(len(t["content"]) for t in kept) > self.max_chars:
            kept.pop(0)
        return kept


def compact(
    memory: ConversationMemory,
    llm: Any = None,
    *,
    tracer: Tracer | None = None,
    budget: Any = None,
) -> ConversationMemory:
    """Replace the oldest turns with one summarised turn, when the window is full.

    **Summarisation loses the thing history is for.** The one job the
    conversation does in this system is resolving a pronoun: turning "what
    about contractors?" into a question that stands alone. A summarised turn
    cannot do that with precision -- "the employee asked about leave" does not
    tell you whether "it" was the carry-over cap or the expiry date. So
    compaction never touches the recent window: the last `KEEP_RECENT` turns
    survive verbatim, and only what has already fallen out of pronoun range is
    condensed.

    Degrades to plain truncation on any failure, and truncation keeps the
    recent turns. A failed summary must cost the conversation its *oldest*
    turns, never its newest -- the opposite would turn a provider outage into a
    follow-up that cannot resolve.
    """
    if not memory.summary_needed() or len(memory.turns) <= KEEP_RECENT:
        return memory

    older = memory.turns[:-KEEP_RECENT]
    recent = memory.turns[-KEEP_RECENT:]
    truncated = replace(memory, turns=list(recent))

    if llm is None:
        return truncated
    if budget is not None and not budget.allows("llm_calls"):
        return truncated

    transcript = replace(memory, turns=list(older)).render()
    tracer = tracer if tracer is not None else null_tracer()
    try:
        with tracer.span("memory.compact", kind="llm", turns=len(older)):
            summary = message_text(
                (HISTORY_SUMMARY_PROMPT | llm).invoke({"transcript": transcript})
            ).strip()
    except Exception:  # noqa: BLE001 - a lost summary must not cost the window
        log.exception("History compaction failed; truncating instead")
        tracer.degraded("memory.compact", reason="summary failed", kind="llm")
        return truncated

    if budget is not None:
        budget.charge_llm()
    if not summary:
        return truncated
    summarised = {"role": "assistant", "content": _SUMMARY_PREFIX + summary}
    return replace(memory, turns=[summarised, *recent])


def remembered(store: Any, subject: str | None) -> str:
    """The long-term block for a prompt, or "" when there is nothing to add.

    The duck-typed seam: `store` is anything with `.render(subject)`, which in
    the application is `app.services.memory_store.FactMemory` and in a test is
    an object with one method. Never raises -- a fact store that is down is an
    employee who gets asked their department again, which is how every
    conversation starts anyway.
    """
    if store is None or not subject:
        return ""
    try:
        return str(store.render(subject) or "")
    except Exception:  # noqa: BLE001 - no prompt is worth a 500
        log.exception("Recalling long-term facts failed; continuing without them")
        return ""
