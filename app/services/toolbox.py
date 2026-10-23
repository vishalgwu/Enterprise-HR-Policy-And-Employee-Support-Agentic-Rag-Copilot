"""The tools the agent actually gets, assembled from real services.

`app/agent/tools.py` is the machinery -- validation, timeouts, retries,
authorization, the idempotency ledger, parallel execution. It deliberately
knows nothing about this product: it may not import `app/services/*`, because
the agent layer sits above the service layer and a registry that reached down
for a retriever would invert that.

So the registry is *populated* here, where the services live. That is the whole
job of this module, and it is why there is a module rather than ten lines in
`create_app`: the names, the authorization actions and the return shapes are a
contract with two other files, and a contract spread across a factory function
is one nobody can read.

**Three of these names are load-bearing.** `app/agent/react.py` derives
`source_used` from which tools *succeeded* rather than from what the model
claims, and it does that by name: `kb_search` means the answer rests on
internal policy, `web_search` means it rests on a public page. A private-KB
search registered under any other name produces an answer that is grounded in
company policy and reports itself as having been answered from memory -- which
is the one claim this product must never get wrong. `KB_TOOLS` and `WEB_TOOLS`
in that module are the lists; these names are in them.

**A search tool returns a dict, not Documents.** `{"chunks", "citations",
"text"}`: `text` is what the model reads next turn, `citations` is what
accumulates onto the answer, `chunks` is what a caller wanting the Documents
back uses. Rendering the evidence here rather than in the agent is deliberate --
`format_documents` is the exact format the grader prompt was calibrated
against, and a second rendering of evidence is a second thing to keep in step
with that calibration.

**What is side-effecting, and what only looks it.** Writing a scratchpad note
is not: it is append-only working state scoped to one conversation, a duplicate
costs a repeated line, and making it idempotent would mean fingerprinting every
note the agent writes. Raising an HR ticket is, and it also needs a human to
agree first -- so it is the one tool here that is fingerprinted into the
ledger *and* gated behind an approval, which is the combination that makes a
retried request safe to allow at all.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.agent.tools import (
    SearchArgs,
    ToolError,
    ToolRegistry,
    ToolSpec,
    register_builtins,
)
from app.core.config import Settings, get_logger, get_settings
from app.rag.clients import web_result_urls, web_results_to_text
from app.rag.loaders import cite_sources, format_documents
from app.services.authorization import ANONYMOUS, can

log = get_logger("toolbox")

__all__ = [
    "EscalateToolArgs",
    "NoteToolArgs",
    "RememberArgs",
    "TicketArgs",
    "authorize",
    "build_registry",
]


# --- Parameter models ---------------------------------------------------------
# Separate from the ones in `app/agent/tools.py`, which are the generic shapes
# the machinery ships with. These describe *this product's* tools, and their
# field descriptions are what the model reads, so they are written for a reader
# who has to decide whether to call the tool rather than for a schema.


class NoteToolArgs(BaseModel):
    """Arguments for `scratchpad_write`."""

    text: str = Field(
        min_length=1,
        max_length=1200,
        description="The note. One finding or one decision, in a sentence or two.",
    )
    kind: str = Field(
        default="finding",
        description="plan | finding | question | decision. Defaults to finding.",
    )


class RememberArgs(BaseModel):
    """Arguments for `remember_fact`."""

    key: str = Field(
        description=(
            "One of: employment_type, department, location, work_mode, "
            "tenure_band, preferred_language. Anything else is refused."
        )
    )
    value: str = Field(
        min_length=1,
        max_length=120,
        description="What the employee said about themselves, in their words.",
    )


class TicketArgs(BaseModel):
    """Arguments for `raise_ticket`."""

    summary: str = Field(
        min_length=8,
        max_length=200,
        description="One line an HR advisor will see in a queue.",
    )
    detail: str = Field(
        default="",
        max_length=2000,
        description="What the employee asked and what policy did or did not cover.",
    )


class EscalateToolArgs(BaseModel):
    """Arguments for `escalate_to_human`."""

    reason: str = Field(
        min_length=8,
        max_length=400,
        description=(
            "Why a person has to decide this: a judgement the documents do not "
            "contain, an individual case, an allegation, or a request for an "
            "exception."
        ),
    )


# --- Handlers -----------------------------------------------------------------
# Each closes over a service. Written as closures rather than methods because
# `ToolSpec.handler` is a plain callable and a class would add a layer whose
# only job is to hold one attribute.


def _search_result(documents: list[Document]) -> dict[str, Any]:
    """The shape every search tool returns.

    `text` goes through `format_documents`, which renders `[n] source=<file>`
    and nothing else -- the format the grader prompt is calibrated against. The
    agent reading this observation is making the same judgement the grader
    makes, so showing it something else would be calibrating one path and not
    the other.
    """
    return {
        "chunks": documents,
        "citations": cite_sources(documents),
        "text": format_documents(documents) or "No matching policy text.",
        "count": len(documents),
    }


def _kb_search_handler(retriever: Any) -> Any:
    def kb_search(query: str, k: int = 4) -> dict[str, Any]:
        documents = list(retriever.invoke(query) or [])
        # `k` narrows what the *model* is shown. The retriever's own k is a
        # configuration decision and a per-call one would mean a tool argument
        # changing how hard Pinecone is asked to work, which is a cost the
        # caller cannot see.
        return _search_result(documents[: max(1, min(k, 10))])

    return kb_search


def _web_search_handler(web_search: Any) -> Any:
    def web_search_tool(query: str, k: int = 4) -> dict[str, Any]:
        raw = web_search.invoke({"query": query})
        text = web_results_to_text(raw)
        if not text.strip():
            # A `ToolError` rather than an empty success: the agent needs to
            # know the search came back with nothing so it can change tack,
            # and an empty `text` observation reads as "the web says nothing
            # about this", which is a different and much stronger claim.
            raise ToolError("the web search returned no results")
        return {
            "chunks": [],
            "citations": [],
            "text": text,
            "urls": web_result_urls(raw),
        }

    return web_search_tool


def _note_handler(scratchpad: Any, conversation_id: str | None) -> Any:
    def scratchpad_write(text: str, kind: str = "finding") -> dict[str, Any]:
        if not conversation_id:
            raise ToolError("there is no conversation to write a note to")
        note = scratchpad.append(
            conversation_id, text, author="agent", kind=kind
        )
        if note is None:
            raise ToolError("the note could not be stored")
        return {"stored": True, "id": note.id, "kind": note.kind}

    return scratchpad_write


def _remember_handler(facts: Any, subject: str) -> Any:
    def remember_fact(key: str, value: str) -> dict[str, Any]:
        fact = facts.remember(subject, key, value, source="conversation")
        if fact is None:
            # The closed key set is the point, so a refused key is reported
            # back as something the model can correct rather than as a failure.
            raise ToolError(
                f"{key!r} is not a fact this assistant keeps; it remembers only "
                f"employment type, department, location, work mode, tenure band "
                f"and preferred language"
            )
        return {"remembered": fact.key, "value": fact.value}

    return remember_fact


def _ticket_handler(approvals: Any, question: str, conversation_id: str | None) -> Any:
    def raise_ticket(summary: str, detail: str = "") -> dict[str, Any]:
        """Record the ticket an approver has already agreed to.

        The registry runs this only *after* an approval was granted -- it
        returns `needs_approval` without calling the handler otherwise -- so
        reaching here means a person said yes. What this does is write the
        decided request back as the ticket itself, which is why there is no
        second store: the approval queue *is* the ticket queue in this
        deployment, and inventing a second one would mean two places to look
        for the same request.
        """
        recorded = approvals.request(
            action="hr_ticket",
            question=question,
            reason=summary,
            payload={"summary": summary, "detail": detail},
            risk="low",
            requested_by="agent",
            conversation_id=conversation_id,
        )
        if recorded is None:
            raise ToolError("the ticket could not be recorded")
        return {"ticket": getattr(recorded, "id", recorded), "summary": summary}

    return raise_ticket


def _escalate_handler(
    approvals: Any, question: str, conversation_id: str | None, subject: str
) -> Any:
    def escalate_to_human(reason: str) -> dict[str, Any]:
        recorded = approvals.request(
            action="escalate_to_human",
            question=question,
            reason=reason,
            payload={"reason": reason},
            risk="high",
            requested_by=subject,
            conversation_id=conversation_id,
        )
        if recorded is None:
            # Fail closed. Telling an employee their question has been
            # escalated when nothing recorded it is worse than telling them it
            # could not be: the first ends with nobody looking at it and the
            # employee waiting.
            raise ToolError("the escalation could not be recorded")
        return {
            "escalated": True,
            "id": getattr(recorded, "id", recorded),
            "reason": reason,
        }

    return escalate_to_human


# --- Assembly -----------------------------------------------------------------


def authorize(principal: Any, action: str) -> bool:
    """The registry's authoriser: `can`, with no principal meaning anonymous.

    `app/services/authorization.can` takes a `Principal` and would fail on
    `None`, and the registry passes whatever its caller gave it -- which for a
    CLI, a script or a test is nothing at all. Defaulting to `ANONYMOUS`
    rather than to allow-everything is the direction that cannot surprise
    anyone: an unauthenticated caller is an employee, which is exactly what
    `/api/chat` being open already means, and the two tools an employee may not
    use stay refused.

    The alternative -- passing `authorize=None`, which the registry reads as
    "allow" -- would make a missing principal grant every action, so a route
    that forgot to resolve one would silently hand an anonymous visitor the HR
    role's tools. Getting that wrong has to fail closed.
    """
    return can(principal or ANONYMOUS, action)


def build_registry(
    *,
    settings: Settings | None = None,
    retriever: Any = None,
    web_search: Any = None,
    scratchpad: Any = None,
    facts: Any = None,
    approvals: Any = None,
    idempotency: Any = None,
    tracer: Any = None,
    budget: Any = None,
    question: str = "",
    conversation_id: str | None = None,
    subject: str = "anonymous",
    authorizer: Any = authorize,
) -> ToolRegistry:
    """The registry for one request.

    Per request, not per process, and that is not an oversight: the ticket and
    escalation tools close over *this* question and *this* conversation id, so
    a shared registry would file one employee's escalation against another's
    question. Registration is a dictionary insert over specs whose expensive
    parts (the retriever, the stores) are the shared ones passed in, so the
    construction cost is nothing.

    Every service is optional. A tool whose service is absent is **not
    registered**, rather than registered and failing when called: the
    catalogue is what the model is shown, and showing it a tool that cannot
    work is inviting it to spend a step discovering that. A deployment with no
    Tavily key therefore has an agent that never tries the web, which is the
    honest shape.
    """
    settings = settings or get_settings()
    registry = ToolRegistry(
        settings=settings,
        tracer=tracer,
        budget=budget,
        idempotency=idempotency,
        approvals=approvals,
        authorize=authorizer,
    )

    # Arithmetic, which is `safe_eval` and needs nothing. Registered first so
    # that an agent with no services at all still has one real tool -- and
    # because a language model doing arithmetic in prose is the classic
    # wrong-number failure an HR product must not have.
    register_builtins(registry)

    if retriever is not None:
        registry.register(
            ToolSpec(
                name="kb_search",
                description=(
                    "Search this company's internal HR policy documents. Use "
                    "this first for anything about leave, pay, benefits, "
                    "conduct, working arrangements or forms. Returns numbered "
                    "passages with their source file names."
                ),
                parameters=SearchArgs,
                handler=_kb_search_handler(retriever),
                requires="kb.read",
                timeout_seconds=min(20.0, settings.tool_timeout_seconds),
                retries=settings.tool_retries,
                idempotent=True,
                cost_hint="network",
            )
        )

    if web_search is not None:
        registry.register(
            ToolSpec(
                name="web_search",
                description=(
                    "Search the public web. Only after kb_search came back "
                    "empty, and only for general information -- a public "
                    "article is not this company's policy and must never be "
                    "presented as one."
                ),
                parameters=SearchArgs,
                handler=_web_search_handler(web_search),
                requires="web.search",
                timeout_seconds=settings.tool_timeout_seconds,
                retries=settings.tool_retries,
                idempotent=True,
                cost_hint="network",
            )
        )

    if scratchpad is not None and conversation_id:
        registry.register(
            ToolSpec(
                name="scratchpad_write",
                description=(
                    "Write one working note so later steps and the person "
                    "reading the trace can see what you established."
                ),
                parameters=NoteToolArgs,
                handler=_note_handler(scratchpad, conversation_id),
                requires="scratchpad.write",
                idempotent=True,
                cost_hint="cheap",
            )
        )

    if facts is not None:
        registry.register(
            ToolSpec(
                name="remember_fact",
                description=(
                    "Remember something the employee stated about themselves, "
                    "so they are not asked again. Only what they said, never "
                    "what you inferred."
                ),
                parameters=RememberArgs,
                handler=_remember_handler(facts, subject),
                requires="memory.write",
                idempotent=True,
                cost_hint="cheap",
            )
        )

    if approvals is not None:
        registry.register(
            ToolSpec(
                name="raise_ticket",
                description=(
                    "Raise an HR ticket for something policy does not answer. "
                    "A person has to approve this before it happens."
                ),
                parameters=TicketArgs,
                handler=_ticket_handler(approvals, question, conversation_id),
                requires="ticket.create",
                idempotent=False,
                side_effecting=True,
                needs_approval=True,
                cost_hint="cheap",
            )
        )
        registry.register(
            ToolSpec(
                name="escalate_to_human",
                description=(
                    "Hand this question to a person. Use it when the answer "
                    "needs a judgement the documents do not contain, concerns "
                    "someone's individual case, involves an allegation, or asks "
                    "for an exception. This ends your turn."
                ),
                parameters=EscalateToolArgs,
                handler=_escalate_handler(
                    approvals, question, conversation_id, subject
                ),
                requires="escalate",
                idempotent=False,
                side_effecting=True,
                # Not gated behind an approval, and the asymmetry with
                # `raise_ticket` is the point: escalating *is* asking for a
                # human, so requiring a human to approve the request for a
                # human is a loop with nobody in it.
                needs_approval=False,
                cost_hint="cheap",
            )
        )

    return registry
