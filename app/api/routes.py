"""The HTTP surface: health, chat, upload, audit.

Two routers, because they answer to different callers. `ops` is unprefixed and
open -- `/health` is what a load balancer polls, and it must work at every
environment and without credentials. `api` is prefixed `/api` and mixes the
employee-facing `/chat` with the admin-only `/upload` and `/audit`.

Every route calls a service, never the agent or Pinecone directly. `/chat` goes
through `Copilot`, `/upload` through `app.services.ingestion` and
`app.rag.ingest`, `/audit` through `AuditStore`. That is what keeps the HTTP
layer from growing a second, subtly different copy of the pipeline.

**No handler returns an exception's text.** A provider error carries URLs,
request ids and sometimes the offending payload, and this application holds API
keys; `detail=str(exc)` is how those reach a browser. Failures are logged with
their traceback and answered with a fixed sentence.

**The blocking work runs in a threadpool.** `/chat` and `/audit` are `def`, not
`async def`, so Starlette runs them off the event loop -- a question takes
seconds of LLM round trips, and an `async def` handler doing that blocks every
other request in the process. `/upload` is `async def` because it streams a
request body, and hands the embed-and-upsert to `run_in_threadpool` explicitly.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import __version__
from app.agent.graph import BRANCHES, EDGES
from app.agent.nodes import NODE_NAMES, TERMINAL_NODES
from app.agent.supervisor import (
    SUPERVISOR_BRANCHES,
    SUPERVISOR_EDGES,
    SUPERVISOR_NODE_NAMES,
    WORKER_DOMAINS,
)
from app.api.deps import (
    get_approvals_dep,
    get_audit_dep,
    get_cache_dep,
    get_copilot_dep,
    get_credentials_dep,
    get_limiter_dep,
    get_pool_dep,
    get_principal_dep,
    get_scratchpad_dep,
    get_settings_dep,
    require_action,
    require_admin,
)
from app.core.config import Settings, get_logger
from app.rag.ingest import ingest_upload
from app.services.audit import AuditStore
from app.services.authorization import POLICY, Principal, can, permissions_for
from app.services.byok import ALLOWED_MODELS, ProviderCredentials
from app.services.copilot import AnswerResult, Copilot
from app.services.ingestion import DEFAULT_DEPARTMENT, SUPPORTED, is_supported
from app.services.routing import MODEL_CATALOG
from app.services.streaming import EVENT_ERROR, EVENT_RESULT, stream_in_thread
from app.services.toolbox import build_registry

log = get_logger("api")

# Read the upload a megabyte at a time so the size cap can stop a large one
# part way through, rather than after it is already resident in memory.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# A department becomes chunk metadata and a Pinecone filter value, so it is
# constrained to something a filter expression can hold unambiguously.
DEPARTMENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

UNAVAILABLE = (
    "The copilot could not answer that right now. The failure has been logged."
)

ops = APIRouter(tags=["ops"])
api = APIRouter(prefix="/api")
# The UI is one page. `include_in_schema=False` keeps it out of /docs, which
# describes the JSON API -- an HTML page in that list is noise to anyone reading
# it to write a client.
pages = APIRouter(include_in_schema=False)


# --- The page ----------------------------------------------------------------


@pages.get("/", response_class=HTMLResponse)
def index(request: Request, settings: Settings = Depends(get_settings_dep)):
    """Serve the single-page console.

    Rendered server-side rather than shipped as a static file so the page starts
    with the deployment's own identity -- its name, environment and whether the
    admin surface is configured at all. The browser then talks to the same
    origin for everything else, which is why this application needs no CORS
    middleware: adding one would be opening a door nothing is knocking on.

    `admin_enabled` decides whether the upload and audit views are *offered*.
    It is not the gate -- `require_admin` is, on every request. Hiding a control
    the server would refuse anyway is a courtesy, not a security boundary.
    """
    return request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {
            "app_name": settings.app_name,
            "app_env": settings.app_env,
            "version": __version__,
            "admin_enabled": settings.admin_enabled,
            "hr_enabled": settings.hr_enabled,
            # Whether to offer the key panel at all. A deployment that refuses
            # caller-supplied keys should not show a box for one.
            "byok_enabled": settings.allow_byok,
            "public_demo": settings.public_demo,
            "agent_mode": settings.agent_mode,
            "retrieval_mode": settings.effective_retrieval_mode,
            # False means every answer comes from the extractive path unless the
            # visitor supplies a key, which is a thing the page has to say
            # up front rather than leave someone to infer from the answers.
            "server_llm": settings.has_llm,
        },
    )


# --- Health ------------------------------------------------------------------


@ops.get("/health")
def health(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """Liveness plus a configuration summary.

    Reports which credentials are *missing* by name and never echoes a value, so
    this is safe to expose to a load balancer. Open at every environment,
    including production, because the load balancer depends on it.
    """
    missing = settings.missing_secrets()
    return {
        "status": "ok" if not missing else "degraded",
        "app": settings.app_name,
        "env": settings.app_env,
        "version": __version__,
        # Whether the admin routes will accept anything at all. They fail closed
        # without a key, and silent-off is exactly the state worth being able to
        # see from outside.
        "admin_enabled": settings.admin_enabled,
        "missing_secrets": missing,
        "pinecone_index": settings.pinecone_index,
        "private_namespace": settings.private_namespace,
        # Which providers this deployment actually resolved to. Reported because
        # "which model answered?" is the first question about a bad answer, and
        # the second is "was that the provider we thought?".
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.active_embedding_model,
        "embedding_dim": settings.embedding_dim,
        # Reported because tracing sends questions and retrieved policy text off
        # the deployment; whether it is on should be visible, not inferred from
        # a config file nobody reads in production.
        "tracing": settings.tracing_enabled,
        # Whether traces carry the employee's question and the retrieved policy
        # text. Surfaced so "we turned redaction off to debug" cannot quietly
        # become the permanent state of production.
        "tracing_redacted": settings.tracing_redacted,
        # What this deployment can actually do, as opposed to what it is
        # configured to prefer. Separate from `missing_secrets` on purpose: a
        # public demo with no vector database is not broken, it is lexical, and
        # a health check that cannot express the difference reports a working
        # deployment as faulty.
        "capabilities": settings.capabilities(),
    }


# --- Chat --------------------------------------------------------------------


class ChatTurn(BaseModel):
    """One prior turn of the conversation, as the browser holds it."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    """One employee question, optionally within a conversation.

    The upper bound is not tidiness: the question is embedded, then rendered
    into the router, grader and answer prompts, so its length is multiplied
    across several billed calls before anything rejects it. `history` is capped
    for the same reason and again in `normalise_history` — it is rendered into
    the contextualise prompt on every follow-up, so an uncapped conversation
    gets more expensive the longer it runs.

    **The client owns the conversation and sends it back.** The server keeps no
    per-user chat state, because `/chat` is deliberately open — there is no
    identity here to scope a stored conversation to, and holding one employee's
    HR questions in a process that serves everyone is not a thing to do by
    accident. `conversation_id` is recorded on the audit row so an admin can
    group a thread; it is not a key anything is fetched by.
    """

    question: str = Field(min_length=2, max_length=3000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    conversation_id: str | None = Field(default=None, max_length=64)
    # Which orchestration answers *this* question, overriding AGENT_MODE.
    # Per-question rather than per-deployment because the interesting
    # comparison is the same question answered three ways, which is what the
    # console's mode switch is for. Constrained to the four that exist rather
    # than passed through: an unknown value would fall through to the graph and
    # silently answer a question the caller thought they had routed elsewhere.
    mode: Literal["graph", "react", "supervisor", "extractive"] | None = None


class ChatResponse(AnswerResult):
    """The agent's answer, plus where it was filed.

    Extends `AnswerResult` rather than restating it, so the API response and the
    facade's contract cannot drift. `audit_id` is null when the audit write
    failed -- the answer stands either way.
    """

    audit_id: int | None = None


def _answer(
    payload: ChatRequest,
    *,
    request: Request,
    copilot: Copilot,
    audit: AuditStore,
    principal: Principal,
    credentials: ProviderCredentials | None,
    limiter: Any,
    pool: Any,
    stream: Any = None,
) -> ChatResponse:
    """The body `/chat` and `/chat/stream` share.

    One function, because the two endpoints must not be able to answer
    differently. The only thing the streamed one does extra is hand an
    `EventStream` to the pipeline, and the pipeline treats that as an observer:
    `test_a_streamed_answer_is_the_same_answer` pins it.
    """
    # The limiter guards the *deployment's own* key. A caller spending their own
    # is not spending ours, and an authenticated admin is operating the
    # deployment rather than consuming it -- `should_limit` is the one place
    # that policy lives so a route cannot forget half of it.
    if limiter is not None and limiter.should_limit(
        byok=principal.byok, is_admin=principal.is_admin
    ):
        decision = limiter.check(principal.subject)
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=decision.reason,
                headers={"Retry-After": str(decision.retry_after)},
            )

    # A caller-supplied key gets its own copilot out of the pool: its own chat
    # model and its own cache over the deployment's shared retriever. The pool
    # holds clients, never keys beyond the life of the entry it built.
    answering = copilot
    if credentials is not None and pool is not None:
        answering = pool.get(credentials)

    started = time.perf_counter()
    try:
        result = answering.ask(
            payload.question,
            history=[turn.model_dump() for turn in payload.history],
            conversation_id=payload.conversation_id,
            principal=principal,
            mode=payload.mode,
            byok=credentials is not None,
            stream=stream,
        )
    except ValueError as exc:
        # An empty question after validation -- cheap to reject, and the message
        # is our own text, not a provider's.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak provider error text
        log.exception("The copilot raised while answering")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=UNAVAILABLE
        ) from exc

    latency_ms = result.latency_ms or int((time.perf_counter() - started) * 1000)
    audit_id = audit.record(
        result,
        latency_ms=latency_ms,
        conversation_id=payload.conversation_id,
        subject=principal.subject,
    )
    _ = request  # the request itself is only needed for the dependencies above
    return ChatResponse(**result.model_dump(), audit_id=audit_id)


@api.post("/chat", response_model=ChatResponse, tags=["copilot"])
def chat(
    payload: ChatRequest,
    request: Request,
    copilot: Copilot = Depends(get_copilot_dep),
    audit: AuditStore = Depends(get_audit_dep),
    principal: Principal = Depends(get_principal_dep),
    credentials: ProviderCredentials | None = Depends(get_credentials_dep),
    limiter: Any = Depends(get_limiter_dep),
    pool: Any = Depends(get_pool_dep),
) -> ChatResponse:
    """Answer one question from internal policy, the web, or not at all.

    The response carries the decision trace, the span timings, the citations,
    the model that answered and what it cost, alongside the answer -- because an
    HR answer that cannot be checked against its source is not much use to the
    person acting on it, and because "which model said that, and had anything
    been redacted?" are the first two questions about a bad one.
    """
    return _answer(
        payload,
        request=request,
        copilot=copilot,
        audit=audit,
        principal=principal,
        credentials=credentials,
        limiter=limiter,
        pool=pool,
    )


@api.post("/chat/stream", tags=["copilot"])
def chat_stream(
    payload: ChatRequest,
    request: Request,
    copilot: Copilot = Depends(get_copilot_dep),
    audit: AuditStore = Depends(get_audit_dep),
    principal: Principal = Depends(get_principal_dep),
    credentials: ProviderCredentials | None = Depends(get_credentials_dep),
    limiter: Any = Depends(get_limiter_dep),
    pool: Any = Depends(get_pool_dep),
) -> StreamingResponse:
    """The same answer, emitted stage by stage as server-sent events.

    **POST, not GET, and therefore not `EventSource`.** The browser's
    `EventSource` only issues GET, which would put the question and the whole
    conversation in a URL -- and a URL reaches the history, the proxy log and
    the `Referer` header. The client reads this with `fetch` and a stream
    reader instead, which is a few more lines of JavaScript and keeps an
    employee's HR question out of three logs it has no business being in.

    The pipeline is driven on a worker thread and this handler yields what it
    emits; `app/services/streaming.py` explains why that is a thread rather
    than an async generator. A `result` event carries the full `ChatResponse`
    at the end, so a client that missed an earlier frame still gets everything.

    `X-Accel-Buffering: no` is for an Nginx in front of this -- the target
    architecture has one -- which otherwise buffers the whole response and
    turns a stream into a slow single reply.
    """

    def work(stream: Any) -> None:
        try:
            response = _answer(
                payload,
                request=request,
                copilot=copilot,
                audit=audit,
                principal=principal,
                credentials=credentials,
                limiter=limiter,
                pool=pool,
                stream=stream,
            )
        except HTTPException as exc:
            # A refusal is part of the protocol, not a crash: a 429 or a 400
            # reaches the client as an `error` event with our own text, because
            # once the response has started streaming there is no status code
            # left to set.
            stream.emit(EVENT_ERROR, message=str(exc.detail), status=exc.status_code)
            stream.close(ok=False, reason="refused")
            return
        stream.emit(EVENT_RESULT, **response.model_dump())
        stream.close(ok=True)

    return StreamingResponse(
        stream_in_thread(work, max_seconds=float(settings_stream_ceiling(request))),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def settings_stream_ceiling(request: Request) -> float:
    """How long a stream may stay open, derived from the agent's own budget.

    Set *above* `max_agent_seconds` rather than independently: a stream that
    times out before the run it is watching would cut an answer that was about
    to arrive, and two unrelated ceilings is how that happens on the one
    question that took longest. The margin covers the stages around the run --
    the guards, the cache lookup, the audit write.
    """
    settings: Settings = request.app.state.settings
    return float(settings.max_agent_seconds) + 20.0


# --- Introspection: what this deployment is and how it is wired --------------
# All three of these are open, and the reasoning is the same one that keeps the
# console's Overview page ungated: they describe the *system*, not its contents.
# None of them can return an employee's question, an answer, a policy chunk or
# a credential. The audience is somebody evaluating the project, which is
# exactly the audience that cannot be issued a key.


@api.get("/config", tags=["meta"])
def config(
    settings: Settings = Depends(get_settings_dep),
    principal: Principal = Depends(get_principal_dep),
) -> dict[str, Any]:
    """Every knob this deployment resolved to, and what the caller may do.

    The console reads this once at load to decide what to offer: which modes
    the mode switch lists, whether the key panel appears, which models a
    caller-supplied key may name, and which nav items to show. It is a
    *description*, never the gate -- `require_action` is, on every request.
    Hiding a control the server would refuse anyway is a courtesy.

    Prices are in `MODEL_CATALOG` and are published list prices used for a
    projection. Said here as well as in the module that holds them, because
    this is where a dashboard reads them from.
    """
    return {
        "app": settings.app_name,
        "env": settings.app_env,
        "version": __version__,
        "capabilities": settings.capabilities(),
        "agent_modes": ["graph", "react", "supervisor", "extractive"],
        "agent_mode": settings.agent_mode,
        "retrieval": {
            "configured": settings.retrieval_mode,
            "effective": settings.effective_retrieval_mode,
            "dense_weight": settings.hybrid_dense_weight,
            "rrf_k": settings.rrf_k,
            "rerank": settings.rerank_strategy,
            "rerank_candidates": settings.rerank_candidates,
            "k": settings.default_k,
            "gate": settings.relevance_threshold,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "fewshot_grading": settings.fewshot_grading,
        },
        "cache": {
            "enabled": settings.cache_enabled,
            "similarity": settings.cache_similarity,
            "ttl_seconds": settings.cache_ttl_seconds,
            "max_entries": settings.cache_max_entries,
        },
        "budgets": {
            "steps": settings.max_agent_steps,
            "tool_calls": settings.max_tool_calls,
            "llm_calls": settings.max_llm_calls,
            "tokens": settings.max_agent_tokens,
            "seconds": settings.max_agent_seconds,
            "cost_usd": settings.max_agent_cost_usd,
            "handoffs": settings.max_handoffs,
            "tool_timeout_seconds": settings.tool_timeout_seconds,
            "tool_retries": settings.tool_retries,
            "tool_parallelism": settings.tool_parallelism,
        },
        "safety": {
            "input_guard": settings.input_guard_enabled,
            "output_guard": settings.output_guard_enabled,
            "pii_redaction": settings.redact_pii,
            "audit_stores_redacted": settings.audit_stores_redacted,
        },
        "byok": {
            "allowed": settings.allow_byok,
            "models": {k: list(v) for k, v in ALLOWED_MODELS.items()},
            "pool_size": settings.byok_pool_size,
        },
        "rate_limit": {
            "enabled": settings.rate_limit_enabled,
            "per_minute": settings.rate_limit_per_minute,
            "burst": settings.rate_limit_burst,
            # Said in the payload and not only in the docstring: a dashboard
            # reporting a limit that is per-instance on a platform that runs
            # several instances is a dashboard that is wrong.
            "scope": "process",
        },
        "models": {
            name: {
                "provider": spec.provider,
                "tier": spec.tier,
                "input_usd_per_mtok": spec.input_usd_per_mtok,
                "output_usd_per_mtok": spec.output_usd_per_mtok,
                "context_window": spec.context_window,
                "note": spec.note,
            }
            for name, spec in MODEL_CATALOG.items()
        },
        "principal": principal.to_dict(),
        "permissions": permissions_for(principal.role),
        "policy": {action: sorted(roles) for action, roles in POLICY.items()},
    }


@api.get("/graph", tags=["meta"])
def graph_definition() -> dict[str, Any]:
    """Both graphs, as the server has compiled them.

    **The console draws from this rather than from a copy.** `CLAUDE.md` warns
    that the UI's diagram and the real edge set can drift, and that renaming a
    trace tag silently stops lighting a node. Serving the node and edge lists
    from the modules that `build_graph` and `build_supervisor_graph` wire from
    removes half of that class of bug: a node added to `NODE_NAMES` appears in
    the diagram without anybody editing JavaScript.

    `test_the_compiled_graph_matches_the_documented_design` keeps these
    constants honest against what actually compiles, so this is not a third
    description of the graph -- it is the second one, published.
    """
    return {
        "pipeline": {
            "nodes": list(NODE_NAMES),
            "terminal": list(TERMINAL_NODES),
            "edges": [{"from": a, "to": b} for a, b in EDGES],
            "branches": {
                source: {"condition": condition, "targets": targets}
                for source, (condition, targets) in BRANCHES.items()
            },
            "entry": "contextualize",
        },
        "supervisor": {
            "nodes": list(SUPERVISOR_NODE_NAMES),
            "terminal": ["compose", "escalate"],
            "edges": [{"from": a, "to": b} for a, b in SUPERVISOR_EDGES],
            "branches": {
                source: {"condition": condition, "targets": targets}
                for source, (condition, targets) in SUPERVISOR_BRANCHES.items()
            },
            "entry": "plan",
            "workers": {
                name: {
                    "label": domain.get("label", name),
                    "description": domain.get("description", ""),
                    "documents": list(domain.get("doc_hints") or []),
                    "departments": list(domain.get("departments") or []),
                }
                for name, domain in WORKER_DOMAINS.items()
            },
        },
    }


@api.get("/tools", tags=["meta"])
def tools(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    principal: Principal = Depends(get_principal_dep),
) -> dict[str, Any]:
    """The tool catalogue this caller would be shown, with its safety flags.

    Built through the same `build_registry` the agent uses and filtered by the
    same permission table, so what is listed here is exactly what the model
    would be offered -- rather than a hand-written list that drifts. The
    handlers are closed over no real question and the registry is thrown away;
    nothing is called.
    """
    state = request.app.state
    # Stand-ins, not clients. `build_registry` registers a tool when its
    # service is present and the question here is "what would this caller be
    # offered?", not "can it run right now" -- so the retriever is a sentinel
    # and the web-search one is present only when there is a key for it, which
    # is the one capability that genuinely changes the catalogue.
    stand_in = object()
    registry = build_registry(
        settings=settings,
        retriever=stand_in,
        web_search=stand_in if settings.has_web_search else None,
        scratchpad=getattr(state, "scratchpad", None),
        facts=getattr(state, "facts", None),
        approvals=getattr(state, "approvals", None),
        conversation_id="catalogue",
        subject=principal.subject,
    )
    specs = []
    for name in registry.names(principal):
        spec = registry.get(name)
        if spec is None:  # pragma: no cover - names() reads the same dict
            continue
        specs.append(
            {
                "name": spec.name,
                "description": spec.description,
                "requires": spec.requires,
                "parameters": spec.parameters.model_json_schema(),
                "timeout_seconds": spec.timeout_seconds,
                "retries": spec.retries,
                "idempotent": spec.idempotent,
                "side_effecting": spec.side_effecting,
                "needs_approval": spec.needs_approval,
                "cost_hint": spec.cost_hint,
            }
        )
    return {"tools": specs, "role": principal.role}


# --- Metrics -----------------------------------------------------------------


@api.get("/metrics", tags=["meta"])
def metrics(
    request: Request,
    audit: AuditStore = Depends(get_audit_dep),
    settings: Settings = Depends(get_settings_dep),
    principal: Principal = Depends(get_principal_dep),
) -> dict[str, Any]:
    """Aggregate health of the answers and of the machinery around them.

    **Open, unlike `/api/audit`, and the line between them is the point.** An
    audit row is one employee's question and the answer they were given, which
    is HR-sensitive by definition. An aggregate is not: "we answered 412
    questions, 86% of them grounded, p95 4.1 seconds" says nothing about anyone.
    So the rows stay behind an admin key and the totals do not, which is what
    makes the dashboard worth building -- the audience for it is somebody
    evaluating this deployment, and they cannot be issued a key.

    **One exception.** Per-model spend is withheld unless the caller holds
    `metrics.read`. The answer counts are not commercially sensitive and the
    dollar split across providers is, and withholding the narrow thing is
    cheaper than gating the whole endpoint.

    **The cost figures are projections**, not invoices: a token estimate from
    character counts multiplied by a published list price. Reported with
    `cost_is_projected: true` so a reader does not have to know that.
    """
    state = request.app.state
    stats = audit.stats()
    if not can(principal, "metrics.read"):
        stats = {key: value for key, value in stats.items() if key != "by_model"}

    def safe(service: Any, name: str) -> dict[str, Any]:
        """A service's stats, or an explanation -- never an exception.

        `/api/metrics` is what the dashboard polls, and one store with a locked
        database must not blank the whole page.
        """
        if service is None:
            return {"available": False}
        try:
            return {"available": True, **service.stats()}
        except Exception:  # noqa: BLE001 - a metric is not worth a 500
            log.exception("Could not read %s stats", name)
            return {"available": False, "error": "unavailable"}

    return {
        "answers": stats,
        "cost_is_projected": True,
        "capabilities": settings.capabilities(),
        "cache": safe(getattr(state, "cache", None), "cache"),
        "rate_limit": safe(getattr(state, "limiter", None), "limiter"),
        "byok_pool": safe(getattr(state, "pool", None), "pool"),
        "approvals": safe(getattr(state, "approvals", None), "approvals"),
        "scratchpad": safe(getattr(state, "scratchpad", None), "scratchpad"),
        "idempotency": safe(getattr(state, "idempotency", None), "idempotency"),
        "knowledge_base": {
            "documents": _kb_document_count(settings),
            "directory": str(settings.kb_dir),
        },
        "role": principal.role,
    }


def _kb_document_count(settings: Settings) -> int:
    """How many documents are on disk. Zero rather than an error if unreadable."""
    try:
        directory = Path(settings.kb_dir)
        if not directory.is_dir():
            return 0
        return sum(1 for path in directory.rglob("*") if is_supported(path))
    except OSError:
        log.warning("Could not count the knowledge base at %s", settings.kb_dir)
        return 0


# --- Scratchpad --------------------------------------------------------------


class NoteRequest(BaseModel):
    """One note a user adds to their own conversation's scratchpad."""

    conversation_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1200)
    kind: Literal["note", "question", "plan", "decision", "user"] = "user"


@api.get("/scratchpad/{conversation_id}", tags=["copilot"])
def scratchpad_read(
    conversation_id: str = PathParam(max_length=64),
    scratchpad: Any = Depends(get_scratchpad_dep),
) -> dict[str, Any]:
    """The working notes for one conversation.

    **Scoped by the conversation id and nothing else, and that is a deliberate
    limit rather than an oversight.** `/api/chat` is open, so there is no
    identity here a note could be scoped to -- `CLAUDE.md` explains why the
    product has none. The id is generated in the browser from
    `crypto.randomUUID()`, so it behaves like a capability: whoever holds it can
    read that conversation's notes, and nothing lists the ids. `
    recent_conversations` -- the call that *would* enumerate them -- is behind
    the admin gate for exactly that reason.

    A deployment that needs more than that needs authentication first, and the
    note in `CLAUDE.md` about `GET /api/conversations` applies here unchanged.
    """
    if scratchpad is None:
        return {"conversation_id": conversation_id, "notes": []}
    notes = scratchpad.notes(conversation_id, limit=200)
    return {
        "conversation_id": conversation_id,
        "notes": [note.to_dict() for note in notes],
    }


@api.post("/scratchpad", status_code=status.HTTP_201_CREATED, tags=["copilot"])
def scratchpad_write(
    payload: NoteRequest,
    scratchpad: Any = Depends(get_scratchpad_dep),
    principal: Principal = Depends(require_action("scratchpad.write")),
) -> dict[str, Any]:
    """Add a note of your own, so the agent's next turn can see it.

    The scratchpad is shared working state, which means a person can put
    something on it: "I'm a contractor", "this is about the Bangalore office".
    The next turn renders it into the prompt, so this is the cheapest way for
    someone to correct a wrong assumption without retyping their question.
    """
    if scratchpad is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The scratchpad is not configured on this deployment.",
        )
    note = scratchpad.append(
        payload.conversation_id,
        payload.text,
        author=principal.role,
        kind=payload.kind,
    )
    if note is None:
        # The store returns None for an invalid conversation id *and* for a
        # failed write, and the difference matters to the caller: one is their
        # mistake and one is ours. The id is the only one this layer can check.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The note could not be stored. A conversation id must be 1-64 "
                "characters of letters, digits, dot, dash or underscore."
            ),
        )
    return note.to_dict()


# --- Approvals: the human in the loop ----------------------------------------
# HR and admin, through the permission table rather than through
# `require_admin`: working the approval queue is HR's job and reading every
# employee's question is not, and one key with one role could not say that.


@api.get("/approvals", tags=["hr"])
def approvals_list(
    limit: int = Query(default=50, ge=1, le=200),
    status_filter: str | None = Query(default=None, alias="status"),
    approvals: Any = Depends(get_approvals_dep),
    _principal: Principal = Depends(require_action("approval.list")),
) -> dict[str, Any]:
    """The queue, newest first, with the pending ones worth acting on.

    `expire_stale` runs on read rather than on a timer: there is no scheduler
    in this process, and an approval queue whose expiry depends on somebody
    having looked at it recently is still better than one with no expiry, which
    becomes a list nobody reads.
    """
    if approvals is None:
        return {"requests": [], "stats": {}, "expired": 0}
    expired = approvals.expire_stale()
    rows = approvals.recent(limit=limit, status=status_filter)
    return {
        "requests": [row.to_dict() for row in rows],
        "stats": approvals.stats(),
        "expired": expired,
    }


class DecisionRequest(BaseModel):
    """A human's decision on one request."""

    decision: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=1000)


@api.post("/approvals/{request_id}/decide", tags=["hr"])
def approvals_decide(
    request_id: int,
    payload: DecisionRequest,
    approvals: Any = Depends(get_approvals_dep),
    principal: Principal = Depends(require_action("approval.decide")),
) -> dict[str, Any]:
    """Approve or reject one request.

    **Single-shot.** Deciding something already decided returns the existing
    row unchanged rather than overwriting it, because two people clicking
    approve and reject within a second must not produce a last-write-wins
    outcome on something that has already had a side effect. The store enforces
    it; this reports what it found.
    """
    if approvals is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The approval queue is not configured on this deployment.",
        )
    before = approvals.get(request_id)
    if before is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No approval request with id {request_id}.",
        )
    decided = approvals.decide(
        request_id,
        payload.decision,
        decided_by=principal.role,
        note=payload.note,
    )
    if decided is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The decision could not be recorded.",
        )
    return {
        "request": decided.to_dict(),
        # True when this call is what decided it. The console shows "already
        # decided by someone else" rather than pretending the click worked.
        "changed": not before.decided_at and bool(decided.decided_at),
    }


# --- Cache -------------------------------------------------------------------


@api.post("/cache/invalidate", tags=["hr"])
def cache_invalidate(
    cache: Any = Depends(get_cache_dep),
    _principal: Principal = Depends(require_action("cache.invalidate")),
) -> dict[str, Any]:
    """Drop every cached answer.

    Needed because the TTL is not enough on its own: a newly ingested document
    can change the answer to a question that mentions none of its words, so
    "the cache expires in fifteen minutes" is fifteen minutes of serving
    superseded policy. `/api/upload` calls this itself after a successful
    ingest; this endpoint is for an ingest that happened elsewhere -- a CLI
    script, a worker.
    """
    if cache is None:
        return {"dropped": 0}
    return {"dropped": cache.invalidate()}


# --- Upload ------------------------------------------------------------------


class UploadResponse(BaseModel):
    """What one upload added to the private knowledge base."""

    file: str
    department: str
    namespace: str
    documents: int
    chunks: int
    vectors_in_namespace: int
    bytes_received: int


def _safe_filename(raw: str | None) -> str:
    """Reduce an uploaded name to a bare file name.

    A multipart filename is attacker-controlled text, not a path. Both
    separators are stripped rather than only the platform's, because a Windows
    client posting to a Linux container sends backslashes that `Path.name` there
    would keep as part of the name.
    """
    candidate = (raw or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    candidate = candidate.replace("\x00", "")
    if not candidate or candidate in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The upload needs a file name.",
        )
    return candidate


def _safe_department(raw: str) -> str:
    """Normalise a department to something a Pinecone filter can hold."""
    candidate = raw.strip().lower().replace(" ", "-")
    if not DEPARTMENT_PATTERN.match(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "department must be 1-64 characters of a-z, 0-9, dot, dash or "
                "underscore."
            ),
        )
    return candidate


async def _stream_to_disk(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Write the upload to disk, refusing it once it passes the cap.

    Checked while streaming rather than afterwards: `await file.read()` with no
    argument materialises the whole body first, which is a way to exhaust the
    process before any limit gets to apply. A refused upload leaves no partial
    file behind.
    """
    written = 0
    try:
        with destination.open("wb") as sink:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploads are limited to {max_bytes // (1024 * 1024)} MB.",
                    )
                sink.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        log.exception("Could not write the upload to %s", destination)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The upload could not be stored.",
        ) from exc
    return written


@api.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
    tags=["admin"],
)
async def upload(
    file: UploadFile = File(...),
    department: str = Form(default=DEFAULT_DEPARTMENT),
    settings: Settings = Depends(get_settings_dep),
    cache: Any = Depends(get_cache_dep),
) -> UploadResponse:
    """Add one document to the private knowledge base. Admin only.

    Admin only because this writes to the corpus the copilot answers employees
    from: whoever can upload here can change what the company's policy appears
    to say.

    The ingest never prunes. Pruning reconciles a namespace against a directory
    on disk, which for a single-file upload would delete the entire rest of the
    corpus.
    """
    name = _safe_filename(file.filename)
    if not is_supported(name):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Supported file types: {', '.join(sorted(SUPPORTED))}.",
        )
    dept = _safe_department(department)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / name
    received = await _stream_to_disk(file, destination, settings.max_upload_bytes)

    try:
        # Embedding and upserting are blocking and slow; off the event loop.
        report = await run_in_threadpool(
            ingest_upload,
            destination,
            namespace=settings.private_namespace,
            department=dept,
            settings=settings,
        )
    except ValueError as exc:
        # No extractable text -- a scanned PDF, or an empty file. The upload is
        # kept on disk so an admin can look at what was actually sent.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak provider error text
        log.exception("Ingesting %s failed", name)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The document was received but could not be indexed.",
        ) from exc

    # A new document can change the answer to a question that mentions none of
    # its words, so the TTL alone is not enough: fifteen minutes of serving
    # superseded policy is exactly what an HR copilot must not do. Dropped
    # whole rather than selectively, because working out which cached answers a
    # new document affects is the retrieval problem again.
    if cache is not None:
        dropped = cache.invalidate()
        if dropped:
            log.info("Dropped %d cached answer(s) after the ingest", dropped)

    log.info("Indexed upload %s into %s (%s)", name, report.namespace, dept)
    return UploadResponse(
        file=name,
        department=dept,
        namespace=report.namespace,
        documents=report.documents,
        chunks=report.chunks,
        vectors_in_namespace=report.vectors_in_namespace,
        bytes_received=received,
    )


# --- Audit -------------------------------------------------------------------


class AuditPage(BaseModel):
    """One page of the audit log, with enough to request the next."""

    total: int
    limit: int
    offset: int
    entries: list[dict[str, Any]]


@api.get(
    "/audit/stats",
    dependencies=[Depends(require_admin)],
    tags=["admin"],
)
def audit_stats(audit: AuditStore = Depends(get_audit_dep)) -> dict[str, Any]:
    """Aggregate answer health: how questions were resolved, and how often.

    Declared before `/audit/{entry_id}` on purpose -- a path parameter would
    otherwise claim "stats" and answer with a validation error.
    """
    return audit.stats()


@api.get(
    "/audit",
    response_model=AuditPage,
    dependencies=[Depends(require_admin)],
    tags=["admin"],
)
def audit_list(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    source_used: str | None = Query(default=None),
    audit: AuditStore = Depends(get_audit_dep),
) -> AuditPage:
    """The most recent answered questions, newest first. Admin only.

    Admin only because every row holds an employee's question and the answer
    they were given -- which is HR-sensitive by definition, and is the reason
    this endpoint exists rather than a log file anyone with shell access can
    read.
    """
    return AuditPage(
        total=audit.count(source_used=source_used),
        limit=limit,
        offset=offset,
        entries=audit.recent(limit=limit, offset=offset, source_used=source_used),
    )


@api.get(
    "/audit/{entry_id}",
    dependencies=[Depends(require_admin)],
    tags=["admin"],
)
def audit_entry(
    entry_id: int, audit: AuditStore = Depends(get_audit_dep)
) -> dict[str, Any]:
    """One answered question in full, including its decision trace."""
    entry = audit.get(entry_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit entry with id {entry_id}.",
        )
    return entry
