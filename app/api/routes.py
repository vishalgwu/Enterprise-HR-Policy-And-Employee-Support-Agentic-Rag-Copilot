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
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import __version__
from app.api.deps import (
    get_audit_dep,
    get_copilot_dep,
    get_settings_dep,
    require_admin,
)
from app.core.config import Settings, get_logger
from app.rag.ingest import ingest_upload
from app.services.audit import AuditStore
from app.services.copilot import AnswerResult, Copilot
from app.services.ingestion import DEFAULT_DEPARTMENT, SUPPORTED, is_supported

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


class ChatResponse(AnswerResult):
    """The agent's answer, plus where it was filed.

    Extends `AnswerResult` rather than restating it, so the API response and the
    facade's contract cannot drift. `audit_id` is null when the audit write
    failed -- the answer stands either way.
    """

    audit_id: int | None = None


@api.post("/chat", response_model=ChatResponse, tags=["copilot"])
def chat(
    payload: ChatRequest,
    copilot: Copilot = Depends(get_copilot_dep),
    audit: AuditStore = Depends(get_audit_dep),
) -> ChatResponse:
    """Answer one question from internal policy, the web, or not at all.

    The response carries the decision trace and the citations alongside the
    answer, because an HR answer that cannot be checked against its source is
    not much use to the person acting on it.
    """
    started = time.perf_counter()
    try:
        result = copilot.ask(
            payload.question,
            history=[turn.model_dump() for turn in payload.history],
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

    latency_ms = int((time.perf_counter() - started) * 1000)
    audit_id = audit.record(
        result, latency_ms=latency_ms, conversation_id=payload.conversation_id
    )
    return ChatResponse(**result.model_dump(), audit_id=audit_id)


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
