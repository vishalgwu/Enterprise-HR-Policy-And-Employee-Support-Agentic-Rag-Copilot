"""FastAPI application factory.

The routes themselves live in `app.api.routes`, one router per audience: `ops`
is unprefixed and open (`/health`), `api` is prefixed `/api` and holds the
employee-facing `/chat` plus the gated admin and HR surfaces, and `pages` serves
the console at `/`. This module only builds the application and decides what
the routes get to depend on.

The console is served from this same application, which is what makes it
same-origin with the API -- so there is no CORS middleware here, deliberately.
Adding one would widen the surface for a request nobody is making.

Both an app *factory* and a module-level instance, and both are needed:

    create_app(settings)   builds an instance over any Settings, which is what
                           lets tests exercise production behaviour (docs
                           withheld, admin off) without touching the process
                           environment. `run.py` uses it with `--factory`.
    app                    the conventional ASGI target, so `uvicorn
                           app.main:app` in a Dockerfile, or `api/index.py` on
                           Vercel, still works.

`app` is built at import, so importing this module reads settings and attaches
the log handler. That is correct for an entry-point module and wrong for a
library one -- which is why nothing under `app.rag` or `app.agent` does it.

**Nothing built here touches the network or the disk.** Every store creates its
schema on first use and `AnswerPipeline` builds its retriever on first use, so
importing this module does not load the embedding model, reach Pinecone, or
write a database file. `test_building_the_app_touches_neither_network_nor_disk`
pins it, and the list of services below has grown enough that the test is now
the only thing keeping that true.

**One SQLite file for working state, a second for the audit log.** The audit log
is the compliance record and must not share a backup policy with a table that
churns on every request; `app/services/state_db.py` argues it at length. Both
paths are settings, because the target architecture mounts them as volumes.

**The BYOK pool shares the expensive clients and not the cheap ones.** A caller
spending their own provider key gets their own `AnswerPipeline` -- their own
chat model, their own semantic cache -- over the *shared* retriever and web
search. The retriever is the expensive one (it loads an embedding model) and it
holds the deployment's own data, which a caller's key has no business paying
for; the cache is cheap and is deliberately not shared, so one visitor cannot
discover another visitor's question by probing for a hit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api.routes import api as api_router
from app.api.routes import ops as ops_router
from app.api.routes import pages as pages_router
from app.core.config import Settings, configure_logging, get_logger, get_settings
from app.services.approval import ApprovalStore
from app.services.audit import AuditStore
from app.services.authorization import PermissionDenied
from app.services.byok import ClientPool, ProviderCredentials
from app.services.cache import SemanticCache
from app.services.copilot import Copilot
from app.services.idempotency import IdempotencyStore
from app.services.memory_store import FactMemory
from app.services.pipeline import AnswerPipeline
from app.services.ratelimit import RateLimiter
from app.services.scratchpad import Scratchpad
from app.services.state_db import StateStore

log = get_logger("main")


class _RevalidatingStatic(StaticFiles):
    """Static assets that must be revalidated, not assumed fresh.

    The console is nine ES modules and two stylesheets served from `/static`,
    and `app.js` imports its siblings by bare path -- so there is no build step
    to hash a filename with, and a `?v=` query on the entry point does not
    reach the modules it imports.

    Without this a browser applies *heuristic* freshness: no `Cache-Control`
    and a `Last-Modified` from a while ago means it may reuse the file for
    hours without asking. That is invisible in development and actively
    misleading after a deploy -- the symptom is a release that half-applies,
    with new markup against an old stylesheet, which reads as a broken build
    rather than as a cache.

    `no-cache` does not mean "do not store"; it means "ask before using". The
    answer is almost always a 304 against the ETag `StaticFiles` already sends,
    so the cost is a conditional request per asset and the bytes are still
    cached. For a console whose assets are a few hundred kilobytes that is the
    right trade; a product serving megabytes of media would want hashed
    filenames and a year-long max-age instead.

    **One caveat, because it is confusing when you hit it.** This only governs
    clients that fetch *after* it is in place. A browser that cached a module
    before this existed is still sitting on a heuristically-fresh copy and will
    not ask, so it keeps running the old file — which looks exactly like the
    deploy not having happened. A hard reload fixes that one profile; nothing
    server-side can, which is the argument for hashed filenames if this ever
    grows a build step.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(
    settings: Settings | None = None,
    copilot: Copilot | None = None,
    audit: AuditStore | None = None,
    state: StateStore | None = None,
) -> FastAPI:
    """Build an application over the given settings and services.

    `copilot`, `audit` and `state` are injectable for the same reason every
    client below them is: it is what lets the API tests run offline, against a
    graph of fakes and a temporary database, rather than against Groq and
    Pinecone.
    """
    settings = settings or get_settings()
    configure_logging()

    # The one configuration rule the *server* enforces, rather than reporting.
    # A missing provider key is deliberately not fatal -- /health comes up and
    # says `degraded`, which is more use to a deploy than a container that
    # exits. An unset admin key in production is different: it is not a degraded
    # state, it is a deployment nobody can administer, and both this README and
    # the console claim it refuses to boot. It did not, until this call existed:
    # `validate_required` holds the rule but only the CLI scripts ever called
    # it, so `python run.py` and `uvicorn app.main:app` both sailed past it.
    settings.validate_admin_surface()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Answers employee HR questions from an approved private knowledge "
            "base, grades the evidence before answering, and falls back to web "
            "search only when internal policy does not cover the question."
        ),
        # The interactive docs enumerate every route, including the admin ones,
        # and are not something an employee-facing deployment needs to publish.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    # The routes read these back off the request; see `app.api.deps`. Held on
    # `state` rather than in `dependency_overrides`, which is a test hook.
    application.state.settings = settings

    state_store = state or StateStore(settings.state_db_path)
    application.state.state_store = state_store
    application.state.scratchpad = Scratchpad(state_store)
    application.state.facts = FactMemory(state_store)
    application.state.approvals = ApprovalStore(state_store)
    application.state.idempotency = IdempotencyStore(state_store)
    application.state.audit = audit or AuditStore(
        settings.audit_db_path, store_redacted=settings.audit_stores_redacted
    )
    application.state.limiter = RateLimiter(settings=settings)

    # The deployment's own cache, shared by every caller using the deployment's
    # own key. A caller-supplied key gets its own, built in the pool factory.
    application.state.cache = SemanticCache(settings=settings)

    application.state.copilot = copilot or Copilot(
        settings=settings,
        pipeline=AnswerPipeline(
            settings,
            cache=application.state.cache,
            scratchpad=application.state.scratchpad,
            facts=application.state.facts,
            approvals=application.state.approvals,
            idempotency=application.state.idempotency,
        ),
    )
    # An *injected* copilot built its own pipeline, which defaulted to a private
    # cache and no stores -- so `/api/cache/invalidate` would drop a cache
    # nothing reads from and an answer would come back with no scratchpad.
    # `adopt_services` fills only what defaulted, so a caller that passed its
    # own keeps it. This is what makes a copilot a seam for the *graph* rather
    # than a way to disconnect the application from its own state.
    #
    # `getattr` rather than an attribute access, because the contract for an
    # injected copilot is `.ask` and nothing more -- a test's stand-in that only
    # raises is a legitimate thing to pass here, and `create_app` refusing to
    # build over one would make the error-path test impossible to write.
    adoptable = getattr(application.state.copilot, "pipeline", None)
    if adoptable is not None and hasattr(adoptable, "adopt_services"):
        adoptable.adopt_services(
            cache=application.state.cache,
            scratchpad=application.state.scratchpad,
            facts=application.state.facts,
            approvals=application.state.approvals,
            idempotency=application.state.idempotency,
        )

    def byok_copilot(
        resolved: Settings, credentials: ProviderCredentials | None
    ) -> Copilot:
        """A copilot for one caller's provider key.

        Shares the retriever and the web-search client with the deployment's
        own pipeline -- reached through the base pipeline's lazy properties, so
        the embedding model still loads on the first question rather than at
        startup -- and gets everything else of its own.
        """
        base = application.state.copilot
        if credentials is None:
            return base
        shared = base.pipeline
        return Copilot(
            settings=resolved,
            pipeline=AnswerPipeline(
                resolved,
                retriever=shared.retriever,
                web_search=shared.web_search,
                cache=SemanticCache(settings=resolved),
                scratchpad=application.state.scratchpad,
                facts=application.state.facts,
                approvals=application.state.approvals,
                idempotency=application.state.idempotency,
            ),
        )

    application.state.pool = ClientPool(settings, byok_copilot)
    application.state.templates = Jinja2Templates(directory=str(settings.templates_dir))

    # A service may raise `PermissionDenied` outside a dependency -- a tool
    # handler, a store called from a route body. Answered here rather than
    # caught per route, and with `str(exc)`, which is safe: the message is built
    # from the action and the role and never holds key material.
    @application.exception_handler(PermissionDenied)
    async def _permission_denied(
        _request: Request, exc: PermissionDenied
    ) -> JSONResponse:
        log.warning("Permission denied: %s", exc)
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    application.include_router(ops_router)
    application.include_router(api_router)

    # Mounted only if it is really there. `StaticFiles` raises at *mount* time
    # on a missing directory, which would turn "the assets were not copied into
    # the image" into "the container does not start" -- and take /health down
    # with it, so nothing could report why.
    if Path(settings.static_dir).is_dir():
        application.mount(
            "/static",
            _RevalidatingStatic(directory=str(settings.static_dir)),
            name="static",
        )
    else:
        log.warning(
            "No static directory at %s; the console will load without assets",
            settings.static_dir,
        )

    application.include_router(pages_router)
    return application


def state_of(request: Any, name: str, default: Any = None) -> Any:
    """Read one service off `app.state`, tolerating its absence.

    For a route that is useful on a reduced application -- one a test built
    without the full set of stores. The alternative is every such route
    growing its own `getattr` dance.
    """
    return getattr(request.app.state, name, default)


# The ASGI target for `uvicorn app.main:app`, `gunicorn -k uvicorn.workers...`,
# Vercel's `api/index.py`, and every platform that expects a module-level
# application. Tests and run.py call create_app() instead -- see the module
# docstring.
app = create_app()
