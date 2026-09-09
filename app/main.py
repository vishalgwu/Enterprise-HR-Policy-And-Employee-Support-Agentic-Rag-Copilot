"""FastAPI application factory.

The routes themselves live in `app.api.routes`, one router per audience: `ops`
is unprefixed and open (`/health`), `api` is prefixed `/api` and holds `/chat`
plus the admin-only `/upload` and `/audit`, and `pages` serves the console at
`/`. This module only builds the application and decides what the routes get to
depend on.

The console is served from this same application, which is what makes it
same-origin with the API -- so there is no CORS middleware here, deliberately.
Adding one would widen the surface for a request nobody is making.

Both an app *factory* and a module-level instance, and both are needed:

    create_app(settings)   builds an instance over any Settings, which is what
                           lets tests exercise production behaviour (docs
                           withheld, admin off) without touching the process
                           environment. `run.py` uses it with `--factory`.
    app                    the conventional ASGI target, so `uvicorn
                           app.main:app` in a Dockerfile or on a platform that
                           does not know about factories still works.

`app` is built at import, so importing this module reads settings and attaches
the log handler. That is correct for an entry-point module and wrong for a
library one -- which is why nothing under `app.rag` or `app.agent` does it.

**Nothing built here touches the network or the disk.** `Copilot` compiles its
graph on first use and `AuditStore` creates its schema on first use, so
importing this module does not load the embedding model, reach Pinecone, or
write a database file. `warmup()` exists for a deployment that would rather pay
that cost at startup than on the first employee's question.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api.routes import api as api_router
from app.api.routes import ops as ops_router
from app.api.routes import pages as pages_router
from app.core.config import Settings, configure_logging, get_logger, get_settings
from app.services.audit import AuditStore
from app.services.copilot import Copilot

log = get_logger("main")


def create_app(
    settings: Settings | None = None,
    copilot: Copilot | None = None,
    audit: AuditStore | None = None,
) -> FastAPI:
    """Build an application over the given settings and services.

    `copilot` and `audit` are injectable for the same reason every client below
    them is: it is what lets the API tests run offline, against a graph of fakes
    and a temporary database, rather than against Groq and Pinecone.
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
    application.state.copilot = copilot or Copilot(settings=settings)
    application.state.audit = audit or AuditStore(settings.audit_db_path)
    application.state.templates = Jinja2Templates(directory=str(settings.templates_dir))

    application.include_router(ops_router)
    application.include_router(api_router)

    # Mounted only if it is really there. `StaticFiles` raises at mount time on
    # a missing directory, which would turn "the assets were not copied into the
    # image" into "the container does not start" -- and take /health down with
    # it, so nothing could report why.
    if Path(settings.static_dir).is_dir():
        application.mount(
            "/static", StaticFiles(directory=str(settings.static_dir)), name="static"
        )
    else:
        log.warning(
            "No static directory at %s; the console will load without assets",
            settings.static_dir,
        )

    application.include_router(pages_router)
    return application


# The ASGI target for `uvicorn app.main:app`, `gunicorn -k uvicorn.workers...`
# and every platform that expects a module-level application. Tests and run.py
# call create_app() instead -- see the module docstring.
app = create_app()
