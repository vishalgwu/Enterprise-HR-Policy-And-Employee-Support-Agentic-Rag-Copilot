"""FastAPI application factory.

Deliberately minimal: only `/health`, which is what a container orchestrator and
a reverse proxy need before anything else exists. The endpoints the product
brief calls for -- /chat, /upload, /ingest, /feedback, /admin, /logs -- are the
next milestone and belong in `app.api`, each router calling `app.services`
rather than the agent directly.

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
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from app import __version__
from app.core.config import Settings, configure_logging, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    api = FastAPI(
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

    @api.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        """Liveness plus a configuration summary.

        Reports which credentials are *missing* by name and never echoes a
        value, so this is safe to expose to a load balancer.
        """
        missing = settings.missing_secrets()
        return {
            "status": "ok" if not missing else "degraded",
            "app": settings.app_name,
            "env": settings.app_env,
            "version": __version__,
            # Whether the admin routes will accept anything at all. They fail
            # closed without a key, and silent-off is exactly the state worth
            # being able to see from outside.
            "admin_enabled": settings.admin_enabled,
            "missing_secrets": missing,
            "pinecone_index": settings.pinecone_index,
            "private_namespace": settings.private_namespace,
            # Which providers this deployment actually resolved to. Reported
            # because "which model answered?" is the first question about a bad
            # answer, and the second is "was that the provider we thought?".
            "llm_provider": settings.llm_provider,
            "llm_model": settings.active_llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.active_embedding_model,
            "embedding_dim": settings.embedding_dim,
            # Reported because tracing sends questions and retrieved policy text
            # off the deployment; whether it is on should be visible, not
            # inferred from a config file nobody reads in production.
            "tracing": settings.tracing_enabled,
            # Whether traces carry the employee's question and the retrieved
            # policy text. Surfaced so "we turned redaction off to debug" cannot
            # quietly become the permanent state of production.
            "tracing_redacted": settings.tracing_redacted,
        }

    return api


# The ASGI target for `uvicorn app.main:app`, `gunicorn -k uvicorn.workers...`
# and every platform that expects a module-level application. Tests and run.py
# call create_app() instead -- see the module docstring.
app = create_app()
