"""FastAPI application factory.

Deliberately minimal: only `/health`, which is what a container orchestrator and
a reverse proxy need before anything else exists. The endpoints the product
brief calls for -- /chat, /upload, /ingest, /feedback, /admin, /logs -- are the
next milestone and belong in `app.api`, each router calling `app.services`
rather than the agent directly.

An app *factory* rather than a module-level `app`, so tests can build an
instance with different settings. `run.py` and uvicorn use `app.main:create_app`
with `--factory`.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI

from app import __version__
from app.core.config import Settings, configure_logging, get_settings

TITLE = "Enterprise HR Policy & Employee Support Agentic RAG Copilot"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    api = FastAPI(
        title=TITLE,
        version=__version__,
        description=(
            "Answers employee HR questions from an approved private knowledge "
            "base, grades the evidence before answering, and falls back to web "
            "search only when internal policy does not cover the question."
        ),
    )

    @api.get("/health", tags=["ops"])
    def health() -> Dict[str, Any]:
        """Liveness plus a configuration summary.

        Reports which credentials are *missing* by name and never echoes a
        value, so this is safe to expose to a load balancer.
        """
        missing = settings.missing_secrets()
        return {
            "status": "ok" if not missing else "degraded",
            "version": __version__,
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


app = create_app()
