"""Start the API and the console.

    python run.py                      # http://localhost:8000, reload off
    python run.py --reload             # development
    python run.py --port 8080          # override API_PORT for one run
    python run.py --help               # flags, without starting anything

Host and port default to API_HOST / API_PORT so a container needs no different
command line from local development; the flags are for overriding one run.

`--factory` rather than a module-level target: `create_app()` is what builds an
application over explicit Settings, and using it here keeps the path this
process takes the same as the one the tests exercise. `uvicorn app.main:app`
still works for a platform that knows nothing about factories.
"""

from __future__ import annotations

import argparse

import uvicorn

from app.core.config import (
    Settings,
    configure_logging,
    configure_stdout,
    get_settings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse flags *before* anything reads settings or a credential.

    That ordering is the point, and it is the house rule for every entry point
    here. `--reload` used to be read as `"--reload" in sys.argv`, which meant
    `--help` matched nothing, was ignored, and started a server on someone
    asking what the flags were.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=None, help="override API_HOST")
    parser.add_argument("--port", type=int, default=None, help="override API_PORT")
    parser.add_argument(
        "--reload", action="store_true", help="restart on source changes (development)"
    )
    return parser.parse_args(argv)


def describe(settings: Settings) -> list[str]:
    """The handful of facts worth seeing in the terminal before the first request.

    Everything here is also on `/health`; this is the same information at the
    moment someone is actually looking. It reports *names* only -- never a
    secret value -- so it is safe in a log a deployment captures.
    """
    admin_state = "enabled" if settings.admin_enabled else "OFF (no ADMIN_API_KEY)"
    lines = [
        f"{settings.app_name} - {settings.app_env}",
        f"  llm         {settings.llm_provider} / {settings.active_llm_model}",
        f"  embeddings  {settings.embedding_provider} / "
        f"{settings.active_embedding_model} ({settings.embedding_dim}d)",
        f"  pinecone    {settings.pinecone_index} / {settings.private_namespace}",
        f"  admin api   {admin_state}",
    ]

    # LangSmith uploads every question and every retrieved chunk to LangChain's
    # servers -- for this product, employee HR questions and internal policy.
    # Whether that is happening should be visible at the moment the process
    # starts, not inferred from a config file nobody reads.
    if not settings.tracing_enabled:
        lines.append("  langsmith   off")
    elif settings.tracing_redacted:
        lines.append(
            f"  langsmith   on, redacted -> project {settings.langsmith_project!r} "
            "(run tree, timings, tokens; no question, policy, decision or answer)"
        )
    else:
        lines.append(
            f"  langsmith   ON, *UNREDACTED* -> project {settings.langsmith_project!r} "
            "-- questions and retrieved policy text are leaving this deployment"
        )
    return lines


def main() -> None:
    args = parse_args()

    configure_stdout()
    configure_logging()
    settings = get_settings()

    for line in describe(settings):
        print(line)

    missing = settings.missing_secrets()
    if missing:
        # A warning, not a failure: /health is meant to come up and report the
        # problem, which is more useful to a deploy than a container that exits.
        print(f"WARNING: missing {', '.join(missing)} -- /health will report degraded")

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"  console     http://{'localhost' if host == '0.0.0.0' else host}:{port}/")

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
