"""Start the API.

    python run.py            # http://localhost:8000, reload off
    python run.py --reload   # development

Host and port come from API_HOST / API_PORT so the container does not need a
different command line from local development.
"""

from __future__ import annotations

import sys

import uvicorn

from app.core.config import configure_logging, configure_stdout, get_settings


def main() -> None:
    configure_stdout()
    configure_logging()
    settings = get_settings()

    missing = settings.missing_secrets()
    if missing:
        # A warning, not a failure: /health is meant to come up and report the
        # problem, which is more useful to a deploy than a container that exits.
        print(f"WARNING: missing {', '.join(missing)} -- /health will report degraded")

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload="--reload" in sys.argv,
    )


if __name__ == "__main__":
    main()
