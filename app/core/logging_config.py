"""Logging and console setup for the whole package.

Split out of `config.py` so the logging policy lives in one small file, but
`config` re-exports every name here -- `from app.core.config import
configure_logging` is still the supported import, and nothing else in the
package imports this module directly.

Two decisions differ from the usual `logging.basicConfig(...)` recipe, and both
are deliberate:

1. **Records go to a package logger, not the root logger.** `basicConfig`
   configures the *root*, which means httpx, urllib3, pinecone and the LangChain
   internals all start printing at INFO through this app's handler. Attaching to
   `hr_copilot` keeps the output this project's own.

2. **The handler writes to stderr.** stdout stays clean for the answer itself,
   so `python scripts/demo.py > out.txt` captures results without the per-node
   trace.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "hr_copilot"

# The node trace is read as prose while a question runs, so the default carries
# no level or timestamp. DETAILED_FORMAT is there for a server, where a record
# without a timestamp is worth much less.
MESSAGE_FORMAT = "%(message)s"
DETAILED_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# A library must not configure logging for its host. NullHandler keeps these
# records silent until an entry point calls configure_logging().
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Logger for one module, under the package's shared root."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO, fmt: str = MESSAGE_FORMAT) -> None:
    """Send this package's logs to stderr. Entry points only, never on import.

    Idempotent: a second call re-uses the handler already attached, so an entry
    point that calls it and then imports another one does not double every line.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt))
    logger.setLevel(level)
    logger.propagate = False


def configure_stdout() -> None:
    """Print UTF-8 regardless of the console code page.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on the
    non-breaking hyphens, em dashes and smart quotes that LLM replies routinely
    contain. The API call succeeds and the crash lands on print().
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
