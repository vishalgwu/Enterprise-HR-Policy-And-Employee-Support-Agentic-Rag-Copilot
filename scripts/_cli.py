"""The preamble every script in this directory shares.

Six scripts opened with the same four lines: force UTF-8 on stdout, attach the
log handler, then build an `ArgumentParser` that renders the module docstring
verbatim. Repeating that is not just noise -- it is repeating an *ordering rule*
that has already been got wrong twice in this project:

    `scripts/demo.py` read `sys.argv` by hand, so `--help` matched nothing, was
    ignored, and fired four live Groq/Pinecone/Tavily round trips at someone
    asking what the flags were. `run.py` had the same bug for longer --
    `reload="--reload" in sys.argv` -- so `python run.py --help` *started a
    server*.

The rule is: **set up the console, parse arguments, and only then look at a
credential.** `parse_args()` exits on `--help` before `validate_required()` is
reached, which is what keeps `--help` working in a checkout with no `.env`.
Holding the first half of that in one function means a new script inherits it
rather than re-deriving it.

`validate_required()` stays at the call site deliberately. It is the line that
says "this script cannot work without credentials", and two scripts here
genuinely can work without them -- `inspect_document.py` and `render_graph.py`
are offline by design. Hiding that behind a shared helper would make the
offline ones look like an oversight rather than a choice.

Imported by scripts, never by `app`. The dependency runs one way: entry points
import the package, and nothing in the package imports an entry point.
"""

from __future__ import annotations

import argparse

from app.core.config import configure_logging, configure_stdout


def parser_for(doc: str | None) -> argparse.ArgumentParser:
    """Prepare the console and return a parser that prints `doc` for `--help`.

    `RawDescriptionHelpFormatter` because every script's docstring is already
    laid out as usage lines followed by prose; the default formatter reflows it
    into one paragraph and loses the commands.

    `configure_stdout()` is here rather than in `main()` because a Windows
    console defaults to cp1252 and raises `UnicodeEncodeError` on the
    non-breaking hyphens and smart quotes that both LLM replies and these
    docstrings contain -- the failure lands on `print()`, after the work is
    done. `configure_logging()` sends the node trace to stderr, so stdout stays
    clean enough to redirect.
    """
    configure_stdout()
    configure_logging()
    return argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
