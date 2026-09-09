"""Command-line entry points.

Thin by design: a script parses arguments, calls into `app`, and prints. All
logic lives in the package, so **nothing under `app/` may import from here** --
the dependency runs one way, and an agent that reaches for a corpus script is
the specific mistake the layering table in CLAUDE.md exists to prevent.

`_cli.py` is the one module here that other scripts import, and it holds only
the preamble they all share: force UTF-8 on stdout, attach the log handler, and
return a parser that renders the caller's docstring. It is here rather than in
`app/` because it is CLI presentation, which the library layer has no business
knowing about.

Each script runs both ways -- `python scripts/demo.py` and
`python -m scripts.demo` -- because of the `__package__` bootstrap at the top of
each file, which puts the project root on sys.path when the file is executed
directly. Library modules need no such guard.
"""
