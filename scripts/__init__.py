"""Command-line entry points.

Thin by design: a script parses arguments, calls into `app`, and prints. All
logic lives in the package, so nothing here is a place another module should
import from.

Each script runs both ways -- `python scripts/demo.py` and
`python -m scripts.demo` -- because of the `__package__` bootstrap at the top of
each file, which puts the project root on sys.path when the file is executed
directly. Library modules need no such guard.
"""
