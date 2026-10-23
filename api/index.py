"""Vercel's entry point: the same ASGI application, nothing else.

Vercel's Python runtime looks for a module under `api/` and serves whatever
ASGI or WSGI callable it exports. `vercel.json` rewrites every path here, so
this one function serves the console, the API and the static assets -- which is
the same shape as the container, just with the process started per request
instead of once.

**Nothing is configured here.** Everything a deployment needs to say comes from
environment variables, which `app/core/config.py` already reads, because a
second place that sets `UPLOAD_DIR` is a second place for it to disagree with
the first. `docs/deploy-vercel.md` lists what to set and why each one matters.

Three things about this platform that the code already handles, listed here
because this file is where somebody looks when it does not work:

**The filesystem is read-only except `/tmp`.** `audit_db_path`, `state_db_path`
and `upload_dir` are settings precisely so they can be pointed there. `/tmp`
does not survive a redeploy or an idle instance being recycled, so the audit
log on Vercel is a demonstration of the feature rather than a compliance
record -- which, for a product whose audit log exists to answer "what did it
tell someone in March?", is a real limitation and not a detail. A deployment
that needs the log needs a database outside the function.

**torch cannot be installed here.** `api/requirements.txt` therefore omits
`sentence-transformers`, so `EMBEDDING_PROVIDER=huggingface` has nothing to
load. Either set `EMBEDDING_PROVIDER=openai` with a key, or leave Pinecone
unconfigured entirely and let `effective_retrieval_mode` resolve to `lexical` --
BM25 over the eleven markdown files, which needs no key, no model and no
network, and which is why there is a deployable demo at all.

**A function has a wall-clock ceiling.** `vercel.json` sets `maxDuration`, and
`MAX_AGENT_SECONDS` should be set below it: a budget that outlives its function
produces a request killed mid-answer, which the client reports as a cut stream
rather than as a slow one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repository root, so `import app` resolves. Vercel invokes this module
# from inside `api/`, and without this the package one directory up is not on
# the path -- which fails as `ModuleNotFoundError: No module named 'app'` at
# cold start, with no other clue about what is wrong.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

# Vercel's runtime looks for `app` (or `handler`). Both names point at the same
# object: `app` is the convention this project already uses everywhere, and
# `handler` is what some of Vercel's own examples show.
handler = app

__all__ = ["app", "handler"]
