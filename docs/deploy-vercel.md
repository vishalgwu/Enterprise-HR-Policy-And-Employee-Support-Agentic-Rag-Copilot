# Deploying to Vercel

This project ships three deployment shapes. Vercel is the one that gets you a
public URL a stranger can try, which is a different problem from the one the
container solves.

| | Docker / DigitalOcean | Vercel | Local |
|---|---|---|---|
| Embeddings | local `all-MiniLM-L6-v2`, free | OpenAI, or none | either |
| Image / bundle | 2.49 GB | well under 250 MB | — |
| Audit log | SQLite on a volume, durable | `/tmp`, **lost on redeploy** | `data/audit.db` |
| Process | one, long-lived | one per request | one |
| Answer with no keys at all | yes, lexical | yes, lexical | yes, lexical |
| Suitable for | the real deployment | a public demo | development |

The honest summary: **Vercel is where this project is demonstrable, not where
it belongs.** The audit log is the feature this product exists to provide, and
a filesystem that does not survive a redeploy cannot provide it. Everything
below says so again at the point where it matters.

---

## What makes it fit at all

Two properties of the codebase, both deliberate and both easy to break.

**Nothing heavy is imported at module scope.** `import app.main` reaches only
`fastapi`, `pydantic`, `pydantic-settings`, `langchain-core`,
`langchain-text-splitters`, `langgraph` and `jinja2`. Every provider SDK is
imported inside the function that constructs its client — `get_embeddings`
branches on the provider, `get_llm` imports `langchain_groq` or
`langchain_openai`, `load_document_text` imports `pypdf` only to read a PDF. So
omitting a package costs the capability it serves rather than the deployment.

A module-scope `import langchain_huggingface` anywhere under `app/` would turn
every cold start here into a crash. If you add one, this deployment target goes
away silently.

**`sentence-transformers` is not in the Vercel requirements.** It pulls torch.
Measured on `python:3.13-slim`, torch from the default PyPI index is 1.2 GB
plus 3.2 GB of `nvidia/` wheels; the CPU index brings that to 769 MB. A
serverless function's limit is 250 MB unzipped, so this is not a tuning problem.

`api/requirements.txt` is the slim set, and Vercel's Python runtime prefers the
`requirements.txt` next to the entry point over the one at the repository root.
That is the only reason the root file can keep pinning torch for the container.

---

## Three ways to run it

Pick one before you set any variables. They differ in what a visitor gets and
in what you pay for.

### 1. No keys at all — "free mode"

Set no provider credentials. `Settings.effective_retrieval_mode` resolves to
`lexical`, and `AnswerPipeline` routes every question down the extractive path:
BM25 over the eleven markdown files in `data/private_kb/`, then the most
relevant sentences stitched together in document order and labelled as what
they are.

Verified on this repository with every key removed:

```
has_llm: False | vector: False | retrieval: lexical
"How many days of paid time off do full-time employees get?"
  -> mode: extractive, source: private_kb
  -> sources: leave-and-time-off.md, payroll-and-compensation.md
"What is the capital of France?"
  -> source: insufficient_evidence
```

The answers read worse than generated ones and are more trustworthy: nothing
wrote them, so there is nothing to have invented a number. `/health` reports
`degraded` and names the missing keys, which is correct — the deployment is
running properly in a reduced shape, and `capabilities.retrieval_mode` is how
the console knows to say so.

This mode costs nothing and cannot run up a bill. It is the right default for a
link on a CV.

### 2. Your key, their key — "demo mode"

Add the deployment's own provider key, leave `ALLOW_BYOK=1` (the default), and
leave the rate limiter on. A visitor gets a handful of answers a minute on your
account, and the console offers them a box to paste their own OpenAI or Groq
key — after which they are answered on their account, at whatever model they
pick, with no limit.

`app/services/byok.py` holds the three rules: the key is used to build one chat
client and never written anywhere; it only ever reaches the chat model, never
Pinecone or the web search; and the whole feature is off when `ALLOW_BYOK=0`.

Set `PUBLIC_DEMO=1`. It makes the console say which model answered and
advertise the shared-key quota rather than hiding it.

### 3. Fully keyed

Pinecone plus OpenAI embeddings plus a chat provider. The same behaviour as the
container, minus the durable audit log. See the variables below.

---

## Environment variables

Set these in the Vercel project settings, not in a file. `.vercelignore`
excludes `.env` from the build deliberately.

### Always

```
APP_ENV=production
AUDIT_DB_PATH=/tmp/hr-audit.db
STATE_DB_PATH=/tmp/hr-state.db
UPLOAD_DIR=/tmp/uploads
```

`/tmp` is the only writable path. The three stores are settings precisely so
they can be pointed there — `kb_dir`, `upload_dir` and `audit_db_path` were
made configurable for the container's volume mounts and the same seam serves
here.

**`APP_ENV=production` makes `ADMIN_API_KEY` mandatory.** `create_app` calls
`Settings.validate_admin_surface()`, which refuses to build without one. That
is intentional — a production deployment nobody can administer is not a
degraded state, it is a broken one — but on Vercel it surfaces as a function
that 500s at cold start with the reason only in the logs. Either set the key or
leave `APP_ENV` unset.

### Budgets, set under the function's ceiling

```
MAX_AGENT_SECONDS=40
```

`vercel.json` sets `maxDuration: 60`. A budget that outlives its function
produces a request killed mid-answer, which the client reports as a cut stream
rather than as a slow one — `app/services/streaming.py` explains how it tells
the difference. Keep the agent budget comfortably below the platform ceiling.

On a Hobby plan `maxDuration` cannot exceed 60 seconds. Lower
`MAX_AGENT_SECONDS` further if you see truncated streams.

### Retrieval

Pick one.

```
# Dense + lexical, fused. Needs both.
PINECONE_API=...
PINECONE_INDEX=hr-copilot-1536
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=...
```

```
# Or nothing at all, which resolves to lexical.
RETRIEVAL_MODE=lexical
```

**The index is 1536-d, not 384-d.** `text-embedding-3-small` emits 1536-wide
vectors and `all-MiniLM-L6-v2` emits 384, and Pinecone cannot resize an index.
A Vercel deployment therefore needs its *own* index, ingested with the OpenAI
embeddings, and cannot share the one a local `huggingface` setup built. `Settings`
derives `embedding_dim` from the active model and `validate_embedding()`
rejects a contradiction rather than letting it fail as an opaque upsert error.

Ingest from a machine, not from the function:

```bash
EMBEDDING_PROVIDER=openai PINECONE_INDEX=hr-copilot-1536 \
  python scripts/ingest_private_kb.py
```

Ingest reconciles a whole namespace and prunes orphans. That is not a thing to
do per request, which is why `langchain-community` and the web loader are
deliberately absent from `api/requirements.txt`.

### The model

```
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API=...
```

or

```
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

Groq's free tier caps tokens per **day**, not per minute, so exhaustion does
not clear in a few seconds. `is_rate_limit` distinguishes it and the affected
paths log at ERROR, because every safe default in this system is "weak" and an
exhausted quota otherwise looks exactly like a knowledge gap.

### Public demo

```
PUBLIC_DEMO=1
ALLOW_BYOK=1
RATE_LIMIT_ENABLED=1
RATE_LIMIT_PER_MINUTE=6
RATE_LIMIT_BURST=3
```

### Operator access

```
ADMIN_API_KEY=<long random string>
HR_API_KEY=<a different long random string>
```

Both default to empty, and empty means that role **cannot be assumed** —
including by a request presenting an empty header. A reference implementation
once shipped `"change-me-in-production"` as the admin default, which is worse
than nothing: forgetting to set it leaves the admin API open behind a string
published in the source.

### Optional

```
AGENT_MODE=auto           # graph | react | supervisor | auto
CACHE_ENABLED=1
MODEL_ROUTING_ENABLED=1
TRACE_SPANS=1
TRACE_PAYLOADS=0          # leave off: spans go onto the HTTP response
LANGSMITH_TRACING=0       # uploads questions and policy text to a third party
```

---

## Deploying

```bash
npm i -g vercel
vercel link
vercel env add APP_ENV production     # repeat per variable, or use the dashboard
vercel --prod
```

`vercel.json` rewrites every path to `api/index.py`, so one function serves the
console, the API and the static assets. `includeFiles` carries `templates/`,
`static/` and `data/private_kb/` into the bundle — without that last one the
lexical retriever has nothing to index and every answer is
`insufficient_evidence`.

Serving `/static` through the function means a static asset costs an invocation.
For a demo that is the right trade against duplicating the files into `public/`
and maintaining two copies; `vercel.json` sets a one-hour `Cache-Control` on
them so a repeat visitor does not pay it again.

---

## Checking it worked

```bash
curl -s https://<your-app>.vercel.app/health | jq '.status, .capabilities'
```

`capabilities.retrieval_mode` is the field to read: `lexical` means Pinecone is
not reachable and the deployment has degraded to BM25, which may be exactly
what you intended. `missing_secrets` names what is absent, by variable name,
and never echoes a value.

Then ask it something:

```bash
curl -s https://<your-app>.vercel.app/api/chat \
  -H 'content-type: application/json' \
  -d '{"question":"How many days of paid time off do full-time employees get?"}' \
  | jq '{answer, answer_mode, source_used, model, sources: [.sources[].source]}'
```

The console's **Dashboard** and **Settings** pages are the quickest check that
everything is wired: Settings shows the resolved configuration the deployment
actually settled on, and Dashboard shows whether answers are landing grounded.

---

## What does not work here, and why

**The audit log does not survive.** `/tmp` is per-instance and goes on redeploy
or when an idle instance is recycled. For a product whose audit log exists to
answer "what did it tell someone about the notice period in March?", that is
the failure the log exists to prevent. A deployment that needs it needs a
database outside the function — Postgres, Turso, Neon — and `AuditStore` would
need a driver rather than `sqlite3`. The container does not have this problem.

**The rate limit is per instance.** `RateLimiter` is a token bucket in process
memory, and Vercel runs several instances. The effective limit is
`instances × rate`, and a caller who reconnects may land on a fresh bucket.
`stats()` reports `"scope": "process"` so a dashboard does not claim otherwise.
A real limit belongs in front of the application — Vercel's own edge limiting,
or an Upstash counter.

**The semantic cache is per instance**, for the same reason, so the hit rate
falls roughly as the instance count rises. Correctness is unaffected; only the
saving is.

**The idempotency ledger is per instance** unless `STATE_DB_PATH` points at
shared storage, which `/tmp` is not. `app/services/idempotency.py` says so
rather than pretending the in-memory mode is a distributed lock. In practice
this bounds how much the `raise_ticket` guarantee is worth on this platform;
the approval gate in front of it is doing most of the work.

**Cold starts.** A function that has not run recently pays to import
`langgraph` and build a `Settings`, then on the first question to construct a
retriever. With `RETRIEVAL_MODE=lexical` that includes loading and chunking the
eleven markdown files — cached per instance by `build_kb_index`'s `lru_cache`,
so it is the first request of an instance and not every request.

**Uploads do not persist.** The file lands in `/tmp` and the vectors go to
Pinecone, so the *content* survives and the file does not. For a demo that is
the right half to keep.

---

## If it fails

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | the repo root is not on `sys.path` | `api/index.py` inserts it; check the file was deployed |
| 500 at cold start, logs mention `ADMIN_API_KEY` | `APP_ENV=production` with no admin key | set one, or unset `APP_ENV` |
| Every answer is `insufficient_evidence` | `data/private_kb/` was not bundled | check `includeFiles` in `vercel.json` |
| Build exceeds the size limit | the root `requirements.txt` was used | `api/requirements.txt` must sit beside `api/index.py` |
| `401 UNAUTHENTICATED` from Pinecone | a quoted value in an env var | Vercel's dashboard stores values raw; do not wrap them in quotes |
| Streams cut off part way | `MAX_AGENT_SECONDS` at or above `maxDuration` | lower the budget |
| Answers work, `/api/audit` is empty | the instance that answered was recycled | expected on `/tmp`; see above |
