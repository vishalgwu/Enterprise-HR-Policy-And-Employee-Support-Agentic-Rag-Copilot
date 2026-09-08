# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An Agentic RAG copilot for internal HR policy and employee support. The reference brief and target
architecture are checked in: `docs/Enterprise_HR_Agentic_RAG_Problem_Statement_DigitalOcean.pdf`
and `docs/architecture.png`. Read them before designing anything new — they name the endpoints, the
data layer, the roles and the deployment target.

The retrieval stack, the 10-node LangGraph agent, the HTTP surface and the browser console are
complete: `/health`, `/api/chat`, the admin-gated `/api/upload` and `/api/audit`, a SQLite audit
log, and a single-page UI at `/` served from `templates/` and `static/`. Docker/DigitalOcean
packaging is not built.

## Environment and commands

Windows-first. The virtualenv lives in `hr/` (Python 3.13, created from Anaconda) and is
self-ignored via `hr/.gitignore`.

```bash
hr\Scripts\activate                    # PowerShell / cmd
pip install -r req.txt                 # req.txt is the source of truth; requirements.txt is "-r req.txt"

pytest                                 # 214 tests, offline, no credentials needed
python -m ruff check app scripts tests run.py ingest_sample_kb.py    # must be clean

python scripts/ingest_private_kb.py    # data/private_kb/** -> "hr-docs"
python scripts/ingest_public_web.py    # scraped article     -> "public-web"
python scripts/demo.py                 # four demos, one per graph path
python scripts/check_retrieval.py      # gated vs ungated retrieval scores
python scripts/inspect_document.py     # load + chunk one file, offline, no keys
python scripts/render_graph.py         # regenerate docs/agent-graph.md
python run.py                          # uvicorn on :8000
```

**ruff is configured in `pyproject.toml`, pinned in `req.txt`, and the tree passes with zero
findings.** Keep it that way; it is a real gate now, not an aspiration. Pinned deliberately — a
minor release adds rules, and a lint gate that starts failing on an unchanged tree is one people
learn to ignore.

**Every entry point parses arguments before it validates credentials, and that ordering is the
point.** `scripts/demo.py` used to read `sys.argv` by hand, so `--help` matched nothing, was
ignored, and fired four live Groq/Pinecone/Tavily round trips at someone asking what the flags
were. `run.py` had the identical bug for longer — `reload="--reload" in sys.argv`, so `python run.py
--help` *started a server* — while this file and the README both claimed every script supported
`--help`. Everything uses `argparse` now, `--help` works in a checkout with no `.env`, and
`tests/test_run.py` pins it so the claim and the code cannot drift apart again.

**`run.py` prints what the process resolved to before the first request.** `describe(settings)`
reports the providers, the index and namespace, whether the admin surface is on, and — the reason it
exists — whether LangSmith is uploading, to which project, and whether payloads are redacted.
Everything there is also on `/health`; this is the same information at the moment someone is
actually looking. It prints **names only, never a secret value**, because it goes to stdout and into
whatever the deployment captures; `test_the_banner_never_prints_a_secret_value` pins that, and
unredacted tracing is deliberately shouted about rather than mentioned.

## Module layering

Lowest first; each layer imports only from those above it, and **nothing imports from `scripts/`**.

| Module | Responsibility |
|---|---|
| `app/core/logging_config.py` | `LOGGER_NAME`, `get_logger`, `configure_logging`, `configure_stdout` |
| `app/core/config.py` | `Settings`, plus a re-export of the logging names. No heavy imports. |
| `app/rag/clients.py` | embeddings (cached), chat model, Tavily search — the only place providers branch |
| `app/rag/loaders.py` | markdown/text/PDF/DOCX/web loading, chunking, chunk ids, evidence rendering |
| `app/rag/vectorstore.py` | Pinecone, parameterised by **index and namespace** |
| `app/rag/ingest.py` | load → chunk → embed → upsert → reconcile, for any corpus |
| `app/rag/retrieval.py` | the private KB and public corpus, namespaces already chosen |
| `app/agent/schemas.py` | decision models and `AgentState` |
| `app/agent/prompts.py` | every prompt |
| `app/agent/decisions.py` | router and grader, with recovery |
| `app/agent/nodes.py` | node bodies, as closures |
| `app/agent/graph.py` | wiring and compilation |
| `app/services/ingestion.py` | load -> chunk -> index, as one door. A facade over `rag`, never a second loader |
| `app/services/copilot.py` | `Copilot` facade and `AnswerResult` — the API contract |
| `app/services/audit.py` | `AuditStore` — the SQLite record of every answered question |
| `app/api/deps.py` | what a route depends on, read off `app.state` |
| `app/api/routes.py` | the `ops` (`/health`) and `api` (`/api/*`) routers |
| `app/main.py` | `create_app` — builds the app, wires services onto `app.state`, mounts `/static` |
| `templates/index.html` | the console, one Jinja2 page |
| `static/css/style.css` | tokens, components, both themes |
| `static/js/app.js` | the console's behaviour; no framework, no build step |
| `scripts/*` | thin CLI callers |

There is no `test.py` or `data/sample_kb/` at the repo root; both existed briefly and were removed.
`test.py` shadowed the stdlib `test` package on `sys.path` and was never collected (`testpaths =
["tests"]`) — it is now `scripts/inspect_document.py`. `data/sample_kb/` was a second HR corpus
overlapping `data/private_kb/` on onboarding, offboarding and payroll, in different vocabulary
("HR portal" against the KB's "HR Service Desk" / "Workday"). See **The KB must not contradict
itself** below: that is the failure grading cannot catch, because each chunk looks fine alone.

An earlier version had `graph.py` importing `get_kb_retriever` from `private_kb.py` — an entry
point. That is the specific mistake this table exists to prevent: **the agent must not depend on a
corpus script.** Retrieval accessors belong in `app/rag/retrieval.py`.

Every `vectorstore` operation takes a namespace *and* an index. Do not add a namespace-specific
copy of an ingest or retrieval helper — pass the namespace instead.

Scripts run both ways (`python scripts/demo.py` and `python -m scripts.demo`) because of the
`if __package__ in (None, "")` bootstrap at the top of each one, which puts the project root on
`sys.path`. Library modules use plain `from app.x import y` and need no such guard.

## Secrets

`.env` at the repo root holds `GROQ_API`, `TAVILY_API`, `PINECONE_API`, `OPENAI_API_KEY`,
`PINECONE_INDEX`, `PINECONE_CLOUD`, `PINECONE_REGION` and the `LANGSMITH_*` group. It is gitignored
and has never been committed; [.env.example](.env.example) documents the shape. Which keys are
*required* depends on the configured providers — see `required_secret_fields()`.

**Missing credentials must not break import.** `Settings` defaults every secret to `""` and
`Settings.require()` raises where a client is constructed. This is deliberate: an earlier version
raised at import time, which made the package unimportable without secrets — no tests, no `--help`,
and no container that gets its configuration from the platform rather than a file. Entry points
call `validate_required()` **after** `parse_args()` so `--help` still works in a fresh checkout.

**`export_client_env()` runs when `app.core.config` is imported.** It copies the short `.env` names
into the `GROQ_API_KEY` / `TAVILY_API_KEY` / `PINECONE_API_KEY` forms that ChatGroq, TavilySearch
and Pinecone read *in their constructors*, plus the `USER_AGENT` that `WebBaseLoader` requires.
Every client module imports its settings from here, which is what guarantees the ordering. Do not
construct a client before importing config, and do not move the export.

**The thing that makes that happen is `settings = get_settings()` at the foot of `config.py`, and
nothing imports it.** It has no readers by design — callers use `get_settings()` or build their own
`Settings`, so they are not coupled to import order — which means it reads exactly like dead code
to a linter, to a reviewer, and to the next cleanup pass. It is not: it is the single call that
runs `export_client_env()` on import. Delete it and every provider client breaks at once, silently,
at construction time, because `GROQ_API_KEY` and friends are simply never in the environment. The
comment above it says so; leave both in place.

**`app/main.py` defines both `create_app()` and a module-level `app`, and both are load-bearing.**
The factory is what lets a test build an instance over arbitrary `Settings` — that is how
production behaviour (docs withheld, admin key mandatory) is exercised without touching the process
environment — and `run.py` uses it with `--factory`. The module-level `app` is the conventional
ASGI target, so `uvicorn app.main:app` in a Dockerfile or on a platform that knows nothing about
factories still works. Building `app` at import means importing `app.main` reads settings and
attaches the log handler; that is correct for an entry-point module and wrong for a library one,
which is why nothing under `app/rag/` or `app/agent/` does it.

**pydantic-settings reads `.env` but does not export it, unlike `load_dotenv()`.** This bit once:
switching to `SettingsConfigDict(env_file=...)` silently killed LangSmith tracing, because the old
`load_dotenv()` had been pushing `LANGSMITH_TRACING` into `os.environ` where LangChain finds it, and
the new loader does not. **Any variable a third-party library discovers through the environment must
be an explicit field plus an explicit export in `export_client_env()`, or it does nothing at all** —
and it fails silently, because a library that finds no variable simply stays off.

**LangSmith tracing is off unless switched on *and* keyed, and `export_client_env` actively clears
the switch when it is off.** Tracing uploads every question and every retrieved chunk to LangChain's
servers — for an HR copilot that is employee questions and internal policy leaving the deployment.
It must be a decision, not something a stray key in the environment turns on. `/health` reports
`tracing` so a running deployment shows it rather than hiding it in a file.

**Traces are redacted by default, and that is what makes tracing safe to enable here.**
`langsmith_hide_inputs` / `langsmith_hide_outputs` default to `True`, exported as
`LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS`. LangSmith still receives the run tree, timings,
routing and grading decisions, token counts and errors — everything needed to debug the agent — but
not the question text or the retrieved policy. Verified end to end against the live service: a probe
run carrying a sentinel string arrived with `inputs={}` and `outputs=None`. `/health` reports
`tracing_redacted` so "we turned it off to debug" cannot quietly become production.

**These must be the literal string `"true"`.** `langsmith.Client.__init__` does
`ls_utils.get_env_var("HIDE_INPUTS") == "true"`, an exact comparison — `"True"` or `"1"` reads as
false and silently uploads every question. `get_env_var` is also `lru_cache`d and searches
`LANGSMITH_*` then `LANGCHAIN_*`, so the variables must be in place before the first LangSmith
client is constructed. They are, because `export_client_env` runs when `app.core.config` is
imported, which precedes every client. Changing them at runtime will not take effect.

**Secrets are `SecretStr`, and that is not decoration.** They render as `**********` in reprs,
tracebacks and log lines, so `log.info("%s", settings)` or an unhandled exception carrying the model
cannot publish a key. Read the real value through `settings.require(field)`, never by attribute.

**Never raise from a `model_validator` on this class.** A model-level validator that raises produces
a pydantic `ValidationError` whose `input_value` is the **whole input mapping** — every API key,
rendered into the error text that then reaches logs and tracebacks. Observed exactly once, in a
dimension-mismatch check: the message began ``input_value={'GROQ_API': 'gsk_...``. Field-level
errors are safe (they echo only that field), but anything cross-field belongs in an ordinary method
like `validate_embedding()`, called from `validate_required()`.

**The admin key defaults to empty, and that is the design.** `ADMIN_API_KEY` gates the admin-only
endpoints. A reference implementation used `"change-me-in-production"` as the default; that is worse
than nothing, because forgetting to set it leaves the admin API open behind a string published in
the source. Empty means `admin_enabled` is False and those routes must **refuse every request** —
fail closed, so the failure mode is "admin is off", never "anyone is admin". `validate_required()`
additionally refuses to start without one when `APP_ENV=production`, and `/health` reports
`admin_enabled` so a silently-off admin surface is visible from outside.

Compare with `check_admin_key()`, never `==`: a plain comparison on a secret leaks its length and
prefix through timing. It also returns False when no key is configured, so an unset key cannot be
matched by an empty header.

**`APP_ENV` is load-bearing, not a label.** `production` withholds `/docs`, `/redoc` and
`/openapi.json` — they enumerate every route, admin ones included — and makes the admin key
mandatory. `/health` stays open at every env, because the load balancer depends on it.

**`kb_dir`, `upload_dir` and `audit_db_path` are settings, not fixed properties.** The target
architecture runs the app and its SQLite volume as separate Docker services, which only works if
those paths can be pointed elsewhere at deploy time.

## Providers

**Groq and OpenAI are both supported, chosen independently** via `LLM_PROVIDER` (`groq` default) and
`EMBEDDING_PROVIDER` (`huggingface` default). `get_llm` and `get_embeddings` in `app/rag/clients.py`
are the only places that branch; no node, prompt or graph edge knows which provider is active,
because both chat models satisfy the same contract the agent needs —
`.with_structured_output(Model)` bound via function calling. Verified live on both.

**`OPENAI_API_KEY` is exported only when a provider actually uses it.** It is the one setting whose
export name equals its input alias, so exporting it feeds straight back into the next `Settings`
built in that process. That also means a test which exports it contaminates every later one, which
is why `tests/conftest.py` clears it per test at *setup* — `monkeypatch.delenv(..., raising=False)`
records nothing for an absent variable, so it has nothing to undo for one the test then creates.

**Credential requirements follow the providers.** `required_secret_fields()` asks for `GROQ_API`
only when the LLM provider is Groq, and `OPENAI_API_KEY` only when something uses OpenAI, so an
all-OpenAI deployment is not reported broken for lacking a Groq key.

**Each embedding provider needs its own Pinecone index.** `all-MiniLM-L6-v2` emits 384-d vectors and
`text-embedding-3-small` 1536-d, and Pinecone cannot resize an index — so the two providers cannot
share one, and switching means a new `PINECONE_INDEX` plus a fresh ingest. `Settings` derives
`embedding_dim` from the active model when `EMBEDDING_DIM` is unset, and `validate_embedding()`
rejects a value that contradicts the model rather than letting it fail as an opaque upsert error.
`KNOWN_EMBEDDING_DIMS` is the table; an unlisted model must state its dimension explicitly.

**`sentence-transformers` cannot be deployed everywhere.** It pulls torch, ~2.5 GB installed, which
is far over a Vercel function's limit. `EMBEDDING_PROVIDER=openai` is what makes a serverless deploy
possible at all; on Docker/DigitalOcean either provider works and local embeddings are cheaper and
more private.

**Write `.gitignore` as UTF-8 and verify the bytes.** Editors on this machine have twice saved it
as UTF-16, which git cannot parse — it silently ignores nothing, and `.env` shows up as untracked
rather than ignored. Check with `git check-ignore -v .env`; if it prints nothing the file is broken
again. `head -c 16 .gitignore | od -c` should show ASCII, not a `fffe` BOM.

**`core.autocrlf` is `true` here and git does not always detect a binary.** Adding the reference
PDF produced `LF will be replaced by CRLF`, which would have corrupted it. `.gitattributes` now
marks `*.pdf`, `*.png`, `*.jpg`, `*.ico` as binary. Verify a binary you add with
`git cat-file -p :path | md5sum` against `md5sum < path`.

**Do not uninstall `openai`, `langchain-openai` or `langchain`.** No module here imports them and
they are deliberately absent from `req.txt`, but `langchain-pinecone` requires `langchain-openai`
(which requires `openai`) and `langchain-tavily` requires `langchain`. Removing them breaks
`pip check`.

## Architecture notes

**Import is side-effect free; the work lives in `main()`.** Building documents or embeddings at
module scope would make every import perform an HTTP fetch and load a ~90 MB model. The only
import-time side effect in the package is `export_client_env`, above.

**`get_embeddings` is `lru_cache`d and must stay that way.** Constructing it loads the model;
without the cache every retrieval call that omits an explicit `embedding` reloads it, which is
ruinous once a request handler retrieves per turn.

**LLM and web search are factories, not module-level singletons.** Constructing a client at import
time — and especially calling `.invoke()` there — would make every import cost an API call.

**Embedding dimension is coupled to the Pinecone index.** `all-MiniLM-L6-v2` emits 384-d normalised
vectors, and `ensure_index` creates a 384-d cosine serverless index to match. Changing
`EMBEDDING_MODEL` requires updating `EMBEDDING_DIM` *and* pointing `PINECONE_INDEX` at a new name —
Pinecone cannot change an existing index's dimension.

**Two corpora, one index, separated by namespace.** Internal policy lives in `hr-docs`, the scraped
public article in `public-web`. Both are explicit strings: Pinecone's default namespace (`""`)
behaves inconsistently between `list` and `delete`. Writer and reader read the same setting — a
namespace mismatch is **silent**, because the upsert succeeds and every later retrieval returns
`[]`. The employee-facing agent only ever reads the private namespace.

> These namespaces changed from `private-kb` and `""`, and both old ones have been deleted from the
> live index — `hr-docs` (31 vectors) is the only namespace that exists. A checkout with stale data
> elsewhere needs `python scripts/ingest_private_kb.py --drop-namespace private-kb`.

**The KB must not contradict itself.** Documents overlap on purpose — notice periods, PTO payout on
termination, expense payment timing, core hours all appear in more than one file — and retrieval
routinely returns chunks from two documents for one question. Two documents disagreeing produces a
confidently wrong answer that grading cannot catch, because each chunk looks fine alone. When
editing a policy, grep the KB for the same fact before changing it.

**Chunk ids include `origin`.** The basis is `origin` + `source` + `start_index`, keyed on position
rather than content so an edited chunk updates in place. `origin` is in there because `source` is
only a file name: once HR staff can upload, `benefits.pdf` from two departments would collide on
one id and the second ingest would silently overwrite the first.

**Ingest reconciles, it does not just upsert.** `prune_orphans` lists the ids in the namespace after
each ingest and deletes any that no longer correspond to a chunk on disk. Without it a deleted or
renamed policy file leaves vectors behind that keep answering questions — an HR copilot serving
withdrawn policy. `ingest_upload` passes `prune=False`, because pruning on a single-file upload
would delete the entire rest of the corpus.

**Pinecone serverless is eventually consistent.** `wait_for_vectors` polls `describe_index_stats`
after ingest because a retrieval fired immediately after an upsert returns nothing. Do not drop
this when refactoring ingest.

**`ensure_index` refuses a dimension of 0, and that guard is not redundant.** `embedding_dim` is 0
when `EMBEDDING_DIM` is unset and the active model is not in `KNOWN_EMBEDDING_DIMS`.
`validate_embedding()` reports it clearly, but only entry points call `validate_required()` — an
`/ingest` request reaches `ensure_index` directly. A 0-d index cannot be repaired, because Pinecone
cannot resize one, so this refuses at the last point that can still name the cause rather than
letting Pinecone reject it later against an index that then has to be recreated under a new name.

**There is exactly one gate, and it lives in Pinecone.** `get_kb_retriever()` applies it through
`similarity_score_threshold`; `retrieve_with_scores()` reports the ungated numbers it is tuned
from. There used to also be a `retrieve_relevant()` that filtered `retrieve_with_scores` by
`relevance_threshold` in Python — weaker in two ways (it re-ranked only the `k` rows that came back
rather than gating the search, and silently lost the `department` / `doc_type` filters its sibling
accepts) and with no callers. Deleted; a comment stands where it was. Do not reintroduce it.

**Pinecone SDK shape-tolerance.** `_index_names`, `_attr` and `_index_ready` exist because the v7
client returns index listings and status as either objects or dicts depending on call path. Keep
that defensiveness if you touch them.

**`score_threshold` is not on the cosine scale.** LangChain's `similarity_score_threshold` search
type maps raw cosine `c` to `(c + 1) / 2`. Passing a raw cosine straight through filters nothing:
`0.15` would be interpreted as raw cosine `-0.70`. `to_relevance_scale` does the conversion so
callers can think in raw cosine, which is what `retrieve_with_scores` reports.

**`get_retriever(score_threshold=...)` uses `-1.0` as its sentinel, not `None`.** `None` is a
meaningful value there — it disables the gate — so it cannot also mean "use the default".

**The 0.15 threshold is an out-of-domain gate, not a relevance grader.** Measured over a
12-question eval: out-of-domain questions peaked at 0.099, the lowest correct-source chunk scored
0.171, so 0.15 sits in the gap. It cannot separate right-document from wrong-document — a
wrong-source chunk scored as high as 0.521 — so grading actual relevance remains the LLM's job.

**Re-check the gate whenever the KB grows.** A fixed threshold degrades as a corpus expands: more
documents means more chances an unrelated question finds something above the gate. Re-measured when
the KB went from 6 documents / 15 chunks to 11 / 31 — out-of-domain still peaked at 0.078 and all
five new areas retrieved top-1 from the correct document, so 0.15 still holds.
`python scripts/check_retrieval.py` is that check.

**The similarity gate and the LLM grader are complementary, not redundant.** The gate rejects
out-of-domain *questions*; the grader rejects *irrelevant evidence* for in-domain questions.
Neither subsumes the other: "quokka grooming sabbatical" clears the gate (it looks lexically like
leave policy) and is caught by the grader, while "capital of France" never reaches the grader at
all. Removing either one opens a hole.

**Empty retrieval skips the grader deliberately.** `grade_kb_evidence` returns `weak` without an
LLM call when there are no chunks — grading an empty context spends a call to be told what the gate
already said. This trips people writing tests: scripting a `weak` grade for an empty retriever
never consumes it. `test_an_empty_retrieval_grades_weak_without_spending_a_call` pins the behaviour.

**Knowledge source for the public corpus is a single hardcoded URL.** `HR_POLICY_URL` points at one
HR Acuity article, scoped to `.entry-content` (~14.8k chars; the unscoped page is ~18.5k of mostly
nav and footer). `class_matcher` is a callable rather than a string because bs4 matches `class_`
against the entire class attribute, not one token at a time.

**Document metadata is what retrieval filters on.** Every chunk carries `source`, `title`, `doc_id`,
`doc_type`, `department`, `origin`, `ingested_on`. A subdirectory under `data/private_kb/` becomes
the department; files in the root are `general`. `build_filter` composes Pinecone filter
expressions, and filters compose *with* the similarity gate rather than replacing it.

**`format_documents` output is calibrated against the grader prompt.** It renders
`[n] source=<file>` and nothing more. Adding department or title there changes what the grader sees
on every question, which is not a free change — re-run the labelled set if you touch it.

## Agent notes

**Every graph node degrades; none may raise.** A node that throws aborts the whole run, so one
flaky Pinecone or Tavily call would cost an answer the other source could have given. `guard()` in
`build_nodes` wraps each network call: retrieval failure becomes zero chunks (which routes to the
web fallback), search failure becomes empty evidence, a rewrite failure keeps the original query
while still incrementing `retry_count` so a broken LLM cannot spin the loop, and a generation
failure returns `GENERATION_FAILED` with `source_used="error"` rather than an empty answer.

**"Inside the guard" is the whole rule, and it is easy to half-observe.** `_generate` used to call
`.content` inside `guard()` and `.strip()` outside it. `.content` is a plain string on Groq and
OpenAI, but the message interface allows a *list of content blocks*, and `.strip()` on a list
raises `AttributeError` — which escaped the node and aborted the run, so a provider shape change
would have cost every answer rather than degrading one. Both generation and the rewrite now flatten
through `message_text()` in `app/rag/clients.py`, inside the guard.
`test_a_reply_of_content_blocks_does_not_abort_the_run` fails against the old code with exactly
that AttributeError; keep it. When adding a node, check where the *last* attribute access happens,
not just where the network call does.

**Provider response shapes are adapted in `app/rag/clients.py` and nowhere else.** `message_text`
for chat replies, `web_results_to_text` and `web_result_urls` for Tavily. A node that reaches into
a provider's response shape directly is a bug waiting for the next SDK release;
`tests/test_clients.py` pins all three offline.

**`message_text` also strips the model's dangling citation markers.**
`openai/gpt-oss-*` emits OpenAI's file-citation tokens — `【2†L1-L4】`, `【4:2†source】` — because it was
trained to cite retrieved documents that way. Nothing in this system numbers documents like that, so
they reference nothing at all. Observed live on a web-fallback answer about paid holidays, which
came back with four of them inside a markdown table. They matter more here than they would
elsewhere: in an HR answer they read as authoritative citations to sources that do not exist,
sitting next to the real ones this product does supply. `CITATION_ARTIFACT` is bounded rather than
greedy so an unmatched `【` in legitimate text costs that fragment and not the rest of the reply, and
it swallows the preceding whitespace so removing a token mid-sentence leaves no double space.
Stripping happens in `message_text` — the one funnel every reply passes through — so the API
response, the CLI and the audit record all hold the same clean text. Do not move it to the
frontend: the audit log would then keep the noise for ever.

**Graph nodes are closures, built once per graph.** `build_nodes` takes the LLM, retriever and
search tool and closes over them, so a run constructs each once and tests can inject fakes — that
is how the rewrite loop and the insufficient-evidence path are tested deterministically, without
waiting for a real question that happens to fail twice. `build_graph` and `Copilot` forward the
same injection points. Do not replace a parameter with a module-level lookup.

**Structured output fails intermittently and must never crash the graph.** Groq returns
`400 tool_use_failed` — "Tool choice is required, but model did not call a tool" — when the model
answers in plain text instead of invoking the bound tool. Observed live on the router, whose call is
on the path of *every* question. The error body carries `error.failed_generation`, which holds what
the model actually said and is usually the correct value (an observed failure carried exactly
`"direct"`). `route_question` and `grade_evidence` recover from it, then fall back to safe defaults:
the router to `kb` (seek evidence rather than answer from memory) and the grader to `weak` (distrust
rather than pass ungraded evidence). **Graph nodes must call those helpers, not `router.invoke` /
`grader.invoke` directly.**

**Never use `method="json_mode"` for the decision models.** It is unconstrained JSON and this model
fills the enum field with prose — a route query came back as
`{"route": "Employees accrue 1.5 days of PTO per month..."}`, which is exactly the failure
structured output exists to prevent. The default (function calling) enforces the enum and is also
the fastest of the three working methods. `json_schema` works too.

**The router prompt is a two-sided calibration; changing one side breaks the other.** It must send
external factual questions to `kb` (otherwise the model answers from memory with no evidence) while
keeping greetings *and questions about the assistant itself* on `direct`. An earlier version pushed
everything informational to `kb`; `"who are you?"` then retrieved nothing, fell through to web
search, and answered from an unrelated web page. Re-test both classes after any edit to it.

**`question` and `current_query` are deliberately separate in `AgentState`.** A rewrite replaces
`current_query`; `question` keeps the employee's original wording for the final answer, citations
and logs. Collapsing them loses what was actually asked.

**`trace` accumulates through a reducer; `citations` deliberately does not.** `trace` is
`Annotated[list[str], add]`, and the reducer is load-bearing: a plain TypedDict key is *replaced* by
whatever a node returns, so a bare `list[str]` would leave only the last node's entry and the
rewrite loop — the one run whose path is worth seeing — would erase its own history. It is a
required key seeded empty by `initial_state`, because a reducer has nothing to fold into on a key
that may be absent. `note()` in `build_nodes` logs a step *and* returns it, so the console line and
the state entry cannot drift apart, and `guard()` appends there too when a call degrades: an outage
and a genuine absence of evidence produce the same answer, and only the trace distinguishes them.
`citations` takes no reducer on purpose — it is derived from `kb_docs`, and a rewrite loops back
through `retrieve_kb` and *replaces* those chunks, so accumulating would cite documents the answer
no longer rests on. `test_the_trace_accumulates_instead_of_being_overwritten` and
`test_citations_are_replaced_by_a_rewrite_not_accumulated` pin the two halves; deleting the reducer
fails three tests rather than none.

**`temperature=0` is not determinism on Groq.** Repeated grading of a genuinely borderline case
("the KB covers employees, the question asks about contractors") returned `weak` 8/8 in isolation
but flipped to `good` once inside a longer suite run. Treat single-run grader accuracy as
approximate, and do not chase a one-off flip as a prompt bug.

## Providers

**Groq's free tier caps tokens per DAY, and exhaustion is invisible without looking.** Observed:
`429 ... tokens per day (TPD): Limit 200000, Used 199771`. Because every safe default is "weak", an
exhausted quota looks exactly like a knowledge gap — the router returns `kb`, every grade returns
`weak`, and every run ends `insufficient_evidence`. `is_rate_limit()` distinguishes it and those
paths log at ERROR saying the results are degraded rather than a judgement. If a whole test run
suddenly grades everything weak, check the quota before touching a prompt. Limits are per-model, so
`GROQ_MODEL=openai/gpt-oss-120b` is a working escape hatch.

**Groq model ids move.** `GROQ_MODEL` defaults to `openai/gpt-oss-20b` and is overridable by env
var. A retired id fails at call time, not at construction, so the traceback points at the invoke.
`groq.Groq().models.list()` prints what the key can actually reach — as of the last check that is
14 models, including `openai/gpt-oss-20b` and `-120b`, `groq/compound`, and two `qwen3` variants.
There is no llama model on this account.

**Do not diagnose Groq with `urllib`.** A raw `urllib` request to the Groq API returns HTTP 403 with
Cloudflare error 1010 — the default `Python-urllib` User-Agent is blocked. That looks exactly like a
revoked key. Use the `groq` SDK or `ChatGroq`, which set proper headers.

**`TavilySearch.invoke()` returns a dict, not Documents.** Keys are `query`, `answer`, `results`,
`images`, `follow_up_questions`, `response_time`, `request_id`. `web_results_to_text` and
`web_result_urls` adapt it; anything mixing web results with retriever output has to go through
them.

## Output and tests

**Library code logs; only an entry point prints.** `config.get_logger()` per module, with a
NullHandler on the package root so importing this library never writes anywhere. Entry points call
`configure_logging()`, which sends records to **stderr** — stdout stays clean so
`python scripts/demo.py > out.txt` captures answers without the node trace. Do not add `print()` to
a non-CLI function; return a value and let the caller print. `IngestReport` and `render_answer`
exist for exactly that reason.

**Windows consoles break on LLM output.** cp1252 raises `UnicodeEncodeError` on the non-breaking
hyphens and smart quotes models emit — the API call succeeds and `print()` is what crashes.
`configure_stdout()` forces UTF-8 and is called at the top of every entry point.

**The test suite runs entirely offline.** No Groq, Pinecone, Tavily or model download; it needs no
`.env`. That is only possible because every layer takes its clients as arguments, and
`tests/conftest.py` supplies `FakeLLM`, `FakeRetriever` and `FakeWebSearch`. If a change makes a
test need the network, the change broke the injection, not the test.

**Assert the exception type, not `Exception`.** `pytest.raises(Exception)` passes when the code
fails for a completely different reason than the one under test — an `AttributeError` from a
renamed field satisfies it just as well as the `ValidationError` the test meant. ruff's `B017`
enforces this.

**The `clean_process_env` autouse fixture is part of that guarantee, not tidiness.** Without it the
suite is offline only by accident: `app.core.config` exports tracing into `os.environ` at import, so
on a machine with LangSmith enabled every graph test uploads its run — observed, complete with
`403 Forbidden` noise in the output. It also clears `OPENAI_API_KEY`, whose export name equals its
input alias, so a test that sets one cannot configure every `Settings` built after it. Do not remove
it. (It was called `no_tracing` in an earlier revision; this file said so long after the code
stopped.)

**A test asserting a default must isolate from *both* config sources.** `Settings()` reads
`os.environ` first and `.env` second, and `export_client_env` has already written to `os.environ` by
the time any test runs. `_env_file=None` alone is not enough — it silences the file and leaves the
environment, so the test still reads local configuration and passes or fails by machine. Use
`monkeypatch.delenv` as well; `test_tracing_is_off_by_default` is the worked example.

**That rule was broken in `tests/test_api.py` and nothing noticed for three commits.** Its `client()`
helper built `Settings(**base)` without `_env_file=None`, so the whole API suite read the
developer's real `.env`. Three tests asserting an *unconfigured* admin surface passed only because
`ADMIN_API_KEY` happened to be empty there; filling it in — an ordinary thing to do to use the admin
UI — broke them, and the suite's claim to need no `.env` was simply false. The helper now passes
`_env_file=None`, and `clean_process_env` clears `ADMIN_API_KEY` too, because
`ADMIN_API_KEY=k python run.py` leaves it exported for the rest of that shell. Verify a change here
the way it was verified: run the suite once normally, once with `ADMIN_API_KEY=leaked-from-shell`
set. Both must pass.

`pyproject.toml` sets `pythonpath = [".", "tests"]`, which is how `tests/test_graph.py` imports
`conftest` directly. Do **not** add `tests/__init__.py` — it turns the directory into a package and
that import stops resolving.

`test_the_compiled_graph_matches_the_documented_design` asserts the compiled edge set exactly. It
is what keeps the README's mermaid diagram and `docs/agent-graph.md` honest, so update it
deliberately when the graph changes rather than loosening it.

## API notes

**`create_app` takes its services as arguments and puts them on `app.state`; the routes read them
back through `app/api/deps.py`.** That is the same injection the rest of the codebase has, carried
up to the HTTP layer: `create_app(settings, copilot=..., audit=...)` is how the API tests run
offline against a graph of fakes and a `tmp_path` database. **Do not wire production through
`dependency_overrides`** — it is a test hook, and once the app uses it there is no way to tell which
overrides are the application and which are the test.

**Nothing `create_app` builds touches the network or the disk.** `Copilot` compiles its graph on
first use and `AuditStore` creates its schema on first use, because `app.main` builds an application
at import — a constructor that reached Pinecone or wrote a database file would make importing the
module do both. `test_building_the_app_touches_neither_network_nor_disk` pins it.

**No handler puts an exception's text in `detail`.** A provider error carries URLs, request ids and
sometimes the offending payload, and this process holds API keys; `detail=str(exc)` is how one
reaches a browser. Failures are logged with their traceback and answered with a fixed sentence.
`test_a_copilot_failure_does_not_leak_the_provider_error` posts a question to a copilot that raises
`"401 from https://api.groq.com key=gsk_live_SECRET"` and asserts neither the key nor the host
appears in the response.

**`/chat` and `/audit` are `def`, not `async def`.** Starlette runs a sync handler in a threadpool; a
question costs seconds of LLM round trips, and an `async def` handler doing that blocks every other
request in the process. `/upload` is `async def` because it streams a request body, and hands the
blocking embed-and-upsert to `run_in_threadpool` explicitly. Adding a route means choosing between
these deliberately.

**The admin gate refuses twice, and the order is the point.** `require_admin` returns 503 when no key
is configured — *before* looking at the header — so an unset key can never be matched by an absent
or empty one. A reference implementation defaulted to `"change-me-in-production"` and compared with
`!=`, which means an unset key plus an absent header compares equal and every caller is an admin.
The key itself goes through `settings.check_admin_key`, never `==`.

**`/audit` is admin-only because every row is an employee's question and the answer they got.** That
is HR-sensitive by definition, and is the reason the log is an endpoint behind a key rather than a
file anyone with shell access can read. `audit_db_path`, `upload_dir` and `kb_dir` are settings so a
container can mount them; `*.db` is gitignored.

**`AuditStore.record` never raises.** A full disk, a locked database or a read-only mount degrades
the audit trail, and that is much cheaper than turning a working answer into a 500. It returns
`None`, `audit_id` comes back null, and the reason is logged at ERROR so an unwritten log is visible
in the logs rather than only in its own absence. It opens a connection per call: FastAPI runs `def`
endpoints in a threadpool, so one long-lived `sqlite3` connection would need `check_same_thread`
and its own locking.

**`/api/audit/stats` is declared before `/api/audit/{entry_id}`.** Declared after, the path parameter
claims `"stats"` and answers with a validation error.
`test_audit_stats_are_not_shadowed_by_the_id_route` pins the ordering.

**An upload's filename is attacker-controlled text, not a path.** `_safe_filename` strips both
separators, not just the platform's, because a Windows client posting to a Linux container sends
backslashes that `Path.name` there would keep. The size cap is enforced *while streaming* — `await
file.read()` with no argument materialises the whole body first, which is a way to exhaust the
process before any limit applies — and a refused upload leaves no partial file behind.

**The upload ingest never prunes,** and `ingest_upload` is where that is decided. Pruning reconciles
a namespace against a directory on disk; for a single-file upload it would delete the entire rest of
the corpus.

## Console notes

**The console is served from the same application, so there is no CORS middleware — deliberately.**
`GET /` renders `templates/index.html` and the browser then calls `/api/*` same-origin. Adding a
CORS middleware would widen the surface for a request nobody is making. If the target
architecture's Nginx container ever serves `static/` from a different origin, that is the moment to
add one, scoped to that origin.

**`static/` is mounted only when the directory exists.** `StaticFiles` raises at *mount* time on a
missing directory, which would turn "the assets were not copied into the image" into "the container
does not start" — and take `/health` down with it, so nothing could report why. Missing assets log a
warning and the API keeps serving.

**The page's `data-admin` attribute hides controls; it is not the gate.** `require_admin` is, on
every request. `/health` is authoritative and the JS re-reads it every 30s, so a deployment that
gains or loses its admin key updates the UI without a reload.

**Nothing untrusted is ever assigned to `innerHTML`.** Answer text comes from an LLM, citation
titles from uploaded documents, audit rows from the database — none of it is content this
application controls, and the admin reading the audit log is exactly the session worth stealing.
Every dynamic string reaches the page through `textContent` or `createTextNode`, and the light
markdown renderer *builds* `<strong>` / `<code>` elements rather than parsing markup, so there is no
escaping step that can be forgotten. Keep it that way: the moment one `innerHTML` appears, a model
reply containing `<img src=x onerror=...>` runs.

**The admin key is held in `sessionStorage`, not `localStorage`.** It is a credential; session
scope means a shared machine does not keep it after the tab closes. It goes in the `X-Admin-Key`
header and never in a query string — a URL reaches the history, the proxy log and the `Referer`.

**`min-height: 0` on every flex and grid child down to `.thread` is load-bearing.** A grid item
defaults to `min-height: auto`, which is its *min-content* height, so without it on `.stage` the
shell's row grew to fit the entire unscrolled conversation, the body gained 169px of scrollable
overflow, and the first `input.focus()` after an answer scrolled the topbar out of view. The symptom
looks like a scroll bug and the cause is a sizing default; do not "fix" it by re-adding `overflow`
somewhere.

**Entrance animations are gated behind a `.stagger-in` class, not put on the element.** The trace
and evidence lists are rendered the moment an answer lands, into whichever inspector panels are
*not* the open tab — and a CSS animation never runs on a `display:none` element. With
`animation-fill-mode: both` the items were then stuck holding the `from` keyframe, so opening the
tab showed an empty panel with the content sitting there at `opacity: 0`. The JS adds the class when
the container is actually on screen.

**The answer renderer handles paragraphs, lists, `**bold**`, `*italic*`, `` `code` `` and pipe
tables — and every one of them builds elements rather than parsing markup.** Tables and italics
were both added after watching real answers: a live web answer sent a correct five-line markdown
table, which fell through to the paragraph branch and came out as one run-on line of pipes and
dashes, and the model routinely cites its source as `*leave-and-time-off.md*`. Bold stays first in
the `INLINE` alternation so `**x**` wins over `*x*`. A table gets its own `overflow-x` box, because
a wide one must not make the whole conversation scroll sideways.

**The graph highlight is derived from the trace, not from a second source of truth.** `TRACE_NODE`
in `app.js` maps a trace entry's tag (`"KB Retriever"`, `"Generate/Web"`) to a node id, and
consecutive visited nodes are exactly the edges traversed — including `rewrite -> retrieve`, the
loop. Renaming a tag in `note()` calls in `app/agent/nodes.py` silently stops lighting that node, so
change both together.

## The build plan

[step.md](step.md) is the user's 16-step plan for this project; steps 1–9 and 16 are done. It
deviates from the code in three ways, all deliberate and all recorded in the README's Roadmap. Do
not "fix" the code to match the plan without asking.

- **Providers.** The plan names OpenAI embeddings and model; this runs local `all-MiniLM-L6-v2` and
  Groq. Both OpenAI paths are supported and verified live — it is a default, not a limitation.
- **Module paths.** The plan puts state and workflow at `app/rag/state.py` and
  `app/rag/workflow.py`; they are at `app/agent/schemas.py` and `app/agent/graph.py`, which keeps
  the agent in one package. `app/services/ingestion.py` does exist as the plan names it, but as a
  thin facade over `app/rag/loaders.py` and `app/rag/ingest.py` — parsing and upserting stay one
  layer down, so there is still exactly one place a PDF is read.
- **One corpus on disk.** The plan names `data/sample_kb/`; the documents are in
  `data/private_kb/` and there is no second directory. A `data/sample_kb/hr_operations_runbook.md`
  existed briefly and was deleted: it re-covered onboarding, offboarding and payroll in vaguer
  terms and different vocabulary than the real KB. Adding a parallel corpus here is not a neutral
  act — see **The KB must not contradict itself**.

## Leftovers

`create_project.py` at the repo root scaffolds the directory layout and is idempotent; re-running it
creates nothing that exists. `Rag/__pycache__/` may still exist from the old package — it is
gitignored bytecode with no source and cannot be imported, but it is safe to delete.
