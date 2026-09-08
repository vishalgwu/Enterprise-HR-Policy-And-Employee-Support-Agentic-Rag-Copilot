# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An Agentic RAG copilot for internal HR policy and employee support. The reference brief and target
architecture are checked in: `docs/Enterprise_HR_Agentic_RAG_Problem_Statement_DigitalOcean.pdf`
and `docs/architecture.png`. Read them before designing anything new — they name the endpoints, the
data layer, the roles and the deployment target.

The retrieval stack and the 10-node LangGraph agent are complete. The FastAPI surface is `/health`
only; SQLite persistence, the frontend, and Docker/DigitalOcean packaging are not built.

## Environment and commands

Windows-first. The virtualenv lives in `hr/` (Python 3.13, created from Anaconda) and is
self-ignored via `hr/.gitignore`.

```bash
hr\Scripts\activate                    # PowerShell / cmd
pip install -r req.txt                 # req.txt is the source of truth; requirements.txt is "-r req.txt"

pytest                                 # 91 tests, offline, no credentials needed
python scripts/ingest_private_kb.py    # data/private_kb/** -> "hr-docs"
python scripts/ingest_public_web.py    # scraped article     -> "public-web"
python scripts/demo.py                 # four demos, one per graph path
python scripts/check_retrieval.py      # gated vs ungated retrieval scores
python scripts/render_graph.py         # regenerate docs/agent-graph.md
python run.py                          # uvicorn on :8000
```

No linter or formatter is configured. `pyproject.toml` carries a ruff section that nothing runs yet.

## Module layering

Lowest first; each layer imports only from those above it, and **nothing imports from `scripts/`**.

| Module | Responsibility |
|---|---|
| `app/core/config.py` | `Settings`, logging, console setup. No heavy imports. |
| `app/rag/clients.py` | embeddings (cached), Groq chat model, Tavily search |
| `app/rag/loaders.py` | markdown/text/PDF/DOCX/web loading, chunking, chunk ids, evidence rendering |
| `app/rag/vectorstore.py` | Pinecone, parameterised by **index and namespace** |
| `app/rag/ingest.py` | load → chunk → embed → upsert → reconcile, for any corpus |
| `app/rag/retrieval.py` | the private KB and public corpus, namespaces already chosen |
| `app/agent/schemas.py` | decision models and `AgentState` |
| `app/agent/prompts.py` | every prompt |
| `app/agent/decisions.py` | router and grader, with recovery |
| `app/agent/nodes.py` | node bodies, as closures |
| `app/agent/graph.py` | wiring and compilation |
| `app/services/copilot.py` | `Copilot` facade and `AnswerResult` — the API contract |
| `app/main.py`, `app/api/` | FastAPI |
| `scripts/*` | thin CLI callers |

An earlier version had `graph.py` importing `get_kb_retriever` from `private_kb.py` — an entry
point. That is the specific mistake this table exists to prevent: **the agent must not depend on a
corpus script.** Retrieval accessors belong in `app/rag/retrieval.py`.

Every `vectorstore` operation takes a namespace *and* an index. Do not add a namespace-specific
copy of an ingest or retrieval helper — pass the namespace instead.

Scripts run both ways (`python scripts/demo.py` and `python -m scripts.demo`) because of the
`if __package__ in (None, "")` bootstrap at the top of each one, which puts the project root on
`sys.path`. Library modules use plain `from app.x import y` and need no such guard.

## Secrets

`.env` at the repo root holds `GROQ_API`, `TAVILY_API`, `PINECONE_API`, `PINECONE_INDEX`,
`PINECONE_CLOUD`, `PINECONE_REGION`. It is gitignored and has never been committed;
[.env.example](.env.example) documents the shape. Only the three keys are required.

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

**Nothing here uses OpenAI.** `.env` carries a lowercase `openai_api` key that no module reads;
it is deliberately lowercase so no library picks it up as `OPENAI_API_KEY`. Do not "normalise" that
name — `langchain-openai` is installed (as a `langchain-pinecone` dependency) and would find it.

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

**The `no_tracing` autouse fixture is part of that guarantee, not tidiness.** Without it the suite
is offline only by accident: `app.core.config` exports tracing into `os.environ` at import, so on a
machine with LangSmith enabled every graph test uploads its run — observed, complete with
`403 Forbidden` noise in the output. Do not remove it.

**A test asserting a default must isolate from *both* config sources.** `Settings()` reads
`os.environ` first and `.env` second, and `export_client_env` has already written to `os.environ` by
the time any test runs. `_env_file=None` alone is not enough — it silences the file and leaves the
environment, so the test still reads local configuration and passes or fails by machine. Use
`monkeypatch.delenv` as well; `test_tracing_is_off_by_default` is the worked example.

`pyproject.toml` sets `pythonpath = [".", "tests"]`, which is how `tests/test_graph.py` imports
`conftest` directly. Do **not** add `tests/__init__.py` — it turns the directory into a package and
that import stops resolving.

`test_the_compiled_graph_matches_the_documented_design` asserts the compiled edge set exactly. It
is what keeps the README's mermaid diagram and `docs/agent-graph.md` honest, so update it
deliberately when the graph changes rather than loosening it.

## The build plan

[step.md](step.md) is the user's 16-step plan for this project; steps 1–9 and 16 are done. It
deviates from the code in two ways, both deliberate and both recorded in the README's Roadmap: it
names OpenAI embeddings and model (this uses local `all-MiniLM-L6-v2` and Groq), and it puts
ingestion, state and workflow at `app/services/ingestion.py`, `app/rag/state.py` and
`app/rag/workflow.py` (they are at `app/rag/ingest.py`, `app/agent/schemas.py`,
`app/agent/graph.py`). Do not "fix" the code to match those paths without asking — the split keeps
the agent in one package.

## Leftovers

`create_project.py` at the repo root scaffolds the directory layout and is idempotent; re-running it
creates nothing that exists. `Rag/__pycache__/` may still exist from the old package — it is
gitignored bytecode with no source and cannot be imported, but it is safe to delete.
