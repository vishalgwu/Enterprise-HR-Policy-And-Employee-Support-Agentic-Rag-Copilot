# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An Agentic RAG copilot for internal HR policy and employee support. The README describes the
target stack as LangGraph + FastAPI + Pinecone + OpenAI + Tavily with an HTML/CSS/JS frontend;
only the retrieval/ingestion layer (`Rag/Agentic_rag.py`) exists so far. The FastAPI app, the
LangGraph agent graph, and the frontend are not written yet.

## Environment and commands

Windows-first. The virtualenv lives in `hr/` (Python 3.13, created from Anaconda) and is
self-ignored via `hr/.gitignore`.

```bash
hr\Scripts\activate                  # PowerShell / cmd
pip install -r req.txt               # note: req.txt, not requirements.txt
python Rag/Agentic_rag.py            # public web ingest -> default namespace
python Rag/private_kb.py             # private KB ingest -> "private-kb" namespace + retrieval demo
```

## Module layering

Lowest first; each layer imports only from those above it.

| Module | Responsibility |
|---|---|
| `config.py` | settings, secrets, all tunable constants. No heavy imports. |
| `clients.py` | embeddings (cached), Groq chat model, Tavily search |
| `loaders.py` | web + markdown loading, chunking, chunk ids, evidence rendering |
| `vectorstore.py` | Pinecone index management and retrieval, **parameterised by namespace** |
| `schemas.py` | structured routing/grading decisions, `AgentState` |
| `Agentic_rag.py` / `private_kb.py` | corpus entry points; thin |

New code should import from the layer modules, not from the entry points. The entry points still
re-export their historical names (`get_llm`, `get_embeddings`, `_chunk_id`, `PINECONE_INDEX_NAME`,
…) purely for backward compatibility.

Every `vectorstore` operation takes a namespace, so the same code drives both corpora. Do not add
a namespace-specific copy of an ingest or retrieval helper — pass the namespace instead.

Both entry points run either way: `python Rag/private_kb.py` or
`from Rag.private_kb import get_kb_retriever`. Script mode works because of the
`if __package__ in (None, "")` bootstrap at the top of each entry point, which puts the project
root on `sys.path`. Library modules use plain `from Rag.x import y` and need no such guard.

**`config` must be imported before any client is constructed.** It exports `GROQ_API_KEY` /
`TAVILY_API_KEY` / `PINECONE_API_KEY` into the environment, which the clients read in their
constructors. This is why `clients.py` imports config above its LangChain imports.

**`get_embeddings` is `lru_cache`d and must stay that way.** Constructing it loads a ~90 MB model;
without the cache every retrieval call that omits an explicit `embedding` reloads it.

Running the module end-to-end downloads the sentence-transformers model on first use, fetches the
source article over the network, creates the Pinecone index if absent, and upserts every chunk.
Chunk ids are derived from `source` + `start_index`, so re-running overwrites rather than
duplicating (verified: two consecutive runs both leave 18 vectors).

There is no test suite, linter, or formatter configured.

## Secrets

`.env` at the repo root holds `GROQ_API`, `TAVILY_API`, `PINECONE_API`, `PINECONE_INDEX`,
`PINECONE_CLOUD`, `PINECONE_REGION`. It is gitignored and has never been committed;
[.env.example](.env.example) documents the shape.

**Write `.gitignore` as UTF-8 and verify the bytes.** Editors on this machine have twice saved it
as UTF-16, which git cannot parse — it silently ignores nothing, and `.env` shows up as untracked
rather than ignored. Check with `git check-ignore -v .env`; if it prints nothing the file is
broken again. `xxd .gitignore | head -1` should show ASCII, not `2300 2000` or a `fffe` BOM.

**Do not uninstall `openai`, `langchain-openai` or `langchain`.** No module here imports them and
they are deliberately absent from `req.txt`, but `langchain-pinecone` requires `langchain-openai`
(which requires `openai`) and `langchain-tavily` requires `langchain`. Removing them breaks
`pip check`.

## Architecture notes

**Env-var remapping happens before imports, on purpose.** `Rag/Agentic_rag.py` reads the
project's short names (`GROQ_API`, `TAVILY_API`, `PINECONE_API`) from `.env`, validates them, and
re-exports them into `os.environ` under the standard names the LangChain/Pinecone/Tavily clients
expect (`GROQ_API_KEY`, etc.), plus a `USER_AGENT` that `WebBaseLoader` requires. All
`langchain_*` / `pinecone` imports sit *below* that block. Do not hoist them to the top of the
file or move them into a shared module without preserving the ordering — the clients read those
variables at import time.

**Import is side-effect free; the work lives in `main()`.** Keep it that way when wiring this into
FastAPI or LangGraph — call `load_hr_policy_documents` / `split_hr_policy_documents` /
`get_embeddings` explicitly. Building documents or embeddings at module scope would make every
import perform an HTTP fetch and load the model.

**Embedding dimension is coupled to the Pinecone index.** `sentence-transformers/all-MiniLM-L6-v2`
emits 384-d normalized vectors, and `ensure_pinecone_index` creates a 384-d cosine serverless
index to match. Changing `EMBEDDING_MODEL` requires updating `EMBEDDING_DIM` and recreating the
index under a new `PINECONE_INDEX` name — Pinecone cannot change an existing index's dimension.

**Pinecone SDK shape-tolerance.** `_index_names` and `_index_ready` exist because the v7 client
returns index listings and status as either objects or dicts depending on call path. Keep that
defensiveness if you touch them.

**Knowledge source is a single hardcoded URL.** `HR_POLICY_URL` points at one public HR Acuity
article, scoped to the `.entry-content` block (~14.8k chars → 18 chunks; the unscoped page is
~18.5k chars of mostly nav/footer boilerplate). `_is_content_class` is a callable rather than a
string because bs4 matches `class_` against the entire class attribute, not one token at a time.
`req.txt` already pins `pypdf` and `python-docx`, but no PDF/DOCX ingestion path is implemented.

**Two corpora, one index, separated by namespace.** The public scraped article lives in the
default namespace (18 vectors); the private KB in `data/private_kb/*.md` lives in `private-kb`
(15 vectors), set by `PRIVATE_NAMESPACE` in [Rag/private_kb.py](Rag/private_kb.py). Writer and
reader must use the same constant — a namespace mismatch fails silently, since the upsert succeeds
and every later retrieval just returns `[]`.

**`score_threshold` is not on the cosine scale.** LangChain's `similarity_score_threshold` search
type maps raw cosine `c` to `(c + 1) / 2`. Passing a raw cosine straight through filters nothing:
`0.15` would be interpreted as raw cosine `-0.70`. `_to_relevance_scale` does the conversion so
callers can think in raw cosine, which is what `retrieve_with_scores` reports.

**The 0.15 threshold is an out-of-domain gate, not a relevance grader.** Measured over a
12-question eval of this KB: out-of-domain questions peaked at 0.099, the lowest correct-source
chunk scored 0.171, so 0.15 sits in the gap and both out-of-domain questions return zero docs with
all 10 in-domain top-1 results intact. It cannot separate right-document from wrong-document — a
wrong-source chunk scored as high as 0.521 — so grading actual relevance remains the LLM's job.

**Ingest reconciles, it does not just upsert.** `_prune_orphans` lists the ids in the namespace
after each ingest and deletes any that no longer correspond to a chunk on disk. Without it a
deleted or renamed policy file leaves vectors behind that keep answering questions — an HR copilot
serving withdrawn policy. Only the private namespace is reconciled; the public web corpus is not.

**Pinecone serverless is eventually consistent.** `_wait_for_vectors` polls
`describe_index_stats` after ingest because a retrieval fired immediately after an upsert returns
nothing. Do not drop this when refactoring ingest.

**Never use `method="json_mode"` for the decision models.** It is unconstrained JSON and this model
fills the enum field with prose — a route query came back as
`{"route": "Employees accrue 1.5 days of PTO per month..."}`, which is exactly the failure
structured output exists to prevent. The default (function calling) enforces the enum and is also
the fastest of the three working methods. `json_schema` works too.

**The similarity gate and the LLM grader are complementary, not redundant.** The 0.15 gate rejects
out-of-domain *questions*; the grader rejects *irrelevant evidence* for in-domain questions. Neither
subsumes the other: "quokka grooming sabbatical" clears the gate (it looks lexically like leave
policy) and is caught by the grader, while "capital of France" never reaches the grader at all.
Removing either one opens a hole.

**`question` and `current_query` are deliberately separate in `AgentState`.** A rewrite replaces
`current_query`; `question` keeps the employee's original wording for the final answer, citations,
and logs. Collapsing them loses what was actually asked.

**LLM and web search are factories, not module-level singletons.** `get_llm()` and
`get_web_search()` follow the same rule as the rest of the module: constructing a client at import
time — and especially calling `.invoke()` there — would make every import cost an API call. The
smoke calls live in `main()`.

**Groq model ids move.** `GROQ_MODEL` defaults to `openai/gpt-oss-20b` and is overridable by env
var. A retired id fails at call time, not at construction, so the traceback points at the invoke.
`groq.Groq().models.list()` prints what the key can actually reach — as of the last check that is
14 models, including `openai/gpt-oss-20b` and `-120b`, `groq/compound`, and two `qwen3` variants.
There is no llama model on this account.

**Do not diagnose Groq with `urllib`.** A raw `urllib` request to the Groq API returns HTTP 403
with Cloudflare error 1010 — the default `Python-urllib` User-Agent is blocked. That looks exactly
like a revoked key. Use the `groq` SDK or `ChatGroq`, which set proper headers.

**`TavilySearch.invoke()` returns a dict, not Documents.** Keys are `query`, `answer`, `results`,
`images`, `follow_up_questions`, `response_time`, `request_id`. Anything mixing web results with
retriever output has to adapt this shape first.

**`req.txt` still pins `openai` and `langchain-openai`,** which nothing imports. Generation and
grading run on Groq. Left in place rather than removed as an unrelated change.

**Windows consoles break on LLM output.** cp1252 raises `UnicodeEncodeError` on the non-breaking
hyphens and smart quotes models emit — the API call succeeds and `print()` is what crashes.
`configure_stdout()` forces UTF-8 and is called at the top of `main()`.
