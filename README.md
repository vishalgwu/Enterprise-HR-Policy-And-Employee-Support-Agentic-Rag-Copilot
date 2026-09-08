# Enterprise HR Policy and Employee Support — Agentic RAG Copilot

An internal HR assistant built as an **agentic** retrieval-augmented generation system: instead of
answering from a single retrieval pass, it scores what it retrieves, rejects out-of-domain matches,
and is designed to fall back to query rewriting and web search rather than hallucinate policy.

Built as a Forward Deployed Engineer (FDE) style project — taking a RAG workflow from notebook
experiment to a measured, reproducible ingestion and retrieval layer.

---

## Status

| Layer | State |
|---|---|
| Public web ingestion → Pinecone | Implemented, verified end to end |
| Private (internal) knowledge base → Pinecone | Implemented, verified end to end |
| Namespace isolation between the two corpora | Implemented |
| Retrieval + out-of-domain gating | Implemented, evaluated on 12 questions, re-verified after the KB grew |
| Private KB covering all nine areas the brief names | 11 documents, 31 chunks |
| Groq LLM client (`openai/gpt-oss-20b`, temperature 0) | Implemented, live call verified |
| Tavily web-search client | Implemented, live call verified |
| Structured routing + evidence grading | Implemented, 21/21 on a labelled set |
| LangGraph agent (10 nodes: route → grade → rewrite → fallback → answer) | Implemented, all paths verified |
| Multi-format ingestion (Markdown, TXT, PDF, DOCX) | Implemented |
| Metadata filtering (`department`, `doc_type`) | Implemented |
| Test suite — 101 tests, no network, no credentials | Implemented |
| LangSmith tracing, redacted by default | Implemented, verified against the live service |
| FastAPI service | `/health` only; `/chat`, `/upload`, `/ingest`, `/feedback`, `/admin`, `/logs` not yet built |
| SQLite audit layer (chat history, feedback, traces) | Not yet built |
| HTML/CSS/JS frontend | Not yet built |
| Docker + DigitalOcean deployment | Not yet built |

The agent graph is complete and runs end to end, and `app.services.copilot` is the seam the API
will call. What remains is the rest of the HTTP surface, persistence, a UI, and packaging.

The reference brief and target architecture this is built against are in
[docs/](docs/): `Enterprise_HR_Agentic_RAG_Problem_Statement_DigitalOcean.pdf` and
`architecture.png`.

## The agent graph

Ten nodes, wired in [app/agent/graph.py](app/agent/graph.py) over bodies in
[app/agent/nodes.py](app/agent/nodes.py). The graph never answers from model memory: every
answer is traceable to internal policy, to cited web results, or to an explicit admission that
neither had the evidence.

```mermaid
graph TD
    S([START]) --> route[route_question]
    route -. "direct" .-> direct[direct_answer]
    route -. "kb" .-> retrieve[retrieve_kb]
    retrieve --> gradekb[grade_kb_evidence]
    gradekb -. "good" .-> genkb[generate_from_kb]
    gradekb -. "weak" .-> web[search_web]
    web --> gradeweb[grade_web_evidence]
    gradeweb -. "good" .-> genweb[generate_from_web]
    gradeweb -. "weak, retries left" .-> rewrite[rewrite_query]
    gradeweb -. "weak, retries spent" .-> insufficient[answer_insufficient]
    rewrite --> retrieve
    direct --> E([END])
    genkb --> E
    genweb --> E
    insufficient --> E
```

Dotted arrows are branches taken on a structured decision. Regenerate the diagram straight from
the compiled graph with `python scripts/render_graph.py`, which writes
[docs/agent-graph.md](docs/agent-graph.md). `test_the_compiled_graph_matches_the_documented_design`
asserts the compiled edge set equals this design exactly, so the picture cannot drift from the
code.

**A rewrite loops back to the KB, not to web search.** A weak retrieval is usually a vocabulary
mismatch between how an employee phrases a question and how policy is written, so the rewritten
query deserves another look at internal policy before falling back to public sources. `retry_count`
bounds the loop at `MAX_RETRIES = 1`.

Verified live, all four terminal paths plus the loop:

| Path | Result |
|---|---|
| `"hi there"` | `direct` |
| `"How many PTO days do I get per year?"` | `private_kb`, answers 20 days, cites the source file |
| `"Who won the football World Cup in 2022?"` | KB returns 0 chunks → `web_search`, flagged as public info, not policy |
| KB and web both empty (injected) | one rewrite, then `insufficient_evidence`; loop terminates |
| KB empty then populated (injected) | recovers to `private_kb` after the rewrite |

---

## Architecture

```
                    ┌──────────────────────────────┐
  public web  ───►  │  WebBaseLoader (scoped to    │
  (HR Acuity)       │  .entry-content)             │
                    └───────────────┬──────────────┘
                                    │
  data/private_kb/**  ───────►  RecursiveCharacterTextSplitter
  .md .txt .pdf .docx               │  1000 chars / 150 overlap
  (subdir = department)             │  markdown splits on headings first
                                    ▼
                    ┌──────────────────────────────┐
                    │  all-MiniLM-L6-v2 (384-d,    │
                    │  normalized, runs locally)   │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Pinecone serverless, cosine │
                    │   namespace "hr-docs"   → 31 │  internal policy
                    │   namespace "public-web"     │  scraped article (optional)
                    └───────────────┬──────────────┘
                                    ▼
                     retriever + raw-cosine gate (0.15)
                                    │
                    ┌───────────────┴──────────────┐
                    │                              │
             chunks returned                 empty result
             → grade & answer          → rewrite / web search
```

**Two corpora, one index, separated by namespace.** Scraped public content and internal policy are
never interleaved in a single retrieval result. The employee-facing agent only ever reads the
private namespace; the public corpus exists to prove the retrieval code is genuinely
namespace-parameterised rather than privately hardcoded. Both namespaces are explicit strings —
Pinecone's default namespace (`""`) behaves inconsistently between `list` and `delete`.

**Retrieval can be narrowed by metadata.** Every chunk carries `department`, `doc_type`, `source`,
`title` and `origin`. A subdirectory under `data/private_kb/` becomes the department, so
`data/private_kb/payroll/bonus.md` is filterable without a manifest, and
`get_kb_retriever(department="payroll")` composes that filter with the similarity gate rather than
replacing it.

**Embeddings run locally.** `all-MiniLM-L6-v2` via `sentence-transformers` — no per-query embedding
cost and no employee question leaving the machine at embed time.

---

## The knowledge base

The problem statement names nine areas PeoplePrime's HR knowledge base holds. All nine are covered
by [data/private_kb/](data/private_kb/):

| Area named in the brief | Document |
|---|---|
| Leave policies | `leave-and-time-off.md` |
| Remote-work guidelines | `remote-and-hybrid-work.md` |
| Attendance rules | `attendance-and-working-hours.md` |
| Payroll information | `payroll-and-compensation.md` |
| Employee benefits | `benefits-and-insurance.md` |
| Onboarding and offboarding | `onboarding-and-offboarding.md` |
| Code-of-conduct guidelines | `code-of-conduct-and-grievance.md` |
| Employee-record policies | `employee-records-and-data-privacy.md` |
| HR forms | `hr-forms-and-requests.md` |

Plus `expense-reimbursement.md` and `hr-copilot-how-it-works.md`, the latter so the assistant can
answer questions about itself from evidence rather than from model memory.

The documents deliberately agree with each other where they overlap — notice periods, PTO payout on
termination, expense payment timing, core hours — because a KB that contradicts itself produces a
confidently wrong answer that no amount of grading catches. A question spanning two documents is
answered from both: see the resignation-and-PTO example, which cites
`onboarding-and-offboarding.md` and `leave-and-time-off.md` together.

---

## Retrieval evaluation

Retrieval quality is measured, not assumed. A 12-question benchmark covers every private document
plus deliberately out-of-domain questions. The numbers below were measured when the KB held six
documents; see [After the KB grew](#after-the-kb-grew) for the re-check at eleven.

| Metric | Result |
|---|---|
| In-domain questions | 10 |
| Top-1 retrieved from correct source document | **10 / 10** |
| Out-of-domain questions | 2 |
| Out-of-domain correctly returning zero documents | **2 / 2** |

### Choosing the threshold

The gate was calibrated from the measured score distribution rather than guessed:

| Score population | n | min | median | max |
|---|---|---|---|---|
| Chunks from the correct source document | 21 | 0.171 | 0.420 | 0.663 |
| Chunks from a wrong source document | 19 | 0.074 | 0.258 | 0.521 |
| Out-of-domain, best match | 2 | 0.005 | — | 0.099 |

Out-of-domain peaks at **0.099**; the weakest correct chunk scores **0.171**. A raw-cosine gate at
**0.15** sits in that gap — it drops every out-of-domain result without losing a single in-domain
top-1.

**The gate is an out-of-domain filter, not a relevance grader.** Wrong-document chunks scored as
high as 0.521, overlapping the correct-document range, so no threshold can separate them. Judging
whether retrieved text actually answers the question stays the LLM grader's job in the agent graph.
The threshold exists to stop obvious noise from reaching it.

### After the KB grew

A fixed threshold is a real risk when a corpus grows: more documents means more chances that an
unrelated question finds something that scores above the gate. So the KB expanding from 6 documents
(15 chunks) to 11 (31 chunks) was re-measured rather than assumed.

| Question | Chunks returned | Top-1 source | Score |
|---|---|---|---|
| "What time must I tell my manager if I am off sick?" | 4 | `attendance-and-working-hours.md` | 0.510 |
| "When am I paid each month and where do I find my payslip?" | 4 | `payroll-and-compensation.md` | 0.697 |
| "How much notice do I have to give when I resign?" | 4 | `onboarding-and-offboarding.md` | 0.472 |
| "How long does the company keep my record after I leave?" | 4 | `employee-records-and-data-privacy.md` | 0.661 |
| "How do I request an employment verification letter?" | 4 | `hr-forms-and-requests.md` | 0.505 |

Top-1 from the correct source on all five new areas. Out-of-domain, against the larger corpus:

| Question | Chunks returned | Best ungated score |
|---|---|---|
| "What is the capital of France?" | **0** | 0.043 |
| "Who won the 2022 world cup?" | **0** | 0.054 |
| "How do I bake sourdough bread?" | **0** | 0.078 |

Out-of-domain still peaks at **0.078**, well under the 0.15 gate, so the calibration holds. Re-run
this with `python scripts/check_retrieval.py` after any further KB expansion — it is the check that
tells you whether the threshold still sits in the gap.

---

## Structured decisions

The graph branches on LLM decisions, so those decisions are Pydantic models bound with
`with_structured_output` rather than parsed out of prose. Two decisions exist today, in
[app/agent/schemas.py](app/agent/schemas.py), with the recovery logic in
[app/agent/decisions.py](app/agent/decisions.py):

| Decision | Values | Purpose |
|---|---|---|
| `RouteDecision.route` | `kb` / `direct` | Does this message need a policy lookup at all? |
| `EvidenceGrade.grade` | `good` / `weak` | Can the retrieved evidence actually answer the question? |

Measured on a labelled set: **router 16/16, grader 7/7.** The router set covers greetings and
questions about the assistant itself (`direct`) alongside policy questions and external factual
questions (`kb`) — both classes matter, because a prompt tightened for one loosens the other.

The grader's value is in the near misses. It correctly returns `weak` for *"What is the mileage
reimbursement rate per kilometre?"* — the evidence comes from the right document, the expense
policy, but that policy never states a rate. It also rejects *"What is the dental waiting period
for contractors?"*, where the benefits document covers employees only.

**The similarity gate and the grader do different jobs.** The 0.15 cosine gate rejects out-of-domain
*questions* before an LLM call is spent; the grader rejects *irrelevant evidence* for questions that
are in domain. "What is the capital of France?" is stopped by the gate and never reaches the grader.
A fabricated *"quokka grooming sabbatical"* question clears the gate — it looks lexically like leave
policy — and is caught by the grader. Removing either leaves a gap.

## Observability

LangSmith tracing is supported and **off unless `LANGSMITH_TRACING` is true and a key is set** —
either alone does nothing, and when tracing is off the switch is actively cleared from the
environment so a stray key cannot turn it on.

The part that matters for an HR product: **traces are redacted by default.**

| | Uploaded to LangSmith |
|---|---|
| Run tree, node sequence, timings | yes |
| Routing and grading decisions, retry count | yes |
| Token counts, errors, latency | yes |
| The employee's question | **no** |
| Retrieved policy text | **no** |

`LANGSMITH_HIDE_INPUTS` and `LANGSMITH_HIDE_OUTPUTS` default to true. That is enough to debug the
agent — which path a question took, where it was slow, what failed — without employee questions or
internal policy leaving the deployment.

Verified end to end rather than assumed: a probe run carrying a sentinel string was sent to the live
service and read back with `inputs={}` and `outputs=None`.

`/health` reports both flags, so a running deployment shows its own posture:

```json
{ "status": "ok", "tracing": true, "tracing_redacted": true }
```

Turn redaction off only to debug a specific answer, and turn it back on — the `/health` field exists
so that a temporary change does not quietly become permanent.

---

## Setup

Requires Python 3.13 and a Pinecone account. Embeddings run locally, so the first run downloads
`sentence-transformers` and PyTorch (~2.5 GB).

```bash
python -m venv hr
hr\Scripts\activate
pip install -r req.txt
```

Copy [.env.example](.env.example) to `.env` and fill in your keys:

```bash
cp .env.example .env
```

`.env` is gitignored and has never been committed. The Pinecone index is created automatically on
first run (384-d, cosine, serverless).

Nothing is required in `.env` beyond the three API keys — every other setting has a default, and
`python run.py` starts even with keys missing so that `/health` can report which ones by name.

> **Migrating from an earlier checkout.** The private namespace default changed from `private-kb`
> to `hr-docs` and the public corpus moved out of Pinecone's default namespace (`""`) into
> `public-web`, both to match [docs/architecture.png](docs/architecture.png) and because the empty
> namespace behaves inconsistently between `list` and `delete`. Re-ingest, then drop the old one:
>
> ```bash
> python scripts/ingest_private_kb.py --drop-namespace private-kb
> ```
>
> Set `PRIVATE_NAMESPACE=private-kb` in `.env` instead if you would rather not move. Writer and
> reader read the same setting, so either choice is consistent — but a namespace mismatch is
> silent, since the upsert succeeds and every later retrieval just returns `[]`.

## Running

```bash
python scripts/ingest_private_kb.py   # rebuild the private KB  -> "hr-docs"
python scripts/ingest_public_web.py   # scrape the public article -> "public-web"
python scripts/demo.py                # four demos, one per path
python scripts/demo.py 2              # just demo 2
python scripts/check_retrieval.py     # what retrieval returns, gated and ungated
python scripts/render_graph.py        # regenerate docs/agent-graph.md
python run.py                         # start the API on :8000
pytest                                # 101 tests, no network, no credentials
```

Ask a single question:

```python
from app.services.copilot import Copilot

copilot = Copilot()                   # builds the graph once, on first ask
result = copilot.ask("How many PTO days do I get per year?")

result.answer          # the text
result.source_used     # private_kb | web_search | direct | insufficient_evidence | error
result.sources         # [SourceRef(source=..., title=..., department=...), ...]
result.model_dump()    # JSON-ready, and the shape /chat will return
```

`AnswerResult` carries the decision trace the UI is meant to show — route taken, source used,
retry count, how each stage graded, and which documents were cited. `render_answer(result)`
formats it for a terminal; `scripts/demo.py` uses exactly that.

The node trace goes to **stderr** via logging, so `python scripts/demo.py > out.txt` captures the
answers without it.

### Demos

```
Demo 1  Answer from the Private KB   "How many PTO days do I get, and do unused days carry over?"
Demo 2  Web Search Fallback          "What is the statutory minimum paid holiday in the UK?"
Demo 3  Direct Answer                "hi there, thanks for the help earlier!"
Demo 4  Current / External Question  "What does recent research say about four-day work weeks?"
```

| Demo | route | source_used | retries |
|---|---|---|---|
| 1 | `kb` | `private_kb` | 0 |
| 2 | `kb` | `web_search` | 0 |
| 3 | `direct` | `direct` | 0 |
| 4 | `kb` | `web_search` | 0 |

Demos 2 and 4 both end at web search, and the difference is the point. Demo 2 is a genuine HR
question this company's KB does not cover; demo 4 is one no internal policy could ever answer. In
both, retrieval returns four chunks that *look* plausible and the grader marks them `weak` — the
similarity gate alone would have let them through.

Both ingests are idempotent — chunk ids are derived from `origin` + `source` + `start_index`, so
re-running overwrites rather than duplicating.

## Repository layout

Four layers. Each imports only from those above it, and nothing imports downward or from a script.

```
app/core/config.py        Settings (pydantic-settings), logging, console setup
app/rag/clients.py        embeddings (cached), Groq chat model, Tavily search
app/rag/loaders.py        markdown/text/PDF/DOCX/web loading, chunking, chunk ids
app/rag/vectorstore.py    Pinecone, parameterised by index *and* namespace
app/rag/ingest.py         load -> chunk -> embed -> upsert -> reconcile, for any corpus
app/rag/retrieval.py      the private KB and public corpus, namespaces already chosen
app/agent/schemas.py      routing/grading decision models, AgentState
app/agent/prompts.py      every prompt, including the two calibrated ones
app/agent/decisions.py    router and grader + recovery from structured-output failures
app/agent/nodes.py        node bodies, as closures over injectable clients
app/agent/graph.py        wiring and compilation
app/agent/diagram.py      render the compiled graph
app/services/copilot.py   Copilot facade and AnswerResult -- the API contract
app/api/                  FastAPI routers (only /health so far, in app/main.py)

scripts/                  CLI entry points: ingest, demo, diagnostics, diagram
tests/                    101 tests over fakes -- no network, no credentials

data/private_kb/          11 internal HR policies; a subdirectory is a department
step.md                   the 16-step build plan this project is working through
docs/                     the reference brief, target architecture, generated graph
req.txt                   pinned dependencies (requirements.txt forwards to it)
CLAUDE.md                 architecture notes and non-obvious constraints
```

Corpora differ only in where text comes from, which separators split it, and which namespace it
lands in. Ingest, freshness waiting, orphan pruning, metadata filtering, gating and retrieval are
one implementation taking a namespace argument.

Every client is injectable at every level — `build_nodes`, `build_graph` and `Copilot` all take
their dependencies as arguments. That is what lets the whole suite run offline, and it is why
`tests/conftest.py` needs only three fakes.

---

## Engineering decisions and problems solved

Notes kept because each of these failed silently rather than raising an error.

**Scoped the loader to the article body.** `WebBaseLoader` over the full page ingested navigation,
related-post cards and footer boilerplate — 18,520 characters, roughly 20% junk. Restricting
extraction to the article container cut it to 14,803 characters of real policy text.

**BeautifulSoup matches the whole class attribute.** `SoupStrainer(class_="entry-content")` silently
matched nothing, because the element's attribute is `"entry-content section-padding relative ..."`.
A callable matcher that splits the attribute into tokens fixed it. An explicit guard now raises if
extraction yields zero text, so a future layout change fails loudly instead of indexing nothing.

**LangChain's `score_threshold` is not on the cosine scale.** It maps raw cosine `c` to `(c + 1) / 2`.
Passing a raw cosine straight through filters nothing — `0.15` is read as raw cosine `−0.70`, which
nothing falls below. The conversion is now explicit so callers reason in raw cosine, the same scale
the debugging output prints.

**Pinecone serverless is eventually consistent.** A retrieval issued immediately after an upsert
returns an empty list. Ingestion polls `describe_index_stats` until the written vectors are actually
queryable before returning.

**A namespace typo fails silently.** The upsert succeeds and every subsequent retrieval returns `[]`.
Writer and reader share a single `PRIVATE_NAMESPACE` constant so the two cannot drift.

**Ingestion was not idempotent.** `PineconeVectorStore.from_documents` mints fresh UUIDs per call, so
every re-run doubled the corpus. Deterministic ids from `source` + `start_index` make re-ingestion an
overwrite — verified by two consecutive runs both leaving 18 vectors.

**Any single service outage took down the whole agent.** Six nodes called Pinecone, Tavily or the
LLM with no error handling, so one flaky call aborted the run — worst of all in `search_web`, which
*is* the fallback and had none of its own. Every node now degrades instead of raising: retrieval
failure becomes zero chunks and routes to the web, search failure becomes empty evidence, a failed
rewrite keeps the original query, and a failed generation returns an honest message rather than
nothing. Verified by injecting failures at each node and asserting the run still terminates with an
answer.

**An exhausted API quota was indistinguishable from a knowledge gap.** Groq's free tier caps tokens
per *day*. When it ran out mid-testing, every grade came back `weak` and every run ended
`insufficient_evidence` — the safe defaults working correctly, but silently. Rate-limit rejections
are now detected and logged at ERROR as degraded results rather than judgements.

**The embedding model was reloaded on every call.** `get_embeddings()` was uncached, so any
retrieval that did not pass an explicit `embedding` rebuilt a ~90 MB sentence-transformers model
from disk. Harmless in a script that passes one through; ruinous in a request handler that
retrieves per turn. Now `lru_cache`d and verified to return the same instance.

**The two corpora had two parallel implementations.** Ingest, chunking, freshness waiting and
retrieval were written twice — once for the public article, once for the private KB — and had
already drifted: the public path had no eventual-consistency wait and no orphan pruning, both of
which the private path needed. They are now one namespace-parameterised implementation in
`vectorstore.py`, so a fix lands in both places at once.

**Upsert-only ingestion served withdrawn policy.** Re-ingesting updates and adds vectors but never
removes them, so deleting a policy document left its vectors in the index still answering employee
questions. Ingestion now reconciles: it lists the namespace and deletes vectors with no
corresponding chunk on disk. Verified by adding a temp policy (retrievable at 0.853), deleting the
file, re-ingesting, and confirming it no longer retrieves.

**Package imports were broken.** An early `from Rag.private_kb import ...` raised
`ModuleNotFoundError` because the module used a script-relative import — fine when run directly,
fatal for the FastAPI and LangGraph layers. Logic now lives in the `app` package and scripts are
thin callers, so imports work regardless of how a file is invoked.

**The agent imported an entry point.** `graph.py` pulled `get_kb_retriever` from `private_kb.py` —
a corpus script — so the agent depended on a CLI. Retrieval accessors moved into
`app/rag/retrieval.py` and the dependency now runs one way only.

**Twenty backward-compatibility aliases had no callers.** Both entry points re-exported names
(`_chunk_id`, `PINECONE_INDEX_NAME`, `_to_relevance_scale`, …) for imports that never existed
outside them. Deleted: a compatibility shim with nothing to be compatible with is just a second
name for everything.

**The index name was reachable only through module globals.** `vectorstore` read
`config.PINECONE_INDEX_NAME` directly in nine places, so nothing could target a second index —
including a test. Every operation now takes `index_name` and `namespace`, defaulting from settings.

**Missing credentials made the package unimportable.** `config.py` raised at import time, which
meant no test could run, no `--help` could print, and a container that gets configuration from the
platform could not start. Secrets are now validated where a client is constructed, with
`validate_required()` for entry points that genuinely need fail-fast, and `/health` reports which
are missing by name.

**Chunk ids could collide across corpora.** Ids were `source` + `start_index`, and `source` is only
a file name. Once HR staff can upload documents, `benefits.pdf` from two departments would produce
one id and the second ingest would silently overwrite the first. `origin` is now part of the basis.

**Import-time side effects removed.** Documents, chunks and embeddings were built at module scope, so
importing the module performed an HTTP fetch and loaded a model. Moved into `main()` so the module can
be imported by the planned FastAPI and LangGraph layers without doing work.

**Unbounded readiness loop bounded.** Waiting for index readiness had no timeout and could hang
forever; it is now capped and raises.

---

## Roadmap

Ordered against [docs/architecture.png](docs/architecture.png) and [step.md](step.md), the 16-step
build plan for this project. Steps 1–9 and 16 are done; 10–15 are what follows.

Two deliberate deviations from step.md, both already in the code:

- **Providers.** The plan says OpenAI embeddings and model. This runs `all-MiniLM-L6-v2` locally
  (no per-query cost, no employee question leaving the machine at embed time) and Groq for
  generation and grading. `openai` and `langchain-openai` are installed only because
  `langchain-pinecone` requires them; nothing imports them.
- **Module paths.** The plan puts ingestion in `app/services/ingestion.py`, state in
  `app/rag/state.py` and the workflow in `app/rag/workflow.py`. They live in `app/rag/ingest.py`,
  `app/agent/schemas.py` and `app/agent/graph.py`, which keeps the agent in one package and stops
  `app/rag/` mixing retrieval with orchestration. Same work, one directory over.

1. `/chat` over `Copilot.ask` — the facade and its response model already exist
2. SQLite data layer: chat history, feedback, document metadata, execution traces
3. `/upload` and `/ingest` for authorised HR staff — the multi-format loader is in place
4. HTML/CSS/JS frontend: chat with citations and a visible execution trace
5. Docker + DigitalOcean: FastAPI app, Nginx frontend, ingestion worker, SQLite volume
6. Conversation memory, so follow-up questions resolve against the previous turn

## Tech stack

Python 3.13 · LangChain · LangGraph · Pinecone serverless · sentence-transformers
(all-MiniLM-L6-v2) · Groq (`openai/gpt-oss-20b`) · Tavily · Pydantic + pydantic-settings ·
BeautifulSoup · pypdf · python-docx · FastAPI · pytest

## License

See [LICENSE](LICENSE).
