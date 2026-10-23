Part of the [HR Copilot](../README.md) documentation set.

# Running this in production

The layers between an HTTP request and a model call: routing, caching,
streaming, limiting, bring-your-own-key, and what each of them is worth behind a
load balancer.

All of them are called from one place —
[`app/services/pipeline.py`](../app/services/pipeline.py), the eleven-stage
`AnswerPipeline` around the agent:

```
 1. trace and budget        one tracer and one budget per run, shared
 2. input guard             refuse, or redact and carry on
 3. semantic cache          an answer already given to this question
 4. model routing           complexity -> tier -> model id
 5. orchestration choice    graph | react | supervisor
 6. run                     streaming the stages as they happen
 7. generation fallback     retrieved evidence, no model -> extract
 8. output guard            redact what the model echoed; flag the rest
 9. accounting              tokens, cost, llm calls, budget report
10. cache store             only an answer worth repeating
11. assemble                one AnswerResult, which is the whole contract
```

Every stage can be switched off by configuration, because a feature nobody can
turn off is a feature nobody can measure. With every stage off, the pipeline is
the behaviour this project shipped with before any of them existed.

Vercel specifics — environment variables, the three key modes, cold starts, and
what does not work on a serverless filesystem — are in
[docs/deploy-vercel.md](deploy-vercel.md) and are not repeated here.

---

## 1. Model routing

Every question in this system paid for the same model. "Hi" paid for it,
"thanks" paid for it, and so did a two-area prorated-accrual question that
genuinely needed the better one. Routing is the largest saving available that
does not involve answering fewer questions — and it is also the one most likely
to quietly damage quality, because **a cheap model on a hard question produces a
plausible answer, not an error.** Nothing in the logs says "this one should have
gone to the big model".

### The ladder

| Complexity | Tier | Model id | Source |
|---|---|---|---|
| `simple` | `fast` | `FAST_MODEL` | `Settings.fast_model` |
| `standard` | `balanced` | `active_llm_model` | the deployment's ordinary model |
| `complex` | `deep` | `DEEP_MODEL` | `Settings.deep_model` |

`_model_for` collapses an unconfigured rung onto `active_llm_model` rather than
failing on an id the configured key cannot reach, so `ModelRouter.tiers()` never
returns an empty string. Two tiers showing the same id is the honest picture of
a deployment that set neither end of the ladder.
`test_tiers_never_returns_an_empty_model_id` pins it.

### Why the classifier is a heuristic

`classify_complexity` is pure Python with no model call. Asking a model which
model to use costs a model call, which is most of what routing was trying to
save on a short question. `COMPLEXITY_PROMPT` exists in
`app/agent/prompts.py` for the deployment that would rather pay it, and
`ModelRouter.__init__` already accepts an `llm` it holds and does not use —
so that classifier can arrive as an injected client rather than as a signature
change at every call site.

| Option | Cost | Rejected because |
|---|---|---|
| **Heuristic (chosen)** | zero | it is lexical, so it will be wrong on wording it has not seen |
| Model call per question | one extra round trip on every question, including "hi" | it taxes the cheap case to classify the cheap case |
| No classification at all | zero | every question pays the deep model, which is the problem |

### The conservative default

```python
return "standard", "no complexity signal; the middle tier is the safe default"
```

Every branch above that line has to *earn* its answer. A question that matched
nothing is a question that was not understood, and the only defensible thing to
do with a question you have not understood is spend the middle amount on it. A
classifier whose default were `simple` would be cheapest precisely on the
questions it failed to read.

### The measured correction

The arithmetic detector was originally **one tuple**, any match promoting to
`complex`. Measured against the console's own suggested questions, it was wrong
in both directions at once:

| Question | Old verdict | Correct |
|---|---|---|
| "How many days of paid time off do full-time employees get?" | `complex` | `standard` |
| "What is the notice period if I resign?" | `complex` | `standard` |
| "I'm a contractor going on parental leave mid-notice, what am I owed?" | `standard` | `complex` |

"How many" is how the single most ordinary HR question there is begins. Treating
it as arithmetic routed two plain lookups to the deep model and — on
`AGENT_MODE=auto` — to the supervisor as well: four model calls for a question
one retrieval answers. Meanwhile the genuinely hard one in the same set came out
`standard`. The classifier was not slightly miscalibrated; it was inverted on
exactly the cases that matter.

Three changes fixed it:

**Strong and weak signals.** `ARITHMETIC_STRONG` ("difference between",
"versus", "prorated", "calculate", …) promotes on its own.
`ARITHMETIC_WEAK` ("how much", "how many", "how long", "if i", "remaining",
"total", …) is ordinary alone and suggestive in company — one weak signal
promotes only alongside a second, or inside a question with some size to it
(`WEAK_WORDS = 18`, `WEAK_CLAUSES = 3`). "How much leave" is a lookup; "how much
leave do I carry over if I started in March" is not.

**The population signal.** `POPULATION` — contractor, intern, part-time,
probation, fixed-term, secondment, … — is the strongest single-area signal
there is, and the reason is in the corpus rather than in the grammar. The policy
documents say "employees", and whether a statement about employees covers a
contractor is precisely the judgement the LLM grader is recorded as finding
borderline (`CLAUDE.md` records the 8/8-in-isolation grade that flipped inside a
longer run). A question naming one of these needs the better model whatever else
it looks like. `"full-time"` is deliberately **absent**: it is the population the
corpus *is* written for, so naming it makes a question more ordinary, not less.

**One area word moved.** `"accrual"` / `"accrue"` look like payroll words and
are not, in this corpus — "full-time employees accrue 20 days of paid time off"
is a leave sentence. Filing them under `pay` made every ordinary PTO question
report two policy areas and route deep. Measured on "Do full-time employees
accrue paid time off?", which came out `complex` for no reason a reader of the
question would recognise.

### The labelled set

`LABELLED_COMPLEXITY` in [`tests/test_routing.py`](../tests/test_routing.py) is
14 questions with expected bands, asserted as a table rather than as 14 tests —
because the value is the *ratio*. A classifier is not right or wrong on a
question, it is right on a proportion of them, and a change that fixes one case
by breaking two is a change individual tests would report as progress.

**14/14 now. 10/13 before the strong/weak split**, with the two
inversions above among the failures.

### The cost-budget override

```python
elif budget is not None and not budget.allows("cost", COST_HEADROOM_USD):
    tier = "fast"
    model = self._model_for("fast")
    reason = "cost budget nearly spent; finishing on the fast model"
```

`COST_HEADROOM_USD` is 0.01. Below that much projected headroom, the run
finishes on the cheap model rather than not finishing: an employee with an
answer from a small model is better served than one whose question stopped
mid-ladder. This is the one case where the ladder is descended for a reason that
has nothing to do with the question.

### It switches off

`MODEL_ROUTING_ENABLED=0` pins every question to `active_llm_model`. That is the
unrouted baseline quality has to be measured against — a saving nobody measured
against a baseline is not a saving. Note that `classify_complexity` still runs
when routing is off: it costs nothing, and a deployment measuring whether
routing hurt quality needs both halves recorded under the same name.

Every decision is reported. `RoutingDecision.to_dict()` carries the model, the
tier, the provider, the complexity band and **the reason in words**, and it goes
into the trace, the API response and the audit row — so "why did that answer
come out thin?" has an answer that does not require re-running it.

---

## 2. Semantic caching

Caching by *meaning* is the most dangerous cache in this product.

> "how much PTO do I get?" and "how much PTO do contractors get?" differ by one
> word and by the entire answer, and the KB has a separate policy for each.

A cache that serves the first to the second is not a slow answer or a missing
answer. It is a confidently wrong HR answer with a citation attached, which is
the one failure this project spends most of its design avoiding. So every knob
is set *against* the cache rather than for it.

### Two layers, exact first

`SemanticCache.get` tries a normalised-text match before it embeds anything.
`normalise_question` casefolds, collapses whitespace and strips trailing
punctuation — so "PTO?" and "PTO" are one key, while nothing internal is touched
("don't" and "dont" are not something to guess at).

The exact layer catches the common case: the same question asked twice, a page
reload, a retry. Paying an embedding round trip to discover you are holding the
identical string is the cache charging rent.

It also means the cache **works with no embedding provider at all**.
`embedding=None` degrades to exact matching, which is the mode a deployment with
no embedding provider runs in, and it still catches reloads.
`stats()["mode"]` reports `semantic`, `exact` or `disabled`, because the two
modes have very different hit rates and a deployment that silently lost its
embedding provider would otherwise look like one whose cache simply stopped
helping.

### The threshold

`CACHE_SIMILARITY` defaults to **0.95 raw cosine**. Measured on the labelled
set and recorded in `app/core/config.py`: paraphrase pairs sat at **0.94–0.99**,
and the closest *different* question pair at **0.91**. So 0.95 buys the reloads
and the retries and declines the near misses.

**A cache that hits more often is not a better cache here.** The PTO/contractor
pair is the worked example: lexically and semantically close, entirely different
correct answers.

`_take_nearest` returns the best score even on a miss, which is what lets
`cache.miss` carry `reason="nearest 0.912"` — the number that tells you whether
the threshold is doing its job or merely sitting above everything.

### The TTL

**Fifteen minutes** (`CACHE_TTL_SECONDS=900`). Policy changes, and a stale HR
answer is a wrong answer that looks exactly like a right one: same prose, same
citation, superseded content. Fifteen minutes covers the burst one announcement
causes without outliving an edit to the policy it is about.

Expiry is lazy, on read, with the lock held. No sweeper thread: a sweeper would
need a lifecycle, and the only entry whose staleness matters is one somebody is
about to be served.

### `CACHEABLE_SOURCES`

```python
CACHEABLE_SOURCES = {"private_kb", "web_search", "direct"}
```

The two that are missing are the point:

| Excluded | Why |
|---|---|
| `error` | caching a failure for fifteen minutes turns one outage into fifteen minutes of outage, and the retry that would have worked never reaches a provider |
| `insufficient_evidence` | caching "I could not find that" means the ingest that fixes it changes nothing for a quarter of an hour, to the one employee most likely to ask again |

An **allowlist** rather than a denylist, so a payload with no `source_used` at
all — a partial state, a hand-built dict — is declined rather than stored on the
assumption that it was fine. `escalated` is not on the list either, so an
escalation is never replayed to the next person.

The whole answer object is stored, citations included. Serving cached text beside
freshly computed citations would attach today's sources to yesterday's words,
which is worse than either being stale alone — the citation is what an employee
checks the answer against. `_from_cache` replays the payload whole and overwrites
only the fields that describe *this* request, then prepends one trace line
saying nothing was retrieved or graded on this request.

### Invalidation on ingest, and why the TTL is not enough

`SemanticCache.invalidate()` drops everything. Whole-cache only, on purpose: an
ingest can change the answer to a question that mentions none of the words in
the file, so there is no honest way to invalidate selectively — working out which
cached answers a new document affects is the retrieval problem again.

`/api/upload` calls it itself after a successful ingest.
`POST /api/cache/invalidate` (gated on the `cache.invalidate` action) is for an
ingest that happened elsewhere — a CLI script, a worker. Without this,
"the cache expires in fifteen minutes" means fifteen minutes of serving
superseded policy after an HR administrator has corrected it.

### The per-caller cache

A caller spending their own key through `app/services/byok.py` gets their own
`Copilot` — and therefore their own pipeline and their own cache — out of
`ClientPool`. That is not only about which key pays.

A shared cache across callers is a cross-caller inference channel: whether an
answer comes back in 40 ms or 4 s tells you whether *somebody else* has recently
asked something within 0.95 cosine of your question. For a product where the
questions are "am I still covered during my notice period" and "how do I file a
grievance", that is a side channel worth closing. One cache per credential
fingerprint closes it, at the cost of a lower hit rate.

---

## 3. Streaming and latency

An answer costs several seconds of provider round trips: contextualise, route,
retrieve, grade, maybe rewrite, maybe search, then generate. A single JSON
response at the end of that is a spinner for eight seconds — and the thing that
makes this product worth looking at, the decisions it took on the way, arrives
only after the user has stopped watching.

### SSE, not WebSockets

| | SSE (chosen) | WebSockets |
|---|---|---|
| Direction | one-way, which is all this needs | duplex, unused |
| Transport | plain HTTP; survives every proxy that would need configuring for an upgrade | needs an upgrade path through each hop |
| Reconnect | built into the browser | hand-rolled |
| Server shape | a generator | a connection to manage |

Nothing here needs a socket.

### POST, not `EventSource`

The browser's `EventSource` only issues GET, which would put the question and
the whole conversation **in a URL** — and a URL reaches the history, the proxy
log and the `Referer` header. `POST /api/chat/stream` is read with `fetch` and a
stream reader instead: a few more lines of JavaScript, and an employee's HR
question stays out of three logs it has no business being in. (The same
reasoning the admin key already follows — it is a header, never a query string.)

A consequence: a refusal arrives as an `error` *event* carrying our own text,
not as a status code. Once the response has started streaming there is no status
left to set, so a 429 or a 400 is emitted in-band and the stream closes with
`ok: false, reason: "refused"`.

### The worker thread

`Copilot.ask` is synchronous and spends its time blocking on HTTP. Running it
inside an async generator would block the event loop for **every other request
in the process** — the same reason `/chat` and `/audit` are `def` rather than
`async def` (Starlette runs a sync handler in a threadpool).

So `stream_in_thread` runs the pipeline on a daemon thread and the generator
drains a `queue.Queue`. The thread is a daemon because a worker still blocked on
a provider when the process is asked to shut down must not hold the shutdown
open; it has already lost its reader by then. `work` does not have to call
`close()` — `stream_in_thread` does, in a `finally`, so a `work` that raises
still terminates the stream rather than leaving the client waiting for a
sentinel that is never coming.

### The heartbeat

A comment frame — `: ping` — every `heartbeat_seconds` (12 by default) keeps the
connection alive through an intermediary with a read timeout. A comment rather
than an event, so no client has to know about it. An opening `: stream open`
frame is sent immediately, so the browser fires `open` and the UI can swap its
spinner for a live indicator before the first stage arrives, which on a cold
start can be several seconds.

### Back-pressure resolved in favour of the worker

A browser that walked away leaves nothing draining the queue. Blocking the
worker on a full queue would pin a thread and its provider connections until the
request times out.

So a full queue **drops the oldest progress event and counts it**, while
`_UNDROPPABLE` — `end`, `result`, `error`, `answer` — makes room by discarding a
droppable event instead. A stream that lost three trace lines is a cosmetic
loss; a leaked thread is not. `EventStream.stats()` reports `emitted` and
`dropped`.

### The terminator

```python
STREAM_END = "end"
```

A client that reaches the end of the response body without having seen this
knows the stream was **cut** — the function timed out, the proxy gave up, the
process died. That is a different thing from an answer that finished. Without a
sentinel the two are indistinguishable and the UI shows a truncated answer as if
it were complete.

The stream's own ceiling is derived from the agent's budget rather than set
independently: `settings_stream_ceiling` returns `max_agent_seconds + 20.0`. Two
unrelated ceilings is how a stream times out just before the answer it was
watching arrives, on the one question that took longest.

### Where the time actually goes

`app/core/tracing.py` builds a span tree inside the request, returns it on the
response and renders it in the console. `tracer.summary()` reports **self-time
per kind** — a span's duration minus the duration of the spans directly inside
it — because nested spans double-count: an `agent` span wrapping four `llm`
spans would otherwise report the LLM time twice and make the percentages sum
past 100.

The spans on the single-graph path, in order:

| Span | Kind | Serial or parallel |
|---|---|---|
| `answer` | `agent` | the root; wraps everything |
| `Contextualiser` | `llm` | serial — and **skipped entirely** with no history |
| `KB Retriever`, wrapping `retrieve.hybrid` in hybrid mode | `retrieval` | **the dense and lexical sides run serially inside it; they could be parallel** |
| `rerank` | `retrieval` | serial after retrieval, by definition |
| `Tavily` | `tool` | serial, and only on the web fallback — **could overlap KB retrieval** |
| `Rewriter` | `llm` | serial; loops back into `KB Retriever` |
| `Generate/KB`, `Generate/Web` or `Generate/Direct` | `llm` | serial, last |

Two honest caveats about reading that tree:

- **The router and the graders are not timed.** `route_question`,
  `grade_kb_evidence` and `grade_web_evidence` record through `note()`, which
  calls `tracer.event(...)` — a zero-duration span. Their model calls happen
  inside `app/agent/decisions.py`, which does not trace. So `by_kind["llm"]`
  self-time is a **lower bound** on model time, not the total. Adding spans
  around `classify_route` and `grade_evidence` is the obvious next change.
- **The two retrieval sides run one after the other, and are not timed
  separately.** `HybridRetriever.search` calls `_side("retrieve.dense", …)`
  then `_side("retrieve.lexical", …)`, and `_side` opens no span — those names
  appear in the tree only as a `degraded` record when a side fails. The
  `retrieve.hybrid` span carries `dense=` and `lexical=` *counts* as
  attributes, not durations, so the tree cannot currently tell you which half
  was slow. The two calls are independent, which makes them the one clear
  parallelisation available on the hot path: `ToolRegistry.call_parallel`
  already shows the shape, and `Tracer` is thread-safe for it (depth is a
  `threading.local`, the span list is behind a lock), so a fan-out reads as a
  fan-out rather than as a chain.

On the supervisor path the handoffs are serial **by construction** — the
supervisor cannot choose the second specialist until the first has reported — so
there is nothing to parallelise there without changing the design. See
[docs/multi-agent.md](multi-agent.md).

### What to measure, rather than a number this repo has not taken

**No latency figures are quoted here, because none have been measured over a
representative population of questions on this deployment.** Two published
numbers from `CLAUDE.md` exist as single observations from LangSmith traces
(`grade_web_evidence` 1511 tokens / 367 ms, `tavily_search` 2387 ms) and one
container measurement (a cold `embed_query` spending ~40 s on failed
huggingface.co HEAD requests without `HF_HUB_OFFLINE=1`, against 4.9 s with it).
Neither is a latency distribution.

The instrumentation to get one is in place:

- `GET /api/metrics` reports `latency_p50_ms`, `latency_p95_ms`,
  `latency_p99_ms` and `latency_samples` over the **newest 1,000 answers** in the
  audit log. Bounded on purpose — percentiles are computed in Python, so an
  unbounded `SELECT` would load the whole log into memory to answer a dashboard
  request, and a p95 over a deployment's entire history is a number that stops
  moving when it matters most.
- Percentiles rather than a mean, because averaging is the statistic that hides
  the problem: a p95 of nine seconds and a mean of two tell completely different
  stories about the same deployment, and only one of them is what a user
  experienced.
- Per-answer, `AnswerResult.span_summary` carries the self-time by kind, so a
  single slow answer can be attributed without re-running it.

The numbers to compare: `by_kind["llm"]` self-time against
`by_kind["retrieval"]`, and p95 against p50. A p95 that is several times p50
with LLM self-time dominating is a provider-queue problem; the same gap with
retrieval dominating is a cold embedding model or a Pinecone region away from
the compute.

---

## 4. Horizontal scaling

### Stateless

- `Copilot` and `AnswerPipeline` hold no per-request state. The conversation
  lives in the **browser** and is posted back with each turn, so one process
  serves every employee and holds nobody's thread.
- Pinecone, Groq/OpenAI and Tavily are external. Any instance can serve any
  question.
- `AnswerPipeline.__init__` touches neither network nor disk; the retriever is
  built on first use and `AuditStore` creates its schema on first use.
- The retriever and its embedding model are a deliberate process-wide cache, not
  state: identical on every instance.
- The tool registry is built **per request**, in
  [`app/services/toolbox.py`](../app/services/toolbox.py): `raise_ticket` and
  `escalate_to_human` close over *this* question and *this* conversation id, so
  a shared one would file one employee's escalation against another's question.
  `GET /api/tools` renders the catalogue for the calling principal, which is
  also how you see what an anonymous visitor may and may not reach.

### Per-process, and what each one means behind a load balancer

| Thing | Behind N instances | Consequence | The fix | Done here? |
|---|---|---|---|---|
| `SemanticCache` | N independent caches | hit rate falls roughly as N rises; **correctness unaffected**, only the saving | a shared cache (Redis, Upstash) keyed the same way | no |
| `RateLimiter` | N token buckets | **the effective limit is `instances × rate`**, and a caller who reconnects may land on a fresh bucket | a limiter *in front of* the application — the platform's edge limiting, or a shared counter | no |
| `ClientPool` (BYOK) | N pools | the first request to each instance rebuilds that caller's client, so a cold instance recompiles a graph | nothing to share; it holds clients, and a shared pool would mean sharing keys across processes | no, and deliberately |
| `IdempotencyLedger` (in-memory) | N dicts | **not a distributed lock**: two concurrent requests in two processes can each win the same claim, so a retried `raise_ticket` can duplicate | `STATE_DB_PATH` on a shared volume, or an external store with a unique constraint | partially — SQLite mode exists; in-memory is the default |

None of this is hidden at run time. `RateLimiter.stats()` reports
`"scope": "process"` **in the payload**, not only in a docstring, because a
dashboard reporting a limit that is actually per-instance is a dashboard that is
wrong. `app/services/idempotency.py` says the in-memory mode is "the right
default for a single container and the wrong one for anything that scales
horizontally; nothing here pretends otherwise."

On the rate limiter specifically, the honest accounting is: on **one container**
it is the real thing. On several instances it still buys something — an
accidental retry loop or a single enthusiastic visitor cannot drain the day's
tokens from one instance — but it is not a limit you can quote to anyone. A
deployment that needs a real one needs a shared counter, and the right place for
it is in front of this application, not inside it.

The idempotency gap bounds how much the `raise_ticket` guarantee is worth on a
multi-instance platform. In practice the **approval gate** in front of it is
doing most of the work there: a ticket needs a human decision before it happens
at all.

### SQLite on a volume, versus an ephemeral filesystem

Two separate SQLite databases, two different stakes:

| Database | Setting | What it holds | Losing it costs |
|---|---|---|---|
| `audit.db` | `AUDIT_DB_PATH` | every question and the answer given | the compliance record — the thing this product exists to provide |
| the state DB | `STATE_DB_PATH` | scratchpad, approvals, facts, idempotency | working state; a decision that can be made again |

`kb_dir`, `upload_dir` and `audit_db_path` are **settings, not fixed
properties**, precisely so a deployment can split them. In the container,
`/app/var` is the only writable path and the whole of the container's state:
the source, the templates and the knowledge base stay root-owned and read-only
to the app user, so a bug in the upload handler cannot rewrite the code that
serves the next request or the policy that answers it. The check is
`find /app -maxdepth 1 -writable`, which must print exactly one line.
`docker-compose.yml` mounts one named volume at that path.

**On a platform with an ephemeral filesystem the audit log does not survive a
redeploy, and that is the failure the log exists to prevent.** A deployment that
needs it needs a database outside the function — Postgres, Turso, Neon — and
`AuditStore` would need a driver rather than `sqlite3`.

**This is the deployer's decision, not something to quietly fix in a
Dockerfile.** Baking a hosted database driver into the image would mean every
deployment carries a dependency and a connection string for a service most of
them do not use, and the honest shapes available differ per platform. The same
goes for the other half: `/api/chat` is unauthenticated by design, so a public
URL with no rate limit in front of it is a bill waiting to happen. Both are
stated in the README's *Deploying* section and in
[docs/deploy-vercel.md](deploy-vercel.md), which lists the per-instance caveats
as they apply there.

What the image does do is fail loudly on the one thing it can check:
`APP_ENV=production` with no `ADMIN_API_KEY` refuses to build the application
(`Settings.validate_admin_surface`, called from `create_app`). A *missing
provider key* deliberately does **not** fail startup — `/health` coming up and
reporting `degraded` with the missing names is more use to a deploy than a
container that exits before it can say why.

---

## 5. Cost projection

### What the numbers are

`MODEL_CATALOG` in `app/services/routing.py` holds one `ModelSpec` per model:
the name, the provider, the tier, input and output dollars per million tokens,
the context window, and a `note` saying how confident the price is.

`estimate_tokens` is **characters over four**, rounded up:

```python
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
```

That is about **±20% on English prose**, and worse on code, tables and the
markdown pipe tables the generator emits — all of which tokenize denser than
words do. A real tokenizer is a dependency (`tiktoken`, and a different one per
provider) plus a download, to improve a number that drives a budget ceiling and
a dashboard. Both tolerate 20%: a run stopped at 115k estimated tokens instead
of 120k actual ones stopped for the right reason.

`project_cost(model, input_tokens, output_tokens)` multiplies the two. An
unlisted model returns **0.0 and logs at WARNING** — inventing a price for a
model nobody put in the table would put a made-up number in a budget ceiling,
but a dashboard reading 0.00 because the model id was renamed looks identical to
one reading 0.00 because the deployment is on a free tier, and those want very
different responses.

`ASSUMED_COMPLETION_TOKENS = 400` is what a completion is assumed to cost when
projecting *before* the call. An HR answer from this system runs two to five
short paragraphs; 400 is the middle of that. It is an assumption, not a
measurement, and it moves only the output half of a projection.

### These are projections, not invoices

Say it plainly, because the decimal places invite the opposite reading. The
projection is a token estimate at ±20% multiplied by a published list price that
moves and that nothing here reconciles against anything. **The product is an
order-of-magnitude figure with decimal places. Read it as a ceiling.**

Two specific reasons not to trust it as a bill:

- **Groq's free tier bills nothing at all.** It caps tokens per *day* instead.
  `CLAUDE.md` records an observed exhaustion:
  `429 ... tokens per day (TPD): Limit 200000, Used 199771`. On the default
  provider this projection is a token-burn estimate wearing a dollar sign, and
  the thing that actually stops a run is the quota. Worse: because every safe
  default in this system is "weak", an exhausted quota looks exactly like a
  knowledge gap — the router returns `kb`, every grade returns `weak`, and every
  run ends `insufficient_evidence`. `is_rate_limit()` distinguishes it and those
  paths log at ERROR saying the results are degraded rather than a judgement.
  Limits are per-model, so `GROQ_MODEL=openai/gpt-oss-120b` is a working escape
  hatch.
- **Agentic endpoints are billed through their own tool calls.**
  `groq/compound` calls its own tools, so it is billed through whichever
  underlying models it used, which is not knowable from here. Its `note` says
  so, and its per-token figures are the 120b's as a stand-in.

**The number to trust is the provider's billing page.** `/api/metrics` returns
`cost_is_projected: true` so a reader does not have to know any of the above,
and per-model spend is withheld from a caller without the `metrics.read`
permission — the answer counts are not commercially sensitive and the dollar
split across providers is.

### Tokens are estimated, not measured

`AnswerPipeline._finish` is explicit about this. LangChain surfaces usage
metadata inconsistently across providers and not at all through a compiled
graph's final state, so the pipeline counts characters instead:

```python
prompt_chars = len(result.question) * max(1, llm_calls)
prompt_chars += result.kb_chunks * self.settings.chunk_size
result.prompt_tokens = (prompt_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
result.completion_tokens = estimate_tokens(result.answer)
```

Every model call sees the question; the generating call also sees the evidence,
approximated as chunk count times the configured chunk size. Closer than
ignoring it, and nowhere near an invoice. `llm_calls` is counted from spans of
kind `llm` — which, per section 3, under-counts the router and the graders.

### The per-run budget

`Budget.from_settings(settings)` carries six ceilings, defaults in brackets:

| Kind | Default | Setting |
|---|---|---|
| `steps` | 8 | `MAX_AGENT_STEPS` |
| `tool_calls` | 12 | `MAX_TOOL_CALLS` |
| `llm_calls` | 16 | `MAX_LLM_CALLS` |
| `tokens` | 120,000 | `MAX_AGENT_TOKENS` |
| `seconds` | 45.0 | `MAX_AGENT_SECONDS` |
| `cost` | $0.25 | `MAX_AGENT_COST_USD` |

One shared mutable object per run, passed to the graph, the supervisor, the
ReAct loop and the tool registry — which is what makes
`budget.report()["tool_calls"]` a fact rather than two opinions. `allows()` is
the graceful path: an agent loop that is out of room composes a best-effort
answer and `BUDGET_STOPPED_NOTE` is appended, so the employee is told, rather
than the run raising.

### The aggregates

`AuditStore.stats()`, surfaced at `GET /api/metrics`:

`total`, `by_source`, `by_mode`, `by_model` (answers, tokens, cost per model),
`grounded` and `grounded_rate`, `rewritten`, `average_latency_ms`,
`latency_p50/p95/p99_ms` and `latency_samples`, `prompt_tokens`,
`completion_tokens`, `cost_usd`, `cost_per_answer_usd`, `cached` and
`cache_hit_rate`, `byok`, `escalated` and `escalation_rate`, `tool_calls`,
`agent_steps`, `guardrail_flagged`, `pii_redacted_answers`.

Plus live service counters — `cache`, `rate_limit`, `byok_pool`, `approvals`,
`scratchpad`, `idempotency` — each read through a `safe()` wrapper, because one
store with a locked database must not blank the whole dashboard.

`/api/metrics` is **open**, unlike `/api/audit`, and the line between them is
the point. An audit row is one employee's question and the answer they were
given, which is HR-sensitive by definition. An aggregate is not: "we answered
412 questions, 86% of them grounded, p95 4.1 seconds" says nothing about
anyone. So the rows stay behind an admin key and the totals do not — which is
what makes the dashboard worth building, since the audience for it is somebody
evaluating the deployment and they cannot be issued a key.

### The levers, in the order they actually pay

| Lever | Mechanism | Switch |
|---|---|---|
| **1. Caching** | a hit costs one embedding (or nothing, on the exact layer) instead of the whole pipeline: no retrieval, no routing, no generation. It is the only lever that removes a question's cost entirely | `CACHE_ENABLED`, `CACHE_SIMILARITY`, `CACHE_TTL_SECONDS` |
| **2. Routing** | the same question answered by a model whose list price is a fraction of the next rung's. The catalog's own spread: `gpt-4o` is roughly 17× `gpt-4o-mini` on input | `MODEL_ROUTING_ENABLED`, `FAST_MODEL`, `DEEP_MODEL` |
| **3. The retrieval gate** | on the dense path, the 0.15 raw-cosine `score_threshold` returns **zero chunks** for an out-of-domain question, and `grade_kb_evidence` then returns `weak` *without an LLM call*. So an off-topic question is refused before it costs a grader round trip — and, because no evidence means no generation, before it costs the expensive call too. `RETRIEVAL_MODE=lexical` has no cosine gate, so this lever does not apply there | `Settings.relevance_threshold`; measured at 0.099 out-of-domain against a weakest correct chunk of 0.171 |
| **4. Budgets** | they do not make the common question cheaper; they bound the pathological one. A rewrite loop, a supervisor consulting six specialists, a ReAct agent calling the same tool repeatedly — each is a run that would otherwise cost an unbounded multiple of a normal one | the six `MAX_AGENT_*` settings |

The ordering matters: caching removes cost, routing reduces it, the gate avoids
spending it on questions that should not be answered, and budgets cap the tail.
Only the first two scale with traffic; the last two protect against the single
expensive run.

No percentage saving is quoted for any of them, because none has been measured
against the unrouted, uncached baseline on this deployment. That measurement is
the first item below.

---

## What to measure first

1. **The unrouted, uncached baseline.** Run the labelled question set with
   `MODEL_ROUTING_ENABLED=0 CACHE_ENABLED=0`, record `/api/metrics`, then turn
   each on in turn. Without this, every claim in section 5 is an argument rather
   than a result — and the quality half matters more than the cost half, since
   a cheap model on a hard question fails quietly.
2. **`grounded_rate` over time.** The share of answers resting on retrieved
   evidence rather than ending in an admission or an error. A fall in it is how
   a broken ingest, an exhausted token quota or a drifting threshold shows up
   from outside.
3. **p95 against p50**, with `span_summary.by_kind` beside it. Fix the
   untimed router and grader spans first, or LLM self-time will keep reading low.
4. **Cache hit rate *and* the near-miss distribution.** `cache.miss` carries the
   nearest score; a cluster just under 0.95 is the signal to look at the
   threshold, and a cluster at 0.91 is the signal to leave it alone.
5. **Actual provider spend against `cost_usd`.** One reconciliation tells you
   the direction and rough size of the projection's error on your traffic. Do it
   once a month, not once.
6. **`escalation_rate` and the pending approval queue depth.** An escalation
   nobody picks up is an employee waiting, and it is invisible in every other
   metric here.
7. **Rate-limiter refusals against instance count.** `refusal_rate` with
   `scope: "process"` is per-instance; if it is near zero while users report
   limits, or near zero while the token quota keeps emptying, the limit is not
   where you think it is.
