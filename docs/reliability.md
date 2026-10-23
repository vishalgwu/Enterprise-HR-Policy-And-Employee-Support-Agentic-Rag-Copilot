Part of the [HR Copilot](../README.md) documentation set.

# Reliability: budgets, degradation and the never-raise contract

Source: [`app/agent/nodes.py`](../app/agent/nodes.py) (`guard`),
[`app/agent/budgets.py`](../app/agent/budgets.py),
[`app/agent/decisions.py`](../app/agent/decisions.py),
[`app/agent/tools.py`](../app/agent/tools.py),
[`app/services/pipeline.py`](../app/services/pipeline.py),
[`app/services/extractive.py`](../app/services/extractive.py),
[`app/services/idempotency.py`](../app/services/idempotency.py),
[`app/rag/hybrid.py`](../app/rag/hybrid.py) (`build_retriever`),
[`app/services/state_db.py`](../app/services/state_db.py),
[`app/services/audit.py`](../app/services/audit.py).
Per-tool flags are declared in
[`app/services/toolbox.py`](../app/services/toolbox.py).
Tests: [`tests/test_graph.py`](../tests/test_graph.py),
[`tests/test_pipeline.py`](../tests/test_pipeline.py),
[`tests/test_tools.py`](../tests/test_tools.py),
[`tests/test_decisions.py`](../tests/test_decisions.py),
[`tests/test_retrieval_hybrid.py`](../tests/test_retrieval_hybrid.py).

---

## 1. The rule

**Every node degrades; none may raise.**

A node that throws aborts the whole run. One flaky Pinecone call would therefore
cost an answer that web search could have given, and one flaky Tavily call would
cost an answer the private KB had already found. So `guard()` in `build_nodes`
wraps each network call and turns a failure into the quietest available state:

| Call | Failure becomes |
|---|---|
| Retrieval | zero chunks — which routes to the web fallback |
| Web search | empty evidence |
| Rewrite | the original query, with `retry_count` still incremented |
| Generation | `GENERATION_FAILED` and `source_used="error"`, never an empty answer |
| Contextualise | the original wording — a follow-up loses its context, not its answer |

The rewrite case is worth a second look: the counter increments even though the
rewrite failed, because otherwise a broken LLM spins the loop. The retry
allowance is spent whether or not the retry was any good.

`guard()` also appends to the trace, via `_degrade`. An outage and a genuine
absence of evidence produce the same answer, and **only the trace distinguishes
them** — a run that answered from the web because Pinecone was down looks
identical to one that answered from the web because the KB evidence was weak.
The exception is caught *inside* the span, so no provider error text can reach
`Span.error` and from there the HTTP response or the audit row; what is recorded
is the exception's type name and nothing else.

### "Inside the guard" is the whole rule, and it is easy to half-observe

`_generate` used to call `.content` inside `guard()` and `.strip()` outside it:

```python
# the bug
answer = guard(tag, lambda: (prompt | llm).invoke(variables).content, "", steps)
if not answer.strip():          # <- outside the guard
```

`.content` is a plain string on Groq and OpenAI, so this worked. But the message
interface allows a **list of content blocks**, and `.strip()` on a list raises
`AttributeError` — which escaped the node and aborted the run. A provider shape
change would have cost *every* answer rather than degrading one, and no test
would have caught it because the fakes returned strings.

Both generation and the rewrite now flatten through `message_text()` in
[`app/rag/clients.py`](../app/rag/clients.py), inside the guard.
`test_a_reply_of_content_blocks_does_not_abort_the_run` fails against the old
code with exactly that `AttributeError`.

The lesson is procedural: when adding a node, check where the **last attribute
access** happens, not just where the network call does. Provider response shapes
are adapted in `app/rag/clients.py` and nowhere else — `message_text` for chat
replies, `web_results_to_text` and `web_result_urls` for Tavily — and a node that
reaches into a provider's response shape directly is a bug waiting for the next
SDK release.

---

## 2. Loop budgets

A graph with a bounded rewrite loop needs one counter. An agent that chooses its
own tools needs a real budget, because the failure mode changes shape: a graph
that loops forever is a bug you can see in the wiring, while an agent that loops
forever is a model deciding, each time, that one more search would help. Nothing
in the wiring is wrong. The only thing that stops it is a ceiling.

Six ceilings rather than one, because they fail differently and a run can hit any
of them first:

| Kind | Default | Bounds |
|---|---|---|
| `steps` | 8 | how many times the agent gets to decide — the loop itself |
| `tool_calls` | 12 | how much work it does — fan-out and retries |
| `llm_calls` | 16 | model round trips — a self-chat |
| `tokens` | 120,000 | context burn — prompt growth as the transcript accumulates |
| `seconds` | 45.0 | wall clock — a provider that is merely slow |
| `cost` | $0.25 | the projection an operator is asked about |

One ceiling cannot do this. A step limit does not stop one step that burns 80k
tokens; a token limit does not stop a provider taking four minutes to answer; a
wall-clock limit does not stop a cheap, fast loop running 200 times. They are
also read in a deliberate order by `exceeded()` — `seconds` first, because a run
that has already taken 45 seconds should say *that* rather than report a step
count that happens to also be full, then cheapest signal to most expensive.

### `allows()` vs `charge()`

**A budget is a stopping condition, not an error.**

```python
if not budget.allows("steps"):
    stopped = True
    trace += self._note("ReAct", budget.stop_reason or "out of budget")
    break
```

The agent asks `allows()` before it spends and composes a best-effort answer when
the answer is no. It does not catch `BudgetExceeded` on the main path. An employee
asking about parental leave should get the partial answer the agent had assembled,
labelled as partial — not a 500 saying `max_steps`.
`test_a_one_step_budget_forces_a_composed_answer_and_says_it_is_partial` pins
that.

`charge()` raising exists for the inner call — a tool, a worker — that genuinely
has nowhere to degrade to and must unwind to a caller that does.
`charge_llm()`, `charge_tool()` and `charge_step()` never raise, deliberately:
the call has already happened by the time its token count is known, so refusing
to record it would make the budget lie about a cost that was genuinely incurred.
The *next* `allows()` is what stops the run.

### One shared mutable object, and why it is thread-safe

The supervisor, every worker, every tool and the pipeline's accounting stage
charge the **same** `Budget`. That is what makes "the run cost 12 calls" a fact
rather than six separate opinions — and `budget.report()["tool_calls"]` the same
number the registry counted.

Shared mutation across `call_parallel`'s threads is why every read-modify-write
sits behind a `threading.Lock`. A budget that undercounts under concurrency is a
budget that does not bound anything: two parallel tools doing
`self.tool_calls += 1` without a lock can land on the same value and the ceiling
silently becomes a suggestion.

`started_at` is set by `start()` and never in `__init__`, and `start()` is
idempotent. A budget constructed at import, or held on a settings object, would
start its clock then and every later run would look as if it had already spent an
hour; and a second `start()` resetting the deadline would make the wall-clock
ceiling unbounded in exactly the run that needed it.

`DEFAULT_BUDGET` exists as a module-level shape for a `--help` string or a docs
example. **Do not charge against it** — it is one object, so charges would
accumulate across every run in the process.

### Saying so in the answer

`stop_reason` is set as a side effect of `allows()` returning False, so a run
that stopped early always carries its reason into the trace and the audit row.
Stage 9 of the pipeline then appends `BUDGET_STOPPED_NOTE`:

> _This answer is partial: I reached my limit for this question before checking
> everything. Ask again for the part I missed and I'll start fresh._

**An answer that ran out of room and does not say so is indistinguishable from a
complete one that happened to be short.** That is the whole argument for the
note: an employee cannot tell "that is all the policy says" from "I stopped
looking", and the two have opposite next actions. `report()` carries `exhausted`
and `stop_reason` for the same reason — neither should be inferable only from a
short answer. A partial result is also not cached (stage 10 skips
`result.partial`), so the truncation is not served to the next person who asks.

---

## 3. Graceful degradation, as a ladder

Each rung is a worse answer and still an answer. What each one costs the user:

| Rung | Trigger | What the user gets instead |
|---|---|---|
| Pinecone unavailable | `_dense_retriever` raises | BM25 over the same documents on disk — no paraphrase matching, exact terms only |
| One hybrid side fails | `HybridRetriever._side` raises | the other side's ranking, unfused — a worse order, same documents |
| No evidence in the KB | zero chunks, or `kb_grade="weak"` | a web-sourced answer, cited to a public page rather than company policy |
| Web evidence weak | `grade_web_evidence` returns `weak` | a rewritten query, looped back to **KB** retrieval first (`rewrite_query -> retrieve_kb`, bounded by `max_retries`, default 1) |
| Still weak | retries spent | `INSUFFICIENT_ANSWER` — it says it does not know, which is the correct answer |
| An orchestration raises | any exception out of graph / ReAct / supervisor | an extractive answer stitched from the retrieved policy text |
| No chat model at all | `has_model` is False | the extractive path *is* the product |
| Tavily absent | no key, `get_web_search` raises | `_NoWebSearch` returns the empty shape, so the graph takes the same path as "the web had nothing" |
| Audit write fails | disk full, locked db, read-only mount | the answer stands; `audit_id` comes back null and the reason is logged at ERROR |

### Pinecone → BM25

`build_retriever` is the one retriever factory the rest of the application calls,
and it degrades rather than propagating:

```python
built_dense = dense if dense is not None else _dense_retriever(settings, embedding, k)
if built_dense is None:
    log.warning("[Retrieval] no dense retriever; falling back to lexical retrieval")
    return _lexical_retriever(settings, k) or LexicalRetriever(BM25Index.from_documents([]), k=k)
```

A deployment that loses Pinecone — an expired key, a deleted index, a region
outage — should give worse answers, not no answers, and BM25 over the same eleven
documents needs nothing from the network. The alternative is an application that
boots, reports healthy, and fails on the first question. Even an unreadable
knowledge base becomes an *empty* index rather than an exception, because zero
chunks is a state the graph already knows what to do with.

`HybridRetriever` holds the same line one level down: one side failing is a worse
ranking, both sides failing is zero chunks, and `invoke` returns `[]` rather than
raising, always.
`test_the_hybrid_retriever_answers_from_the_sparse_side_when_the_dense_one_fails`
and its mirror pin both.

### The extractive floor

`AnswerPipeline._run` catches anything that escapes an orchestration and tries
extraction before giving up:

```python
except Exception:
    log.exception("The %s orchestration raised; falling back", mode)
    tracer.degraded("orchestration", reason=f"{mode} raised", kind="agent")
    fallback = self._run_extractive(question, asked, tracer)
    if fallback.answer:
        return fallback
    return AnswerResult(question=question, answer=GENERATION_FAILED, source_used="error", ...)
```

Retrieval may still work when generation does not: at that point the evidence is
sitting there, and handing back the policy text beats handing back an apology.
`answer_from_documents` ranks the sentences of the retrieved chunks by how much
of the question's content words they cover, keeps the best few **in document
order**, and labels the result with `EXTRACTIVE_PREFIX`.

Document order rather than score order matters more than it sounds: policy text
is sequential — a condition and its exception, a rate and the period it applies
to — and reordering by relevance routinely puts the exception before the rule it
is an exception to, which reads as a contradiction.

**The extractive path is structurally incapable of inventing a policy number.**
No model composes it; every sentence is a substring of a retrieved chunk. That
makes the worst-case answer the most trustworthy one in the system — it reads
worse than a generated answer and it is harder to be wrong about. It is also what
makes a no-key deployment a product rather than an error page:
`test_no_model_credentials_answers_from_the_documents_alone` and
`test_an_orchestration_that_raises_falls_back_to_the_documents` pin the two
entries to it.

One honest limit. A generation failure *inside* a guarded node does not reach
this path: `_generate` returns `GENERATION_FAILED` without raising, so the
pipeline sees a completed run with `source_used="error"` and does not re-route it
to extraction. The extractive fallback fires when an orchestration **raises**, or
when there is no model configured at all. Wiring the in-node case to it is a
one-line change in the pipeline and is not currently made.

### `_NoWebSearch`

```python
def invoke(self, payload=None, *args, **kwargs) -> dict[str, Any]:
    query = payload.get("query") if isinstance(payload, Mapping) else ""
    return {"query": query, "answer": "", "results": []}
```

Returning the real client's empty shape rather than `None` means the keyless
deployment takes the identical code path to the keyed one on a query that found
nothing — the trace reads `Tavily: 0 characters` and the graph proceeds. A `None`
would have to be special-cased in the node, which is how a no-key deployment
starts behaving differently from the keyed one in ways nobody tested.

---

## 4. Structured-output failures

Groq returns `400 tool_use_failed` — "Tool choice is required, but model did not
call a tool" — when the model answers in plain text instead of invoking the bound
tool. Observed live on the router, whose call is on the path of *every* question.

The error body carries `error.failed_generation`, holding what the model actually
said, and it is usually the right value: an observed failure carried exactly
`"direct"`. So the recovery is three steps, in `app/agent/decisions.py`:

```python
def route_question(question: str, router: Any = None) -> Route:
    try:
        return (router or get_router()).invoke({"question": question}).route
    except Exception as exc:
        return _recover(exc, ROUTE_VALUES, "Router", "kb")
```

1. `failed_generation(exc)` digs the raw text out of the error body.
2. `coerce(text, allowed)` matches it against the enum — handling `direct`,
   `"direct"` and `{"route": "direct"}`, and falling back to containment only
   when **exactly one** candidate appears. With two it would be a guess, and
   guessing is what the safe default is for.
3. Failing that, a safe default.

**The defaults are asymmetric on purpose:**

| Decision | Default | Why that direction |
|---|---|---|
| Router | `kb` | Seek evidence rather than answer from memory. An unclassifiable message costs a retrieval; `direct` would make the assistant answer an HR question with no evidence behind it. |
| Grader | `weak` | Distrust rather than pass ungraded evidence. A `weak` grade sends the run to its fallback path; `good` would let an unchecked chunk become a cited answer. |

Graph nodes must call `route_question` / `grade_evidence`, never
`router.invoke` / `grader.invoke` directly — the recovery lives in the helper and
a direct call bypasses it.

### Why `method="json_mode"` is banned

It is unconstrained JSON, and this model fills the enum field with prose.
Observed, from a routing call:

```json
{"route": "Employees accrue 1.5 days of PTO per month..."}
```

That is precisely the failure structured output exists to prevent. The default
(function calling) enforces the enum and is also the fastest of the three working
methods; `json_schema` works too.

One more thing worth knowing before chasing a prompt bug: **`temperature=0` is
not determinism on Groq.** Repeated grading of a genuinely borderline case ("the
KB covers employees, the question asks about contractors") returned `weak` 8/8 in
isolation and flipped to `good` inside a longer suite run. Treat a single-run
grader result as approximate.

---

## 5. Rate limits are not knowledge gaps

Every safe default in this system is "weak". So an exhausted quota looks exactly
like a knowledge gap: the router returns `kb`, every grade returns `weak`, and
every run ends `insufficient_evidence`. Nothing in the answer says "the provider
refused me".

`is_rate_limit(exc)` in `app/rag/clients.py` distinguishes it — class name
`RateLimitError`, HTTP 429, or the phrase in the message — and both `_degrade` in
the nodes and `_recover` in the decisions log that case at **ERROR** rather than
WARNING:

```python
log.error(
    "[%s] RATE LIMITED by the provider - results below are "
    "degraded, not a knowledge gap (%s)", tag, str(exc)[:160],
)
```

For a guarded node call the trace line says the same thing — `_degrade` appends
`"rate limited by the provider; results are degraded"` — so it reaches the audit
row and the console rather than only the server log. The router and grader
recovery in `_recover` logs but does **not** write a trace line, so a
quota-degraded routing decision is visible in the logs only. That is the gap to
know about when reading an audit row that looks like a knowledge gap.

The observed exhaustion, from Groq's free tier:

```
429 ... tokens per day (TPD): Limit 200000, Used 199771
```

Tokens per **day**, not per minute — so it does not clear while you stare at it.
**If a whole run suddenly grades everything weak, check the quota before touching
a prompt.** Limits are per model, so `GROQ_MODEL=openai/gpt-oss-120b` is a
working escape hatch. `test_router_survives_a_rate_limit` and
`test_grader_survives_a_rate_limit` pin the behaviour offline.

---

## 6. Retries and timeouts

Both live in `ToolRegistry`, not in handlers, so a tool that forgets cannot hang
a request.

**Retries only for a tool declared idempotent.**

```python
allowed = 1 + (self._retries(spec) if spec.idempotent else 0)
```

Retrying a call that may already have taken effect is how one request becomes
two. `kb_search` and `web_search` are idempotent and retried; `raise_ticket` and
`escalate_to_human` declare `idempotent=False` and get exactly one attempt.
`test_a_non_idempotent_tool_is_not_retried` pins it.

**`min(spec, settings)` for both ceilings.**

```python
def _timeout(self, spec):
    ceiling = float(getattr(self._settings, "tool_timeout_seconds", 20.0))
    return min(float(spec.timeout_seconds), ceiling)
```

The deployment's value is a *ceiling*, not a default, so a tool declaring a
generous timeout of its own still cannot hang a request on a deployment that will
not wait that long. `_retries` is the same shape against `tool_retries`
(default 2, so three attempts at most).

**Exponential backoff with jitter.**

```python
delay = base * (2**attempt)          # base = tool_backoff_seconds, 0.4s
return delay + random.random() * delay * 0.25
```

The jitter is not decoration. Several calls that failed *together* — which is
exactly what a rate limit or a brief outage produces — would otherwise retry at
the same instant and reproduce the burst that caused the failure, so the second
attempt fails for the same reason as the first. Up to 25% of the delay is enough
to spread a handful of calls.

The budget is charged **once per `call`**, not once per attempt: a retry is the
registry's own decision about an unreliable provider, and billing the agent for
it would make a flaky Tavily look like an expensive agent and shorten the run
that needed the budget most.
`test_the_budget_is_charged_once_however_many_attempts_it_took` pins it.

### The honest caveat about the timeout

```python
executor.submit(lambda: spec.handler(**payload)).result(timeout=timeout)
```

**This bounds the caller's wait, not the work.** Python cannot kill a thread and
there is no safe general way to interrupt one, so when the timeout fires the
handler may still be mid-request. Two consequences, both deliberate:

- `tool_parallelism` is small (4 by default), because a timed-out call keeps its
  thread until it finishes; `shutdown(wait=False)` avoids re-joining the very
  thread the timeout existed to stop waiting for.
- A handler is still expected to pass its own timeout to its HTTP client. This is
  a backstop, not the primary control.

`test_a_tool_over_its_timeout_fails_without_blocking_the_caller` pins the
caller's side, which is the part this can actually promise.

---

## 7. Idempotency

A retried tool call is the whole point of retries, and it is also how an HR
ticket gets raised twice. The registry cannot tell a request that failed from a
request whose *reply* failed, so retrying is the only way to survive a flaky
provider and also the only way to duplicate a side effect. The ledger is what
makes the second kind safe.

A `side_effecting` tool claims a fingerprint of its **validated arguments**
before the handler runs. `fingerprint()` is sorted-key JSON through sha256,
truncated to 32 hex characters, and nothing that varies per process goes into it
— no `hash()` (PYTHONHASHSEED randomises it for strings), no `id()`, no
timestamp — because the claim is written by one request and matched by another,
which after a container restart is a different interpreter.
`test_a_fingerprint_does_not_depend_on_the_order_the_keys_were_written` pins it.

### The atomic claim, and the SELECT-then-INSERT race

```sql
BEGIN IMMEDIATE;
INSERT INTO idempotency (key, created_at, expires_at, state, action)
VALUES (?,?,?,?,?) ON CONFLICT(key) DO NOTHING;
SELECT state, result FROM idempotency WHERE key = ?;
COMMIT;
```

`rowcount == 1` on the insert is what says whether **we** created the row. The
obvious implementation — SELECT, and INSERT if nothing came back — is the bug
this exists to prevent, reintroduced by the check that was supposed to prevent
it: two threads both see no row, both insert, both perform the side effect.
`BEGIN IMMEDIATE` rather than sqlite3's implicit DEFERRED, because DEFERRED takes
its write lock late and can lose the upgrade to a concurrent writer.
`test_only_one_of_two_concurrent_claims_wins` pins it.

A `Claim` has three states, not two:

| State | Meaning | Caller does |
|---|---|---|
| `won` | the row is ours | run the work, then `complete()` |
| `replayed` | somebody already finished it | return `.result`; do not repeat |
| `in_flight` | somebody is doing it right now | **refuse** |

`in_flight` is neither a yes nor a safe no: replaying a result we do not have
would invent one, and running anyway is the duplicate. The caller refuses and the
employee is told to try again.
`test_a_claim_already_in_flight_refuses_rather_than_racing_it` pins it.

### The stall window

An `in_flight` row older than `stall_seconds` (60) is treated as abandoned and
deleted, with a warning. A process killed mid-call otherwise leaves a key nobody
will ever complete or release, and because the row survives the full TTL (86,400
seconds — a day) that turns one crash into a day of refused retries. Sixty
seconds is comfortably past any tool timeout the registry allows, so a live call
is never stolen from.
`test_an_abandoned_claim_is_re_won_rather_than_blocking_for_the_whole_ttl` pins
it.

### `release` on a terminal failure, deliberately not on a timeout

```python
if claim is not None and not timed_out:
    self._release(claim)
elif claim is not None:
    log.warning(
        "[tool.%s] timed out with a claim held; leaving it in flight so "
        "a retry cannot duplicate the side effect", spec.name,
    )
```

A terminal failure means the side effect did not happen, so the claim is deleted
(not marked failed — a failed row would have to be distinguished from a live one
by every later reader, and the only thing anyone wants to know is whether the side
effect happened). A **timeout** means we stopped waiting; the worker thread is
still running and may yet perform the side effect. Releasing there would let a
retry duplicate it, so the claim is left in flight and the stall window decides
when it is safe to re-win.

### In-memory mode is not a distributed lock

With no store and no path the ledger is a dict behind a lock: correct for one
process, worth nothing across two. Vercel runs each function instance separately,
so two concurrent requests can land in two processes that share no memory and
each win the same claim. Idempotency across instances needs the SQLite path on a
shared volume, or an external store — Redis, or a unique constraint in the
provider's own database. `IdempotencyStore.stats()` reports `backend` for exactly
this reason: `"memory"` is the single fact an operator needs before trusting the
ledger behind more than one instance.

`call_parallel` also runs a side-effecting tool on the calling thread, before the
fan-out. Two side effects racing on the same key is the case the ledger cannot
fix — both claims are decided before either completes, so one wins and the other
refuses the employee's request for no reason the employee can see.

---

## 8. The never-raise stores

| Call | On failure | What the caller sees |
|---|---|---|
| `AuditStore.record` | returns `None`, logs at ERROR | the answer, with `audit_id` null |
| `StateStore.execute` | returns `None`, logs at ERROR | a feature degraded: no note, no fact |
| `StateStore.query` | returns `[]`, logs at ERROR | an empty list where rows were expected |
| `ApprovalStore.request` | returns `None`, logs at ERROR | **the caller must refuse** |

A full disk, a locked database or a read-only mount degrades a feature. That is
much cheaper than turning a working answer into a 500, so the write is swallowed
and the answer stands. A scratchpad that cannot be read is an agent with no notes,
which is how it starts every conversation anyway.

**The failure is logged at ERROR, and that is half the design.** The symptom of a
swallowed audit write is an audit log that silently stops growing — the one
failure this store exists to prevent, and one that is invisible in the product.
Logging it at ERROR makes an unwritten log visible *in the logs* rather than only
in its own absence. (The same reasoning drives `ADDED_COLUMNS` and `_migrate` in
both stores: `CREATE TABLE IF NOT EXISTS` does nothing to an existing table, so a
database written by an earlier version keeps its old shape and every INSERT naming
a new column fails — and `record()` swallows it.)

Both stores open a connection per call. FastAPI runs `def` endpoints in a
threadpool, so one long-lived `sqlite3` connection would need
`check_same_thread=False` plus its own locking; opening costs microseconds
against an LLM round trip that costs seconds. Neither creates its schema in
`__init__`, because `app.main` builds an application at import and a constructor
that touched the filesystem would make importing the module write a database
file. `test_building_the_app_touches_neither_network_nor_disk` pins that.

### The one place the reasoning inverts

`ApprovalStore.request` returning `None` means **the approval was not recorded**.
There is no queue entry for anyone to decide and no record that the agent ever
asked. The tempting reading of `None` is "carry on without it", and carrying on is
precisely the thing the approval existed to prevent — so the tool refuses. See
[safety.md](safety.md) §6.

The two state files are also separate on purpose. `audit.db` is the compliance
record — every question and the answer given, admin-gated, on a volume, the one
file a deployment must not lose. `state.db` is working state that churns on every
request. One file would mean the backup policy for the compliance record had to
cover a churning table, and that the audit log's size told you nothing about how
many questions were asked.

---

## 9. Failure modes

| What fails | What the user sees | What the trace says | What an operator should check |
|---|---|---|---|
| Pinecone unreachable | answers that miss paraphrased questions | `retrieve.dense` degraded, or "no dense retriever; falling back to lexical" in the log | `/health` for `vector_store`; the Pinecone key and index name |
| Both retrieval sides fail | a web-sourced or insufficient answer | `KB Retriever: 0 chunk(s) for ...` then the web path | the KB directory is readable; the index exists |
| Tavily key absent | KB answers only; weak-KB questions end `insufficient_evidence` | `Tavily: 0 characters` | `/health` for `web_search` |
| Groq daily quota exhausted | every question ends "I don't have enough information" | `RATE LIMITED` at ERROR on the router and graders | the Groq dashboard, **before** editing a prompt |
| Structured output fails | a correct answer, usually | nothing in the trace; `Router`/`Grader` recovery at DEBUG or WARNING in the log | nothing, unless it is constant — then the model id |
| No chat model configured | an extractive answer led by `EXTRACTIVE_PREFIX` | `Extractive: N sentence(s), coverage X` | intended on a keyless demo; otherwise `/health` for `server_llm` |
| An orchestration raises | an extractive answer, or `GENERATION_FAILED` | `orchestration` degraded, reason `"<mode> raised"` | the ERROR traceback — this one is a bug, not a degradation |
| Budget exhausted | a partial answer ending in the partial-answer note | `stop_reason` in the trace and `budget.exhausted` in the response | whether the ceiling is too tight or the question too broad |
| Tool timeout | the agent answers without that tool's result | `tool.<name>` degraded, "did not answer within Ns" | the handler's own HTTP timeout; `TOOL_TIMEOUT_SECONDS` |
| Approval queue unwritable | "needs human approval and the request could not be recorded, so it was not run" | the refusal, as the tool result | disk space and the `state.db` mount — nothing was queued |
| Audit write fails | a normal answer; `audit_id` is null | nothing | ERROR lines from the `audit` logger; the volume mount |
| Idempotency claim in flight | "already running with these arguments" | the refusal | whether a previous call crashed; it clears after 60s |
| Multi-instance deployment | duplicated side effects, a weaker rate limit | nothing — each instance looks healthy | `/api/metrics` for `backend: "memory"`; move the ledger to a shared volume |
