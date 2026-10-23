Part of the [HR Copilot](../README.md) documentation set.

# Agent patterns

A field guide to the agentic machinery in this repository. Each section says
what the pattern is, why it is here, where it lives, and what it cost —
because the cost is the part that shows whether the pattern was chosen or
copied.

The fixed graph (`app/agent/graph.py`, eleven nodes, documented in
[agent-graph.md](agent-graph.md)) is still the default path. Everything below is
what you need once an agent chooses its own sequence, plus the pipeline that
wraps whichever orchestration runs. `app/services/pipeline.py` is that wrapper:
eleven stages in a fixed order — tracer and budget, input guard, semantic cache,
model routing, orchestration choice, run, generation fallback, output guard,
accounting, cache store, assemble — each of which can be switched off, because a
feature nobody can turn off is a feature nobody can measure.

---

## The minimal ReAct loop

**What it is.** Think, act, observe, repeat. One tool per turn, chosen by the
model from a rendered catalogue, with the previous observations on the
transcript. `app/agent/react.py`.

**Why it is here.** The graph decides its sequence once, at build time, and
every question walks the same eleven nodes. That is right for the questions it
was built for and wrong for one that needs two lookups and a sum: there is no
edge for "retrieve, work out the arithmetic, retrieve again". The loop has no
fixed sequence at all.

**What it cost.** Every guarantee the wiring used to provide, which is why the
module is mostly guarantees.

### The turn

```python
for index in range(self._max_steps):
    if not self._budget.allows("steps"):
        stopped = True
        break
    self._budget.charge_step()
    decision = self._decide(question, catalogue, transcript, index)
```

`ReActDecision` is a structured output (`thought`, `action`, `action_input`,
`final_answer`) bound once at construction, not per turn. The recovery is the
same three stages every structured call in this project uses: try, recover the
model's raw text from a Groq `tool_use_failed` body, then fall back to a safe
default.

### `action` wins when the model fills both

```python
# Both set: prefer the action.
if action:
    ...
    continue
if final:
    answer = final
    break
```

An agent that chose to look something up has told you its answer is not ready,
and the answer it wrote alongside was written **before the evidence arrived**.
Preferring `final_answer` would systematically serve the pre-evidence answer on
exactly the turns where the model knew it needed evidence.

### The safe default never answers from memory

`_safe_default` is what runs when the decision cannot be read at all. On step 0
it reaches for a search tool, resolved against the registry rather than assumed;
after that it composes from the observations already gathered. Both paths end in
evidence somebody retrieved. Filling the gap from training data is the single
worst output an HR copilot can produce, and it is indistinguishable from a good
one to the person reading it.

`_parse_decision` is tolerant about shape (a JSON object anywhere in the text,
several spellings per field) and strict about kind: **JSON only**. A
`failed_generation` body usually holds the model's prose, and treating that prose
as `final_answer` would turn every structured-output failure into an answer
written from memory.

### Repeated identical calls are intercepted, and the model is told

```python
key = (action, _canonical_args(step.action_input))
if key in seen:
    step.observation = (
        "That exact call has already run. It returned: "
        f"{seen[key]}\nChange the arguments or use a different tool."
    )
    step.ok = False
    continue
```

The budget stops a repeating model eventually — but it stops it *by spending the
budget*, and the run ends with a truncated answer instead of a different
approach. Telling the model beats silently re-serving the cached observation for
the same reason every `ToolResult.observation()` branch says what to do next: a
model told "that already ran" changes tack, a model handed the same text again
has learned nothing. `_canonical_args` sorts keys, so reordered arguments are
recognised as the same call.

### Provenance from what ran, not from what was claimed

`_source_used` derives the answer's source from the tools that **succeeded**:
`KB_TOOLS` wins over `WEB_TOOLS`, an attempt with nothing returned becomes
`insufficient_evidence` rather than `direct`, and `escalated` and `error` are
their own values. A model asked to cite its sources will cheerfully write
"according to the company handbook" over an answer composed from a web article.

---

## The tool registry

**What it is.** A declaration per tool of everything the wiring used to
guarantee: timeout, retries, whether a retry is safe at all, who may ask, and
whether a person must say yes first. `app/agent/tools.py`.

**Why it is here.** A graph's wiring *is* its policy. An agent that chooses its
own tools has no wiring, so the properties have to be declared somewhere a
handler cannot forget them. That somewhere is `ToolSpec`, and the registry
enforces it.

**What it cost.** One more place to look when a tool misbehaves — and a hard
layering rule: this module is **agent-layer and may not import
`app/services/*`**. Every service arrives duck-typed (`idempotency`,
`approvals`, `authorize`, `budget`, `tracer`), used through a small documented
interface rather than a type.

### `ToolRegistry.call`, in its real order

`call` opens a span, calls `_dispatch`, and fills in the duration — so the span
in step 11 brackets everything below it, and is recorded whichever way the call
goes. `_dispatch` and `_run` do the rest, in this order:

| # | Step | Why it is *there* |
|---|---|---|
| 1 | **Unknown tool** → answer with the names that *are* available | A model that hallucinated a tool name recovers from being told the real ones. It cannot recover from a stack trace. First, because everything below needs a spec. |
| 2 | **Budget** — refuse with the budget's own `stop_reason` if the tool-call ceiling is spent | Nothing is called and nothing is charged. Before authorization because a spent budget is cheaper to check than a policy lookup, and the answer is the same for every caller. |
| 3 | **Authorization** — `authorize(principal, spec.requires)` | The refusal names the action and says nothing about what the tool would have done, because the refusal itself is returned to the caller. An authoriser that *raises* denies. |
| 4 | **Argument validation** against `spec.parameters` | Before approval, so a malformed request is never queued for a human to look at. |
| 5 | **Approval** — record the request, return without running | A store that answers `None` has not recorded it, so that refuses too. Fail closed: running anyway would be running unapproved. |
| 6 | **Idempotency** — a side-effecting tool claims a fingerprint of its *validated* arguments | After approval, so an unapproved call never reserves a fingerprint it will not use. A replay returns the first call's result; a claim already in flight refuses rather than racing it. |
| 7 | **Run**, with a timeout, retrying only an idempotent tool | `allowed = 1 + (retries if spec.idempotent else 0)`. A non-idempotent tool gets exactly one attempt. |
| 8 | **Back off** exponentially with jitter between attempts | See *Retries and timeouts* below. |
| 9 | **Charge the budget** once per `call`, not once per attempt | A retry is the registry's own decision about an unreliable provider. Billing the agent for it would make a flaky Tavily look like an expensive agent and shorten the run that needed the budget most. |
| 10 | **Complete** the claim on success; **release** it on a terminal failure | Except on a timeout — see below. |
| 11 | **Record a span either way** | A failure that is neither a denial nor an approval also fires `tracer.degraded(...)`, with the error text the registry wrote rather than a provider's. |

The order is the design. Validation before approval; approval before the ledger;
the charge after both, so a refusal is free.

### Argument validation gets its own paragraph

```python
def _argument_error(name: str, exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(piece) for piece in error.get("loc", ())) or "(root)"
        parts.append(f"{location} ({error.get('type', 'invalid')})")
    return f"{name} was called with invalid arguments: {'; '.join(parts) or 'unknown'}"
```

`str(exc)` would be the obvious thing to put in that message, and it is the
wrong thing: pydantic prints `input_value=...`. The input here is whatever a
language model put in `action_input` on behalf of an employee who may have
pasted a key, an id or a salary into the chat box, and this message is rendered
into the next prompt *and* returned on the HTTP response. So the message is
rebuilt from `exc.errors()` — field locations and error *types*, both of which
are schema facts rather than data.

The same failure, with the same cause, is on record elsewhere in this codebase:
a model-level pydantic validator on `Settings` once rendered every API key into
an error message beginning `input_value={'GROQ_API': 'gsk_...`. That is why
`CLAUDE.md` forbids raising from a `model_validator` on that class, and why this
function exists rather than a format string.

More generally: **a tool's error text is ours, never a provider's.** A handler
signals an expected failure by raising `ToolError` with words it wrote; anything
else is logged with its traceback and reported as `f"{spec.name} failed with
{type(exc).__name__}"`.

### One place a model's output is executed

`safe_eval` walks the AST and refuses every node type that is not a number or an
arithmetic operator — rather than `eval` with a restricted globals dict, which
has been escaped repeatedly through dunder traversal
(`().__class__.__bases__[0].__subclasses__()` reaches the whole class hierarchy
from a literal with no names in scope). Refusing by node type means a new Python
syntax feature arrives refused rather than arriving allowed. `MAX_EXPONENT` and
`MAX_RESULT_DIGITS` bound the one denial of service available in pure
arithmetic.

It exists because an HR question contains arithmetic — "I have 12 days left and
take 5, what is my carry-over?" — and a language model doing arithmetic in prose
is the classic wrong-number failure this product must not have.

---

## Error handling: every node degrades, none may raise

**What it is.** `guard()` inside `build_nodes` in `app/agent/nodes.py`, wrapping
every network call.

**Why it is here.** A node that throws aborts the whole run, so one flaky
Pinecone or Tavily call costs an answer the *other* source could have given.

```python
with tracer.span(tag, kind=kind) as span:
    try:
        return action()
    except Exception as exc:
        span.status = "degraded"
        span.set(failure=type(exc).__name__)
        return _degrade(tag, exc, fallback, steps)
```

Each fallback is chosen rather than generic: retrieval failure becomes zero
chunks (which routes to the web fallback, the same path a genuinely empty
retrieval takes), search failure becomes empty evidence, a rewrite failure keeps
the original query **while still incrementing `retry_count`** so a broken LLM
cannot spin the loop, and a generation failure returns `GENERATION_FAILED` with
`source_used="error"` rather than an empty answer.

`_degrade` writes the degradation into the trace as well as the log. An outage
and a genuine absence of evidence produce the *same answer*, and only the trace
distinguishes them. A rate limit logs at ERROR, because every safe default in
this graph is "weak" — so an exhausted quota looks exactly like a knowledge gap
unless it is named.

### "Inside the guard" is the whole rule, and it is easy to half-observe

`_generate` used to call `.content` inside `guard()` and `.strip()` outside it.
`.content` is a plain string on Groq and OpenAI — but the message interface
allows a **list of content blocks**, and `.strip()` on a list raises
`AttributeError`, which escaped the node and aborted the run. A provider shape
change would have cost every answer rather than degrading one.

Both generation and the rewrite now flatten through `message_text()` in
`app/rag/clients.py`, inside the guard.
`test_a_reply_of_content_blocks_does_not_abort_the_run` fails against the old
code with exactly that `AttributeError`.

**What it cost.** A rule you have to apply by reading, not by pattern-matching:
when adding a node, check where the *last attribute access* happens, not just
where the network call does.

---

## Memory

**What it is.** Two layers that are deliberately different things. The
short-term window is the conversation (`app/agent/memory.py`, agent-layer). The
long-term store is a handful of facts about one person
(`app/services/memory_store.py`, service-layer, duck-typed into the agent
through anything with `.render(subject)` and `.recall(subject)`).

### The short-term window has a character budget, not only a turn count

```python
turns: list[dict[str, str]] = field(default_factory=list)
max_turns: int = 8
max_chars: int = 6000
```

Eight short turns and eight 3000-character turns cost very different amounts,
and the whole history is rendered into the contextualise prompt on **every**
follow-up. So an unbounded window makes a conversation more expensive the longer
it runs, and the turn where somebody pasted a policy document into the chat box
is the turn that makes every later one expensive. Counting turns is free and
catches the ordinary case; counting characters is what bounds the bill.

`_trimmed` never drops the last turn however long it is — a window holding
nothing is worse than a window holding one oversized thing.

The window is immutable in use: `add()` returns a new window, so a node holding
one cannot have it changed underneath by a sibling.

### Compaction never touches the recent window

```python
if not memory.summary_needed() or len(memory.turns) <= KEEP_RECENT:
    return memory
older, recent = memory.turns[:-KEEP_RECENT], memory.turns[-KEEP_RECENT:]
```

`KEEP_RECENT = 4` — two exchanges. The one job the conversation does in this
system is resolving a pronoun: turning "what about contractors?" into a question
that stands alone. A summarised turn cannot do that with precision. "The
employee asked about leave" does not tell you whether "it" was the carry-over cap
or the expiry date.

So only what has already fallen out of pronoun range is condensed, and a failed
summary degrades to plain truncation — which keeps the *recent* turns. A failed
summary must cost the conversation its oldest turns, never its newest; the
opposite turns a provider outage into a follow-up that cannot resolve.

### `MEMORY_KEYS` is a closed set

```python
MEMORY_KEYS = frozenset({
    "employment_type", "department", "location",
    "work_mode", "tenure_band", "preferred_language",
})
```

Six keys, and each is there because some document in `data/private_kb/` answers
differently depending on it: employment type (salaried against contract staff),
department and location (local law on PTO payout), work mode (the hybrid
minimum), tenure band (notice periods), and the language the reply is written
in.

Free-text keys would turn this into a profile store, and a profile store is a
thing that needs a privacy policy. An unknown key returns `None` and logs at
debug rather than raising — the caller is a language model extracting facts from
prose, and an unexpected key is its ordinary failure mode.

Three more refusals in the same module: the subject is a hashed pseudonym rather
than a name (pseudonymisation, not anonymisation — the docstring says plainly
what that does and does not buy against someone holding both the file and a
staff list); every fact carries a confidence, and below `CONFIDENT = 0.5` it
renders with "(unconfirmed)"; and `remember()` **replaces** rather than appends,
because two contradictory values for `employment_type` are permanently unusable
where one stale value can be corrected. The higher confidence of the two
survives, so a fact the employee stated outright is not downgraded by a later
guess.

**What it cost.** The useful thing memory could do — remembering anything
interesting — is exactly what the closed key set forbids. That is the trade, and
it is the right way round for an HR product.

---

## RAG as a tool

**What it is.** `app/services/toolbox.py` populates the registry with the real
services. The machinery is agent-layer and may not import services, so the
registry is *built* where the services live.

**Why it is a module and not ten lines in `create_app`.** The names, the
authorization actions and the return shapes are a contract with two other files,
and a contract spread across a factory function is one nobody can read.

### `kb_search` is the exact name it is

`app/agent/react.py` derives `source_used` from which tools **succeeded**, by
name:

```python
KB_TOOLS = frozenset({"kb_search", "policy_search", "kb_lookup"})
WEB_TOOLS = frozenset({"web_search", "public_search"})
```

A private-KB search registered under any other name produces an answer that *is*
grounded in company policy and reports itself as `direct` — answered from
memory. That is the one claim this product must never get wrong, and it would
fail silently. The names are a contract; change both ends together.

### The return shape

```python
{"chunks": [Document, ...],
 "citations": [{"source": ..., "title": ..., "department": ...}, ...],
 "text": "[1] source=leave-and-time-off.md\n..."}
```

`text` is what the model reads next turn, `citations` is what accumulates onto
the answer, `chunks` is the evidence itself. `text` goes through
`format_documents` — the exact format the grader prompt was calibrated against —
because the agent reading that observation is making the same judgement the
grader makes, and showing it something else would calibrate one path and not the
other. `ToolResult._render_value` recognises a dict carrying `text` and uses it,
which keeps a page of chunk metadata out of the prompt.

`react.py` reads the shape tolerantly on purpose: it may not import the services
layer, so the only thing holding the two ends together is a docstring. A tool
returning a bare list of Documents, or a dict with only `text`, degrades to no
citations rather than to an exception.

### A tool whose service is absent is not registered

Not registered-and-failing. The catalogue is what the model is shown, and
showing it a tool that cannot work is inviting it to spend a step discovering
that. A deployment with no Tavily key therefore has an agent that never tries
the web, which is the honest shape. `render_catalogue(principal)` filters the
same way for authorization.

---

## Retries and timeouts

**Only for an idempotent tool.**

```python
allowed = 1 + (self._retries(spec) if spec.idempotent else 0)
```

Retrying a call that may already have taken effect is how one request becomes
two. The registry cannot tell a request that failed from a request whose *reply*
failed, so retrying is simultaneously the only way to survive a flaky provider
and the only way to duplicate a side effect.

**The deployment's setting is a ceiling, not a default.**

```python
return min(float(spec.timeout_seconds), ceiling)      # TOOL_TIMEOUT_SECONDS
return max(0, min(int(spec.retries), ceiling))        # TOOL_RETRIES
```

A tool that declares a generous timeout of its own still cannot hang a request
on a deployment that will not wait that long.

**Backoff is exponential with jitter, and the jitter is not decoration.**

```python
delay = base * (2**attempt)
return delay + random.random() * delay * 0.25
```

Several calls that failed together — which is what a rate limit or a brief
outage produces — would otherwise retry at the same instant and reproduce the
burst that caused the failure, so the second attempt fails for the same reason as
the first.

**The honest caveat.** `_invoke` submits the handler to a
`ThreadPoolExecutor` and calls `.result(timeout=...)`. That bounds **the
caller's wait, not the work**: Python cannot kill a thread, and there is no safe
general way to interrupt one. The handler may still be mid-request when we have
given up on it. Two consequences are deliberate rather than accidental:
`tool_parallelism` is kept small, because a timed-out call keeps its thread until
it finishes; and `executor.shutdown(wait=False)`, because waiting would re-join
the very thread the timeout exists to stop waiting for. A handler is still
expected to pass its own timeout to its HTTP client rather than relying on this.

---

## Idempotency and budgets

### The atomic claim

`app/services/idempotency.py`. A fingerprint of the request is claimed *before*
the side effect runs, and a second call carrying the same fingerprint is handed
the first call's result.

```python
connection.execute("BEGIN IMMEDIATE")
self._clear_stale(connection, key, now)
cursor = connection.execute(
    "INSERT INTO idempotency (key, created_at, expires_at, state, action) "
    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(key) DO NOTHING", ...)
won = cursor.rowcount == 1
```

`INSERT ... ON CONFLICT DO NOTHING` then read, inside one `BEGIN IMMEDIATE` —
**not** a SELECT and then an INSERT. Two threads running the latter both see no
row, both insert, and both perform the side effect: precisely the duplicate the
ledger exists to prevent, reintroduced by the check that was supposed to prevent
it. `rowcount` on the insert is what says whether *we* created the row.
`BEGIN IMMEDIATE` rather than sqlite3's implicit `DEFERRED`, which takes its
write lock late and can lose the upgrade to a concurrent writer.

`Claim` has three states rather than a bool, because "somebody else is doing it
right now" is neither a yes nor a safe no: replaying a result we do not have
would invent one, and running anyway is the duplicate. The caller refuses.

`fingerprint` is stable across *processes* — sorted keys, canonical separators,
nothing from `hash()` (`PYTHONHASHSEED` randomises it for strings), `id()` or
the clock — because the claim is written by one request and matched by another,
which after a container restart is a different interpreter.

### The stall window

```python
if row["state"] == STATE_IN_FLIGHT and _older_than(row["created_at"], self.stall_seconds, now):
    log.warning("...treating it as abandoned so a retry is not refused for the whole TTL")
    connection.execute("DELETE FROM idempotency WHERE key = ?", (key,))
```

`ttl_seconds` is a day: long enough that a retry an hour later still finds the
result, short enough that the table does not grow without bound.
`stall_seconds` is 60 and does something else entirely. A process killed
mid-call leaves a key nobody will ever complete or release, and since the row
survives for the full TTL, **one crash would become a day of refused retries**.
Sixty seconds is comfortably past any tool timeout the registry allows, so a
live call is never stolen from.

### A timed-out side-effecting call keeps its claim

```python
if claim is not None and not timed_out:
    self._release(claim)
elif claim is not None:
    log.warning("[tool.%s] timed out with a claim held; leaving it in flight so "
                "a retry cannot duplicate the side effect", spec.name)
```

This is the `ThreadPoolExecutor` caveat above, cashed out. The worker thread may
yet perform the side effect, so releasing the claim would let a retry duplicate
it. The claim is left in flight and the stall window decides when it is safe to
re-win. The cost is a real one: for up to sixty seconds, a legitimate retry of a
genuinely-failed call is refused. That is the correct direction — a duplicated HR
ticket is worse than a refused one.

**In-memory mode is per process, and it is not a distributed lock.** With no
store and no path the ledger is a dict behind a lock: correct for one process,
worth nothing across two. `stats()["backend"]` reports `"memory"` so an operator
cannot be in that mode without knowing.

### Five ceilings, and two ways to spend

`app/agent/budgets.py`. A graph with a bounded rewrite loop needs one counter.
An agent that chooses its own tools needs a real budget, because the failure
changes shape: a graph that loops forever is a bug visible in the wiring, while
an agent that loops forever is a model deciding, each time, that one more search
would help. Nothing in the wiring is wrong.

| Ceiling | Bounds |
|---|---|
| `steps` | the loop — how many times the agent gets to decide |
| `tool_calls` | how much work it does, including fan-out and retries |
| `llm_calls` | a self-chat |
| `tokens` | prompt growth |
| `seconds` | a provider that is merely slow |
| `cost` | the number an operator is asked about — a *projection*, not a bill |

`exceeded()` checks wall clock first: a run that has already taken 45 seconds
should say so rather than report a step count that happens to also be full.

**`allows()` is the graceful path; `charge()` raising is for the inner call.**

```python
if not self._budget.allows("steps"):       # agent loop: compose what we have
    stopped = True
    break
budget.charge("tool_calls")                # inner call: raises BudgetExceeded
```

An employee asking about parental leave should get the partial answer the agent
assembled, labelled as partial, not a 500 saying `max_steps`. So the loop asks
before it spends and never catches `BudgetExceeded` on the main path. `allows()`
records `stop_reason` as a side effect, so a run that stopped early always
carries its reason into the trace and the audit row, and
`ReActResult.stopped_on_budget` sits beside the answer rather than being
inferable from it — a partial answer that does not say it is partial is
indistinguishable from a complete one that happened to be short.

`charge_llm` / `charge_tool` / `charge_step` never raise, deliberately: the call
has already happened by the time its token count is known, so refusing to record
it would make the budget lie about a cost genuinely incurred. The *next*
`allows()` is what stops the run.

The budget is **one shared mutable object**, passed to everything in a run,
which is what makes "the run cost 12 calls" a fact rather than six opinions. It
is therefore locked: parallel tool calls charge concurrently, and a budget that
undercounts under concurrency does not bound anything. `DEFAULT_BUDGET` exists
for a caller that wants the shape without the numbers and must never be charged
against — it is one object, so it would accumulate across every run in the
process.

---

## Authorization and approval

### Deny by default

`app/services/authorization.py` is one table from action to the roles that hold
it. Three roles, because the product has three kinds of user: an **employee**
asks questions and may ask for a person, **HR** additionally uploads policy and
decides approvals, an **admin** operates the deployment.

```python
POLICY: dict[str, frozenset[Role]] = {
    "kb.read":          frozenset({"employee", "hr", "admin"}),
    "escalate":         frozenset({"employee", "hr", "admin"}),
    "ticket.create":    frozenset({"employee", "hr", "admin"}),
    "kb.write":         frozenset({"hr", "admin"}),
    "approval.decide":  frozenset({"hr", "admin"}),
    "audit.read":       frozenset({"admin"}),
}
```

**An action not in the table is denied to everybody**, admin included, and
logged. A table that defaulted open would mean adding a tool quietly grants it
to every anonymous visitor — and the whole value of having a table is that
granting is a visible edit to it.

`audit.read` is the line worth defending: every audit row is an employee's
question and the answer they were given, which is HR-sensitive by definition, so
reading it is an admin action and not an HR one *even though HR is the
department the product is for*.

Roles resolve from the same two keys the HTTP layer already gates on, through
`Settings.check_admin_key` / `check_hr_key` — constant-time, and `False` when
the role is not configured at all, so an unconfigured role can never be assumed
by an empty header. Admin beats HR when both are presented, so a caller gets the
wider role rather than whichever header the framework read first.

`app/services/toolbox.authorize` defaults a missing principal to `ANONYMOUS`
rather than to allow-everything. Passing `authorize=None` — which the registry
reads as "allow" — would make a route that forgot to resolve a principal hand an
anonymous visitor the HR role's tools.

### The asymmetry between `raise_ticket` and `escalate_to_human`

Both are `side_effecting=True`, `idempotent=False`. They differ on one flag:

| Tool | `needs_approval` | Why |
|---|---|---|
| `raise_ticket` | `True` | It files something in an HR queue in somebody's name. A language model should be able to propose that; a person should agree to it. It is the one tool here fingerprinted into the ledger **and** gated behind an approval — which is the combination that makes a retried request safe to allow at all. |
| `escalate_to_human` | `False` | Escalating **is** asking for a human. Requiring a human to approve the request for a human is a loop with nobody in it. |

The registry treats the two outcomes asymmetrically too. `needs_approval` ends
the ReAct run — a person now owns the request, and letting the model carry on
would have it looking for a way round a control that exists to stop it.
`denied` is only an observation, because this caller not being allowed one tool
says nothing about the others: an employee refused the ticket tool can still be
told the policy.

`app/services/approval.py` holds the queue. Its one inverted contract is worth
stating: everywhere else in this codebase a failed write degrades a feature, but
a `None` from `request()` means **the approval was not recorded** — no queue
entry, no record that the agent ever asked — so the caller must refuse. Telling
an employee their question has been escalated when nothing recorded it ends with
nobody looking at it and the employee waiting.

---

## Tracing

**What it is.** `app/core/tracing.py`. A span tree built in the request itself,
returned on the API response, rendered in the console. No dependency, no
network, no key.

**Why it is here.** LangSmith answers "where did the time and tokens go, and
what broke?" for a development machine with a key. It is off by default, redacts
payloads when on, and needs a third party — and an interviewer clicking through
the console cannot be asked to open somebody else's LangSmith project.

### Self time, not total

```python
bucket["self_ms"] += max(0.0, span.duration_ms - child_ms)
```

Nested spans double-count. An `agent` span wrapping four `llm` spans would
report the LLM time twice and make the percentages **sum past 100**, which is
the specific way a latency breakdown becomes useless: every number looks
plausible and the chart is wrong. Self time is a span's duration minus the
duration of the spans *directly* inside it, which is what "where did the time
go" actually means.

`Span.depth` is the nesting level at the moment the span opened, so a flat list
reconstructs a tree with no parent pointers — and with no parent pointer there
is nothing to get wrong when a span is dropped. The span is appended on
**entry**, not on exit, so a run that dies half-way still shows where it got to.

### Payload redaction is the default

A span's attributes are serialised onto an HTTP response and written to the
audit log, so an attribute holding the employee's question or a retrieved policy
chunk is that text leaving the span and entering somewhere it was not accounted
for. Keys in `PAYLOAD_KEYS` — `question`, `answer`, `evidence`, `query`,
`observation`, `thought`, and ten more — are replaced with their character count
when `redact_payloads` is on, which keeps the shape of what happened ("the
grader saw 2.1k characters") and drops the content.

The list is in one place so a new caller cannot invent a synonym that escapes
redaction. A non-string value under a payload key is kept as-is: `{"chunks": 4}`
is a count, and replacing it with a character count would be a loss.

Three places deliberately record a type name and nothing else —
`HybridRetriever._side`, `Span.error`, `guard()`'s `span.set(failure=...)` —
because a provider error carries URLs, request ids and sometimes the offending
payload, and this process holds API keys.

### Thread-local depth makes a fan-out read as a fan-out

```python
parent_depth = self._tracer.depth        # read on the calling thread
...
def _worker(self, parent_depth, ...):
    self._tracer.adopt_depth(parent_depth)
```

`ToolRegistry.call_parallel` runs tool calls on a threadpool and every one of
them traces, so the span list is mutated from several threads at once. Depth is a
`threading.local` and the list is behind a lock. A worker reading `tracer.depth`
would find zero, so the parent reads it and passes it in; without that, a
parallel fan-out reads as four unrelated root spans instead of four siblings
under the span that fanned out.

`null_tracer()` is returned rather than `None` so every call site can write
`tracer.span(...)` unconditionally.

---

## Routing

**What it is.** `app/services/routing.py` picks a model per question from an
offline heuristic, and says why in words.

**Why it is here.** Every question currently pays for the same model. "Hi" pays
for it, and so does a two-area prorated-accrual question that genuinely needs
the better one. Routing is the largest cost saving available that does not
involve answering fewer questions — **and it is the one most likely to quietly
damage quality**, because a cheap model on a hard question produces a plausible
answer, not an error. Nothing in the logs says "this one should have gone to the
big model".

Three things hold it in place: the classifier's uncertain answer is the
**middle** tier and never the cheap one (a classifier whose default is "fast"
silently degrades exactly the questions it failed to understand); the decision is
reported on every answer, with its reason in prose, into the trace and the audit
row; and it switches off, because a saving nobody measured against an unrouted
baseline is not a saving. The classifier runs even when routing is disabled, so
both halves are recorded under the same name.

It is a heuristic and not a model call, because asking a model which model to use
costs a model call — which is most of what routing was trying to save on a short
question. `COMPLEXITY_PROMPT` exists in `app/agent/prompts.py` for the deployment
that would rather pay it.

### The measured correction

The arithmetic signals were originally **one tuple**, any match promoting to
`complex`. "How many" is how the single most ordinary HR question there is
begins. Measured against the console's own suggested questions:

| Question | Before | After | Should be |
|---|---|---|---|
| "How many days of paid time off do full-time employees get?" | `complex` | `standard` | `standard` |
| "What is the notice period if I resign?" | `complex` | `standard` | `standard` |
| "I'm a contractor going on parental leave mid-notice, what am I owed?" | `standard` | `complex` | `complex` |

Wrong in both directions at once. Two plain lookups routed to the deep model and
— on `AGENT_MODE=auto` — to the supervisor, which is four model calls for a
question one retrieval answers; while the genuinely hard one in the same set went
to the middle tier. The classifier was not slightly miscalibrated, it was
inverted on exactly the cases that matter.

### The strong/weak split, and the population signal

```python
ARITHMETIC_STRONG = ("difference between", "versus", " vs ", "compared to",
                     "prorated", "pro rata", "calculate", "work out", "add up")
ARITHMETIC_WEAK   = ("how much", "how many", "how long", "if i", "more than",
                     "less than", "carry over", "remaining", "total")
```

A strong phrase promotes on its own. A weak one is ordinary alone and suggestive
in company: it promotes only alongside a second weak signal, or inside a question
with some size to it (`WEAK_WORDS = 18`, `WEAK_CLAUSES = 3`). "How much leave" is
a lookup; "how much leave do I carry over if I started in March" is not.

The signal that fixed the hard case is `POPULATION` — contractor, intern,
part-time, fixed-term, probation, secondment. It is the strongest single-area
signal there is, and the reason is in the corpus rather than the grammar: these
policy documents say "employees", and whether a statement about employees covers
a contractor is precisely the judgement the LLM grader is recorded as finding
borderline (`weak` 8/8 in isolation, flipping to `good` inside a longer suite
run). "Full-time" is deliberately absent — it is the population the corpus *is*
written for, so naming it makes a question more ordinary, not less.

Two more corrections of the same kind are recorded in the module: "accrual" and
"accrue" are kept out of the `pay` area list, because in this corpus accrual is a
leave concept ("full-time employees accrue 20 days of paid time off") and filing
it under pay made every ordinary PTO question report two policy areas; and
`GREETING` accepts an optional closed list of forms of address, because "hello
there" was classified `standard` and spent the middle model on the word "hello".

### The labelled set

`LABELLED_COMPLEXITY` in `tests/test_routing.py` — fourteen entries, three
`simple`, six `standard`, five `complex`. It scores **14/14** now; before the
strong/weak split it scored **10/13**.

One table rather than a test per entry, because the value is the ratio: a
classifier is not right or wrong on a question, it is right on a proportion of
them, and a change that fixes one case by breaking two is a change the
individual tests would report as progress.

**What it cost.** A heuristic that has to be re-measured every time the corpus's
vocabulary changes, and a labelled set small enough that one case is seven
percent of it.

---

## Semantic caching

**What it is.** `app/services/cache.py` — an LRU of answers, keyed by normalised
text and then by meaning.

**Why it is the most dangerous cache in this product.** Two questions can be
arbitrarily close in embedding space and have different correct answers. "How
much PTO do I get?" and "how much PTO do contractors get?" differ by one word
and by the entire answer, and the KB has a separate policy for each. A cache
that serves the first to the second is not a slow answer or a missing answer —
it is a confidently wrong HR answer **with a citation attached**, which is the
failure this project spends most of its design avoiding.

So every knob is set against the cache rather than for it.

### Exact match first

```python
exact = self._take_exact(normalise_question(question))
if exact is not None:
    return self._note_hit(CacheHit(entry=exact, similarity=1.0, kind="exact"), tracer)
if self.embedding is None:
    return self._note_miss(tracer, reason="no embedding provider")
```

A normalised-text match catches the common case — the same question twice, a
page reload, a retry — and costs nothing. Paying an embedding round trip to
discover you are holding the identical string is the cache charging rent. It also
means the cache **works with no embedding at all**, which is the mode a
deployment with no embedding provider runs in, and `stats()["mode"]` says which
mode it is in so a deployment cannot quietly be in the other one.

### The threshold is high, and the TTL is short

`cache_similarity` is **0.95 raw cosine**. Measured on the labelled set,
paraphrase pairs sat at 0.94–0.99 and the closest *different* question pair at
0.91. So 0.95 buys the reloads and the retries and declines the near misses. **A
cache that hits more often is not a better cache here.** `_take_nearest` returns
the best score even on a miss, so "nearest was 0.91" is available — it is the
number that tells you whether the threshold is doing its job or merely sitting
above everything.

`cache_ttl_seconds` is fifteen minutes. Policy changes, and a stale HR answer is
a wrong answer that looks exactly like a right one: same prose, same citation,
superseded content. Fifteen minutes covers the burst one announcement causes
without outliving an edit to the policy it is about. `invalidate()` is
whole-cache only, on purpose — an ingest can change the answer to a question
mentioning none of the words in the file, so there is no honest way to invalidate
selectively.

The whole answer object is stored, citations included. Serving cached text beside
freshly computed citations would attach today's sources to yesterday's words,
which is worse than either being stale alone: the citation is what an employee
checks the answer against.

### `CACHEABLE_SOURCES`

```python
CACHEABLE_SOURCES = {"private_kb", "web_search", "direct"}
```

An allowlist, and the two values missing from it are the point.

| Not cached | Why |
|---|---|
| `error` | Caching a failure for fifteen minutes turns one outage into fifteen minutes of outage, and the retry that would have worked never reaches a provider. |
| `insufficient_evidence` | Caching "I could not find that" means the ingest that fixes it changes nothing for a quarter of an hour — to the one employee most likely to ask again. |

An allowlist rather than a denylist, so a payload with no `source_used` at all —
a partial state, a hand-built dict — is declined rather than stored on the
assumption that it was fine. A declined `put` is recorded in the trace, because
"why did that not cache?" is otherwise a question only the source answers.

Embedding failure is absorbed into a miss and counted in `stats()`: the cache is
an optimisation, so its failure mode is "no cache", never "no reply" — and a
provider failing every call shows up as a number rather than as a hit rate that
quietly went to zero.

---

## Parallel tools

`ToolRegistry.call_parallel`.

### Results come back in call order, never completion order

```python
requested = [(index, name, arguments) for index, (name, arguments) in enumerate(calls)]
...
return [results[index] for index, _, _ in requested]
```

The results are rendered into the next prompt as observations, and a model
reading them out of order attributes the wrong answer to the wrong question —
which for an HR answer means a correct sentence about the wrong policy.
Completion order is also nondeterministic, so the bug would appear in roughly one
run in three and never in a test.

### A side-effecting tool is not parallelised

```python
if spec is not None and spec.side_effecting:
    self._tracer.event("tool.sequential", kind="tool", tool=name)
    results[index] = self.call(name, arguments, principal=principal)
```

It runs on the calling thread, before the fan-out. Two side effects racing on
the same idempotency key is exactly the case the ledger **cannot** fix: both
claims are decided before either completes, so one wins and the other refuses the
employee's request for no reason the employee can see.

### `tracer.adopt_depth`

Each worker adopts the depth read on the parent thread, so the fan-out reads as
a fan-out rather than as four root spans. See *Tracing* above.

One call raising does not lose the others: `_worker` catches anything that
escapes `call`'s own error handling and answers with a failed `ToolResult`.

---

## Summary

| Pattern | File | Why it exists |
|---|---|---|
| ReAct loop | `app/agent/react.py` | the question the fixed graph has no edge for |
| Tool registry | `app/agent/tools.py` | an agent that chooses its tools has no wiring to enforce the rules |
| Real tool set | `app/services/toolbox.py` | the agent layer may not import services, so the registry is populated where they live |
| Answer pipeline | `app/services/pipeline.py` | eleven stages around the agent, each switchable, so each is measurable |
| Degrading nodes | `app/agent/nodes.py` (`guard`) | one flaky provider must not cost an answer the other source could give |
| Short-term memory | `app/agent/memory.py` | a bounded window, because history is re-rendered on every follow-up |
| Long-term facts | `app/services/memory_store.py` | six closed keys, because a seventh makes it a profile store |
| Budgets | `app/agent/budgets.py` | a model deciding "one more search" is not a bug in the wiring |
| Idempotency | `app/services/idempotency.py` | retrying is also how one HR ticket becomes two |
| Authorization | `app/services/authorization.py` | one deny-by-default table, so granting is a visible edit |
| Approvals | `app/services/approval.py` | some things should not happen because a model decided they should |
| Tracing | `app/core/tracing.py` | observability with no key, no network and no third party |
| Model routing | `app/services/routing.py` | the largest saving available, and the one most likely to hurt quietly |
| Semantic cache | `app/services/cache.py` | the most dangerous cache here, so every knob is set against it |
