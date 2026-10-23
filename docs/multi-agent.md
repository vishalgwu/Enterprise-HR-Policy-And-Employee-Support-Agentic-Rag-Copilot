Part of the [HR Copilot](../README.md) documentation set.

# The supervisor-worker team

Source: [`app/agent/supervisor.py`](../app/agent/supervisor.py),
[`app/agent/prompts.py`](../app/agent/prompts.py),
[`app/services/scratchpad.py`](../app/services/scratchpad.py),
[`tests/test_supervisor.py`](../tests/test_supervisor.py). The pipeline that
chooses between this and the single-agent graph is
[`app/services/pipeline.py`](../app/services/pipeline.py).

---

## 1. The argument against this, first

A multi-agent system is a worse engineering choice than a single graph on almost
every axis that is easy to measure.

| | Single graph (`app/agent/graph.py`) | Supervisor team |
|---|---|---|
| Model calls for an ordinary question | 2–4 | 4+ (delegate, worker, delegate, compose) |
| Latency | one retrieval, one generation | serial: each handoff waits for the last |
| Failure surface | one path, every node guarded | every worker can degrade independently |
| Determinism | branches are deterministic given a decision | the *sequence* of workers is model-chosen |
| Can it contradict itself? | no — one evidence set, one generation | yes — two specialists can report conflicting policy |

That last row is the one that matters and it is not hypothetical here. The
knowledge base overlaps on purpose: notice periods, PTO payout on termination,
expense payment timing and core hours each appear in more than one document. Two
specialists reading two documents can report two different versions of the same
fact, and a composer that merges them silently produces a confidently wrong HR
answer. `COMPOSE_PROMPT` is written against exactly that:

> Where two specialists disagree, say so explicitly and name both documents
> rather than silently picking one -- a confidently merged contradiction is the
> worst answer this team can produce.

So the team has to earn its place. It earns it on one specific shape of question,
and the reason is a property of the corpus rather than a preference for agents.

**A single retrieval over a two-area question fills its top-k with whichever
area the corpus covers better, and the other half goes unanswered while the
answer reads as complete.** "How is my notice period paid, and does my
insurance continue?" needs `payroll-and-compensation.md` and
`benefits-and-insurance.md`. One `k=4` retrieval over both will usually return
four chunks about notice pay, because that is the better-represented topic, and
the generation will then write a fluent, correctly-cited answer that never
mentions insurance. Nothing in the trace says a half was dropped. The grader
cannot catch it either: every chunk it sees is relevant.

Two retrievals, each scoped to one specialist, is the fix. The supervisor is the
machinery for deciding there are two.

### The graph

```
START -> plan -> supervise
  supervise -> policy | benefits | payroll | workplace | web | general
  supervise -> escalate                              (a person must decide)
  supervise -> compose                               (done, or out of room)
  <any worker> -> supervise                          (handoff back)
  compose  -> END
  escalate -> END
```

It is a compiled `StateGraph`, not a `while` loop, and that is deliberate. A
loop would be shorter and invisible: `docs/agent-graph.md` is generated from the
compiled edges, the console lights nodes from the trace, and
`test_the_compiled_graph_matches_the_documented_design` is what stops the
picture drifting from the code. A multi-agent system is the one whose control
flow a reader most needs drawn.

---

## 2. Supervisor and workers

**The supervisor does not answer.** It has exactly two jobs: pick the next
specialist, or say the team is done. `SUPERVISOR_PROMPT` opens with it — "You do
not answer questions yourself" — and the node body backs that up: `supervise`
returns a `next_worker` and a reason, and nothing else. The only prose the
employee ever reads comes from `compose` or from `ESCALATED_ANSWER`.

Six specialists, mapped onto the eleven files that actually exist in
`data/private_kb/` rather than onto a generic HR org chart. `WORKER_DOMAINS` is
the table, and each entry carries `doc_hints` holding real file names — which is
what makes a specialism checkable rather than aspirational.
`test_every_worker_domain_names_documents_that_exist` fails if a hint names a
file that is not there.

| Worker | Covers | Documents |
|---|---|---|
| `policy` | leave, time off, attendance, working hours, remote and hybrid | `leave-and-time-off.md`, `attendance-and-working-hours.md`, `remote-and-hybrid-work.md` |
| `benefits` | insurance, medical cover, allowances, wellbeing | `benefits-and-insurance.md` |
| `payroll` | pay, salary, payslips, tax, expenses, reimbursement | `payroll-and-compensation.md`, `expense-reimbursement.md` |
| `workplace` | conduct, grievance, data privacy, records, forms | `code-of-conduct-and-grievance.md`, `employee-records-and-data-privacy.md`, `hr-forms-and-requests.md` |
| `web` | public, non-company information | none — Tavily |
| `general` | anything that does not obviously belong above | `onboarding-and-offboarding.md`, `hr-copilot-how-it-works.md` |

`onboarding-and-offboarding.md` sits under `general` rather than `policy`
because joining and leaving cut across every other specialism — payroll
registration, benefit elections, equipment, notice — so filing it under one
would hide it from the other three. `general` is also unfiltered by
declaration — it is the one specialist meant to see the whole corpus.

**On the shipped corpus, though, no worker filters at all, and that is
deliberate.** `departments` is the metadata filter a *foldered* deployment uses:
`app/rag/loaders` derives a chunk's `department` from its subdirectory under
`data/private_kb/`, and files in the root become `general`. The shipped corpus
is flat, so `_foldered_departments` returns an empty set and `retriever_for`
applies no filter — because an unconditional per-worker filter would return
**zero chunks for five of the six specialists**, which looks exactly like an
empty knowledge base. A deployment that files benefits under
`data/private_kb/benefits/` gets the filter with no other change. Until then,
the specialisms differ by the prompt each worker is given and by what the
supervisor asks them, not by what they can retrieve.

The `description` strings in `WORKER_DOMAINS` are word-for-word what
`SUPERVISOR_PROMPT` lists, because the prompt is what the model routes on and
the table is what the code routes on. Two descriptions of one specialist is how
a question gets delegated to a worker whose documents do not cover it.

### A worker that found nothing did its job

`WORKER_PROMPT` asks a specialist to report in two or three sentences from the
evidence given, and then:

> If it does not cover the question, say so in one sentence and stop -- a
> specialist reporting 'my documents do not cover this' is doing their job, and
> guessing to look useful is the one failure the supervisor cannot detect.

`covered` is therefore a flag on the worker's own `Finding`, not a grade the
supervisor awards. `NOT_COVERED_MARKERS` reads the sentence the prompt asked
for, and an uncovered finding has its `text` and `citations` dropped, so hedging
prose cannot be merged into the answer as fact. A phrasing the marker list
misses costs the composer one paragraph of scaffolding to drop — the cheap
direction for this check to fail in.

### The delegation decision

`DelegationDecision(worker, reason, done)`, bound through
`llm.with_structured_output`. It follows the same recovery shape as
`route_question` and `grade_evidence` in `app/agent/decisions.py`: try the
structured call, recover the model's raw text from a Groq `tool_use_failed`
body with `failed_generation` + `coerce`, then fall back to a safe default.

The default is asymmetric, and `_delegate`'s docstring says why:

| Hop | Unreadable decision falls back to | Why |
|---|---|---|
| First | `general`, `done=False` | one hop to the unfiltered specialist is worth a call |
| Any later | `general`, `done=True` | the second is a loop |

Defaulting to another handoff on an unreadable decision is how a broken model
spends the whole budget: every hop fails the same way, every failure says
"delegate", and the run consults six specialists before composing from whatever
the first one happened to find.
`test_an_unreadable_delegation_decision_defaults_to_general_then_stops` and
`test_a_failed_structured_call_recovers_the_worker_the_model_named` pin both
halves.

### Why the hard rules are in the branch, not the prompt

Three things must hold on every run:

1. The web specialist is unreachable until a private one has reported.
2. No specialist is consulted twice.
3. The loop stops at `max_handoffs` (default 4).

`SUPERVISOR_PROMPT` asks for all three. **A prompt instruction is a suggestion**
— it holds on most questions and fails on the one where the model is confident —
and these failures are not cosmetic:

| Rule broken | What the employee gets |
|---|---|
| Web before private | a public article answering a question company policy covered, and no way to tell which they were given |
| Same worker twice | a handoff spent re-reading documents already read, taken from the question's other half |
| No ceiling | a model deciding, each time, that one more opinion would help |

So `delegate_after_supervise` enforces them in code, and the prompt's version
exists only to make the model's first guess usually right. The function is the
graph's only conditional branch:

```python
def delegate_after_supervise(state: SupervisorState) -> str:
    stop = forced_stop(state)
    if stop is not None:
        return "compose"
    target = state.get("next_worker") or "general"
    ...
    if target in consulted:
        return "compose"
    if target == "web" and not _private_reported(state):
        fallback = _first_unconsulted_private(state)
        return fallback or "compose"
    return target
```

Note the web rule: the rule is "private first", not "never web", so an early
request for the web becomes the private attempt it should have been rather than
a refusal. `test_the_web_worker_is_unreachable_before_a_private_worker_has_reported`
and `test_the_web_worker_is_reachable_once_a_private_worker_has_reported` cover
both sides.

`forced_stop` is one implementation read by three callers that each need the
same answer: `supervise` (to skip a pointless model call when the decision is
already made), the branch (to force the edge), and `compose` (to say in the
trace why it was reached). The loop never runs on the model's judgement alone:
`done` is a model output, and `forced_stop` is not.

---

## 3. Handoff

Every worker turn writes one handoff record:

```python
{"from": "supervise", "to": worker, "reason": reason, "at": now_iso()}
```

`reason` is the supervisor's own words, carried from `supervise` through
`state["delegation_reason"]` and read by the worker node. That is what puts a
reason *on* the record rather than leaving it to be reconstructed from the trace
later. `test_a_handoff_record_names_who_was_handed_what_and_why` asserts all
four fields.

Four keys on `SupervisorState` use the `add` reducer:

```python
trace: Annotated[list[str], add]
worker_path: Annotated[list[str], add]
findings: Annotated[list[dict[str, Any]], add]
handoffs: Annotated[list[dict[str, Any]], add]
```

**A plain TypedDict key is *replaced* by whatever a node returns.** Without the
reducer, a bare `list` keeps only the last node's contribution — and on this
graph that is catastrophic rather than untidy, because the whole point of the
run is several workers' several reports. Four specialists would compose from one
finding. `trace` loses the rewrite history, `worker_path` loses the path the
console lights, `handoffs` loses everything but the last hop.

All four are seeded empty by `initial_supervisor_state`, because a reducer has
nothing to fold into on a missing key.
`test_the_seeded_state_has_every_key_a_reducer_folds_into` pins that, and
`test_the_accumulating_keys_are_appended_to_rather_than_overwritten` is the test
that fails if any one reducer is removed — it makes four assertions over a
three-worker run for exactly that reason.

Two keys deliberately do **not** accumulate:

| Key | Why no reducer |
|---|---|
| `consulted` | a worker returns the whole new list, so there is one writer and no chance of a duplicate making `len(consulted)` disagree with `handoff_count` |
| `handoff_count` | a plain `int`; a reducer would sum every node's view of the total |

---

## 4. Escalation

Some HR questions are not lookups. They need a judgement the documents do not
contain, they concern someone's individual case, they allege misconduct, or they
ask for an exception. `SUPERVISOR_PROMPT` lists `escalate` beside the six
specialists with exactly that definition, and it is a destination rather than a
worker — nothing hands back from it.

**Escalation outranks `done`.** In `supervise`:

```python
if worker == "escalate":
    target = "escalate"
elif done:
    target = "compose"
else:
    target = worker
```

A supervisor that has decided a person must see this has told you its answer is
not the product, so a `done=True` alongside it is not a conflict to resolve in
favour of answering. `test_an_escalation_request_outranks_a_done_flag` pins it.

**The escalate node must not invent an answer.** It renders `ESCALATION_PROMPT`
into a handover note — what was asked, what policy did and did not establish,
and the one decision the person has to make — and writes that note to the
**scratchpad**, not to the response. The employee gets `ESCALATED_ANSWER` and
nothing else. The whole reason the node exists is that the documents do not hold
the judgement the question needs; a composed reply here would be precisely the
guess the escalation was meant to prevent.
`test_escalation_produces_the_escalation_copy_and_does_not_invent_an_answer`
asserts that the note's text does not appear in the answer.

`source_used` becomes `"escalated"`, which matters downstream: it is not in
`CACHEABLE_SOURCES`, so an escalation is never replayed from cache to the next
person who asks something similar.

### Where the queue is

This graph writes the note; it does not file the request. The agent layer may
not import `app/services/*` (see the layering table in `CLAUDE.md`), so
`scratchpad` arrives duck-typed — anything with
`.append(conversation_id, text, author=, kind=)`.

The durable queue is [`app/services/approval.py`](../app/services/approval.py),
reached from the **ReAct** orchestration rather than from this one: the
`escalate_to_human` tool in `app/services/toolbox.py` calls
`ApprovalStore.request(...)` with `risk="high"`. Two details of that are worth
carrying across:

- **It fails closed.** A `None` from `request()` means the approval was *not
  recorded*, so there is no queue entry for anyone to decide. The tool raises
  `ToolError` rather than proceeding. Telling an employee their question has
  been escalated when nothing recorded it ends with nobody looking at it and the
  employee waiting.
- **`escalate_to_human` is not itself gated behind an approval**, unlike
  `raise_ticket`. Escalating *is* asking for a human, so requiring a human to
  approve the request for a human is a loop with nobody in it.

A supervisor escalation that should also reach the queue is the obvious next
change: pass the store in as a second duck-typed service beside `scratchpad`.
It is not wired today, and the docstring on `escalate` is honest about what the
node does — it writes a note for whoever picks the conversation up.

---

## 5. Shared state

`SupervisorState`, field by field:

| Field | Reducer | What it holds |
|---|---|---|
| `question` | — | what the employee typed, verbatim; what the audit row and the UI show |
| `standalone_question` | — | the resolved wording every prompt sees; seeded equal to `question` |
| `history` | — | normalised turns, carried for the audit row; **no prompt on this graph renders it** |
| `trace` | `add` | one line per step, the console's node highlight and the degradation record |
| `worker_path` | `add` | the specialists in the order they ran |
| `findings` | `add` | one `Finding.to_dict()` per worker turn |
| `handoffs` | `add` | `{"from","to","reason","at"}` per hop |
| `consulted` | — | whole list rewritten by each worker; one writer |
| `handoff_count` | — | plain `int`; the ceiling `forced_stop` reads |
| `citations` | — | deduped across findings by `_dedupe_citations` |
| `subquestions` | — | what `plan` found, if anything |
| `next_worker` | — | what the supervisor *wanted*; the branch decides what happens |
| `delegation_reason` | — | the supervisor's words, read by the worker for the handoff record |
| `escalated` / `escalation_reason` | — | set by `escalate` |
| `answer` / `source_used` | — | written by `compose` or `escalate` only |

There is no contextualise node. This graph is reached either directly (a single
turn) or from `AnswerPipeline`, which has already resolved the follow-up and
passes the resolved wording in as `asked`.

`_dedupe_citations` keeps one entry per source document, in the order the
workers found them. Two specialists routinely cite the same file — the corpus
overlaps on purpose — and an answer listing `leave-and-time-off.md` twice reads
as two sources agreeing when it is one source counted twice.
`test_citations_are_deduped_across_workers` covers it.

### The scratchpad

A single-agent graph keeps its working state in `AgentState` and that is enough,
because one node's output is the next node's input — the wiring carries the
information forward. A supervisor and five workers are a different shape: a
worker needs to see what another worker established *without the supervisor
copying it into a prompt*, and a person reading the trace afterwards needs the
same view, or "why did the payroll specialist say that?" is answerable only by
re-running the question.

`app/services/scratchpad.py` is that surface: append-only, scoped to one
conversation, readable by every participant. It is also the only place in this
system where the *reasoning* is durable rather than reconstructed — the span
trace says what ran, the audit log says what was decided, and neither holds the
sentence a worker wrote for the next worker to read.

The supervisor writes three kinds of note through its `pad()` helper: `plan`
(the planner's verdict), `finding` (each worker's report, or "my documents do
not cover: …"), and `handoff` (who was handed what, and why).
`SupervisorTeam._note_question` also records the employee's question as `user`,
so the pad reads as a transcript. All of it is wrapped: a lost note must not
cost the answer, and `test_a_scratchpad_that_raises_does_not_cost_the_answer`
pins that.

**The renderer is budgeted; the table is not.** `Scratchpad.render()` takes
`max_chars` (default 4000) and drops the oldest unpinned notes to fit.
`Scratchpad.notes()` and the `scratchpad` table do not:

- The table is the **record**. A note that was written stays written, because
  the audit view and the UI both read it, and a record that drops its own
  middle is worse than no record.
- The prompt is the **cost**. Every note in it is paid for on every model call
  that includes the pad, so a pad that grows all conversation makes the last
  turn the most expensive one.

So the ceiling lives where the money is spent. Pinned notes are included
*before* the budget applies at all — pinning is the one way a caller says "this
must reach the model", and a budget that could drop it would make pinning a
suggestion.

Two bounds sit on the write side instead: `max_chars` per note (2000,
**truncated not refused** — a worker that wrote three paragraphs of finding
should lose the tail, not the finding) and `max_notes` per conversation (200 — a
conversation that has written 200 notes is a loop, and the 201st note is almost
certainly the loop's).

The conversation id is matched against `CONVERSATION_ID_PATTERN` and **rejected
rather than sanitised**. Sanitising `a/b` and `a:b` to `ab` merges two
conversations into one, which means one employee's notes rendered into another
employee's prompt. A rejected id costs this conversation its scratchpad; a
silently merged one costs the wrong person their privacy.

---

## 6. When to split

`plan` runs on every question and *spends* on very few. The gate is
`looks_multi_part`, which is deliberately lexical:

```python
SPLIT_MIN_WORDS = 8
SPLIT_HINTS = (" and ", " also ", " plus ", " as well as ",
               " versus ", " vs ", ";")
```

Under eight words, or with no joining word and one question mark, it is one
question. Paying a `SplitDecision` round trip to be told that taxes the common
case to serve the rare one — the same trade `contextualize` makes in
`app/agent/nodes.py`, where no history means no LLM call.
`test_the_plan_node_spends_no_call_on_a_short_single_clause_question` asserts
`"SplitDecision" not in llm.calls`.

Past the gate, `SPLIT_PROMPT` decides, under three rules:

- Split **only** when the question asks about two or more separate policy areas
  that different documents would cover — leave and payroll, benefits and
  conduct.
- Do **not** split a single question that merely has several clauses, asks for a
  number and its deadline, or names two things inside one policy area.
- Two or three parts, never more, each written to stand on its own.

Splitting does not fan out retrieval. It tells the *supervisor* that two policy
areas are in play, which is what turns one handoff into two: `supervise` appends
the parts to the question it sends the delegator.

The parametrised cases in `tests/test_supervisor.py` and the corpus this KB
actually holds:

| Question | Split? | Why |
|---|---|---|
| "What are the core hours?" | no | four words; under the heuristic gate, no call spent |
| "How many PTO days?" | no | under the gate |
| "How is my notice period paid, and does my insurance continue?" | yes | `payroll-and-compensation.md` and `benefits-and-insurance.md` are different documents |
| "How many PTO days do I get and when is payday each month?" | yes | leave and payroll |
| "Tell me about hybrid work rules as well as the expense caps" | yes | `remote-and-hybrid-work.md` and `expense-reimbursement.md` |
| "What is my notice period? What about my PTO payout?" | yes | two question marks; and the KB keeps these facts in different files |
| "How much leave do I carry over if I started in March?" | gate opens, planner should decline | one area (leave) with a condition on it; arithmetic, not two documents |
| "What is the difference between sick leave and casual leave?" | gate opens, planner should decline | two things inside one policy area, one document |

The last two rows are the interesting ones: they reach the planner because they
are long enough and contain a joining word, and the prompt's second rule is what
is meant to stop them. The heuristic is cheap and over-admits; the model is the
precise half.

**The safe default is: do not split.** `plan` only records subquestions when
`needs_split` is true *and* the list comes back non-empty; a failed `SplitDecision`
call degrades through `guard()` to `None` and the run proceeds as a single-area
question. The asymmetry is right:

| Mistake | Cost |
|---|---|
| Wrongly split | one extra delegation call and one extra worker: latency and money |
| Wrongly merged | the employee gets half their answer, and it still reads as complete |

Only one of those is visible to the person who asked.

---

## 7. What it costs

Per run, counting model calls:

| Stage | Calls | When |
|---|---|---|
| `plan` | 0 or 1 | 1 only if `looks_multi_part` opens the gate |
| `supervise` | 1 per hop, plus the final one | a `DelegationDecision` each time; 0 when `forced_stop` already decided |
| each worker | 1 generation | 0 if its retrieval came back empty — no evidence means no prose |
| each worker | 1 retrieval | a Pinecone query, or Tavily for `web` |
| `compose` | 1 | 0 when no specialist covered the question (`INSUFFICIENT_ANSWER` is canned) |
| `escalate` | 1 | instead of `compose` |

So a two-worker supervisor run is, in the ordinary case: 1 plan + 2 delegations
+ 1 final delegation + 2 worker generations + 1 composition = **7 model calls
and 2 retrievals**. Against the single graph on the same question: 1 router + 1
grader + 1 generation = **3 calls and 1 retrieval** (plus a contextualise call
only on a follow-up).

Roughly double the calls, and worse than double in wall-clock terms, because
the handoffs are serial by construction: the supervisor cannot choose the second
specialist until the first has reported.

**That arithmetic is why `AGENT_MODE=auto` is narrow.** `AnswerPipeline._pick_mode`:

```python
if complexity == "complex" and self._graph is None:
    return "supervisor"
return "graph"
```

The supervisor is chosen for exactly the questions `classify_complexity` calls
`complex`, and the graph for everything else. ReAct is never chosen
automatically — it is the most expensive and least predictable of the three, and
a question needing a *tool* is one a human is better placed to recognise than a
keyword heuristic.

**Which makes the classifier's accuracy matter more than it looks.** A
misclassification here is not a slightly worse model on one question; it is four
extra model calls, or a two-area question answered by one retrieval. The
measured regression is recorded in `docs/production.md` and in
`tests/test_routing.py`: the first version of the classifier sent "How many days
of paid time off do full-time employees get?" to `complex` — and therefore, on
`auto`, to the supervisor — while sending "I'm a contractor going on parental
leave mid-notice, what am I owed?" to `standard`. It was inverted on exactly the
cases that matter. `LABELLED_COMPLEXITY` scores 14/14 after the fix and scored
10/13 before it.

Two ceilings bound the worst case regardless:

- `max_handoffs` (default 4), checked by `forced_stop` and enforced in the
  branch. `test_max_handoffs_forces_composition_and_says_so_in_the_trace`.
- The shared `Budget` — `forced_stop` also returns when `budget.allows("steps")`
  is false, and `compose` appends `BUDGET_STOPPED_NOTE` to the answer so the
  employee is told the run was cut short.
  `test_an_exhausted_step_budget_forces_composition`.

LangGraph's own `recursion_limit` is set to `2 * max_handoffs + 10` — generously,
because tripping it *raises* where every other limit here degrades.

---

## 8. What is deliberately absent

| Not built | What it would buy | What it would cost |
|---|---|---|
| **Agent-to-agent messaging** | a worker could ask another worker directly — "payroll, does notice pay include the bonus?" — instead of routing the question back through the supervisor | the supervisor stops being the only control point, so `max_handoffs` no longer bounds the run and the trace stops being a path; two workers can also deadlock waiting on each other, which nothing in `forced_stop` can detect |
| **A shared mutable blackboard** | a worker could correct or retract another's finding, rather than only appending beside it | the append-only scratchpad is what makes the record readable afterwards; a mutable surface means the note that explains an answer can be gone by the time anyone reads it, and the last writer silently wins a genuine disagreement that `COMPOSE_PROMPT` is written to surface |
| **Dynamic team composition** | the model could invent a specialist for a question the six do not fit, instead of falling through to `general` | a specialism is only meaningful if it maps to documents that exist — `WORKER_DOMAINS.doc_hints` and `test_every_worker_domain_names_documents_that_exist` are what keep that honest. A model-invented worker has no slice to retrieve from, so it is `general` with a better name and a worse audit trail |

The scratchpad is the one shared surface, and it is append-only. That is the
whole of the inter-agent communication in this design, and the trace is readable
as a result.
