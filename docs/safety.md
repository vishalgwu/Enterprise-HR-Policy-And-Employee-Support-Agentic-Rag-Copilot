Part of the [HR Copilot](../README.md) documentation set.

# Safety: what goes in, what comes out, and who may ask

Source: [`app/services/pii.py`](../app/services/pii.py),
[`app/services/guardrails.py`](../app/services/guardrails.py),
[`app/services/authorization.py`](../app/services/authorization.py),
[`app/services/approval.py`](../app/services/approval.py),
[`app/services/byok.py`](../app/services/byok.py),
[`app/api/deps.py`](../app/api/deps.py),
[`app/core/config.py`](../app/core/config.py),
[`static/js/dom.js`](../static/js/dom.js).
The guards are *called* from stages 2 and 8 of
[`app/services/pipeline.py`](../app/services/pipeline.py); the per-tool approval
flags are declared in [`app/services/toolbox.py`](../app/services/toolbox.py).
Tests: [`tests/test_pii.py`](../tests/test_pii.py),
[`tests/test_guardrails.py`](../tests/test_guardrails.py),
[`tests/test_authorization.py`](../tests/test_authorization.py),
[`tests/test_approval.py`](../tests/test_approval.py),
[`tests/test_tools.py`](../tests/test_tools.py).

---

## 1. The threat model, first

Nothing below makes sense without saying who is trusted.

| Party | Trusted? | What that means here |
|---|---|---|
| An employee asking a question | Not an attacker, but **the data subject** | Their question is likely to contain their own salary, phone number, national id or medical situation. The job is to keep that out of the places it would otherwise travel, not to defend against them. |
| An internet visitor on a public demo | **Untrusted** | `/api/chat` has no authentication by design, so anyone who can reach the URL can spend the deployment's quota and probe its prompts. |
| A model's output | **Untrusted input to the page** | It is a string from a third-party service, rendered into a browser session that may belong to an admin reading the audit log. |
| An uploaded document | **Untrusted content with a delay** | It is admin-gated on the way in, and then becomes retrieved context for every later question. Whatever is in it reaches a prompt and an answer. |
| The deployment's own operator | Trusted | Holds the keys, reads the audit log, decides the approvals. |

**Explicitly not in scope.** There is no authentication of employees — no SSO,
no per-person identity, no session. There is therefore no tenant isolation and
no per-employee access control on the knowledge base: every chunk in `hr-docs`
is answerable to every caller. There is no DLP on outbound mail, because nothing
here sends mail. There is no encryption at rest beyond whatever the host volume
provides, and the `/api/audit` key is the only thing between a reader and every
question the deployment has answered.

Those are deliberate boundaries of a product that answers questions about
published internal policy. They would all have to change before this held
anything employee-specific, and the one that has to change *first* is identity:
several controls below are only as strong as "the caller is anonymous and
therefore equal to every other caller".

---

## 2. Input filtering

`InputGuard.check` runs as stage 2 of the pipeline — before the semantic cache,
before routing, before any embedding or retrieval. Order inside it:

| Step | On a match | Severity / category |
|---|---|---|
| Guard disabled (`INPUT_GUARD=0`) | returns the question untouched | — |
| Empty or whitespace | refuse | `low` / `empty` |
| Over `MAX_QUESTION_CHARS` (8000) | refuse | `low` / `too_long` |
| Instruction-shaped span (`INJECTION_PATTERNS`) | replace with `[instruction removed]`, carry on | `high` / `injection` |
| …and nothing of substance left | refuse | `high` / `injection` |
| API-key shape | redact, **always**, even with `REDACT_PII=0` | `high` / `secret` |
| Other PII (when `REDACT_PII=1`) | redact | `low` / `pii` |
| `REFUSAL_TOPICS` match | refuse | `high` / `unsafe` |

The verdict is a `GuardVerdict`, never an exception: a guard that raises turns a
judgement about content into a 500.

The topic check runs last but matches against the **original** question, not the
rewritten one — `_refused_topic(question)`. Neutralising an instruction span or
redacting an identifier therefore cannot accidentally remove the words a refusal
rule was looking for.

### Injection is neutralised, not refused

This is the decision worth arguing about. The obvious design refuses anything
containing "ignore previous instructions". Three reasons this does not:

- **The prompts are already retrieval-grounded.** The generation prompt is handed
  numbered context and told to answer from it; the graders judge evidence rather
  than follow it. An instruction smuggled into a question arrives at a model that
  has been told where its truth comes from.
- **A real employee types it.** "Ignore what I said before" is an ordinary
  mid-conversation correction about their own earlier question. Refusing it costs
  a real person their answer for looking like an attacker.
- **A refusal is a free oracle.** It tells whoever is probing precisely which
  phrasing tripped the check — the single most useful piece of information to an
  attacker, and the one thing flag-and-continue never hands over.

So the span is replaced with `[instruction removed]`, the verdict is recorded at
`severity="high"` on the trace and in the audit row where a human can review it,
and the question is answered. `test_an_injection_attempt_is_flagged_and_neutralised_but_still_answered`
pins that.

Refusal is reserved for the case where neutralising leaves nothing behind.
`_has_substance(text, minimum=3)` strips placeholders and non-alphanumerics and
counts words: a message that is *only* an attack has no question in it, so
refusing costs nobody an answer. Three words rather than one, because "and",
"then" and "please" survive the removal of the instruction around them.

Each pattern is bounded rather than greedy — `[\w\s'-]{0,40}` and similar —
because neutralising removes the span, and a pattern running to the end of the
string would take the employee's actual question with it.

### The pasted key

`pii.redact(text, kinds=["api_key"])` runs unconditionally, before and
independently of the `REDACT_PII` switch, and the reason is that a pasted key is
not the deployment's data to choose about: it is a live credential belonging to
someone else, and the audit log is designed to still be holding it in a year.
This deployment cannot rotate it.

The user is **told**, not silently logged:

> That looked like an API key, so it was removed before anything saw it. If you
> meant to use your own key, put it in the Key panel rather than the chat box.

The message names the right destination, because the realistic cause is someone
aiming at the BYOK panel (section 7) and hitting the chat box. Silently
swallowing it would leave them wondering why their key did not work, and
probably pasting it again.

### Off-topic questions are not refused

`"off_topic"` exists in the category vocabulary and this module never produces
it. Two reasons:

- A keyword list that refuses "what is the capital of France?" makes the product
  feel broken.
- It is not even the better detector. The similarity gate scores an
  out-of-domain question below the 0.15 raw-cosine threshold (measured:
  out-of-domain peaked at 0.099 against a weakest correct chunk of 0.171 — see
  the README's retrieval benchmark), and the router sends a greeting straight to
  a direct answer. Both classify the general case better than any word list.

`REFUSAL_TOPICS` is therefore short and specific: a request for a method of
self-harm, violence, fraud, forgery, impersonation, or a legal opinion. Every
entry has to be one a reader would agree with, because the cost of a wrong entry
is a real employee told no. `test_a_general_off_topic_question_is_not_refused`
guards the boundary from the other side.

Note what is *not* there: a statement of distress. "I am being harassed and it is
making me want to hurt myself" is not a request for instructions, and refusing it
would be the worst available answer. That is a case for the escalate path
(section 6).

---

## 3. PII

### The patterns

`PII_PATTERNS` in `app/services/pii.py`, in insertion order — which is
load-bearing, as the final tie-break for two patterns matching the *same* span:

| Kind | Anchored on |
|---|---|
| `api_key` | known provider prefixes: `sk-`, `sk-proj-`, `gsk_`, `pcsk_`, `tvly-`, `lsv2_pt_`/`lsv2_sk_`, `xoxb-`, `ghp_`, `github_pat_`, `AKIA` |
| `email` | the local-part/domain shape |
| `iban` | two letters, two digits, then 10–30 alphanumerics |
| `pan` | five letters, four digits, one letter — matched **case-sensitively**, because case-insensitively it starts eating ordinary words next to a number |
| `ssn` | `\d{3}-\d{2}-\d{4}` |
| `credit_card` | 13–19 digits **plus a Luhn check** |
| `aadhaar` | twelve digits, leading `[2-9]` (an aadhaar never starts 0 or 1) |
| `passport` | **a nearby label**: `passport no:` and then `[A-Z]{1,2}\d{6,8}` |
| `date_of_birth` | **a nearby label**: `dob`, `date of birth`, `born on`, then a date |
| `employee_id` | `EMP-12345`, or the words `employee id/no/number/code` and digits |
| `ip` | dotted quad |
| `phone` | a country code, or bracketed/separated groups, or a bare ten-digit Indian mobile `[6-9]\d{9}` |
| `bank_account` | 9–18 bare digits |

**False positives are the expensive failure here, not misses.** The redacted
question is what the model is actually asked, so a pattern that eats "20 days" or
"Form 16" out of an HR question corrupts the question and produces a worse answer
than leaving the text alone. That single fact explains most of the table:

- **Luhn on cards.** Without it, a 13-to-16-digit pattern eats order numbers,
  invoice references, employee ids and plain amounts.
  `test_a_sixteen_digit_number_that_fails_luhn_is_not_a_card` pins it.
- **Separators or a country code on phones.** An HR corpus is full of bare
  numbers — "20 days", "Form 16", "section 4.2", "10:00 to 16:00" — and a pattern
  loose enough to catch `12345` redacts all of them.
- **Labels on passport and date of birth.** In this corpus `HR2024001` is a policy
  reference and `2024-04-01` is an effective date, not a passport and a birthday.
  Every policy document contains dates. The label is redacted along with the
  value, rather than left behind pointing at a hole.

`test_ordinary_policy_text_is_left_completely_alone` is the test that would catch
a regression here.

### The overlap rule

Patterns genuinely overlap, and picking wrong leaves digits behind. `_resolve`
keeps the **longest** match of any overlapping group, then the earliest, then
insertion order:

```python
ranked = sorted(matches, key=lambda m: (-(m.end - m.start), m.start, _KIND_ORDER[m.kind]))
```

A card written `4111 1111 1111 1111` also contains an aadhaar-shaped twelve-digit
prefix. Redacting the prefix would leave the last group of four sitting in the
text, which is both a disclosure and a lie about having redacted it. Longest-first
makes the span covering the whole identifier win; insertion order settles the case
where an unseparated sixteen-digit card and a generic long account number are the
*same* span, so one reading has to be declared more specific and it is written
down rather than left to whichever pattern ran first.

### A measured fix

The phone pattern's trailing lookahead is now:

```python
(?![\w-])(?!\.\d)
```

The plain `(?![\w.-])` it replaced refused every phone number at the end of a
sentence — which is most of them, because "call me on +44 7700 900123." is how a
person actually writes one. The original intent was to refuse a phone-shaped run
inside a longer dotted number; `(?!\.\d)` keeps exactly that and nothing more.

### What regex PII detection does not catch

It catches *structured identifiers* reliably: a thing with a shape is exactly what
a pattern is good at. It catches the following **not at all**:

- free text about a medical condition — "I need leave because my father has
  dementia";
- a divorce, a bereavement, a financial hardship;
- a disciplinary matter or an allegation against a named colleague;
- **someone else's name**, in any form.

There is no pattern for any of it, and pretending otherwise would be worse than
having nothing, because it invites the assumption that a redacted question is a
safe question. **A redacted question is not a safe question.** It is a question
with the identifiers taken out of it.

Two controls exist because of that gap, and they are not redundant with this one:

1. **The output guard** (section 4) catches a model echoing sensitive detail back
   out of retrieved context.
2. **The escalate path** (section 6) hands a question to a person instead of
   answering it when the situation is one a policy lookup should not be deciding.

This module is the cheap, certain layer underneath them, not the whole of it.

### Where the text would otherwise travel

Redaction happens **once, at the edge** — stage 2, before the three places the
raw text would otherwise end up:

| Destination | Why it matters |
|---|---|
| A provider's servers | Every question is sent to a model. |
| The vector index | A question is embedded to retrieve against it, and the semantic cache keeps the text it is keyed on. |
| The audit log | Deliberately long-lived and admin-readable: the one store whose whole purpose is to still be there in a year. |

`AnswerResult.audited_question` is the one place the rule lives: it returns the
redacted form when redaction changed something and the original otherwise, and
`AuditStore.record` writes what it is handed. The employee's verbatim wording is
what the UI shows them back; the redacted form is what goes in the record.

One honest note on the configuration. `AUDIT_STORES_REDACTED` (default `True`) is
currently **reported** on `/api/config` and not read by `AuditStore.record` —
`audited_question` is unconditional, so the audit log always keeps the redacted
form when `REDACT_PII` produced one. The setting is a declared intent that no
code branch consumes yet; turning it off today changes nothing. `REDACT_PII=0` is
the switch that actually changes what is stored, by making the redacted and the
original question the same string.

The guard also runs **before the cache**, not after, for two reasons: a blocked
question must not be able to poison the cache, and the redacted question is the
right cache key — otherwise two employees asking the same thing with different
phone numbers in it are two misses, and the entry that does get stored holds
somebody's phone number. `test_a_blocked_question_cannot_poison_the_cache` and
`test_personal_data_is_redacted_before_it_reaches_retrieval` pin both halves.

`pii.py` has **no logger**, deliberately. A log line per match would be a log of
where the PII was, which is most of the way to a log of the PII itself. The guard
records the verdict, and it records kinds and counts (`summarise`) rather than
values.

---

## 4. Output filtering

`OutputGuard.check` runs as stage 8, on the finished answer, before it reaches a
browser or the audit log. It takes `grounded` as a parameter — the run's own
verdict on whether the answer rested on retrieved evidence — because only the
caller knows it (`AnswerResult.grounded` is `source_used in ("private_kb",
"web_search")`).

| Finding | Outcome |
|---|---|
| Something key-shaped | **Blocked entirely.** `allowed=False`, logged at ERROR. |
| PII (when `REDACT_PII=1`) | Redacted, `severity="low"`. |
| A figure with no evidence behind it | **Flagged, not blocked.** `severity="medium"`. |
| Redaction emptied the answer | Replaced with `REDACTED_ANSWER`, never `""`. |

**A leaked credential blocks the answer rather than being redacted out of it.**
A key in an answer does not mean the answer needs tidying; it means something in
the context or the prompt chain is leaking a credential. An answer shown with a
hole where the key was still tells the reader there is one to find, and tells
them where. So the whole answer is withheld, the run is logged for someone to
look at, and `result.blocked` is set.
`test_an_answer_containing_a_credential_is_blocked_entirely` pins it.

**The ungrounded-number flag is a flag.** `_POLICY_NUMBER` matches a money
amount, a percentage, or a count of days/weeks/months/years/hours. When the run
had no evidence and the answer states one of those, the verdict gets
`severity="medium"` and a reason — and the answer is still shown. An answer
saying "20 days" with nothing under it may well be right, and whether a
possibly-ungrounded answer is acceptable is a judgement for the human reading the
trace rather than for a regex. What the flag buys is a **searchable audit log**:
a confident number with nothing behind it is the exact failure worth being able
to query for after the fact, and it is invisible unless it is recorded.
`test_an_ungrounded_answer_quoting_a_figure_is_flagged_but_not_blocked` and
`test_the_same_figure_is_not_flagged_when_the_run_had_evidence` pin both sides.

**Never return empty.** A caller handed `""` renders a blank bubble and cannot
tell whether the model said nothing or the guard took it all. `REDACTED_ANSWER`
stands in — but only when redaction is what emptied it, judged at `minimum=1`
rather than 3, so a genuinely short answer ("Yes, 20 days.") is not replaced.
`test_a_short_answer_is_not_mistaken_for_an_emptied_one` is the guard against
over-triggering.

`GuardVerdict.to_dict()` deliberately omits both `text` and `original`. The
verdict is serialised onto the HTTP response and into the audit row, which is
exactly where the unredacted original must not go.
`test_the_verdict_dict_never_carries_the_text_it_judged` pins that.

### The frontend half

A model reply is untrusted input to the page, and the admin reading the audit log
is exactly the session worth stealing. So:

**The word `innerHTML` does not appear in any executable line of this console.**
`grep -rn innerHTML static/ templates/` returns only comments explaining why. Every
dynamic string reaches the page through `textContent` or `createTextNode`, and the
markdown renderer in [`static/js/dom.js`](../static/js/dom.js) *builds*
`<strong>`, `<em>`, `<code>` and `<table>` elements rather than parsing markup
into HTML.

That is a stronger property than escaping, because **there is no escaping step
that can be forgotten**: the unsafe operation is not available. The one place a
namespaced attribute is set is `<use href="#icon">`, which references a symbol
already in the document and takes no user input, and even that goes through
`setAttributeNS` rather than a string of HTML. The moment one `innerHTML`
appears, a model reply containing `<img src=x onerror=...>` runs.

The admin key is held in `sessionStorage`, not `localStorage` — it is a
credential, and session scope means a shared machine does not keep it after the
tab closes. It travels in the `X-Admin-Key` header and never in a query string,
because a URL reaches the history, the proxy log and the `Referer`.

The page's `data-admin` attribute hides controls; it is **not** the gate.
`require_admin` is, on every request.

---

## 5. Authorization

Three roles, because the product has three kinds of user:

| Role | Does |
|---|---|
| `employee` | asks questions, asks for a person |
| `hr` | that, plus uploads policy and works the approval queue |
| `admin` | operates the deployment, and reads the audit log |

The dividing line that matters is the audit log. Every row in it is an employee's
question and the answer they were given, which is HR-sensitive by definition — so
reading it is an **admin** action and not an HR one, even though HR is the
department the product is for.

### The table, and deny-by-default

`POLICY` in `app/services/authorization.py` maps an action to the roles that hold
it. It is dotted and coarse — one entry per capability, not one per endpoint — so
a second route onto the same capability cannot disagree with the first about who
may use it.

```python
POLICY: dict[str, frozenset[Role]] = {
    "kb.read":           frozenset({"employee", "hr", "admin"}),
    "escalate":          frozenset({"employee", "hr", "admin"}),
    ...
    "kb.write":          frozenset({"hr", "admin"}),
    "approval.decide":   frozenset({"hr", "admin"}),
    ...
    "audit.read":        frozenset({"admin"}),
}
```

**An action that is not in the table is denied to everybody, admin included.**

```python
holders = POLICY.get(action)
if holders is None:
    log.warning("Denied unknown action %r; it is not in the policy table", action)
    return False
```

A table that defaulted open would mean adding a tool quietly grants it to every
anonymous visitor — and the whole value of having a table is that granting is a
visible edit to it. Exempting admin would be the same hole with one extra step:
the console and the tool registry both ask this function, so an unknown action
answered "yes, for admin" is an action nobody ever wrote down being offered to
whoever holds the key. `test_an_unknown_action_is_denied_to_every_role_including_admin`
pins it.

### Why a table rather than a second admin key

The worked example: **HR can work the approval queue without gaining the ability
to read every employee's question.** One key and one role cannot express that —
either HR gets the admin key and reads everything, or approvals become an admin
chore. Two keys plus a table expresses it in one line per capability, and
`require_action("approval.decide")` both gates the route and hands it the
principal whose name goes into `decided_by`.

### Constant-time comparison

```python
def check_admin_key(self, presented: str | None) -> bool:
    expected = self._secret("admin_api_key")
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected, presented.strip())
```

A plain `==` on a secret returns as soon as two bytes differ, so the time it
takes leaks the key's length and then its prefix, one byte at a time. Over
enough requests that is a key. `hmac.compare_digest` does not short-circuit.
`check_hr_key` is the same shape for the same reasons.

**An unconfigured role can never be assumed, including by an empty header.**
`not expected` returns False before the comparison, both keys default to `""`
rather than to a placeholder, and `require_admin` refuses with 503 *before it
looks at the header at all*:

```python
if not settings.admin_enabled:
    raise HTTPException(503, "The admin API is not configured on this deployment.")
if not settings.check_admin_key(x_admin_key):
    raise HTTPException(401, "A valid X-Admin-Key header is required.")
```

The failure mode is "admin is off", never "anyone is admin". A reference
implementation shipped `"change-me-in-production"` as the default and compared
with `!=`, which means an unset key plus an absent header compares equal and
every caller is an admin. In production `Settings.validate_admin_surface()` makes
the key mandatory and `create_app` refuses to start without one.

`require_action` answers 403 rather than 404. Pretending the route does not exist
would also hide it from the console, which decides what to *offer* from the same
table. The refusal names the action and the role and nothing else — it cannot
name the key, because it never has it: a `Principal` holds a role and a hash.

### `fingerprint_subject`: a salted hash, not an address

Every caller gets a `Principal`, including an anonymous one. Its `subject` is

```python
subject = fingerprint_subject(
    request.client.host if request.client else None,
    request.headers.get("user-agent"),
    settings=settings,
)
```

— 16 hex characters of sha256, never the address. The subject is written to the
approval queue, the fact store and the audit row, all of which outlive the
request, and it only ever has to answer "is this the same caller as last time?".
A hash answers that; an address answers it no better and is personal data this
product has no reason to keep.

**It is salted with the admin key when one is configured, and the reason is not
hypothetical: an unsalted sha256 of an IPv4 address is trivially reversible.**
There are only about four billion of them, so the whole space can be enumerated
in seconds and the "anonymous" subject in the approval queue becomes the address
again. A per-deployment salt means that table would have to be rebuilt for this
deployment specifically, using a secret already being kept.
`test_a_configured_admin_key_salts_the_fingerprint` pins it.

The user agent is in the material deliberately: the address alone collapses an
office behind one NAT into a single rate-limit bucket, and adding the agent
separates most of them without identifying anybody. Note the honest limit — with
no admin key configured there is no salt, so a deployment with admin off has an
unsalted fingerprint.

---

## 6. Human in the loop

Some things should not happen because a language model decided they should.
Raising a ticket in someone's name, writing to a system of record, mailing a
manager: the agent may propose one, a person approves it.

### Where the agent stops

The flags are per tool, declared once in
[`app/services/toolbox.py`](../app/services/toolbox.py) rather than decided at a
call site:

| Tool | `requires` | `idempotent` | `side_effecting` | `needs_approval` |
|---|---|---|---|---|
| `kb_search` | `kb.read` | yes | no | no |
| `web_search` | `web.search` | yes | no | no |
| `scratchpad_write` | `scratchpad.write` | yes | no | no |
| `remember_fact` | `memory.write` | yes | no | no |
| `raise_ticket` | `ticket.create` | no | yes | **yes** |
| `escalate_to_human` | `escalate` | no | yes | no |

When `needs_approval` is set, `ToolRegistry._dispatch` records the request and
**returns without calling the handler** — the result carries
`needs_approval=True` and an `approval_id`, and the run stops there. The handler
is not reached at all, which is why `raise_ticket`'s handler can be written on
the assumption that a person already said yes.

The order in `call()` matters: validation before approval, so a malformed request
is not queued for a human to read; approval before the idempotency ledger, so an
unapproved call never reserves a fingerprint it will not use; the budget charge
after both, so a refusal is free.

### Fail closed when the queue cannot record it

```python
approval_id = getattr(recorded, "id", recorded)
if approval_id is None:
    # Not recorded means no person will ever see it, so running anyway
    # would be running unapproved. Fail closed.
```

This is the one place the codebase's usual reasoning is **inverted**.
`StateStore.execute`, `AuditStore.record` and `ApprovalStore.request` all return
`None` on a write failure rather than raising, because a degraded feature beats a
500 (see [reliability.md](reliability.md) §8). But a `None` from
`ApprovalStore.request` means **the approval is not in the queue** — no entry for
anyone to decide, and no record that the agent ever asked. The tempting reading is
"carry on without it", and carrying on is precisely the thing the approval existed
to prevent. The same applies when no approval queue is configured at all: the
tool refuses.
`test_an_approval_store_that_records_nothing_refuses_rather_than_running` pins it.

### Single-shot decisions, and the two-people race

The guard is in the UPDATE, not in a read followed by a write:

```sql
UPDATE approvals
   SET status = ?, decided_at = ?, decided_by = ?, note = ?
 WHERE id = ? AND status = 'pending'
```

Two HR staff clicking approve and reject a second apart would otherwise
last-write-wins, and by then the approval may already have had its side effect —
leaving the queue claiming the thing was rejected when it had in fact been done.
With the status in the `WHERE`, `rowcount` says whether *this* caller was the
winner; the loser gets the existing row back unchanged and a warning in the log.
**The first decision is the decision.**
`test_deciding_twice_returns_the_first_decision_unchanged` pins it.

### Expiry is not rejection

`expire_stale()` moves pending rows older than `ttl_hours` (72 by default) to
`expired`. A queue with no expiry becomes a list nobody reads: a month of
undecided rows at the top is indistinguishable from a broken feature, and the one
request that mattered is below them.

`expired` is its own status and must not be read as `rejected`. **Nobody declined
this — nobody looked.** The follow-ups differ: a rejection is an answer to give
the employee, an expiry is a queue that is not being worked.
`ApprovalStore.stats()` therefore reports `oldest_pending_age_hours`, which is the
number worth watching — four pending rows is healthy, an oldest pending row three
days old is not. Expiry runs on read (in `GET /api/approvals`) because there is no
scheduler in this process.

### The asymmetry: `raise_ticket` vs `escalate_to_human`

Both are `side_effecting` and non-idempotent. Only `raise_ticket` needs approval.

> Escalating *is* asking for a human, so requiring a human to approve the request
> for a human is a loop with nobody in it.

`raise_ticket` creates an obligation in someone else's queue in the employee's
name, which is the thing a model should not do unsupervised. `escalate_to_human`
hands the question *to* a person — the worst case is a person reading a question
they did not need to read, which is the same outcome as the approval step itself.
Gating it would mean a distressed employee waiting for an approver before anyone
sees that they asked.

---

## 7. Bring your own key

`/api/chat` is unauthenticated by design, so every question a public URL answers
spends the deployment's own quota. Letting a visitor paste their own Groq or
OpenAI key moves the cost to the person who wanted the answer. Three rules, and
they are the whole design:

1. **The key is used and forgotten.** It arrives in `X-Provider-Key`, builds one
   chat client, and is never written to the audit log, a span attribute, a log
   line, the vector index or the scratchpad. `ProviderCredentials.describe()` is
   what every other layer sees: provider, model, a `prefix…tail` hint and a
   fingerprint. `__repr__` is overridden on top of `SecretStr` so a traceback
   cannot publish it.
2. **The key only ever reaches the chat model.** Not Pinecone, not Tavily, not
   embeddings. `settings_with_credentials` substitutes `llm_provider`, the model
   id and the one secret, and nothing else — a caller's key paying for embeddings
   would also mean their key deciding which index this deployment writes to, and
   the embedding dimension is coupled to an index Pinecone cannot resize.
3. **It is off unless switched on.** `ALLOW_BYOK=0` means every answer is on the
   company's account, which is what an HR department auditing its own copilot
   wants.

A supplied key is validated before it costs a round trip: length between 20 and
256, no whitespace, a known provider prefix, and a model from the per-provider
`ALLOWED_MODELS` allow-list. The allow-list is there because an arbitrary model
id is a way to make this deployment issue requests of an unknown shape and cost
against someone's account. A refusal names the rule, never the value — "a groq
key normally starts with `gsk_`" — because echoing what they pasted is how a key
reaches a log, and forwarding a wrong key instead would produce a provider 401
this layer is not allowed to repeat.

`X-Provider-Key`, not `Authorization`: `Authorization` is what a reverse proxy, an
API gateway or a future auth layer claims, and a caller's provider key sitting in
it would be consumed or logged by whichever arrives first.

### The honest paragraph

A key in a header over HTTPS is encrypted in transit and is not in the URL, the
history, the proxy log or the `Referer` — the same reasoning the admin key
follows. **It is still a secret the visitor is trusting this deployment with for
the length of their session.** It is in this process's memory, inside a pooled
client (bounded at `byok_pool_size`, 8 by default, LRU-evicted), and a reverse
proxy in front of this application *could* log the header. Nothing here can prove
otherwise to the person pasting it. So the README says so plainly and the console
says so beside the input, `ClientPool.clear()` exists so an operator can drop
every held key at once, and the right answer for anyone who does not want to
extend that trust is to run the container themselves.

---

## 8. Residual risks

| Risk | What exists | What does not |
|---|---|---|
| An employee's free-text situation reaching a provider | Structured identifiers redacted at the edge; the audit log keeps the redacted form by default | No detection of a medical condition, a divorce, a disciplinary matter or a third party's name described in prose. A redacted question is not a safe question. |
| Anyone who can reach the URL reading HR policy | Admin key on `/api/audit` and `/api/upload`; HR key on the approval queue | No employee authentication, no per-document access control, no tenant isolation. Every KB chunk is answerable to every caller. |
| Prompt injection via an uploaded document | Upload is admin-gated; generation is retrieval-grounded; the input guard neutralises instruction spans in the *question* | Retrieved document text is **not** passed through `INJECTION_PATTERNS`. A malicious instruction inside an uploaded policy file reaches the generation prompt as context. |
| A model asserting a policy figure it invented | The ungrounded-number flag in the audit row; the extractive path is structurally incapable of it | The flag does not block, and nothing re-checks a stated figure against the retrieved chunks. |
| Audit log disclosure | Admin-only endpoint, constant-time key, `*.db` gitignored, excluded from the Docker image | No encryption at rest, no per-row access control, no key rotation mechanism. Shell or volume access is full access. |
| A caller's BYOK key | Header not query string, never logged or persisted, pool clearable | In-process memory for the session; a reverse proxy could log the header; nothing here can prove to the visitor that it does not. |
| Quota exhaustion / cost abuse | Per-subject token bucket, loop budgets, projected cost ceiling | The bucket is per process, so on a multi-instance platform the effective limit is `instances × rate`. A real limit needs a shared counter in front of this application. |
| A duplicated side effect across instances | SQLite idempotency ledger on a shared volume | The in-memory ledger is per process and is **not** a distributed lock. See [reliability.md](reliability.md) §7. |
| XSS into an admin session | No `innerHTML` anywhere; every dynamic string via `textContent`; markdown renderer builds elements | No CSP header is set. The property rests on the code staying this way, which is a convention rather than an enforced header. |
| An unsalted caller fingerprint | Salted with the admin key when one is configured | With admin off there is no salt, so the subject is a plain truncated sha256 of address plus user agent. |
