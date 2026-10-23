Part of the [HR Copilot](../README.md) documentation set.

# Retrieval

How a policy document becomes a chunk, how a chunk is found, and why each of
those steps is the shape it is. Every number below is either measured in this
repository — the figures are in the [README](../README.md#retrieval-evaluation)
and in the comments beside the settings they justify — or labelled as a
published list price with its source. Where a section would read better with a
measurement nobody has taken, it says what the measurement would be instead of
inventing one.

---

## 1. Chunking

`app/rag/loaders.py` is the whole of it: four file readers, one splitter, one id
function. 1000 characters per chunk with 150 of overlap, from
`Settings.chunk_size` / `Settings.chunk_overlap` in `app/core/config.py`.

```python
from app.rag.loaders import load_directory, split_documents, chunk_id

documents = load_directory("data/private_kb", origin="hr-docs")
chunks = split_documents(documents)          # per-document separators
chunk_id(chunks[0])                          # 'b1c4...', stable across ingests
```

`split_documents` builds a `RecursiveCharacterTextSplitter` with
`add_start_index=True`. That flag is not a convenience — the offset it writes
into `metadata["start_index"]` is one third of the chunk id, and without it two
chunks of one file cannot be told apart by anything except their text.

### Separators are chosen per format, not per deployment

`PROSE_SEPARATORS` and `MARKDOWN_SEPARATORS` live in `app/core/config.py` and
are deliberately *not* settings — they are structural facts about how a format
splits, not knobs:

```python
PROSE_SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]
MARKDOWN_SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " ", ""]
```

`separators_for(doc_type)` picks between them, so a directory holding markdown
and PDF is one `split_documents` call rather than one per format.

Heading-first splitting means a markdown chunk *begins* at its section title, so
the words "Notice period" are inside the text that gets embedded rather than
two paragraphs above it. The private KB is eleven markdown files organised under
`##` headings, so almost every chunk boundary is a real topic boundary.

**This one is reasoned, not measured.** The comment in `config.py` says the
split "measurably improves" scoring, and no number backing that claim is
recorded anywhere in this repository. The measurement to run is cheap:
`scripts/check_retrieval.py` already reports ungated top-1 source and score for
a fixed question set, so forcing `split_documents(..., separators=PROSE_SEPARATORS)`
and re-running it would give a before/after on the same twelve questions. It has
not been run.

### Chunk ids are keyed on position, not on content

```python
basis = ":".join(str(metadata.get(k, "")) for k in ("origin", "source", "start_index"))
return hashlib.sha1(basis.encode("utf-8")).hexdigest()
```

Position rather than content, so an *edited* chunk updates in place instead of
leaving its previous version behind as a second vector answering the same
questions.

`origin` is in the basis because `source` is only a file name. Once authorised
HR staff can upload — which `/api/upload` allows — `benefits.pdf` from two
departments would collide on one id and the second ingest would silently
overwrite the first. The failure is invisible: the upsert succeeds, the vector
count is right, and one department's benefits policy has simply been replaced by
another's.

### What each metadata field is for

| Field | Written by | What it is for |
|---|---|---|
| `source` | file name | citing (`cite_sources`) and the `[n] source=` line the grader reads |
| `title` | first heading, else first line, else stem | citing, shown in the console's evidence panel |
| `doc_id` | file stem | grouping chunks of one document |
| `doc_type` | extension, via `SUPPORTED_SUFFIXES` | filtering (`build_filter`) and choosing separators |
| `department` | the subdirectory under `data/private_kb/` | filtering; files in the root are `general` |
| `origin` | the corpus (`hr-docs`, `public-web`, an upload) | chunk identity, so two corpora cannot collide |
| `start_index` | the splitter | chunk identity and deterministic tie-breaking |
| `ingested_on` | ingest date | telling a stale vector from a fresh one by eye |

Two of those are filters, three are citations, two are identity. `build_filter`
in `app/rag/retrieval.py` composes the filter clauses, and they compose *with*
the similarity gate rather than replacing it.

---

## 2. Embeddings

`app/rag/clients.py` is the only module that branches on provider. Two are
supported, chosen independently of the chat model by `EMBEDDING_PROVIDER`:

```python
from app.rag.clients import get_embeddings
embedding = get_embeddings()        # cached per (provider, model, key)
```

| | `huggingface` (default) | `openai` |
|---|---|---|
| Model | `all-MiniLM-L6-v2` | `text-embedding-3-small` |
| Vector width | 384-d | 1536-d |
| Normalised | yes, `normalize_embeddings=True` | yes, already unit-normalised |

Both are unit-normalised, which is what keeps cosine on the Pinecone index
equivalent to a dot product and lets one set of measured thresholds mean the
same thing under either.

### The dimension is welded to the index

Pinecone cannot resize an index. `ensure_index` creates it at
`Settings.embedding_dim` and then verifies the dimension of an index that
already exists, because the interesting failure is not "no index" but "the model
changed and the index did not" — which otherwise surfaces much later as an
opaque upsert error.

So switching embedding provider is not a configuration change, it is a
migration: a new `PINECONE_INDEX` name and a fresh ingest.

`KNOWN_EMBEDDING_DIMS` in `app/core/config.py` is the table, and matching is
exact after `_normalise_model` drops the org prefix (`all-MiniLM-L6-v2` and
`sentence-transformers/all-MiniLM-L6-v2` are one model). There is deliberately
no substring fallback — treating anything containing `all-minilm` as 384-d would
be a *guess*, and a wrong guess builds an index that cannot be repaired, so it
costs a full teardown and re-ingest. An unlisted model costs one
`EMBEDDING_DIM=` line instead. `ensure_index` refuses a dimension of 0 (the
value `embedding_dim` takes when the model is unknown and `EMBEDDING_DIM` is
unset) rather than letting Pinecone create an index nothing can use.

### Why `get_embeddings` is cached

`_build_embeddings` carries `@lru_cache(maxsize=4)`. Constructing the local
provider loads a ~90 MB model from disk; without the cache, every retrieval call
that omits an explicit `embedding` reloads it. That is survivable in a script
and ruinous in a request handler that retrieves once per turn. The cache key is
`(provider, model_name, api_key)`, so switching provider inside one process
still builds the right client.

### The torch trade-off

| Provider | Per-query cost | Question leaves the machine | Serverless-deployable | Installed size |
|---|---|---|---|---|
| `huggingface` (`all-MiniLM-L6-v2`) | none — local inference | no | no | torch ~2.5 GB; a 2.49 GB container image, measured on `python:3.13-slim` |
| `openai` (`text-embedding-3-small`) | $0.02 per million input tokens (published list price, recorded in `MODEL_CATALOG` in `app/services/routing.py`) | yes, to OpenAI | yes | no torch; total not measured here |

The 2.49 GB figure is the measured image for this repository, and it depends on
installing torch from PyTorch's CPU index *before* `req.txt`: the default PyPI
wheel pulls the CUDA runtime for a GPU no container has, which measured 8.93 GB
for the same image. The canary is `torch.__version__` ending in `+cpu`.

`sentence-transformers` is far over a Vercel function's size limit, so
`EMBEDDING_PROVIDER=openai` is what makes a serverless deploy possible at all.
On Docker or DigitalOcean either works, and the local one is cheaper and keeps
the question on the machine — which for employee HR questions is the better
default even though it is also the slower one to start.

---

## 3. Vector search

`app/rag/vectorstore.py` parameterises every operation by *both* index and
namespace. `app/rag/retrieval.py` is the same module with the namespace already
chosen, so exactly one place decides which corpus lives where.

### Namespaces, and the typo that says nothing

Two corpora share one index: internal policy in `hr-docs`, the scraped public
article in `public-web`. Both are explicit strings, because Pinecone's default
namespace (`""`) behaves inconsistently between `list` and `delete`.

A namespace mismatch between writer and reader is **silent**. The upsert
succeeds, the vector count is correct, `/health` is green, and every later
retrieval returns `[]` — which the graph reads as "the knowledge base does not
cover this" and answers from web search. Writer and reader both read the same
setting (`Settings.private_namespace`) for exactly that reason.

### `to_relevance_scale`, and the trap it exists for

LangChain's `similarity_score_threshold` search type does not compare against
raw cosine. It compares against `(c + 1) / 2`:

```python
def to_relevance_scale(raw_cosine: float) -> float:
    return (raw_cosine + 1) / 2
```

Passing a raw cosine straight through filters nothing useful: `0.15` would be
interpreted as raw cosine **-0.70**, which admits essentially everything. The
conversion is done in one place so callers — and `retrieve_with_scores`, which
reports the ungated numbers the gate is tuned from — can all think in raw
cosine.

### `USE_CONFIGURED_GATE` is a sentinel because `None` is taken

```python
USE_CONFIGURED_GATE = -1.0    # app/rag/vectorstore.py
```

`get_retriever(score_threshold=None)` means *disable the gate*, which is a
meaningful request. So "use the configured default" needs a different value, and
it is defined in the module that interprets it; `app/rag/retrieval.py`
re-exports the name rather than spelling `-1.0` again. It was briefly written in
both places, which is two chances for one copy to be changed alone — and the
thing that produces is a silently disabled gate.

### Ingest reconciles; it does not only upsert

`prune_orphans` lists the ids in a namespace after each ingest and deletes any
that no longer correspond to a chunk on disk. Without it, a deleted or renamed
policy file leaves vectors behind that keep answering questions — an HR copilot
serving withdrawn policy, which is the worst thing in this failure class because
the answer looks completely normal.

`ingest_upload` passes `prune=False`, and that is not an oversight: pruning
reconciles a namespace against a *directory*, so running it after a single-file
upload would delete the entire rest of the corpus.

### Eventual consistency

Pinecone serverless is eventually consistent. A retrieval fired immediately
after an upsert returns nothing, so `wait_for_vectors` polls
`describe_index_stats` until the expected count is queryable
(`freshness_timeout` 120s, polling every 2s). This is load-bearing in ingest and
in the tests that follow an ingest with a query; do not drop it in a refactor.

`_index_names`, `_attr` and `_index_ready` exist because the Pinecone v7 client
returns index listings and status as either objects or dicts depending on the
call path. That defensiveness is deliberate.

---

## 4. The similarity gate

One gate, and it lives inside Pinecone rather than in Python:
`get_kb_retriever()` applies it through `similarity_score_threshold`.
`retrieve_with_scores()` reports the ungated numbers it is tuned from. There
used to also be a `retrieve_relevant()` that filtered in Python — weaker in two
ways (it re-ranked only the `k` rows that came back rather than gating the
search, and silently lost the `department` / `doc_type` filters its sibling
accepts) — and it is gone, with a comment standing where it was.

### The measured distribution

Measured over a 12-question eval of the private KB:

| Score population | n | min | median | max |
|---|---|---|---|---|
| Chunks from the correct source document | 21 | **0.171** | 0.420 | 0.663 |
| Chunks from a wrong source document | 19 | 0.074 | 0.258 | **0.521** |
| Out-of-domain, best match | 2 | 0.005 | — | **0.099** |

Out-of-domain peaks at 0.099; the weakest correct chunk scores 0.171. A raw
cosine gate at **0.15** sits in that gap: 10/10 in-domain questions kept their
correct-source top-1, and 2/2 out-of-domain questions returned zero documents.

### It is an out-of-domain filter, not a relevance grader

The third row of that table is the one that decides the architecture. A
wrong-document chunk scored as high as **0.521**, which is inside the
correct-document range, so **no threshold can separate right-document from
wrong-document**. Judging whether retrieved text actually answers the question
stays the LLM grader's job. The gate's only job is stopping obvious noise from
reaching it.

### Why both, and what each one catches

| Question | Gate | Grader |
|---|---|---|
| "What is the capital of France?" | rejected — nothing scores above 0.15 | never reached |
| "quokka grooming sabbatical" | **passes** — it reads lexically like leave policy | rejected as irrelevant evidence |
| "What is the mileage reimbursement rate per kilometre?" | passes — the expense policy is the right document | `weak` — that policy never states a rate |

The gate rejects out-of-domain *questions*; the grader rejects irrelevant
*evidence* for in-domain questions. Neither subsumes the other, and removing
either opens a hole.

One optimisation falls out of this: `grade_kb_evidence` in `app/agent/nodes.py`
returns `weak` **without an LLM call** when there are no chunks. Grading an
empty context spends a round trip to be told what the gate already said. This
trips people writing tests — scripting a `weak` grade for an empty retriever
never consumes it.

### A fixed threshold degrades as the corpus grows

More documents means more chances that an unrelated question finds something
above the gate, so the KB growing from 6 documents (15 chunks) to 11 (31 chunks)
was **re-measured rather than assumed**: out-of-domain still peaked at **0.078**,
and all five newly covered areas retrieved their top-1 from the correct
document. `python scripts/check_retrieval.py` is that check. Run it after any
further expansion; it is the thing that tells you whether 0.15 still sits in the
gap.

---

## 5. Hybrid search

Dense and sparse retrieval fail *differently*, and that — not a general belief
that more retrievers are better — is the argument for running both.

`app/rag/lexical.py` implements Okapi BM25 over the knowledge base on disk.
`app/rag/hybrid.py` fuses it with the dense ranking and owns `build_retriever`,
the one retriever factory the rest of the application calls.

```python
from app.rag.hybrid import build_retriever
retriever = build_retriever(settings=settings)   # resolves dense | hybrid | lexical
retriever.invoke("how much notice do I owe?")    # -> list[Document], never raises
```

### BM25 implemented rather than depended on

`rank_bm25` would do the arithmetic. It is forty lines against one more pin —
and, more to the point, the IDF variant is a decision this corpus has an opinion
about.

```python
def idf(self, term: str) -> float:
    df = self.doc_freq.get(term, 0)
    return math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
```

The textbook Okapi form is `log((N - df + 0.5) / (df + 0.5))`, which goes
**negative** as soon as a term appears in more than half the corpus. A negative
IDF is not a rounding problem — it inverts the ranking, because a document then
scores *lower* for containing the query term than for not containing it.

With 31 chunks that is the normal case rather than the corner one. "Policy",
"employee" and "leave" each appear in well over half of this knowledge base, and
those are the words an HR question is made of. Arithmetic from the formula at
N=31: a term in 20 chunks gives `log(11.5 / 20.5) = -0.58` under the textbook
form. A question about leave policy would rank the chunks that never mention
leave first. The `1 +` floors every term's contribution at zero instead, so a
near-universal term simply stops discriminating, and `search` skips any term
whose weight is not positive.

The stopword list is kept short and deliberately free of this corpus's own
vocabulary for the same reason: "policy", "leave", "employee" and "pay" look
like noise and are exactly what an HR question is about. Holding their weight
down is the IDF's job — a term removed from the tokeniser cannot be matched at
all.

### Fusion is by rank, not by score

```python
scores[key] += weight / (k + rank)      # k = 60, rank from 1
```

A cosine similarity of 0.34 and a BM25 score of 7.8 are not the same kind of
number, and no normalisation makes them comparable *across* queries: min-max
over one result set rescales by whatever the best hit happened to score, so the
same chunk gets a different fused score depending on what else came back with
it. Reciprocal rank fusion throws the magnitudes away and keeps the one thing
both lists mean identically — the order. It is also why there is nothing here to
re-tune when the embedding model changes.

`k = 60` is the value from the original RRF paper and the one every
implementation defaults to. It damps the top of each list, which produces the
property the whole thing exists for. Arithmetic from the formula, with
`HYBRID_DENSE_WEIGHT = 0.6`:

| Candidate | Dense rank | Lexical rank | Fused score |
|---|---|---|---|
| found by dense only, first | 1 | — | `0.6/61` = 0.00984 |
| found by both, third in each | 3 | 3 | `0.6/63 + 0.4/63` = 0.01587 |

A chunk both retrievers found at rank 3 outranks a chunk one retriever found at
rank 1. There is no "both agreed" bonus to get wrong — it falls out of the sum.

Weights are `[dense_weight, 1 - dense_weight]`, so the dense list carries 0.6
and the sparse list the remainder. 0.6 rather than 0.5 because the dense side
carries the similarity gate and is the half measured against the labelled set;
BM25 is there for the questions embeddings are weakest at.

De-duplication across the two rankings uses `document_key`, which is `chunk_id`
when the metadata is there — so both sides agree about what "the same chunk" is
without comparing text — and falls back to hashing `source` plus the first 200
characters when a Document has no `start_index` (a Tavily-derived one, a
hand-built one in a test). Without the fallback, `chunk_id` would hash every
chunk of one file to the same value and quietly fuse a whole document into one
entry.

### What BM25 is genuinely better at here

| | Embeddings | BM25 |
|---|---|---|
| "how much notice do I owe?" against a document headed *Notice period* | strong — paraphrase is what it does | depends on the words overlapping |
| "Form 16", "IJP", a form number | **weak** — an embedding of "form 16" sits very near one of "form 12", and cosine has no way to care which digits were asked for | **strong** — it cares about nothing else |
| a question using none of the document's words | finds it | finds nothing |
| a word in most of the corpus | handled by the vector geometry | handled by the IDF, and only because of the `1 +` guard |

Exact terms are the case worth naming: the private KB has a forms-and-requests
document, and "IJP" is an internal acronym that no general-purpose embedding
model has useful geometry for. BM25 is worse at everything involving synonymy,
and it has no notion of paraphrase at all.

### It is also the no-key fallback

The dense retriever needs a Pinecone key, an embedding model and a network round
trip. BM25 over the markdown already sitting in `data/private_kb/` needs none of
those, so `effective_retrieval_mode` resolves `auto` downwards: hybrid when
Pinecone is configured, lexical when it is not. A deployment that loses Pinecone
gives worse answers rather than no answers. `HybridRetriever._side` catches a
failure per side and records it as `tracer.degraded(...)` with the exception's
**type only**, because that span is serialised onto the HTTP response and
written to the audit row.

`build_kb_index` is cached, and the newest mtime under the directory is part of
the cache key — not the directory alone. The failure that guards against is an
*edited* policy continuing to be answered from the text it used to have, which
is the same failure `prune_orphans` prevents on the dense side. A file count
would notice a new document and miss a correction, and corrections are what
editing a policy is for.

---

## 6. Re-ranking

Retrieval ranks by similarity to the *question*, which is not the same thing as
"answers the question". The gate's own measurement is the evidence: a
wrong-source chunk scored **0.521** while the lowest correct chunk scored
**0.171**, so similarity order is demonstrably not answer order.

That is tolerable while the numbers only decide whether anything is in domain at
all. It stops being tolerable at the point where the top four chunks become the
only thing the generator will ever see.

So retrieve wider, then spend a second and cheaper judgement on the order.
`RERANK_CANDIDATES` is 12 against a `default_k` of 4, and `HybridRetriever`
clamps with `self.candidates = max(candidates, k)` — a re-ranker can only
reorder what it is handed, so a candidate list already cut to `k` leaves it
nothing to do.

| Strategy | Cost | Determinism | What it scores on |
|---|---|---|---|
| `lexical` (default) | free, offline | deterministic | 0.60 × question-term coverage + 0.25 × heading-term match + 0.15 × exact adjacent-pair match |
| `llm` | one model call per question | not determinate | `RERANK_PROMPT`, 0–10 per passage, into `RelevanceScores` |
| `none` | free | deterministic | nothing — keeps the fused order, which is already a ranking |

**The lexical weights were chosen to be explainable, not tuned.** No labelled
set was run over them. Coverage leads because it has the most behind it, a
heading match is the cheapest available proxy for "this chunk is about that",
and a phrase match is a bonus rather than a judgement because one adjacent pair
can line up by accident. A number that looked measured and was not would be
worse here than an honest weight.

### The LLM re-ranker falls back to *lexical*, never to the input order

```python
raw = self._ask(query, docs)
if raw is None:
    span.status = "degraded"
    return lexical_rerank(query, docs, k)
```

Returning the order it was handed is indistinguishable from the feature being
switched off, and it would make a provider having a bad afternoon look like a
design decision. Falling back to a lexical re-rank costs nothing and still
produces a judgement.

Three more details in `app/rag/rerank.py` worth knowing:

- `_fit` makes the model's score list exactly as long as the candidate list. It
  comes back the wrong length often enough to plan for: a model that merges two
  passages returns N-1 numbers, and then every passage after the merge silently
  takes its neighbour's score. Short lists are padded *from the lexical score* —
  a judgement rather than a guess — and long ones truncated.
- The budget is charged whether or not the reply was usable, because the round
  trip happened. A budget that only counts successful calls does not bound a
  provider that is failing expensively. If `budget.allows("llm_calls")` is
  already false, nothing is spent and the lexical path runs instead.
- `tiebreak_key` (shared with `app/rag/lexical.py`) gives a total order over
  documents, because an LLM re-rank hands out integers and four candidates
  sharing a 7 is the common case. A tie broken by arrival order makes the top-k
  flap between identical runs, which reads as a retrieval bug and is a sorting
  one.

---

## 7. Few-shot examples for the grader

`app/agent/prompts.py` holds two grader templates. `GRADER_PROMPT` is the one in
use. `GRADER_PROMPT_FEWSHOT` is the same instruction plus four labelled
examples from `GRADER_EXAMPLES`, rendered as `Question / Evidence / Grade`
blocks.

### Four examples, because the decision has four corners

| # | Shape | Label |
|---|---|---|
| 1 | right document, right fact — PTO days from `leave-and-time-off.md` | `good` |
| 2 | right subject, wrong specific — resignation notice asked, offboarding *process* supplied | `weak` |
| 3 | adjacent subject, plausible-looking — taxi home asked, home-office allowance supplied | `weak` |
| 4 | several chunks, one of which has the fact — core hours plus expense payment timing | `good` |

Example 2 is the hard one and the reason the grader exists at all: the evidence
comes from a document a human would call relevant, and it does not contain the
answer. Example 4 is its mirror — irrelevant company does not disqualify the
chunk that does answer.

One of each is enough to show the boundary. More examples mostly add tokens to
every graded question, and the grader runs once or twice per answer.

### Why it is a separate template, not an edit

This is the methodological point of the section.

`GRADER_PROMPT` is what the **0.15 gate** and the **21/21 labelled decision
set** (router 16/16, grader 7/7) were both measured against. Editing it in place
would invalidate both numbers at once, with no way afterwards to tell which
change moved which figure. The gate's calibration depends on the grader only
indirectly — the threshold sits in a gap in a cosine distribution — but the
grader half of the labelled set depends on the prompt entirely, and a repository
that claims 21/21 against a prompt that has since been rewritten is claiming
something it cannot show.

So the few-shot version is a second template behind `FEWSHOT_GRADING`
(`Settings.fewshot_grading`, default `False`), reported on `/health` and in the
console's settings view. It is an **A/B switch, not an improvement already
made.**

### The A/B to run, which has not been run

1. Baseline: the existing labelled grader set (7 cases) plus the router set (16)
   against `GRADER_PROMPT`, recording per-case decisions and not only the ratio.
2. Switch `FEWSHOT_GRADING=1` and re-run the grader half only. The router prompt
   is untouched, so its 16/16 is a control rather than a second variable.
3. Compare on the near misses specifically — "mileage rate not stated in the
   expense policy", "dental waiting period for contractors" — because those are
   where a grader is worth anything. A prompt change that fixes an easy case and
   breaks a near miss is a regression the ratio would report as a draw.
4. Repeat the grader half two or three times. `temperature=0` is not determinism
   on Groq: a genuinely borderline case ("the KB covers employees, the question
   asks about contractors") returned `weak` 8/8 in isolation here and flipped to
   `good` inside a longer suite run. Treat a single-run grader figure as
   approximate.
5. Only then decide. A labelled set of 7 grader cases is too small to move on a
   one-case difference, which is itself an argument for growing the set before
   trusting the A/B.

**Two honest caveats.** The eval script referenced in the comment beside
`GRADER_EXAMPLES` (`scripts/eval_decisions.py`) does not exist in this
repository — the labelled sets were run by hand, and step 1 above means writing
that script first. And `app/agent/decisions.py` currently builds the grader from
`GRADER_PROMPT` unconditionally, so setting `FEWSHOT_GRADING=1` today changes
what `/health` reports and nothing about how a question is graded. Wiring it is
one line in `get_evidence_grader`; it is deliberately not written yet, because
an unmeasured switch that *does* change behaviour is worse than one that is
visibly inert.

---

## 8. RAG against the alternatives

| | Retrieval (this system) | Fine-tuning | Everything in a long context |
|---|---|---|---|
| **How a policy change propagates** | re-run `scripts/ingest_private_kb.py`; `prune_orphans` removes the withdrawn text in the same pass. Live in seconds, and the old vectors are *gone* | a training run, then an evaluation, then a deployment. The old answer stays available until the new weights ship, and partially after | edit the file; live on the next request |
| **Can it cite** | yes — `cite_sources` returns `source` / `title` / `department` per chunk, and the generator is shown `[n] source=<file>` | no. Weights carry no provenance, so a citation would have to be invented, and an invented citation to a real document is worse than none | yes, if the prompt asks for it, but nothing constrains the model to a document it was actually shown |
| **Cost per question** | 4 chunks × ~1000 characters of evidence, bounded by `default_k` whatever the corpus size | inference only — the cheapest per question of the three | the whole corpus, every question |
| **Cost as the corpus grows** | flat. `k` does not grow with the index | flat per question, but every corpus change costs another training run | linear in the corpus, per question, for ever |
| **What it is actually good at** | facts that change, must be cited, and must be removable | tone, format, a task shape the model does not naturally produce | a small corpus read end to end — cross-document reasoning with nothing retrieved away |

### At this size, the long-context argument is real

Eleven documents, 31 chunks. At `CHARS_PER_TOKEN = 4` (the approximation in
`app/services/routing.py`, itself ±20%) and roughly 1000 characters per chunk,
the entire private KB is on the order of 8k tokens. The context windows in this
repository's own model catalog are 131,072 for the Groq `gpt-oss` models and
128,000 for `gpt-4o-mini`. The whole corpus fits more than fifteen times over.

So "just put it all in the prompt" would work today, and it is worth saying so
rather than pretending retrieval is load-bearing for a reason it is not. Three
reasons survive the size argument:

1. **Citation.** An HR answer that cannot be checked against a named document is
   not usable by the person reading it. Retrieval gives the generator a numbered
   list of specific chunks and the UI the same list; long context gives the model
   everything and hopes it attributes correctly.
2. **Bounded cost per question.** Stuffing the corpus makes every question cost
   the corpus. That is affordable at 31 chunks and is the line that breaks first
   at 300 — and it breaks by getting gradually more expensive, which nobody
   notices until a bill.
3. **An ingest is a deployment of data, not of a model.** HR editing
   `leave-and-time-off.md` and running one script is a change an HR team can
   own. A fine-tune is a change only the people who own the pipeline can make,
   and the thing being changed is policy — which moves on HR's schedule, not on
   an ML team's.

The honest summary: retrieval here is chosen for provenance and for the
operational story, not for the context window. At ten times the corpus it is
chosen for cost as well.

---

## 9. Knowledge base sizing

What changes as the corpus grows, and which line in this codebase breaks first
at each step.

### ~10 documents — where it is now

11 documents, 31 chunks, 9 HR areas. Everything above holds, and the gate has
already been re-measured once (at 6 documents, and again at 11).

The one live hazard at this size is self-contradiction. Documents overlap on
purpose — notice periods, PTO payout on termination, expense payment timing and
core hours each appear in more than one file — and retrieval routinely returns
chunks from two documents for one question. Two documents disagreeing produces a
confidently wrong answer that **grading cannot catch**, because each chunk looks
fine alone. Grep the KB for a fact before changing it.

### ~100 documents

- **The gate goes first.** `Settings.relevance_threshold` is a fixed 0.15. The
  out-of-domain peak happened to *fall* between 6 and 11 documents, 0.099 to
  0.078, and that is one re-measurement rather than a trend. Ten times the
  corpus is ten times the chances an unrelated question finds something above
  0.15. Fix: `scripts/check_retrieval.py`, and move the threshold if the gap has
  closed from either side.
- **`default_k = 4` stops being enough.** With a hundred documents a question
  about leave has more than four relevant chunks, and the fourth-best is no
  longer the marginal one. Fix: raise `default_k` and lean harder on
  `RERANK_CANDIDATES` / the re-ranker, which is the cheap half of the change.
- **BM25 rebuild cost becomes visible.** `build_kb_index` loads and splits every
  file on a cache miss, and the newest mtime is part of the cache key, so *any*
  edit rebuilds the whole index on the next request. At 11 files that is
  invisible; at 100 it is a slow first question after every policy edit. Fix:
  build the index in a worker on ingest rather than lazily on the request path.

### ~1,000 documents

- **Metadata filtering stops being optional.** `build_filter(department=...)`
  and `get_kb_retriever(department=...)` exist, but nothing in the chat path
  uses them: `retrieve_kb` in `app/agent/nodes.py` calls
  `retriever.invoke(query)` against a retriever built with no filter. At 1,000
  documents a payroll question searching the whole corpus is paying for — and
  ranking against — 950 documents that cannot answer it. Fix: resolve a
  department (from the principal, from `FactMemory`, or from the question) and
  pass it through `build_retriever`.
- **The LLM grader becomes the bottleneck.** `grade_kb_evidence` renders *all*
  `k` chunks through `format_documents` into one call. Raise `k` to compensate
  for a larger corpus and the evidence block grows linearly, so the grader's
  prompt — on the path of every KB question — grows with it, and
  `Budget.max_tokens` starts being the thing that stops runs. Fix: grade per
  chunk and fan out through `ToolRegistry.call_parallel`, or replace the grader
  with a cross-encoder and keep the LLM for the final judgement only.
- **BM25 in-process stops being free.** `BM25Index.from_documents` holds a
  `Counter` per chunk in memory. Thousands of chunks is still megabytes, so this
  is a build-time problem before it is a memory one — but it is the point at
  which "implemented rather than depended on" should be revisited against a real
  sparse index.

### ~10,000 documents

- **A flat namespace stops working.** Everything lives in one namespace,
  `hr-docs`, and `prune_orphans` pages `index.list(namespace=...)` over *every
  id in it* after each ingest to find orphans. At 10,000 documents that is tens
  of thousands of ids walked on every full ingest. (Uploads already escape this
  — `ingest_upload` passes `prune=False` — but the trade is that uploads never
  reconcile.) Fix: a namespace per department, so reconciliation is scoped to
  the part of the corpus that changed, and `kb_namespace()` becomes a function
  of the document rather than a setting.
- **One `k` for every question stops being defensible.** A question answered by
  one sentence and a question spanning three policies want different amounts of
  evidence. Fix: let the complexity classifier in `app/services/routing.py`
  choose `k` as well as the model — the signal is already computed on every
  question and already reported in the trace.

### Signals that the corpus has outgrown this setup

- `scripts/check_retrieval.py` shows the out-of-domain peak creeping toward 0.15,
  or the weakest correct chunk falling toward it. The gap closing from either
  side is the same problem.
- The grader returns `weak` on questions the corpus demonstrably *does* answer.
  That is the top-`k` no longer containing the right chunk, not a grading fault —
  check it by widening `k` and seeing whether the answer appears.
- Answers start citing one document when a human would cite two. Retrieval is
  filling the slots with near-duplicates from a single file.
- A full ingest's `prune_orphans` time, or the first question after a policy
  edit, becomes something you notice.
- `Budget.report()` shows `exhausted: "tokens"` on ordinary questions. The
  evidence block has outgrown the budget, which means `k` and the grader need
  re-thinking together rather than separately.

---

## Where each piece lives

| Concern | Module |
|---|---|
| Reading files, chunking, chunk ids, evidence rendering | `app/rag/loaders.py` |
| Embeddings, chat model, web search, provider response shapes | `app/rag/clients.py` |
| Pinecone: index, namespaces, gate, upsert, prune | `app/rag/vectorstore.py` |
| The two corpora, namespaces already chosen | `app/rag/retrieval.py` |
| load → chunk → embed → upsert → reconcile | `app/rag/ingest.py` |
| BM25, tokenisation, the on-disk index | `app/rag/lexical.py` |
| RRF, `HybridRetriever`, `build_retriever` | `app/rag/hybrid.py` |
| Candidate re-ordering, three strategies | `app/rag/rerank.py` |
| Grader and re-rank prompts, the few-shot examples | `app/agent/prompts.py` |
| Threshold, chunk size, `k`, retrieval mode, weights | `app/core/config.py` |
