# Enterprise HR Policy & Employee Support — Agentic RAG Copilot

An internal HR assistant that answers employee questions from approved company policy, **grades its
own evidence before answering**, and says it does not know rather than inventing an entitlement.

**▶ Live demo — [hr-policy-copilot.vercel.app](https://hr-policy-copilot.vercel.app)**
— runs with **no API keys at all**; paste your own OpenAI or Groq key in Settings for written answers.

```
Python · FastAPI · LangGraph · LangChain · Pinecone · Groq / OpenAI · SQLite · Docker · Vercel
```

---

## What it does

- **Retrieves before it answers** — from an approved private knowledge base, never from the model's
  memory, and every answer carries the documents it rests on.
- **Grades its own evidence** with an LLM, then falls back to query rewriting → web search → an
  honest "I don't know", rather than hallucinating a policy number.
- **Shows its work** — the decision trace, span timings, citations, model used, tokens and projected
  cost come back with every answer and render in the console.
- **Degrades instead of failing** — no vector database falls back to BM25 on disk; no language model
  at all still answers, from the policy text itself.

---

## Generative AI concepts implemented

The point of the project. Every row is built, tested and measured — not described.

### Retrieval-augmented generation → [docs/rag.md](docs/rag.md)

| Concept | How it is implemented here |
|---|---|
| **Chunking strategy** | Recursive splitting at 1000/150, markdown split on headings first; chunk ids keyed on position so an edited chunk updates in place instead of duplicating |
| **Embeddings** | Local `all-MiniLM-L6-v2` (384-d) or OpenAI `text-embedding-3-small` (1536-d), chosen independently of the chat model |
| **Vector search** | Pinecone serverless, cosine, two corpora isolated by namespace, metadata filtering on department and document type |
| **Hybrid search** | BM25 (implemented directly, not a dependency) fused with dense retrieval by **reciprocal rank fusion** — by *rank*, because a cosine score and a BM25 score are not on the same scale |
| **Re-ranking** | Lexical term-overlap and LLM cross-scoring over a wider candidate list |
| **Out-of-domain gating** | A similarity threshold calibrated against a 12-question benchmark, rejecting unrelated questions before they cost a model call |
| **Few-shot prompting** | A four-example evidence-grader prompt behind an A/B switch, deliberately off until measured |
| **RAG vs fine-tuning vs long context** | Analysed as a trade-off table; retrieval chosen for citation, bounded cost and same-day policy updates |
| **Knowledge base sizing** | What breaks at 10 / 100 / 1,000 / 10,000 documents — and which line of code breaks first |

### Agents & orchestration → [docs/agent-patterns.md](docs/agent-patterns.md) · [docs/multi-agent.md](docs/multi-agent.md)

| Concept | How it is implemented here |
|---|---|
| **Graph orchestration** | An **11**-node LangGraph agent: contextualise → route → retrieve → grade → rewrite → web fallback → answer |
| **Structured output** | Pydantic schemas bound by function calling, with recovery from malformed replies and safe defaults (seek evidence, distrust evidence) |
| **LLM-as-judge** | An evidence grader that rejects chunks on the right topic which do not contain the fact asked for |
| **Query rewriting** | A bounded rewrite-and-retry loop when retrieval comes back weak |
| **Query decomposition** | A split decision for questions spanning two policy areas, whose safe default is *don't* |
| **ReAct loop** | Think → act → observe over a tool registry, with repeated-call interception |
| **Tool / function calling** | A registry with JSON-schema parameters, timeouts, retries, authorization, approval and parallel execution |
| **Multi-agent supervisor-worker** | A supervisor delegating to six specialists, each owning real documents, asserted by test to cover the corpus exactly once |
| **Handoff & escalation** | Recorded handoffs, a forced stop at a ceiling, and escalation into a human approval queue |
| **Shared state** | A per-conversation scratchpad workers read each other's findings from |
| **Conversational & long-term memory** | A token-budgeted window with compaction, plus durable facts over a closed key set |

### Production & LLMOps → [docs/production.md](docs/production.md) · [docs/reliability.md](docs/reliability.md)

| Concept | How it is implemented here |
|---|---|
| **Model routing** | Complexity → tier → model, measured against a labelled set, switchable off for a baseline |
| **Semantic caching** | Exact match then embedding similarity, with a high threshold and short TTL because HR answers go stale |
| **Streaming** | Server-sent events over POST, with a terminator so a client can tell a finished stream from a cut one |
| **Cost projection** | Token estimation against published list prices, reported honestly as a projection, never an invoice |
| **Loop budgets** | Six ceilings per run — steps, tool calls, model calls, tokens, wall clock, cost — where exceeding one returns a partial answer *labelled partial* |
| **Retries, timeouts, backoff** | Exponential backoff with jitter, applied only to tools declared idempotent |
| **Idempotency** | An atomic-claim ledger so a retried side effect happens once |
| **Graceful degradation** | A documented ladder from vector-store outage down to answering with no model at all |
| **Observability** | An in-process span tree rendered in the console, plus optional LangSmith with payloads redacted by default |
| **Horizontal scaling** | What is stateless and what is not, stated plainly rather than assumed |

### Safety & governance → [docs/safety.md](docs/safety.md)

| Concept | How it is implemented here |
|---|---|
| **Prompt-injection defence** | Detected, flagged and **neutralised rather than refused** — a refusal teaches an attacker which phrasing tripped it and costs a real user their answer |
| **Input & output guardrails** | Ordered checks on the way in and on the way out, each independently switchable |
| **PII detection & redaction** | 13 identifier types, Luhn-validated cards, applied once at the edge before a question reaches a provider, the vector index or the audit log |
| **Hallucination mitigation** | Citation-carrying answers, an ungrounded-figure flag, and an extractive path **structurally incapable** of inventing a number |
| **Authorization** | Three roles over one deny-by-default permission table |
| **Human-in-the-loop** | Side-effecting tools stop for approval; decisions are single-shot and expiry is a distinct outcome from rejection |
| **Audit logging** | Every answered question with its decisions, citations, model, tokens, cost and latency, behind an admin key |
| **Bring your own key** | A caller's provider key builds one chat client and reaches nothing else — not the audit log, a span, a log line, the index or the web search |

---

## Measured

Every figure below was measured in this repository, not estimated.

| | |
|---|---|
| Knowledge base | **11** documents, **31** chunks, 9 policy areas |
| Out-of-domain retrieval peak | **0.099** raw cosine — **0.078** after the corpus grew |
| Weakest correct-source chunk | **0.171** — so a gate at **0.15** sits in the gap between them |
| Strongest *wrong*-source chunk | **0.521** — which is why an LLM grader is needed as well as a gate |
| Routing + evidence-grading accuracy | **21/21** on a labelled set |
| Complexity classification | 14/14 on a labelled set — was 10/13 before a measured fix |
| Test suite | **652** tests, fully offline, no credentials required |
| Lint | zero findings, enforced as a gate |
| Container image | 2.49 GB, down from 8.93 GB by pinning CPU-only torch |

---

## Architecture

```
Browser console ──► FastAPI ──► AnswerPipeline (11 ordered stages)
                                     |
            input guard -> cache -> routing -> orchestration -> output guard -> audit
                                     |
             +-----------------------+-----------------------+
             v                       v                       v
      LangGraph agent          ReAct + tools         Supervisor + 6 workers
      (11 nodes)               (registry)            (handoff, escalation)
             |                       |                       |
             +---------------- Retrieval --------------------+
                     BM25 + Pinecone -> RRF fusion -> re-rank
```

Layered so each tier imports only from the one above it — `core → rag → agent → services → api` —
and every client is injected, which is why the whole test suite runs with no network and no keys.

---

## Quick start

```bash
pip install -r req.txt
cp .env.example .env            # add keys, or run with none — it still answers
python scripts/ingest_private_kb.py
python run.py                   # console + API on :8000
```

```bash
pytest                          # 652 tests, offline
docker compose up -d --build    # the container, production defaults
```

With **no credentials at all**, retrieval resolves to BM25 over the documents on disk and answers
come from the extractive path. Add a Groq or OpenAI key for written answers, and Pinecone for dense
retrieval.

---

## Documentation

| Document | Covers |
|---|---|
| [rag.md](docs/rag.md) | Chunking, embeddings, vector search, hybrid search, re-ranking, few-shot, RAG vs fine-tuning vs long context, KB sizing |
| [multi-agent.md](docs/multi-agent.md) | Supervisor-worker, handoff and escalation, shared state, when to split |
| [agent-patterns.md](docs/agent-patterns.md) | ReAct, tool registry, error handling, memory, retries, budgets, idempotency, tracing, caching, parallel tools |
| [production.md](docs/production.md) | Model routing, semantic caching, streaming and latency, horizontal scaling, cost projection |
| [safety.md](docs/safety.md) | Input filtering, output filtering and PII, authorization, human-in-the-loop |
| [reliability.md](docs/reliability.md) | Loop budgets, graceful degradation and fallbacks, idempotency |
| [deploy-vercel.md](docs/deploy-vercel.md) | Deploying the demo, and what does not work on serverless |
| [engineering-notes.md](docs/engineering-notes.md) | The long form — full evaluation tables, the API surface, Docker packaging, and the problems solved along the way |

Design decisions, including the ones that were wrong first, are in [CLAUDE.md](CLAUDE.md).

---

## Worth a reviewer's attention

Three bugs this work surfaced. Each failed **silently** rather than raising, and each has a test now:

- The **complexity classifier was inverted** — it sent *"how many days of PTO do I get?"*, the
  commonest HR question there is, to the deep model and the supervisor, while treating a
  contractor's mid-notice parental-leave question as routine. Wrong in both directions at once.
- **Three configuration switches did nothing.** The semantic cache defines `__len__`, so an empty
  cache was *falsy* — the truth test guarding it skipped both the lookup and the store, so the cache
  could never fill and therefore never became truthy.
- **`.vercelignore` follows gitignore semantics where `.dockerignore` does not**, so one copied
  pattern silently excluded the entire knowledge base from the deployment. Every answer came back
  "insufficient evidence" in 3 ms.

---

## License

[MIT](LICENSE)
