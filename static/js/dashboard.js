/* =============================================================================
   The Dashboard view.

   Everything here comes from `/api/metrics`, which is open, aggregate-only, and
   cannot emit a question or an answer. That is the line worth keeping: a
   deployment's totals say nothing about anybody, and an audit row says
   everything about one person. The dashboard is for the first.

   **Every dollar figure on this page is a projection**, not an invoice: an
   estimated token count (characters/4) times a published list price. The tile
   says so, because a number that looks like accounting and is not is worse than
   no number.
   ========================================================================== */

import { api } from "./api.js";
import { bars, donut, latency, legend } from "./charts.js";
import {
  ago,
  el,
  fill,
  ms,
  num,
  pct,
  qs,
  show,
  titleCase,
  toast,
  tokens,
  usd,
} from "./dom.js";

let lastFetched = null;
let timer = null;

/* --- Tiles ---------------------------------------------------------------- */

function tile(label, value, note = "", flavour = "") {
  const classes = ["stat", "b3"];
  if (flavour) classes.push(`stat--${flavour}`);
  return el("div", { class: classes.join(" ") }, [
    el("span", { class: "stat__label", text: label }),
    el("span", { class: "stat__value" }, [value]),
    el("span", { class: "stat__note", text: note }),
  ]);
}

/** A big number with a small unit, so "4.1s" reads as one figure. */
function figure(main, unit = "") {
  return el("span", {}, [main, unit ? el("span", { class: "unit", text: unit }) : null]);
}

function renderTiles(metrics) {
  const answers = metrics.answers || {};
  const cache = metrics.cache || {};
  const host = qs("#tiles");
  if (!host) return;

  const grounded = answers.grounded_rate ?? 0;
  const groundedFlavour = grounded >= 0.75 ? "accent" : grounded >= 0.5 ? "warm" : "warm";

  fill(host, [
    tile(
      "Questions answered",
      figure(num(answers.total ?? 0)),
      answers.total
        ? `${num(answers.rewritten ?? 0)} needed a rewrite`
        : "ask something to populate this",
      "accent",
    ),
    tile(
      "Grounded in evidence",
      figure(pct(grounded, 0)),
      "the share resting on retrieved policy rather than an admission",
      groundedFlavour,
    ),
    tile(
      "Median latency",
      figure(ms(answers.latency_p50_ms)),
      `p95 ${ms(answers.latency_p95_ms)} · p99 ${ms(answers.latency_p99_ms)}`,
      "cool",
    ),
    tile(
      "Cache hit rate",
      figure(pct(answers.cache_hit_rate ?? 0, 0)),
      cache.available
        ? `${num(cache.entries ?? 0)} entries · ${cache.mode || "—"} matching`
        : "cache unavailable",
      "accent",
    ),
    tile(
      "Projected spend",
      figure(usd(answers.cost_usd ?? 0)),
      `${usd(answers.cost_per_answer_usd ?? 0)} per answer · estimated, not billed`,
      "warm",
    ),
    tile(
      "Tokens",
      figure(tokens((answers.prompt_tokens ?? 0) + (answers.completion_tokens ?? 0))),
      `${tokens(answers.prompt_tokens ?? 0)} in · ${tokens(answers.completion_tokens ?? 0)} out`,
      "cool",
    ),
    tile(
      "Escalated to a person",
      figure(num(answers.escalated ?? 0)),
      `${pct(answers.escalation_rate ?? 0, 0)} of answers`,
      "warm",
    ),
    tile(
      "Guardrail flags",
      figure(num(answers.guardrail_flagged ?? 0)),
      `${num(answers.pii_redacted_answers ?? 0)} answers had personal data removed`,
      "warm",
    ),
  ]);
}

/* --- Capabilities --------------------------------------------------------- */

const CAP_LABELS = {
  vector_store: "Vector store",
  web_search: "Web search",
  server_llm: "Server model",
  server_embeddings: "Server embeddings",
  retrieval_mode: "Retrieval",
  byok: "Bring your own key",
  semantic_cache: "Semantic cache",
  model_routing: "Model routing",
  input_guard: "Input guard",
  output_guard: "Output guard",
  pii_redaction: "PII redaction",
  approvals: "Approval queue",
  span_tracing: "Span tracing",
  agent_mode: "Agent mode",
  admin: "Admin surface",
  hr: "HR surface",
  public_demo: "Public demo",
};

function renderCaps(capabilities) {
  const host = qs("#caps");
  if (!host) return;
  fill(
    host,
    Object.entries(capabilities || {}).map(([key, value]) => {
      const isBool = typeof value === "boolean";
      const on = isBool ? value : Boolean(value);
      return el("div", { class: `cap ${on ? "is-on" : "is-off"}` }, [
        el("span", { class: "cap__dot" }),
        el("span", { class: "cap__name", text: CAP_LABELS[key] || titleCase(key) }),
        el("span", {
          class: "cap__value",
          text: isBool ? (value ? "on" : "off") : String(value),
        }),
      ]);
    }),
  );
}

/* --- Machinery ------------------------------------------------------------ */

function panel(title, rows, note = "") {
  return el("div", { class: "card" }, [
    el("div", { class: "card__head" }, [el("h3", { text: title })]),
    el("div", { class: "card__body" }, [
      el(
        "dl",
        { class: "kv" },
        rows.flatMap(([label, value]) => [
          el("dt", { text: label }),
          el("dd", { text: String(value) }),
        ]),
      ),
      note ? el("p", { class: "field__hint", style: { marginTop: "12px" }, text: note }) : null,
    ]),
  ]);
}

function renderMachinery(metrics) {
  const host = qs("#machinery");
  if (!host) return;
  const cache = metrics.cache || {};
  const limiter = metrics.rate_limit || {};
  const pool = metrics.byok_pool || {};
  const approvals = metrics.approvals || {};
  const pad = metrics.scratchpad || {};
  const ledger = metrics.idempotency || {};

  fill(host, [
    panel(
      "Semantic cache",
      [
        ["Mode", cache.mode ?? "—"],
        ["Entries", `${num(cache.entries ?? 0)} / ${num(cache.max_entries ?? 0)}`],
        ["Hits", `${num(cache.hits ?? 0)} (${num(cache.exact_hits ?? 0)} exact)`],
        ["Hit rate", pct(cache.hit_rate ?? 0, 1)],
        ["Threshold", cache.threshold ?? "—"],
        ["TTL", `${num(cache.ttl_seconds ?? 0)}s`],
        ["Evictions", num(cache.evictions ?? 0)],
        ["Embedding failures", num(cache.embedding_failures ?? 0)],
      ],
      "A high threshold and a short TTL on purpose: two questions that differ " +
        "by one word can have different correct answers, and a stale HR answer " +
        "is a wrong answer with a citation attached.",
    ),
    panel(
      "Rate limit",
      [
        ["Enabled", limiter.enabled ? "yes" : "no"],
        ["Allowance", `${num(limiter.per_minute ?? 0)}/min, burst ${num(limiter.burst ?? 0)}`],
        ["Subjects tracked", num(limiter.subjects ?? 0)],
        ["Allowed", num(limiter.allowed ?? 0)],
        ["Refused", num(limiter.refused ?? 0)],
        ["Scope", limiter.scope ?? "—"],
      ],
      "It bounds the deployment's own key; a caller spending their own is not " +
        "limited. Scope is one process, so on a platform running several " +
        "instances the effective limit is instances × rate.",
    ),
    panel(
      "Caller keys",
      [
        ["Pooled clients", `${num(pool.size ?? 0)} / ${num(pool.max_size ?? 0)}`],
        ["Caller-supplied", num(pool.byok_entries ?? 0)],
        ["Reuse rate", pct(pool.hit_rate ?? 0, 1)],
        ["Evictions", num(pool.evictions ?? 0)],
      ],
      "The pool holds compiled clients, never keys beyond the life of the " +
        "entry that built them.",
    ),
    panel(
      "Human in the loop",
      [
        ["Pending", num(approvals.pending ?? 0)],
        ["Approved", num(approvals.approved ?? 0)],
        ["Rejected", num(approvals.rejected ?? 0)],
        ["Expired", num(approvals.expired ?? 0)],
        [
          "Oldest pending",
          approvals.oldest_pending_age_hours
            ? `${Number(approvals.oldest_pending_age_hours).toFixed(1)}h`
            : "—",
        ],
        ["Scratchpad notes", num(pad.notes ?? 0)],
        ["Idempotency keys", num(ledger.entries ?? ledger.live ?? 0)],
      ],
      "An expired request is a different outcome from a rejected one and is " +
        "not conflated with it.",
    ),
  ]);
}

/* --- Models --------------------------------------------------------------- */

function renderModels(metrics) {
  const body = qs("#model-body");
  const empty = qs("#model-empty");
  const byModel = metrics.answers?.by_model;
  if (!body) return;

  if (!byModel) {
    // Withheld rather than empty: per-model spend needs `metrics.read`.
    fill(body, []);
    show(empty, true);
    return;
  }
  show(empty, false);
  const rows = Object.entries(byModel).sort((a, b) => b[1].answers - a[1].answers);
  fill(
    body,
    rows.length
      ? rows.map(([model, row]) =>
          el("tr", {}, [
            el("td", { text: model }),
            el("td", { class: "num", text: num(row.answers) }),
            el("td", { class: "num", text: tokens(row.tokens) }),
            el("td", { class: "num", text: usd(row.cost_usd) }),
          ]),
        )
      : [el("tr", {}, [el("td", { colspan: "4", class: "empty", text: "No answers yet." })])],
  );
}

/* --- Load ----------------------------------------------------------------- */

export async function loadDashboard({ quiet = false } = {}) {
  try {
    const metrics = await api.metrics();
    lastFetched = new Date().toISOString();

    renderTiles(metrics);
    renderCaps(metrics.capabilities);
    renderMachinery(metrics);
    renderModels(metrics);

    const sourceRows = donut(qs("#chart-source"), metrics.answers?.by_source || {}, {
      label: "answers",
    });
    legend(qs("#legend-source"), sourceRows.map((row) => ({ ...row, name: titleCase(row.name) })));

    const modeRows = bars(qs("#chart-mode"), metrics.answers?.by_mode || {});
    legend(qs("#legend-mode"), modeRows.map((row) => ({ ...row, name: titleCase(row.name) })));

    latency(qs("#chart-latency"), metrics.answers || {});

    const age = qs("#metrics-age");
    if (age) {
      age.textContent = `${num(metrics.answers?.latency_samples ?? 0)} timed · read ${ago(lastFetched)}`;
    }
  } catch (error) {
    if (!quiet) toast(error.detail || "Metrics could not be read.", "bad");
  }
}

export function initDashboard() {
  qs("#metrics-refresh")?.addEventListener("click", () => loadDashboard());
}

/**
 * Poll while the dashboard is the visible view, and stop when it is not.
 *
 * Stopping matters more than polling: a dashboard left open in a background tab
 * otherwise asks a serverless deployment for metrics every ten seconds for the
 * rest of the day, which on a per-invocation platform is a bill for a page
 * nobody is looking at.
 */
export function watchDashboard(active) {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (active) {
    loadDashboard({ quiet: true });
    timer = setInterval(() => loadDashboard({ quiet: true }), 15000);
  }
}
