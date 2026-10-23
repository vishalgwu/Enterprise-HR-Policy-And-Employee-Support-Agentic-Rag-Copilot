/* =============================================================================
   The Settings view: the two kinds of key, the mode switch, and what the
   deployment resolved to.

   Both keys are held in `sessionStorage` and sent as headers -- see api.js for
   why that store and not the other. Neither is ever put in a URL.

   The caller-supplied provider key is the feature that makes a public demo
   workable: without one, a visitor spends the deployment's quota and is rate
   limited to a handful of questions a minute. With one, they spend their own
   account on whatever model they choose. The copy here says plainly that they
   are trusting this deployment with it for the length of their session, because
   they are.
   ========================================================================== */

import { api, providerKey, staffKey } from "./api.js";
import { el, fill, num, pct, qs, show, titleCase, toast, usd } from "./dom.js";

let config = null;
let onModeChange = () => {};

/* --- Provider key --------------------------------------------------------- */

function renderProviderModels() {
  const select = qs("#provider-model");
  const provider = qs("#provider").value;
  if (!select) return;

  const models = config?.byok?.models || {};
  const options = [el("option", { value: "", text: "Provider default" })];
  const names = provider
    ? models[provider] || []
    : // No provider chosen yet: list everything, labelled, so the field is
      // useful before the provider is known rather than empty until it is.
      Object.entries(models).flatMap(([name, list]) =>
        list.map((model) => `${name}:${model}`),
      );

  for (const model of names) {
    const [, bare] = model.includes(":") ? model.split(":") : [null, model];
    options.push(el("option", { value: bare, text: model }));
  }
  fill(select, options);
}

function renderProviderState() {
  const pill = qs("#byok-state");
  const stored = providerKey.get();
  if (!pill) return;
  if (!stored?.key) {
    pill.textContent = "not set";
    pill.className = "pill";
    return;
  }
  const tail = stored.key.slice(-4);
  pill.textContent = `${stored.provider || "auto"} · …${tail}`;
  pill.className = "pill pill--ok";
}

function saveProvider() {
  const key = qs("#provider-key").value.trim();
  if (!key) {
    toast("Paste a key first.", "warn");
    return;
  }
  const credentials = {
    key,
    provider: qs("#provider").value || "",
    model: qs("#provider-model").value || "",
  };
  if (!providerKey.set(credentials)) {
    toast("This browser will not let the console store the key.", "bad");
    return;
  }
  // Cleared from the field immediately. Leaving a credential sitting in a
  // visible input is how it ends up in a screenshot or a screen share.
  qs("#provider-key").value = "";
  renderProviderState();
  toast("Your key is in use for this tab. Questions are answered on your account.", "ok");
}

function clearProvider() {
  providerKey.clear();
  qs("#provider-key").value = "";
  renderProviderState();
  toast("Key forgotten. Back to this deployment's own model.", "ok");
}

/* --- Staff key ------------------------------------------------------------ */

async function renderRole() {
  const pill = qs("#role-state");
  const host = qs("#permissions");
  try {
    const body = await api.config();
    config = body;
    const role = body.principal?.role || "employee";
    if (pill) {
      pill.textContent = role;
      pill.className = role === "employee" ? "pill" : "pill pill--ok";
    }
    if (host) {
      fill(
        host,
        (body.permissions || []).map((action) =>
          el("span", { class: "tag", text: action }),
        ),
      );
    }
    // The body attributes decide what the rail offers, and /api/config is
    // authoritative -- so a key added here updates the nav without a reload.
    document.body.dataset.role = role;
    document.body.dataset.admin = String(role === "admin");
    document.body.dataset.hr = String(role === "admin" || role === "hr");
    return body;
  } catch {
    if (pill) pill.textContent = "unknown";
    return null;
  }
}

function saveStaff(afterChange) {
  const key = qs("#staff-key").value.trim();
  const role = qs("#staff-role").value;
  if (!key) {
    toast("Paste a key first.", "warn");
    return;
  }
  if (!staffKey.set(key, role)) {
    toast("This browser will not let the console store the key.", "bad");
    return;
  }
  qs("#staff-key").value = "";
  renderRole().then(() => afterChange?.());
  toast(`Saved as ${role}. Operator surfaces are unlocked for this tab.`, "ok");
}

function clearStaff(afterChange) {
  staffKey.clear();
  qs("#staff-key").value = "";
  renderRole().then(() => afterChange?.());
  toast("Staff key forgotten.", "ok");
}

/* --- Mode switch ---------------------------------------------------------- */

const MODE_COPY = {
  graph: {
    title: "Graph",
    body:
      "The 11-node pipeline: contextualise, route, retrieve, grade, maybe " +
      "rewrite, maybe search the web, generate. Deterministic and the cheapest.",
  },
  supervisor: {
    title: "Supervisor",
    body:
      "A supervisor delegates to the specialist whose documents hold each part " +
      "of the question, then composes one answer. For questions spanning two " +
      "policy areas.",
  },
  react: {
    title: "ReAct",
    body:
      "The agent chooses its own tools one step at a time: search, arithmetic, " +
      "a ticket, an escalation. The most capable and the least predictable.",
  },
  extractive: {
    title: "Extractive",
    body:
      "No language model at all. Retrieves, then stitches the most relevant " +
      "sentences. Structurally incapable of inventing a policy number.",
  },
};

function renderModes(current) {
  const host = qs("#mode-cards");
  if (!host) return;
  const modes = config?.agent_modes || Object.keys(MODE_COPY);

  const card = (mode, label, body, selected) =>
    el(
      "button",
      {
        class: `card card--pad${selected ? " card--glow" : ""}`,
        style: {
          textAlign: "left",
          cursor: "pointer",
          borderColor: selected ? "rgba(124,92,255,.5)" : "",
        },
        onclick: () => {
          onModeChange(mode);
          renderModes(mode);
        },
      },
      [
        el("div", {
          style: { fontWeight: "650", fontSize: "14.5px", marginBottom: "6px" },
          text: label,
        }),
        el("div", { style: { fontSize: "12.5px", color: "var(--text-dim)" }, text: body }),
      ],
    );

  fill(host, [
    card(
      null,
      `Deployment default (${config?.agent_mode || "auto"})`,
      "Let the deployment choose. On `auto` that is the graph, or the " +
        "supervisor when a question spans policy areas.",
      !current,
    ),
    ...modes.map((mode) =>
      card(
        mode,
        MODE_COPY[mode]?.title || titleCase(mode),
        MODE_COPY[mode]?.body || "",
        current === mode,
      ),
    ),
  ]);
}

/* --- Resolved configuration ---------------------------------------------- */

function renderConfigTable() {
  const host = qs("#config-kv");
  if (!host || !config) return;

  const rows = [
    ["Application", `${config.app} v${config.version} (${config.env})`],
    ["Agent mode", config.agent_mode],
    [
      "Retrieval",
      `${config.retrieval?.effective} (configured: ${config.retrieval?.configured})`,
    ],
    [
      "Hybrid weights",
      `dense ${config.retrieval?.dense_weight} / sparse ` +
        `${(1 - Number(config.retrieval?.dense_weight || 0)).toFixed(2)}, RRF k=${config.retrieval?.rrf_k}`,
    ],
    [
      "Re-ranking",
      `${config.retrieval?.rerank} over ${config.retrieval?.rerank_candidates} candidates`,
    ],
    ["Chunking", `${config.retrieval?.chunk_size} / ${config.retrieval?.chunk_overlap} overlap`],
    ["Similarity gate", `${config.retrieval?.gate} raw cosine, top-${config.retrieval?.k}`],
    ["Few-shot grading", config.retrieval?.fewshot_grading ? "on" : "off"],
    [
      "Semantic cache",
      config.cache?.enabled
        ? `on · ${config.cache.similarity} similarity · ${config.cache.ttl_seconds}s TTL · ${config.cache.max_entries} entries`
        : "off",
    ],
    [
      "Budgets",
      `${config.budgets?.steps} steps · ${config.budgets?.tool_calls} tools · ` +
        `${config.budgets?.llm_calls} model calls · ${num(config.budgets?.tokens)} tokens · ` +
        `${config.budgets?.seconds}s · ${usd(config.budgets?.cost_usd)}`,
    ],
    ["Handoffs", `${config.budgets?.handoffs} max`],
    [
      "Tools",
      `${config.budgets?.tool_timeout_seconds}s timeout · ${config.budgets?.tool_retries} retries · ` +
        `${config.budgets?.tool_parallelism} in parallel`,
    ],
    [
      "Guards",
      `input ${config.safety?.input_guard ? "on" : "off"} · ` +
        `output ${config.safety?.output_guard ? "on" : "off"} · ` +
        `PII redaction ${config.safety?.pii_redaction ? "on" : "off"} · ` +
        `audit stores ${config.safety?.audit_stores_redacted ? "redacted" : "raw"}`,
    ],
    [
      "Rate limit",
      config.rate_limit?.enabled
        ? `${config.rate_limit.per_minute}/min, burst ${config.rate_limit.burst}, per ${config.rate_limit.scope}`
        : "off",
    ],
    ["Own keys", config.byok?.allowed ? `accepted · pool of ${config.byok.pool_size}` : "not accepted"],
    ["Your role", config.principal?.role],
  ];

  fill(
    host,
    rows.flatMap(([label, value]) => [
      el("dt", { text: label }),
      el("dd", { text: String(value ?? "—") }),
    ]),
  );
}

/* --- Public --------------------------------------------------------------- */

export async function initSettings({ onMode, onRoleChange } = {}) {
  onModeChange = onMode || (() => {});

  if (document.body.dataset.byok !== "true") {
    for (const node of document.querySelectorAll("[data-byok-only]")) {
      show(node, false);
    }
  }

  await renderRole();
  renderProviderModels();
  renderProviderState();
  renderModes(null);
  renderConfigTable();

  qs("#provider")?.addEventListener("change", renderProviderModels);
  qs("#byok-save")?.addEventListener("click", saveProvider);
  qs("#byok-clear")?.addEventListener("click", clearProvider);
  qs("#staff-save")?.addEventListener("click", () => saveStaff(onRoleChange));
  qs("#staff-clear")?.addEventListener("click", () => clearStaff(onRoleChange));
  qs("#staff-role")?.addEventListener("change", () => {
    const stored = staffKey.get();
    if (stored) {
      // Re-saving under the new role, because the role decides which header
      // the key is sent in -- changing the dropdown and not the stored role
      // would send an HR key as X-Admin-Key and read as a wrong key.
      staffKey.set(stored, qs("#staff-role").value);
      renderRole().then(() => onRoleChange?.());
    }
  });
}

export function settingsConfig() {
  return config;
}

export { renderModes, pct };
