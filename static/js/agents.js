/* =============================================================================
   The Agents view: the specialist team, the handoffs, and the tool registry.

   The worker cards and the tool cards are both read from the server rather than
   written here: `/api/graph` serves the worker domains the supervisor actually
   delegates to, and `/api/tools` serves the catalogue the model is actually
   offered, filtered through the same permission table that would refuse a call.
   A hand-written list of either would be a third description of something that
   already has two, and the third one is always the one that goes stale.
   ========================================================================== */

import { api } from "./api.js";
import { clock, el, fill, qs, show, tag, titleCase, toast } from "./dom.js";
import { renderGraph, supervisorPath } from "./graph.js";

const WORKER_ICONS = {
  policy: "🗓️",
  benefits: "🩺",
  payroll: "💸",
  workplace: "⚖️",
  web: "🌐",
  general: "📁",
};

let definition = null;

/* --- Workers -------------------------------------------------------------- */

function workerCard(name, domain, order, finding) {
  const consulted = order > 0;
  return el("div", { class: `worker${consulted ? " is-consulted" : ""}` }, [
    el("div", { class: "worker__head" }, [
      el("span", { class: "worker__icon", text: WORKER_ICONS[name] || "📄" }),
      el("span", { class: "worker__name", text: domain.label || titleCase(name) }),
      consulted
        ? el("span", { class: "worker__order", text: `#${order}` })
        : null,
    ]),
    el("div", { class: "worker__desc", text: domain.description || "" }),
    el(
      "div",
      { class: "worker__docs" },
      (domain.documents || []).map((doc) => el("span", { class: "tag", text: doc })),
    ),
    finding
      ? el("div", { class: "worker__finding" }, [
          el("div", { style: { marginBottom: "6px" } }, [
            tag(finding.covered ? "good" : "weak", finding.covered ? "covered" : "no coverage"),
            el("span", {
              class: "tag",
              text: `${finding.chunks ?? 0} chunk(s)`,
              style: { marginLeft: "5px" },
            }),
          ]),
          el("div", { text: finding.text || "(reported nothing)" }),
        ])
      : null,
  ]);
}

function renderWorkers(result) {
  const host = qs("#workers");
  if (!host || !definition) return;

  const path = result?.worker_path || [];
  const findings = new Map(
    (result?.findings || []).map((finding) => [finding.worker, finding]),
  );

  fill(
    host,
    Object.entries(definition.supervisor.workers || {}).map(([name, domain]) =>
      workerCard(name, domain, path.indexOf(name) + 1, findings.get(name)),
    ),
  );
}

/* --- Handoffs ------------------------------------------------------------- */

function renderHandoffs(result) {
  const host = qs("#handoffs");
  const empty = qs("#handoffs-empty");
  const hops = result?.handoffs || [];
  if (!host) return;

  show(empty, hops.length === 0);
  fill(
    host,
    hops.map((hop, index) =>
      el(
        "div",
        { class: `hop${hop.to === "escalate" ? " hop--escalate" : ""}` },
        [
          el("div", { class: "hop__rail" }, [
            el("div", { class: "hop__dot", text: String(index + 1) }),
          ]),
          el("div", { class: "hop__body" }, [
            el("div", { class: "hop__title" }, [
              el("code", { text: hop.from || "supervisor" }),
              " → ",
              el("code", { text: hop.to || "?" }),
            ]),
            hop.reason ? el("div", { class: "hop__why", text: hop.reason }) : null,
            hop.at
              ? el("div", { class: "hop__why", text: clock(hop.at) })
              : null,
          ]),
        ],
      ),
    ),
  );
}

/* --- Tools ---------------------------------------------------------------- */

const FLAG_LABELS = [
  ["side_effecting", "side-effecting", "warn"],
  ["needs_approval", "needs approval", "warn"],
  ["idempotent", "idempotent", "ok"],
];

function toolCard(tool) {
  const flags = [];
  for (const [key, label, kind] of FLAG_LABELS) {
    if (tool[key]) flags.push(tag(kind === "ok" ? "good" : "weak", label));
  }
  flags.push(el("span", { class: "tag", text: tool.cost_hint }));
  flags.push(el("span", { class: "tag", text: `${tool.timeout_seconds}s timeout` }));
  if (tool.retries) {
    flags.push(el("span", { class: "tag", text: `${tool.retries} retries` }));
  }
  flags.push(el("span", { class: "tag", text: tool.requires }));

  const properties = Object.keys(tool.parameters?.properties || {});
  return el("div", { class: "tool" }, [
    el("div", { class: "tool__name", text: tool.name }),
    el("div", { class: "tool__desc", text: tool.description }),
    el("div", { class: "tool__flags" }, flags),
    properties.length
      ? el("div", { class: "tool__args", text: `(${properties.join(", ")})` })
      : null,
  ]);
}

async function loadTools() {
  const host = qs("#tools");
  if (!host) return;
  try {
    const body = await api.tools();
    fill(host, (body.tools || []).map(toolCard));
  } catch (error) {
    fill(
      host,
      el("p", {
        class: "empty",
        text: error.detail || "The tool catalogue could not be read.",
      }),
    );
  }
}

/* --- Public --------------------------------------------------------------- */

export function renderAgents(result) {
  if (!definition) return;
  const nodes = definition.supervisor.nodes || [];
  const path = result?.answer_mode === "supervisor"
    ? supervisorPath(result.worker_path, result.escalated, nodes)
    : [];
  renderGraph(qs("#graph-supervisor"), definition.supervisor, path);
  renderWorkers(result);
  renderHandoffs(result);

  const pill = qs("#supervisor-mode");
  if (pill) {
    pill.textContent = result?.answer_mode === "supervisor"
      ? `last run · ${(result.worker_path || []).length} worker(s)`
      : "compiled";
  }
}

export async function initAgents() {
  try {
    definition = await api.graph();
  } catch (error) {
    toast(error.detail || "The agent graph could not be read.", "bad");
    return;
  }
  renderAgents(null);
  loadTools();
}
