/* =============================================================================
   The operator views: Knowledge, Approvals and Audit.

   All three are gated server-side and only *offered* here. `data-admin` and
   `data-hr` on <body> decide what appears; `require_admin` and
   `require_action` decide what happens. Hiding a control the server would
   refuse anyway is a courtesy, not a boundary.

   The audit table is the one place in this console where every value came out
   of a database somebody else wrote to, so it is also the place where the
   no-innerHTML rule earns its keep most directly.
   ========================================================================== */

import { api } from "./api.js";
import {
  ago,
  clock,
  el,
  fill,
  icon,
  ms,
  num,
  qs,
  show,
  tag,
  titleCase,
  toast,
  usd,
} from "./dom.js";

/* ========================= Knowledge ====================================== */

let chosen = null;

function receipt(report, ok = true) {
  return el("div", { class: `receipt${ok ? "" : " receipt--bad"}` }, [
    icon(ok ? "i-check" : "i-close", 16),
    el("span", { class: "receipt__file", text: report.file }),
    ok
      ? el("span", {
          text:
            `${num(report.chunks)} chunk(s) from ${num(report.documents)} document(s)` +
            ` → ${report.namespace} · ${num(report.vectors_in_namespace)} vectors now`,
        })
      : el("span", { text: report.error }),
    ok ? tag(report.department) : null,
  ]);
}

function pick(file) {
  chosen = file || null;
  const label = qs("#drop-file");
  const button = qs("#do-upload");
  if (label) {
    show(label, Boolean(chosen));
    if (chosen) {
      label.textContent = `${chosen.name} · ${(chosen.size / 1024).toFixed(0)} KB`;
    }
  }
  if (button) button.disabled = !chosen;
}

async function doUpload() {
  if (!chosen) return;
  const button = qs("#do-upload");
  const department = qs("#department").value.trim() || "general";
  button.disabled = true;
  try {
    const report = await api.upload(chosen, department);
    qs("#receipts").prepend(receipt(report, true));
    toast(`Indexed ${report.file}. The answer cache was dropped.`, "ok");
    pick(null);
    qs("#file").value = "";
  } catch (error) {
    qs("#receipts").prepend(
      receipt({ file: chosen.name, error: error.detail || "Upload failed." }, false),
    );
    toast(error.detail || "The upload failed.", "bad");
    button.disabled = false;
  }
}

export function initKnowledge() {
  const drop = qs("#drop");
  const input = qs("#file");
  if (!drop || !input) return;

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => pick(input.files?.[0] || null));

  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
    });
  }
  drop.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) pick(file);
  });

  qs("#do-upload").addEventListener("click", doUpload);
}

/* ========================= Approvals ====================================== */

const RISK_KIND = { low: "good", medium: "weak", high: "error" };

function approvalCard(request, onDecided) {
  const pending = request.status === "pending";
  const noteInput = el("input", {
    class: "input",
    maxlength: "1000",
    placeholder: "Why? (optional, recorded with the decision)",
  });

  const decide = async (decision) => {
    try {
      const body = await api.decide(request.id, {
        decision,
        note: noteInput.value.trim() || null,
      });
      if (!body.changed) {
        toast(
          `Already ${body.request.status} by ${body.request.decided_by || "someone"}.`,
          "warn",
        );
      } else {
        toast(`Request #${request.id} ${decision}.`, "ok");
      }
      onDecided();
    } catch (error) {
      toast(error.detail || "The decision could not be recorded.", "bad");
    }
  };

  return el("div", { class: `approval${pending ? " approval--pending" : ""}` }, [
    el("div", { class: "approval__head" }, [
      el("span", { class: "approval__action", text: request.action }),
      tag(request.status),
      tag(RISK_KIND[request.risk] || "weak", `${request.risk} risk`),
      el("span", { class: "approval__id", text: `#${request.id} · ${ago(request.created_at)}` }),
    ]),
    request.question
      ? el("div", { class: "approval__q", text: request.question })
      : null,
    request.reason
      ? el("div", { class: "approval__why", text: request.reason })
      : null,
    request.conversation_id
      ? el("div", { class: "approval__why" }, [
          "conversation ",
          el("code", { text: request.conversation_id }),
        ])
      : null,
    pending
      ? el("div", { class: "approval__row" }, [
          noteInput,
          el("button", {
            class: "btn btn--sm btn--primary",
            text: "Approve",
            onclick: () => decide("approved"),
          }),
          el("button", {
            class: "btn btn--sm btn--danger",
            text: "Reject",
            onclick: () => decide("rejected"),
          }),
        ])
      : el("div", { class: "approval__why" }, [
          el("span", {
            text:
              `${titleCase(request.status)} by ${request.decided_by || "nobody"}` +
              `${request.decided_at ? ` · ${clock(request.decided_at)}` : ""}`,
          }),
          request.note ? el("span", { text: ` — ${request.note}` }) : null,
        ]),
  ]);
}

export async function loadApprovals({ quiet = false } = {}) {
  const host = qs("#approvals");
  const empty = qs("#approvals-empty");
  if (!host) return;

  try {
    const body = await api.approvals();
    const rows = body.requests || [];
    show(empty, rows.length === 0);
    fill(host, rows.map((row) => approvalCard(row, () => loadApprovals({ quiet: true }))));

    const stats = body.stats || {};
    const pill = qs("#approvals-stats");
    if (pill) {
      pill.textContent =
        `${num(stats.pending ?? 0)} pending · ${num(stats.approved ?? 0)} approved · ` +
        `${num(stats.rejected ?? 0)} rejected · ${num(stats.expired ?? 0)} expired`;
    }
    setBadge(stats.pending ?? 0);
  } catch (error) {
    if (error.status === 403) {
      show(empty, true);
      if (empty) {
        empty.textContent =
          "The approval queue needs an HR or admin key. Add one in Settings.";
      }
      fill(host, []);
      return;
    }
    if (!quiet) toast(error.detail || "The queue could not be read.", "bad");
  }
}

export function setBadge(count) {
  const badge = qs("#approvals-badge");
  if (!badge) return;
  badge.textContent = String(count);
  show(badge, Number(count) > 0);
}

export function initApprovals() {
  qs("#approvals-refresh")?.addEventListener("click", () => loadApprovals());
}

/* =========================== Audit ======================================== */

function flagsFor(entry) {
  const flags = [];
  if (entry.cached) flags.push(tag("cached"));
  if (entry.escalated) flags.push(tag("escalated"));
  if (entry.byok) flags.push(el("span", { class: "tag", text: "own key" }));
  if (entry.stop_reason) flags.push(tag("degraded", "partial"));
  for (const kind of entry.pii_redacted || []) flags.push(tag("pii", kind));
  for (const side of Object.keys(entry.guard_flags || {})) {
    const verdict = entry.guard_flags[side];
    flags.push(tag(verdict.allowed === false ? "blocked" : "weak", `${side}: ${verdict.category}`));
  }
  return flags;
}

function drawer(entry) {
  const existing = qs("#audit-drawer");
  if (existing) existing.remove();

  const close = () => qs("#audit-drawer")?.remove();

  const node = el("aside", { class: "drawer", id: "audit-drawer", role: "dialog" }, [
    el("div", { class: "drawer__head" }, [
      el("h3", { text: `Audit entry #${entry.id}` }),
      el("button", { class: "btn btn--ghost btn--icon", onclick: close, title: "Close" }, [
        icon("i-close", 16),
      ]),
    ]),
    el("div", { class: "drawer__body" }, [
      el("div", {}, [
        el("div", { class: "field__label", text: "Question (as stored)" }),
        el("div", { class: "approval__q", text: entry.question }),
      ]),
      el("div", {}, [
        el("div", { class: "field__label", text: "Answer" }),
        el("div", { class: "approval__q", text: entry.answer }),
      ]),
      el(
        "dl",
        { class: "kv" },
        [
          ["Asked at", entry.asked_at],
          ["Conversation", entry.conversation_id || "—"],
          ["Subject", entry.subject || "—"],
          ["Mode", entry.answer_mode || "—"],
          ["Route", entry.route || "—"],
          ["Source", entry.source_used || "—"],
          ["Grounded", entry.grounded ? "yes" : "no"],
          ["KB grade", entry.kb_grade || "—"],
          ["Web grade", entry.web_grade || "—"],
          ["Chunks", num(entry.kb_chunks)],
          ["Rewrites", num(entry.retry_count)],
          ["Rewritten as", entry.rewritten_query || "—"],
          ["Resolved as", entry.resolved_question || "—"],
          ["Model", entry.model || "—"],
          ["Provider", entry.provider || "—"],
          ["Tier", entry.model_tier || "—"],
          ["Complexity", entry.complexity || "—"],
          ["Tokens", `${num(entry.prompt_tokens)} in / ${num(entry.completion_tokens)} out`],
          ["Projected cost", usd(entry.cost_usd)],
          ["Latency", ms(entry.latency_ms)],
          ["Steps / tools", `${num(entry.steps)} / ${num(entry.tool_calls)}`],
          ["Stopped early", entry.stop_reason || "no"],
          ["Workers", (entry.worker_path || []).join(" → ") || "—"],
        ].flatMap(([label, value]) => [
          el("dt", { text: label }),
          el("dd", { text: String(value) }),
        ]),
      ),
      entry.sources?.length
        ? el("div", {}, [
            el("div", { class: "field__label", text: "Sources" }),
            el(
              "div",
              { class: "source__meta" },
              entry.sources.map((source) =>
                el("span", { class: "tag", text: source.source }),
              ),
            ),
          ])
        : null,
      entry.trace?.length
        ? el("div", {}, [
            el("div", { class: "field__label", text: "Trace" }),
            el(
              "ol",
              { class: "trace" },
              entry.trace.map((step) => el("li", {}, [
                el("span", { class: "trace__text", text: step }),
              ])),
            ),
          ])
        : null,
    ]),
  ]);

  document.body.appendChild(node);
  const onEscape = (event) => {
    if (event.key === "Escape") {
      close();
      document.removeEventListener("keydown", onEscape);
    }
  };
  document.addEventListener("keydown", onEscape);
}

export async function loadAudit({ quiet = false } = {}) {
  const body = qs("#audit-body");
  const empty = qs("#audit-empty");
  if (!body) return;

  const source = qs("#audit-filter")?.value || "";
  try {
    const page = await api.audit({ limit: 100, source_used: source });
    const rows = page.entries || [];
    show(empty, rows.length === 0);

    const total = qs("#audit-total");
    if (total) total.textContent = `${num(page.total)} entries`;

    fill(
      body,
      rows.map((entry) =>
        el(
          "tr",
          {
            dataset: { clickable: "1" },
            onclick: async () => {
              try {
                drawer(await api.auditEntry(entry.id));
              } catch (error) {
                toast(error.detail || "That entry could not be read.", "bad");
              }
            },
          },
          [
            el("td", { text: `#${entry.id}` }),
            el("td", { text: ago(entry.asked_at) }),
            el("td", { class: "wrap", text: entry.question }),
            el("td", {}, [tag(entry.source_used || "unknown")]),
            el("td", { text: entry.answer_mode || "—" }),
            el("td", { text: entry.model || "—" }),
            el("td", { class: "num", text: num(entry.kb_chunks) }),
            el("td", { class: "num", text: entry.latency_ms ?? "—" }),
            el("td", {}, flagsFor(entry)),
          ],
        ),
      ),
    );
  } catch (error) {
    if (error.status === 401 || error.status === 503) {
      show(empty, true);
      if (empty) {
        empty.textContent =
          "The audit log needs an admin key. Add one in Settings.";
      }
      fill(body, []);
      return;
    }
    if (!quiet) toast(error.detail || "The audit log could not be read.", "bad");
  }
}

export function initAudit() {
  qs("#audit-refresh")?.addEventListener("click", () => loadAudit());
  qs("#audit-filter")?.addEventListener("change", () => loadAudit());
}
