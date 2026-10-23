/* =============================================================================
   Bootstrap: the theme, the rail, view switching, and the health poll.

   No framework and no build step -- `CLAUDE.md` says so, the Dockerfile
   depends on it, and the Vercel deployment does too. These are ES modules
   loaded directly by the browser, which is how a console gets to be several
   files without a bundler.

   View switching is a class toggle, deliberately. A static section plus a rail
   button is the whole change needed to add a screen, and the Pipeline page --
   a page of prose written for a reader who cannot check it against the code --
   is the last thing that should acquire a render path.
   ========================================================================== */

import { api, providerKey, staffKey } from "./api.js";
import { el, fill, qs, qsa, show, toast } from "./dom.js";
import { initAgents, renderAgents } from "./agents.js";
import { initApprovals, initAudit, initKnowledge, loadApprovals, loadAudit, setBadge } from "./admin.js";
import { initChat, lastResult, renderInspector, setMode } from "./chat.js";
import { initDashboard, loadDashboard, watchDashboard } from "./dashboard.js";
import { initSettings } from "./settings.js";

const THEME_KEY = "hr-copilot/theme";
const RAIL_KEY = "hr-copilot/rail";

const TITLES = {
  ask: "Ask",
  dashboard: "Dashboard",
  agents: "Agents",
  pipeline: "Pipeline",
  knowledge: "Knowledge base",
  approvals: "Approvals",
  audit: "Audit log",
  settings: "Settings",
};

let current = "ask";

/* --- Theme ---------------------------------------------------------------- */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const use = qs("#theme-icon")?.firstElementChild;
  if (use) use.setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // A browser with storage blocked still gets the theme for this page load.
  }
}

function initTheme() {
  let theme = null;
  try {
    theme = localStorage.getItem(THEME_KEY);
  } catch {
    theme = null;
  }
  if (!theme) {
    theme = window.matchMedia?.("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }
  applyTheme(theme);
  qs("#theme-toggle")?.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
}

/* --- Rail ----------------------------------------------------------------- */

function initRail() {
  const shell = qs("#shell");
  let stored = null;
  try {
    stored = localStorage.getItem(RAIL_KEY);
  } catch {
    stored = null;
  }
  shell.dataset.rail = stored === "mini" ? "mini" : "wide";

  qs("#rail-toggle")?.addEventListener("click", () => {
    const next = shell.dataset.rail === "mini" ? "wide" : "mini";
    shell.dataset.rail = next;
    try {
      localStorage.setItem(RAIL_KEY, next);
    } catch {
      /* not worth reporting */
    }
  });
}

/* --- Role-gated navigation ------------------------------------------------ */

/**
 * Show or hide what the current role may reach.
 *
 * `data-admin-only` and `data-hr-only` are *offers*, not gates --
 * `require_admin` and `require_action` are, on every request. This exists so
 * somebody without a key is not shown three buttons that answer 403, and so a
 * key added in Settings takes effect without a reload.
 */
function applyRole() {
  const isAdmin = document.body.dataset.admin === "true";
  const isHr = isAdmin || document.body.dataset.hr === "true";

  for (const node of qsa("[data-admin-only]")) {
    // A group marked both ways appears for either role; a button marked
    // admin-only inside it stays hidden for HR.
    const alsoHr = node.hasAttribute("data-hr-only");
    show(node, isAdmin || (alsoHr && isHr));
  }
  for (const node of qsa("[data-hr-only]:not([data-admin-only])")) {
    show(node, isHr);
  }

  // If the role just lost access to the open view, fall back to Ask rather
  // than leaving someone looking at a panel full of 403s.
  const hidden = { knowledge: !isHr, approvals: !isHr, audit: !isAdmin };
  if (hidden[current]) switchTo("ask");
}

/* --- Views ---------------------------------------------------------------- */

function switchTo(name) {
  current = name;
  for (const section of qsa(".view")) {
    section.classList.toggle("is-active", section.id === `view-${name}`);
  }
  for (const button of qsa(".rail__btn[data-view]")) {
    const active = button.dataset.view === name;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  const title = qs("#view-title");
  if (title) title.textContent = TITLES[name] || name;

  // Polling only runs while its view is on screen; see `watchDashboard`.
  watchDashboard(name === "dashboard");

  if (name === "agents") renderAgents(lastResult());
  if (name === "approvals") loadApprovals({ quiet: true });
  if (name === "audit") loadAudit({ quiet: true });
  if (name === "ask" && lastResult()) renderInspector(lastResult());

  // The hash is the only navigation state, so a reload and a shared link both
  // land on the same screen. Not pushState: these are panels of one page, and
  // a back button that walks through eight panel switches is worse than one
  // that leaves the page.
  if (window.location.hash !== `#${name}`) {
    window.history.replaceState(null, "", `#${name}`);
  }
}

function initViews() {
  for (const button of qsa(".rail__btn[data-view]")) {
    button.addEventListener("click", () => switchTo(button.dataset.view));
  }
  window.addEventListener("hashchange", () => {
    const name = window.location.hash.replace("#", "");
    if (TITLES[name]) switchTo(name);
  });
  const initial = window.location.hash.replace("#", "");
  switchTo(TITLES[initial] ? initial : "ask");
}

/* --- Health --------------------------------------------------------------- */

function renderHealth(health) {
  const dot = qs("#health-dot");
  const text = qs("#health-text");
  const pill = qs("#health-pill");
  if (!pill) return;

  const missing = health.missing_secrets || [];
  const capabilities = health.capabilities || {};
  const ok = health.status === "ok";

  pill.className = `pill ${ok ? "pill--ok" : "pill--warn"}`;
  if (text) {
    text.textContent = ok
      ? `${capabilities.retrieval_mode || "ready"} · ${health.llm_model || "no model"}`
      : `degraded · ${missing.length} key(s) missing`;
  }
  pill.title = ok
    ? `${health.app} ${health.version} · ${health.env}\n` +
      `retrieval: ${capabilities.retrieval_mode}\n` +
      `model: ${health.llm_model}\n` +
      `embeddings: ${health.embedding_model} (${health.embedding_dim}-d)\n` +
      `tracing: ${health.tracing ? "on" : "off"}` +
      (health.tracing ? ` (${health.tracing_redacted ? "redacted" : "UNREDACTED"})` : "")
    : `Missing: ${missing.join(", ")}\n` +
      `Retrieval has degraded to: ${capabilities.retrieval_mode}`;

  if (dot) dot.style.color = ok ? "var(--ok)" : "var(--warn)";

  // The server is authoritative about the admin surface, so a deployment that
  // gains or loses its key updates the nav on the next poll rather than on a
  // reload.
  document.body.dataset.admin = String(Boolean(capabilities.admin && staffKey.isSet));
  document.body.dataset.serverLlm = String(Boolean(capabilities.server_llm));
  applyRole();

  // The one thing worth interrupting someone about: no model and no key means
  // every answer comes from the extractive path, which is a different product
  // and should not have to be inferred from the answers.
  if (!capabilities.server_llm && !providerKey.isSet && !sessionStorage.getItem("hr-copilot/warned")) {
    toast(
      "No language model is configured here. Answers come straight from the " +
        "policy documents — add your own key in Settings for written answers.",
      "warn",
    );
    try {
      sessionStorage.setItem("hr-copilot/warned", "1");
    } catch {
      /* the toast has already been shown; the flag only stops a repeat */
    }
  }
}

async function pollHealth() {
  try {
    renderHealth(await api.health());
  } catch {
    const text = qs("#health-text");
    const pill = qs("#health-pill");
    if (text) text.textContent = "unreachable";
    if (pill) pill.className = "pill pill--bad";
  }
}

/* --- Key dialog ----------------------------------------------------------- */

function initKeyDialog() {
  const dialog = qs("#key-dialog");
  qs("#key-open")?.addEventListener("click", () => {
    if (dialog?.showModal) dialog.showModal();
    else switchTo("settings");
  });
  qs("#key-form")?.addEventListener("submit", (event) => {
    if (event.submitter?.value === "settings") switchTo("settings");
  });
}

/* --- Inspector ------------------------------------------------------------ */

const INSPECTOR_KEY = "hr-copilot/inspector";

function initInspectorToggle() {
  const ask = qs("#ask");
  if (!ask) return;
  let stored = null;
  try {
    stored = localStorage.getItem(INSPECTOR_KEY);
  } catch {
    stored = null;
  }
  // Defaults to shown. At every width the inspector is the reason this page is
  // worth looking at, so the one state that hides it is the one somebody asked
  // for.
  ask.dataset.inspector = stored === "off" ? "off" : "on";

  qs("#inspector-toggle")?.addEventListener("click", () => {
    const next = ask.dataset.inspector === "off" ? "on" : "off";
    ask.dataset.inspector = next;
    try {
      localStorage.setItem(INSPECTOR_KEY, next);
    } catch {
      /* not worth reporting */
    }
    // Re-rendered on reveal so the entrance animations run now that the panels
    // are on screen; see `stagger` in dom.js.
    if (next === "on" && lastResult()) renderInspector(lastResult());
  });
}

/* --- Mode pill ------------------------------------------------------------ */

const MODE_CYCLE = [null, "graph", "supervisor", "react", "extractive"];

function initModePill() {
  let index = 0;
  qs("#mode-pill")?.addEventListener("click", () => {
    index = (index + 1) % MODE_CYCLE.length;
    setMode(MODE_CYCLE[index]);
    const mode = MODE_CYCLE[index];
    toast(
      mode
        ? `Next question will use the ${mode} orchestration.`
        : "Back to the deployment's own choice of orchestration.",
      "ok",
    );
  });
}

/* --- Approvals badge ------------------------------------------------------ */

/**
 * Keep the rail badge current without opening the view.
 *
 * Only for a caller who can actually read the queue -- otherwise this would be
 * a 403 every half-minute from a page that is not even showing the button.
 */
async function pollApprovals() {
  if (document.body.dataset.hr !== "true" && document.body.dataset.admin !== "true") {
    setBadge(0);
    return;
  }
  try {
    const body = await api.approvals("pending");
    setBadge((body.stats?.pending ?? 0) || 0);
  } catch {
    setBadge(0);
  }
}

/* --- Boot ----------------------------------------------------------------- */

async function boot() {
  initTheme();
  initRail();
  initKeyDialog();
  initInspectorToggle();
  initModePill();
  setMode(null);
  applyRole();

  initViews();
  initDashboard();
  initKnowledge();
  initApprovals();
  initAudit();

  await initChat();
  await initSettings({
    onMode: (mode) => setMode(mode),
    onRoleChange: () => {
      applyRole();
      pollApprovals();
      if (current === "audit") loadAudit({ quiet: true });
      if (current === "approvals") loadApprovals({ quiet: true });
    },
  });
  await initAgents();

  await pollHealth();
  await pollApprovals();
  // Thirty seconds rather than ten: /health is cheap but this page is left open
  // all day, and on a per-invocation platform every poll is billed.
  setInterval(pollHealth, 30000);
  setInterval(pollApprovals, 45000);

  // The dashboard is pre-loaded once so switching to it is instant rather than
  // showing eight empty tiles for a round trip.
  loadDashboard({ quiet: true });
}

boot().catch((error) => {
  // Something in the boot sequence threw, which means part of the console is
  // not wired up. Say so rather than leaving a page that looks fine and does
  // nothing when clicked.
  console.error(error);
  toast("The console did not finish loading. Reload the page.", "bad");
});

// Exported for the browser console, which is a legitimate way to poke at this
// while developing and is why they are namespaced rather than global.
window.HR = { api, switchTo, loadDashboard, lastResult, el, fill };
