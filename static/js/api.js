/* =============================================================================
   Talking to the API, and holding the two kinds of key.

   **Both keys live in sessionStorage, not localStorage.** They are credentials;
   session scope means a shared machine does not keep them after the tab closes.
   They go in headers and never in a query string -- a URL reaches the history,
   the proxy log and the `Referer`.

   **The streaming endpoint is POST, so this cannot use `EventSource`.**
   `EventSource` only issues GET, which would put the question and the whole
   conversation in a URL. So the SSE frames are parsed by hand from a `fetch`
   body reader, which is about thirty lines and keeps an employee's HR question
   out of three logs it has no business being in.
   ========================================================================== */

const KEY_STAFF = "hr-copilot/staff";
const KEY_ROLE = "hr-copilot/staff-role";
const KEY_PROVIDER = "hr-copilot/provider";

/* --- Key storage ---------------------------------------------------------- */

/*
  Every read is wrapped. `sessionStorage` throws outright in a few real
  situations -- a browser with site data blocked, some private modes, an iframe
  with third-party storage partitioned off -- and a console that fails to load
  because it could not read a key it does not have is worse than one that treats
  the key as absent.
*/
function read(key) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key, value) {
  try {
    if (value === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export const staffKey = {
  get: () => read(KEY_STAFF),
  role: () => read(KEY_ROLE) || "admin",
  set(value, role = "admin") {
    write(KEY_ROLE, role);
    return write(KEY_STAFF, value);
  },
  clear() {
    write(KEY_STAFF, null);
    write(KEY_ROLE, null);
  },
  get isSet() {
    return Boolean(read(KEY_STAFF));
  },
};

export const providerKey = {
  get() {
    const raw = read(KEY_PROVIDER);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      // A malformed entry is dropped rather than repaired: it came from an
      // older shape of this console, and guessing at it would send a key to
      // the wrong provider.
      write(KEY_PROVIDER, null);
      return null;
    }
  },
  set(credentials) {
    return write(KEY_PROVIDER, JSON.stringify(credentials));
  },
  clear() {
    write(KEY_PROVIDER, null);
  },
  get isSet() {
    return Boolean(read(KEY_PROVIDER));
  },
};

/* --- Headers -------------------------------------------------------------- */

function headers(extra = {}) {
  const out = { ...extra };
  const staff = staffKey.get();
  if (staff) {
    // The header name follows the role the key was saved under. Sending an HR
    // key as `X-Admin-Key` would be refused, and the refusal would read as a
    // wrong key rather than as the wrong header.
    out[staffKey.role() === "hr" ? "X-HR-Key" : "X-Admin-Key"] = staff;
  }
  const provider = providerKey.get();
  if (provider?.key) {
    out["X-Provider-Key"] = provider.key;
    if (provider.provider) out["X-Provider"] = provider.provider;
    if (provider.model) out["X-Provider-Model"] = provider.model;
  }
  return out;
}

/* --- Errors --------------------------------------------------------------- */

export class ApiError extends Error {
  constructor(status, detail, body = null) {
    super(detail || `Request failed (${status})`);
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

async function detailOf(response) {
  /*
    FastAPI's `detail` is either a string or a list of validation errors. Both
    are rendered to one sentence here, because a toast showing a JSON array of
    pydantic error objects tells the user nothing they can act on.
  */
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return [detail, body];
    if (Array.isArray(detail)) {
      const first = detail[0];
      const where = Array.isArray(first?.loc) ? first.loc.slice(-1)[0] : "";
      return [`${where ? `${where}: ` : ""}${first?.msg || "invalid request"}`, body];
    }
    return [response.statusText || `Request failed (${response.status})`, body];
  } catch {
    return [response.statusText || `Request failed (${response.status})`, null];
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: headers(options.headers || {}),
  });
  if (!response.ok) {
    const [detail, body] = await detailOf(response);
    throw new ApiError(response.status, detail, body);
  }
  if (response.status === 204) return null;
  return response.json();
}

export const get = (path) => request(path);

export const post = (path, payload) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });

/** A multipart POST. No Content-Type: the browser has to set the boundary. */
export const postForm = (path, form) =>
  request(path, { method: "POST", body: form });

/* --- Endpoints ------------------------------------------------------------ */

export const api = {
  health: () => get("/health"),
  config: () => get("/api/config"),
  graph: () => get("/api/graph"),
  tools: () => get("/api/tools"),
  metrics: () => get("/api/metrics"),
  chat: (payload) => post("/api/chat", payload),
  notes: (conversationId) =>
    get(`/api/scratchpad/${encodeURIComponent(conversationId)}`),
  addNote: (payload) => post("/api/scratchpad", payload),
  approvals: (status = "") =>
    get(`/api/approvals${status ? `?status=${encodeURIComponent(status)}` : ""}`),
  decide: (id, payload) => post(`/api/approvals/${id}/decide`, payload),
  invalidateCache: () => post("/api/cache/invalidate", {}),
  audit: (params = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== "" && value !== null && value !== undefined) {
        query.set(key, String(value));
      }
    }
    const suffix = query.toString();
    return get(`/api/audit${suffix ? `?${suffix}` : ""}`);
  },
  auditEntry: (id) => get(`/api/audit/${id}`),
  upload: (file, department) => {
    const form = new FormData();
    form.append("file", file);
    form.append("department", department || "general");
    return postForm("/api/upload", form);
  },
};

/* --- Streaming ------------------------------------------------------------ */

/**
 * POST a question and call `onEvent(name, data)` for each server-sent event.
 *
 * Returns the `result` event's payload, which is the full answer -- so a caller
 * can use this exactly like `api.chat` and treat the events as progress.
 *
 * Resolves rather than rejecting on a server-side refusal: a 429 arrives as an
 * `error` event because by then the response has started and there is no status
 * code left to set. The caller distinguishes them through the event, which is
 * why `onEvent` sees everything.
 */
export async function streamChat(payload, onEvent, { signal } = {}) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    const [detail, body] = await detailOf(response);
    throw new ApiError(response.status, detail, body);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  let sawEnd = false;

  const dispatch = (frame) => {
    let name = "message";
    const data = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith(":")) continue; // a comment, e.g. the heartbeat
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (!data.length) return;
    let parsed;
    try {
      parsed = JSON.parse(data.join("\n"));
    } catch {
      parsed = { raw: data.join("\n") };
    }
    if (name === "result") result = parsed;
    if (name === "end") sawEnd = true;
    onEvent(name, parsed);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Frames are separated by a blank line. Split on both line endings,
    // because a proxy may normalise one and not the other.
    let boundary = buffer.search(/\r?\n\r?\n/);
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + (buffer[boundary] === "\r" ? 4 : 2));
      if (frame.trim()) dispatch(frame);
      boundary = buffer.search(/\r?\n\r?\n/);
    }
  }
  if (buffer.trim()) dispatch(buffer);

  if (!sawEnd) {
    /*
      The body ended without the terminator the server always sends last. That
      means the stream was *cut* -- a serverless function hit its time limit, a
      proxy gave up, the process died -- which is a different thing from an
      answer that finished and must not be shown as a short one.
    */
    throw new ApiError(0, "The connection closed before the answer finished.");
  }
  return result;
}
