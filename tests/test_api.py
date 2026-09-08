"""The HTTP surface: /health, /api/chat, /api/upload and /api/audit.

Offline like the rest of the suite. `/chat` runs the real graph over the fakes
in conftest, `/upload` monkeypatches the one call that would reach Pinecone, and
the audit database is a file under tmp_path.
"""

from __future__ import annotations

from conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc
from fastapi.testclient import TestClient

from app.agent.graph import build_graph
from app.core.config import Settings
from app.main import create_app
from app.rag.ingest import IngestReport
from app.services.copilot import Copilot

ADMIN = {"X-Admin-Key": "secret-admin-key"}


def client(copilot=None, **overrides) -> TestClient:
    base = dict(
        GROQ_API="g", TAVILY_API="t", PINECONE_API="p", PINECONE_INDEX="test-index"
    )
    base.update(overrides)
    return TestClient(create_app(Settings(**base), copilot=copilot))


def kb_copilot(**llm_kwargs) -> Copilot:
    """A real Copilot over a real graph over fakes -- no network, no keys."""
    defaults = dict(route="kb", grades=["good"], reply="Twenty days.")
    defaults.update(llm_kwargs)
    return Copilot(
        graph=build_graph(
            llm=FakeLLM(**defaults),
            retriever=FakeRetriever([kb_doc()]),
            web_search=FakeWebSearch(),
            verbose=False,
        )
    )


# --- /health -----------------------------------------------------------------


def test_health_reports_ok_when_configured():
    body = client().get("/health").json()
    assert body["status"] == "ok"
    assert body["missing_secrets"] == []
    assert body["pinecone_index"] == "test-index"


def test_health_reports_degraded_and_names_what_is_missing():
    body = client(GROQ_API="").get("/health").json()
    assert body["status"] == "degraded"
    assert body["missing_secrets"] == ["GROQ_API"]


def test_health_never_echoes_a_secret_value():
    """It is exposed to a load balancer, so it may only report names."""
    text = client(GROQ_API="super-secret-value").get("/health").text
    assert "super-secret-value" not in text


def test_health_reports_the_app_identity_and_admin_state():
    body = client(APP_NAME="HR Copilot", APP_ENV="staging").get("/health").json()
    assert body["app"] == "HR Copilot"
    assert body["env"] == "staging"
    assert body["admin_enabled"] is False


def test_admin_enabled_follows_the_key():
    assert client(ADMIN_API_KEY="k").get("/health").json()["admin_enabled"] is True


def test_interactive_docs_are_served_outside_production():
    c = client(APP_ENV="development")
    assert c.get("/docs").status_code == 200
    assert c.get("/openapi.json").status_code == 200


def test_interactive_docs_are_withheld_in_production():
    """They enumerate every route, admin ones included."""
    c = client(APP_ENV="production")
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
    # /health still works -- the load balancer depends on it.
    assert c.get("/health").status_code == 200


def test_health_is_reachable_without_any_credentials():
    """A container must come up and report the problem, not fail to start."""
    response = client(GROQ_API="", TAVILY_API="", PINECONE_API="").get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_building_the_app_touches_neither_network_nor_disk(tmp_path):
    """`app.main` builds an application at import. If that reached Pinecone or
    wrote the audit database, importing the module would do both."""
    db = tmp_path / "audit.db"
    create_app(Settings(PINECONE_API="p", AUDIT_DB_PATH=db))
    assert not db.exists()


# --- /api/chat ---------------------------------------------------------------


def test_chat_answers_and_returns_its_evidence(tmp_path):
    c = client(copilot=kb_copilot(), AUDIT_DB_PATH=tmp_path / "audit.db")
    body = c.post("/api/chat", json={"question": "How many PTO days do I get?"}).json()

    assert body["answer"] == "Twenty days."
    assert body["source_used"] == "private_kb"
    assert body["sources"][0]["source"] == "leave-and-time-off.md"
    assert body["trace"][0] == "Router: kb"
    assert body["audit_id"] == 1


def test_chat_rejects_a_question_too_short_to_route(tmp_path):
    c = client(copilot=kb_copilot(), AUDIT_DB_PATH=tmp_path / "audit.db")
    assert c.post("/api/chat", json={"question": "?"}).status_code == 422


def test_chat_needs_a_question_at_all(tmp_path):
    c = client(copilot=kb_copilot(), AUDIT_DB_PATH=tmp_path / "audit.db")
    assert c.post("/api/chat", json={}).status_code == 422


class BoomCopilot:
    """A copilot whose failure carries text that must never reach a caller."""

    def ask(self, question: str):
        raise RuntimeError("401 from https://api.groq.com key=gsk_live_SECRET")


def test_a_copilot_failure_does_not_leak_the_provider_error(tmp_path):
    """`detail=str(exc)` is how an API key reaches a browser: a provider error
    carries URLs, request ids and sometimes the offending payload."""
    c = client(copilot=BoomCopilot(), AUDIT_DB_PATH=tmp_path / "audit.db")
    response = c.post("/api/chat", json={"question": "How many PTO days?"})

    assert response.status_code == 500
    assert "gsk_live_SECRET" not in response.text
    assert "api.groq.com" not in response.text


def test_chat_is_open_to_employees_without_an_admin_key(tmp_path):
    """The employee-facing route must not be behind the admin gate."""
    c = client(
        copilot=kb_copilot(),
        AUDIT_DB_PATH=tmp_path / "audit.db",
        ADMIN_API_KEY="secret-admin-key",
    )
    assert c.post("/api/chat", json={"question": "PTO days?"}).status_code == 200


def test_an_unwritable_audit_log_still_returns_the_answer(tmp_path):
    """Recording must never cost the employee their answer."""
    unwritable = tmp_path / "audit.db"
    unwritable.mkdir()
    c = client(copilot=kb_copilot(), AUDIT_DB_PATH=unwritable)

    body = c.post("/api/chat", json={"question": "PTO days?"}).json()
    assert body["answer"] == "Twenty days."
    assert body["audit_id"] is None


# --- The admin gate ----------------------------------------------------------


def test_admin_routes_refuse_everything_when_no_key_is_configured(tmp_path):
    """Fail closed. The failure mode must be "admin is off", never "anyone is
    admin" -- which is what `!=` against an empty default produces."""
    c = client(AUDIT_DB_PATH=tmp_path / "audit.db")
    assert c.get("/api/audit").status_code == 503
    assert c.get("/api/audit", headers=ADMIN).status_code == 503
    # An empty presented key must not match an unset configured one.
    assert c.get("/api/audit", headers={"X-Admin-Key": ""}).status_code == 503


def test_admin_routes_reject_a_wrong_key(tmp_path):
    c = client(ADMIN_API_KEY="secret-admin-key", AUDIT_DB_PATH=tmp_path / "audit.db")
    assert c.get("/api/audit", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert c.get("/api/audit").status_code == 401


def test_an_admin_refusal_never_echoes_the_configured_key(tmp_path):
    c = client(ADMIN_API_KEY="secret-admin-key", AUDIT_DB_PATH=tmp_path / "audit.db")
    response = c.get("/api/audit", headers={"X-Admin-Key": "wrong"})
    assert "secret-admin-key" not in response.text


# --- /api/audit --------------------------------------------------------------


def audit_client(tmp_path, **overrides) -> TestClient:
    return client(
        copilot=kb_copilot(),
        ADMIN_API_KEY="secret-admin-key",
        AUDIT_DB_PATH=tmp_path / "audit.db",
        **overrides,
    )


def test_an_answered_question_appears_in_the_audit_log(tmp_path):
    c = audit_client(tmp_path)
    c.post("/api/chat", json={"question": "How many PTO days do I get?"})

    body = c.get("/api/audit", headers=ADMIN).json()
    assert body["total"] == 1
    assert body["entries"][0]["question"] == "How many PTO days do I get?"
    assert body["entries"][0]["source_used"] == "private_kb"
    assert body["entries"][0]["trace"][0] == "Router: kb"


def test_the_audit_log_pages(tmp_path):
    c = audit_client(tmp_path)
    for n in range(3):
        c.post("/api/chat", json={"question": f"question number {n}"})

    body = c.get("/api/audit?limit=2&offset=0", headers=ADMIN).json()
    assert body["total"] == 3
    assert len(body["entries"]) == 2
    assert body["limit"] == 2


def test_the_audit_log_can_be_narrowed_to_one_source(tmp_path):
    c = audit_client(tmp_path)
    c.post("/api/chat", json={"question": "How many PTO days do I get?"})

    assert c.get("/api/audit?source_used=web_search", headers=ADMIN).json()["total"] == 0
    assert c.get("/api/audit?source_used=private_kb", headers=ADMIN).json()["total"] == 1


def test_one_audit_entry_can_be_fetched_by_id(tmp_path):
    c = audit_client(tmp_path)
    entry_id = c.post("/api/chat", json={"question": "PTO days?"}).json()["audit_id"]

    entry = c.get(f"/api/audit/{entry_id}", headers=ADMIN).json()
    assert entry["id"] == entry_id
    assert entry["answer"] == "Twenty days."


def test_a_missing_audit_entry_is_a_404(tmp_path):
    assert audit_client(tmp_path).get("/api/audit/999", headers=ADMIN).status_code == 404


def test_audit_stats_are_not_shadowed_by_the_id_route(tmp_path):
    """`/audit/stats` is declared before `/audit/{entry_id}`; declared after, a
    path parameter would claim "stats" and answer 422."""
    c = audit_client(tmp_path)
    c.post("/api/chat", json={"question": "PTO days?"})

    stats = c.get("/api/audit/stats", headers=ADMIN).json()
    assert stats["total"] == 1
    assert stats["by_source"] == {"private_kb": 1}
    assert stats["grounded_rate"] == 1.0


# --- /api/upload -------------------------------------------------------------


def fake_report(*args, **kwargs) -> IngestReport:
    return IngestReport(
        namespace=kwargs.get("namespace", "hr-docs"),
        documents=1,
        chunks=3,
        vectors_in_namespace=34,
        sources=["policy.md"],
    )


def upload_client(tmp_path, **overrides) -> TestClient:
    return client(
        ADMIN_API_KEY="secret-admin-key",
        UPLOAD_DIR=tmp_path / "uploads",
        AUDIT_DB_PATH=tmp_path / "audit.db",
        **overrides,
    )


def test_an_upload_is_stored_and_indexed(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.ingest_upload", fake_report)
    c = upload_client(tmp_path)

    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("policy.md", b"# Leave\n\nTwenty days.", "text/markdown")},
        data={"department": "People Ops"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["file"] == "policy.md"
    assert body["chunks"] == 3
    # The department is normalised to something a Pinecone filter can hold.
    assert body["department"] == "people-ops"
    assert (tmp_path / "uploads" / "policy.md").exists()


def test_uploading_needs_the_admin_key(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.ingest_upload", fake_report)
    c = upload_client(tmp_path)
    files = {"file": ("policy.md", b"text", "text/markdown")}

    assert c.post("/api/upload", files=files).status_code == 401
    assert not (tmp_path / "uploads" / "policy.md").exists()


def test_an_unsupported_file_type_is_refused_before_it_is_written(tmp_path):
    c = upload_client(tmp_path)
    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert not (tmp_path / "uploads" / "payload.exe").exists()


def test_a_filename_cannot_escape_the_upload_directory(monkeypatch, tmp_path):
    """A multipart filename is attacker-controlled text, not a path."""
    monkeypatch.setattr("app.api.routes.ingest_upload", fake_report)
    c = upload_client(tmp_path)

    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("../../escaped.md", b"# Policy", "text/markdown")},
    )
    assert response.status_code == 201
    assert response.json()["file"] == "escaped.md"
    assert (tmp_path / "uploads" / "escaped.md").exists()
    assert not (tmp_path / "escaped.md").exists()


def test_a_windows_style_path_is_stripped_too(monkeypatch, tmp_path):
    """`Path.name` on Linux keeps backslashes, so both separators are stripped."""
    monkeypatch.setattr("app.api.routes.ingest_upload", fake_report)
    c = upload_client(tmp_path)

    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": (r"C:\\Users\\hr\\policy.md", b"# Policy", "text/markdown")},
    )
    assert response.json()["file"] == "policy.md"


def test_an_upload_over_the_cap_is_refused_and_leaves_nothing_behind(tmp_path):
    """Checked while streaming: reading the whole body first is a way to exhaust
    the process before any limit gets to apply."""
    c = upload_client(tmp_path, MAX_UPLOAD_MB=1)

    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("big.md", b"x" * (2 * 1024 * 1024), "text/markdown")},
    )
    assert response.status_code == 413
    assert not (tmp_path / "uploads" / "big.md").exists()


def test_a_bad_department_is_refused(tmp_path):
    c = upload_client(tmp_path)
    response = c.post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("policy.md", b"# Policy", "text/markdown")},
        data={"department": "../etc"},
    )
    assert response.status_code == 400


def test_a_file_with_no_extractable_text_is_reported_not_indexed(
    monkeypatch, tmp_path
):
    def empty(*args, **kwargs):
        raise ValueError("No extractable text in policy.md.")

    monkeypatch.setattr("app.api.routes.ingest_upload", empty)
    response = upload_client(tmp_path).post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("policy.md", b"", "text/markdown")},
    )
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_an_indexing_failure_does_not_leak_the_provider_error(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("pinecone 403 key=pcsk_live_SECRET")

    monkeypatch.setattr("app.api.routes.ingest_upload", boom)
    response = upload_client(tmp_path).post(
        "/api/upload",
        headers=ADMIN,
        files={"file": ("policy.md", b"# Policy", "text/markdown")},
    )
    assert response.status_code == 502
    assert "pcsk_live_SECRET" not in response.text
