"""The API surface that exists today: /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def client(**overrides) -> TestClient:
    base = dict(
        GROQ_API="g", TAVILY_API="t", PINECONE_API="p", PINECONE_INDEX="test-index"
    )
    base.update(overrides)
    return TestClient(create_app(Settings(**base)))


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
