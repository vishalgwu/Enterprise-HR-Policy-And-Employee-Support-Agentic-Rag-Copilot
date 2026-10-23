"""The HTTP surface the production layers added.

`tests/test_api.py` covers the original endpoints -- health, chat, upload,
audit -- and is left alone. This covers what was added around them: the
introspection endpoints, the metrics aggregate, streaming, bring-your-own-key,
the rate limiter, the scratchpad and the approval queue.

The dividing line worth asserting here is **what each endpoint is allowed to
return**. `/api/metrics` is open and must never be able to emit a question or
an answer; `/api/audit` is admin-gated because it emits exactly those. Several
tests below exist only to pin that difference, because it is the one that gets
eroded by a convenient addition.

Entirely offline. Every store lands in `tmp_path`.
"""

from __future__ import annotations

import json
from typing import Any

from conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc, settings_for
from fastapi.testclient import TestClient

from app.agent.graph import build_graph
from app.main import create_app
from app.services.copilot import Copilot

ADMIN = {"X-Admin-Key": "admin-key"}
HR = {"X-HR-Key": "hr-key"}

QUESTION = "How many days of paid time off do full-time employees get?"


def app_for(tmp_path: Any, **overrides: Any) -> TestClient:
    """A whole application over a graph of fakes and temporary databases."""
    base: dict[str, Any] = dict(
        ADMIN_API_KEY="admin-key",
        HR_API_KEY="hr-key",
        AUDIT_DB_PATH=tmp_path / "audit.db",
        STATE_DB_PATH=tmp_path / "state.db",
        RATE_LIMIT_ENABLED=False,
    )
    base.update(overrides)
    settings = settings_for(**base)
    copilot = Copilot(
        graph=build_graph(
            llm=FakeLLM(route="kb", grades=["good"], reply="Twenty days of PTO."),
            retriever=FakeRetriever([kb_doc()]),
            web_search=FakeWebSearch(),
            verbose=False,
            settings=settings,
        ),
        settings=settings,
    )
    return TestClient(create_app(settings, copilot=copilot))


def sse_events(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, Any]]:
    """Drive the streaming endpoint and return its (event, data) pairs."""
    collected: list[tuple[str, Any]] = []
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        name = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                collected.append((name, json.loads(line[6:])))
    return collected


# --- Introspection -----------------------------------------------------------


def test_health_reports_what_the_deployment_can_actually_do(tmp_path):
    body = app_for(tmp_path).get("/health").json()

    capabilities = body["capabilities"]
    assert capabilities["retrieval_mode"] in ("dense", "hybrid", "lexical")
    assert capabilities["admin"] is True
    assert capabilities["hr"] is True
    assert "pii_redaction" in capabilities


def test_config_describes_the_knobs_and_the_callers_permissions(tmp_path):
    client = app_for(tmp_path)
    body = client.get("/api/config").json()

    assert body["agent_modes"] == ["graph", "react", "supervisor", "extractive"]
    assert body["retrieval"]["gate"] == 0.15
    assert body["budgets"]["steps"] > 0
    assert body["models"], "the console prices a run from this catalogue"
    # An anonymous caller is an employee with real, small permissions -- not an
    # absence of a principal.
    assert body["principal"]["role"] == "employee"
    assert "kb.read" in body["permissions"]
    assert "audit.read" not in body["permissions"]

    as_admin = client.get("/api/config", headers=ADMIN).json()
    assert as_admin["principal"]["role"] == "admin"
    assert "audit.read" in as_admin["permissions"]


def test_config_never_returns_a_secret(tmp_path):
    body = app_for(tmp_path).get("/api/config", headers=ADMIN).text
    assert "admin-key" not in body
    assert "hr-key" not in body
    assert "test-pinecone" not in body


def test_the_graph_endpoint_serves_the_compiled_edge_set(tmp_path):
    """The console draws from this, so it must be the real wiring."""
    body = app_for(tmp_path).get("/api/graph").json()

    pipeline = body["pipeline"]
    assert "contextualize" in pipeline["nodes"]
    assert "answer_insufficient" in pipeline["terminal"]
    assert {"from": "rewrite_query", "to": "retrieve_kb"} in pipeline["edges"], (
        "the rewrite loop is the one edge worth seeing in the diagram"
    )
    assert "grade_web_evidence" in pipeline["branches"]

    supervisor = body["supervisor"]
    assert "supervise" in supervisor["nodes"]
    assert "escalate" in supervisor["terminal"]
    assert supervisor["workers"]["payroll"]["documents"], (
        "a worker with no documents is a worker that answers nothing"
    )


def test_the_tool_catalogue_is_filtered_by_role(tmp_path):
    client = app_for(tmp_path)
    employee = client.get("/api/tools").json()
    names = {tool["name"] for tool in employee["tools"]}

    assert "kb_search" in names, (
        "the ReAct loop derives source_used from this exact name"
    )
    assert "calculate" in names
    assert employee["role"] == "employee"

    # The flags the registry enforces are published, so the console can show
    # which tools stop for a human.
    escalate = next(t for t in employee["tools"] if t["name"] == "escalate_to_human")
    assert escalate["side_effecting"] is True
    ticket = next(t for t in employee["tools"] if t["name"] == "raise_ticket")
    assert ticket["needs_approval"] is True


# --- Metrics -----------------------------------------------------------------


def test_metrics_are_open_and_aggregate_only(tmp_path):
    client = app_for(tmp_path)
    secret_question = "Does my zymography sabbatical carry over?"
    client.post("/api/chat", json={"question": secret_question})

    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.text

    assert "zymography" not in body, (
        "an open endpoint must never be able to emit an employee's question"
    )
    assert "Twenty days of PTO" not in body, "nor an answer"

    data = response.json()
    assert data["answers"]["total"] == 1
    assert data["cost_is_projected"] is True
    assert data["cache"]["available"] is True
    assert data["knowledge_base"]["documents"] >= 0


def test_per_model_spend_is_withheld_from_an_anonymous_caller(tmp_path):
    client = app_for(tmp_path)
    client.post("/api/chat", json={"question": QUESTION})

    assert "by_model" not in client.get("/api/metrics").json()["answers"]
    assert "by_model" in client.get("/api/metrics", headers=ADMIN).json()["answers"]
    assert "by_model" in client.get("/api/metrics", headers=HR).json()["answers"]


def test_metrics_survive_a_store_that_cannot_be_read(tmp_path):
    client = app_for(tmp_path)

    class Broken:
        def stats(self) -> dict[str, Any]:
            raise RuntimeError("the database is locked")

    client.app.state.cache = Broken()
    body = client.get("/api/metrics").json()

    assert body["cache"]["available"] is False
    assert body["answers"]["total"] == 0, "the rest of the page still renders"


# --- Chat: the added fields --------------------------------------------------


def test_an_answer_reports_the_model_the_cost_and_the_spans(tmp_path):
    body = app_for(tmp_path).post("/api/chat", json={"question": QUESTION}).json()

    assert body["answer_mode"] == "graph"
    assert body["model"]
    assert body["retrieval_mode"]
    assert body["prompt_tokens"] > 0
    assert body["budget"]["max_steps"] > 0
    assert body["spans"], "the console's latency breakdown reads this"
    assert body["audit_id"] == 1


def test_a_mode_can_be_chosen_per_question(tmp_path):
    client = app_for(tmp_path, RETRIEVAL_MODE="lexical")
    body = client.post(
        "/api/chat", json={"question": QUESTION, "mode": "extractive"}
    ).json()

    assert body["answer_mode"] == "extractive"


def test_an_unknown_mode_is_refused_rather_than_silently_ignored(tmp_path):
    response = app_for(tmp_path).post(
        "/api/chat", json={"question": QUESTION, "mode": "telepathy"}
    )
    assert response.status_code == 422


def test_a_redacted_question_is_what_reaches_the_audit_log(tmp_path):
    client = app_for(tmp_path)
    client.post(
        "/api/chat",
        json={"question": f"{QUESTION} I'm on +44 7700 900123."},
    )
    entry = client.get("/api/audit/1", headers=ADMIN).json()

    assert "900123" not in entry["question"]
    assert "phone" in entry["pii_redacted"]


# --- Streaming ---------------------------------------------------------------


def test_the_stream_emits_stages_a_result_and_a_terminator(tmp_path):
    events = sse_events(app_for(tmp_path), {"question": QUESTION})
    names = [name for name, _ in events]

    assert "stage" in names
    assert "answer" in names
    assert names[-1] == "end"

    result = next(data for name, data in events if name == "result")
    assert result["answer"] == "Twenty days of PTO."
    assert result["audit_id"] == 1, "a streamed answer is audited like any other"


def test_a_streamed_refusal_arrives_as_an_event_not_a_status(tmp_path):
    """Once the body has started there is no status code left to set."""
    client = app_for(
        tmp_path, RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1, RATE_LIMIT_BURST=1
    )
    client.post("/api/chat", json={"question": QUESTION})

    events = sse_events(client, {"question": QUESTION})
    errors = [data for name, data in events if name == "error"]

    assert errors, "the refusal has to reach the client somehow"
    assert errors[0]["status"] == 429
    assert [name for name, _ in events][-1] == "end"


def test_a_streamed_answer_matches_the_posted_one(tmp_path):
    posted = app_for(tmp_path).post("/api/chat", json={"question": QUESTION}).json()
    events = sse_events(app_for(tmp_path), {"question": QUESTION})
    streamed = next(data for name, data in events if name == "result")

    assert posted["answer"] == streamed["answer"]
    assert posted["source_used"] == streamed["source_used"]


# --- Rate limiting -----------------------------------------------------------


def test_the_shared_key_is_rate_limited_and_says_when_to_come_back(tmp_path):
    client = app_for(
        tmp_path, RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1, RATE_LIMIT_BURST=1
    )

    assert client.post("/api/chat", json={"question": QUESTION}).status_code == 200
    refused = client.post("/api/chat", json={"question": "And for contractors?"})

    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) >= 1
    assert "your own key" in refused.json()["detail"]


def test_an_admin_is_not_rate_limited(tmp_path):
    client = app_for(
        tmp_path, RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1, RATE_LIMIT_BURST=1
    )
    client.post("/api/chat", json={"question": QUESTION}, headers=ADMIN)
    again = client.post("/api/chat", json={"question": QUESTION}, headers=ADMIN)

    assert again.status_code == 200, (
        "an admin is operating the deployment, not consuming it"
    )


# --- Bring your own key ------------------------------------------------------


def test_a_caller_supplied_key_of_the_wrong_shape_is_named_not_forwarded(tmp_path):
    response = app_for(tmp_path).post(
        "/api/chat",
        json={"question": QUESTION},
        headers={"X-Provider": "groq", "X-Provider-Key": "sk-this-is-an-openai-key"},
    )

    assert response.status_code == 400
    assert "gsk_" in response.json()["detail"], (
        "the message names the rule so the person can fix it"
    )


def test_a_caller_supplied_key_is_refused_when_byok_is_off(tmp_path):
    response = app_for(tmp_path, ALLOW_BYOK=False).post(
        "/api/chat",
        json={"question": QUESTION},
        headers={"X-Provider-Key": "gsk_" + "a" * 40},
    )

    assert response.status_code == 400
    assert "does not accept" in response.json()["detail"]


def test_a_model_this_deployment_will_not_call_is_refused(tmp_path):
    response = app_for(tmp_path).post(
        "/api/chat",
        json={"question": QUESTION},
        headers={
            "X-Provider": "groq",
            "X-Provider-Key": "gsk_" + "a" * 40,
            "X-Provider-Model": "some-unvetted-model",
        },
    )

    assert response.status_code == 400
    assert "not one of the models" in response.json()["detail"]


def test_a_caller_supplied_key_never_appears_in_a_response(tmp_path):
    client = app_for(tmp_path)
    key = "gsk_" + "b" * 40
    # The key is valid in shape, so the request reaches the pool, which builds
    # a copilot over it -- and that copilot has no reachable provider, so the
    # answer degrades. What matters is that nothing echoes the key.
    response = client.post(
        "/api/chat",
        json={"question": QUESTION},
        headers={"X-Provider": "groq", "X-Provider-Key": key},
    )

    assert key not in response.text
    assert key not in client.get("/api/metrics").text


# --- Scratchpad --------------------------------------------------------------


def test_a_note_round_trips_and_is_scoped_to_its_conversation(tmp_path):
    client = app_for(tmp_path)
    created = client.post(
        "/api/scratchpad",
        json={"conversation_id": "conv-1", "text": "I am a contractor"},
    )
    assert created.status_code == 201

    mine = client.get("/api/scratchpad/conv-1").json()
    assert [note["text"] for note in mine["notes"]] == ["I am a contractor"]

    other = client.get("/api/scratchpad/conv-2").json()
    assert other["notes"] == []


def test_an_invalid_conversation_id_is_refused_rather_than_sanitised(tmp_path):
    """A sanitised id would silently merge two conversations."""
    response = app_for(tmp_path).post(
        "/api/scratchpad",
        json={"conversation_id": "../../etc/passwd", "text": "hello"},
    )
    assert response.status_code == 400


def test_the_scratchpad_reaches_the_answer_it_belongs_to(tmp_path):
    client = app_for(tmp_path)
    client.post(
        "/api/scratchpad",
        json={"conversation_id": "conv-9", "text": "Bangalore office"},
    )
    body = client.post(
        "/api/chat", json={"question": QUESTION, "conversation_id": "conv-9"}
    ).json()

    assert [note["text"] for note in body["scratchpad"]] == ["Bangalore office"]


# --- Approvals ---------------------------------------------------------------


def test_the_approval_queue_is_hr_and_admin_but_not_employee(tmp_path):
    client = app_for(tmp_path)

    assert client.get("/api/approvals").status_code == 403
    assert client.get("/api/approvals", headers=HR).status_code == 200
    assert client.get("/api/approvals", headers=ADMIN).status_code == 200


def test_hr_can_work_the_queue_without_being_able_to_read_the_audit_log(tmp_path):
    """The reason there is a permission table and not a second admin key."""
    client = app_for(tmp_path)

    assert client.get("/api/approvals", headers=HR).status_code == 200
    assert client.get("/api/audit", headers=HR).status_code == 401


def test_a_request_can_be_decided_once(tmp_path):
    client = app_for(tmp_path)
    recorded = client.app.state.approvals.request(
        action="hr_ticket",
        question=QUESTION,
        reason="needs a person",
        payload={"summary": "carry-over"},
    )
    assert recorded is not None

    first = client.post(
        f"/api/approvals/{recorded.id}/decide",
        json={"decision": "approved", "note": "fine"},
        headers=HR,
    ).json()
    assert first["changed"] is True
    assert first["request"]["status"] == "approved"
    assert first["request"]["decided_by"] == "hr"

    second = client.post(
        f"/api/approvals/{recorded.id}/decide",
        json={"decision": "rejected"},
        headers=ADMIN,
    ).json()
    assert second["changed"] is False
    assert second["request"]["status"] == "approved", (
        "two people deciding at once must not be last-write-wins on a side effect"
    )


def test_deciding_something_that_does_not_exist_is_a_404(tmp_path):
    response = app_for(tmp_path).post(
        "/api/approvals/9999/decide", json={"decision": "approved"}, headers=HR
    )
    assert response.status_code == 404


def test_an_invalid_decision_is_refused_by_the_schema(tmp_path):
    response = app_for(tmp_path).post(
        "/api/approvals/1/decide", json={"decision": "maybe"}, headers=HR
    )
    assert response.status_code == 422


# --- Cache -------------------------------------------------------------------


def test_only_hr_and_above_may_drop_the_cache(tmp_path):
    client = app_for(tmp_path)

    assert client.post("/api/cache/invalidate").status_code == 403
    assert client.post("/api/cache/invalidate", headers=HR).status_code == 200


def test_dropping_the_cache_makes_the_next_question_run_again(tmp_path):
    client = app_for(tmp_path)
    client.post("/api/chat", json={"question": QUESTION})
    assert client.post("/api/chat", json={"question": QUESTION}).json()["cached"]

    dropped = client.post("/api/cache/invalidate", headers=ADMIN).json()["dropped"]
    assert dropped == 1

    assert client.post("/api/chat", json={"question": QUESTION}).json()["cached"] is False
