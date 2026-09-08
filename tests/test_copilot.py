"""The service facade: the projection from AgentState onto the API contract."""

from __future__ import annotations

from conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc

from app.agent.graph import build_graph
from app.services.copilot import AnswerResult, Copilot, render_answer, to_answer


def _state(**overrides):
    state = {
        "question": "How many PTO days do I get?",
        "current_query": "How many PTO days do I get?",
        "retry_count": 0,
        "route": "kb",
        "kb_docs": [kb_doc()],
        "kb_grade": "good",
        "answer": "  Twenty days.  ",
        "source_used": "private_kb",
        "trace": ["Router: kb", "KB Retriever: 1 chunk(s)", "KB Grader: good"],
    }
    state.update(overrides)
    return state


def test_answer_is_trimmed_and_typed():
    result = to_answer(_state())
    assert isinstance(result, AnswerResult)
    assert result.answer == "Twenty days."
    assert result.kb_chunks == 1


def test_sources_are_projected_from_chunk_metadata():
    result = to_answer(_state())
    assert [s.source for s in result.sources] == ["leave-and-time-off.md"]
    assert result.sources[0].department == "general"


def test_rewritten_query_is_only_set_when_it_actually_changed():
    assert to_answer(_state()).rewritten_query is None
    assert to_answer(_state(current_query="formal")).rewritten_query == "formal"


def test_grounded_distinguishes_evidence_backed_answers():
    assert to_answer(_state()).grounded
    assert to_answer(_state(source_used="web_search")).grounded
    assert not to_answer(_state(source_used="insufficient_evidence")).grounded
    assert not to_answer(_state(source_used="direct")).grounded
    assert not to_answer(_state(source_used="error")).grounded


def test_the_trace_is_carried_through_to_the_api_contract():
    result = to_answer(_state())
    assert result.trace[0] == "Router: kb"
    assert len(result.trace) == 3


def test_sources_come_from_the_state_citations_when_the_graph_wrote_them():
    """`retrieve_kb` already cited the chunks; the projection must not re-derive
    them and quietly disagree with what the run recorded."""
    cited = [{"source": "pay.md", "title": "Pay", "department": "hr",
              "doc_type": "markdown"}]
    result = to_answer(_state(citations=cited))
    assert [s.source for s in result.sources] == ["pay.md"]


def test_a_missing_trace_does_not_break_the_projection():
    result = to_answer({"question": "q", "answer": "a"})
    assert result.route is None and result.kb_chunks == 0 and result.sources == []


def test_answer_result_serialises_for_an_api_response():
    payload = to_answer(_state()).model_dump()
    assert payload["source_used"] == "private_kb"
    assert payload["sources"][0]["title"]


def test_render_answer_shows_the_decision_trace():
    text = render_answer(to_answer(_state()))
    assert "ROUTE         kb" in text
    assert "SOURCE USED   private_kb" in text
    assert "leave-and-time-off.md" in text
    assert "Twenty days." in text


def test_render_answer_lists_the_trace_steps():
    text = render_answer(to_answer(_state()))
    assert "TRACE" in text
    assert "1. Router: kb" in text
    assert "TRACE" not in render_answer(to_answer(_state()), show_trace=False)


def test_render_answer_omits_the_rewrite_line_when_there_was_none():
    assert "REWRITTEN AS" not in render_answer(to_answer(_state()))
    assert "REWRITTEN AS" in render_answer(to_answer(_state(current_query="formal")))


# --- Copilot -----------------------------------------------------------------


def test_copilot_answers_end_to_end_over_fakes():
    graph = build_graph(
        llm=FakeLLM(route="kb", grades=["good"], reply="Twenty days."),
        retriever=FakeRetriever([kb_doc()]),
        web_search=FakeWebSearch(),
        verbose=False,
    )
    result = Copilot(graph=graph).ask("How many PTO days do I get?")
    assert result.answer == "Twenty days."
    assert result.source_used == "private_kb"


def test_copilot_builds_its_graph_once():
    graph = build_graph(
        llm=FakeLLM(), retriever=FakeRetriever([]), web_search=FakeWebSearch(),
        verbose=False,
    )
    copilot = Copilot(graph=graph)
    assert copilot.graph is copilot.graph
