"""Lexical search, rank fusion and re-ranking, all offline.

None of this needs a key, which is most of the point of it: BM25 is the half of
retrieval a deployment with no Pinecone key can still serve, so a test suite
that needed a key to exercise it would be testing the wrong thing.

The arithmetic tests are the ones to keep honest. A negative IDF and a tie
broken by dict order are both failures that produce a plausible-looking ranking,
so neither shows up as an error anywhere -- only as answers that are quietly
worse, or that differ between two identical runs.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

from conftest import FakeRetriever, kb_doc, settings_for
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.agent.budgets import Budget
from app.agent.schemas import RelevanceScores
from app.core.tracing import Tracer
from app.rag import hybrid
from app.rag.hybrid import (
    HybridRetriever,
    build_retriever,
    document_key,
    reciprocal_rank_fusion,
)
from app.rag.lexical import (
    BM25Index,
    LexicalRetriever,
    build_kb_index,
    tokenize,
)
from app.rag.rerank import (
    LLMReranker,
    documents_of,
    lexical_rerank,
    rerank,
)


class ScriptedLLM(Runnable):
    """A chat model that returns scripted `RelevanceScores`, and counts calls.

    Lives here rather than in `conftest.FakeLLM`, which only knows the router
    and grader schemas.
    """

    def __init__(self, decisions: Any, reply: str = "ok") -> None:
        self.decisions = list(decisions)
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append("invoke")
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def decide(_input: Any) -> Any:
            self.calls.append(schema.__name__)
            if len(self.decisions) > 1:
                return self.decisions.pop(0)
            return self.decisions[0]

        return RunnableLambda(decide)


class BrokenLLM(Runnable):
    """A model whose structured call always fails, with no recoverable text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        raise RuntimeError("simulated provider outage")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def boom(_input: Any) -> Any:
            self.calls.append(schema.__name__)
            raise RuntimeError("simulated provider outage")

        return RunnableLambda(boom)


# Two chunks that disagree about which retriever should find them: the first
# names an exact term, the second is longer and shares only common vocabulary.
EXACT = kb_doc(
    content="Submit Form 16 to the HR Service Desk by 30 June.",
    source="payroll.md",
)
COMMON = kb_doc(
    content=(
        "The employee handbook explains how to submit a form to the HR service "
        "desk, and the policy covers every form the company uses for expenses "
        "and travel."
    ),
    source="handbook.md",
)

# One chunk that states the asked-about fact, one that is merely on the subject.
WITH_TERM = kb_doc(
    content=(
        "# Offboarding and resignation\n"
        "The notice period for a resignation is 30 days for all full-time "
        "employees."
    ),
    source="offboarding.md",
)
ON_TOPIC = kb_doc(
    content=(
        "# Leave and time off\n"
        "Employees accrue paid time off each month and should discuss plans "
        "with their manager."
    ),
    source="leave-and-time-off.md",
)

NOTICE_QUESTION = "What is the notice period for a resignation?"


# --- Tokenising --------------------------------------------------------------


def test_tokenize_drops_stopwords_and_splits_a_hyphenated_policy_term():
    assert tokenize("What is the work-from-home policy?") == [
        "work",
        "home",
        "policy",
    ]


def test_tokenize_drops_single_characters_and_keeps_a_form_number():
    assert tokenize("Do I file Form 16 or a P-60?") == ["file", "form", "16", "60"]


# --- BM25 scoring ------------------------------------------------------------


def test_bm25_puts_an_exact_rare_term_above_a_longer_chunk_of_common_words():
    index = BM25Index.from_documents([COMMON, EXACT])
    results = index.search("form 16", k=2)

    assert [doc.metadata["source"] for doc, _ in results] == [
        "payroll.md",
        "handbook.md",
    ]


def test_a_term_in_most_of_the_corpus_still_scores_positively():
    """The IDF guard. Without the `1 +`, a common term inverts the ranking."""
    corpus = [
        kb_doc(content="This policy covers paid leave.", source="a.md"),
        kb_doc(content="This policy covers expenses.", source="b.md"),
        kb_doc(content="This policy covers travel.", source="c.md"),
    ]
    index = BM25Index.from_documents(corpus)

    # "policy" is in every chunk, which is where the textbook form breaks.
    assert index.doc_freq["policy"] == 3
    assert math.log((3 - 3 + 0.5) / (3 + 0.5)) < 0
    assert index.idf("policy") > 0

    results = index.search("policy", k=3)
    assert len(results) == 3
    assert all(score > 0 for _, score in results)


def test_an_empty_or_all_stopword_query_returns_nothing_without_raising():
    index = BM25Index.from_documents([EXACT, COMMON])

    assert index.search("", k=4) == []
    assert index.search("   ", k=4) == []
    assert index.search("what is the and of it", k=4) == []


def test_searching_an_empty_index_returns_nothing():
    index = BM25Index.from_documents([])

    assert index.is_empty
    assert len(index) == 0
    assert index.search("form 16", k=4) == []


def test_tied_scores_come_back_in_the_same_order_every_time():
    """Identical chunks score identically, so only the tie-break decides."""
    tied = [
        kb_doc(content="Core hours are 10:00 to 16:00.", source=f"{name}.md")
        for name in ("alpha", "beta", "gamma", "delta")
    ]

    shuffled = list(tied)
    random.Random(0).shuffle(shuffled)
    first = BM25Index.from_documents(shuffled).search("core hours", k=4)

    random.Random(7).shuffle(shuffled)
    second = BM25Index.from_documents(shuffled).search("core hours", k=4)

    assert [doc.metadata["source"] for doc, _ in first] == [
        doc.metadata["source"] for doc, _ in second
    ]


def test_the_lexical_retriever_returns_documents_for_a_query():
    retriever = LexicalRetriever(BM25Index.from_documents([EXACT, COMMON]), k=1)

    assert retriever.invoke("form 16") == [EXACT]


# --- Building the index from disk -------------------------------------------


def write_kb(directory: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


PAYROLL_KB = {"payroll.md": "# Payroll\nPayday is the 25th of each month.\n"}


def test_an_edited_knowledge_base_rebuilds_rather_than_serving_the_cache(tmp_path):
    write_kb(tmp_path, PAYROLL_KB)
    settings = settings_for(KB_DIR=str(tmp_path))

    first = build_kb_index(settings=settings)
    assert first.search("payday", k=1)
    assert not first.search("sabbatical", k=1)

    added = tmp_path / "leave.md"
    added.write_text("# Leave\nA sabbatical needs twelve months service.\n", "utf-8")
    # Forced rather than assumed: a file written in the same filesystem tick as
    # the first build would leave the newest mtime unchanged, and this test
    # would then pass or fail by disk speed.
    stamp = added.stat().st_mtime + 10
    os.utime(added, (stamp, stamp))

    rebuilt = build_kb_index(settings=settings)
    assert rebuilt.search("sabbatical", k=1)


def test_a_repeated_build_of_an_unchanged_directory_is_the_same_object(tmp_path):
    write_kb(tmp_path, PAYROLL_KB)
    settings = settings_for(KB_DIR=str(tmp_path))

    assert build_kb_index(settings=settings) is build_kb_index(settings=settings)


# --- Identity ----------------------------------------------------------------


def test_two_chunks_of_one_file_without_an_offset_keep_separate_keys():
    """`chunk_id` alone would hash both to the same value and fuse them."""
    first = kb_doc(content="Core hours are 10:00 to 16:00.", source="working.md")
    second = kb_doc(content="Expenses are paid within 14 days.", source="working.md")

    assert document_key(first) != document_key(second)


def test_a_chunk_with_an_offset_is_keyed_by_its_chunk_id():
    chunk = Document(
        page_content="Payday is the 25th.",
        metadata={"source": "payroll.md", "origin": "hr-docs", "start_index": 0},
    )
    other = Document(
        page_content="Payday is the 25th.",
        metadata={"source": "payroll.md", "origin": "hr-docs", "start_index": 940},
    )

    assert document_key(chunk) != document_key(other)


# --- Rank fusion -------------------------------------------------------------


def test_a_document_both_retrievers_found_outranks_one_only_the_dense_side_found():
    dense = [ON_TOPIC, WITH_TERM]
    sparse = [WITH_TERM]

    fused = reciprocal_rank_fusion([dense, sparse])

    # WITH_TERM was second on the dense side and first on the sparse side;
    # ON_TOPIC was first on the dense side and absent from the other. Agreement
    # beats a better rank in one list, which is what RRF is for.
    assert [doc for doc, _ in fused] == [WITH_TERM, ON_TOPIC]


def test_fusion_respects_the_weights():
    dense_only = [EXACT]
    sparse_only = [COMMON]

    dense_heavy = reciprocal_rank_fusion([dense_only, sparse_only], weights=[0.9, 0.1])
    sparse_heavy = reciprocal_rank_fusion([dense_only, sparse_only], weights=[0.1, 0.9])

    assert dense_heavy[0][0] is EXACT
    assert sparse_heavy[0][0] is COMMON


def test_fusion_keeps_the_first_document_object_seen_for_a_key():
    same_chunk_again = kb_doc(content=EXACT.page_content, source="payroll.md")

    fused = reciprocal_rank_fusion([[EXACT], [same_chunk_again]])

    assert len(fused) == 1
    assert fused[0][0] is EXACT


def test_fusion_of_nothing_is_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


# --- Degrading ---------------------------------------------------------------


def test_the_hybrid_retriever_answers_from_the_sparse_side_when_the_dense_one_fails():
    tracer = Tracer()
    retriever = HybridRetriever(
        dense=FakeRetriever(fail=True),
        sparse=FakeRetriever(results=[WITH_TERM]),
        tracer=tracer,
    )

    assert retriever.invoke(NOTICE_QUESTION) == [WITH_TERM]
    assert [span.name for span in tracer.spans() if span.status == "degraded"] == [
        "retrieve.dense"
    ]


def test_the_hybrid_retriever_answers_from_the_dense_side_when_the_sparse_one_fails():
    tracer = Tracer()
    retriever = HybridRetriever(
        dense=FakeRetriever(results=[WITH_TERM]),
        sparse=FakeRetriever(fail=True),
        tracer=tracer,
    )

    assert retriever.invoke(NOTICE_QUESTION) == [WITH_TERM]
    assert [span.name for span in tracer.spans() if span.status == "degraded"] == [
        "retrieve.lexical"
    ]


def test_both_sides_failing_returns_no_chunks_rather_than_raising():
    tracer = Tracer()
    retriever = HybridRetriever(
        dense=FakeRetriever(fail=True),
        sparse=FakeRetriever(fail=True),
        tracer=tracer,
    )

    # Zero chunks is a state the graph already handles: it routes to the web
    # fallback. An exception would abort the run instead.
    assert retriever.invoke(NOTICE_QUESTION) == []
    assert len([span for span in tracer.spans() if span.status == "degraded"]) == 2


def test_the_mode_reports_which_sides_are_present():
    dense, sparse = FakeRetriever(), FakeRetriever()

    assert HybridRetriever(dense=dense, sparse=sparse).mode == "hybrid"
    assert HybridRetriever(dense=dense).mode == "dense"
    assert HybridRetriever(sparse=sparse).mode == "lexical"
    assert HybridRetriever().mode == "empty"


# --- Re-ranking --------------------------------------------------------------


def test_lexical_rerank_promotes_the_chunk_that_contains_the_asked_about_term():
    ranked = lexical_rerank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM])

    assert documents_of(ranked) == [WITH_TERM, ON_TOPIC]
    assert ranked[0].rank == 1
    assert ranked[0].score > ranked[1].score
    assert ranked[0].reason.startswith("term overlap")


def test_lexical_rerank_keeps_the_fused_order_when_no_term_matches():
    ranked = lexical_rerank("capital of France", [ON_TOPIC, WITH_TERM])

    assert documents_of(ranked) == [ON_TOPIC, WITH_TERM]


def test_rerank_with_no_strategy_keeps_the_fused_order_and_says_so():
    ranked = rerank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM], strategy="none")

    assert documents_of(ranked) == [ON_TOPIC, WITH_TERM]
    assert [item.reason for item in ranked] == ["fused", "fused"]


def test_the_llm_reranker_reorders_as_the_model_says():
    llm = ScriptedLLM([RelevanceScores(scores=[2, 9])])
    reranker = LLMReranker(llm=llm)

    ranked = reranker.rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM])

    assert documents_of(ranked) == [WITH_TERM, ON_TOPIC]
    assert ranked[0].reason == "llm 9/10"
    assert llm.calls == ["RelevanceScores"]


def test_a_score_list_of_the_wrong_length_keeps_every_candidate():
    llm = ScriptedLLM([RelevanceScores(scores=[5])])
    reranker = LLMReranker(llm=llm)

    ranked = reranker.rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM, EXACT])

    assert len(ranked) == 3
    assert sorted(id(doc) for doc in documents_of(ranked)) == sorted(
        id(doc) for doc in (ON_TOPIC, WITH_TERM, EXACT)
    )


def test_a_score_list_that_is_too_long_is_truncated():
    llm = ScriptedLLM([RelevanceScores(scores=[1, 2, 3, 4, 5])])

    ranked = LLMReranker(llm=llm).rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM])

    assert len(ranked) == 2


def test_the_llm_reranker_falls_back_to_lexical_order_not_input_order():
    llm = BrokenLLM()

    ranked = LLMReranker(llm=llm).rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM])

    # The input order was [ON_TOPIC, WITH_TERM]; falling back to it would make
    # a provider outage look like a design decision.
    assert documents_of(ranked) == [WITH_TERM, ON_TOPIC]
    assert llm.calls == ["RelevanceScores"]


def test_a_spent_budget_skips_the_model_and_is_not_charged_for_it():
    llm = ScriptedLLM([RelevanceScores(scores=[9, 1])])
    budget = Budget(max_llm_calls=1).charge_llm()

    ranked = LLMReranker(llm=llm).rank(
        NOTICE_QUESTION, [ON_TOPIC, WITH_TERM], budget=budget
    )

    assert llm.calls == []
    assert budget.llm_calls == 1
    assert documents_of(ranked) == [WITH_TERM, ON_TOPIC]


def test_a_model_rerank_charges_the_budget_once():
    llm = ScriptedLLM([RelevanceScores(scores=[2, 9])])
    budget = Budget(max_llm_calls=4)

    LLMReranker(llm=llm).rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM], budget=budget)

    assert budget.llm_calls == 1


def test_the_model_rerank_records_a_span():
    tracer = Tracer()
    llm = ScriptedLLM([RelevanceScores(scores=[2, 9])])

    LLMReranker(llm=llm).rank(NOTICE_QUESTION, [ON_TOPIC, WITH_TERM], tracer=tracer)

    spans = [span for span in tracer.spans() if span.name == "rerank"]
    assert spans and spans[0].kind == "retrieval"
    assert spans[0].attributes["strategy"] == "llm"


# --- Choosing a retriever ----------------------------------------------------


def test_build_retriever_is_lexical_when_the_mode_says_lexical(tmp_path):
    write_kb(tmp_path, PAYROLL_KB)
    settings = settings_for(RETRIEVAL_MODE="lexical", KB_DIR=str(tmp_path))

    retriever = build_retriever(settings=settings)

    assert isinstance(retriever, LexicalRetriever)
    assert retriever.invoke("payday")


def test_build_retriever_falls_back_to_lexical_when_the_dense_side_fails(
    tmp_path, monkeypatch
):
    write_kb(tmp_path, PAYROLL_KB)
    settings = settings_for(KB_DIR=str(tmp_path))
    assert settings.effective_retrieval_mode == "hybrid"

    def no_pinecone(**kwargs: Any) -> Any:
        raise RuntimeError("simulated Pinecone outage")

    monkeypatch.setattr(hybrid, "get_kb_retriever", no_pinecone)

    retriever = build_retriever(settings=settings)

    # Worse answers, not no answers.
    assert isinstance(retriever, LexicalRetriever)
    assert retriever.invoke("payday")


def test_build_retriever_wires_a_hybrid_from_the_settings(tmp_path):
    write_kb(tmp_path, PAYROLL_KB)
    settings = settings_for(KB_DIR=str(tmp_path), HYBRID_DENSE_WEIGHT=0.7, RRF_K=30)

    retriever = build_retriever(settings=settings, dense=FakeRetriever(results=[EXACT]))

    assert isinstance(retriever, HybridRetriever)
    assert retriever.mode == "hybrid"
    assert retriever.dense_weight == 0.7
    assert retriever.rrf_k == 30


def test_build_retriever_in_dense_mode_returns_the_dense_retriever_unchanged(tmp_path):
    settings = settings_for(RETRIEVAL_MODE="dense", KB_DIR=str(tmp_path))
    dense = FakeRetriever(results=[EXACT])

    assert build_retriever(settings=settings, dense=dense) is dense
