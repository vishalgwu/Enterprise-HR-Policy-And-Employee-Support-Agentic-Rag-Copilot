"""Reordering candidates before they reach a prompt.

Retrieval ranks by similarity to the *question*, which is not the same thing as
"answers the question". The 12-question eval the 0.15 gate was set from records
the gap directly: a wrong-source chunk scored 0.521 while the lowest correct
chunk scored 0.171, so similarity order is demonstrably not answer order. That
is tolerable while the numbers are only deciding whether anything is in domain
at all, and it stops being tolerable at the point where the top four chunks
become the only thing the generator will ever see.

So retrieve a wider list than the answer needs, then spend a second, cheaper
judgement on ordering it. Two strategies, because they trade differently:

    lexical  free, offline, deterministic. Term overlap with the question.
    llm      better, and costs a model call per question.

`lexical_rerank` is also the fallback for every way the LLM path can fail,
which is the reason it is first in the file: falling back to the order the
re-ranker was handed would silently turn the feature off, and this costs
nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.core.config import Settings, get_logger
from app.core.tracing import KIND_RETRIEVAL, null_tracer
from app.rag.clients import get_llm, is_rate_limit
from app.rag.lexical import tiebreak_key, tokenize
from app.rag.loaders import format_documents

log = get_logger("rerank")

__all__ = [
    "LLMReranker",
    "Ranked",
    "documents_of",
    "lexical_rerank",
    "rerank",
]

# The blend `lexical_rerank` scores with. Three signals, each already 0..1, and
# weights that sum to 1 so the result is too:
#
#   0.60  coverage -- how much of the question's vocabulary the chunk contains
#   0.25  heading  -- whether the chunk's first line shares a term with it
#   0.15  phrase   -- an exact adjacent-pair match somewhere in the chunk
#
# **These were chosen to be explainable, not tuned.** No labelled set was run
# over them. Coverage leads because it is the signal with the most behind it, a
# heading match is the cheapest available proxy for "this chunk is about that",
# and a phrase match is a bonus rather than a judgement because one adjacent
# pair can line up by accident. A number that looks measured and is not would be
# worse here than an honest weight, since the whole point of this module is to
# be the cheap, defensible half of re-ranking.
WEIGHT_COVERAGE = 0.60
WEIGHT_HEADING = 0.25
WEIGHT_PHRASE = 0.15


@dataclass(slots=True)
class Ranked:
    """One candidate, its score, its position and why.

    `reason` is for the console's evidence panel and for a bug report: "term
    overlap 0.42" and "llm 8/10" are different claims about the same ordering,
    and an operator asking why a chunk came third should not have to infer
    which strategy produced it.
    """

    document: Document
    score: float
    rank: int
    reason: str = ""


def documents_of(ranked: Sequence[Ranked]) -> list[Document]:
    return [item.document for item in ranked]


# --- The offline re-rank -----------------------------------------------------


def _heading_terms(document: Document) -> set[str]:
    for line in document.page_content.splitlines():
        if line.strip():
            return set(tokenize(line))
    return set()


def _adjacent_pairs(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _score_one(
    document: Document, query_terms: set[str], query_pairs: set[tuple[str, str]]
) -> float:
    tokens = tokenize(document.page_content)
    coverage = len(query_terms & set(tokens)) / len(query_terms)
    heading = 1.0 if query_terms & _heading_terms(document) else 0.0
    phrase = 1.0 if query_pairs & _adjacent_pairs(tokens) else 0.0
    return (
        WEIGHT_COVERAGE * coverage
        + WEIGHT_HEADING * heading
        + WEIGHT_PHRASE * phrase
    )


def _lexical_scores(query: str, documents: Sequence[Document]) -> list[float]:
    """A 0..1 score per document, aligned to the order it was given in.

    Aligned by position rather than keyed by document identity because the LLM
    path pads its own scores from these, and a mapping would need an identity
    function that `app.rag.hybrid` owns -- importing it would be a cycle.
    """
    query_tokens = tokenize(query)
    query_terms = set(query_tokens)
    if not query_terms:
        return [0.0] * len(documents)
    query_pairs = _adjacent_pairs(query_tokens)
    return [_score_one(doc, query_terms, query_pairs) for doc in documents]


def _assemble(
    documents: Sequence[Document],
    scores: Sequence[float],
    reasons: Sequence[str],
    secondary: Sequence[float] | None = None,
) -> list[Ranked]:
    """Sort by score, number the result, and break every tie the same way twice."""
    second = list(secondary) if secondary is not None else [0.0] * len(documents)
    order = sorted(
        range(len(documents)),
        key=lambda i: (-scores[i], -second[i], *tiebreak_key(documents[i])),
    )
    return [
        Ranked(
            document=documents[i],
            score=round(scores[i], 4),
            rank=position,
            reason=reasons[i],
        )
        for position, i in enumerate(order, 1)
    ]


def lexical_rerank(
    query: str, documents: Sequence[Document], k: int | None = None
) -> list[Ranked]:
    """Reorder by term overlap with the question. Offline, deterministic, free."""
    docs = list(documents)
    if not docs:
        return []

    scores = _lexical_scores(query, docs)
    if not any(scores):
        # Nothing to judge on -- an empty or all-stopword question, or a
        # question whose every term is absent from every candidate. Keep the
        # order the caller arrived with rather than imposing one by file name:
        # that order came from the fusion, which knew more than this does.
        return [
            Ranked(document=doc, score=0.0, rank=position, reason="no term overlap")
            for position, doc in enumerate(docs, 1)
        ]

    reasons = [f"term overlap {score:.2f}" for score in scores]
    ranked = _assemble(docs, scores, reasons)
    return ranked[:k] if k else ranked


# --- The model re-rank -------------------------------------------------------


def _parse_scores(text: str | None) -> list[int] | None:
    """Pull a score list out of whatever the model actually said.

    The same recovery shape `app.agent.decisions` uses for the router and the
    grader: Groq answers `400 tool_use_failed` when the model writes prose
    instead of calling the bound tool, and the raw text it wrote is usually the
    right answer in the wrong wrapper -- here, a list of numbers inside a
    sentence or a JSON fragment.
    """
    if not text:
        return None
    digits = [int(found) for found in re.findall(r"\d+", text)]
    return digits or None


def _fit(scores: Sequence[int], lexical: Sequence[float]) -> list[int]:
    """Make the model's score list exactly as long as the candidate list.

    It comes back the wrong length often enough to plan for. A model that
    merges two passages returns N-1 numbers, and then every passage after the
    merge silently takes its neighbour's score -- the list still looks
    well-formed, so nothing downstream can notice. Too many numbers is the same
    problem from the other end.

    Short lists are padded from the lexical score, which is a judgement rather
    than a guess; long ones are truncated. Nothing indexes past the end.
    """
    fitted = [max(0, min(10, int(score))) for score in scores[: len(lexical)]]
    fitted.extend(round(lexical[i] * 10) for i in range(len(fitted), len(lexical)))
    return fitted


class LLMReranker:
    """Asks the model to score each candidate 0-10, using `RERANK_PROMPT`.

    Survives a malformed reply the way every other structured call in this
    project does -- try, recover the raw text, then fall back -- with one
    difference in the last step: **the fallback is `lexical_rerank`, never the
    order this was handed.** Returning the input order is indistinguishable
    from the feature being switched off, and would make a provider having a bad
    afternoon look like a design decision.
    """

    def __init__(self, llm: Any = None, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings

    def _chain(self) -> Any:
        # Imported inside the function on purpose. `prompts` and `schemas` are
        # pure declarations -- a prompt template and a Pydantic model, with no
        # `app.rag` imports of their own -- so there is no cycle, but
        # `app/rag/*` importing `app/agent/*` at module scope would point this
        # package's import graph the wrong way round against the layering.
        from app.agent.prompts import RERANK_PROMPT
        from app.agent.schemas import RelevanceScores

        llm = self._llm if self._llm is not None else get_llm(settings=self._settings)
        return RERANK_PROMPT | llm.with_structured_output(RelevanceScores)

    def _ask(self, query: str, documents: Sequence[Document]) -> list[int] | None:
        """One structured call, or None when nothing usable came back."""
        from app.agent.decisions import failed_generation

        try:
            decision = self._chain().invoke(
                {"question": query, "passages": format_documents(documents)}
            )
            return [int(score) for score in (getattr(decision, "scores", None) or [])]
        except Exception as exc:  # noqa: BLE001 - a re-rank must never cost the answer
            recovered = _parse_scores(failed_generation(exc))
            if recovered:
                log.debug(
                    "[Rerank] recovered %d score(s) from a failed call", len(recovered)
                )
                return recovered
            if is_rate_limit(exc):
                # Error level for the same reason the grader logs it that way:
                # an exhausted quota degrades silently here, because a lexical
                # re-rank still returns a plausible order and nothing in the
                # answer says the model never saw the candidates.
                log.error(
                    "[Rerank] RATE LIMITED; falling back to the lexical re-rank -- "
                    "this is a quota failure, not a judgement"
                )
            else:
                log.exception("[Rerank] the model re-rank failed (%s)", type(exc).__name__)
            return None

    def rank(
        self,
        query: str,
        documents: Sequence[Document],
        k: int | None = None,
        tracer: Any = None,
        budget: Any = None,
    ) -> list[Ranked]:
        docs = list(documents)
        if not docs:
            return []
        tracer = tracer or null_tracer()
        lexical = _lexical_scores(query, docs)

        if budget is not None and not budget.allows("llm_calls"):
            # Not charged: nothing was spent. `allows()` has already recorded
            # the stop reason, so the run can still say why the answer is the
            # cheaper shape.
            tracer.event(
                "rerank", kind=KIND_RETRIEVAL, strategy="lexical", reason="budget"
            )
            return lexical_rerank(query, docs, k)

        with tracer.span(
            "rerank", kind=KIND_RETRIEVAL, strategy="llm", candidates=len(docs)
        ) as span:
            raw = self._ask(query, docs)
            if budget is not None:
                # Charged whether or not the reply was usable: the round trip
                # happened, and a budget that only counts successful calls does
                # not bound a provider that is failing expensively.
                budget.charge_llm()
            if raw is None:
                span.status = "degraded"
                return lexical_rerank(query, docs, k)
            span.set(scored=len(raw))

        fitted = _fit(raw, lexical)
        reasons = [f"llm {score}/10" for score in fitted]
        ranked = _assemble(
            docs, [score / 10 for score in fitted], reasons, secondary=lexical
        )
        return ranked[:k] if k else ranked


def rerank(
    query: str,
    documents: Sequence[Document],
    k: int | None = None,
    strategy: str = "lexical",
    llm: Any = None,
    tracer: Any = None,
    budget: Any = None,
    settings: Settings | None = None,
) -> list[Ranked]:
    """Reorder candidates by the configured strategy.

    `strategy` comes from `Settings.rerank_strategy`; it is passed rather than
    read here so one request can override it and so a test does not have to
    build a `Settings` to exercise a branch.
    """
    docs = list(documents)
    if not docs:
        return []

    if strategy == "llm":
        return LLMReranker(llm=llm, settings=settings).rank(
            query, docs, k=k, tracer=tracer, budget=budget
        )

    tracer = tracer or null_tracer()
    with tracer.span(
        "rerank", kind=KIND_RETRIEVAL, strategy=strategy, candidates=len(docs)
    ):
        if strategy == "none":
            # The fused order is already a ranking. "none" means do not spend
            # anything improving it, not throw it away.
            ranked = [
                Ranked(document=doc, score=0.0, rank=position, reason="fused")
                for position, doc in enumerate(docs, 1)
            ]
            return ranked[:k] if k else ranked
        return lexical_rerank(query, docs, k)
