"""Answering with no language model at all.

This is the floor the product stands on. Every other path needs a key: the graph
needs a chat model to route, grade and generate, the supervisor needs one per
worker, and a deployment with no provider key answers nothing — which, for a
public demo that a stranger should be able to try, means a product that looks
broken rather than one that looks free.

So: retrieve, then stitch. Rank the sentences of the retrieved chunks by how
much of the question they cover, keep the best few in document order, and label
the result as what it is. No model wrote it, so there is nothing for a model to
invent — which makes this the one path in the system that is *structurally*
incapable of hallucinating a policy number. It reads worse than a generated
answer and it is more trustworthy than one.

**It is a fallback, and it says so.** `EXTRACTIVE_PREFIX` leads the answer
rather than trailing it, because the thing a reader needs to know about this
text is that nothing composed it and the sentences may not follow on from each
other.

Two jobs, again:

1. The no-key path, as above.
2. The last fallback when a configured model is unreachable. `_generate` in
   `app/agent/nodes.py` currently returns `GENERATION_FAILED` — an apology —
   when generation fails after retrieval succeeded. But at that point the
   evidence is sitting in state: the run found the policy and then failed to
   write prose about it. Handing back the policy text beats handing back an
   apology, and this is what makes that possible.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.agent.prompts import EXTRACTIVE_PREFIX
from app.core.config import get_logger
from app.rag.lexical import tokenize
from app.rag.loaders import cite_sources

log = get_logger("extractive")

__all__ = [
    "ExtractiveAnswer",
    "answer_from_documents",
    "split_sentences",
]

# Sentence boundaries, approximately. A full stop, question mark or exclamation
# followed by whitespace and a capital or a digit.
#
# Approximately is the right amount of effort here. A real sentence segmenter
# (nltk, pysbd) is a dependency and a model download to split eleven markdown
# files whose sentences are short, declarative and already one per line more
# often than not. The failure mode of getting it slightly wrong is a slightly
# long or slightly clipped quote, which a reader absorbs; the failure mode of
# adding a 40 MB dependency to the one path that exists *because* the
# deployment is too small for dependencies is the path not existing.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Markdown furniture that should not become a sentence of an answer: a heading
# marker, a list bullet, a table rule, trailing pipes.
_MARKDOWN_NOISE = re.compile(r"^[\s>#*\-|]+|[\s|]+$")

# A table row, which is meaningful content but is not a sentence. Kept whole
# rather than split or dropped: the benefit tiers and expense caps in this KB
# live in tables, and a tier table with its rows separated reads as nonsense.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# Below this share of the question's content words, a sentence is not about the
# question. Tuned to be generous: this path has no grader behind it, so the
# alternative to a marginal sentence is usually nothing at all, and a reader
# shown four sentences can discard one. A reader shown none cannot.
_MIN_COVERAGE = 0.12


@dataclass(slots=True)
class ExtractiveAnswer:
    """What was assembled, and from where."""

    text: str
    citations: list[dict[str, str]] = field(default_factory=list)
    sentences_used: int = 0
    coverage: float = 0.0
    sources: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.sentences_used == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": self.citations,
            "sentences_used": self.sentences_used,
            "coverage": round(self.coverage, 3),
            "sources": self.sources,
        }


def split_sentences(text: str) -> list[str]:
    """Break a chunk into sentence-ish pieces, keeping table rows whole."""
    pieces: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _TABLE_ROW.match(stripped):
            # A table rule row (|---|---|) carries no information for a reader.
            if set(stripped) <= set("|-: "):
                continue
            pieces.append(_MARKDOWN_NOISE.sub("", stripped))
            continue
        cleaned = _MARKDOWN_NOISE.sub("", stripped)
        if not cleaned:
            continue
        pieces.extend(part.strip() for part in _SENTENCE_BREAK.split(cleaned))
    return [piece for piece in pieces if len(piece) > 2]


def _coverage(question_terms: Counter[str], sentence: str) -> float:
    """What share of the question's content words this sentence contains.

    Counted over the *distinct* question terms rather than weighted by
    frequency: a question asks each thing once, and a sentence that repeats one
    term six times has not answered more of the question than one that mentions
    it twice. This is deliberately not BM25 -- that ranks documents against a
    corpus, and there is no corpus here, only one question and a handful of
    sentences already known to be relevant.
    """
    if not question_terms:
        return 0.0
    present = set(tokenize(sentence)) & set(question_terms)
    return len(present) / len(question_terms)


def answer_from_documents(
    question: str,
    documents: list[Document],
    *,
    max_sentences: int = 6,
    max_chars: int = 1400,
    prefix: str | None = EXTRACTIVE_PREFIX,
    tracer: Any = None,
) -> ExtractiveAnswer:
    """Assemble an answer from retrieved chunks without a language model.

    Sentences are selected by coverage and then emitted in **document order**,
    not in score order. That matters more than it sounds: policy text is
    sequential -- a condition and its exception, a rate and the period it
    applies to -- and reordering by relevance routinely puts the exception
    before the rule it is an exception to, which reads as a contradiction.

    Returns an empty `ExtractiveAnswer` for no documents or no sentence over the
    coverage floor, so the caller can fall through to its own "insufficient
    evidence" copy rather than this module inventing one.
    """
    documents = [d for d in documents if (d.page_content or "").strip()]
    if not documents:
        return ExtractiveAnswer(text="")

    question_terms = Counter(tokenize(question))

    # (document index, sentence index, coverage, text) so document order can be
    # restored after ranking.
    scored: list[tuple[int, int, float, str]] = []
    for doc_index, document in enumerate(documents):
        for sent_index, sentence in enumerate(split_sentences(document.page_content)):
            score = _coverage(question_terms, sentence)
            if score >= _MIN_COVERAGE:
                scored.append((doc_index, sent_index, score, sentence))

    if not scored:
        if tracer is not None:
            tracer.event(
                "extractive.empty", kind="step", candidates=len(documents)
            )
        return ExtractiveAnswer(text="", citations=cite_sources(documents))

    # Best by coverage, longer sentence breaking a tie -- a longer sentence at
    # equal coverage carries more of the surrounding condition.
    scored.sort(key=lambda row: (-row[2], -len(row[3])))
    chosen = scored[:max_sentences]

    # Then back into document order for the reason above.
    chosen.sort(key=lambda row: (row[0], row[1]))

    lines: list[str] = []
    used_documents: list[Document] = []
    budget = max_chars
    last_doc = -1
    for doc_index, _sent_index, _score, sentence in chosen:
        if budget <= 0:
            break
        if doc_index != last_doc:
            document = documents[doc_index]
            source = str((document.metadata or {}).get("source", "unknown"))
            lines.append(f"\n**{source}**")
            used_documents.append(document)
            last_doc = doc_index
        clipped = sentence[:budget]
        lines.append(f"- {clipped}")
        budget -= len(clipped)

    body = "\n".join(lines).strip()
    coverage = max(row[2] for row in chosen)
    answer = ExtractiveAnswer(
        text=f"{prefix or ''}{body}",
        citations=cite_sources(used_documents),
        sentences_used=sum(1 for line in lines if line.startswith("- ")),
        coverage=coverage,
        sources=[str((d.metadata or {}).get("source", "unknown")) for d in used_documents],
    )
    if tracer is not None:
        tracer.event(
            "extractive.answer",
            kind="step",
            sentences=answer.sentences_used,
            coverage=round(coverage, 3),
            sources=len(answer.sources),
        )
    return answer
