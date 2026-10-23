"""BM25 over the knowledge base on disk: the no-key half, and the sparse half.

Two jobs, and it earns its place through the second one even on a deployment
that has every key.

**The fallback.** The dense retriever needs a Pinecone key, an embedding model
and a network round trip. A deployment with none of those -- a public demo, a
serverless function that cannot ship torch's 2.5 GB -- answers nothing at all,
which is a worse failure than answering less well. BM25 over the markdown
already sitting in `data/private_kb/` needs no key, no model and no network.

**The sparse side of hybrid search.** It is also genuinely better than
embeddings at the questions employees actually ask with exact terms in them: a
form number, "IJP", "Form 16", the name of one policy. An embedding of "form 16"
sits very near an embedding of "form 12" and cosine similarity has no way to
care which digits were asked for; BM25 cares about nothing else. So the two
retrievers fail differently, which is the argument for `app.rag.hybrid`.

Implemented here rather than taken from `rank_bm25`: it is forty lines of
arithmetic against one more pin, and the IDF variant is a decision this corpus
has an opinion about -- see `BM25Index.idf`, which is the one part of this file
that is not textbook.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings, get_logger, get_settings
from app.rag.loaders import load_directory, split_documents

log = get_logger("lexical")

__all__ = [
    "STOPWORDS",
    "BM25Index",
    "LexicalRetriever",
    "build_kb_index",
    "tiebreak_key",
    "tokenize",
]

_NON_WORD = re.compile(r"[^a-z0-9]+")

# A small English list. Deliberately short, and deliberately free of this
# corpus's own vocabulary: "policy", "leave", "employee" and "pay" appear in
# most of the knowledge base and look like noise, but they are exactly the words
# an HR question is made of. Holding their weight down is the IDF's job, not a
# stopword list's -- a term removed here cannot be matched at all.
STOPWORDS: frozenset[str] = frozenset(
    {
        "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
        "at", "be", "been", "being", "but", "by", "can", "did", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "him", "his", "how",
        "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of",
        "on", "or", "our", "out", "over", "she", "should", "so", "such", "than",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "to", "too", "up", "was", "we", "were", "what", "when",
        "where", "which", "who", "why", "will", "with", "would", "you", "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit, then filter.

    Splitting on non-alphanumerics is what makes a hyphenated policy term
    useful rather than unmatchable: `work-from-home` becomes `work` and `home`,
    so a question typed without the hyphens still hits it, and
    `leave-and-time-off.md` cited by a model still matches the file it names.

    Single characters go too. They are initials, list bullets and the stray `s`
    left behind by a possessive, and every one of them appears in most chunks,
    so they cost a posting list each and separate nothing.
    """
    return [
        token
        for token in _NON_WORD.split(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def tiebreak_key(document: Document) -> tuple[str, int, str]:
    """A total order over documents, for breaking a scoring tie the same way twice.

    Shared with `app.rag.rerank` because both modules sort by a score that ties
    constantly -- an LLM re-rank hands out integers, and four candidates sharing
    a 7 is the common case. A tie broken by whatever order the candidates
    happened to arrive in makes the top-k flap between identical runs, which
    reads as a retrieval bug and is a sorting one.

    `page_content` is the last resort rather than the first: two chunks of one
    file are separated by `start_index`, and only a document with no offset at
    all (hand-built, or derived from a web result) needs its text compared.
    """
    metadata = document.metadata or {}
    try:
        start = int(metadata.get("start_index") or 0)
    except (TypeError, ValueError):
        start = 0
    return str(metadata.get("source", "")), start, document.page_content


# --- The index ---------------------------------------------------------------


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 over a fixed set of chunks.

    Built once and read many times: nothing here mutates after
    `from_documents`, which is what makes it safe to cache per directory and to
    share between threads.
    """

    documents: list[Document] = field(default_factory=list)
    term_counts: list[Counter[str]] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    doc_freq: Counter[str] = field(default_factory=Counter)
    postings: dict[str, list[int]] = field(default_factory=dict)
    avg_length: float = 0.0
    # Term-frequency saturation: the point past which repeating a term stops
    # adding much. 1.5 is the usual value and there is nothing in this corpus
    # that argues for another.
    k1: float = 1.5
    # Length normalisation. 0.75 means a long chunk is penalised, but not in
    # full proportion to its length -- which matters here because the splitter
    # emits chunks of very uneven size from headings-first markdown.
    b: float = 0.75

    @classmethod
    def from_documents(
        cls, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        docs = list(documents)
        term_counts = [Counter(tokenize(doc.page_content)) for doc in docs]
        lengths = [sum(counts.values()) for counts in term_counts]

        doc_freq: Counter[str] = Counter()
        postings: dict[str, list[int]] = {}
        for position, counts in enumerate(term_counts):
            for term in counts:
                doc_freq[term] += 1
                postings.setdefault(term, []).append(position)

        return cls(
            documents=docs,
            term_counts=term_counts,
            lengths=lengths,
            doc_freq=doc_freq,
            postings=postings,
            avg_length=(sum(lengths) / len(docs)) if docs else 0.0,
            k1=k1,
            b=b,
        )

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def idf(self, term: str) -> float:
        """Inverse document frequency, with the `1 +` guard inside the logarithm.

        The textbook Okapi form is `log((N - df + 0.5) / (df + 0.5))`, which
        turns **negative** as soon as a term appears in more than half the
        corpus. A negative IDF means a document scores *lower* for containing
        the query term than for not containing it, which is not a rounding
        problem -- it inverts the ranking.

        With 31 chunks that is the normal case rather than the corner one:
        "policy", "employee" and "leave" each appear in well over half of this
        knowledge base, and those are the words an HR question is made of. A
        question about leave policy would rank the chunks that never mention
        leave first. The `1 +` floors every term's contribution at zero
        instead, so a near-universal term simply stops discriminating.
        """
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """The top `k` chunks for a query, with their BM25 scores.

        An empty query, an all-stopword query and an empty index all return
        nothing without touching the postings: there is no term to look up, and
        "no evidence" is a state the graph already handles.
        """
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or self.is_empty or self.avg_length <= 0:
            return []

        scores: dict[int, float] = {}
        for term in terms:
            hits = self.postings.get(term)
            if not hits:
                continue
            weight = self.idf(term)
            if weight <= 0:
                continue
            for position in hits:
                tf = self.term_counts[position][term]
                length_norm = 1 - self.b + self.b * self.lengths[position] / self.avg_length
                scores[position] = scores.get(position, 0.0) + weight * (
                    tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                )

        ranked = [
            (self.documents[position], score)
            for position, score in scores.items()
            if score > 0
        ]
        ranked.sort(key=lambda pair: (-pair[1], *tiebreak_key(pair[0])))
        return ranked[:k] if k else ranked


class LexicalRetriever:
    """A retriever interface over a `BM25Index`, drop-in for the dense one.

    `invoke(query) -> list[Document]` is the whole contract the retrieval node
    and `app.rag.hybrid` depend on, which is why this is a plain class rather
    than a LangChain `BaseRetriever`: nothing here needs a callback manager or a
    run tree, and subclassing one would make the keyless fallback depend on the
    framework machinery it exists to do without.
    """

    def __init__(self, index: BM25Index, k: int = 4, min_score: float = 0.0) -> None:
        self.index = index
        self.k = k
        self.min_score = min_score

    def invoke(self, query: str, *args: object, **kwargs: object) -> list[Document]:
        return [
            document
            for document, score in self.index.search(query, k=self.k)
            if score >= self.min_score
        ]

    def __len__(self) -> int:
        return len(self.index)


# --- Building one from the corpus on disk ------------------------------------


def _newest_mtime(directory: Path) -> float:
    """The newest modification time under a directory, or 0.0 when unreadable."""
    try:
        return max(
            (path.stat().st_mtime for path in directory.rglob("*") if path.is_file()),
            default=0.0,
        )
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _cached_index(
    directory: str,
    newest_mtime: float,
    origin: str,
    chunk_size: int,
    chunk_overlap: int,
) -> BM25Index:
    documents = load_directory(directory, origin=origin)
    chunks = split_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    log.info(
        "Built a BM25 index over %d chunk(s) from %d document(s) in %s",
        len(chunks),
        len(documents),
        directory,
    )
    return BM25Index.from_documents(chunks)


def build_kb_index(
    kb_dir: Path | str | None = None,
    settings: Settings | None = None,
    origin: str = "hr-docs",
) -> BM25Index:
    """A BM25 index over the private knowledge base. Pure disk, no keys.

    Cached, because in the worst case this is called once per request and
    loading plus splitting eleven markdown files is not free.

    **The newest mtime is part of the cache key, and that is the whole reason
    the key is not just the directory.** The failure this guards against is an
    *edited* policy continuing to be answered from the text it used to have --
    an HR copilot serving withdrawn policy, which is the same failure
    `prune_orphans` exists to prevent on the dense side. A file count or a
    directory name would notice a new document and miss a correction, which is
    the direction that matters: corrections are what editing a policy is for.
    """
    settings = settings or get_settings()
    directory = Path(kb_dir or settings.kb_dir)
    return _cached_index(
        str(directory.resolve()),
        _newest_mtime(directory),
        origin,
        settings.chunk_size,
        settings.chunk_overlap,
    )
