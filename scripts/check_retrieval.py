"""Inspect what the private KB returns, gated and ungated.

    python scripts/check_retrieval.py
    python scripts/check_retrieval.py "how do I claim a taxi fare?"
    python scripts/check_retrieval.py --department payroll "bonus schedule"

This is the diagnostic the 0.15 relevance gate was set from: it prints the raw
cosine score of every candidate alongside whether the gate keeps it, so the
threshold can be re-derived from observed numbers rather than guessed.

It also checks an out-of-domain question, which must come back empty -- that is
what stops the agent answering from whatever happened to rank highest.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/check_retrieval.py`
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from app.core.config import configure_logging, configure_stdout, get_settings
from app.rag.clients import get_embeddings
from app.rag.retrieval import get_kb_retriever, retrieve_with_scores
from app.rag.vectorstore import namespace_counts

DEFAULT_QUESTION = "How many PTO days do I get per year?"
OUT_OF_DOMAIN = "What is the capital of France?"


def main() -> None:
    configure_stdout()
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--department", default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args()
    # After parse_args, so --help works in a checkout with no credentials.
    settings.validate_required()

    embeddings = get_embeddings()
    print(f"Index         {settings.pinecone_index}")
    print(f"Namespace     {settings.private_namespace}")
    print(f"Namespaces    {namespace_counts()}")
    print(f"Gate          {settings.relevance_threshold} raw cosine\n")

    retriever = get_kb_retriever(
        k=args.k,
        embedding=embeddings,
        department=args.department,
        doc_type=args.doc_type,
        settings=settings,
    )

    print(f"--- {args.question!r} ---")
    docs = retriever.invoke(args.question)
    print(f"gated retriever returned {len(docs)} chunk(s)")
    for doc, score in retrieve_with_scores(
        args.question,
        k=args.k,
        embedding=embeddings,
        department=args.department,
        doc_type=args.doc_type,
        settings=settings,
    ):
        keep = "keep" if score >= settings.relevance_threshold else "drop"
        metadata = doc.metadata or {}
        print(
            f"  {score:.4f}  {keep}  {metadata.get('source')} "
            f"[{metadata.get('department')}/{metadata.get('doc_type')}]"
        )

    print(f"\n--- OUT OF DOMAIN: {OUT_OF_DOMAIN!r} ---")
    print(f"gated retriever returned {len(retriever.invoke(OUT_OF_DOMAIN))} chunk(s)")
    for doc, score in retrieve_with_scores(
        OUT_OF_DOMAIN, k=2, embedding=embeddings, settings=settings
    ):
        print(f"  ungated would have returned {score:.4f}  {doc.metadata.get('source')}")


if __name__ == "__main__":
    main()
