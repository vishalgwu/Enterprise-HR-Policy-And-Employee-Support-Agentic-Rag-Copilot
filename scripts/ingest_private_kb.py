"""Rebuild the private HR knowledge base in Pinecone.

    python scripts/ingest_private_kb.py
    python scripts/ingest_private_kb.py --dir data/other_kb
    python scripts/ingest_private_kb.py --drop-namespace private-kb

Reconciles rather than only upserting: anything in the namespace without a file
on disk is deleted, so a withdrawn policy stops answering questions.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/ingest_private_kb.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from app.core.config import configure_logging, configure_stdout, get_settings
from app.rag.clients import embedding_dimension, get_embeddings
from app.rag.ingest import ingest_directory
from app.rag.vectorstore import delete_namespace, namespace_counts


def main() -> None:
    configure_stdout()
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", default=None, help="source directory")
    parser.add_argument(
        "--namespace", default=None, help="target namespace (default: PRIVATE_NAMESPACE)"
    )
    parser.add_argument(
        "--drop-namespace",
        default=None,
        metavar="NAME",
        help="delete this namespace first -- for cleaning up after a rename",
    )
    args = parser.parse_args()
    # After parse_args, so --help works in a checkout with no credentials.
    settings.validate_required()

    namespace = args.namespace or settings.private_namespace
    directory = args.dir or settings.kb_dir

    embeddings = get_embeddings()
    dim = embedding_dimension(embeddings)
    if dim != settings.embedding_dim:
        # active_embedding_model, not embedding_model: the latter is the
        # HuggingFace model regardless of provider, so naming it here would
        # misdirect anyone running EMBEDDING_PROVIDER=openai.
        raise SystemExit(
            f"{settings.active_embedding_model} returned {dim}-d vectors but "
            f"embedding_dim is {settings.embedding_dim}. Update EMBEDDING_DIM "
            f"and point PINECONE_INDEX at a fresh name -- Pinecone cannot "
            f"resize an index."
        )

    if args.drop_namespace:
        dropped = delete_namespace(args.drop_namespace)
        print(
            f"Namespace {args.drop_namespace!r}: "
            + ("deleted" if dropped else "did not exist")
        )

    report = ingest_directory(directory, namespace=namespace, embedding=embeddings)

    print(f"Source        {directory}")
    print(f"Index         {settings.pinecone_index}")
    print(f"Namespace     {namespace}")
    print(f"Ingested      {report.summary()}")
    print(f"Departments   {report.departments}")
    for source in report.sources:
        print(f"  - {source}")
    print(f"\nAll namespaces in this index: {namespace_counts()}")


if __name__ == "__main__":
    main()
