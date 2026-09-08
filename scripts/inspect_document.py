"""Show what ingest would do to one document, without touching the network.

    python scripts/inspect_document.py
    python scripts/inspect_document.py data/private_kb/leave-and-time-off.md
    python scripts/inspect_document.py path/to/handbook.pdf     # .md .txt .pdf .docx

Loading and chunking are the bottom of the stack the agent stands on, and they
are the half of ingest that can be checked for free: no embeddings, no Pinecone,
no credentials. If this prints a plausible chunk count and stable ids, the only
thing left for a real ingest to get wrong is the upsert.

Chunk ids are printed because they are the thing to watch after editing a
policy: an edited chunk should keep its id and update in place, and a chunk that
changed id will be written as a second vector and pruned only on a full ingest.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/inspect_document.py`
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from app.core.config import configure_logging, configure_stdout, get_settings
from app.services.ingestion import (
    DEFAULT_DEPARTMENT,
    SUPPORTED,
    chunk_documents,
    chunk_id,
    load_file,
)

DEFAULT_DOCUMENT = Path("data/private_kb/hr-copilot-how-it-works.md")


def main() -> int:
    configure_stdout()
    configure_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_DOCUMENT,
        help=f"document to inspect (default: {DEFAULT_DOCUMENT})",
    )
    # origin is part of the chunk id, so leaving it at load_file's "upload"
    # default would print ids that no real ingest ever produces -- which defeats
    # the point of printing them. Default to the corpus this file most likely
    # belongs to, and let an upload be simulated with --origin upload.
    parser.add_argument(
        "--origin",
        default=None,
        help="corpus the document belongs to (default: PRIVATE_NAMESPACE)",
    )
    parser.add_argument(
        "--department",
        default=DEFAULT_DEPARTMENT,
        help=f"department metadata (default: {DEFAULT_DEPARTMENT})",
    )
    args = parser.parse_args()
    path: Path = args.path

    if not path.is_file():
        print(f"No such file: {path}")
        return 1
    if path.suffix.lower() not in SUPPORTED:
        print(
            f"{path.suffix or '(no extension)'} is not ingestable. "
            f"Supported: {', '.join(sorted(SUPPORTED))}"
        )
        return 1

    # Settings for the chunk sizes and the namespace, so what is printed is what
    # a real ingest would produce rather than library defaults.
    settings = get_settings()
    origin = args.origin or settings.private_namespace

    documents = load_file(path, origin=origin, department=args.department)
    if not documents:
        print(f"{path.name} has no extractable text")
        return 1

    chunks = chunk_documents(documents, settings=settings)

    document = documents[0]
    print(f"File          {path}")
    print(f"Origin        {origin}")
    print(f"Chunking      size={settings.chunk_size} overlap={settings.chunk_overlap}")
    print(f"Result        {len(document.page_content)} chars -> {len(chunks)} chunk(s)")
    print(f"Metadata      {document.metadata}")
    print()
    for chunk in chunks:
        preview = " ".join(chunk.page_content.split())[:72]
        print(f"  {chunk_id(chunk)}  {preview}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
