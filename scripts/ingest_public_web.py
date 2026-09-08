"""Scrape the public HR policy article into its own namespace.

    python scripts/ingest_public_web.py

A development corpus, not part of the product in docs/: it exercises the
web-loading path end to end and gives the retrieval code a second corpus to
prove it is genuinely namespace-parameterised. The employee-facing agent only
ever reads the private namespace.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/ingest_public_web.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from app.core.config import (
    HR_POLICY_CONTENT_CLASS,
    HR_POLICY_URL,
    configure_logging,
    configure_stdout,
    get_settings,
)
from app.rag.ingest import ingest_web_page
from app.rag.vectorstore import namespace_counts


def main() -> None:
    configure_stdout()
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=HR_POLICY_URL)
    parser.add_argument(
        "--content-class",
        default=HR_POLICY_CONTENT_CLASS,
        help="restrict extraction to this CSS class; empty string to disable",
    )
    args = parser.parse_args()
    # After parse_args, so --help works in a checkout with no credentials.
    settings.validate_required()

    report = ingest_web_page(
        args.url,
        namespace=settings.public_namespace,
        content_class=args.content_class or None,
        title="HR Policies and Procedures",
    )

    print(f"Source        {args.url}")
    print(f"Namespace     {settings.public_namespace}")
    print(f"Ingested      {report.summary()}")
    print(f"\nAll namespaces in this index: {namespace_counts()}")


if __name__ == "__main__":
    main()
