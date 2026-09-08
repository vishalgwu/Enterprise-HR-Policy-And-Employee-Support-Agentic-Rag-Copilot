"""Convenience wrapper: rebuild the private KB from data/private_kb/.

Equivalent to `python scripts/ingest_private_kb.py`, kept at the repo root
because that is where the project brief expects to find it.
"""

from scripts.ingest_private_kb import main

if __name__ == "__main__":
    main()
