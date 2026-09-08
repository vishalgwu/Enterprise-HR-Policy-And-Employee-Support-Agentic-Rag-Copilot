"""Rebuild the HR knowledge base in Pinecone. Root entry point.

    python ingest_sample_kb.py
    python ingest_sample_kb.py --dir data/other_kb
    python ingest_sample_kb.py --drop-namespace private-kb

A pass-through to `scripts/ingest_private_kb.py`, arguments and all -- there is
one ingest implementation, and this is a second door onto it rather than a
second copy of it.

The name is the one the project brief asks for (step 7). The corpus it actually
loads is `data/private_kb/`, which holds the documents the brief calls "sample":
leave, remote work, attendance, payroll, benefits, onboarding and offboarding,
conduct, employee records and HR forms. There is deliberately no second
`data/sample_kb/` alongside it -- two overlapping HR corpora is precisely the
self-contradiction this KB is maintained to avoid, and retrieval routinely draws
chunks from more than one file for a single question.
"""

from scripts.ingest_private_kb import main

if __name__ == "__main__":
    main()
