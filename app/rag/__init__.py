"""Retrieval-augmented generation plumbing.

    clients      embeddings (cached), Groq chat model, Tavily search
    loaders      markdown/text/PDF/DOCX/web loading, chunking, chunk ids
    vectorstore  Pinecone index management, parameterised by index and namespace
    ingest       load -> chunk -> embed -> upsert -> reconcile, for any corpus
    retrieval    the private KB and public corpus, with namespaces chosen
"""
