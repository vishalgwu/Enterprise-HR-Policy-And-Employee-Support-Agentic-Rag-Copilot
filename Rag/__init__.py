"""HR Copilot retrieval layer.

Layering, lowest first:

    config      settings and secrets; no heavy imports
    clients     embeddings, chat LLM, web search
    loaders     document loading, chunking, rendering
    vectorstore Pinecone index management and retrieval, per namespace
    schemas     structured routing/grading decisions and agent state

`Agentic_rag` and `private_kb` are the two corpus entry points built on those.
Import from this package rather than reaching into the entry points:

    from Rag import clients, config, vectorstore
    from Rag.private_kb import get_kb_retriever
"""
