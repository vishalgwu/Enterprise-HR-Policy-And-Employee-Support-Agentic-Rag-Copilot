"""Enterprise HR Policy & Employee Support Agentic RAG Copilot.

Layering, lowest first. Each layer imports only from those above it, and nothing
imports downward:

    core      settings, logging, console setup; no heavy dependencies
    rag       embeddings, loading, chunking, Pinecone, ingest, retrieval
    agent     decision schemas, prompts, nodes, graph wiring
    services  the facade the API and the CLI both call
    api       FastAPI routes

Import from the layer modules, never from a script:

    from app.core.config import get_settings
    from app.rag.retrieval import get_kb_retriever
    from app.services.copilot import Copilot
"""

__version__ = "0.2.0"
