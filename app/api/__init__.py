"""FastAPI routes.

Only /health so far. The endpoints in docs/architecture.png -- /chat, /upload,
/ingest, /feedback, /admin, /logs -- are the next milestone and belong here, one
router per concern, each calling `app.services` rather than the agent directly.
"""
