"""FastAPI routes, one router per audience.

    routes.ops   unprefixed and open: /health, which a load balancer polls and
                 which must answer at every environment and without credentials
    routes.api   prefixed /api: /chat for employees, /upload and /audit for
                 admins behind the X-Admin-Key gate in `deps.require_admin`

Routes call `app.services`, never the agent or Pinecone directly, and take what
they need through the dependencies in `deps` so an application can be built over
fakes. `app.main.create_app` wires the two together.
"""
