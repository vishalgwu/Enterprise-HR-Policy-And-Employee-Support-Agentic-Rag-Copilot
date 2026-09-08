"""What the routes depend on, and how a request gets hold of it.

Everything a route needs is built once in `create_app` and put on
`app.state`; these functions read it back off the request. That keeps the
injection property the rest of the codebase has -- `build_nodes`, `build_graph`
and `Copilot` all take their dependencies as arguments -- all the way up to the
HTTP layer, so a test builds an app over a fake copilot and a temporary audit
database without touching the network or the real settings.

`dependency_overrides` would do the same job, but it is a test hook. Wiring
production through a test hook makes it impossible to tell, later, which
overrides are the app and which are the test.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import Settings, get_logger
from app.services.audit import AuditStore
from app.services.copilot import Copilot

log = get_logger("api")


def get_settings_dep(request: Request) -> Settings:
    """The Settings this application was built over -- not the process-wide one."""
    return request.app.state.settings


def get_copilot_dep(request: Request) -> Copilot:
    """The shared Copilot. Its graph is built on first use, not at startup."""
    return request.app.state.copilot


def get_audit_dep(request: Request) -> AuditStore:
    """The shared audit store. Its schema is created on first use."""
    return request.app.state.audit


def require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    settings: Settings = Depends(get_settings_dep),
) -> None:
    """Refuse anything that is not an authenticated admin.

    Two refusals, and the order matters. **An unconfigured admin key refuses
    every request**, including one presenting an empty key: the default is empty
    precisely so that forgetting to set it leaves admin *off* rather than open.
    A reference implementation shipped `"change-me-in-production"` as the
    default and compared with `!=`, which means an unset key plus an absent
    header compares equal and every caller is an admin.

    The key itself goes through `check_admin_key`, never `==`: a plain
    comparison on a secret leaks its length and prefix through timing.
    """
    if not settings.admin_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The admin API is not configured on this deployment.",
        )
    if not settings.check_admin_key(x_admin_key):
        log.warning("Rejected an admin request with a missing or wrong key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-Admin-Key header is required.",
            headers={"WWW-Authenticate": "X-Admin-Key"},
        )
