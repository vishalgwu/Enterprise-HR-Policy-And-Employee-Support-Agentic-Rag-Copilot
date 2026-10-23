"""What the routes depend on, and how a request gets hold of it.

Everything a route needs is built once in `create_app` and put on
`app.state`; these functions read it back off the request. That keeps the
injection property the rest of the codebase has -- `build_nodes`, `build_graph`,
`AnswerPipeline` and `Copilot` all take their dependencies as arguments -- all
the way up to the HTTP layer, so a test builds an app over a fake copilot and a
temporary database without touching the network or the real settings.

`dependency_overrides` would do the same job, but it is a test hook. Wiring
production through a test hook makes it impossible to tell, later, which
overrides are the app and which are the test.

**Two gates, and they are not alternatives.** `require_admin` is the original
one: a single key, a single role, refusing before it looks at the header. It
still guards `/api/audit`, because every row there is an employee's question.
`require_action` is the newer and finer one: it resolves a `Principal` from the
presented keys and asks the permission table in
`app/services/authorization.py`. That is what lets HR work the approval queue
without also gaining the ability to read everyone's questions -- which one key
and one role could not express.

**Every caller gets a `Principal`, including an anonymous one.** `/api/chat` is
open by design, so the common case is `role="employee"` with a hashed subject
and no key at all. That is a real principal with real (small) permissions, not
an absence of one -- which is what makes the tool registry's authorization
checks meaningful on the open endpoint rather than only on the gated ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import Settings, get_logger
from app.services.audit import AuditStore
from app.services.authorization import (
    Principal,
    can,
    fingerprint_subject,
    principal_for,
)
from app.services.byok import (
    CredentialError,
    ProviderCredentials,
    credentials_from_headers,
)
from app.services.copilot import Copilot

log = get_logger("api")


# --- Services off app.state --------------------------------------------------


def get_settings_dep(request: Request) -> Settings:
    """The Settings this application was built over -- not the process-wide one."""
    return request.app.state.settings


def get_copilot_dep(request: Request) -> Copilot:
    """The shared Copilot. Its pipeline is built on first use, not at startup."""
    return request.app.state.copilot


def get_audit_dep(request: Request) -> AuditStore:
    """The shared audit store. Its schema is created on first use."""
    return request.app.state.audit


def get_approvals_dep(request: Request) -> Any:
    """The approval queue -- the human in the loop."""
    return request.app.state.approvals


def get_scratchpad_dep(request: Request) -> Any:
    """The scratchpad the agent and the user both write to."""
    return request.app.state.scratchpad


def get_cache_dep(request: Request) -> Any:
    """The semantic cache, so an admin can drop it after an ingest."""
    return request.app.state.cache


def get_limiter_dep(request: Request) -> Any:
    """The rate limiter in front of the deployment's own provider key."""
    return request.app.state.limiter


def get_pool_dep(request: Request) -> Any:
    """The pool of per-credential copilots. See `app/services/byok.py`."""
    return request.app.state.pool


# --- Identity ----------------------------------------------------------------


def get_credentials_dep(
    request: Request, settings: Settings = Depends(get_settings_dep)
) -> ProviderCredentials | None:
    """A caller-supplied provider key, or None -- which is the ordinary case.

    A key that *was* supplied and is refused becomes a 400 naming the rule,
    never the value: `CredentialError`'s messages are written for the person who
    pasted the key, and they say "a groq key normally starts with gsk_" rather
    than echoing what they pasted. Forwarding a wrong key instead would produce
    a provider 401 this layer is not allowed to repeat, so the caller would see
    "something went wrong" for a mistake we could name exactly.
    """
    try:
        return credentials_from_headers(request.headers, settings)
    except CredentialError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


def get_principal_dep(
    request: Request,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    x_hr_key: str | None = Header(default=None, alias="X-HR-Key"),
    settings: Settings = Depends(get_settings_dep),
    credentials: ProviderCredentials | None = Depends(get_credentials_dep),
) -> Principal:
    """Who is asking, as far as this deployment can tell.

    The subject is a salted hash of the client address and user agent, never
    either of them: it is written to the approval queue, the fact store and the
    audit row, all of which outlive the request, and it only ever has to answer
    "is this the same caller as last time?". `fingerprint_subject` explains why
    an unsalted hash of an IPv4 address is not anonymisation.

    The user agent is in the material deliberately. The address alone collapses
    an office behind one NAT into a single rate-limit bucket; adding the agent
    separates most of them without identifying anybody.
    """
    subject = fingerprint_subject(
        request.client.host if request.client else None,
        request.headers.get("user-agent"),
        settings=settings,
    )
    return principal_for(
        settings,
        x_admin_key,
        x_hr_key,
        subject=subject,
        byok=credentials is not None,
    )


# --- Gates -------------------------------------------------------------------


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


def require_action(action: str) -> Callable[..., Principal]:
    """A dependency refusing anyone who does not hold `action`.

    Returns the principal so a route that needs both the check and the identity
    gets them from one dependency -- `Depends(require_action("approval.decide"))`
    both gates the route and hands it the principal whose name goes in
    `decided_by`.

    The refusal is a 403 naming the action and the role, and nothing else. It
    cannot name the key, because it never has it; `Principal` holds a role and a
    hash. And it is 403 rather than 404: pretending the route does not exist
    would also hide it from the console, which decides what to *offer* from the
    same permission table.

    An HR or admin action attempted with no key configured for that role is
    refused here like any other, which is the same fail-closed property
    `require_admin` has -- an unconfigured role cannot be assumed.
    """

    def dependency(
        principal: Principal = Depends(get_principal_dep),
    ) -> Principal:
        if not can(principal, action):
            log.warning(
                "Refused %r to role %r", action, principal.role
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"The {principal.role!r} role may not {action}. "
                    f"Present the key for a role that holds it."
                ),
            )
        return principal

    return dependency
