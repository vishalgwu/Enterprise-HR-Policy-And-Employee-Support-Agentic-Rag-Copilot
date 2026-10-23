"""Who the caller is, and what that lets them do.

Three roles, because the product has three kinds of user and they want
different things. An **employee** asks questions and may ask for a person. **HR
staff** additionally upload policy and decide approvals. An **admin** operates
the deployment. The dividing line that matters is the audit log: every row in
it is an employee's question and the answer they were given, which is
HR-sensitive by definition, so reading it is an admin action and not an HR one
even though HR is the department the product is for.

Roles come from keys, the same two keys `app/api/deps.py` already gates the
admin surface with, and they resolve through `Settings.check_admin_key` /
`check_hr_key` -- constant-time, and False when the role is not configured at
all. **An unconfigured role can never be assumed, including by an empty
header.** That is the rule `require_admin` documents at length and the reason
both keys default to empty rather than to a placeholder: forgetting to set one
leaves that role *off*, never open.

This module decides nothing about HTTP. It raises `PermissionDenied`, and the
API layer is what turns that into a status code -- which keeps the table
usable from the CLI, from a tool call and from a test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings, get_logger

log = get_logger("authorization")

__all__ = [
    "ANONYMOUS",
    "POLICY",
    "PermissionDenied",
    "Principal",
    "Role",
    "can",
    "fingerprint_subject",
    "permissions_for",
    "principal_for",
    "require",
]

Role = Literal["employee", "hr", "admin"]


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is asking. Deliberately almost nothing.

    `subject` is a stable but non-identifying id -- see `fingerprint_subject`.
    It exists so an approval request can say which caller raised it and a
    remembered fact can be scoped to one person, neither of which needs to know
    who that person is.
    """

    role: Role = "employee"
    subject: str = "anonymous"
    department: str | None = None
    byok: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "subject": self.subject,
            "department": self.department,
            "byok": self.byok,
        }


# The caller with no key and no fingerprint. `/api/chat` is unauthenticated by
# design, so this is the common case rather than an edge one.
ANONYMOUS = Principal()


class PermissionDenied(Exception):
    """A role tried to do something its role does not hold.

    Carries the action and the role and nothing else: no key material, no
    provider detail, and nothing about which key was presented. This message
    reaches a log and may reach a response body.
    """

    def __init__(self, action: str, role: str) -> None:
        super().__init__(f"The {role} role may not {action}.")
        self.action = action
        self.role = role


# Action -> the roles that hold it. Dotted and coarse: one entry per thing a
# caller can do, not one per endpoint, so a second route onto the same
# capability cannot disagree with the first about who may use it.
#
# **An action that is not in this table is denied to everybody**, admin
# included -- see `can`. A table that defaults open means adding a tool quietly
# grants it to every anonymous visitor, and the whole value of having a table
# is that granting is a visible edit to it.
POLICY: dict[str, frozenset[Role]] = {
    # Everyone, including an anonymous visitor: this is what the product is.
    "kb.read": frozenset({"employee", "hr", "admin"}),
    "web.search": frozenset({"employee", "hr", "admin"}),
    "tool.run": frozenset({"employee", "hr", "admin"}),
    "scratchpad.write": frozenset({"employee", "hr", "admin"}),
    "memory.write": frozenset({"employee", "hr", "admin"}),
    "escalate": frozenset({"employee", "hr", "admin"}),
    "approval.request": frozenset({"employee", "hr", "admin"}),
    "ticket.create": frozenset({"employee", "hr", "admin"}),
    # HR and above: changing what the copilot answers from, and deciding the
    # things it stopped to ask about.
    "kb.write": frozenset({"hr", "admin"}),
    "approval.list": frozenset({"hr", "admin"}),
    "approval.decide": frozenset({"hr", "admin"}),
    "cache.invalidate": frozenset({"hr", "admin"}),
    "metrics.read": frozenset({"hr", "admin"}),
    # Admin only. Every audit row is an employee's question and the answer they
    # got; HR needing to upload a policy is not a reason to read those.
    "audit.read": frozenset({"admin"}),
}


def can(principal: Principal, action: str) -> bool:
    """Whether this principal holds this action. Unknown actions are denied."""
    holders = POLICY.get(action)
    if holders is None:
        log.warning("Denied unknown action %r; it is not in the policy table", action)
        return False
    return principal.role in holders


def require(principal: Principal, action: str) -> None:
    """Raise `PermissionDenied` unless the principal holds the action."""
    if not can(principal, action):
        raise PermissionDenied(action, principal.role)


def permissions_for(role: Role) -> list[str]:
    """Every action a role holds, sorted. For `/health` and the console."""
    return sorted(action for action, holders in POLICY.items() if role in holders)


def principal_for(
    settings: Settings,
    admin_header: str | None,
    hr_header: str | None = None,
    subject: str = "anonymous",
    byok: bool = False,
) -> Principal:
    """Resolve a role from the presented keys.

    **Admin beats HR**, so a caller presenting both gets the wider role rather
    than whichever header the framework happened to read first. Both checks go
    through `Settings`, which compares in constant time and returns False when
    that role has no key configured -- an unconfigured role cannot be assumed,
    including by an empty header, exactly as `require_admin` in
    `app/api/deps.py` refuses before it looks at the header at all.
    """
    if settings.check_admin_key(admin_header):
        return Principal(role="admin", subject=subject, byok=byok)
    if settings.check_hr_key(hr_header):
        return Principal(role="hr", subject=subject, byok=byok)
    return Principal(role="employee", subject=subject, byok=byok)


def fingerprint_subject(
    *parts: str | None, settings: Settings | None = None
) -> str:
    """A stable, non-identifying id for a caller: 16 hex characters of sha256.

    A hash and not the address it was made from, because the subject is written
    to the approval queue and the fact store -- both of which outlive the
    request -- and an IP address there is personal data this product has no
    reason to keep. It only ever needs to answer "is this the same caller as
    last time?", which a hash answers and an address answers no better.

    **Salted with the admin key when one is configured, and the reason is not
    hypothetical: an unsalted sha256 of an IPv4 address is trivially
    reversible.** There are only four billion of them, so the whole space can be
    enumerated in seconds and the "anonymous" subject in the approval queue
    becomes the address again. A per-deployment salt means the table would have
    to be rebuilt for this deployment specifically, and the salt is a secret
    that is already being kept.

    `settings` is a keyword argument rather than read from the process, so this
    stays usable with no configuration at all and the caller that *has*
    settings is the one that supplies them.
    """
    material = "\x1f".join(part for part in parts if part)
    if not material:
        return "anonymous"
    salt = ""
    if settings is not None and settings.admin_enabled:
        salt = settings.require("admin_api_key")
    return hashlib.sha256(f"{salt}\x1f{material}".encode()).hexdigest()[:16]
