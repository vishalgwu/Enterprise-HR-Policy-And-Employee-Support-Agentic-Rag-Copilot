"""The permission table, and how a role is resolved from a presented key."""

from __future__ import annotations

import pytest
from conftest import settings_for

from app.services.authorization import (
    ANONYMOUS,
    POLICY,
    PermissionDenied,
    Principal,
    can,
    fingerprint_subject,
    permissions_for,
    principal_for,
    require,
)

ROLES = ("employee", "hr", "admin")


def test_the_table_grants_what_it_says_and_nothing_more():
    employee = Principal(role="employee")
    hr = Principal(role="hr")
    admin = Principal(role="admin")

    assert can(employee, "kb.read") is True
    assert can(employee, "escalate") is True
    assert can(employee, "kb.write") is False
    assert can(employee, "approval.decide") is False
    assert can(employee, "audit.read") is False

    assert can(hr, "kb.write") is True
    assert can(hr, "approval.decide") is True
    assert can(hr, "metrics.read") is True
    # Every audit row is an employee's question and the answer they got.
    assert can(hr, "audit.read") is False

    assert can(admin, "audit.read") is True
    assert all(can(admin, action) for action in POLICY)


def test_an_unknown_action_is_denied_to_every_role_including_admin():
    # A table that defaults open means adding a tool quietly grants it to
    # everyone, which is the opposite of why the table exists.
    for role in ROLES:
        assert can(Principal(role=role), "kb.incinerate") is False


def test_permissions_for_lists_a_role_and_grows_with_it():
    employee = permissions_for("employee")
    hr = permissions_for("hr")
    admin = permissions_for("admin")

    assert employee == sorted(employee)
    assert set(employee) < set(hr) < set(admin)
    assert "audit.read" in admin
    assert "audit.read" not in hr


def test_require_raises_permission_denied_naming_the_action():
    with pytest.raises(PermissionDenied) as raised:
        require(Principal(role="employee"), "approval.decide")
    assert raised.value.action == "approval.decide"
    assert raised.value.role == "employee"
    assert "approval.decide" in str(raised.value)


def test_require_is_silent_when_the_role_holds_the_action():
    assert require(Principal(role="hr"), "kb.write") is None


def test_the_anonymous_principal_is_an_employee_and_not_an_admin():
    assert ANONYMOUS.role == "employee"
    assert ANONYMOUS.is_admin is False
    assert ANONYMOUS.subject == "anonymous"
    assert ANONYMOUS.to_dict()["byok"] is False


def test_admin_beats_hr_when_both_keys_are_presented():
    settings = settings_for(ADMIN_API_KEY="admin-key", HR_API_KEY="hr-key")
    principal = principal_for(settings, "admin-key", "hr-key")
    assert principal.role == "admin"
    assert principal.is_admin is True


def test_the_hr_key_alone_resolves_to_hr_and_not_to_admin():
    settings = settings_for(ADMIN_API_KEY="admin-key", HR_API_KEY="hr-key")
    assert principal_for(settings, None, "hr-key").role == "hr"
    # The HR key in the admin header is still only the HR key.
    assert principal_for(settings, "hr-key", None).role == "employee"


def test_an_unconfigured_role_cannot_be_assumed_by_an_empty_or_absent_header():
    settings = settings_for()
    assert settings.admin_enabled is False
    assert settings.hr_enabled is False
    for admin_header, hr_header in ((None, None), ("", ""), ("anything", "anything")):
        assert principal_for(settings, admin_header, hr_header).role == "employee"


def test_a_wrong_key_resolves_to_employee():
    settings = settings_for(ADMIN_API_KEY="admin-key", HR_API_KEY="hr-key")
    assert principal_for(settings, "nearly-admin-key", "nope").role == "employee"


def test_the_subject_and_byok_flag_travel_with_the_principal():
    settings = settings_for(ADMIN_API_KEY="admin-key")
    principal = principal_for(settings, "admin-key", subject="abc123", byok=True)
    assert principal.subject == "abc123"
    assert principal.byok is True
    assert principal.to_dict() == {
        "role": "admin",
        "subject": "abc123",
        "department": None,
        "byok": True,
    }


def test_the_same_inputs_fingerprint_to_the_same_subject():
    first = fingerprint_subject("203.0.113.7", "Mozilla/5.0")
    second = fingerprint_subject("203.0.113.7", "Mozilla/5.0")
    assert first == second
    assert len(first) == 16
    assert all(char in "0123456789abcdef" for char in first)


def test_different_inputs_fingerprint_differently():
    assert fingerprint_subject("203.0.113.7") != fingerprint_subject("203.0.113.8")
    assert fingerprint_subject("a", "b") != fingerprint_subject("ab")


def test_the_fingerprint_is_not_the_raw_input():
    # The subject is written to the approval queue and the fact store, both of
    # which outlive the request. An address there is personal data this
    # product has no reason to keep.
    address = "203.0.113.7"
    assert address not in fingerprint_subject(address, "Mozilla/5.0")


def test_a_configured_admin_key_salts_the_fingerprint():
    # An unsalted sha256 of an IPv4 address is trivially reversible: the whole
    # space is four billion entries. A per-deployment salt is what stops one
    # rainbow table covering every deployment.
    salted = fingerprint_subject(
        "203.0.113.7", settings=settings_for(ADMIN_API_KEY="admin-key")
    )
    assert salted != fingerprint_subject("203.0.113.7")
    assert salted != fingerprint_subject(
        "203.0.113.7", settings=settings_for(ADMIN_API_KEY="other-key")
    )


def test_no_parts_fingerprint_to_the_anonymous_subject():
    assert fingerprint_subject() == "anonymous"
    assert fingerprint_subject(None, "") == "anonymous"
