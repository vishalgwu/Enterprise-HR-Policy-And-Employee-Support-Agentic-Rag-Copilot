"""Settings behaviour, including the two rules that are easy to regress."""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings, get_settings


def test_short_env_names_map_to_fields():
    """.env uses GROQ_API; the field is groq_api_key."""
    s = Settings(GROQ_API="abc", TAVILY_API="def", PINECONE_API="ghi")
    assert s.groq_api_key == "abc"
    assert s.tavily_api_key == "def"
    assert s.pinecone_api_key == "ghi"


def test_missing_secret_is_not_fatal_at_construction():
    """Importing without credentials must stay possible: tests, --help, Docker."""
    s = Settings(GROQ_API="", TAVILY_API="", PINECONE_API="")
    assert s.missing_secrets() == ["GROQ_API", "TAVILY_API", "PINECONE_API"]


def test_require_names_the_env_var_the_user_has_to_set():
    s = Settings(GROQ_API="")
    with pytest.raises(ValueError, match="GROQ_API"):
        s.require("groq_api_key")


def test_require_returns_the_stripped_value():
    assert Settings(GROQ_API="  abc  ").require("groq_api_key") == "abc"


def test_validate_required_lists_everything_missing_at_once():
    s = Settings(GROQ_API="set", TAVILY_API="", PINECONE_API="")
    with pytest.raises(ValueError) as excinfo:
        s.validate_required()
    message = str(excinfo.value)
    assert "TAVILY_API" in message and "PINECONE_API" in message
    assert "GROQ_API," not in message


def test_export_client_env_publishes_the_names_clients_read():
    """ChatGroq and friends read the environment in their constructors."""
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"):
        os.environ.pop(name, None)
    Settings(GROQ_API="g", TAVILY_API="t", PINECONE_API="p").export_client_env()
    assert os.environ["GROQ_API_KEY"] == "g"
    assert os.environ["TAVILY_API_KEY"] == "t"
    assert os.environ["PINECONE_API_KEY"] == "p"


def test_export_client_env_does_not_blank_an_existing_value():
    os.environ["GROQ_API_KEY"] = "already-set"
    Settings(GROQ_API="").export_client_env()
    assert os.environ["GROQ_API_KEY"] == "already-set"


def test_namespaces_are_explicit_strings():
    """The Pinecone default namespace ("") behaves inconsistently across calls."""
    s = Settings()
    assert s.private_namespace and s.public_namespace
    assert s.private_namespace != s.public_namespace


def test_settings_are_immutable():
    with pytest.raises(Exception):
        Settings().pinecone_index = "something-else"  # type: ignore[misc]


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
