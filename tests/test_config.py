"""Settings behaviour, including the two rules that are easy to regress."""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings, get_settings


def test_short_env_names_map_to_fields():
    """.env uses GROQ_API; the field is groq_api_key."""
    s = Settings(GROQ_API="abc", TAVILY_API="def", PINECONE_API="ghi")
    assert s.require("groq_api_key") == "abc"
    assert s.require("tavily_api_key") == "def"
    assert s.require("pinecone_api_key") == "ghi"


def test_secrets_never_appear_in_a_repr_or_a_traceback():
    """`log.info("%s", settings)` or an unhandled exception must not leak keys."""
    s = Settings(
        GROQ_API="gsk-supersecret",
        TAVILY_API="tvly-supersecret",
        PINECONE_API="pc-supersecret",
        OPENAI_API_KEY="sk-supersecret",
        LANGSMITH_API_KEY="lsv2-supersecret",
    )
    for rendered in (repr(s), str(s), str(s.model_dump())):
        assert "supersecret" not in rendered
    # ...and the real values are still reachable deliberately.
    assert s.require("groq_api_key") == "gsk-supersecret"


def test_a_validation_error_does_not_echo_the_input_secrets():
    """A model_validator that raises puts the whole input dict in the error.

    Observed: a dimension mismatch produced a pydantic ValidationError whose
    text began `input_value={'GROQ_API': 'gsk_...`. That error reaches logs and
    tracebacks, so the check that can fail lives outside validation.
    """
    s = Settings(
        GROQ_API="gsk-supersecret", EMBEDDING_PROVIDER="openai", EMBEDDING_DIM=384
    )
    with pytest.raises(ValueError) as excinfo:
        s.validate_embedding()
    assert "supersecret" not in str(excinfo.value)
    assert "contradicts" in str(excinfo.value)


def test_missing_secret_is_not_fatal_at_construction():
    """Importing without credentials must stay possible: tests, --help, Docker."""
    s = Settings(GROQ_API="", TAVILY_API="", PINECONE_API="")
    assert sorted(s.missing_secrets()) == ["GROQ_API", "PINECONE_API", "TAVILY_API"]


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


def _clear_tracing_env():
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        os.environ.pop(name, None)


def test_tracing_is_off_by_default(monkeypatch):
    """It uploads employee questions, so it must be a decision, not a default.

    Isolated from both sources a real process has: the developer's `.env`
    (`_env_file=None`) and `os.environ`, which `export_client_env` has already
    written to at import. Environment outranks the file, so clearing only one of
    them still reads local configuration and the test would pass or fail by
    machine rather than by behaviour.
    """
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert not Settings(_env_file=None).tracing_enabled


def test_tracing_needs_both_the_switch_and_a_key():
    assert not Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="").tracing_enabled
    assert not Settings(LANGSMITH_TRACING=False, LANGSMITH_API_KEY="k").tracing_enabled
    assert Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="k").tracing_enabled


def test_enabled_tracing_reaches_the_environment_langchain_reads():
    """pydantic-settings reads .env without exporting it, unlike load_dotenv.

    LangChain discovers tracing only through os.environ, so a var that is not
    exported here does nothing at all.
    """
    _clear_tracing_env()
    Settings(
        LANGSMITH_TRACING=True, LANGSMITH_API_KEY="k", LANGSMITH_PROJECT="proj"
    ).export_client_env()
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "k"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    _clear_tracing_env()


def test_traces_are_redacted_by_default():
    """This system traces employee HR questions and internal policy text."""
    s = Settings(_env_file=None)
    assert s.langsmith_hide_inputs and s.langsmith_hide_outputs
    assert s.tracing_redacted


def test_redaction_exports_the_literal_string_langsmith_compares_against():
    """langsmith does `get_env_var("HIDE_INPUTS") == "true"` -- exactly that.

    "True" or "1" would be read as false and silently upload every question.
    """
    _clear_tracing_env()
    Settings(
        _env_file=None, LANGSMITH_TRACING=True, LANGSMITH_API_KEY="k"
    ).export_client_env()
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    _clear_tracing_env()


def test_redaction_can_be_turned_off_deliberately():
    _clear_tracing_env()
    s = Settings(
        _env_file=None,
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="k",
        LANGSMITH_HIDE_INPUTS=False,
        LANGSMITH_HIDE_OUTPUTS=False,
    )
    s.export_client_env()
    assert not s.tracing_redacted
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "false"
    _clear_tracing_env()


def test_openai_key_is_not_exported_while_no_provider_uses_it(monkeypatch):
    """An unused key stays out of the process environment.

    langchain-openai is installed as a langchain-pinecone dependency, so a key
    in os.environ is enough for it to configure itself. It only belongs there
    when a provider is actually set to openai.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    Settings(_env_file=None, OPENAI_API_KEY="sk-unused").export_client_env()
    assert "OPENAI_API_KEY" not in os.environ


def test_openai_key_is_exported_when_a_provider_uses_it(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    Settings(
        _env_file=None, OPENAI_API_KEY="sk-used", LLM_PROVIDER="openai"
    ).export_client_env()
    assert os.environ["OPENAI_API_KEY"] == "sk-used"


# --- Provider selection ------------------------------------------------------


def test_groq_is_the_default_provider_for_both():
    s = Settings(_env_file=None)
    assert s.llm_provider == "groq"
    assert s.embedding_provider == "huggingface"
    assert not s.uses_openai


def test_an_unknown_provider_is_rejected_at_construction():
    with pytest.raises(Exception):
        Settings(_env_file=None, LLM_PROVIDER="anthropic")


def test_active_model_follows_the_provider():
    groq = Settings(_env_file=None)
    assert groq.active_llm_model == groq.groq_model
    assert groq.active_embedding_model == groq.embedding_model

    openai = Settings(_env_file=None, LLM_PROVIDER="openai", EMBEDDING_PROVIDER="openai")
    assert openai.active_llm_model == openai.openai_model
    assert openai.active_embedding_model == openai.openai_embedding_model


def test_required_secrets_follow_the_providers():
    """An all-OpenAI deployment must not be reported broken for lacking Groq."""
    groq = Settings(_env_file=None, PINECONE_API="p", TAVILY_API="t")
    assert groq.missing_secrets() == ["GROQ_API"]

    openai = Settings(
        _env_file=None,
        PINECONE_API="p",
        TAVILY_API="t",
        LLM_PROVIDER="openai",
        EMBEDDING_PROVIDER="openai",
    )
    assert openai.missing_secrets() == ["OPENAI_API_KEY"]

    mixed = Settings(
        _env_file=None, PINECONE_API="p", TAVILY_API="t", EMBEDDING_PROVIDER="openai"
    )
    assert mixed.missing_secrets() == ["GROQ_API", "OPENAI_API_KEY"]


# --- Embedding dimension -----------------------------------------------------


def test_embedding_dim_is_derived_from_the_active_model():
    """Switching provider without the dimension is the easy mistake."""
    assert Settings(_env_file=None).embedding_dim == 384
    assert Settings(_env_file=None, EMBEDDING_PROVIDER="openai").embedding_dim == 1536
    assert (
        Settings(
            _env_file=None,
            EMBEDDING_PROVIDER="openai",
            OPENAI_EMBEDDING_MODEL="text-embedding-3-large",
        ).embedding_dim
        == 3072
    )


def test_a_dimension_contradicting_the_model_is_rejected():
    """Otherwise it surfaces as an opaque Pinecone upsert failure."""
    with pytest.raises(ValueError, match="contradicts"):
        Settings(
            _env_file=None, EMBEDDING_PROVIDER="openai", EMBEDDING_DIM=384
        ).validate_embedding()


def test_an_unknown_model_must_state_its_dimension():
    with pytest.raises(ValueError, match="must be set explicitly"):
        Settings(_env_file=None, EMBEDDING_MODEL="some/unlisted-model").validate_embedding()

    ok = Settings(
        _env_file=None, EMBEDDING_MODEL="some/unlisted-model", EMBEDDING_DIM=512
    )
    ok.validate_embedding()
    assert ok.embedding_dim == 512


def test_validate_required_also_checks_the_embedding_dimension():
    s = Settings(
        _env_file=None,
        PINECONE_API="p",
        TAVILY_API="t",
        GROQ_API="g",
        EMBEDDING_PROVIDER="openai",
        OPENAI_API_KEY="o",
        EMBEDDING_DIM=384,
    )
    with pytest.raises(ValueError, match="contradicts"):
        s.validate_required()


def test_disabled_tracing_clears_a_switch_left_in_the_environment():
    """A stray key must not start shipping questions to a third party."""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    Settings(LANGSMITH_TRACING=False).export_client_env()
    assert "LANGSMITH_TRACING" not in os.environ
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_namespaces_are_explicit_strings():
    """The Pinecone default namespace ("") behaves inconsistently across calls."""
    s = Settings(_env_file=None)
    assert s.private_namespace and s.public_namespace
    assert s.private_namespace != s.public_namespace


def test_defaults_match_the_reference_architecture():
    """docs/architecture.png names these; a local .env may override them."""
    s = Settings(_env_file=None)
    assert s.pinecone_index == "peopleprime-hr-kb"
    assert s.private_namespace == "hr-docs"


def test_exported_settings_feed_back_into_a_later_settings_object(monkeypatch):
    """export_client_env writes os.environ, which outranks the .env file.

    Worth pinning: it means a Settings built after an export sees the exported
    value, not the file's. Consistent in a process, surprising in a test.
    """
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    Settings(
        _env_file=None, LANGSMITH_TRACING=True, LANGSMITH_API_KEY="k"
    ).export_client_env()
    assert Settings(_env_file=None).langsmith_tracing is True


def test_settings_are_immutable():
    with pytest.raises(Exception):
        Settings().pinecone_index = "something-else"  # type: ignore[misc]


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


# --- Admin key ---------------------------------------------------------------


def test_admin_is_disabled_when_no_key_is_configured():
    """No working placeholder default: unset must mean off, not open."""
    s = Settings(_env_file=None)
    assert not s.admin_enabled
    assert not s.check_admin_key("")
    assert not s.check_admin_key(None)
    assert not s.check_admin_key("change-me-in-production")


def test_admin_key_matches_only_the_configured_value():
    s = Settings(_env_file=None, ADMIN_API_KEY="s3cret")
    assert s.admin_enabled
    assert s.check_admin_key("s3cret")
    assert s.check_admin_key("  s3cret  ")  # header whitespace
    assert not s.check_admin_key("s3cre")
    assert not s.check_admin_key("wrong")


def test_production_refuses_to_start_without_an_admin_key():
    base = dict(
        _env_file=None, PINECONE_API="p", TAVILY_API="t", GROQ_API="g",
        EMBEDDING_DIM=384,
    )
    with pytest.raises(ValueError, match="ADMIN_API_KEY"):
        Settings(**base, APP_ENV="production").validate_required()
    # Development is allowed to run without one; the routes fail closed anyway.
    Settings(**base, APP_ENV="development").validate_required()
    Settings(**base, APP_ENV="production", ADMIN_API_KEY="k").validate_required()


def test_is_production_accepts_the_usual_spellings():
    assert Settings(_env_file=None, APP_ENV="production").is_production
    assert Settings(_env_file=None, APP_ENV="Prod").is_production
    assert not Settings(_env_file=None, APP_ENV="development").is_production


# --- Paths -------------------------------------------------------------------


def test_paths_are_configurable_for_a_container(tmp_path):
    """Docker mounts the KB, uploads and the SQLite volume at its own paths."""
    s = Settings(
        _env_file=None,
        KB_DIR=str(tmp_path / "kb"),
        UPLOAD_DIR=str(tmp_path / "up"),
        AUDIT_DB_PATH=str(tmp_path / "audit.db"),
    )
    assert s.kb_dir == tmp_path / "kb"
    assert s.upload_dir == tmp_path / "up"
    assert s.audit_db_path == tmp_path / "audit.db"


def test_paths_default_under_the_project_root():
    s = Settings(_env_file=None)
    assert s.kb_dir == s.project_root / "data" / "private_kb"
    assert s.audit_db_path == s.project_root / "data" / "audit.db"
