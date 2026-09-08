"""The entry point: flags are parsed before anything reads a credential."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from run import describe, parse_args


def settings_for(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        GROQ_API="g",
        TAVILY_API="t",
        PINECONE_API="p",
        PINECONE_INDEX="test-index",
    )
    base.update(overrides)
    return Settings(**base)


# --- Arguments ---------------------------------------------------------------


def test_defaults_leave_host_and_port_to_the_settings():
    args = parse_args([])
    assert args.host is None and args.port is None and args.reload is False


def test_flags_override_one_run():
    args = parse_args(["--host", "127.0.0.1", "--port", "9000", "--reload"])
    assert (args.host, args.port, args.reload) == ("127.0.0.1", 9000, True)


def test_help_exits_instead_of_starting_a_server():
    """`--reload` used to be read as `"--reload" in sys.argv`, so `--help`
    matched nothing, was ignored, and started a server on someone asking what
    the flags were. argparse is the house rule for every entry point here."""
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"])
    assert exit_info.value.code == 0


def test_an_unknown_flag_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--relaod"])  # a plausible typo
    assert exit_info.value.code != 0


# --- The startup banner ------------------------------------------------------


def test_the_banner_names_the_resolved_providers():
    lines = "\n".join(describe(settings_for(APP_NAME="HR Copilot")))
    assert "HR Copilot" in lines
    assert "groq / openai/gpt-oss-20b" in lines
    assert "test-index" in lines


def test_the_banner_never_prints_a_secret_value():
    """It goes to stdout and into whatever the deployment captures."""
    lines = "\n".join(describe(settings_for(GROQ_API="gsk_live_SECRET")))
    assert "gsk_live_SECRET" not in lines


def test_an_unconfigured_admin_surface_is_stated_not_implied():
    """Silent-off is exactly the state worth seeing at startup."""
    assert "OFF (no ADMIN_API_KEY)" in "\n".join(describe(settings_for()))
    assert "enabled" in "\n".join(describe(settings_for(ADMIN_API_KEY="k")))


def test_tracing_off_is_reported_as_off():
    assert "langsmith   off" in "\n".join(describe(settings_for()))


def test_redacted_tracing_says_what_it_does_and_does_not_send():
    lines = "\n".join(
        describe(
            settings_for(
                LANGSMITH_TRACING=True,
                LANGSMITH_API_KEY="ls-secret",
                LANGSMITH_PROJECT="hr-copilot",
            )
        )
    )
    assert "on, redacted" in lines
    assert "hr-copilot" in lines
    assert "ls-secret" not in lines


def test_unredacted_tracing_is_shouted_about():
    """Questions and retrieved policy text leaving the deployment is not a
    detail to discover later in a config file."""
    lines = "\n".join(
        describe(
            settings_for(
                LANGSMITH_TRACING=True,
                LANGSMITH_API_KEY="k",
                LANGSMITH_HIDE_INPUTS=False,
                LANGSMITH_HIDE_OUTPUTS=False,
            )
        )
    )
    assert "*UNREDACTED*" in lines
    assert "leaving this deployment" in lines
