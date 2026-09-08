"""Settings for the whole project, plus the logging names re-exported from
`logging_config` so callers have one import to remember.

The lowest layer: imported by everything, and deliberately free of heavy
dependencies so importing configuration never drags in torch, bs4 or Pinecone.

Two things about the import-time behaviour are load-bearing:

1. Importing this module calls `export_client_env()`, which copies the short
   names used in `.env` (`GROQ_API`) into the names the LangChain, Groq, Tavily
   and Pinecone clients read when they are constructed (`GROQ_API_KEY`). Those
   clients read the environment in their constructors, so the export has to
   happen before any of them are built -- which it does, because every client
   module imports its settings from here.

2. A missing key is *not* fatal at import. It raises when a client that needs it
   is actually constructed, via `Settings.require`. Failing at import would make
   the package unimportable without secrets, which breaks the test suite, `--help`
   on any script, and a container that gets its configuration from the platform
   rather than from a `.env` file. `validate_required()` restores fail-fast for
   entry points that genuinely need the keys up front.
"""

from __future__ import annotations

import hmac
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Re-exported: `from app.core.config import configure_logging, get_logger` is
# the import every entry point and module already uses, and stays supported.
from app.core.logging_config import (  # noqa: F401
    DETAILED_FORMAT,
    LOGGER_NAME,
    MESSAGE_FORMAT,
    configure_logging,
    configure_stdout,
    get_logger,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Output width of every embedding model this project knows about. Used to fill
# `embedding_dim` when it is not set explicitly, and to reject a value that
# contradicts the model. Getting this wrong is expensive: Pinecone cannot resize
# an index, so the mismatch surfaces as an opaque upsert failure against an index
# that then has to be recreated under a new name.
# Field name -> the env var a user has to set. One map so an error message and
# a health check cannot disagree about what to tell someone.
SECRET_ALIASES: dict[str, str] = {
    "groq_api_key": "GROQ_API",
    "tavily_api_key": "TAVILY_API",
    "pinecone_api_key": "PINECONE_API",
    "openai_api_key": "OPENAI_API_KEY",
    "admin_api_key": "ADMIN_API_KEY",
    "langsmith_api_key": "LANGSMITH_API_KEY",
}

KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _normalise_model(name: str) -> str:
    """Lower-case a model id and drop its org prefix, for dimension lookup.

    `all-MiniLM-L6-v2` and `sentence-transformers/all-MiniLM-L6-v2` name the
    same model -- sentence-transformers resolves the bare form -- so writing the
    short one in `.env` must not read as an unknown model.
    """
    return name.strip().lower().rsplit("/", 1)[-1]


_DIMS_BY_NORMALISED_MODEL: dict[str, int] = {
    _normalise_model(model): dim for model, dim in KNOWN_EMBEDDING_DIMS.items()
}

# Two entries that normalise to one key would silently shadow each other, and
# the survivor would hand out the wrong dimension for the loser. Fail at import
# instead: this is a source-code mistake, not a configuration one.
if len(_DIMS_BY_NORMALISED_MODEL) != len(KNOWN_EMBEDDING_DIMS):
    raise RuntimeError(
        "Two entries in KNOWN_EMBEDDING_DIMS normalise to the same key; "
        "distinct models must stay distinguishable after _normalise_model."
    )


def known_embedding_dim(model: str) -> int | None:
    """Output width of a known embedding model, or None if it is not listed.

    Matching is **exact** after normalisation, deliberately. A tempting
    alternative is to fall back to substring matching -- treat anything
    containing "all-minilm" as 384-d -- but a guessed dimension builds a
    Pinecone index that cannot be resized, so a wrong guess costs a full
    teardown and re-ingest while an unknown model costs one `EMBEDDING_DIM=`
    line. Returning None so the caller can say "state it explicitly" is the
    cheaper failure.
    """
    return _DIMS_BY_NORMALISED_MODEL.get(_normalise_model(model))


class Settings(BaseSettings):
    """Every tunable value in one validated object.

    A class rather than module globals so a test can build a variant
    (`Settings(pinecone_index="throwaway")`) instead of monkeypatching module
    state, and so FastAPI can inject it as a dependency.

    Field names are snake_case; `validation_alias` maps each to the short env
    var the project's `.env` actually uses.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Application identity ------------------------------------------------
    app_name: str = Field(
        default="Enterprise HR Policy Agentic RAG Copilot", validation_alias="APP_NAME"
    )
    # "development" | "production". Load-bearing rather than decorative: it
    # decides whether the interactive API docs are exposed, and it is what
    # `validate_required` checks the admin key against.
    app_env: str = Field(default="development", validation_alias="APP_ENV")

    # --- Secrets -------------------------------------------------------------
    # Empty rather than required: see the module docstring. Use `require()`.
    #
    # SecretStr, not str: it renders as `SecretStr('**********')` in reprs,
    # tracebacks and log lines, so `log.info("%s", settings)` or an unhandled
    # exception carrying the model cannot publish a key. Read the real value
    # through `require()`.
    groq_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GROQ_API")
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="TAVILY_API"
    )
    pinecone_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="PINECONE_API"
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="OPENAI_API_KEY"
    )

    # Gates the admin-only endpoints (/upload, /ingest, /admin, /logs).
    #
    # Empty by default, deliberately. A working placeholder like
    # "change-me-in-production" is worse than none: forget to set it and the
    # admin API is open, guarded by a string that is in the source. Empty means
    # `admin_enabled` is False and those routes must refuse every request --
    # fail closed, so the failure is "admin is off" rather than "anyone is
    # admin". `validate_required` additionally refuses to start without one when
    # APP_ENV=production.
    admin_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="ADMIN_API_KEY"
    )

    # --- Providers -----------------------------------------------------------
    # Groq and OpenAI are both supported so a deployment can pick. They are
    # chosen independently: a host that cannot run torch still wants a cheap
    # LLM, and Groq has no embedding endpoint.
    llm_provider: Literal["groq", "openai"] = Field(
        default="groq", validation_alias="LLM_PROVIDER"
    )
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface", validation_alias="EMBEDDING_PROVIDER"
    )

    # --- Embeddings ----------------------------------------------------------
    # embedding_dim must match whichever model is active: the Pinecone index is
    # created with it and Pinecone cannot resize an index, so changing model
    # means a new index name as well as a new dimension. `validate_embedding`
    # catches the mismatch at construction instead of as an opaque upsert error.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=0, validation_alias="EMBEDDING_DIM")

    # --- Pinecone ------------------------------------------------------------
    # The default matches the name in docs/architecture.png. A different name in
    # .env wins, which is why nothing in the code hardcodes either one.
    pinecone_index: str = Field(
        default="peopleprime-hr-kb", validation_alias="PINECONE_INDEX"
    )
    pinecone_cloud: str = Field(default="aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", validation_alias="PINECONE_REGION")
    pinecone_metric: str = "cosine"

    # Namespaces keep the internal policy corpus and the scraped public article
    # apart, so one retrieval never mixes them. Writer and reader share these
    # values: a namespace typo is silent -- the upsert succeeds and every later
    # retrieval returns []. Both are explicit strings; the Pinecone default
    # namespace ("") behaves inconsistently across list and delete calls.
    private_namespace: str = Field(default="hr-docs", validation_alias="PRIVATE_NAMESPACE")
    public_namespace: str = Field(default="public-web", validation_alias="PUBLIC_NAMESPACE")

    index_ready_timeout: int = 300
    index_ready_poll_interval: int = 2

    # Pinecone serverless is eventually consistent: an upsert is not immediately
    # queryable, so a retrieval fired straight after ingest returns nothing.
    freshness_timeout: int = 120
    freshness_poll_interval: int = 2

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval -----------------------------------------------------------
    default_k: int = 4

    # Minimum raw cosine similarity for a chunk to be worth grading, measured on
    # a 12-question eval of the private KB: out-of-domain questions peaked at
    # 0.099 while the lowest-scoring correct-source chunk was 0.171, so 0.15 sits
    # in the gap. This is an out-of-domain gate, not a relevance grader -- a
    # wrong-document chunk scored as high as 0.521, so judging whether evidence
    # actually answers the question remains the LLM's job.
    relevance_threshold: float = 0.15

    # --- LLM and web search --------------------------------------------------
    # Groq retires model ids fairly often; `groq.Groq().models.list()` prints
    # what a key can reach. A wrong id fails at call time, not at construction.
    groq_model: str = Field(default="openai/gpt-oss-20b", validation_alias="GROQ_MODEL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    # Routing and grading are classification decisions, so they must not vary
    # between runs.
    llm_temperature: float = Field(default=0.0, validation_alias="LLM_TEMPERATURE")
    web_search_max_results: int = 5

    # One rewrite-and-retry before admitting defeat. Higher values mostly buy
    # latency: if a rewrite does not fix retrieval, a second rarely does.
    max_retries: int = 1

    # --- HTTP ----------------------------------------------------------------
    # WebBaseLoader requires a User-Agent, and the Groq API sits behind
    # Cloudflare, which blocks the default Python-urllib agent with a 403 that
    # looks exactly like a revoked key.
    user_agent: str = Field(default=DEFAULT_USER_AGENT, validation_alias="USER_AGENT")

    # --- API -----------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")

    # A cap on one upload, enforced while the body streams to disk rather than
    # after it is read into memory. The uploaded file is the only input this API
    # takes from a caller by the megabyte, so it is the only one that can
    # exhaust the process or fill the mounted volume.
    max_upload_mb: int = Field(default=10, ge=1, validation_alias="MAX_UPLOAD_MB")

    # --- Paths ---------------------------------------------------------------
    # Settings rather than fixed properties so a container can mount them: the
    # target architecture runs the app and its SQLite volume as separate Docker
    # services, which only works if the paths are configurable.
    kb_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "private_kb", validation_alias="KB_DIR"
    )
    upload_dir: Path = Field(
        default=PROJECT_ROOT / "uploads", validation_alias="UPLOAD_DIR"
    )
    audit_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "audit.db", validation_alias="AUDIT_DB_PATH"
    )

    # The UI. Settings rather than fixed paths for the same reason as the rest:
    # the target architecture serves these from an Nginx container that does not
    # have the Python package, and a test needs to point them at tmp_path.
    templates_dir: Path = Field(
        default=PROJECT_ROOT / "templates", validation_alias="TEMPLATES_DIR"
    )
    static_dir: Path = Field(
        default=PROJECT_ROOT / "static", validation_alias="STATIC_DIR"
    )

    # --- Observability -------------------------------------------------------
    # LangSmith tracing is declared here rather than left to the environment,
    # because it has to be a decision rather than an accident. Tracing uploads
    # every question and every retrieved chunk to LangChain's servers -- for
    # this product that means employee HR questions and internal policy text
    # leaving the deployment. Off unless explicitly switched on *and* keyed.
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGSMITH_API_KEY"
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", validation_alias="LANGSMITH_ENDPOINT"
    )
    langsmith_project: str = Field(
        default="hr-copilot", validation_alias="LANGSMITH_PROJECT"
    )

    # Payload redaction, defaulting to ON because this system traces employee HR
    # questions and internal policy text. With these set, every run reaches
    # LangSmith with empty `inputs` and `outputs` -- so the UI shows "No inputs /
    # No outputs" on every node, which is the control working rather than a
    # misconfiguration. Verified over 12 live runs: node names, the run tree,
    # timings, token counts and errors all still arrive.
    #
    # A node's *decision* is its output, so the route and the grades are
    # redacted too; the path they produced is still readable from which nodes
    # ran. The values live in the audit log behind /api/audit, beside the
    # question they were made about. Set these false in .env only when debugging
    # a specific answer, deliberately and temporarily.
    langsmith_hide_inputs: bool = Field(
        default=True, validation_alias="LANGSMITH_HIDE_INPUTS"
    )
    langsmith_hide_outputs: bool = Field(
        default=True, validation_alias="LANGSMITH_HIDE_OUTPUTS"
    )

    @property
    def tracing_enabled(self) -> bool:
        """Tracing needs both the switch and a key; either alone does nothing."""
        return bool(self.langsmith_tracing and self._secret("langsmith_api_key"))

    @property
    def tracing_redacted(self) -> bool:
        """True when no question or policy text is uploaded with a trace."""
        return bool(self.langsmith_hide_inputs and self.langsmith_hide_outputs)

    # --- Provider resolution -------------------------------------------------
    @model_validator(mode="after")
    def _derive_embedding_dim(self) -> Settings:
        """Fill embedding_dim from the active model when it was not given.

        Derives only; it must never raise. A `model_validator` that raises
        produces a pydantic ValidationError whose `input_value` is the whole
        input mapping -- which for this class is every API key. That error text
        reaches logs and tracebacks, so the contradiction check lives in
        `validate_embedding()` instead, which raises a plain ValueError naming
        only the dimension and the model.
        """
        if not self.embedding_dim:
            expected = known_embedding_dim(self.active_embedding_model)
            if expected:
                object.__setattr__(self, "embedding_dim", expected)
        return self

    def validate_embedding(self) -> None:
        """Reject a dimension that contradicts the active embedding model.

        Switching provider without switching the dimension is the easy mistake,
        and it otherwise surfaces as an opaque Pinecone upsert failure against an
        index that then has to be recreated under a new name.
        """
        model = self.active_embedding_model
        expected = known_embedding_dim(model)
        if not self.embedding_dim:
            raise ValueError(
                f"EMBEDDING_DIM must be set explicitly for unknown model {model!r}."
            )
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} contradicts {model!r}, which "
                f"emits {expected}-d vectors. Pinecone cannot resize an index -- "
                f"fix the dimension and point PINECONE_INDEX at a name matching "
                f"the new model."
            )

    @property
    def active_embedding_model(self) -> str:
        """The embedding model the configured provider will actually use."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.embedding_model

    @property
    def active_llm_model(self) -> str:
        """The chat model the configured provider will actually use."""
        return self.openai_model if self.llm_provider == "openai" else self.groq_model

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in ("production", "prod")

    @property
    def max_upload_bytes(self) -> int:
        """The upload cap in bytes, which is what a streaming read counts."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def admin_enabled(self) -> bool:
        """False when no admin key is configured; admin routes must refuse."""
        return bool(self._secret("admin_api_key"))

    def check_admin_key(self, presented: str | None) -> bool:
        """Constant-time comparison of a presented admin key.

        Returns False when admin is not configured at all, so an unset key can
        never be matched by an empty header. `compare_digest` because a plain
        `==` on a secret leaks its length and prefix through timing.
        """
        expected = self._secret("admin_api_key")
        if not expected or not presented:
            return False
        return hmac.compare_digest(expected, presented.strip())

    # --- Secret access -------------------------------------------------------
    def require(self, field: str) -> str:
        """Read a secret, failing with an actionable message when it is unset.

        Called where a client is constructed, so importing the package without
        credentials stays possible while using it without them does not.
        """
        value = self._secret(field)
        if not value:
            raise ValueError(
                f"{SECRET_ALIASES.get(field, field.upper())} is not set. Add it "
                f"to {PROJECT_ROOT / '.env'} (see .env.example) or export it in "
                f"the environment."
            )
        return value

    @property
    def uses_openai(self) -> bool:
        """True when either configured provider needs an OpenAI key."""
        return self.llm_provider == "openai" or self.embedding_provider == "openai"

    def required_secret_fields(self) -> list[str]:
        """Which credentials this configuration actually needs.

        Provider-aware on purpose: a deployment running entirely on OpenAI must
        not be reported as broken for having no Groq key, and vice versa.
        """
        fields = ["pinecone_api_key", "tavily_api_key"]
        if self.llm_provider == "groq":
            fields.append("groq_api_key")
        if self.uses_openai:
            fields.append("openai_api_key")
        return fields

    def _secret(self, field: str) -> str:
        """The plain value behind a SecretStr field, stripped."""
        value = getattr(self, field, None)
        if value is None:
            return ""
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        return raw.strip()

    def missing_secrets(self) -> list[str]:
        return [
            SECRET_ALIASES[field]
            for field in self.required_secret_fields()
            if not self._secret(field)
        ]

    def validate_required(self) -> None:
        """Fail fast, for entry points that need a usable configuration."""
        missing = self.missing_secrets()
        if missing:
            raise ValueError(
                f"Missing required setting(s): {', '.join(missing)}. "
                f"Add them to {PROJECT_ROOT / '.env'} (see .env.example)."
            )
        self.validate_embedding()
        if self.is_production and not self.admin_enabled:
            # Only in production: development wants to run without one, and
            # admin routes already fail closed when it is unset.
            raise ValueError(
                "ADMIN_API_KEY must be set when APP_ENV=production -- the admin "
                "endpoints are refused entirely without it."
            )

    def export_client_env(self) -> None:
        """Publish settings under the names third-party clients look for.

        ChatGroq, TavilySearch and Pinecone read these from the environment in
        their constructors rather than taking them as arguments, so they have to
        be in place before any client is built.

        This is also the *only* thing that puts `.env` values into `os.environ`.
        pydantic-settings reads the file without exporting it, unlike
        `load_dotenv()`. Anything a third-party library discovers through the
        environment has to be listed here or it silently does nothing.
        """
        for field, env_name in (
            ("groq_api_key", "GROQ_API_KEY"),
            ("tavily_api_key", "TAVILY_API_KEY"),
            ("pinecone_api_key", "PINECONE_API_KEY"),
        ):
            value = self._secret(field)
            if value:
                os.environ[env_name] = value

        # Exported only when a provider is actually configured to use it. A key
        # sitting unused in .env stays out of the process environment, so it
        # cannot be picked up by langchain-openai -- which is installed here
        # purely as a langchain-pinecone dependency.
        if self.uses_openai and self._secret("openai_api_key"):
            os.environ["OPENAI_API_KEY"] = self._secret("openai_api_key")

        os.environ.setdefault("USER_AGENT", self.user_agent)

        # LangChain discovers tracing purely through the environment. Exported
        # only when switched on and keyed, so a stray key cannot start shipping
        # employee questions to a third party on its own.
        if self.tracing_enabled:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"  # older SDKs read this
            os.environ["LANGSMITH_API_KEY"] = self._secret("langsmith_api_key")
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            # langsmith compares these against the literal string "true"
            # (Client.__init__ -> ls_utils.get_env_var("HIDE_INPUTS")), so they
            # must be exactly that, not "True" or "1".
            os.environ["LANGSMITH_HIDE_INPUTS"] = (
                "true" if self.langsmith_hide_inputs else "false"
            )
            os.environ["LANGSMITH_HIDE_OUTPUTS"] = (
                "true" if self.langsmith_hide_outputs else "false"
            )
        else:
            for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
                os.environ.pop(name, None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings object. Cached so `.env` is read once."""
    settings = Settings()
    settings.export_client_env()
    return settings


# DO NOT DELETE, and do not import it either.
#
# Nothing in this project reads this name -- callers use get_settings(), or
# build their own Settings, so they are not coupled to import order. What this
# line is for is its *side effect*: it is the single call that runs
# export_client_env() when `app.core.config` is imported, which is what puts
# GROQ_API_KEY / TAVILY_API_KEY / PINECONE_API_KEY / USER_AGENT into the
# environment before any client constructor reads them. Every client module
# imports its settings from here, so this ordering is guaranteed -- and it is
# guaranteed by this line alone. Removing it as an unused global would break
# every provider client at once, silently, at construction time.
settings = get_settings()


# --- Text splitting ----------------------------------------------------------
# Not settings: these are structural choices about how a format is best split,
# not values anyone would tune per environment.
PROSE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
# Markdown splits on headings first so a chunk starts at its section title,
# which measurably improves how it scores against a question about that section.
MARKDOWN_SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " ", ""]


# --- Public web corpus -------------------------------------------------------
# A development corpus, not part of the product described in docs/. Kept because
# it exercises the web-loading path end to end.
HR_POLICY_URL = (
    "https://www.hracuity.com/blog/hr-policies-and-procedures-to-consider-for-your-company/"
)
# The article body lives in .entry-content; parsing only that keeps site
# navigation, related-post cards and the footer out of the vector index.
HR_POLICY_CONTENT_CLASS = "entry-content"
