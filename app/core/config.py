"""Settings, logging and console setup.

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

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGER_NAME = "hr_copilot"
# A library must not configure logging for its host. NullHandler keeps these
# records silent until an entry point calls configure_logging().
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


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

    # --- Secrets -------------------------------------------------------------
    # Empty rather than required: see the module docstring. Use `require()`.
    groq_api_key: str = Field(default="", validation_alias="GROQ_API")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API")
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API")

    # --- Embeddings ----------------------------------------------------------
    # all-MiniLM-L6-v2 emits 384-d vectors. embedding_dim must match the model:
    # the Pinecone index is created with it and Pinecone cannot resize an index,
    # so a new model needs a new index name as well as a new dimension.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=384, validation_alias="EMBEDDING_DIM")

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
    # Routing and grading are classification decisions, so they must not vary
    # between runs.
    groq_temperature: float = 0.0
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

    # --- Paths ---------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def kb_dir(self) -> Path:
        """Private HR documents. Subdirectories are read as department names."""
        return PROJECT_ROOT / "data" / "private_kb"

    @property
    def upload_dir(self) -> Path:
        """Where authorised HR staff drop documents for ingestion."""
        return PROJECT_ROOT / "uploads"

    # --- Secret access -------------------------------------------------------
    def require(self, field: str) -> str:
        """Read a secret, failing with an actionable message when it is unset.

        Called where a client is constructed, so importing the package without
        credentials stays possible while using it without them does not.
        """
        value = (getattr(self, field, "") or "").strip()
        if not value:
            alias = {
                "groq_api_key": "GROQ_API",
                "tavily_api_key": "TAVILY_API",
                "pinecone_api_key": "PINECONE_API",
            }.get(field, field.upper())
            raise ValueError(
                f"{alias} is not set. Add it to {PROJECT_ROOT / '.env'} "
                f"(see .env.example) or export it in the environment."
            )
        return value

    def missing_secrets(self) -> List[str]:
        return [
            alias
            for field, alias in (
                ("groq_api_key", "GROQ_API"),
                ("tavily_api_key", "TAVILY_API"),
                ("pinecone_api_key", "PINECONE_API"),
            )
            if not (getattr(self, field, "") or "").strip()
        ]

    def validate_required(self) -> None:
        """Fail fast, for entry points that need every credential up front."""
        missing = self.missing_secrets()
        if missing:
            raise ValueError(
                f"Missing required setting(s): {', '.join(missing)}. "
                f"Add them to {PROJECT_ROOT / '.env'} (see .env.example)."
            )

    def export_client_env(self) -> None:
        """Publish secrets under the names third-party clients look for.

        ChatGroq, TavilySearch and Pinecone read these from the environment in
        their constructors rather than taking them as arguments, so they have to
        be in place before any client is built.
        """
        for field, env_name in (
            ("groq_api_key", "GROQ_API_KEY"),
            ("tavily_api_key", "TAVILY_API_KEY"),
            ("pinecone_api_key", "PINECONE_API_KEY"),
        ):
            value = (getattr(self, field, "") or "").strip()
            if value:
                os.environ[env_name] = value
        os.environ.setdefault("USER_AGENT", self.user_agent)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings object. Cached so `.env` is read once."""
    settings = Settings()
    settings.export_client_env()
    return settings


# Module-level convenience for the common case. Tests and FastAPI dependencies
# should call get_settings() (or build their own Settings) rather than reaching
# for this, so they are not coupled to import order.
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


# --- Logging -----------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Logger for one module, under the package's shared root."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Send this package's logs to stderr. Entry points only, never on import.

    stderr keeps stdout clean for the answer itself, so `python scripts/demo.py
    > out.txt` captures results without the per-node trace.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def configure_stdout() -> None:
    """Print UTF-8 regardless of the console code page.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on the
    non-breaking hyphens, em dashes and smart quotes that LLM replies routinely
    contain. The API call succeeds and the crash lands on print().
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
