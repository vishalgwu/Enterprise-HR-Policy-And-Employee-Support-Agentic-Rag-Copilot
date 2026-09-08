"""Environment and configuration for the HR Copilot.

Imported by every other module and deliberately free of heavy dependencies, so
importing configuration never drags in torch, bs4 or the Pinecone client.

Importing this module exports the API keys under the names the LangChain, Groq,
Tavily and Pinecone clients expect. Those clients read the environment when they
are constructed, so this module must be imported before any of them are built --
which it is, because they all import their settings from here.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

LOGGER_NAME = "hr_copilot"
# A library must not configure logging for its host. NullHandler keeps these
# records silent until an entry point calls configure_logging().
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Logger for one module, under the package's shared root."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Send this package's logs to stderr. Entry points only, never on import.

    Keeps stdout clean for the answer itself, so `python Rag/demo.py > out.txt`
    captures results without the per-node trace.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def require_env(name: str) -> str:
    """Read a required setting, rejecting missing and whitespace-only values."""
    value = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(f"{name} is missing. Set it in the .env file.")
    return value.strip()


# The .env uses short names; the client libraries expect the _API_KEY forms.
GROQ_API_KEY = require_env("GROQ_API")
TAVILY_API_KEY = require_env("TAVILY_API")
PINECONE_API_KEY = require_env("PINECONE_API")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
)
USER_AGENT = os.environ["USER_AGENT"]

# --- Embeddings --------------------------------------------------------------
# all-MiniLM-L6-v2 produces 384-dimensional vectors. EMBEDDING_DIM must match the
# model: the Pinecone index is created with it and Pinecone cannot change an
# existing index's dimension, so a new model needs a new PINECONE_INDEX name.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# --- Pinecone ----------------------------------------------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "hr-policy-copilot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = "cosine"

# Namespaces keep the scraped public article and the internal policy corpus
# apart, so a retrieval never mixes them. Writer and reader share these
# constants: a namespace typo is silent -- the upsert succeeds and every later
# retrieval returns [].
PUBLIC_NAMESPACE = ""
PRIVATE_NAMESPACE = "private-kb"

INDEX_READY_TIMEOUT = 300
INDEX_READY_POLL_INTERVAL = 2

# Pinecone serverless is eventually consistent: an upsert is not immediately
# queryable, so a retrieval fired straight after ingest returns nothing.
FRESHNESS_TIMEOUT = 120
FRESHNESS_POLL_INTERVAL = 2

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

PROSE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
# Markdown splits on headings first so a chunk starts at its section title,
# which measurably improves how it scores against a question about that section.
MARKDOWN_SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " ", ""]

# --- Retrieval ---------------------------------------------------------------
DEFAULT_K = 4

# Minimum raw cosine similarity for a chunk to be worth grading, measured on a
# 12-question eval of the private KB: out-of-domain questions peaked at 0.099
# while the lowest-scoring correct-source chunk was 0.171, so 0.15 sits in the
# gap. This is an out-of-domain gate, not a relevance grader -- a wrong-document
# chunk scored as high as 0.521, so judging relevance is still the LLM's job.
RELEVANCE_THRESHOLD = 0.15

# --- LLM and web search ------------------------------------------------------
# Groq retires models fairly often; `groq.Groq().models.list()` prints what this
# key can reach. A wrong id fails at call time, not at construction.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Grading is a classification decision, so it must not vary between runs.
GROQ_TEMPERATURE = 0
WEB_SEARCH_MAX_RESULTS = 5

# --- Data sources ------------------------------------------------------------
KB_DIR = PROJECT_ROOT / "data" / "private_kb"

HR_POLICY_URL = (
    "https://www.hracuity.com/blog/hr-policies-and-procedures-to-consider-for-your-company/"
)
# The article body lives in .entry-content; parsing only that keeps site
# navigation, related-post cards and the footer out of the vector index.
HR_POLICY_CONTENT_CLASS = "entry-content"


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
