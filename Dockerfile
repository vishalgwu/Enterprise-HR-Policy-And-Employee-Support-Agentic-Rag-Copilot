# Enterprise HR Policy Agentic RAG Copilot -- runtime image.
#
#   docker build -t hr-copilot .
#   docker run --rm -p 8080:8080 -v hr-copilot-data:/app/var \
#     -e GROQ_API=... -e TAVILY_API=... -e PINECONE_API=... hr-copilot
#
# **`--env-file .env` will not work with this repo's .env, and the way it fails
# is worth knowing.** Docker's env-file parser is not dotenv: it does not strip
# surrounding quotes and it does not strip the CR from CRLF line endings, both
# of which this .env has. `PINECONE_API="pcsk_..."` therefore arrives at the
# client as a key with literal quote characters, and Pinecone answers
# `401 UNAUTHENTICATED / Invalid API key` -- which looks exactly like a revoked
# key, and sends you to the dashboard to rotate a credential that was fine.
# Observed here, on the first containerised question.
#
# Pass secrets with `-e`, or through the platform's own secret store (which is
# what a DigitalOcean or Kubernetes deployment does anyway), or render a
# Docker-shaped file first:
#
#   python -c "from dotenv import dotenv_values; \
#     print('\n'.join(f'{k}={v}' for k, v in dotenv_values('.env').items() if v))" \
#     > .env.docker
#
# Four things here are decided by this project rather than by habit:
#
# 1. **Both requirement files are copied.** `requirements.txt` in this repo is a
#    one-line forward to `req.txt`, which holds the actual pins. Copying only
#    the conventional name gives `pip` a file whose sole instruction is to read
#    one that is not in the image, and the build fails on a missing `req.txt`.
#
# 2. **torch is installed CPU-only, before anything else.** `sentence-transformers`
#    pulls torch, and the default PyPI wheel drags in the CUDA runtime for a GPU
#    no container here has. Both sides measured on this base image rather than
#    estimated:
#
#      default PyPI    torch 1.2 GB + nvidia/ 3.2 GB -> 8.93 GB image
#      CPU index       torch 769 MB, no nvidia/      -> 2.49 GB image
#
#    Installing torch from PyTorch's CPU index *first* means pip finds the
#    requirement already satisfied when `sentence-transformers` asks for it, so
#    the CUDA wheel is never fetched. `python -c "import torch; torch.__version__"`
#    must end in `+cpu`; if it ever reads `+cu###`, this step stopped working
#    and the image is about to quadruple.
#
# 3. **The embedding model is baked in at build time.** `EMBEDDING_PROVIDER`
#    defaults to `huggingface`, which loads `all-MiniLM-L6-v2` from disk on
#    first use -- and in a fresh container "from disk" means a ~90 MB download
#    from huggingface.co while an employee waits. Worse, a deployment with
#    locked-down egress would fail on the first question rather than at build.
#    Pre-fetching makes the image self-contained and the first answer as fast as
#    the second.
#
# 4. **`app.main:app`, not the factory.** `run.py` uses `--factory`, but the
#    module-level `app` exists exactly so a platform that knows nothing about
#    factories can still be handed an ASGI target. This is that platform.

FROM python:3.13-slim

# 3.13 matches the interpreter req.txt was pinned against. pyproject declares
# >=3.11, so 3.11 or 3.12 also work if a base image is mandated -- change the
# tag, and re-run the suite on it before trusting the pins.

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# Everything the app writes lives under /app/var, and nothing else does. The
# repo defaults put the audit database inside data/ beside the knowledge base,
# which in a container would mean granting the runtime user write access to the
# corpus it answers employees from. `kb_dir`, `upload_dir` and `audit_db_path`
# are settings precisely so a deployment can make that split.
#
# One mount point as a result: `-v hr-copilot-data:/app/var` keeps the audit log
# and any uploaded documents across a redeploy.
ENV AUDIT_DB_PATH=/app/var/audit.db \
    UPLOAD_DIR=/app/var/uploads \
    KB_DIR=/app/data/private_kb

# HF_HOME is where the pre-fetched model lands. Set before the download so the
# cache is written somewhere predictable and world-readable, rather than into a
# root-owned $HOME the runtime user cannot read.
ENV HF_HOME=/opt/hf-cache

# curl is for HEALTHCHECK. No build toolchain: every pinned dependency ships a
# manylinux wheel, so nothing is compiled here.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies ------------------------------------------------------------
# Copied and installed before the source, so editing a template does not
# reinstall torch.
COPY req.txt requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# --- Embedding model ---------------------------------------------------------
# Fails the build if the model cannot be fetched, which is the right time to
# find out -- the alternative is discovering it on the first employee question.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chmod -R a+rX /opt/hf-cache

# Set *after* the download, which needs the network the line then switches off.
#
# Not belt-and-braces: without it the pre-fetch above buys much less than it
# looks. Measured on this image -- model already cached, container run with
# `--network none` -- the first `embed_query` still spent ~40s failing HEAD
# requests to huggingface.co through five retries before falling back to disk.
# It answers correctly either way, so the symptom is not an error in any log; it
# is the first employee of every deployment waiting forty seconds.
#
# Safe because the model is baked in: changing EMBEDDING_MODEL already means
# rebuilding this image *and* pointing PINECONE_INDEX at a new name, since
# Pinecone cannot resize an index. Pass `-e HF_HUB_OFFLINE=0` to let a
# container fetch a different model at run time.
ENV HF_HUB_OFFLINE=1

# --- Application -------------------------------------------------------------
COPY . .

# Non-root, and /app/var is the only thing it owns. The source, the templates
# and the knowledge base stay root-owned and read-only to the app, so a bug in
# the upload handler cannot rewrite the code that serves the next request or the
# policy that answers it.
RUN useradd --create-home --uid 10001 copilot \
    && mkdir -p /app/var/uploads \
    && chown -R copilot:copilot /app/var

VOLUME ["/app/var"]

USER copilot

EXPOSE 8080

# /health is open at every environment and reports `missing_secrets` rather than
# failing, so this checks that the process serves -- not that it is configured.
# `start-period` covers the graph and embedding warm-up on the first request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
