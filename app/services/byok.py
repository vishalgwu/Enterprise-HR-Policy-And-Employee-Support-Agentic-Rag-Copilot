"""Bring your own key: a caller spending their own provider account.

The problem this solves is specific to a deployment anyone can reach. `/api/chat`
is unauthenticated by design, and every question it answers costs the
deployment's own key — so a public URL is either rate-limited into
uselessness, or it is a bill. Letting a visitor paste their own OpenAI or Groq
key moves the cost to the person who wanted the answer, and it is also the only
way someone can try this on a better model than a free tier allows.

Three rules, and they are the whole design:

**1. The key is used and forgotten.** It arrives in a request header, it is used
to build one chat client, and it is never written to the audit log, a span
attribute, a log line, the vector index or the scratchpad. It is held in memory
only for as long as the pooled client that holds it, and the pool is bounded.
`ProviderCredentials.describe()` is what every other layer gets: the provider,
the model, and a four-character tail. That is enough to show a user which key is
in use and not enough to use it.

**2. The key only ever reaches the chat model.** Not Pinecone, not Tavily, not
embeddings. The LLM is the expensive, per-question, interchangeable component;
the vector store is the deployment's own data and a caller's key has no business
there. A caller-supplied key therefore changes `llm_provider`, the model id and
the one secret, and nothing else — which is also why the pool can share one
retriever across every caller.

**3. It is off unless switched on.** `ALLOW_BYOK=0` on an internal deployment
means every answer is on the company's account, which is what an HR department
auditing its own copilot wants.

**The pool holds clients, not keys, and that is a cost decision.** Building a
`Copilot` compiles a graph; doing that per request would recompile it per
question. Keyed on a hash of the credentials, bounded by `byok_pool_size`, and
LRU-evicted, so a hundred visitors do not pin a hundred graphs in memory.

**On the transport.** A key in a header over HTTPS is encrypted in transit and
is not in the URL, the history, the proxy log or the `Referer` — which is the
same reasoning the admin key already follows. It is still a secret in a header
that a reverse proxy *could* log, so the README says plainly that a visitor is
trusting the deployment with it, and the console says so beside the input. Never
accept one in a query string.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import SecretStr

from app.core.config import Settings, get_logger

log = get_logger("byok")

__all__ = [
    "ClientPool",
    "ProviderCredentials",
    "credentials_from_headers",
    "settings_with_credentials",
]

Provider = Literal["groq", "openai"]

# Header names. `X-Provider-Key` rather than `Authorization`, deliberately:
# `Authorization` is what a reverse proxy, an API gateway or a future auth layer
# claims, and a caller's provider key sitting in it would be consumed or logged
# by whichever of those arrives first.
HEADER_PROVIDER = "X-Provider"
HEADER_KEY = "X-Provider-Key"
HEADER_MODEL = "X-Provider-Model"

# Key prefixes per provider, used to reject a key posted to the wrong provider
# before it costs a round trip and an error the caller cannot interpret. These
# are the shapes observed in this project's own `.env`; a provider that
# introduces a new prefix will be rejected here, which is why the check is a
# *warning path* with an explicit escape rather than a hard refusal.
KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("sk-", "sk-proj-"),
    "groq": ("gsk_",),
}

# A caller-supplied key that is obviously not a key at all. Rejecting these
# saves a provider round trip and gives a message the user can act on.
MIN_KEY_LENGTH = 20
MAX_KEY_LENGTH = 256

# Models a caller may name, per provider. An allow-list rather than passing the
# string through: an arbitrary model id is a way to make this deployment issue
# requests of an unknown shape and cost against someone's account, and a typo'd
# id fails at call time with a provider error this layer must not echo.
ALLOWED_MODELS: dict[str, tuple[str, ...]] = {
    "groq": (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound",
        "qwen/qwen3-32b",
    ),
    "openai": (
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ),
}


class CredentialError(ValueError):
    """A caller-supplied credential this layer refuses, with a safe message.

    The message is shown to the caller, so it names the rule and never the
    value. Raised rather than returned because every caller must handle it and a
    `None` return would be indistinguishable from "no key supplied", which is
    the ordinary case.
    """


@dataclass(frozen=True, slots=True)
class ProviderCredentials:
    """One caller's provider configuration. Never logged, never serialised."""

    provider: Provider
    api_key: SecretStr
    model: str = ""

    @property
    def secret(self) -> str:
        return self.api_key.get_secret_value()

    @property
    def fingerprint(self) -> str:
        """A stable, non-reversing id for pooling and for a log line.

        sha256 of provider + key + model, truncated. Used as the pool key and
        as the only form of the key that appears in a log: two requests from the
        same caller share a fingerprint, so a rate limiter or a metric can count
        them without holding the key.

        Truncation to 16 hex characters is 64 bits, which is far past collision
        risk for a pool of eight and keeps a log line readable.
        """
        basis = f"{self.provider}:{self.model}:{self.secret}".encode()
        return hashlib.sha256(basis).hexdigest()[:16]

    @property
    def key_hint(self) -> str:
        """The last four characters, for a UI that must show *which* key.

        Four, not eight: enough for a person with two keys to tell them apart,
        and not enough to narrow a brute force. The prefix is shown instead of
        the body because a provider prefix is public information.
        """
        secret = self.secret
        tail = secret[-4:] if len(secret) >= 8 else ""
        prefix = next(
            (p for p in KEY_PREFIXES.get(self.provider, ()) if secret.startswith(p)),
            "",
        )
        return f"{prefix}…{tail}" if tail else "set"

    def describe(self) -> dict[str, Any]:
        """What every other layer is allowed to see. No key material."""
        return {
            "provider": self.provider,
            "model": self.model,
            "key_hint": self.key_hint,
            "fingerprint": self.fingerprint,
        }

    def __repr__(self) -> str:  # pragma: no cover - defensive, like SecretStr
        # Overridden so a traceback, a log line or a repr in a debugger cannot
        # publish the key. `SecretStr` already masks the field, but a dataclass
        # repr printing `api_key=SecretStr('**********')` still invites someone
        # to reach for `.get_secret_value()` in the same breath.
        return f"ProviderCredentials({self.provider}, {self.key_hint})"


def _validate_key(provider: str, key: str) -> None:
    """Refuse an obviously wrong key before it costs a round trip."""
    if not (MIN_KEY_LENGTH <= len(key) <= MAX_KEY_LENGTH):
        raise CredentialError(
            "That does not look like an API key. Check you pasted the whole "
            "thing and nothing else."
        )
    if any(character.isspace() for character in key):
        raise CredentialError(
            "The key contains a space or a line break. Paste it without "
            "surrounding quotes or whitespace."
        )
    expected = KEY_PREFIXES.get(provider, ())
    if expected and not key.startswith(expected):
        # A warning path rather than a refusal would be wrong here: the common
        # cause is a key pasted against the wrong provider, and forwarding it
        # produces a provider 401 that this layer is not allowed to echo -- so
        # the caller would see "something went wrong" for a mistake we could
        # name exactly.
        raise CredentialError(
            f"A {provider} key normally starts with "
            f"{' or '.join(expected)}. Did you pick the right provider?"
        )


def credentials_from_headers(
    headers: Mapping[str, str], settings: Settings
) -> ProviderCredentials | None:
    """Read a caller's provider configuration out of the request headers.

    Returns None when no key was supplied, which is the ordinary case and not an
    error. Raises `CredentialError` when a key *was* supplied and is refused, so
    the caller can answer 400 with a message naming the rule.

    Header lookup is case-insensitive via the mapping Starlette provides; a
    plain dict in a test works because the names are matched in both cases.
    """

    def read(name: str) -> str:
        return str(
            headers.get(name) or headers.get(name.lower()) or ""
        ).strip()

    key = read(HEADER_KEY)
    if not key:
        return None

    if not settings.allow_byok:
        raise CredentialError(
            "This deployment does not accept caller-supplied API keys."
        )

    provider = (read(HEADER_PROVIDER) or "").lower()
    if not provider:
        # Inferred from the prefix rather than defaulted, because defaulting
        # sends an OpenAI key to Groq and produces a 401 that reads as a bad
        # key.
        provider = next(
            (
                name
                for name, prefixes in KEY_PREFIXES.items()
                if key.startswith(prefixes)
            ),
            "",
        )
    if provider not in ("groq", "openai"):
        raise CredentialError(
            "Set X-Provider to 'groq' or 'openai' (or use a key whose prefix "
            "identifies the provider)."
        )

    _validate_key(provider, key)

    model = read(HEADER_MODEL)
    if model:
        allowed = ALLOWED_MODELS.get(provider, ())
        if model not in allowed:
            raise CredentialError(
                f"{model!r} is not one of the models this deployment will call "
                f"on your behalf. Supported: {', '.join(allowed)}."
            )

    return ProviderCredentials(
        provider=provider,  # type: ignore[arg-type]
        api_key=SecretStr(key),
        model=model,
    )


def settings_with_credentials(
    settings: Settings, credentials: ProviderCredentials
) -> Settings:
    """A copy of the settings with the caller's chat provider substituted.

    **Only the chat provider.** `embedding_provider`, the Pinecone index and the
    namespaces are untouched: see rule 2 in the module docstring. A caller's key
    paying for embeddings would also mean their key deciding which index this
    deployment writes to, and the embedding dimension is coupled to an index
    Pinecone cannot resize.

    `model_copy(update=...)` rather than constructing a new `Settings`: the
    model is frozen, and rebuilding it from scratch would re-read `.env` and the
    environment, which would silently drop every override the caller's
    `Settings` was built with.
    """
    update: dict[str, Any] = {"llm_provider": credentials.provider}
    if credentials.provider == "openai":
        update["openai_api_key"] = credentials.api_key
        if credentials.model:
            update["openai_model"] = credentials.model
    else:
        update["groq_api_key"] = credentials.api_key
        if credentials.model:
            update["groq_model"] = credentials.model
    # Routing picks from the configured ladder, and the deployment's ladder is
    # not the caller's. Collapse it: one model, the one they asked for.
    update["fast_model"] = ""
    update["deep_model"] = ""
    return settings.model_copy(update=update)


class ClientPool:
    """A bounded, LRU pool of per-credential clients.

    Generic over what it holds: `factory(settings, credentials)` returns
    whatever the caller wants cached — a `Copilot`, a chat client, a compiled
    graph. The pool only knows how to key it and when to drop it.

    **The server's own client is the pool's default entry**, keyed on the empty
    fingerprint, so a caller with no key and a caller with one go through the
    same call site and there is no second code path for the ordinary case.
    """

    def __init__(
        self,
        settings: Settings,
        factory: Callable[[Settings, ProviderCredentials | None], Any],
        *,
        size: int | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._size = max(1, int(size or settings.byok_pool_size))
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, credentials: ProviderCredentials | None = None) -> Any:
        """The client for these credentials, built on first use."""
        key = credentials.fingerprint if credentials else ""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._hits += 1
                return self._entries[key]
            self._misses += 1

        # Built outside the lock: constructing a client may compile a graph, and
        # holding the lock through that would serialise every first request in
        # the process behind one slow one.
        resolved = (
            settings_with_credentials(self._settings, credentials)
            if credentials
            else self._settings
        )
        client = self._factory(resolved, credentials)

        with self._lock:
            # Another thread may have built the same entry while this one was
            # outside the lock. Keep theirs and drop ours rather than replacing
            # a client another request may already be using.
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            self._entries[key] = client
            while len(self._entries) > self._size:
                # Never evict the server's own entry: it is the one every
                # unkeyed caller uses, and rebuilding it recompiles the graph.
                evict_key, _ = next(
                    (
                        item
                        for item in self._entries.items()
                        if item[0] != ""
                    ),
                    (None, None),
                )
                if evict_key is None:
                    break
                self._entries.pop(evict_key, None)
                self._evictions += 1
            return client

    def clear(self) -> int:
        """Drop every caller-supplied entry, keeping the server's own.

        Exists so an operator can revoke every held key at once -- the pool is
        the only place a caller's key lives, so this is what "forget all keys"
        means.
        """
        with self._lock:
            keys = [key for key in self._entries if key != ""]
            for key in keys:
                self._entries.pop(key, None)
            return len(keys)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._entries),
                "max_size": self._size,
                "byok_entries": sum(1 for key in self._entries if key),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }
