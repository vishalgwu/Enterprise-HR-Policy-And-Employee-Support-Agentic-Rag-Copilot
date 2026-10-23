"""A token bucket in front of the deployment's own key.

`/api/chat` is unauthenticated by design -- `CLAUDE.md` explains why, and the
reason has not changed: there is no identity here to scope a conversation to.
What follows from that is a public URL where every question spends the
deployment's provider quota, and Groq's free tier caps tokens per *day*, so one
script can exhaust tomorrow's answers in a minute.

**It limits the shared key, not the caller.** A caller spending their own key
through `app/services/byok.py` is not spending the deployment's quota, so they
are not limited -- which also turns the limiter into the reason to paste a key
rather than a wall for its own sake. `should_limit()` is where that decision
lives, in one place, so a route cannot forget it.

**Honesty about what this is worth.** A token bucket in process memory bounds
*one process*. On a single container it is the real thing. On a platform that
runs several instances behind a load balancer -- Vercel, which is a deployment
target here -- each instance has its own bucket, so the effective limit is
`instances x rate` and a caller who reconnects may land on a fresh one. That is
a meaningful weakening and it is written here rather than left to be discovered:
a deployment that needs a real limit needs a shared counter (Redis, Upstash, the
platform's own edge limiter), and the right place for it is in front of this
application, not inside it. What this does buy, even on Vercel, is that an
accidental retry loop or a single enthusiastic visitor cannot drain the day's
tokens from one instance.

**The subject is a hash, never an address.** `app/services/authorization.py`
produces it. An IP address held in a counter is personal data this product has
no reason to keep, and the limiter needs only equality.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_logger

log = get_logger("ratelimit")

__all__ = ["Decision", "RateLimiter"]


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether one request may proceed, and what to tell the caller.

    `retry_after` is seconds, rounded up, and is what goes in the
    `Retry-After` header -- a 429 with no hint of when to come back is a 429
    that gets retried immediately.
    """

    allowed: bool
    remaining: float = 0.0
    retry_after: int = 0
    limit: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": round(self.remaining, 2),
            "retry_after": self.retry_after,
            "limit": self.limit,
            "reason": self.reason,
        }


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """A per-subject token bucket.

    A bucket rather than a fixed window, because a fixed window lets a caller
    spend the whole minute's allowance in its last second and the next minute's
    in its first -- twice the rate across the boundary, which is exactly when a
    retry loop is hammering. A bucket refills continuously, so the burst is
    `burst` and the sustained rate is `per_minute`, both of which are the
    numbers an operator set.
    """

    def __init__(
        self,
        *,
        per_minute: int = 10,
        burst: int | None = None,
        enabled: bool = True,
        max_subjects: int = 4096,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            per_minute = settings.rate_limit_per_minute
            burst = settings.rate_limit_burst
            enabled = settings.rate_limit_enabled
        self.per_minute = max(1, int(per_minute))
        self.burst = max(1, int(burst or per_minute))
        self.enabled = bool(enabled)
        # Bounded, so a flood of distinct subjects cannot grow this without
        # limit -- which would make the limiter the memory exhaustion it exists
        # to prevent. LRU eviction: a subject evicted is a subject that has not
        # asked recently, and giving them a fresh bucket is the right answer.
        self.max_subjects = max(16, int(max_subjects))
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()
        self._allowed = 0
        self._refused = 0

    @property
    def _refill_per_second(self) -> float:
        return self.per_minute / 60.0

    def check(self, subject: str, *, cost: float = 1.0) -> Decision:
        """Spend one token for `subject`, or refuse.

        `cost` is there so an expensive request can cost more than a cheap one
        -- a streamed supervisor run against four workers is not one question's
        worth of quota. Default 1.
        """
        if not self.enabled:
            return Decision(
                allowed=True, remaining=float(self.burst), limit=self.per_minute
            )

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(subject)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), updated=now)
                self._buckets[subject] = bucket
                while len(self._buckets) > self.max_subjects:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(subject)
                elapsed = max(0.0, now - bucket.updated)
                bucket.tokens = min(
                    float(self.burst),
                    bucket.tokens + elapsed * self._refill_per_second,
                )
                bucket.updated = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                self._allowed += 1
                return Decision(
                    allowed=True,
                    remaining=bucket.tokens,
                    limit=self.per_minute,
                )

            shortfall = cost - bucket.tokens
            # Rounded up, and never zero: a Retry-After of 0 is an invitation to
            # retry in the same millisecond.
            retry_after = max(1, int(shortfall / self._refill_per_second + 0.999))
            self._refused += 1
            return Decision(
                allowed=False,
                remaining=bucket.tokens,
                retry_after=retry_after,
                limit=self.per_minute,
                reason=(
                    f"This demo allows {self.per_minute} questions a minute on "
                    f"its own API key. Add your own key to remove the limit, or "
                    f"try again in {retry_after}s."
                ),
            )

    def should_limit(self, *, byok: bool, is_admin: bool = False) -> bool:
        """Whether this caller's request should be charged against the bucket.

        The whole policy, in one place: a caller spending their own key is not
        spending ours, and an authenticated admin is operating the deployment
        rather than consuming it. Everyone else shares the quota.
        """
        if not self.enabled:
            return False
        return not (byok or is_admin)

    def reset(self, subject: str | None = None) -> int:
        """Forget one subject's bucket, or all of them."""
        with self._lock:
            if subject is None:
                count = len(self._buckets)
                self._buckets.clear()
                return count
            return 1 if self._buckets.pop(subject, None) is not None else 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._allowed + self._refused
            return {
                "enabled": self.enabled,
                "per_minute": self.per_minute,
                "burst": self.burst,
                "subjects": len(self._buckets),
                "allowed": self._allowed,
                "refused": self._refused,
                "refusal_rate": round(self._refused / total, 3) if total else 0.0,
                # Said plainly in the payload, not only in the docstring: a
                # dashboard reporting a limit that is actually per-instance is a
                # dashboard that is wrong on the platform this deploys to.
                "scope": "process",
            }
