"""
Distributed Rate Limiting

Redis-backed rate limiting for horizontal scalability.

Features:
- Sliding window algorithm (accurate rate limiting)
- Distributed across multiple servers (Redis-backed)
- IP reputation scoring (auto-block bad actors)
- Adaptive throttling (cost-based limiting)
- Temporary IP bans
- Persistent across restarts

References:
- OWASP API Security Top 10 (API4:2023 - Unrestricted Resource Consumption)
- Cloudflare Rate Limiting Best Practices

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional Redis dependency (graceful degradation)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️  redis not installed. Rate limiting will be in-memory only!")

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(message)


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[int] = None  # Seconds to wait (if denied)


class DistributedRateLimiter:
    """
    Redis-backed distributed rate limiter using sliding window algorithm.

    Works across multiple servers (horizontally scalable).

    Algorithms:
    - Sliding window: Accurate rate limiting (no burst at window boundary)
    - IP reputation: Track bad behavior, auto-block repeat offenders
    - Adaptive throttling: Different costs for different operations

    Usage:
        >>> limiter = DistributedRateLimiter(redis_url="redis://localhost")
        >>> allowed, remaining = await limiter.check_rate_limit("user_123")
        >>> if not allowed:
        ...     raise RateLimitExceeded("Too many requests")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_requests: int = 60,
        window_seconds: int = 60,
        enable_ip_reputation: bool = True
    ):
        """
        Initialize distributed rate limiter.

        Args:
            redis_url: Redis connection URL
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            enable_ip_reputation: Enable IP reputation tracking
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enable_ip_reputation = enable_ip_reputation

        # Connect to Redis
        if REDIS_AVAILABLE:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            logger.info(f"✅ Connected to Redis: {redis_url}")
        else:
            self.redis = None
            logger.warning("⚠️  Redis not available, using in-memory fallback")
            # In-memory fallback (not distributed!)
            self._memory_store = {}

    async def check_rate_limit(
        self,
        client_id: str,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if client is within rate limit.

        Args:
            client_id: Client identifier (user ID, IP, API key)
            cost: Request cost (default: 1, expensive ops: 10)

        Returns:
            RateLimitResult with allowed status and remaining quota

        Example:
            >>> result = await limiter.check_rate_limit("user_123", cost=5)
            >>> if result.allowed:
            ...     # Process request
            >>> else:
            ...     raise RateLimitExceeded(
            ...         "Rate limit exceeded",
            ...         retry_after=result.retry_after
            ...     )
        """
        if self.redis:
            return await self._check_rate_limit_redis(client_id, cost)
        else:
            return await self._check_rate_limit_memory(client_id, cost)

    async def _check_rate_limit_redis(
        self,
        client_id: str,
        cost: int
    ) -> RateLimitResult:
        """Check rate limit using Redis (distributed)."""
        key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - self.window_seconds

        # Redis transaction (atomic)
        pipe = self.redis.pipeline()

        # Remove old requests outside window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        pipe.zcard(key)

        # Execute read operations
        pipe.execute()

        # Get current count
        current_count = self.redis.zcard(key)

        # Check if limit would be exceeded
        if (current_count + cost) > self.max_requests:
            # Denied
            remaining = max(0, self.max_requests - current_count)
            reset_at = now + self.window_seconds
            retry_after = int(self.window_seconds)

            logger.warning(
                f"🚫 Rate limit exceeded: {client_id} "
                f"(current: {current_count}, cost: {cost}, limit: {self.max_requests})"
            )

            return RateLimitResult(
                allowed=False,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after
            )

        # Allowed - add this request
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self.window_seconds)
        pipe.execute()

        remaining = max(0, self.max_requests - current_count - cost)
        reset_at = now + self.window_seconds

        logger.debug(
            f"✅ Rate limit ok: {client_id} "
            f"(used: {current_count + cost}/{self.max_requests}, remaining: {remaining})"
        )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at
        )

    async def _check_rate_limit_memory(
        self,
        client_id: str,
        cost: int
    ) -> RateLimitResult:
        """Check rate limit using in-memory store (single-server only)."""
        key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - self.window_seconds

        # Get or create request list
        if key not in self._memory_store:
            self._memory_store[key] = []

        requests = self._memory_store[key]

        # Remove old requests
        requests = [t for t in requests if t >= window_start]
        self._memory_store[key] = requests

        # Check limit
        current_count = len(requests)

        if (current_count + cost) > self.max_requests:
            # Denied
            remaining = max(0, self.max_requests - current_count)
            reset_at = now + self.window_seconds
            retry_after = int(self.window_seconds)

            return RateLimitResult(
                allowed=False,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after
            )

        # Allowed - add this request
        for _ in range(cost):
            self._memory_store[key].append(now)

        remaining = max(0, self.max_requests - current_count - cost)
        reset_at = now + self.window_seconds

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=reset_at
        )

    async def get_ip_reputation(self, ip: str) -> float:
        """
        Get IP reputation score (0.0 = bad, 1.0 = good).

        Tracks:
        - Failed auth attempts
        - Rate limit violations
        - Suspicious patterns

        Args:
            ip: IP address

        Returns:
            Reputation score (0.0-1.0)
        """
        if not self.enable_ip_reputation:
            return 1.0

        key = f"reputation:{ip}"

        if self.redis:
            score = self.redis.get(key)
        else:
            score = self._memory_store.get(key)

        if score is None:
            return 1.0  # Innocent until proven guilty

        return float(score)

    async def decrease_reputation(
        self,
        ip: str,
        amount: float = 0.1,
        reason: str = "rate_limit_violation"
    ):
        """
        Decrease IP reputation (bad behavior).

        Args:
            ip: IP address
            amount: Amount to decrease (0.0-1.0)
            reason: Reason for decrease
        """
        if not self.enable_ip_reputation:
            return

        key = f"reputation:{ip}"
        current = await self.get_ip_reputation(ip)
        new_score = max(0.0, current - amount)

        if self.redis:
            self.redis.set(key, str(new_score), ex=86400)  # 24-hour TTL
        else:
            self._memory_store[key] = new_score

        logger.warning(
            f"⬇️  Decreased reputation: {ip} "
            f"({current:.2f} → {new_score:.2f}, reason: {reason})"
        )

        # Auto-block if reputation too low
        if new_score < 0.3:
            await self.block_ip(ip, duration=3600, reason="low_reputation")

    async def block_ip(
        self,
        ip: str,
        duration: int = 3600,
        reason: str = "manual_block"
    ):
        """
        Block IP temporarily.

        Args:
            ip: IP address to block
            duration: Block duration in seconds
            reason: Reason for block
        """
        key = f"blocked:{ip}"

        if self.redis:
            self.redis.set(key, reason, ex=duration)
        else:
            self._memory_store[key] = (time.time() + duration, reason)

        logger.warning(
            f"🚫 Blocked IP: {ip} "
            f"(duration: {duration}s, reason: {reason})"
        )

    async def unblock_ip(self, ip: str):
        """
        Unblock IP manually.

        Args:
            ip: IP address to unblock
        """
        key = f"blocked:{ip}"

        if self.redis:
            self.redis.delete(key)
        else:
            self._memory_store.pop(key, None)

        logger.info(f"✅ Unblocked IP: {ip}")

    async def is_blocked(self, ip: str) -> Tuple[bool, Optional[str]]:
        """
        Check if IP is blocked.

        Args:
            ip: IP address

        Returns:
            (is_blocked, reason)
        """
        key = f"blocked:{ip}"

        if self.redis:
            reason = self.redis.get(key)
            return reason is not None, reason
        else:
            if key in self._memory_store:
                expires_at, reason = self._memory_store[key]
                if time.time() < expires_at:
                    return True, reason
                else:
                    # Expired, clean up
                    del self._memory_store[key]

            return False, None

    async def get_statistics(self, client_id: str) -> dict:
        """
        Get rate limit statistics for client.

        Args:
            client_id: Client identifier

        Returns:
            Statistics dictionary
        """
        key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - self.window_seconds

        if self.redis:
            # Count requests in window
            count = self.redis.zcount(key, window_start, now)
        else:
            requests = self._memory_store.get(key, [])
            count = len([t for t in requests if t >= window_start])

        remaining = max(0, self.max_requests - count)
        reset_at = now + self.window_seconds

        return {
            "current_count": count,
            "max_requests": self.max_requests,
            "remaining": remaining,
            "window_seconds": self.window_seconds,
            "reset_at": reset_at,
            "utilization": count / self.max_requests if self.max_requests > 0 else 0.0
        }


# Factory function
def create_rate_limiter(
    redis_url: Optional[str] = None,
    max_requests: int = 60,
    window_seconds: int = 60,
    enable_ip_reputation: bool = True
) -> DistributedRateLimiter:
    """
    Create distributed rate limiter with Redis URL from environment or provided.

    Args:
        redis_url: Optional Redis URL (defaults to REDIS_URL env var)
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds
        enable_ip_reputation: Enable IP reputation tracking

    Returns:
        DistributedRateLimiter instance

    Usage:
        >>> import os
        >>> os.environ["REDIS_URL"] = "redis://localhost:6379"
        >>> limiter = create_rate_limiter()
    """
    import os

    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    return DistributedRateLimiter(
        redis_url=redis_url,
        max_requests=max_requests,
        window_seconds=window_seconds,
        enable_ip_reputation=enable_ip_reputation
    )


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate rate limiting."""
        print("=" * 70)
        print("Distributed Rate Limiting Demo")
        print("=" * 70)

        # Initialize limiter (in-memory fallback if Redis not available)
        limiter = DistributedRateLimiter(
            redis_url="redis://localhost:6379",
            max_requests=10,  # 10 requests
            window_seconds=60  # per minute
        )

        # Simulate requests
        client_id = "user_123"

        print(f"\nSimulating requests for client: {client_id}")
        print(f"Limit: {limiter.max_requests} requests per {limiter.window_seconds}s\n")

        for i in range(15):
            result = await limiter.check_rate_limit(client_id, cost=1)

            if result.allowed:
                print(
                    f"✅ Request {i+1:2d}: Allowed "
                    f"(remaining: {result.remaining:2d}/{limiter.max_requests})"
                )
            else:
                print(
                    f"🚫 Request {i+1:2d}: Rate limited "
                    f"(retry after: {result.retry_after}s)"
                )

        # Statistics
        print(f"\n{'=' * 70}")
        print("Statistics")
        print("=" * 70)

        stats = await limiter.get_statistics(client_id)
        print(f"\n   Current usage: {stats['current_count']}/{stats['max_requests']}")
        print(f"   Remaining: {stats['remaining']}")
        print(f"   Utilization: {stats['utilization']:.1%}")
        print(f"   Resets in: {int(stats['reset_at'] - time.time())}s")

        # IP Reputation
        print(f"\n{'=' * 70}")
        print("IP Reputation Demo")
        print("=" * 70)

        ip = "192.168.1.100"
        reputation = await limiter.get_ip_reputation(ip)
        print(f"\n   IP: {ip}")
        print(f"   Reputation: {reputation:.2f}/1.00")

        # Simulate bad behavior
        print(f"\n   Simulating rate limit violations...")
        for i in range(3):
            await limiter.decrease_reputation(ip, amount=0.2, reason="rate_limit_violation")
            reputation = await limiter.get_ip_reputation(ip)
            print(f"   Violation {i+1}: Reputation = {reputation:.2f}")

            if reputation < 0.3:
                is_blocked, reason = await limiter.is_blocked(ip)
                if is_blocked:
                    print(f"   🚫 IP auto-blocked! Reason: {reason}")

        print(f"\n{'=' * 70}\n")

    # Run demo
    asyncio.run(demo())
