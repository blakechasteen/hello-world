"""
Rate Limiting for Promptly Matrix Bot

Provides:
- Token bucket algorithm for rate limiting
- Per-user and per-room limits
- Sliding window tracking
- Burst allowance

Usage:
    from bot.rate_limiter import RateLimiter, RateLimitExceeded

    limiter = RateLimiter(
        requests_per_minute=10,
        burst_size=20
    )

    # Check if request allowed
    if await limiter.is_allowed(user_id="@user:matrix.org"):
        # Process request
        pass
    else:
        raise RateLimitExceeded("Too many requests")
"""

import asyncio
import logging
import time
from typing import Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting

    Algorithm:
    - Bucket holds tokens (max = burst_size)
    - Tokens refilled at constant rate
    - Request consumes 1 token
    - Request allowed if tokens available
    """
    capacity: int  # Max tokens (burst size)
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill_time: float = field(init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill_time = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill_time

        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate

        # Refill bucket (capped at capacity)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens available and consumed, False otherwise
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False

    def peek(self) -> float:
        """Check available tokens without consuming"""
        self._refill()
        return self.tokens

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens available

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens available now)
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        # Calculate tokens needed
        tokens_needed = tokens - self.tokens

        # Calculate wait time
        return tokens_needed / self.refill_rate


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 10
    burst_size: Optional[int] = None  # If None, = requests_per_minute * 2
    per_user: bool = True
    per_room: bool = False
    # Track request history for metrics
    history_size: int = 100


@dataclass
class RateLimitStats:
    """Rate limit statistics"""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=100))

    def denial_rate(self) -> float:
        """Calculate denial rate"""
        if self.total_requests == 0:
            return 0.0
        return self.denied_requests / self.total_requests

    def requests_per_second(self) -> float:
        """Calculate requests per second from recent history"""
        if len(self.recent_requests) < 2:
            return 0.0

        # Calculate time span of recent requests
        oldest = self.recent_requests[0]["timestamp"]
        newest = self.recent_requests[-1]["timestamp"]
        time_span = newest - oldest

        if time_span == 0:
            return 0.0

        return len(self.recent_requests) / time_span


class RateLimiter:
    """
    Rate limiter using token bucket algorithm

    Supports:
    - Per-user rate limiting
    - Per-room rate limiting
    - Combined user+room rate limiting
    - Burst allowance
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        burst_size: Optional[int] = None,
        per_user: bool = True,
        per_room: bool = False,
        history_size: int = 100
    ):
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size if burst_size else requests_per_minute * 2,
            per_user=per_user,
            per_room=per_room,
            history_size=history_size
        )

        # Token buckets keyed by identifier (user_id, room_id, or both)
        self.buckets: Dict[str, TokenBucket] = {}

        # Statistics
        self.stats = RateLimitStats(recent_requests=deque(maxlen=history_size))

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Refill rate (tokens per second)
        self.refill_rate = self.config.requests_per_minute / 60.0

        logger.info(
            f"RateLimiter initialized: {self.config.requests_per_minute} req/min, "
            f"burst={self.config.burst_size}, per_user={per_user}, per_room={per_room}"
        )

    def _get_key(self, user_id: Optional[str] = None, room_id: Optional[str] = None) -> str:
        """
        Generate key for token bucket

        Args:
            user_id: User identifier
            room_id: Room identifier

        Returns:
            Key for bucket lookup
        """
        parts = []

        if self.config.per_user and user_id:
            parts.append(f"user:{user_id}")

        if self.config.per_room and room_id:
            parts.append(f"room:{room_id}")

        if not parts:
            # Global rate limit
            return "global"

        return ":".join(parts)

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get existing bucket or create new one"""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=self.refill_rate
            )
            logger.debug(f"Created token bucket for '{key}'")

        return self.buckets[key]

    async def is_allowed(
        self,
        user_id: Optional[str] = None,
        room_id: Optional[str] = None,
        tokens: int = 1
    ) -> bool:
        """
        Check if request is allowed under rate limit

        Args:
            user_id: User identifier
            room_id: Room identifier
            tokens: Number of tokens to consume (default: 1)

        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            self.stats.total_requests += 1

            # Get bucket
            key = self._get_key(user_id, room_id)
            bucket = self._get_or_create_bucket(key)

            # Try to consume tokens
            allowed = bucket.consume(tokens)

            # Record request
            self.stats.recent_requests.append({
                "timestamp": time.time(),
                "user_id": user_id,
                "room_id": room_id,
                "allowed": allowed,
                "tokens": tokens,
                "key": key
            })

            if allowed:
                self.stats.allowed_requests += 1
                logger.debug(f"Request allowed for '{key}' ({bucket.peek():.1f} tokens remaining)")
            else:
                self.stats.denied_requests += 1
                wait_time = bucket.wait_time(tokens)
                logger.warning(f"Request denied for '{key}' (rate limit exceeded, retry in {wait_time:.1f}s)")

            return allowed

    async def wait_if_needed(
        self,
        user_id: Optional[str] = None,
        room_id: Optional[str] = None,
        tokens: int = 1,
        max_wait: float = 10.0
    ) -> bool:
        """
        Wait if rate limited, then allow request

        Args:
            user_id: User identifier
            room_id: Room identifier
            tokens: Number of tokens to consume
            max_wait: Maximum wait time in seconds

        Returns:
            True if request allowed (after waiting if needed), False if max_wait exceeded
        """
        async with self._lock:
            key = self._get_key(user_id, room_id)
            bucket = self._get_or_create_bucket(key)

            # Calculate wait time
            wait_time = bucket.wait_time(tokens)

            if wait_time == 0:
                # No wait needed
                return await self.is_allowed(user_id, room_id, tokens)

            if wait_time > max_wait:
                # Wait time exceeds max
                logger.warning(
                    f"Rate limit wait time ({wait_time:.1f}s) exceeds max ({max_wait:.1f}s) for '{key}'"
                )
                return False

        # Wait outside of lock
        logger.info(f"Waiting {wait_time:.1f}s for rate limit on '{key}'")
        await asyncio.sleep(wait_time)

        # Retry after wait
        return await self.is_allowed(user_id, room_id, tokens)

    async def get_wait_time(
        self,
        user_id: Optional[str] = None,
        room_id: Optional[str] = None,
        tokens: int = 1
    ) -> float:
        """
        Get wait time until request would be allowed

        Args:
            user_id: User identifier
            room_id: Room identifier
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if allowed now)
        """
        async with self._lock:
            key = self._get_key(user_id, room_id)
            bucket = self._get_or_create_bucket(key)
            return bucket.wait_time(tokens)

    def get_stats(self) -> RateLimitStats:
        """Get rate limiting statistics"""
        return self.stats

    def get_bucket_stats(self) -> Dict[str, dict]:
        """Get statistics for all buckets"""
        return {
            key: {
                "tokens_available": bucket.peek(),
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "utilization": 1.0 - (bucket.peek() / bucket.capacity)
            }
            for key, bucket in self.buckets.items()
        }

    async def reset(self, user_id: Optional[str] = None, room_id: Optional[str] = None):
        """
        Reset rate limit for user/room

        Args:
            user_id: User identifier (if None, reset all users)
            room_id: Room identifier (if None, reset all rooms)
        """
        async with self._lock:
            if user_id is None and room_id is None:
                # Reset all
                self.buckets.clear()
                logger.info("Reset all rate limit buckets")
            else:
                # Reset specific bucket
                key = self._get_key(user_id, room_id)
                if key in self.buckets:
                    del self.buckets[key]
                    logger.info(f"Reset rate limit bucket for '{key}'")


def rate_limit(
    requests_per_minute: int = 10,
    burst_size: Optional[int] = None,
    per_user: bool = True,
    per_room: bool = False,
    key_func: Optional[callable] = None
):
    """
    Decorator for rate limiting async functions

    Args:
        requests_per_minute: Request limit per minute
        burst_size: Burst allowance (default: 2x requests_per_minute)
        per_user: Apply limit per user
        per_room: Apply limit per room
        key_func: Custom function to extract (user_id, room_id) from args

    Example:
        @rate_limit(requests_per_minute=10, per_user=True)
        async def handle_command(user_id: str, command: str):
            # Command handler
            pass
    """
    limiter = RateLimiter(
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        per_user=per_user,
        per_room=per_room
    )

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id and room_id
            if key_func:
                user_id, room_id = key_func(*args, **kwargs)
            else:
                # Default: assume first arg is user_id, second is room_id
                user_id = args[0] if len(args) > 0 else None
                room_id = args[1] if len(args) > 1 else None

            # Check rate limit
            if not await limiter.is_allowed(user_id=user_id, room_id=room_id):
                wait_time = await limiter.get_wait_time(user_id=user_id, room_id=room_id)
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry in {wait_time:.1f}s"
                )

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Global rate limiters for different scopes
global_user_limiter = RateLimiter(requests_per_minute=20, per_user=True)
global_room_limiter = RateLimiter(requests_per_minute=50, per_room=True)
