"""
Rate Limiting: Resource Management and Fair Usage

Part 5: Production Hardening - Day 23

Implements rate limiting for the Context Department:
- Token bucket (for bursty traffic)
- Sliding window (for precise rate limiting)
- Concurrent semaphore (for parallelism control)
- Memory usage limiting

Prevents:
- Resource exhaustion
- Runaway queries
- Unfair resource monopolization
"""

import asyncio
import time
from typing import Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

from HoloLoom.context.error_handling import RateLimitExceededError


# ============================================================================
# Rate Limiter Types
# ============================================================================

class RateLimiterType(Enum):
    """Type of rate limiter"""
    TOKEN_BUCKET = "token_bucket"      # Bursty traffic
    SLIDING_WINDOW = "sliding_window"  # Precise limiting
    CONCURRENT = "concurrent"          # Parallelism control


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RateLimiterConfig:
    """Rate limiter configuration"""
    # Token bucket settings
    rate: float = 10.0           # Tokens per second
    capacity: int = 20           # Bucket capacity (burst size)

    # Sliding window settings
    window_size: float = 1.0     # Window size in seconds
    max_requests: int = 10       # Max requests per window

    # Concurrent settings
    max_concurrent: int = 10     # Max concurrent operations

    # General
    name: str = "default"        # Rate limiter name


# ============================================================================
# Token Bucket Rate Limiter
# ============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter

    Good for:
    - Bursty traffic patterns
    - Allowing temporary bursts
    - Smooth long-term rate control

    Algorithm:
    - Bucket has capacity C tokens
    - Tokens refill at rate R per second
    - Each request consumes 1 token
    - Request allowed if tokens available
    """

    def __init__(self, rate: float, capacity: int, name: str = "default"):
        """
        Initialize token bucket rate limiter

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Bucket capacity (max burst size)
            name: Rate limiter name
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.time()

        # Statistics
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rate limit exceeded
        """
        self.total_requests += 1
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.total_allowed += 1
            return True
        else:
            self.total_rejected += 1
            return False

    async def acquire_wait(self, tokens: int = 1, timeout: Optional[float] = None):
        """
        Acquire tokens, waiting if necessary

        Args:
            tokens: Number of tokens to acquire
            timeout: Max wait time in seconds (None = no timeout)

        Raises:
            RateLimitExceededError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            if await self.acquire(tokens):
                return

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                raise RateLimitExceededError(
                    f"Rate limit timeout exceeded for '{self.name}' "
                    f"(rate: {self.rate}/s, capacity: {self.capacity})"
                )

            # Calculate wait time
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.rate
            await asyncio.sleep(min(wait_time, 0.1))  # Max 100ms wait per iteration

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            "name": self.name,
            "type": "token_bucket",
            "rate": self.rate,
            "capacity": self.capacity,
            "current_tokens": self.tokens,
            "total_requests": self.total_requests,
            "total_allowed": self.total_allowed,
            "total_rejected": self.total_rejected,
            "rejection_rate": self.total_rejected / max(1, self.total_requests)
        }

    def reset(self):
        """Reset rate limiter"""
        self.tokens = float(self.capacity)
        self.last_update = time.time()
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0


# ============================================================================
# Sliding Window Rate Limiter
# ============================================================================

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter

    Good for:
    - Precise rate limiting
    - Preventing bursts at window boundaries
    - Strict QPS enforcement

    Algorithm:
    - Track timestamps of last N requests
    - Allow request if count in window < max
    - Slide window based on current time
    """

    def __init__(self, window_size: float, max_requests: int, name: str = "default"):
        """
        Initialize sliding window rate limiter

        Args:
            window_size: Window size in seconds
            max_requests: Max requests per window
            name: Rate limiter name
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.name = name
        self.request_times = deque()

        # Statistics
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0

    async def acquire(self) -> bool:
        """
        Try to acquire request slot

        Returns:
            True if acquired, False if rate limit exceeded
        """
        self.total_requests += 1
        self._cleanup_old_requests()

        if len(self.request_times) < self.max_requests:
            self.request_times.append(time.time())
            self.total_allowed += 1
            return True
        else:
            self.total_rejected += 1
            return False

    async def acquire_wait(self, timeout: Optional[float] = None):
        """
        Acquire request slot, waiting if necessary

        Args:
            timeout: Max wait time in seconds (None = no timeout)

        Raises:
            RateLimitExceededError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            if await self.acquire():
                return

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                raise RateLimitExceededError(
                    f"Rate limit timeout exceeded for '{self.name}' "
                    f"(window: {self.window_size}s, max: {self.max_requests})"
                )

            # Wait for oldest request to expire
            if self.request_times:
                oldest = self.request_times[0]
                wait_time = (oldest + self.window_size) - time.time()
                await asyncio.sleep(max(0.01, min(wait_time, 0.1)))
            else:
                await asyncio.sleep(0.01)

    def _cleanup_old_requests(self):
        """Remove requests outside window"""
        cutoff = time.time() - self.window_size
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()

    def get_current_rate(self) -> float:
        """Get current request rate (requests per second)"""
        self._cleanup_old_requests()
        if not self.request_times:
            return 0.0
        return len(self.request_times) / self.window_size

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        self._cleanup_old_requests()
        return {
            "name": self.name,
            "type": "sliding_window",
            "window_size": self.window_size,
            "max_requests": self.max_requests,
            "current_requests": len(self.request_times),
            "current_rate": self.get_current_rate(),
            "total_requests": self.total_requests,
            "total_allowed": self.total_allowed,
            "total_rejected": self.total_rejected,
            "rejection_rate": self.total_rejected / max(1, self.total_requests)
        }

    def reset(self):
        """Reset rate limiter"""
        self.request_times.clear()
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0


# ============================================================================
# Concurrent Limiter
# ============================================================================

class ConcurrentLimiter:
    """
    Concurrent operation limiter

    Good for:
    - Limiting parallelism
    - Preventing resource exhaustion
    - Controlling concurrent backend calls

    Algorithm:
    - Semaphore with max concurrent count
    - Async context manager for automatic release
    """

    def __init__(self, max_concurrent: int, name: str = "default"):
        """
        Initialize concurrent limiter

        Args:
            max_concurrent: Max concurrent operations
            name: Limiter name
        """
        self.max_concurrent = max_concurrent
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self.current_concurrent = 0
        self.total_acquires = 0
        self.peak_concurrent = 0

    async def __aenter__(self):
        """Acquire semaphore (context manager entry)"""
        await self.semaphore.acquire()
        self.current_concurrent += 1
        self.total_acquires += 1
        self.peak_concurrent = max(self.peak_concurrent, self.current_concurrent)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore (context manager exit)"""
        self.current_concurrent -= 1
        self.semaphore.release()

    async def acquire(self):
        """Acquire semaphore (manual)"""
        await self.semaphore.acquire()
        self.current_concurrent += 1
        self.total_acquires += 1
        self.peak_concurrent = max(self.peak_concurrent, self.current_concurrent)

    def release(self):
        """Release semaphore (manual)"""
        self.current_concurrent -= 1
        self.semaphore.release()

    def get_stats(self) -> dict:
        """Get limiter statistics"""
        return {
            "name": self.name,
            "type": "concurrent",
            "max_concurrent": self.max_concurrent,
            "current_concurrent": self.current_concurrent,
            "peak_concurrent": self.peak_concurrent,
            "total_acquires": self.total_acquires,
            "utilization": self.current_concurrent / max(1, self.max_concurrent)
        }

    def reset_stats(self):
        """Reset statistics (keep semaphore state)"""
        self.total_acquires = 0
        self.peak_concurrent = self.current_concurrent


# ============================================================================
# Unified Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Unified rate limiter combining multiple strategies

    Combines:
    - Token bucket (bursty traffic)
    - Sliding window (precise limiting)
    - Concurrent limiter (parallelism control)
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Initialize unified rate limiter

        Args:
            config: Rate limiter configuration (uses defaults if None)
        """
        self.config = config or RateLimiterConfig()

        # Create component limiters
        self.token_bucket = TokenBucketRateLimiter(
            rate=self.config.rate,
            capacity=self.config.capacity,
            name=f"{self.config.name}_token"
        )

        self.sliding_window = SlidingWindowRateLimiter(
            window_size=self.config.window_size,
            max_requests=self.config.max_requests,
            name=f"{self.config.name}_window"
        )

        self.concurrent = ConcurrentLimiter(
            max_concurrent=self.config.max_concurrent,
            name=f"{self.config.name}_concurrent"
        )

    async def acquire(self) -> bool:
        """
        Try to acquire rate limit

        Checks both token bucket and sliding window

        Returns:
            True if acquired, False if rate limit exceeded
        """
        # Check token bucket (fast path)
        if not await self.token_bucket.acquire():
            return False

        # Check sliding window (precise)
        if not await self.sliding_window.acquire():
            return False

        return True

    async def __aenter__(self):
        """Acquire rate limit and concurrency limit (context manager)"""
        # Acquire rate limit
        if not await self.acquire():
            raise RateLimitExceededError(
                f"Rate limit exceeded for '{self.config.name}'"
            )

        # Acquire concurrency limit
        await self.concurrent.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release concurrency limit (context manager)"""
        self.concurrent.release()

    def get_stats(self) -> dict:
        """Get combined statistics"""
        return {
            "name": self.config.name,
            "token_bucket": self.token_bucket.get_stats(),
            "sliding_window": self.sliding_window.get_stats(),
            "concurrent": self.concurrent.get_stats()
        }

    def reset(self):
        """Reset all limiters"""
        self.token_bucket.reset()
        self.sliding_window.reset()
        self.concurrent.reset_stats()


# ============================================================================
# Factory Functions
# ============================================================================

def create_rate_limiter(
    rate: float = 10.0,
    capacity: int = 20,
    window_size: float = 1.0,
    max_requests: int = 10,
    max_concurrent: int = 10,
    name: str = "default"
) -> RateLimiter:
    """
    Create unified rate limiter

    Args:
        rate: Token refill rate (tokens per second)
        capacity: Token bucket capacity (burst size)
        window_size: Sliding window size (seconds)
        max_requests: Max requests per window
        max_concurrent: Max concurrent operations
        name: Rate limiter name

    Returns:
        Configured RateLimiter instance
    """
    config = RateLimiterConfig(
        rate=rate,
        capacity=capacity,
        window_size=window_size,
        max_requests=max_requests,
        max_concurrent=max_concurrent,
        name=name
    )
    return RateLimiter(config)


def create_token_bucket_limiter(
    rate: float = 10.0,
    capacity: int = 20,
    name: str = "default"
) -> TokenBucketRateLimiter:
    """
    Create token bucket rate limiter

    Args:
        rate: Token refill rate (tokens per second)
        capacity: Bucket capacity (burst size)
        name: Rate limiter name

    Returns:
        TokenBucketRateLimiter instance
    """
    return TokenBucketRateLimiter(rate, capacity, name)


def create_sliding_window_limiter(
    window_size: float = 1.0,
    max_requests: int = 10,
    name: str = "default"
) -> SlidingWindowRateLimiter:
    """
    Create sliding window rate limiter

    Args:
        window_size: Window size in seconds
        max_requests: Max requests per window
        name: Rate limiter name

    Returns:
        SlidingWindowRateLimiter instance
    """
    return SlidingWindowRateLimiter(window_size, max_requests, name)


def create_concurrent_limiter(
    max_concurrent: int = 10,
    name: str = "default"
) -> ConcurrentLimiter:
    """
    Create concurrent limiter

    Args:
        max_concurrent: Max concurrent operations
        name: Limiter name

    Returns:
        ConcurrentLimiter instance
    """
    return ConcurrentLimiter(max_concurrent, name)
