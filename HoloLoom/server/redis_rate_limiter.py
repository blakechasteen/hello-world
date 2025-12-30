"""
Redis-based Distributed Rate Limiter for Multi-Instance Deployments

This module implements a distributed rate limiter using Redis to ensure
rate limits are enforced globally across multiple HoloLoom instances.

Features:
- Sliding window algorithm using Redis sorted sets
- Atomic operations with Redis pipelines
- Automatic expiration of old data
- Graceful fallback to in-memory limiting if Redis unavailable
- Support for different limits per endpoint
- Prometheus metrics for monitoring

Created: 2025-11-26
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import HTTPException, Request
from prometheus_client import Counter, Histogram, Gauge

from HoloLoom.utils.security import sanitize_uri

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

rate_limit_checks = Counter(
    'hololoom_rate_limit_checks_total',
    'Total number of rate limit checks',
    ['endpoint', 'result']
)

rate_limit_rejections = Counter(
    'hololoom_rate_limit_rejections_total',
    'Total number of rate limited requests',
    ['endpoint']
)

rate_limit_latency = Histogram(
    'hololoom_rate_limit_check_duration_seconds',
    'Time spent checking rate limits',
    ['endpoint', 'backend']  # backend: redis or memory
)

redis_connection_status = Gauge(
    'hololoom_redis_connection_status',
    'Redis connection status (1=connected, 0=disconnected)'
)


# ============================================================================
# In-Memory Fallback Rate Limiter
# ============================================================================

class InMemoryRateLimiter:
    """
    In-memory rate limiter for fallback when Redis is unavailable.
    Uses sliding window algorithm with deque for efficiency.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize in-memory rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)

    async def check_rate_limit(
        self,
        key: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None
    ) -> Tuple[bool, int, float]:
        """
        Check if request exceeds rate limit.

        Args:
            key: Rate limit key (e.g., "endpoint:ip")
            max_requests: Override max requests (uses default if None)
            window_seconds: Override window (uses default if None)

        Returns:
            Tuple of (allowed, current_count, time_until_reset)
        """
        max_requests = max_requests or self.max_requests
        window_seconds = window_seconds or self.window_seconds

        now = time.time()
        queue = self.requests[key]

        # Remove old requests outside the window
        while queue and queue[0] < now - window_seconds:
            queue.popleft()

        current_count = len(queue)
        allowed = current_count < max_requests

        if allowed:
            queue.append(now)
            current_count += 1

        # Calculate time until oldest request expires
        time_until_reset = 0.0
        if queue:
            time_until_reset = max(0, window_seconds - (now - queue[0]))

        return allowed, current_count, time_until_reset

    async def cleanup_old_entries(self, older_than_seconds: int = 300):
        """
        Clean up old entries that haven't been accessed.

        Args:
            older_than_seconds: Remove entries older than this
        """
        now = time.time()
        keys_to_remove = []

        for key, queue in self.requests.items():
            if not queue or now - queue[-1] > older_than_seconds:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.requests[key]


# ============================================================================
# Redis-based Distributed Rate Limiter
# ============================================================================

class RedisRateLimiter:
    """
    Distributed rate limiter using Redis sorted sets for sliding window.
    Falls back to in-memory limiting if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_max_requests: int = 60,
        default_window_seconds: int = 60,
        key_prefix: str = "ratelimit",
        enable_fallback: bool = True,
        fallback_limiter: Optional[InMemoryRateLimiter] = None,
        connection_timeout: float = 5.0,
        operation_timeout: float = 1.0
    ):
        """
        Initialize Redis rate limiter.

        Args:
            redis_url: Redis connection URL
            default_max_requests: Default max requests in window
            default_window_seconds: Default window size in seconds
            key_prefix: Prefix for Redis keys
            enable_fallback: Enable fallback to in-memory limiting
            fallback_limiter: Custom fallback limiter instance
            connection_timeout: Redis connection timeout
            operation_timeout: Redis operation timeout
        """
        self.redis_url = redis_url
        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self.key_prefix = key_prefix
        self.enable_fallback = enable_fallback
        self.connection_timeout = connection_timeout
        self.operation_timeout = operation_timeout

        # Redis client (initialized on first use)
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._connection_lock = asyncio.Lock()

        # Fallback limiter
        if enable_fallback:
            self.fallback_limiter = fallback_limiter or InMemoryRateLimiter(
                max_requests=default_max_requests,
                window_seconds=default_window_seconds
            )
        else:
            self.fallback_limiter = None

        # Background cleanup task
        self._cleanup_task = None

    async def connect(self) -> bool:
        """
        Connect to Redis with proper error handling.

        Returns:
            True if connected successfully
        """
        async with self._connection_lock:
            if self._connected and self._redis:
                try:
                    # Test connection
                    await asyncio.wait_for(
                        self._redis.ping(),
                        timeout=self.operation_timeout
                    )
                    return True
                except Exception:
                    self._connected = False
                    redis_connection_status.set(0)

            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=self.connection_timeout,
                    socket_timeout=self.operation_timeout
                )

                # Test connection
                await asyncio.wait_for(
                    self._redis.ping(),
                    timeout=self.operation_timeout
                )

                self._connected = True
                redis_connection_status.set(1)
                logger.info(f"Connected to Redis at {sanitize_uri(self.redis_url)}")

                # Start cleanup task if not running
                if not self._cleanup_task or self._cleanup_task.done():
                    self._cleanup_task = asyncio.create_task(self._cleanup_loop())

                return True

            except Exception as e:
                self._connected = False
                redis_connection_status.set(0)
                logger.warning(f"Failed to connect to Redis: {e}")
                return False

    async def disconnect(self):
        """Disconnect from Redis and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.close()
            self._redis = None

        self._connected = False
        redis_connection_status.set(0)

    async def check_rate_limit(
        self,
        key: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None
    ) -> Tuple[bool, int, float]:
        """
        Check if request exceeds rate limit using Redis sliding window.

        Args:
            key: Rate limit key (e.g., "endpoint:ip")
            max_requests: Override max requests
            window_seconds: Override window

        Returns:
            Tuple of (allowed, current_count, time_until_reset)
        """
        max_requests = max_requests or self.default_max_requests
        window_seconds = window_seconds or self.default_window_seconds

        # Try Redis first
        if await self.connect():
            try:
                with rate_limit_latency.labels(endpoint=key.split(':')[0], backend='redis').time():
                    return await self._check_redis(key, max_requests, window_seconds)
            except Exception as e:
                logger.warning(f"Redis rate limit check failed: {e}")
                self._connected = False
                redis_connection_status.set(0)

        # Fallback to in-memory if enabled
        if self.enable_fallback and self.fallback_limiter:
            with rate_limit_latency.labels(endpoint=key.split(':')[0], backend='memory').time():
                return await self.fallback_limiter.check_rate_limit(
                    key, max_requests, window_seconds
                )

        # If no fallback, deny request (fail-closed)
        return False, 0, float(window_seconds)

    async def _check_redis(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int, float]:
        """
        Internal Redis rate limit check using sorted sets.

        Args:
            key: Rate limit key
            max_requests: Max requests in window
            window_seconds: Window size

        Returns:
            Tuple of (allowed, current_count, time_until_reset)
        """
        redis_key = f"{self.key_prefix}:{key}"
        now = time.time()
        window_start = now - window_seconds

        # Use pipeline for atomic operations
        async with self._redis.pipeline() as pipe:
            # Remove old entries outside the window
            pipe.zremrangebyscore(redis_key, '-inf', window_start)

            # Count current entries in window
            pipe.zcard(redis_key)

            # Execute pipeline
            results = await asyncio.wait_for(
                pipe.execute(),
                timeout=self.operation_timeout
            )

            current_count = results[1]

            # Check if under limit
            if current_count >= max_requests:
                # Get oldest entry to calculate reset time
                oldest = await asyncio.wait_for(
                    self._redis.zrange(redis_key, 0, 0, withscores=True),
                    timeout=self.operation_timeout
                )

                if oldest:
                    oldest_timestamp = oldest[0][1]
                    time_until_reset = max(0, window_seconds - (now - oldest_timestamp))
                else:
                    time_until_reset = 0.0

                return False, current_count, time_until_reset

            # Add current request
            async with self._redis.pipeline() as pipe:
                pipe.zadd(redis_key, {str(now): now})
                pipe.expire(redis_key, window_seconds + 60)  # Extra buffer for safety
                await asyncio.wait_for(
                    pipe.execute(),
                    timeout=self.operation_timeout
                )

            return True, current_count + 1, float(window_seconds)

    async def _cleanup_loop(self):
        """Background task to clean up expired keys."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Cleanup in-memory fallback if enabled
                if self.fallback_limiter:
                    await self.fallback_limiter.cleanup_old_entries()

                # Redis handles expiration automatically via TTL

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    @asynccontextmanager
    async def rate_limit_context(
        self,
        request: Request,
        endpoint: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None
    ):
        """
        Context manager for rate limiting with automatic metrics.

        Args:
            request: FastAPI request
            endpoint: Endpoint name for metrics
            max_requests: Override max requests
            window_seconds: Override window

        Yields:
            None if allowed

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        key = f"{endpoint}:{client_ip}"

        # Check rate limit
        allowed, count, reset_time = await self.check_rate_limit(
            key, max_requests, window_seconds
        )

        # Update metrics
        rate_limit_checks.labels(endpoint=endpoint, result='allowed' if allowed else 'rejected').inc()

        if not allowed:
            rate_limit_rejections.labels(endpoint=endpoint).inc()

            # Add rate limit headers to response
            headers = {
                'X-RateLimit-Limit': str(max_requests or self.default_max_requests),
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Reset': str(int(time.time() + reset_time))
            }

            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {int(reset_time)} seconds.",
                headers=headers
            )

        # Yield for request processing
        yield


# ============================================================================
# Factory Functions
# ============================================================================

def create_redis_rate_limiter(
    redis_url: Optional[str] = None,
    **kwargs
) -> RedisRateLimiter:
    """
    Create a Redis rate limiter with environment configuration.

    Args:
        redis_url: Override Redis URL
        **kwargs: Additional arguments for RedisRateLimiter

    Returns:
        Configured RedisRateLimiter instance
    """
    import os

    # Get Redis URL from environment or use default
    if not redis_url:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

    return RedisRateLimiter(redis_url=redis_url, **kwargs)


# ============================================================================
# Endpoint-Specific Limiters
# ============================================================================

class EndpointRateLimiter:
    """
    Manager for endpoint-specific rate limits.
    Different endpoints can have different rate limits.
    """

    def __init__(self, redis_rate_limiter: Optional[RedisRateLimiter] = None):
        """
        Initialize endpoint rate limiter.

        Args:
            redis_rate_limiter: Redis limiter instance (creates default if None)
        """
        self.redis_limiter = redis_rate_limiter or create_redis_rate_limiter()

        # Endpoint-specific configurations
        self.endpoint_configs = {
            # Vision endpoints: computationally expensive
            'vision/detect_objects': (10, 60),      # 10 req/min
            'vision/analyze_scene': (10, 60),       # 10 req/min
            'vision/track_hands': (30, 60),         # 30 req/min (lighter)
            'vision/estimate_depth': (5, 60),       # 5 req/min (heavy)
            'vision/segment_image': (5, 60),        # 5 req/min (heavy)
            'vision/estimate_pose': (10, 60),       # 10 req/min
            'vision/track_camera': (20, 60),        # 20 req/min

            # AR endpoints: moderate load
            'ar/query': (60, 60),                   # 60 req/min
            'ar/context': (120, 60),                # 120 req/min (updates)
            'ar/session': (60, 60),                 # 60 req/min

            # WebSocket: connection limits
            'ws/ar': (5, 60),                       # 5 connections/min per IP

            # Default for unknown endpoints
            'default': (60, 60),                    # 60 req/min
        }

    async def check_endpoint_limit(
        self,
        request: Request,
        endpoint: str
    ) -> None:
        """
        Check rate limit for specific endpoint.

        Args:
            request: FastAPI request
            endpoint: Endpoint path

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Get configuration for endpoint
        config = self.endpoint_configs.get(
            endpoint,
            self.endpoint_configs['default']
        )
        max_requests, window_seconds = config

        # Use context manager for automatic metrics
        async with self.redis_limiter.rate_limit_context(
            request=request,
            endpoint=endpoint,
            max_requests=max_requests,
            window_seconds=window_seconds
        ):
            pass  # Request allowed

    async def disconnect(self):
        """Cleanup resources."""
        if self.redis_limiter:
            await self.redis_limiter.disconnect()


# ============================================================================
# Global Instance
# ============================================================================

# Create global endpoint rate limiter
# This will be initialized in the FastAPI lifespan
_global_limiter: Optional[EndpointRateLimiter] = None


def get_rate_limiter() -> EndpointRateLimiter:
    """Get global rate limiter instance."""
    global _global_limiter
    if not _global_limiter:
        _global_limiter = EndpointRateLimiter()
    return _global_limiter


async def init_rate_limiter():
    """Initialize global rate limiter (called in FastAPI lifespan)."""
    global _global_limiter
    _global_limiter = EndpointRateLimiter()
    # Test connection
    await _global_limiter.redis_limiter.connect()


async def cleanup_rate_limiter():
    """Cleanup global rate limiter (called in FastAPI shutdown)."""
    global _global_limiter
    if _global_limiter:
        await _global_limiter.disconnect()
        _global_limiter = None