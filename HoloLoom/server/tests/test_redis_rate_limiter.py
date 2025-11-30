"""
Tests for Redis-based Distributed Rate Limiter

Tests both Redis and in-memory fallback implementations.

Created: 2025-11-26
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HoloLoom.server.redis_rate_limiter import (
    InMemoryRateLimiter,
    RedisRateLimiter,
    EndpointRateLimiter,
    create_redis_rate_limiter
)


# ============================================================================
# In-Memory Rate Limiter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_in_memory_basic_rate_limiting():
    """Test basic rate limiting with in-memory implementation."""
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=1)

    # First 3 requests should succeed
    for i in range(3):
        allowed, count, reset_time = await limiter.check_rate_limit("test_key")
        assert allowed is True
        assert count == i + 1

    # 4th request should be denied
    allowed, count, reset_time = await limiter.check_rate_limit("test_key")
    assert allowed is False
    assert count == 3
    assert reset_time > 0

    # Wait for window to expire
    await asyncio.sleep(1.1)

    # Should allow requests again
    allowed, count, reset_time = await limiter.check_rate_limit("test_key")
    assert allowed is True
    assert count == 1


@pytest.mark.asyncio
async def test_in_memory_sliding_window():
    """Test sliding window behavior."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=2)

    # First request
    allowed, _, _ = await limiter.check_rate_limit("test_key")
    assert allowed is True

    # Wait 1 second
    await asyncio.sleep(1)

    # Second request (still in window)
    allowed, _, _ = await limiter.check_rate_limit("test_key")
    assert allowed is True

    # Third request (should be denied)
    allowed, _, _ = await limiter.check_rate_limit("test_key")
    assert allowed is False

    # Wait for first request to expire (1 more second)
    await asyncio.sleep(1.1)

    # Should allow new request (only second request in window)
    allowed, count, _ = await limiter.check_rate_limit("test_key")
    assert allowed is True
    assert count == 2  # Second and this new request


@pytest.mark.asyncio
async def test_in_memory_cleanup():
    """Test cleanup of old entries."""
    limiter = InMemoryRateLimiter(max_requests=5, window_seconds=1)

    # Create entries for multiple keys
    for i in range(5):
        await limiter.check_rate_limit(f"key_{i}")

    assert len(limiter.requests) == 5

    # Wait for entries to become old
    await asyncio.sleep(2)

    # Cleanup entries older than 1 second
    await limiter.cleanup_old_entries(older_than_seconds=1)

    assert len(limiter.requests) == 0


# ============================================================================
# Redis Rate Limiter Tests (with Mocking)
# ============================================================================

@pytest.mark.asyncio
async def test_redis_connection_failure_with_fallback():
    """Test fallback to in-memory when Redis is unavailable."""
    limiter = RedisRateLimiter(
        redis_url="redis://invalid:6379",
        default_max_requests=3,
        enable_fallback=True
    )

    # Should fall back to in-memory
    allowed, count, _ = await limiter.check_rate_limit("test_key")
    assert allowed is True
    assert count == 1

    # Verify using in-memory limiter
    for i in range(2):
        allowed, count, _ = await limiter.check_rate_limit("test_key")
        assert allowed is True

    # Should be rate limited
    allowed, _, _ = await limiter.check_rate_limit("test_key")
    assert allowed is False


@pytest.mark.asyncio
async def test_redis_connection_failure_without_fallback():
    """Test fail-closed behavior when Redis unavailable and no fallback."""
    limiter = RedisRateLimiter(
        redis_url="redis://invalid:6379",
        default_max_requests=3,
        enable_fallback=False
    )

    # Should deny request (fail-closed)
    allowed, count, reset_time = await limiter.check_rate_limit("test_key")
    assert allowed is False
    assert count == 0


@pytest.mark.asyncio
async def test_redis_rate_limiting_with_mock():
    """Test Redis rate limiting with mocked Redis client."""
    limiter = RedisRateLimiter(
        redis_url="redis://localhost:6379",
        default_max_requests=3,
        default_window_seconds=60
    )

    # Mock Redis client
    mock_redis = AsyncMock()
    mock_pipeline = AsyncMock()

    # Setup mock responses
    mock_redis.ping.return_value = "PONG"
    mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline

    # First check: under limit
    mock_pipeline.execute.return_value = [0, 2]  # removed_count, current_count
    mock_redis.zrange.return_value = []

    with patch('redis.asyncio.from_url', return_value=mock_redis):
        # Connect and check
        connected = await limiter.connect()
        assert connected is True

        # Set the mocked redis client
        limiter._redis = mock_redis

        # Should allow request
        allowed, count, _ = await limiter._check_redis("test_key", 3, 60)
        assert allowed is True
        assert count == 3

        # Second check: at limit
        mock_pipeline.execute.return_value = [0, 3]
        mock_redis.zrange.return_value = [(b'1234567890', 1234567890.0)]

        allowed, count, _ = await limiter._check_redis("test_key", 3, 60)
        assert allowed is False
        assert count == 3

    await limiter.disconnect()


# ============================================================================
# Endpoint Rate Limiter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_endpoint_specific_limits():
    """Test different rate limits for different endpoints."""
    # Create mock Redis limiter
    mock_redis_limiter = AsyncMock(spec=RedisRateLimiter)

    limiter = EndpointRateLimiter(redis_rate_limiter=mock_redis_limiter)

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "192.168.1.1"

    # Test vision endpoint (should use 10 req/min limit)
    await limiter.check_endpoint_limit(mock_request, "vision/detect_objects")

    # Verify correct limits were used
    mock_redis_limiter.rate_limit_context.assert_called_with(
        request=mock_request,
        endpoint="vision/detect_objects",
        max_requests=10,
        window_seconds=60
    )

    # Test AR endpoint (should use 60 req/min limit)
    await limiter.check_endpoint_limit(mock_request, "ar/query")

    mock_redis_limiter.rate_limit_context.assert_called_with(
        request=mock_request,
        endpoint="ar/query",
        max_requests=60,
        window_seconds=60
    )

    # Test unknown endpoint (should use default)
    await limiter.check_endpoint_limit(mock_request, "unknown/endpoint")

    mock_redis_limiter.rate_limit_context.assert_called_with(
        request=mock_request,
        endpoint="unknown/endpoint",
        max_requests=60,
        window_seconds=60
    )


@pytest.mark.asyncio
async def test_rate_limit_context_manager():
    """Test rate limit context manager with HTTPException."""
    limiter = RedisRateLimiter(
        redis_url="redis://invalid:6379",  # Force fallback
        default_max_requests=1,
        default_window_seconds=60,
        enable_fallback=True
    )

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "192.168.1.1"

    # First request should succeed
    async with limiter.rate_limit_context(mock_request, "test_endpoint"):
        pass  # Request processing

    # Second request should raise HTTPException
    with pytest.raises(HTTPException) as exc_info:
        async with limiter.rate_limit_context(mock_request, "test_endpoint"):
            pass

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert 'X-RateLimit-Limit' in exc_info.value.headers


@pytest.mark.asyncio
async def test_create_redis_rate_limiter():
    """Test factory function with environment variables."""
    with patch.dict(os.environ, {'REDIS_URL': 'redis://custom:6380'}):
        limiter = create_redis_rate_limiter()
        assert limiter.redis_url == 'redis://custom:6380'

    # Without environment variable
    with patch.dict(os.environ, {}, clear=True):
        limiter = create_redis_rate_limiter()
        assert limiter.redis_url == 'redis://localhost:6379'

    # With override
    limiter = create_redis_rate_limiter(redis_url='redis://override:6381')
    assert limiter.redis_url == 'redis://override:6381'


# ============================================================================
# Integration Tests (requires Redis)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv('RUN_INTEGRATION_TESTS'),
    reason="Integration tests require Redis (set RUN_INTEGRATION_TESTS=1)"
)
async def test_redis_integration():
    """Integration test with real Redis (if available)."""
    limiter = RedisRateLimiter(
        redis_url="redis://localhost:6379",
        default_max_requests=3,
        default_window_seconds=2
    )

    try:
        # Test connection
        connected = await limiter.connect()
        if not connected:
            pytest.skip("Redis not available")

        # Clear any existing data
        test_key = "integration_test"

        # First 3 requests should succeed
        for i in range(3):
            allowed, count, _ = await limiter.check_rate_limit(test_key)
            assert allowed is True
            assert count == i + 1

        # 4th request should be denied
        allowed, count, reset_time = await limiter.check_rate_limit(test_key)
        assert allowed is False
        assert count == 3
        assert 0 < reset_time <= 2

        # Wait for window to expire
        await asyncio.sleep(2.1)

        # Should allow requests again
        allowed, count, _ = await limiter.check_rate_limit(test_key)
        assert allowed is True
        assert count == 1

    finally:
        await limiter.disconnect()


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test rate limiting under concurrent load."""
    limiter = InMemoryRateLimiter(max_requests=10, window_seconds=1)

    async def make_request(key: str, delay: float = 0):
        """Simulate a request with optional delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        return await limiter.check_rate_limit(key)

    # Launch 15 concurrent requests
    tasks = []
    for i in range(15):
        # Stagger requests slightly to avoid exact simultaneity
        tasks.append(make_request("concurrent_test", delay=i * 0.01))

    results = await asyncio.gather(*tasks)

    # Count allowed and denied
    allowed_count = sum(1 for allowed, _, _ in results if allowed)
    denied_count = sum(1 for allowed, _, _ in results if not allowed)

    assert allowed_count == 10  # Exactly 10 should be allowed
    assert denied_count == 5    # Exactly 5 should be denied


@pytest.mark.asyncio
async def test_multiple_keys():
    """Test that different keys have independent limits."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    # Use limit for key1
    for _ in range(2):
        allowed, _, _ = await limiter.check_rate_limit("key1")
        assert allowed is True

    # key1 should be limited
    allowed, _, _ = await limiter.check_rate_limit("key1")
    assert allowed is False

    # key2 should still have quota
    for _ in range(2):
        allowed, _, _ = await limiter.check_rate_limit("key2")
        assert allowed is True

    # Now key2 should be limited
    allowed, _, _ = await limiter.check_rate_limit("key2")
    assert allowed is False


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_performance_in_memory():
    """Test performance of in-memory rate limiter."""
    limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)

    start_time = time.perf_counter()

    # Perform 1000 checks
    for i in range(1000):
        await limiter.check_rate_limit(f"perf_test_{i % 100}")

    elapsed = time.perf_counter() - start_time

    # Should complete in under 100ms (very generous for CI)
    assert elapsed < 0.1, f"Performance test took {elapsed:.3f}s"

    # Average per check should be under 0.1ms
    avg_per_check = elapsed / 1000
    assert avg_per_check < 0.0001, f"Average per check: {avg_per_check*1000:.3f}ms"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])