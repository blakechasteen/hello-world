"""
Tests for Rate Limiter

Part 5: Production Hardening - Day 23

Tests:
1. Token bucket (burst handling)
2. Token bucket wait (acquire_wait)
3. Sliding window (precise limiting)
4. Sliding window wait (acquire_wait)
5. Concurrent limiter (parallelism control)
6. Unified rate limiter (combined)
"""

import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
import pytest
import time
from HoloLoom.context.rate_limiter import (
    RateLimiterType,
    RateLimiterConfig,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    ConcurrentLimiter,
    RateLimiter,
    create_rate_limiter,
    create_token_bucket_limiter,
    create_sliding_window_limiter,
    create_concurrent_limiter
)
from HoloLoom.context.error_handling import RateLimitExceededError


# ============================================================================
# Test 1: Token Bucket (Burst Handling)
# ============================================================================

@pytest.mark.asyncio
async def test_token_bucket_burst():
    """Test 1: Token bucket allows bursts"""
    print("\n" + "=" * 80)
    print("Test 1: Token Bucket (Burst Handling)")
    print("=" * 80)

    limiter = create_token_bucket_limiter(rate=10.0, capacity=10)

    # Burst of 10 requests (should all succeed)
    burst_results = []
    for i in range(10):
        burst_results.append(await limiter.acquire())

    assert all(burst_results), "Burst should succeed"
    print(f"  [OK] Burst of 10 requests allowed (capacity: 10)")

    # 11th request should fail (bucket empty)
    result_11 = await limiter.acquire()
    assert not result_11, "11th request should fail"
    print(f"  [OK] 11th request rejected (bucket empty)")

    # Wait for refill (0.1s = 1 token at 10/s rate)
    await asyncio.sleep(0.15)
    result_after_wait = await limiter.acquire()
    assert result_after_wait, "Request after wait should succeed"
    print(f"  [OK] Request after 0.15s succeeds (token refilled)")

    # Statistics
    stats = limiter.get_stats()
    assert stats["total_requests"] == 12  # 10 + 1 + 1
    assert stats["total_allowed"] == 11   # 10 + 0 + 1
    assert stats["total_rejected"] == 1   # 0 + 1 + 0
    print(f"  [OK] Stats: {stats['total_requests']} requests, {stats['total_rejected']} rejected")

    print("\n[PASS] Token bucket burst handling working correctly")


# ============================================================================
# Test 2: Token Bucket Wait
# ============================================================================

@pytest.mark.asyncio
async def test_token_bucket_wait():
    """Test 2: Token bucket acquire_wait"""
    print("\n" + "=" * 80)
    print("Test 2: Token Bucket Wait")
    print("=" * 80)

    limiter = create_token_bucket_limiter(rate=10.0, capacity=5)

    # Exhaust bucket
    for i in range(5):
        await limiter.acquire()
    print(f"  [OK] Exhausted bucket (5 tokens)")

    # acquire_wait should wait and succeed
    start = time.time()
    await limiter.acquire_wait(tokens=1, timeout=1.0)
    elapsed = time.time() - start

    assert elapsed >= 0.08, f"Should wait at least 0.1s (waited {elapsed:.3f}s)"
    print(f"  [OK] acquire_wait waited {elapsed:.3f}s and succeeded")

    # acquire_wait with timeout should fail if too long
    for i in range(5):
        await limiter.acquire()  # Exhaust again

    with pytest.raises(RateLimitExceededError):
        await limiter.acquire_wait(tokens=10, timeout=0.1)  # Need 1s to refill 10 tokens
    print(f"  [OK] acquire_wait with timeout=0.1s raised RateLimitExceededError")

    print("\n[PASS] Token bucket wait working correctly")


# ============================================================================
# Test 3: Sliding Window (Precise Limiting)
# ============================================================================

@pytest.mark.asyncio
async def test_sliding_window_precise():
    """Test 3: Sliding window precise limiting"""
    print("\n" + "=" * 80)
    print("Test 3: Sliding Window (Precise Limiting)")
    print("=" * 80)

    limiter = create_sliding_window_limiter(window_size=1.0, max_requests=5)

    # First 5 requests should succeed
    for i in range(5):
        result = await limiter.acquire()
        assert result, f"Request {i+1} should succeed"
    print(f"  [OK] First 5 requests allowed (window: 1.0s, max: 5)")

    # 6th request should fail (window full)
    result_6 = await limiter.acquire()
    assert not result_6, "6th request should fail"
    print(f"  [OK] 6th request rejected (window full)")

    # Current rate
    rate = limiter.get_current_rate()
    assert 4.5 <= rate <= 5.5, f"Current rate should be ~5.0 (actual: {rate:.2f})"
    print(f"  [OK] Current rate: {rate:.2f} requests/second")

    # Wait for window to slide
    await asyncio.sleep(1.1)

    # New request should succeed (old requests expired)
    result_after_slide = await limiter.acquire()
    assert result_after_slide, "Request after window slide should succeed"
    print(f"  [OK] Request after window slide succeeds")

    # Statistics
    stats = limiter.get_stats()
    assert stats["total_requests"] == 7  # 5 + 1 + 1
    assert stats["total_rejected"] == 1  # Just the 6th request
    print(f"  [OK] Stats: {stats['total_requests']} requests, {stats['total_rejected']} rejected")

    print("\n[PASS] Sliding window precise limiting working correctly")


# ============================================================================
# Test 4: Sliding Window Wait
# ============================================================================

@pytest.mark.asyncio
async def test_sliding_window_wait():
    """Test 4: Sliding window acquire_wait"""
    print("\n" + "=" * 80)
    print("Test 4: Sliding Window Wait")
    print("=" * 80)

    limiter = create_sliding_window_limiter(window_size=0.5, max_requests=3)

    # Exhaust window
    for i in range(3):
        await limiter.acquire()
    print(f"  [OK] Exhausted window (3 requests)")

    # acquire_wait should wait and succeed
    start = time.time()
    await limiter.acquire_wait(timeout=1.0)
    elapsed = time.time() - start

    assert elapsed >= 0.4, f"Should wait at least 0.5s (waited {elapsed:.3f}s)"
    print(f"  [OK] acquire_wait waited {elapsed:.3f}s and succeeded")

    print("\n[PASS] Sliding window wait working correctly")


# ============================================================================
# Test 5: Concurrent Limiter
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_limiter():
    """Test 5: Concurrent limiter (parallelism control)"""
    print("\n" + "=" * 80)
    print("Test 5: Concurrent Limiter")
    print("=" * 80)

    limiter = create_concurrent_limiter(max_concurrent=3)

    # Track concurrent operations
    concurrent_count = []

    async def operation(duration: float):
        async with limiter:
            concurrent_count.append(limiter.current_concurrent)
            await asyncio.sleep(duration)

    # Launch 5 operations concurrently
    tasks = [operation(0.1) for _ in range(5)]
    await asyncio.gather(*tasks)

    # Max concurrent should be 3
    max_concurrent = max(concurrent_count)
    assert max_concurrent == 3, f"Max concurrent should be 3 (actual: {max_concurrent})"
    print(f"  [OK] Max concurrent: {max_concurrent} (limit: 3)")

    # Statistics
    stats = limiter.get_stats()
    assert stats["peak_concurrent"] == 3
    assert stats["total_acquires"] == 5
    print(f"  [OK] Stats: {stats['total_acquires']} acquires, peak: {stats['peak_concurrent']}")

    # Current should be 0 (all released)
    assert limiter.current_concurrent == 0
    print(f"  [OK] All released: {limiter.current_concurrent} current")

    print("\n[PASS] Concurrent limiter working correctly")


# ============================================================================
# Test 6: Unified Rate Limiter
# ============================================================================

@pytest.mark.asyncio
async def test_unified_rate_limiter():
    """Test 6: Unified rate limiter (combined)"""
    print("\n" + "=" * 80)
    print("Test 6: Unified Rate Limiter")
    print("=" * 80)

    limiter = create_rate_limiter(
        rate=20.0,        # 20 tokens/second
        capacity=10,      # Burst of 10
        window_size=1.0,  # 1 second window
        max_requests=15,  # Max 15 per second
        max_concurrent=5  # Max 5 concurrent
    )

    # Test burst handling (should allow 10)
    burst_count = 0
    for i in range(15):
        if await limiter.acquire():
            burst_count += 1
        else:
            break

    assert burst_count == 10, f"Should allow burst of 10 (actual: {burst_count})"
    print(f"  [OK] Burst: {burst_count} allowed (capacity: 10)")

    # Wait for some tokens to refill before concurrent test
    await asyncio.sleep(0.3)  # Refill ~6 tokens at 20/s

    # Test context manager with concurrency limit
    async def concurrent_operation():
        async with limiter:
            await asyncio.sleep(0.05)

    # This should work (respects concurrency limit)
    tasks = [concurrent_operation() for _ in range(3)]
    await asyncio.gather(*tasks)
    print(f"  [OK] Context manager with concurrency limit works")

    # Statistics
    stats = limiter.get_stats()
    assert "token_bucket" in stats
    assert "sliding_window" in stats
    assert "concurrent" in stats
    print(f"  [OK] Combined stats available for all components")

    # Check individual components
    tb_stats = stats["token_bucket"]
    sw_stats = stats["sliding_window"]
    cc_stats = stats["concurrent"]

    print(f"  Token Bucket: {tb_stats['total_requests']} requests, {tb_stats['total_rejected']} rejected")
    print(f"  Sliding Window: {sw_stats['total_requests']} requests, {sw_stats['total_rejected']} rejected")
    print(f"  Concurrent: {cc_stats['total_acquires']} acquires, peak: {cc_stats['peak_concurrent']}")

    print("\n[PASS] Unified rate limiter working correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all rate limiter tests"""
    print("=" * 80)
    print("Rate Limiter Test Suite")
    print("=" * 80)
    print()
    print("Testing token bucket, sliding window, and concurrent limiters...")
    print()

    # Run async tests
    asyncio.run(test_token_bucket_burst())
    asyncio.run(test_token_bucket_wait())
    asyncio.run(test_sliding_window_precise())
    asyncio.run(test_sliding_window_wait())
    asyncio.run(test_concurrent_limiter())
    asyncio.run(test_unified_rate_limiter())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Tests Passed: 6/6")
    print()
    print("[PASS] Token Bucket (Burst Handling)")
    print("[PASS] Token Bucket Wait")
    print("[PASS] Sliding Window (Precise Limiting)")
    print("[PASS] Sliding Window Wait")
    print("[PASS] Concurrent Limiter")
    print("[PASS] Unified Rate Limiter")
    print()
    print("[SUCCESS] All rate limiter tests passed!")
    print()


if __name__ == "__main__":
    main()
