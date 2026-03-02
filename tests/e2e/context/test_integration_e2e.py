"""
End-to-End Integration Tests

Part 5: Production Hardening - Day 25

Tests all production hardening features working together:
1. Complete production scenario (happy path)
2. Error recovery with retry and fallback
3. Circuit breaker protection
4. Rate limiting enforcement
5. Health degradation and recovery
"""

import sys
from pathlib import Path

# Add repository root to path (tests/e2e/context -> repo root)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
import pytest
import time
from hololoom.context import (
    # Configuration
    ProductionConfig,
    # Error handling
    create_error_handler,
    BackendError,
    RateLimitExceededError,
    # Monitoring
    create_system_monitor,
    # Circuit breakers
    create_circuit_breaker_registry,
    CircuitState,
    # Rate limiting
    create_rate_limiter,
    # Health checks
    create_health_checker,
    HealthStatus
)


# ============================================================================
# Test 1: Complete Production Scenario (Happy Path)
# ============================================================================

@pytest.mark.asyncio
async def test_complete_production_scenario():
    """Test 1: Complete production scenario (happy path)"""
    print("\n" + "=" * 80)
    print("Test 1: Complete Production Scenario")
    print("=" * 80)

    # Load production configuration
    config = ProductionConfig.production()
    print(f"  [OK] Loaded production configuration")

    # Create production components
    monitor = create_system_monitor()
    breaker_registry = create_circuit_breaker_registry()
    rate_limiter = create_rate_limiter(
        rate=100.0,
        capacity=200,
        max_concurrent=50
    )
    error_handler = create_error_handler()
    health_checker = create_health_checker(
        performance_monitor=monitor.performance,
        resource_monitor=monitor.resources,
        learning_monitor=monitor.learning,
        circuit_breaker_registry=breaker_registry
    )
    print(f"  [OK] Created all production components")

    # Simulate successful queries
    async def mock_query(query_text: str):
        """Mock query that succeeds"""
        await asyncio.sleep(0.01)  # Simulate work
        return {"result": f"Success: {query_text}", "confidence": 0.95}

    backend = "mock_backend"
    breaker = breaker_registry.get_or_create(backend)

    for i in range(10):
        # Rate limiting
        assert await rate_limiter.acquire(), f"Query {i+1} should pass rate limit"

        # Circuit breaker protection
        start_time = time.time()
        result = await breaker.call(mock_query, f"query_{i+1}")

        # Monitor
        latency = (time.time() - start_time) * 1000
        monitor.performance.record_query(
            latency_ms=latency,
            cache_hit=(i % 2 == 0)  # Simulate 50% cache hit
        )

        assert result["confidence"] == 0.95
        print(f"  [OK] Query {i+1}/10: latency={latency:.1f}ms, confidence={result['confidence']:.2f}")

    # Check health
    health = await health_checker.check_health()
    assert health.healthy, "System should be healthy"
    assert health.status == HealthStatus.HEALTHY
    print(f"  [OK] System health: {health.status.value}")

    # Check metrics
    metrics = monitor.performance.get_metrics()
    assert metrics["query_count"] == 10
    assert metrics["error_rate"] == 0.0
    assert metrics["cache_hit_rate"] == 0.5
    print(f"  [OK] Metrics: {metrics['query_count']} queries, {metrics['error_rate']*100:.1f}% errors, {metrics['cache_hit_rate']*100:.1f}% cache hit")

    print("\n[PASS] Complete production scenario working correctly")


# ============================================================================
# Test 2: Error Recovery with Retry and Fallback
# ============================================================================

@pytest.mark.asyncio
async def test_error_recovery():
    """Test 2: Error recovery with retry and fallback"""
    print("\n" + "=" * 80)
    print("Test 2: Error Recovery with Retry and Fallback")
    print("=" * 80)

    monitor = create_system_monitor()
    error_handler = create_error_handler()

    # Mock query that fails twice then succeeds
    call_count = 0

    async def flaky_query():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise BackendError("Transient failure")
        return {"result": "Success after retry"}

    # Retry with error handler
    from hololoom.context.error_handling import retry

    @retry(max_attempts=3, initial_delay=0.01, retry_on=(BackendError,))
    async def query_with_retry():
        return await flaky_query()

    # Should succeed after retries
    result = await query_with_retry()
    assert result["result"] == "Success after retry"
    assert call_count == 3
    print(f"  [OK] Query succeeded after {call_count} attempts")

    # Test fallback
    async def always_fails():
        raise BackendError("Permanent failure")

    async def fallback_result():
        return {"result": "Fallback result", "confidence": 0.75}

    # Should use fallback
    result = await error_handler.handle(
        BackendError("Permanent failure"),
        context="test_query",
        fallback=fallback_result
    )
    assert result["result"] == "Fallback result"
    print(f"  [OK] Fallback executed: {result['result']}")

    print("\n[PASS] Error recovery working correctly")


# ============================================================================
# Test 3: Circuit Breaker Protection
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_protection():
    """Test 3: Circuit breaker protection"""
    print("\n" + "=" * 80)
    print("Test 3: Circuit Breaker Protection")
    print("=" * 80)

    monitor = create_system_monitor()
    breaker_registry = create_circuit_breaker_registry()
    health_checker = create_health_checker(
        performance_monitor=monitor.performance,
        circuit_breaker_registry=breaker_registry
    )

    # Create backend breaker
    backend = "failing_backend"
    breaker = breaker_registry.get_or_create(backend)

    # Configure for quick testing
    breaker.config.failure_threshold = 3
    breaker.config.recovery_timeout = 0.5

    # Simulate failures
    async def failing_query():
        raise BackendError("Backend down")

    # Cause 3 failures to open circuit
    for i in range(3):
        try:
            await breaker.call(failing_query)
        except BackendError:
            print(f"  [OK] Failure {i+1}/3 recorded")

    # Circuit should be open
    assert breaker.state == CircuitState.OPEN
    print(f"  [OK] Circuit opened after {breaker.config.failure_threshold} failures")

    # Check health (should be degraded)
    health = await health_checker.check_health()
    assert not health.checks["backends"].healthy
    assert health.checks["backends"].status == HealthStatus.DEGRADED
    print(f"  [OK] Health degraded: {health.checks['backends'].message}")

    # Wait for recovery timeout
    await asyncio.sleep(0.6)

    # Simulate recovery (working backend)
    async def working_query():
        return {"result": "Success"}

    # Should enter HALF_OPEN
    result = await breaker.call(working_query)
    assert breaker.state == CircuitState.HALF_OPEN
    print(f"  [OK] Circuit in HALF_OPEN state")

    # Another success should close circuit
    result = await breaker.call(working_query)
    assert breaker.state == CircuitState.CLOSED
    print(f"  [OK] Circuit closed after recovery")

    # Health should be restored
    health = await health_checker.check_health()
    assert health.checks["backends"].healthy
    print(f"  [OK] Health restored: {health.checks['backends'].message}")

    print("\n[PASS] Circuit breaker protection working correctly")


# ============================================================================
# Test 4: Rate Limiting Enforcement
# ============================================================================

@pytest.mark.asyncio
async def test_rate_limiting_enforcement():
    """Test 4: Rate limiting enforcement"""
    print("\n" + "=" * 80)
    print("Test 4: Rate Limiting Enforcement")
    print("=" * 80)

    monitor = create_system_monitor()

    # Create rate limiter with strict limits
    rate_limiter = create_rate_limiter(
        rate=10.0,      # 10 QPS
        capacity=10,    # Burst of 10
        max_requests=20,  # Allow more than burst in sliding window
        max_concurrent=3
    )

    # Burst of 10 should succeed
    burst_count = 0
    for i in range(10):
        if await rate_limiter.acquire():
            burst_count += 1

    assert burst_count == 10
    print(f"  [OK] Burst of {burst_count} allowed (capacity: 10)")

    # 11th should fail
    assert not await rate_limiter.acquire()
    print(f"  [OK] 11th request rejected (bucket empty)")

    # Concurrent limit
    async def concurrent_work():
        async with rate_limiter:
            await asyncio.sleep(0.05)

    # Launch 5 concurrent (only 3 should run at once)
    concurrent_count = []

    async def tracked_work():
        # Use token_bucket.acquire_wait to wait for tokens instead of immediate fail
        await rate_limiter.token_bucket.acquire_wait(timeout=2.0)
        # Also check sliding window
        if not await rate_limiter.sliding_window.acquire():
            raise RateLimitExceededError("Sliding window limit exceeded")
        # Acquire concurrent limit
        await rate_limiter.concurrent.acquire()
        try:
            concurrent_count.append(rate_limiter.concurrent.current_concurrent)
            await asyncio.sleep(0.05)
        finally:
            rate_limiter.concurrent.release()

    tasks = [tracked_work() for _ in range(5)]
    await asyncio.gather(*tasks)

    max_concurrent = max(concurrent_count)
    assert max_concurrent <= 3
    print(f"  [OK] Max concurrent: {max_concurrent} (limit: 3)")

    # Stats
    stats = rate_limiter.get_stats()
    tb_stats = stats["token_bucket"]
    print(f"  [OK] Rate limiter stats: {tb_stats['total_requests']} requests, {tb_stats['total_rejected']} rejected")

    print("\n[PASS] Rate limiting enforcement working correctly")


# ============================================================================
# Test 5: Health Degradation and Recovery
# ============================================================================

@pytest.mark.asyncio
async def test_health_degradation_recovery():
    """Test 5: Health degradation and recovery"""
    print("\n" + "=" * 80)
    print("Test 5: Health Degradation and Recovery")
    print("=" * 80)

    monitor = create_system_monitor()
    health_checker = create_health_checker(
        performance_monitor=monitor.performance,
        learning_monitor=monitor.learning
    )

    # Initial health (should be healthy)
    health = await health_checker.check_health()
    assert health.healthy
    print(f"  [OK] Initial health: {health.status.value}")

    # Simulate high error rate (>10%)
    for i in range(10):
        if i < 2:
            # 2 errors
            monitor.performance.record_query(
                latency_ms=100.0,
                error="BackendError"
            )
        else:
            # 8 successes
            monitor.performance.record_query(latency_ms=100.0)

    # Error rate = 2/10 = 20% (>10% threshold)
    health = await health_checker.check_health()
    assert not health.checks["overall"].healthy
    assert health.checks["overall"].status == HealthStatus.UNHEALTHY
    print(f"  [OK] Health degraded due to high error rate: {health.checks['overall'].message}")

    # Simulate recovery (no errors)
    for i in range(50):
        monitor.performance.record_query(latency_ms=100.0)

    # Error rate should drop to 2/60 = 3.3% (<10% threshold)
    health = await health_checker.check_health()
    assert health.checks["overall"].healthy
    print(f"  [OK] Health recovered: {health.checks['overall'].message}")

    # Simulate poor calibration
    for i in range(5):
        monitor.learning.record_calibration(ece=0.20)  # Poor calibration (>0.15)

    health = await health_checker.check_health()
    assert not health.checks["learning"].healthy
    assert health.checks["learning"].status == HealthStatus.DEGRADED
    print(f"  [OK] Learning degraded: {health.checks['learning'].message}")

    # Simulate calibration improvement
    for i in range(5):
        monitor.learning.record_calibration(ece=0.08)  # Good calibration (<0.15)

    health = await health_checker.check_health()
    assert health.checks["learning"].healthy
    print(f"  [OK] Learning recovered: {health.checks['learning'].message}")

    print("\n[PASS] Health degradation and recovery working correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all end-to-end integration tests"""
    print("=" * 80)
    print("End-to-End Integration Test Suite")
    print("=" * 80)
    print()
    print("Testing all production features working together...")
    print()

    # Run async tests
    asyncio.run(test_complete_production_scenario())
    asyncio.run(test_error_recovery())
    asyncio.run(test_circuit_breaker_protection())
    asyncio.run(test_rate_limiting_enforcement())
    asyncio.run(test_health_degradation_recovery())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Tests Passed: 5/5")
    print()
    print("[PASS] Complete Production Scenario")
    print("[PASS] Error Recovery with Retry and Fallback")
    print("[PASS] Circuit Breaker Protection")
    print("[PASS] Rate Limiting Enforcement")
    print("[PASS] Health Degradation and Recovery")
    print()
    print("[SUCCESS] All end-to-end integration tests passed!")
    print()


if __name__ == "__main__":
    main()
