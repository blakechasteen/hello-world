"""
Tests for Monitoring and Circuit Breaker

Part 5: Production Hardening - Day 22

Tests:
1. Performance monitoring (QPS, latency, errors)
2. Resource monitoring (memory, CPU)
3. Learning metrics monitoring
4. Prometheus metrics export
5. Circuit breaker state transitions
6. Circuit breaker timeout handling
7. Circuit breaker registry
8. System monitor integration
"""

import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
import pytest
import time
from HoloLoom.context.monitoring import (
    LatencyHistogram,
    PerformanceMonitor,
    ResourceMonitor,
    LearningMetricsMonitor,
    SystemMonitor,
    create_system_monitor
)
from HoloLoom.context.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerRegistry,
    create_circuit_breaker,
    create_circuit_breaker_registry
)
from HoloLoom.context.error_handling import CircuitBreakerOpenError


# ============================================================================
# Test 1: Performance Monitoring
# ============================================================================

def test_performance_monitoring():
    """Test 1: Performance monitoring (QPS, latency, errors)"""
    print("\n" + "=" * 80)
    print("Test 1: Performance Monitoring")
    print("=" * 80)

    monitor = PerformanceMonitor()

    # Record some queries
    monitor.record_query(latency_ms=50.0, cache_hit=True)
    monitor.record_query(latency_ms=150.0, cache_hit=False)
    monitor.record_query(latency_ms=100.0, error="BackendError", cache_hit=False)
    monitor.record_query(latency_ms=75.0, cache_hit=True, fallback_used=True)

    metrics = monitor.get_metrics()

    # Query count
    assert metrics["query_count"] == 4
    print(f"  [OK] Query count: {metrics['query_count']}")

    # QPS (should be >0 since queries just happened)
    assert metrics["qps"] > 0
    print(f"  [OK] QPS: {metrics['qps']:.2f}")

    # Latency percentiles
    assert 50 <= metrics["latency_p50"] <= 150
    print(f"  [OK] Latency p50: {metrics['latency_p50']:.1f}ms")

    # Error rate
    assert metrics["error_rate"] == 0.25  # 1/4
    print(f"  [OK] Error rate: {metrics['error_rate']*100:.1f}%")

    # Cache hit rate
    assert metrics["cache_hit_rate"] == 0.5  # 2/4
    print(f"  [OK] Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")

    # Fallback rate
    assert metrics["fallback_rate"] == 0.25  # 1/4
    print(f"  [OK] Fallback rate: {metrics['fallback_rate']*100:.1f}%")

    # Reset
    monitor.reset()
    assert monitor.query_count == 0
    print(f"  [OK] Reset: {monitor.query_count} queries")

    print("\n[PASS] Performance monitoring working correctly")


# ============================================================================
# Test 2: Resource Monitoring
# ============================================================================

def test_resource_monitoring():
    """Test 2: Resource monitoring (memory, CPU)"""
    print("\n" + "=" * 80)
    print("Test 2: Resource Monitoring")
    print("=" * 80)

    monitor = ResourceMonitor()

    metrics = monitor.get_metrics()

    # Memory usage (should be >=0, may be 0 if psutil not available)
    assert metrics["memory_mb"] >= 0
    if metrics["memory_mb"] > 0:
        print(f"  [OK] Memory usage: {metrics['memory_mb']:.1f} MB")
    else:
        print(f"  [OK] Memory usage: N/A (psutil not available)")

    # CPU usage (should be >=0)
    assert metrics["cpu_percent"] >= 0
    if metrics["cpu_percent"] > 0:
        print(f"  [OK] CPU usage: {metrics['cpu_percent']:.1f}%")
    else:
        print(f"  [OK] CPU usage: N/A (psutil not available)")

    print("\n[PASS] Resource monitoring working correctly")


# ============================================================================
# Test 3: Learning Metrics Monitoring
# ============================================================================

def test_learning_metrics_monitoring():
    """Test 3: Learning metrics monitoring"""
    print("\n" + "=" * 80)
    print("Test 3: Learning Metrics Monitoring")
    print("=" * 80)

    monitor = LearningMetricsMonitor()

    # Record calibration
    monitor.record_calibration(ece=0.05)
    monitor.record_calibration(ece=0.08)

    # Record strategy update
    monitor.record_strategy_update({"sql": 0.1, "neo4j": -0.05})
    monitor.record_strategy_update({"qdrant": 0.15})

    metrics = monitor.get_metrics()

    # Current ECE
    assert metrics["calibration_ece"] == 0.08  # Most recent
    print(f"  [OK] Current ECE: {metrics['calibration_ece']:.3f}")

    # Calibration observations
    assert metrics["calibration_observations"] == 2
    print(f"  [OK] Calibration observations: {metrics['calibration_observations']}")

    # Strategy updates
    assert metrics["strategy_update_count"] == 2
    print(f"  [OK] Strategy updates: {metrics['strategy_update_count']}")

    # Mean weight adjustment
    assert metrics["mean_weight_adjustment"] > 0
    print(f"  [OK] Mean weight adjustment: {metrics['mean_weight_adjustment']:.3f}")

    print("\n[PASS] Learning metrics monitoring working correctly")


# ============================================================================
# Test 4: Prometheus Metrics Export
# ============================================================================

def test_prometheus_metrics_export():
    """Test 4: Prometheus metrics export"""
    print("\n" + "=" * 80)
    print("Test 4: Prometheus Metrics Export")
    print("=" * 80)

    monitor = create_system_monitor()

    # Record some activity
    monitor.performance.record_query(latency_ms=100.0)
    monitor.learning.record_calibration(ece=0.07)

    # Export Prometheus metrics
    prometheus_output = monitor.get_prometheus_metrics()

    # Verify format
    assert "# HELP" in prometheus_output
    assert "# TYPE" in prometheus_output
    assert "context_queries_total" in prometheus_output
    assert "context_qps" in prometheus_output
    assert "context_latency_seconds" in prometheus_output
    print(f"  [OK] Prometheus format valid")

    # Verify values
    assert "context_queries_total 1" in prometheus_output
    print(f"  [OK] Query count exported")

    # Print sample (first 5 lines)
    lines = prometheus_output.split("\n")[:5]
    for line in lines:
        if line:
            print(f"  Sample: {line}")

    print("\n[PASS] Prometheus metrics export working correctly")


# ============================================================================
# Test 5: Circuit Breaker State Transitions
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions():
    """Test 5: Circuit breaker state transitions"""
    print("\n" + "=" * 80)
    print("Test 5: Circuit Breaker State Transitions")
    print("=" * 80)

    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=0.5,  # Short timeout for testing
        success_threshold=2,
        timeout=1.0
    )
    breaker = CircuitBreaker(config)

    # Initial state: CLOSED
    assert breaker.state == CircuitState.CLOSED
    print(f"  [OK] Initial state: {breaker.state.value}")

    # Successful calls stay CLOSED
    async def succeeds():
        return "success"

    result = await breaker.call(succeeds)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    print(f"  [OK] After success: {breaker.state.value}")

    # Failures -> OPEN
    async def fails():
        raise RuntimeError("Simulated failure")

    for i in range(3):
        try:
            await breaker.call(fails)
        except RuntimeError:
            pass

    assert breaker.state == CircuitState.OPEN
    print(f"  [OK] After 3 failures: {breaker.state.value}")

    # OPEN -> fail fast
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeeds)
    print(f"  [OK] OPEN state rejects calls")

    # Wait for recovery timeout
    await asyncio.sleep(0.6)

    # Next call transitions to HALF_OPEN
    result = await breaker.call(succeeds)
    assert result == "success"
    assert breaker.state == CircuitState.HALF_OPEN
    print(f"  [OK] After timeout: {breaker.state.value}")

    # Another success -> CLOSED
    result = await breaker.call(succeeds)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    print(f"  [OK] After 2 successes in HALF_OPEN: {breaker.state.value}")

    print("\n[PASS] Circuit breaker state transitions working correctly")


# ============================================================================
# Test 6: Circuit Breaker Timeout Handling
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_timeout():
    """Test 6: Circuit breaker timeout handling"""
    print("\n" + "=" * 80)
    print("Test 6: Circuit Breaker Timeout Handling")
    print("=" * 80)

    config = CircuitBreakerConfig(
        failure_threshold=2,
        timeout=0.1  # 100ms timeout
    )
    breaker = CircuitBreaker(config)

    # Slow operation should timeout
    async def slow_operation():
        await asyncio.sleep(0.5)
        return "too slow"

    # First timeout
    with pytest.raises(asyncio.TimeoutError):
        await breaker.call(slow_operation)
    print(f"  [OK] First timeout detected")

    # Second timeout -> OPEN
    with pytest.raises(asyncio.TimeoutError):
        await breaker.call(slow_operation)
    print(f"  [OK] Second timeout detected")

    assert breaker.state == CircuitState.OPEN
    print(f"  [OK] Circuit opened after timeouts: {breaker.state.value}")

    # Stats
    stats = breaker.get_stats()
    assert stats["total_timeouts"] == 2
    print(f"  [OK] Timeout count: {stats['total_timeouts']}")

    print("\n[PASS] Circuit breaker timeout handling working correctly")


# ============================================================================
# Test 7: Circuit Breaker Registry
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_registry():
    """Test 7: Circuit breaker registry"""
    print("\n" + "=" * 80)
    print("Test 7: Circuit Breaker Registry")
    print("=" * 80)

    registry = create_circuit_breaker_registry()

    # Create breakers for different backends
    mcp_breaker = registry.get_or_create("mcp_backend")
    neo4j_breaker = registry.get_or_create("neo4j")
    qdrant_breaker = registry.get_or_create("qdrant")

    assert mcp_breaker is not neo4j_breaker
    assert len(registry.breakers) == 3
    print(f"  [OK] Created 3 breakers: {list(registry.breakers.keys())}")

    # Get existing breaker
    mcp_breaker_2 = registry.get_or_create("mcp_backend")
    assert mcp_breaker is mcp_breaker_2
    print(f"  [OK] Get existing breaker returns same instance")

    # Health summary (all closed initially)
    health = registry.get_health_summary()
    assert health["healthy"]
    assert health["all_closed"]
    print(f"  [OK] All breakers closed: {health['all_closed']}")

    # Force one open
    mcp_breaker.force_open()
    health = registry.get_health_summary()
    assert not health["healthy"]
    assert "mcp_backend" in health["open_breakers"]
    print(f"  [OK] Health summary detects open breaker: {health['open_breakers']}")

    # Reset all
    registry.reset_all()
    health = registry.get_health_summary()
    assert health["healthy"]
    print(f"  [OK] Reset all breakers: {health['all_closed']}")

    print("\n[PASS] Circuit breaker registry working correctly")


# ============================================================================
# Test 8: System Monitor Integration
# ============================================================================

def test_system_monitor_integration():
    """Test 8: System monitor integration"""
    print("\n" + "=" * 80)
    print("Test 8: System Monitor Integration")
    print("=" * 80)

    monitor = create_system_monitor()

    # Record activity
    monitor.performance.record_query(latency_ms=120.0, cache_hit=True)
    monitor.performance.record_query(latency_ms=90.0, error="TimeoutError")
    monitor.learning.record_calibration(ece=0.06)
    monitor.learning.record_strategy_update({"sql": 0.1})

    # Get all metrics
    all_metrics = monitor.get_all_metrics()

    assert "performance" in all_metrics
    assert "resources" in all_metrics
    assert "learning" in all_metrics
    assert "timestamp" in all_metrics
    print(f"  [OK] All metric categories present")

    # Verify performance metrics
    perf = all_metrics["performance"]
    assert perf["query_count"] == 2
    assert perf["error_rate"] == 0.5
    print(f"  [OK] Performance metrics: {perf['query_count']} queries, {perf['error_rate']*100:.0f}% errors")

    # Verify learning metrics
    learn = all_metrics["learning"]
    assert learn["calibration_ece"] == 0.06
    assert learn["strategy_update_count"] == 1
    print(f"  [OK] Learning metrics: ECE={learn['calibration_ece']:.2f}, updates={learn['strategy_update_count']}")

    # Get summary
    summary = monitor.get_summary()
    assert "Performance:" in summary
    assert "Resources:" in summary
    assert "Learning:" in summary
    print(f"  [OK] Summary generated ({len(summary)} chars)")

    print("\n[PASS] System monitor integration working correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all monitoring and circuit breaker tests"""
    print("=" * 80)
    print("Monitoring and Circuit Breaker Test Suite")
    print("=" * 80)
    print()
    print("Testing monitoring, resource tracking, and circuit breakers...")
    print()

    # Run sync tests
    test_performance_monitoring()
    test_resource_monitoring()
    test_learning_metrics_monitoring()
    test_prometheus_metrics_export()
    test_system_monitor_integration()

    # Run async tests
    asyncio.run(test_circuit_breaker_state_transitions())
    asyncio.run(test_circuit_breaker_timeout())
    asyncio.run(test_circuit_breaker_registry())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Tests Passed: 8/8")
    print()
    print("[PASS] Performance Monitoring")
    print("[PASS] Resource Monitoring")
    print("[PASS] Learning Metrics Monitoring")
    print("[PASS] Prometheus Metrics Export")
    print("[PASS] Circuit Breaker State Transitions")
    print("[PASS] Circuit Breaker Timeout Handling")
    print("[PASS] Circuit Breaker Registry")
    print("[PASS] System Monitor Integration")
    print()
    print("[SUCCESS] All monitoring and circuit breaker tests passed!")
    print()


if __name__ == "__main__":
    main()
