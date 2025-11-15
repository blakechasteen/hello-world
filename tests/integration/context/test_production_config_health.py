"""
Tests for Production Configuration and Health Checks

Part 5: Production Hardening - Day 24

Tests:
1. Configuration profiles (dev, staging, production)
2. Configuration validation
3. Environment detection
4. Health check overall
5. Health check backends
6. Health check learning
7. Health check resources
"""

import sys
from pathlib import Path

# Add repository root to path (tests/integration/context -> repo root)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
import pytest
import os
from HoloLoom.context.production_config import (
    Environment,
    ProductionConfig,
    MonitoringConfig,
    ErrorHandlingConfig,
    CircuitBreakerConfig as ConfigCircuitBreakerConfig,
    RateLimitConfig,
    ResourceConfig,
    LearningConfig,
    create_config,
    detect_environment
)
from HoloLoom.context.health_check import (
    HealthStatus,
    ComponentCheck,
    HealthCheckResult,
    HealthChecker,
    create_health_checker
)
from HoloLoom.context.monitoring import (
    PerformanceMonitor,
    ResourceMonitor,
    LearningMetricsMonitor
)
from HoloLoom.context.circuit_breaker import (
    CircuitBreakerRegistry,
    create_circuit_breaker_registry
)


# ============================================================================
# Test 1: Configuration Profiles
# ============================================================================

def test_configuration_profiles():
    """Test 1: Configuration profiles (dev, staging, production)"""
    print("\n" + "=" * 80)
    print("Test 1: Configuration Profiles")
    print("=" * 80)

    # Development config
    dev_config = ProductionConfig.development()
    assert dev_config.environment == Environment.DEVELOPMENT
    assert dev_config.monitoring.log_level == "DEBUG"
    assert not dev_config.circuit_breaker.enabled
    assert not dev_config.rate_limit.enabled
    print(f"  [OK] Development: log_level=DEBUG, circuit_breaker=disabled, rate_limit=disabled")

    # Staging config
    staging_config = ProductionConfig.staging()
    assert staging_config.environment == Environment.STAGING
    assert staging_config.monitoring.log_level == "INFO"
    assert staging_config.circuit_breaker.enabled
    assert staging_config.rate_limit.enabled
    print(f"  [OK] Staging: log_level=INFO, circuit_breaker=enabled, rate_limit=enabled")

    # Production config
    prod_config = ProductionConfig.production()
    assert prod_config.environment == Environment.PRODUCTION
    assert prod_config.monitoring.log_level == "WARNING"
    assert prod_config.circuit_breaker.enabled
    assert prod_config.rate_limit.enabled
    assert prod_config.circuit_breaker.failure_threshold == 3  # Stricter in prod
    print(f"  [OK] Production: log_level=WARNING, failure_threshold=3 (strict)")

    # Verify different settings across environments
    assert dev_config.resource.max_memory_mb > prod_config.resource.max_memory_mb
    assert dev_config.rate_limit.global_qps > staging_config.rate_limit.global_qps
    print(f"  [OK] Resource limits differ across environments")

    print("\n[PASS] Configuration profiles working correctly")


# ============================================================================
# Test 2: Configuration Validation
# ============================================================================

def test_configuration_validation():
    """Test 2: Configuration validation"""
    print("\n" + "=" * 80)
    print("Test 2: Configuration Validation")
    print("=" * 80)

    # Valid config
    valid_config = ProductionConfig.production()
    errors = valid_config.validate()
    assert len(errors) == 0, f"Valid config should have no errors: {errors}"
    assert valid_config.is_valid()
    print(f"  [OK] Valid production config: {len(errors)} errors")

    # Invalid: negative retries
    invalid_config = ProductionConfig.development()
    invalid_config.error_handling.max_retries = -1
    errors = invalid_config.validate()
    assert len(errors) > 0
    assert any("max_retries" in error for error in errors)
    print(f"  [OK] Negative max_retries detected: {len(errors)} errors")

    # Invalid: session_qps > global_qps
    invalid_config2 = ProductionConfig.production()
    invalid_config2.rate_limit.session_qps = 2000.0
    invalid_config2.rate_limit.global_qps = 1000.0
    errors = invalid_config2.validate()
    assert len(errors) > 0
    assert any("session_qps" in error and "global_qps" in error for error in errors)
    print(f"  [OK] session_qps > global_qps detected: {len(errors)} errors")

    # Invalid: low memory limit
    invalid_config3 = ProductionConfig.production()
    invalid_config3.resource.max_memory_mb = 128
    errors = invalid_config3.validate()
    assert len(errors) > 0
    assert any("max_memory_mb" in error for error in errors)
    print(f"  [OK] Low memory limit detected: {len(errors)} errors")

    print("\n[PASS] Configuration validation working correctly")


# ============================================================================
# Test 3: Environment Detection
# ============================================================================

def test_environment_detection():
    """Test 3: Environment detection"""
    print("\n" + "=" * 80)
    print("Test 3: Environment Detection")
    print("=" * 80)

    # Test load() by name
    dev_config = ProductionConfig.load("development")
    assert dev_config.environment == Environment.DEVELOPMENT
    print(f"  [OK] load('development') -> DEVELOPMENT")

    staging_config = ProductionConfig.load("staging")
    assert staging_config.environment == Environment.STAGING
    print(f"  [OK] load('staging') -> STAGING")

    prod_config = ProductionConfig.load("production")
    assert prod_config.environment == Environment.PRODUCTION
    print(f"  [OK] load('production') -> PRODUCTION")

    # Test aliases
    dev_alias = ProductionConfig.load("dev")
    assert dev_alias.environment == Environment.DEVELOPMENT
    print(f"  [OK] load('dev') -> DEVELOPMENT (alias)")

    prod_alias = ProductionConfig.load("prod")
    assert prod_alias.environment == Environment.PRODUCTION
    print(f"  [OK] load('prod') -> PRODUCTION (alias)")

    # Test environment variable detection
    os.environ["CONTEXT_ENV"] = "production"
    detected = ProductionConfig.from_environment()
    assert detected.environment == Environment.PRODUCTION
    print(f"  [OK] from_environment() with CONTEXT_ENV=production -> PRODUCTION")

    # Cleanup
    del os.environ["CONTEXT_ENV"]

    # Test invalid environment
    try:
        ProductionConfig.load("invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()
        print(f"  [OK] load('invalid') raises ValueError")

    print("\n[PASS] Environment detection working correctly")


# ============================================================================
# Test 4: Health Check Overall
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_overall():
    """Test 4: Health check overall"""
    print("\n" + "=" * 80)
    print("Test 4: Health Check Overall")
    print("=" * 80)

    # Create monitors
    perf_monitor = PerformanceMonitor()
    resource_monitor = ResourceMonitor()

    # Record some healthy metrics
    perf_monitor.record_query(latency_ms=100.0, cache_hit=True)
    perf_monitor.record_query(latency_ms=120.0, cache_hit=False)

    # Create health checker
    checker = create_health_checker(
        performance_monitor=perf_monitor,
        resource_monitor=resource_monitor
    )

    # Check health
    result = await checker.check_health()

    assert result.healthy, "System should be healthy"
    assert result.status == HealthStatus.HEALTHY
    print(f"  [OK] Overall health: {result.status.value}")

    # Check individual components
    assert "overall" in result.checks
    assert result.checks["overall"].healthy
    print(f"  [OK] Overall component check: healthy")

    # Convert to dict (HTTP response)
    response_dict = result.to_dict()
    assert "healthy" in response_dict
    assert "status" in response_dict
    assert "checks" in response_dict
    print(f"  [OK] HTTP response format valid")

    print("\n[PASS] Health check overall working correctly")


# ============================================================================
# Test 5: Health Check Backends
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_backends():
    """Test 5: Health check backends"""
    print("\n" + "=" * 80)
    print("Test 5: Health Check Backends")
    print("=" * 80)

    # Create circuit breaker registry
    registry = create_circuit_breaker_registry()
    mcp_breaker = registry.get_or_create("mcp_backend")
    neo4j_breaker = registry.get_or_create("neo4j")

    # Create health checker
    checker = create_health_checker(circuit_breaker_registry=registry)

    # Check health (all closed)
    result = await checker.check_health()
    assert result.checks["backends"].healthy
    print(f"  [OK] All backends healthy (all breakers closed)")

    # Force one open
    mcp_breaker.force_open()

    # Check health again (one open)
    result = await checker.check_health()
    assert not result.checks["backends"].healthy
    assert result.checks["backends"].status == HealthStatus.DEGRADED
    print(f"  [OK] Backends degraded (mcp_backend open)")

    # Verify details
    details = result.checks["backends"].details
    assert "mcp_backend" in details.get("open_breakers", [])
    print(f"  [OK] Open breaker details: {details['open_breakers']}")

    print("\n[PASS] Health check backends working correctly")


# ============================================================================
# Test 6: Health Check Learning
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_learning():
    """Test 6: Health check learning"""
    print("\n" + "=" * 80)
    print("Test 6: Health Check Learning")
    print("=" * 80)

    # Create learning monitor
    learning_monitor = LearningMetricsMonitor()

    # Record good calibration
    learning_monitor.record_calibration(ece=0.05)

    # Create health checker
    checker = create_health_checker(learning_monitor=learning_monitor)

    # Check health (good calibration)
    result = await checker.check_health()
    assert result.checks["learning"].healthy
    print(f"  [OK] Learning healthy (ECE=0.05)")

    # Record poor calibration
    learning_monitor.record_calibration(ece=0.20)

    # Check health again (poor calibration)
    result = await checker.check_health()
    assert not result.checks["learning"].healthy
    assert result.checks["learning"].status == HealthStatus.DEGRADED
    print(f"  [OK] Learning degraded (ECE=0.20)")

    print("\n[PASS] Health check learning working correctly")


# ============================================================================
# Test 7: Health Check Resources
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_resources():
    """Test 7: Health check resources"""
    print("\n" + "=" * 80)
    print("Test 7: Health Check Resources")
    print("=" * 80)

    # Create resource monitor
    resource_monitor = ResourceMonitor()

    # Create health checker
    checker = create_health_checker(resource_monitor=resource_monitor)

    # Check health
    result = await checker.check_health()

    # Resources should be healthy (or N/A if psutil not available)
    assert "resources" in result.checks
    print(f"  [OK] Resources check present: {result.checks['resources'].status.value}")

    # Get quick status (non-async)
    quick_status = checker.get_quick_status()
    assert "healthy" in quick_status
    assert "checks" in quick_status
    print(f"  [OK] Quick status: {quick_status['healthy']}")

    print("\n[PASS] Health check resources working correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all production config and health check tests"""
    print("=" * 80)
    print("Production Configuration and Health Check Test Suite")
    print("=" * 80)
    print()
    print("Testing configuration profiles, validation, and health checks...")
    print()

    # Run sync tests
    test_configuration_profiles()
    test_configuration_validation()
    test_environment_detection()

    # Run async tests
    asyncio.run(test_health_check_overall())
    asyncio.run(test_health_check_backends())
    asyncio.run(test_health_check_learning())
    asyncio.run(test_health_check_resources())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Tests Passed: 7/7")
    print()
    print("[PASS] Configuration Profiles")
    print("[PASS] Configuration Validation")
    print("[PASS] Environment Detection")
    print("[PASS] Health Check Overall")
    print("[PASS] Health Check Backends")
    print("[PASS] Health Check Learning")
    print("[PASS] Health Check Resources")
    print()
    print("[SUCCESS] All production config and health check tests passed!")
    print()


if __name__ == "__main__":
    main()
