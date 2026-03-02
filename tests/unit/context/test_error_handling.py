"""
Tests for Error Handling

Part 5: Production Hardening - Day 21

Tests:
1. Exception hierarchy
2. Error categorization
3. Retry decorator with exponential backoff
4. Fallback strategy cascade
5. Error handler integration
"""

import sys
from pathlib import Path

# Add repository root to path (tests/unit/context -> repo root)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio
import pytest
import time
from hololoom.context.error_handling import (
    # Exceptions
    ContextError,
    RoutingError,
    BackendError,
    CalibrationError,
    LearningError,
    RateLimitExceededError,
    CircuitBreakerOpenError,
    # Categorization
    ErrorCategory,
    categorize_error,
    is_retryable,
    should_fallback,
    # Retry
    RetryConfig,
    retry,
    # Fallback
    FallbackStrategy,
    # Handler
    ErrorHandler,
    create_error_handler
)


# ============================================================================
# Test 1: Exception Hierarchy
# ============================================================================

def test_exception_hierarchy():
    """Test 1: Exception hierarchy is correct"""
    print("\n" + "=" * 80)
    print("Test 1: Exception Hierarchy")
    print("=" * 80)

    # All context errors inherit from ContextError
    assert issubclass(RoutingError, ContextError)
    assert issubclass(BackendError, ContextError)
    assert issubclass(CalibrationError, ContextError)
    assert issubclass(LearningError, ContextError)
    assert issubclass(RateLimitExceededError, ContextError)
    assert issubclass(CircuitBreakerOpenError, ContextError)

    # Can catch specific errors
    try:
        raise BackendError("Backend failed")
    except BackendError as e:
        print(f"  [OK] Caught BackendError: {e}")
        caught_backend = True
    assert caught_backend

    # Can catch all context errors
    try:
        raise RoutingError("Routing failed")
    except ContextError as e:
        print(f"  [OK] Caught RoutingError as ContextError: {e}")
        caught_context = True
    assert caught_context

    print("\n[PASS] Exception hierarchy working correctly")


# ============================================================================
# Test 2: Error Categorization
# ============================================================================

def test_error_categorization():
    """Test 2: Error categorization for handling decisions"""
    print("\n" + "=" * 80)
    print("Test 2: Error Categorization")
    print("=" * 80)

    test_cases = [
        (BackendError("DB down"), ErrorCategory.TRANSIENT, True, True),
        (RateLimitExceededError("Too many requests"), ErrorCategory.RATE_LIMIT, False, False),
        (CircuitBreakerOpenError("Circuit open"), ErrorCategory.CIRCUIT_OPEN, False, False),
        (asyncio.TimeoutError("Timeout"), ErrorCategory.TIMEOUT, True, True),
        (CalibrationError("Invalid ECE"), ErrorCategory.PERMANENT, False, True),
        (LearningError("Learning failed"), ErrorCategory.PERMANENT, False, True),
    ]

    results = []
    for error, expected_category, expected_retryable, expected_fallback in test_cases:
        category = categorize_error(error)
        retryable = is_retryable(error)
        fallback = should_fallback(error)

        match_category = category == expected_category
        match_retryable = retryable == expected_retryable
        match_fallback = fallback == expected_fallback

        status = "[OK]" if (match_category and match_retryable and match_fallback) else "[X]"
        print(f"  {status} {type(error).__name__}: {category.value}, retryable={retryable}, fallback={fallback}")

        results.append(match_category and match_retryable and match_fallback)

    assert all(results), "Some categorizations failed"
    print("\n[PASS] Error categorization working correctly")


# ============================================================================
# Test 3: Retry Decorator
# ============================================================================

@pytest.mark.asyncio
async def test_retry_decorator():
    """Test 3: Retry decorator with exponential backoff"""
    print("\n" + "=" * 80)
    print("Test 3: Retry Decorator")
    print("=" * 80)

    # Test Case 1: Succeeds on first attempt
    call_count_success = 0

    @retry(max_attempts=3, initial_delay=0.01, retry_on=(BackendError,))
    async def succeeds_immediately():
        nonlocal call_count_success
        call_count_success += 1
        return "success"

    result = await succeeds_immediately()
    assert result == "success"
    assert call_count_success == 1
    print(f"  [OK] Succeeds immediately: 1 attempt")

    # Test Case 2: Succeeds on retry
    call_count_retry = 0

    @retry(max_attempts=3, initial_delay=0.01, retry_on=(BackendError,))
    async def succeeds_on_retry():
        nonlocal call_count_retry
        call_count_retry += 1
        if call_count_retry < 2:
            raise BackendError("Transient failure")
        return "success"

    result = await succeeds_on_retry()
    assert result == "success"
    assert call_count_retry == 2
    print(f"  [OK] Succeeds on retry: {call_count_retry} attempts")

    # Test Case 3: Exhausts retries
    call_count_fail = 0

    @retry(max_attempts=3, initial_delay=0.01, retry_on=(BackendError,))
    async def always_fails():
        nonlocal call_count_fail
        call_count_fail += 1
        raise BackendError("Permanent failure")

    with pytest.raises(BackendError):
        await always_fails()

    assert call_count_fail == 3
    print(f"  [OK] Exhausts retries: {call_count_fail} attempts before giving up")

    # Test Case 4: Exponential backoff timing
    start = time.time()
    call_count_timing = 0

    @retry(max_attempts=3, backoff_factor=2.0, initial_delay=0.05, jitter=0.0, retry_on=(BackendError,))
    async def test_timing():
        nonlocal call_count_timing
        call_count_timing += 1
        raise BackendError("Testing backoff")

    try:
        await test_timing()
    except BackendError:
        pass

    elapsed = time.time() - start
    # Expected delays: 0.05 (first retry) + 0.10 (second retry) = 0.15s minimum
    assert elapsed >= 0.15, f"Backoff too fast: {elapsed:.3f}s"
    assert elapsed < 0.5, f"Backoff too slow: {elapsed:.3f}s"
    print(f"  [OK] Exponential backoff: {elapsed:.3f}s elapsed (expected ~0.15s)")

    print("\n[PASS] Retry decorator working correctly")


# ============================================================================
# Test 4: Fallback Strategy
# ============================================================================

@pytest.mark.asyncio
async def test_fallback_strategy():
    """Test 4: Fallback strategy cascade"""
    print("\n" + "=" * 80)
    print("Test 4: Fallback Strategy")
    print("=" * 80)

    strategy = FallbackStrategy(default_confidence=0.75)

    # Test Case 1: Primary succeeds
    async def primary_success():
        return 0.95

    confidence = await strategy.get_confidence(primary=primary_success)
    assert confidence == 0.95
    print(f"  [OK] Primary succeeds: {confidence:.2f}")

    # Test Case 2: Primary fails, fallback 1 succeeds
    async def primary_fails():
        raise CalibrationError("Primary failed")

    async def fallback_1_success():
        return 0.85

    confidence = await strategy.get_confidence(
        primary=primary_fails,
        fallback_1=fallback_1_success
    )
    assert confidence == 0.85
    print(f"  [OK] Primary fails, fallback 1 succeeds: {confidence:.2f}")

    # Test Case 3: Primary and fallback 1 fail, fallback 2 succeeds
    async def fallback_1_fails():
        raise CalibrationError("Fallback 1 failed")

    async def fallback_2_success():
        return 0.80

    confidence = await strategy.get_confidence(
        primary=primary_fails,
        fallback_1=fallback_1_fails,
        fallback_2=fallback_2_success
    )
    assert confidence == 0.80
    print(f"  [OK] Primary and fallback 1 fail, fallback 2 succeeds: {confidence:.2f}")

    # Test Case 4: All fail, use default
    async def fallback_2_fails():
        raise CalibrationError("Fallback 2 failed")

    confidence = await strategy.get_confidence(
        primary=primary_fails,
        fallback_1=fallback_1_fails,
        fallback_2=fallback_2_fails
    )
    assert confidence == 0.75  # default
    print(f"  [OK] All fail, use default: {confidence:.2f}")

    # Test Case 5: Fallback rate tracking
    fallback_rate = strategy.get_fallback_rate()
    assert fallback_rate > 0.0
    print(f"  [OK] Fallback rate: {fallback_rate:.2%}")

    print("\n[PASS] Fallback strategy working correctly")


# ============================================================================
# Test 5: Error Handler Integration
# ============================================================================

@pytest.mark.asyncio
async def test_error_handler():
    """Test 5: Error handler integration"""
    print("\n" + "=" * 80)
    print("Test 5: Error Handler Integration")
    print("=" * 80)

    handler = create_error_handler()

    # Test Case 1: Handle transient error with fallback
    async def transient_fallback():
        return "fallback_result"

    result = await handler.handle(
        BackendError("Transient failure"),
        context="test_transient",
        fallback=transient_fallback
    )
    assert result == "fallback_result"
    print(f"  [OK] Transient error with fallback: {result}")

    # Test Case 2: Handle permanent error with fallback
    result = await handler.handle(
        CalibrationError("Permanent failure"),
        context="test_permanent",
        fallback=transient_fallback
    )
    assert result == "fallback_result"
    print(f"  [OK] Permanent error with fallback: {result}")

    # Test Case 3: Handle rate limit (no fallback execution)
    with pytest.raises(RateLimitExceededError):
        await handler.handle(
            RateLimitExceededError("Too many requests"),
            context="test_rate_limit",
            fallback=transient_fallback
        )
    print(f"  [OK] Rate limit error raised (no fallback)")

    # Test Case 4: Handle circuit open (no fallback execution)
    with pytest.raises(CircuitBreakerOpenError):
        await handler.handle(
            CircuitBreakerOpenError("Circuit open"),
            context="test_circuit_open",
            fallback=transient_fallback
        )
    print(f"  [OK] Circuit open error raised (no fallback)")

    # Test Case 5: Error statistics
    stats = handler.get_error_stats()
    assert stats["total_errors"] == 4  # 2 handled with fallback + 2 raised
    print(f"  [OK] Error stats: {stats['total_errors']} total errors")

    # Verify error counts
    assert "BackendError" in stats["error_counts"]
    assert "CalibrationError" in stats["error_counts"]
    assert "RateLimitExceededError" in stats["error_counts"]
    assert "CircuitBreakerOpenError" in stats["error_counts"]
    print(f"  [OK] Error counts: {stats['error_counts']}")

    # Reset stats
    handler.reset_stats()
    stats = handler.get_error_stats()
    assert stats["total_errors"] == 0
    print(f"  [OK] Reset stats: {stats['total_errors']} errors")

    print("\n[PASS] Error handler integration working correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all error handling tests"""
    print("=" * 80)
    print("Error Handling Test Suite")
    print("=" * 80)
    print()
    print("Testing error handling, retry, and fallback mechanisms...")
    print()

    # Run sync tests
    test_exception_hierarchy()
    test_error_categorization()

    # Run async tests
    asyncio.run(test_retry_decorator())
    asyncio.run(test_fallback_strategy())
    asyncio.run(test_error_handler())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("Tests Passed: 5/5")
    print()
    print("[PASS] Exception Hierarchy")
    print("[PASS] Error Categorization")
    print("[PASS] Retry Decorator")
    print("[PASS] Fallback Strategy")
    print("[PASS] Error Handler Integration")
    print()
    print("[SUCCESS] All error handling tests passed!")
    print()


if __name__ == "__main__":
    main()
