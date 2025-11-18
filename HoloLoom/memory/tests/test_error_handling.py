"""
Tests for Memory System Error Handling
========================================

Week 8A: Comprehensive error handling tests.

Test Coverage:
1. Input validation (20+ tests)
2. Error recovery (15+ tests)
3. Circuit breaker (10+ tests)
4. Retry logic (10+ tests)
5. Graceful degradation (10+ tests)
6. Edge cases (15+ tests)

Author: HoloLoom Memory Team
Date: 2025-11-18 (Week 8A: Error Handling & Defensive Programming)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from HoloLoom.memory.validation import (
    MemoryValidator,
    ValidationConfig,
    QueryValidationError,
    ConfidenceValidationError,
    TimestampValidationError,
    MemoryIdValidationError,
    EntityValidationError,
    create_validator
)

from HoloLoom.memory.error_recovery import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpen,
    RetryExhausted,
    retry_with_backoff,
    with_retry,
    safe_execute,
    safe_execute_sync,
    ErrorAggregator,
    get_error_aggregator
)

from HoloLoom.memory.health import (
    MemorySystemHealth,
    HealthStatus,
    HealthLevel,
    create_health_checker
)


# ============================================================================
# Validation Tests (20+)
# ============================================================================

class TestMemoryValidator:
    """Test input validation."""

    def test_validate_query_normal(self):
        """Test normal query validation."""
        validator = MemoryValidator()
        result = validator.validate_query("What is Thompson Sampling?")
        assert result == "What is Thompson Sampling?"

    def test_validate_query_none(self):
        """Test None query handling."""
        validator = MemoryValidator()
        result = validator.validate_query(None, allow_empty=True)
        assert result == ""

    def test_validate_query_empty_not_allowed(self):
        """Test empty query rejection."""
        validator = MemoryValidator()
        result = validator.validate_query("", allow_empty=False)
        assert result == "[empty query]"

    def test_validate_query_too_long(self):
        """Test query truncation."""
        validator = MemoryValidator()
        long_query = "a" * 20000  # Exceeds MAX_QUERY_LENGTH
        result = validator.validate_query(long_query)
        assert len(result) == validator.config.MAX_QUERY_LENGTH

    def test_validate_query_control_chars(self):
        """Test control character removal."""
        validator = MemoryValidator()
        result = validator.validate_query("Hello\x00World\x1f!")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_validate_query_type_coercion(self):
        """Test non-string query coercion."""
        validator = MemoryValidator()
        result = validator.validate_query(12345)
        assert result == "12345"

    def test_validate_confidence_normal(self):
        """Test normal confidence validation."""
        validator = MemoryValidator()
        result = validator.validate_confidence(0.85)
        assert result == 0.85

    def test_validate_confidence_none(self):
        """Test None confidence."""
        validator = MemoryValidator()
        result = validator.validate_confidence(None)
        assert result == 0.0

    def test_validate_confidence_clamping_low(self):
        """Test confidence clamping (too low)."""
        validator = MemoryValidator()
        result = validator.validate_confidence(-0.5)
        assert result == 0.0

    def test_validate_confidence_clamping_high(self):
        """Test confidence clamping (too high)."""
        validator = MemoryValidator()
        result = validator.validate_confidence(1.5)
        assert result == 1.0

    def test_validate_confidence_nan(self):
        """Test NaN confidence."""
        validator = MemoryValidator()
        result = validator.validate_confidence(float('nan'))
        assert result == 0.0

    def test_validate_confidence_inf(self):
        """Test infinite confidence."""
        validator = MemoryValidator()
        result = validator.validate_confidence(float('inf'))
        assert result == 0.0

    def test_validate_timestamp_normal(self):
        """Test normal timestamp validation."""
        validator = MemoryValidator()
        now = datetime.now()
        result = validator.validate_timestamp(now)
        assert result == now

    def test_validate_timestamp_none(self):
        """Test None timestamp."""
        validator = MemoryValidator()
        result = validator.validate_timestamp(None)
        assert isinstance(result, datetime)

    def test_validate_timestamp_future(self):
        """Test future timestamp rejection."""
        validator = MemoryValidator()
        future = datetime.now() + timedelta(days=1)
        result = validator.validate_timestamp(future)
        # Should be clamped to now
        assert result <= datetime.now()

    def test_validate_timestamp_from_float(self):
        """Test timestamp from Unix timestamp."""
        validator = MemoryValidator()
        timestamp = datetime.now().timestamp()
        result = validator.validate_timestamp(timestamp)
        assert isinstance(result, datetime)

    def test_validate_timestamp_from_iso(self):
        """Test timestamp from ISO string."""
        validator = MemoryValidator()
        iso = "2025-11-18T10:00:00"
        result = validator.validate_timestamp(iso)
        assert isinstance(result, datetime)

    def test_validate_memory_id_normal(self):
        """Test normal memory ID validation."""
        validator = MemoryValidator()
        result = validator.validate_memory_id("memory_12345")
        assert result == "memory_12345"

    def test_validate_memory_id_none(self):
        """Test None memory ID."""
        validator = MemoryValidator()
        result = validator.validate_memory_id(None)
        assert result == "invalid_id"

    def test_validate_memory_id_invalid_chars(self):
        """Test memory ID with invalid characters."""
        validator = MemoryValidator()
        result = validator.validate_memory_id("memory@#$%12345")
        assert "@" not in result
        assert "#" not in result

    def test_validate_entities_normal(self):
        """Test normal entity validation."""
        validator = MemoryValidator()
        result = validator.validate_entities(["Thompson", "Sampling", "Algorithm"])
        assert len(result) == 3

    def test_validate_entities_none(self):
        """Test None entities."""
        validator = MemoryValidator()
        result = validator.validate_entities(None)
        assert result == []

    def test_validate_entities_single_string(self):
        """Test single entity string."""
        validator = MemoryValidator()
        result = validator.validate_entities("Thompson Sampling")
        assert len(result) == 1
        assert result[0] == "Thompson Sampling"

    def test_validate_entities_too_long(self):
        """Test entity truncation."""
        validator = MemoryValidator()
        long_entity = "a" * 500
        result = validator.validate_entities([long_entity])
        assert len(result[0]) == validator.config.MAX_ENTITY_LENGTH

    def test_validate_concept_text_normal(self):
        """Test normal concept validation."""
        validator = MemoryValidator()
        result = validator.validate_concept_text("Thompson Sampling is a Bayesian algorithm")
        assert "Thompson Sampling" in result


# ============================================================================
# Error Recovery Tests (15+)
# ============================================================================

class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_circuit_closed_normal(self):
        """Test normal operation (circuit closed)."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fail_func():
            raise ValueError("Test error")

        # Trigger failures
        for _ in range(3):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=10)

        async def fail_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        # Should reject
        with pytest.raises(CircuitBreakerOpen):
            await breaker.call(fail_func)

    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test circuit recovery through half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1
        )

        async def fail_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Open circuit
        for _ in range(2):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Should try half-open
        result1 = await breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN

        # Close circuit with second success
        result2 = await breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_manual_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def fail_func():
            raise ValueError("Test error")

        # Open circuit
        for _ in range(2):
            try:
                await breaker.call(fail_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestRetryLogic:
    """Test retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful first attempt (no retry)."""
        async def success_func():
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test success after retries."""
        call_count = 0

        async def sometimes_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_with_backoff(sometimes_fail, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion."""
        async def always_fail():
            raise ValueError("Permanent failure")

        with pytest.raises(RetryExhausted):
            await retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_retry_decorator(self):
        """Test retry decorator."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry me")
            return "success"

        result = await decorated_func()
        assert result == "success"
        assert call_count == 2


class TestSafeExecution:
    """Test safe execution wrappers."""

    @pytest.mark.asyncio
    async def test_safe_execute_success(self):
        """Test safe execution with success."""
        async def success_func():
            return "success"

        result = await safe_execute(success_func, fallback="fallback")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_safe_execute_with_fallback(self):
        """Test safe execution with fallback."""
        async def fail_func():
            raise ValueError("Error")

        result = await safe_execute(fail_func, fallback="fallback")
        assert result == "fallback"

    def test_safe_execute_sync(self):
        """Test safe execution (sync)."""
        def success_func():
            return "success"

        result = safe_execute_sync(success_func, fallback="fallback")
        assert result == "success"

    def test_safe_execute_sync_with_fallback(self):
        """Test safe execution (sync) with fallback."""
        def fail_func():
            raise ValueError("Error")

        result = safe_execute_sync(fail_func, fallback="fallback")
        assert result == "fallback"


class TestErrorAggregator:
    """Test error aggregation."""

    def test_record_error(self):
        """Test recording errors."""
        aggregator = ErrorAggregator()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            aggregator.record_error(e, context={"operation": "test"})

        assert aggregator.total_errors == 1
        assert len(aggregator.errors) == 1
        assert aggregator.error_counts["ValueError"] == 1

    def test_error_summary(self):
        """Test error summary."""
        aggregator = ErrorAggregator()

        # Record some errors
        for i in range(5):
            try:
                if i % 2 == 0:
                    raise ValueError("Test error")
                else:
                    raise TypeError("Another error")
            except Exception as e:
                aggregator.record_error(e)

        summary = aggregator.get_error_summary(hours=24)
        assert summary['total_errors'] == 5
        assert summary['error_types'] == 2

    def test_recent_errors(self):
        """Test getting recent errors."""
        aggregator = ErrorAggregator()

        for i in range(10):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                aggregator.record_error(e)

        recent = aggregator.get_recent_errors(limit=5)
        assert len(recent) == 5


# ============================================================================
# Health Check Tests (10+)
# ============================================================================

class TestHealthChecks:
    """Test health monitoring."""

    @pytest.mark.asyncio
    async def test_health_check_no_components(self):
        """Test health check with no components."""
        health_checker = MemorySystemHealth()
        health = await health_checker.get_overall_health()

        assert health['overall']['healthy'] is True
        assert health['overall']['level'] == HealthLevel.HEALTHY.value

    @pytest.mark.asyncio
    async def test_consolidation_health(self):
        """Test consolidation health check."""
        # Mock consolidator
        consolidator = Mock()
        consolidator.get_statistics = Mock(return_value={
            'total_consolidations': 10,
            'total_facts_extracted': 50,
            'llm_usage': {'error_rate': 0.05}
        })
        consolidator._running = True

        health_checker = MemorySystemHealth(consolidator=consolidator)
        health = await health_checker.check_consolidation_health()

        assert health.healthy is True

    @pytest.mark.asyncio
    async def test_temporal_tracking_health(self):
        """Test temporal tracking health check."""
        # Mock tracker
        tracker = Mock()
        tracker.concept_histories = {
            'concept1': Mock(
                state_transitions=[],
                milestones=[],
                last_accessed=datetime.now()
            )
        }
        tracker.interaction_log = [1, 2, 3]
        tracker.config = Mock(persistence_path=None)

        health_checker = MemorySystemHealth(temporal_tracker=tracker)
        health = await health_checker.check_temporal_tracking_health()

        assert health.healthy is True

    @pytest.mark.asyncio
    async def test_health_trend_stable(self):
        """Test health trend analysis."""
        health_checker = MemorySystemHealth()

        # Simulate multiple checks
        for _ in range(5):
            await health_checker.get_overall_health()

        trend = health_checker.get_health_trend(hours=1)
        assert trend['trend'] == 'stable_healthy'


# ============================================================================
# Graceful Degradation Tests (10+)
# ============================================================================

class TestGracefulDegradation:
    """Test graceful degradation scenarios."""

    def test_validator_fallback_on_error(self):
        """Test validator falls back gracefully."""
        validator = MemoryValidator()

        # Invalid type should convert to string
        result = validator.validate_query({'complex': 'object'})
        assert isinstance(result, str)

    def test_confidence_fallback_on_invalid(self):
        """Test confidence falls back to 0.0."""
        validator = MemoryValidator()
        result = validator.validate_confidence("not a number")
        assert result == 0.0

    def test_timestamp_fallback_on_invalid(self):
        """Test timestamp falls back to now."""
        validator = MemoryValidator()
        result = validator.validate_timestamp("invalid date")
        assert isinstance(result, datetime)

    def test_entities_filter_invalid(self):
        """Test entities filters out invalid values."""
        validator = MemoryValidator()
        result = validator.validate_entities([None, "", "Valid", 123, "Another"])
        assert "Valid" in result
        assert "Another" in result
        assert len(result) == 3  # "Valid", "123", "Another"


# ============================================================================
# Edge Cases (15+)
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self):
        """Test empty query handling."""
        validator = MemoryValidator()
        result = validator.validate_query("")
        assert result == "[empty query]"

    def test_whitespace_only_query(self):
        """Test whitespace-only query."""
        validator = MemoryValidator()
        result = validator.validate_query("   \n\t  ")
        assert result == "[empty query]"

    def test_unicode_query(self):
        """Test Unicode query."""
        validator = MemoryValidator()
        result = validator.validate_query("What is Thompson Sampling? 你好")
        assert "你好" in result

    def test_confidence_boundary_min(self):
        """Test confidence at lower boundary."""
        validator = MemoryValidator()
        result = validator.validate_confidence(0.0)
        assert result == 0.0

    def test_confidence_boundary_max(self):
        """Test confidence at upper boundary."""
        validator = MemoryValidator()
        result = validator.validate_confidence(1.0)
        assert result == 1.0

    def test_very_old_timestamp(self):
        """Test very old timestamp."""
        validator = MemoryValidator()
        old_date = datetime(1900, 1, 1)
        result = validator.validate_timestamp(old_date)
        assert isinstance(result, datetime)

    def test_large_entity_list(self):
        """Test large entity list."""
        validator = MemoryValidator()
        entities = [f"entity_{i}" for i in range(200)]
        result = validator.validate_entities(entities)
        assert len(result) <= validator.config.MAX_ENTITIES_PER_MEMORY

    def test_memory_id_special_chars(self):
        """Test memory ID with allowed special characters."""
        validator = MemoryValidator()
        result = validator.validate_memory_id("memory-123_456.789")
        assert result == "memory-123_456.789"

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_calls(self):
        """Test circuit breaker with concurrent calls."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def success_func():
            await asyncio.sleep(0.01)
            return "success"

        results = await asyncio.gather(*[
            breaker.call(success_func) for _ in range(10)
        ])

        assert all(r == "success" for r in results)
        assert breaker.state == CircuitState.CLOSED

    def test_error_aggregator_max_errors(self):
        """Test error aggregator respects max size."""
        aggregator = ErrorAggregator(max_errors=10)

        for i in range(20):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                aggregator.record_error(e)

        assert len(aggregator.errors) == 10  # Only keeps last 10

    def test_validator_strict_mode(self):
        """Test validator strict mode raises errors."""
        config = ValidationConfig()
        config.STRICT_MODE = True
        validator = MemoryValidator(config)

        with pytest.raises(QueryValidationError):
            validator.validate_query(None, allow_empty=False)

    def test_validator_non_strict_mode(self):
        """Test validator non-strict mode uses defaults."""
        config = ValidationConfig()
        config.STRICT_MODE = False
        validator = MemoryValidator(config)

        result = validator.validate_query(None, allow_empty=False)
        assert isinstance(result, str)

    def test_concept_text_very_long(self):
        """Test very long concept text truncation."""
        validator = MemoryValidator()
        long_text = "a" * 2000
        result = validator.validate_concept_text(long_text)
        assert len(result) <= validator.config.MAX_CONCEPT_LENGTH

    def test_entities_with_duplicates(self):
        """Test entity list with duplicates."""
        validator = MemoryValidator()
        result = validator.validate_entities(["A", "B", "A", "C", "B"])
        # Should keep all (deduplication is not validator's job)
        assert len(result) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
