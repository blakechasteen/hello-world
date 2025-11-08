"""
Phase 6C: Error Recovery & Resilience
Circuit breakers, retry logic with exponential backoff, graceful degradation.

Features:
- Circuit breaker pattern for external services
- Retry with exponential backoff
- Fallback mechanisms
- Timeout handling
- Error aggregation and reporting

Patterns:
- Circuit Breaker: Prevent cascading failures
- Retry: Handle transient errors
- Fallback: Graceful degradation
- Bulkhead: Isolate failures
"""

import time
import asyncio
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
    import logging
except ImportError:
    print("⚠️  tenacity not installed. Run: pip install tenacity==8.2.3")
    retry = None


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60  # Time before attempting half-open
    expected_exception: type = Exception  # Exception type to track


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.utcnow()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        # Check if circuit should transition to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Wait {self.config.timeout_seconds}s before retry."
                )

        try:
            result = func(*args, **kwargs)

            # Success
            self._on_success()

            return result

        except self.config.expected_exception as e:
            # Expected failure
            self._on_failure()
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call()."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Wait {self.config.timeout_seconds}s before retry."
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            # Failed while testing, go back to open
            self._transition_to(CircuitState.OPEN)

        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1

            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.utcnow()

        # Reset counters
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0

        print(
            f"🔄 Circuit breaker '{self.name}': {old_state.value} → {new_state.value}"
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }

    def reset(self):
        """Manually reset circuit breaker."""
        self._transition_to(CircuitState.CLOSED)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RetryConfig:
    """Retry configuration."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,),
    ):
        """
        Initialize retry config.

        Args:
            max_attempts: Maximum retry attempts
            initial_wait: Initial wait time (seconds)
            max_wait: Maximum wait time (seconds)
            exponential_base: Base for exponential backoff
            exceptions: Exception types to retry
        """
        self.max_attempts = max_attempts
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.exponential_base = exponential_base
        self.exceptions = exceptions


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retry with exponential backoff.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry logic
    """
    if retry is None:
        raise ImportError("tenacity not installed")

    cfg = config or RetryConfig()

    return retry(
        stop=stop_after_attempt(cfg.max_attempts),
        wait=wait_exponential(
            multiplier=cfg.initial_wait,
            max=cfg.max_wait,
            exp_base=cfg.exponential_base,
        ),
        retry=retry_if_exception_type(cfg.exceptions),
        reraise=True,
    )


async def with_timeout(coro, timeout_seconds: float, fallback: Any = None) -> Any:
    """
    Execute coroutine with timeout.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        fallback: Fallback value if timeout

    Returns:
        Result or fallback

    Raises:
        asyncio.TimeoutError: If timeout and no fallback
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if fallback is not None:
            return fallback
        raise


class FallbackHandler:
    """Fallback handler for graceful degradation."""

    def __init__(self):
        """Initialize fallback handler."""
        self.fallback_funcs: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Register fallback function."""
        self.fallback_funcs[name] = func

    async def execute_with_fallback(
        self, primary_func: Callable, fallback_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute function with fallback.

        Args:
            primary_func: Primary function to try
            fallback_name: Name of registered fallback
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Result from primary or fallback
        """
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)

        except Exception as e:
            print(f"⚠️  Primary function failed: {e}. Using fallback: {fallback_name}")

            fallback_func = self.fallback_funcs.get(fallback_name)

            if not fallback_func:
                raise Exception(f"No fallback registered for '{fallback_name}'")

            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)


class ErrorAggregator:
    """Aggregate errors for analysis."""

    def __init__(self, max_errors: int = 1000):
        """Initialize error aggregator."""
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)

    def record_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record an error."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {},
        }

        self.errors.append(entry)

    def get_recent_errors(self, limit: int = 100) -> list:
        """Get recent errors."""
        return list(self.errors)[-limit:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        if not self.errors:
            return {"total": 0, "by_type": {}}

        error_counts = {}
        for error in self.errors:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {"total": len(self.errors), "by_type": error_counts}


# Example usage
if __name__ == "__main__":
    import random

    # Circuit Breaker Example
    print("=" * 60)
    print("Circuit Breaker Example")
    print("=" * 60)

    breaker = CircuitBreaker("github_api", CircuitBreakerConfig(failure_threshold=3, timeout_seconds=5))

    def flaky_api_call():
        """Simulated flaky API call."""
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("API error")
        return "Success"

    for i in range(10):
        try:
            result = breaker.call(flaky_api_call)
            print(f"Attempt {i+1}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Attempt {i+1}: ⚠️  {e}")
        except Exception as e:
            print(f"Attempt {i+1}: ❌ {e}")

        time.sleep(0.5)

    print(f"\nCircuit State: {breaker.get_state()}")

    # Retry Example
    print("\n" + "=" * 60)
    print("Retry with Exponential Backoff Example")
    print("=" * 60)

    attempt_count = 0

    @with_retry(RetryConfig(max_attempts=5, initial_wait=0.5, max_wait=5.0))
    def flaky_operation():
        """Simulated operation that eventually succeeds."""
        global attempt_count
        attempt_count += 1
        print(f"Attempt {attempt_count}...")

        if attempt_count < 3:
            raise Exception("Temporary failure")

        return "Success after retries"

    try:
        result = flaky_operation()
        print(f"✅ Final result: {result}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")

    # Fallback Example
    print("\n" + "=" * 60)
    print("Fallback Handler Example")
    print("=" * 60)

    fallback = FallbackHandler()

    def primary_service():
        """Primary service (fails)."""
        raise Exception("Primary service unavailable")

    def backup_service():
        """Backup service (works)."""
        return "Response from backup service"

    fallback.register("backup", backup_service)

    async def test_fallback():
        result = await fallback.execute_with_fallback(primary_service, "backup")
        print(f"✅ Result: {result}")

    asyncio.run(test_fallback())

    # Error Aggregator Example
    print("\n" + "=" * 60)
    print("Error Aggregator Example")
    print("=" * 60)

    aggregator = ErrorAggregator()

    # Simulate some errors
    aggregator.record_error("APIError", "GitHub API rate limit exceeded")
    aggregator.record_error("DatabaseError", "Connection timeout")
    aggregator.record_error("APIError", "GitHub API unreachable")
    aggregator.record_error("ValidationError", "Invalid input")

    summary = aggregator.get_error_summary()
    print(f"Total errors: {summary['total']}")
    print("Errors by type:")
    for error_type, count in summary["by_type"].items():
        print(f"  {error_type}: {count}")

    recent = aggregator.get_recent_errors(limit=2)
    print(f"\nRecent errors (last 2):")
    for error in recent:
        print(f"  [{error['timestamp']}] {error['error_type']}: {error['message']}")
