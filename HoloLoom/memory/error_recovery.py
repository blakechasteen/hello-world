"""
Memory System Error Recovery
==============================

Week 8A: Production-grade error recovery for memory systems.

Provides comprehensive error recovery strategies:
- Circuit breaker pattern (prevent cascading failures)
- Retry logic with exponential backoff
- Graceful degradation
- Error aggregation and reporting

Philosophy:
"Fail gracefully, recover automatically, learn from failures"

Author: HoloLoom Memory Team
Date: 2025-11-18 (Week 8A: Error Handling & Defensive Programming)
"""

from typing import Any, Callable, Optional, Dict, List, TypeVar, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to delays


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60  # Time before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


# ============================================================================
# Circuit Breaker States
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


# ============================================================================
# Errors
# ============================================================================

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryExhausted(Exception):
    """Raised when max retries exhausted."""
    pass


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject all requests
    - HALF_OPEN: Testing if service recovered

    Usage:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        >>>
        >>> async def risky_operation():
        >>>     return await breaker.call(some_async_function, arg1, arg2)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening
            success_threshold: Successes to close from half-open
            timeout_seconds: Time before trying half-open
            half_open_max_calls: Max concurrent calls in half-open
            name: Circuit breaker name (for logging)
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

        # Statistics
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_rejections = 0

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, timeout={timeout_seconds}s"
        )

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Original exception from function
        """
        self.total_calls += 1

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit breaker '{self.name}' attempting reset (half-open)")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                self.total_rejections += 1
                logger.warning(f"Circuit breaker '{self.name}' is OPEN, rejecting call")
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is open, "
                    f"retry after {self._seconds_until_retry():.1f}s"
                )

        # Check half-open concurrency limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.total_rejections += 1
                logger.warning(
                    f"Circuit breaker '{self.name}' half-open concurrency limit reached"
                )
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is testing recovery, "
                    f"try again later"
                )

        # Execute function
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

        finally:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls -= 1

    def _on_success(self):
        """Handle successful call."""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"Circuit breaker '{self.name}' success in half-open "
                f"({self.success_count}/{self.success_threshold})"
            )

            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' closing (recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.total_failures += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' failure in half-open, reopening"
            )
            self.state = CircuitState.OPEN
            self.failure_count = 0
            self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            logger.debug(
                f"Circuit breaker '{self.name}' failure "
                f"({self.failure_count}/{self.failure_threshold})"
            )

            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' OPENING (threshold reached)"
                )
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def _seconds_until_retry(self) -> float:
        """Calculate seconds until retry is allowed."""
        if self.last_failure_time is None:
            return 0.0

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        remaining = self.timeout_seconds - elapsed
        return max(0.0, remaining)

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "success_rate": (
                self.total_successes / self.total_calls
                if self.total_calls > 0 else 0.0
            ),
            "seconds_until_retry": self._seconds_until_retry() if self.state == CircuitState.OPEN else 0.0
        }

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


# ============================================================================
# Retry Logic
# ============================================================================

async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs
) -> T:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to delay
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        RetryExhausted: If all retries failed
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_error = e

            if attempt == max_retries:
                # Final attempt failed
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                )
                raise RetryExhausted(
                    f"Exhausted {max_retries + 1} retries for {func.__name__}"
                ) from e

            # Calculate delay
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter
            if jitter:
                import random
                delay *= (0.5 + random.random())  # 50-150% of base delay

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}, "
                f"retrying in {delay:.2f}s"
            )

            await asyncio.sleep(delay)

    # Should never reach here
    raise RetryExhausted("Retry logic error") from last_error


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        >>> @with_retry(max_retries=3, base_delay=1.0)
        >>> async def my_function():
        >>>     return await risky_operation()
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                **kwargs
            )
        return wrapper
    return decorator


# ============================================================================
# Safe Execution Wrapper
# ============================================================================

async def safe_execute(
    func: Callable[..., Awaitable[T]],
    *args,
    fallback: Optional[T] = None,
    error_message: str = "Operation failed",
    **kwargs
) -> T:
    """
    Execute async function with error handling and fallback.

    Args:
        func: Async function to execute
        *args: Positional arguments
        fallback: Fallback value on error
        error_message: Custom error message
        **kwargs: Keyword arguments

    Returns:
        Function result or fallback value
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}", exc_info=True)
        if fallback is not None:
            logger.info(f"Using fallback value: {fallback}")
            return fallback
        else:
            raise


def safe_execute_sync(
    func: Callable[..., T],
    *args,
    fallback: Optional[T] = None,
    error_message: str = "Operation failed",
    **kwargs
) -> T:
    """
    Execute sync function with error handling and fallback.

    Args:
        func: Sync function to execute
        *args: Positional arguments
        fallback: Fallback value on error
        error_message: Custom error message
        **kwargs: Keyword arguments

    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}", exc_info=True)
        if fallback is not None:
            logger.info(f"Using fallback value: {fallback}")
            return fallback
        else:
            raise


# ============================================================================
# Error Aggregation
# ============================================================================

@dataclass
class ErrorRecord:
    """Record of error occurrence."""
    timestamp: datetime
    error_type: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


class ErrorAggregator:
    """
    Aggregate and analyze errors across memory systems.

    Tracks error patterns, frequencies, and trends.
    """

    def __init__(self, max_errors: int = 1000):
        """
        Initialize error aggregator.

        Args:
            max_errors: Maximum errors to keep in memory
        """
        self.max_errors = max_errors
        self.errors: List[ErrorRecord] = []

        # Statistics
        self.error_counts: Dict[str, int] = {}
        self.total_errors = 0

    def record_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record error occurrence.

        Args:
            error: Exception that occurred
            context: Additional context (operation, args, etc.)
        """
        import traceback as tb

        error_type = type(error).__name__
        error_message = str(error)

        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            context=context or {},
            traceback=tb.format_exc()
        )

        self.errors.append(record)
        self.total_errors += 1

        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Prune old errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]

        logger.debug(f"Recorded error: {error_type} - {error_message}")

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get error summary for recent time window.

        Args:
            hours: Time window in hours

        Returns:
            Error summary statistics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.errors if e.timestamp >= cutoff]

        # Count by type
        type_counts = {}
        for error in recent_errors:
            type_counts[error.error_type] = type_counts.get(error.error_type, 0) + 1

        # Most common errors
        top_errors = sorted(
            type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_errors": len(recent_errors),
            "error_types": len(type_counts),
            "top_errors": [
                {"type": error_type, "count": count}
                for error_type, count in top_errors
            ],
            "time_window_hours": hours,
            "all_time_total": self.total_errors
        }

    def get_recent_errors(self, limit: int = 10) -> List[ErrorRecord]:
        """
        Get most recent errors.

        Args:
            limit: Maximum errors to return

        Returns:
            Recent error records
        """
        return self.errors[-limit:]

    def clear_errors(self):
        """Clear all error records."""
        self.errors.clear()
        self.error_counts.clear()
        logger.info("Error records cleared")


# ============================================================================
# Global Error Aggregator
# ============================================================================

_global_error_aggregator = ErrorAggregator()


def get_error_aggregator() -> ErrorAggregator:
    """Get global error aggregator instance."""
    return _global_error_aggregator


# ============================================================================
# Factory Functions
# ============================================================================

def create_circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    name: str = "default"
) -> CircuitBreaker:
    """
    Create circuit breaker with configuration.

    Args:
        failure_threshold: Failures before opening
        timeout_seconds: Time before retry
        name: Circuit breaker name

    Returns:
        CircuitBreaker instance
    """
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        name=name
    )
