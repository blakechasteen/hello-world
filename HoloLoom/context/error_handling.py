"""
Error Handling: Graceful Degradation and Recovery

Part 5: Production Hardening - Day 21

Provides comprehensive error handling for the Context Department:
- Custom exception hierarchy
- Retry logic with exponential backoff
- Fallback strategies for failed operations
- Error categorization (transient vs permanent)
- Automatic recovery mechanisms
"""

import asyncio
import time
import functools
from typing import Callable, Optional, TypeVar, Any, Type, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Exception Hierarchy
# ============================================================================

class ContextError(Exception):
    """Base exception for all context operations"""
    pass


class RoutingError(ContextError):
    """Errors during query routing"""
    pass


class BackendError(ContextError):
    """Errors from backend operations (MCP, Neo4j, Qdrant)"""
    pass


class CalibrationError(ContextError):
    """Errors in confidence calibration"""
    pass


class LearningError(ContextError):
    """Errors in learning mechanisms"""
    pass


class RateLimitExceededError(ContextError):
    """Rate limit exceeded"""
    pass


class CircuitBreakerOpenError(ContextError):
    """Circuit breaker is open"""
    pass


class ErrorCategory(Enum):
    """Error category for handling decisions"""
    TRANSIENT = "transient"      # Temporary, retry likely to succeed
    PERMANENT = "permanent"      # Permanent, retry won't help
    RATE_LIMIT = "rate_limit"    # Rate limit, need to back off
    TIMEOUT = "timeout"          # Timeout, may succeed with more time
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open


# ============================================================================
# Retry Configuration
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 0.1  # seconds
    max_delay: float = 10.0     # seconds
    jitter: float = 0.1         # Random jitter (0.0 - 1.0)
    retry_on: Tuple[Type[Exception], ...] = (BackendError, RoutingError)


# ============================================================================
# Retry Decorator
# ============================================================================

T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    retry_on: Tuple[Type[Exception], ...] = (BackendError,)
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff

    Args:
        max_attempts: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Random jitter factor (0.0 - 1.0)
        retry_on: Exception types to retry on

    Example:
        @retry(max_attempts=3, backoff_factor=2.0, retry_on=(BackendError,))
        async def risky_operation():
            # This will retry up to 3 times on BackendError
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        # Add jitter to prevent thundering herd
                        import random
                        jittered_delay = delay * (1 + random.uniform(-jitter, jitter))
                        actual_delay = min(jittered_delay, max_delay)

                        await asyncio.sleep(actual_delay)
                        delay *= backoff_factor
                    else:
                        # Last attempt, raise
                        raise

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


# ============================================================================
# Error Categorization
# ============================================================================

def categorize_error(error: Exception) -> ErrorCategory:
    """
    Categorize error for handling decisions

    Args:
        error: Exception to categorize

    Returns:
        ErrorCategory indicating how to handle
    """
    # Network/backend errors are usually transient
    if isinstance(error, BackendError):
        return ErrorCategory.TRANSIENT

    # Rate limits need backoff
    if isinstance(error, RateLimitExceededError):
        return ErrorCategory.RATE_LIMIT

    # Circuit breaker open
    if isinstance(error, CircuitBreakerOpenError):
        return ErrorCategory.CIRCUIT_OPEN

    # Timeouts may succeed with retry
    if isinstance(error, asyncio.TimeoutError):
        return ErrorCategory.TIMEOUT

    # Calibration/learning errors are permanent
    if isinstance(error, (CalibrationError, LearningError)):
        return ErrorCategory.PERMANENT

    # Unknown errors treated as permanent
    return ErrorCategory.PERMANENT


# ============================================================================
# Fallback Strategies
# ============================================================================

class FallbackStrategy:
    """
    Fallback strategy for failed operations

    Implements cascading fallback:
    1. Primary: Use calibrated confidence
    2. Fallback 1: Use raw confidence (if calibration fails)
    3. Fallback 2: Use fixed confidence (if prediction fails)
    """

    def __init__(self, default_confidence: float = 0.75):
        """
        Initialize fallback strategy

        Args:
            default_confidence: Default confidence if all else fails
        """
        self.default_confidence = default_confidence
        self.fallback_count = 0
        self.fallback_history = []

    async def get_confidence(
        self,
        primary: Callable,
        fallback_1: Optional[Callable] = None,
        fallback_2: Optional[Callable] = None
    ) -> float:
        """
        Get confidence with fallback cascade

        Args:
            primary: Primary confidence source
            fallback_1: First fallback (optional)
            fallback_2: Second fallback (optional)

        Returns:
            Confidence value (0.0 - 1.0)
        """
        # Try primary
        try:
            confidence = await primary()
            return float(confidence)
        except Exception as e:
            self.fallback_count += 1
            self.fallback_history.append({
                "level": 1,
                "error": str(e),
                "timestamp": time.time()
            })

        # Try fallback 1
        if fallback_1:
            try:
                confidence = await fallback_1()
                return float(confidence)
            except Exception as e:
                self.fallback_count += 1
                self.fallback_history.append({
                    "level": 2,
                    "error": str(e),
                    "timestamp": time.time()
                })

        # Try fallback 2
        if fallback_2:
            try:
                confidence = await fallback_2()
                return float(confidence)
            except Exception as e:
                self.fallback_count += 1
                self.fallback_history.append({
                    "level": 3,
                    "error": str(e),
                    "timestamp": time.time()
                })

        # Final fallback: use default
        return self.default_confidence

    def get_fallback_rate(self, window_seconds: float = 3600.0) -> float:
        """
        Get fallback rate in recent window

        Args:
            window_seconds: Time window in seconds

        Returns:
            Fallback rate (0.0 - 1.0)
        """
        cutoff = time.time() - window_seconds
        recent = [fb for fb in self.fallback_history if fb["timestamp"] > cutoff]

        if not recent:
            return 0.0

        return len(recent) / max(1, self.fallback_count)


# ============================================================================
# Error Handler
# ============================================================================

class ErrorHandler:
    """
    Centralized error handling for context operations

    Provides:
    - Error categorization
    - Automatic retry for transient errors
    - Fallback strategies
    - Error logging and metrics
    """

    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        Initialize error handler

        Args:
            retry_config: Retry configuration (uses defaults if None)
        """
        self.retry_config = retry_config or RetryConfig()
        self.fallback_strategy = FallbackStrategy()
        self.error_counts = {}
        self.error_history = []

    async def handle(
        self,
        error: Exception,
        context: str,
        fallback: Optional[Callable] = None
    ) -> Any:
        """
        Handle error with appropriate strategy

        Args:
            error: Exception to handle
            context: Context string for logging
            fallback: Optional fallback function

        Returns:
            Result from fallback (if provided and executed)

        Raises:
            Original exception if no fallback or fallback fails
        """
        # Categorize error
        category = categorize_error(error)

        # Record error
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_history.append({
            "error": error_type,
            "category": category.value,
            "context": context,
            "timestamp": time.time()
        })

        # Handle based on category
        if category == ErrorCategory.TRANSIENT:
            # Transient errors should have been retried already
            # If we're here, retries exhausted
            if fallback:
                return await fallback()
            raise error

        elif category == ErrorCategory.RATE_LIMIT:
            # Rate limit - need to back off
            # Don't retry, fail fast
            raise error

        elif category == ErrorCategory.CIRCUIT_OPEN:
            # Circuit open - fail fast
            raise error

        elif category == ErrorCategory.TIMEOUT:
            # Timeout - may succeed with fallback
            if fallback:
                return await fallback()
            raise error

        elif category == ErrorCategory.PERMANENT:
            # Permanent error - try fallback immediately
            if fallback:
                return await fallback()
            raise error

        # Unknown category - raise
        raise error

    def get_error_stats(self) -> dict:
        """
        Get error statistics

        Returns:
            Dictionary with error counts and rates
        """
        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "error_counts": dict(self.error_counts),
            "fallback_rate": self.fallback_strategy.get_fallback_rate(),
            "error_history_size": len(self.error_history)
        }

    def reset_stats(self):
        """Reset error statistics"""
        self.error_counts.clear()
        self.error_history.clear()
        self.fallback_strategy.fallback_count = 0
        self.fallback_strategy.fallback_history.clear()


# ============================================================================
# Utility Functions
# ============================================================================

def is_retryable(error: Exception) -> bool:
    """
    Check if error is retryable

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    category = categorize_error(error)
    return category in (ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT)


def should_fallback(error: Exception) -> bool:
    """
    Check if error should trigger fallback

    Args:
        error: Exception to check

    Returns:
        True if fallback should be used, False otherwise
    """
    category = categorize_error(error)
    return category not in (ErrorCategory.RATE_LIMIT, ErrorCategory.CIRCUIT_OPEN)


# ============================================================================
# Factory Function
# ============================================================================

def create_error_handler(retry_config: Optional[RetryConfig] = None) -> ErrorHandler:
    """
    Create error handler with configuration

    Args:
        retry_config: Retry configuration (uses defaults if None)

    Returns:
        Configured ErrorHandler instance
    """
    return ErrorHandler(retry_config=retry_config)
