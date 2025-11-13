"""
Error Recovery Module for Promptly Matrix Bot

Provides:
- Exponential backoff with jitter
- Retry decorators for async functions
- Circuit breaker pattern
- Graceful degradation

Usage:
    from bot.error_recovery import with_retry, BackoffStrategy

    @with_retry(max_attempts=3, backoff=BackoffStrategy.EXPONENTIAL)
    async def unstable_operation():
        # Operation that might fail
        pass
"""

import asyncio
import logging
import time
import random
from typing import Callable, Optional, Type, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategy for retries"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    # Don't retry on these exceptions
    fatal_exceptions: Tuple[Type[Exception], ...] = (KeyboardInterrupt, SystemExit)


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay for retry attempt based on backoff strategy

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    if config.backoff == BackoffStrategy.CONSTANT:
        delay = config.base_delay

    elif config.backoff == BackoffStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)

    elif config.backoff == BackoffStrategy.EXPONENTIAL:
        delay = config.base_delay * (2 ** attempt)

    elif config.backoff == BackoffStrategy.EXPONENTIAL_JITTER:
        # Exponential backoff with full jitter
        # Prevents thundering herd problem
        delay = random.uniform(0, config.base_delay * (2 ** attempt))

    else:
        delay = config.base_delay

    # Cap at max_delay
    return min(delay, config.max_delay)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    fatal_exceptions: Tuple[Type[Exception], ...] = (KeyboardInterrupt, SystemExit),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator to add retry logic with exponential backoff to async functions

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff: Backoff strategy to use
        exceptions: Tuple of exceptions to retry on
        fatal_exceptions: Tuple of exceptions that should not be retried
        on_retry: Optional callback called on each retry (attempt_num, exception)

    Example:
        @with_retry(max_attempts=3, backoff=BackoffStrategy.EXPONENTIAL_JITTER)
        async def fetch_data():
            # Operation that might fail
            pass
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff=backoff,
        exceptions=exceptions,
        fatal_exceptions=fatal_exceptions
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.fatal_exceptions:
                    # Don't retry fatal exceptions
                    raise

                except config.exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        # Calculate delay
                        delay = calculate_delay(attempt, config)

                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            on_retry(attempt + 1, e)

                        # Wait before retry
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations with exponential backoff

    Usage:
        async with RetryableOperation(max_attempts=3) as op:
            result = await op.execute(some_async_function, arg1, arg2)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        fatal_exceptions: Tuple[Type[Exception], ...] = (KeyboardInterrupt, SystemExit)
    ):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff=backoff,
            exceptions=exceptions,
            fatal_exceptions=fatal_exceptions
        )
        self.attempt_count = 0
        self.success_count = 0
        self.failure_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Log statistics
        logger.info(
            f"RetryableOperation stats: "
            f"{self.success_count} successes, "
            f"{self.failure_count} failures, "
            f"{self.attempt_count} total attempts"
        )
        return False

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            self.attempt_count += 1

            try:
                result = await func(*args, **kwargs)
                self.success_count += 1
                return result

            except self.config.fatal_exceptions:
                # Don't retry fatal exceptions
                self.failure_count += 1
                raise

            except self.config.exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    # Calculate delay
                    delay = calculate_delay(attempt, self.config)

                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    # Wait before retry
                    await asyncio.sleep(delay)
                else:
                    self.failure_count += 1
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed: {e}"
                    )

        # All retries exhausted
        raise last_exception


class GracefulDegradation:
    """
    Provide graceful degradation with fallback values

    Usage:
        degradation = GracefulDegradation()

        # Try operation with fallback
        result = await degradation.try_with_fallback(
            operation=fetch_from_api,
            fallback=get_cached_value,
            fallback_value="default"
        )
    """

    def __init__(self):
        self.degradation_count = 0
        self.last_degradation_time: Optional[float] = None

    async def try_with_fallback(
        self,
        operation: Callable,
        fallback: Optional[Callable] = None,
        fallback_value: Any = None,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Any:
        """
        Try operation with fallback on failure

        Args:
            operation: Primary operation to attempt
            fallback: Fallback operation (called if primary fails)
            fallback_value: Static fallback value (used if fallback operation also fails)
            exceptions: Exceptions to catch

        Returns:
            Result from operation, fallback, or fallback_value

        Priority:
            1. Try operation
            2. If fails, try fallback (if provided)
            3. If fallback fails or not provided, return fallback_value
        """
        try:
            # Try primary operation
            if asyncio.iscoroutinefunction(operation):
                return await operation()
            else:
                return operation()

        except exceptions as e:
            logger.warning(f"Primary operation failed: {e}. Attempting fallback...")

            # Track degradation
            self.degradation_count += 1
            self.last_degradation_time = time.time()

            # Try fallback operation
            if fallback:
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        result = await fallback()
                    else:
                        result = fallback()

                    logger.info("Fallback operation succeeded")
                    return result

                except exceptions as fallback_error:
                    logger.error(f"Fallback operation also failed: {fallback_error}")

            # Return fallback value
            logger.warning(f"Using fallback value: {fallback_value}")
            return fallback_value

    def get_stats(self) -> dict:
        """Get degradation statistics"""
        return {
            "degradation_count": self.degradation_count,
            "last_degradation_time": self.last_degradation_time,
            "last_degradation_ago": time.time() - self.last_degradation_time if self.last_degradation_time else None
        }


# Global degradation tracker (for monitoring)
global_degradation = GracefulDegradation()


def with_graceful_degradation(
    fallback: Optional[Callable] = None,
    fallback_value: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for graceful degradation

    Args:
        fallback: Fallback function to call on failure
        fallback_value: Static fallback value
        exceptions: Exceptions to catch

    Example:
        @with_graceful_degradation(fallback=get_cached_data, fallback_value=[])
        async def fetch_fresh_data():
            # Operation that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await global_degradation.try_with_fallback(
                operation=lambda: func(*args, **kwargs),
                fallback=fallback,
                fallback_value=fallback_value,
                exceptions=exceptions
            )

        return wrapper
    return decorator
