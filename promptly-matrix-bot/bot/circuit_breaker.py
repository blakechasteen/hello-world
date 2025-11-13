"""
Circuit Breaker Pattern for Promptly Matrix Bot

Protects against cascading failures by:
- Opening circuit after consecutive failures
- Half-open state for recovery testing
- Automatic circuit reset after timeout

Usage:
    from bot.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(
        failure_threshold=5,
        timeout=60.0,
        expected_exception=Exception
    )

    @breaker
    async def call_external_service():
        # Service call that might fail
        pass
"""

import asyncio
import logging
import time
from typing import Callable, Optional, Type, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 60.0  # Seconds before attempting recovery
    expected_exception: Type[Exception] = Exception
    # Track recent calls for metrics
    call_history_size: int = 100


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    open_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))

    def success_rate(self) -> float:
        """Calculate success rate from recent calls"""
        if not self.recent_calls:
            return 0.0
        successes = sum(1 for call in self.recent_calls if call["success"])
        return successes / len(self.recent_calls)

    def avg_response_time(self) -> float:
        """Calculate average response time from recent calls"""
        if not self.recent_calls:
            return 0.0
        total_time = sum(call["duration"] for call in self.recent_calls)
        return total_time / len(self.recent_calls)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation

    States:
    - CLOSED: Normal operation, all calls allowed
    - OPEN: Circuit open after failures, reject all calls
    - HALF_OPEN: Test if service recovered, allow single call

    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After timeout seconds
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: If test call fails
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        call_history_size: int = 100
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            call_history_size=call_history_size
        )

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.open_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.last_state_change_time = time.time()
        self.recent_calls = deque(maxlen=call_history_size)
        self._lock = asyncio.Lock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If func fails
        """
        async with self._lock:
            self.total_calls += 1

            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.timeout:
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN (timeout elapsed)")
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    # Circuit still open
                    logger.warning(f"Circuit breaker '{self.name}' is OPEN. Rejecting call.")
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open. "
                        f"Retry after {self.config.timeout - (time.time() - self.last_failure_time):.1f}s"
                    )

        # Execute function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # Record success
            await self._on_success(duration)

            return result

        except self.config.expected_exception as e:
            duration = time.time() - start_time

            # Record failure
            await self._on_failure(duration, e)

            raise

    async def _on_success(self, duration: float):
        """Handle successful call"""
        async with self._lock:
            self.success_count += 1
            self.last_success_time = time.time()
            self.recent_calls.append({
                "success": True,
                "duration": duration,
                "timestamp": time.time()
            })

            if self.state == CircuitState.HALF_OPEN:
                # Increment consecutive successes in half-open state
                self.failure_count = 0  # Reset failure count
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED (service recovered)")
                    self._transition_to(CircuitState.CLOSED)

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _on_failure(self, duration: float, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.recent_calls.append({
                "success": False,
                "duration": duration,
                "timestamp": time.time(),
                "error": str(exception)
            })

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure ({self.failure_count}/{self.config.failure_threshold}): {exception}"
            )

            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open state → back to open
                logger.warning(f"Circuit breaker '{self.name}' transitioning back to OPEN (recovery test failed)")
                self._transition_to(CircuitState.OPEN)

            elif self.state == CircuitState.CLOSED:
                # Check if we should open circuit
                if self.failure_count >= self.config.failure_threshold:
                    logger.error(
                        f"Circuit breaker '{self.name}' transitioning to OPEN "
                        f"(threshold {self.config.failure_threshold} reached)"
                    )
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = time.time()

        if new_state == CircuitState.OPEN:
            self.open_count += 1

        if new_state == CircuitState.CLOSED:
            # Reset counters
            self.failure_count = 0
            self.success_count = 0

        if new_state == CircuitState.HALF_OPEN:
            # Reset success count for half-open test
            self.success_count = 0

        logger.info(f"Circuit breaker '{self.name}' transitioned: {old_state.value} → {new_state.value}")

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics"""
        return CircuitBreakerStats(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            total_calls=self.total_calls,
            open_count=self.open_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            recent_calls=self.recent_calls
        )

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change_time = time.time()
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerRegistry:
    """
    Global registry of circuit breakers for monitoring

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("github", failure_threshold=5)

        # Get all breaker stats
        stats = registry.get_all_stats()
    """

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    expected_exception=expected_exception
                )
                logger.info(f"Created circuit breaker '{name}'")

            return self.breakers[name]

    def get_all_stats(self) -> dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    def get_health_status(self) -> dict:
        """
        Get overall health status

        Returns:
            {
                "healthy": bool,
                "total_breakers": int,
                "open_breakers": int,
                "breakers": {name: {state, success_rate, ...}}
            }
        """
        all_stats = self.get_all_stats()

        open_breakers = sum(
            1 for stats in all_stats.values()
            if stats.state == CircuitState.OPEN
        )

        breaker_details = {
            name: {
                "state": stats.state.value,
                "success_rate": stats.success_rate(),
                "avg_response_time": stats.avg_response_time(),
                "failure_count": stats.failure_count,
                "total_calls": stats.total_calls
            }
            for name, stats in all_stats.items()
        }

        return {
            "healthy": open_breakers == 0,
            "total_breakers": len(all_stats),
            "open_breakers": open_breakers,
            "breakers": breaker_details
        }


# Global registry
global_registry = CircuitBreakerRegistry()
