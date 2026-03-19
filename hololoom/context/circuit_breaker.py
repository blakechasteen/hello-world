"""
Circuit Breaker: Prevent Cascading Failures

Part 5: Production Hardening - Day 22

Implements circuit breaker pattern for external dependencies:
- MCP backend calls
- Neo4j graph queries
- Qdrant vector searches
- External enrichment services

States:
- CLOSED: Normal operation (all requests pass through)
- OPEN: Failure threshold exceeded (fail fast)
- HALF_OPEN: Testing recovery (limited requests)
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from hololoom.context.error_handling import CircuitBreakerOpenError

# ============================================================================
# Circuit States
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, blocking requests
    HALF_OPEN = "half_open"    # Testing recovery


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5        # Failures before opening
    recovery_timeout: float = 60.0    # Seconds before HALF_OPEN
    success_threshold: int = 2        # Successes in HALF_OPEN to close
    timeout: float = 5.0              # Request timeout (seconds)
    name: str = "default"             # Circuit breaker name


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

T = TypeVar('T')


class CircuitBreaker:
    """
    Circuit breaker for external dependencies

    Usage:
        breaker = CircuitBreaker(config)

        result = await breaker.call(risky_operation, arg1, arg2)

    The breaker will:
    - Track failures and open after threshold
    - Fail fast when open
    - Test recovery in half-open state
    - Close automatically after successful recovery
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker

        Args:
            config: Circuit breaker configuration (uses defaults if None)
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()

        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_rejections = 0  # Calls rejected while open

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function

        Raises:
            CircuitBreakerOpenError: If circuit is open
            asyncio.TimeoutError: If function times out
            Exception: Any exception from the function
        """
        self.total_calls += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.total_rejections += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.config.name}' is OPEN "
                    f"(failures: {self.failure_count}/{self.config.failure_threshold})"
                )

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            self._on_success()
            return result

        except asyncio.TimeoutError:
            self.total_timeouts += 1
            self._on_failure()
            raise

        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.last_state_change = time.time()

    def _on_success(self):
        """Handle successful call"""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open -> back to open
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_timeouts": self.total_timeouts,
            "total_rejections": self.total_rejections,
            "failure_rate": self.total_failures / max(1, self.total_calls),
            "rejection_rate": self.total_rejections / max(1, self.total_calls),
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_rejections = 0

    def force_open(self):
        """Manually open circuit (for testing/operations)"""
        self._transition_to_open()

    def force_close(self):
        """Manually close circuit (for testing/operations)"""
        self._transition_to_closed()


# ============================================================================
# Circuit Breaker Registry
# ============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers

    Allows creating circuit breakers for different backends:
    - MCP backend
    - Neo4j
    - Qdrant
    - Ollama enrichment
    """

    def __init__(self):
        """Initialize circuit breaker registry"""
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker by name

        Args:
            name: Circuit breaker name
            config: Configuration (uses defaults if None)

        Returns:
            CircuitBreaker instance
        """
        if name not in self.breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            else:
                config.name = name
            self.breakers[name] = CircuitBreaker(config)

        return self.breakers[name]

    def get_all_stats(self) -> dict:
        """
        Get statistics for all circuit breakers

        Returns:
            Dictionary mapping name -> stats
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    def get_health_summary(self) -> dict:
        """
        Get health summary for all circuit breakers

        Returns:
            Dictionary with overall health status
        """
        all_closed = all(
            breaker.state == CircuitState.CLOSED
            for breaker in self.breakers.values()
        )

        open_breakers = [
            name for name, breaker in self.breakers.items()
            if breaker.state == CircuitState.OPEN
        ]

        half_open_breakers = [
            name for name, breaker in self.breakers.items()
            if breaker.state == CircuitState.HALF_OPEN
        ]

        return {
            "healthy": all_closed,
            "total_breakers": len(self.breakers),
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "all_closed": all_closed
        }

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()


# ============================================================================
# Factory Functions
# ============================================================================

def create_circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2,
    timeout: float = 5.0
) -> CircuitBreaker:
    """
    Create circuit breaker with configuration

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before HALF_OPEN
        success_threshold: Successes to close from HALF_OPEN
        timeout: Request timeout in seconds

    Returns:
        Configured CircuitBreaker instance
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout
    )
    return CircuitBreaker(config)


def create_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Create circuit breaker registry

    Returns:
        CircuitBreakerRegistry instance
    """
    return CircuitBreakerRegistry()
