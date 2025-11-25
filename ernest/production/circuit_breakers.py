"""
Ernest Production Circuit Breakers
===================================
Production-grade fault tolerance for Ernest creative writing system.

Features:
- Rate limiting (refinements per minute)
- Circuit breakers (auto-disable on errors)
- Graceful degradation
- Integration with HoloLoom's production hardening

Philosophy: "Fail gracefully. Protect the writer's flow."
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
from datetime import datetime, timedelta


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Tripped, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_refinements_per_minute: int = 30
    max_learning_updates_per_minute: int = 10
    max_parallel_passes: int = 4
    burst_allowance: int = 5  # Allow short bursts


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Trip after 5 failures
    recovery_timeout: float = 60.0  # 60 seconds
    success_threshold: int = 2  # Need 2 successes to close
    enable_auto_recovery: bool = True


@dataclass
class RateLimitState:
    """Current rate limit state"""
    refinements_this_minute: int = 0
    learning_updates_this_minute: int = 0
    parallel_passes_active: int = 0
    minute_start: float = field(default_factory=time.time)

    def reset_if_new_minute(self):
        """Reset counters if we're in a new minute"""
        now = time.time()
        if now - self.minute_start >= 60.0:
            self.refinements_this_minute = 0
            self.learning_updates_this_minute = 0
            self.minute_start = now


@dataclass
class CircuitBreakerState:
    """Current circuit breaker state"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    trip_time: Optional[float] = None


class ErnestRateLimiter:
    """Rate limiter for Ernest operations"""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.state = RateLimitState()
        self.logger = logging.getLogger(f"{__name__}.ErnestRateLimiter")

    def check_refinement_allowed(self) -> tuple[bool, Optional[str]]:
        """Check if refinement is allowed under rate limits"""
        self.state.reset_if_new_minute()

        if self.state.refinements_this_minute >= self.config.max_refinements_per_minute:
            return False, f"Rate limit exceeded: {self.config.max_refinements_per_minute} refinements/minute"

        return True, None

    def check_learning_update_allowed(self) -> tuple[bool, Optional[str]]:
        """Check if learning update is allowed under rate limits"""
        self.state.reset_if_new_minute()

        if self.state.learning_updates_this_minute >= self.config.max_learning_updates_per_minute:
            return False, f"Rate limit exceeded: {self.config.max_learning_updates_per_minute} updates/minute"

        return True, None

    def check_parallel_pass_allowed(self) -> tuple[bool, Optional[str]]:
        """Check if starting a parallel pass is allowed"""
        if self.state.parallel_passes_active >= self.config.max_parallel_passes:
            return False, f"Parallel limit exceeded: {self.config.max_parallel_passes} concurrent passes"

        return True, None

    def record_refinement(self):
        """Record a refinement (increment counter)"""
        self.state.reset_if_new_minute()
        self.state.refinements_this_minute += 1

    def record_learning_update(self):
        """Record a learning update (increment counter)"""
        self.state.reset_if_new_minute()
        self.state.learning_updates_this_minute += 1

    def acquire_parallel_pass(self):
        """Acquire a parallel pass slot"""
        self.state.parallel_passes_active += 1

    def release_parallel_pass(self):
        """Release a parallel pass slot"""
        self.state.parallel_passes_active = max(0, self.state.parallel_passes_active - 1)

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        self.state.reset_if_new_minute()
        return {
            "refinements_this_minute": self.state.refinements_this_minute,
            "learning_updates_this_minute": self.state.learning_updates_this_minute,
            "parallel_passes_active": self.state.parallel_passes_active,
            "refinements_remaining": max(0, self.config.max_refinements_per_minute - self.state.refinements_this_minute),
            "learning_updates_remaining": max(0, self.config.max_learning_updates_per_minute - self.state.learning_updates_this_minute),
            "parallel_slots_remaining": max(0, self.config.max_parallel_passes - self.state.parallel_passes_active)
        }


class ErnestCircuitBreaker:
    """Circuit breaker for Ernest learning engine"""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self.logger = logging.getLogger(f"{__name__}.ErnestCircuitBreaker")

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        if self.state.state == CircuitState.OPEN:
            # Check if we should try recovery
            if self.config.enable_auto_recovery and self.state.trip_time:
                elapsed = time.time() - self.state.trip_time
                if elapsed >= self.config.recovery_timeout:
                    self.logger.info("Circuit breaker entering HALF_OPEN state for recovery testing")
                    self.state.state = CircuitState.HALF_OPEN
                    self.state.success_count = 0
                    return False
            return True

        return False

    def record_success(self):
        """Record a successful operation"""
        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1
            self.logger.info(f"Circuit breaker success in HALF_OPEN: {self.state.success_count}/{self.config.success_threshold}")

            if self.state.success_count >= self.config.success_threshold:
                self.logger.info("Circuit breaker CLOSED after successful recovery")
                self.state.state = CircuitState.CLOSED
                self.state.failure_count = 0
                self.state.success_count = 0
                self.state.trip_time = None
        elif self.state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.state.failure_count = 0

    def record_failure(self, error: Exception):
        """Record a failed operation"""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()

        self.logger.warning(
            f"Circuit breaker failure: {self.state.failure_count}/{self.config.failure_threshold} "
            f"(error: {error})"
        )

        if self.state.state == CircuitState.HALF_OPEN:
            # Immediate trip on failure during recovery
            self.logger.error("Circuit breaker tripped during HALF_OPEN recovery attempt")
            self._trip()
        elif self.state.state == CircuitState.CLOSED:
            if self.state.failure_count >= self.config.failure_threshold:
                self.logger.error(f"Circuit breaker TRIPPED after {self.state.failure_count} failures")
                self._trip()

    def _trip(self):
        """Trip the circuit breaker"""
        self.state.state = CircuitState.OPEN
        self.state.trip_time = time.time()
        self.state.success_count = 0

    def reset(self):
        """Manually reset circuit breaker"""
        self.logger.info("Circuit breaker manually reset to CLOSED state")
        self.state.state = CircuitState.CLOSED
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.trip_time = None

    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        stats = {
            "state": self.state.state.value,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count
        }

        if self.state.trip_time:
            elapsed = time.time() - self.state.trip_time
            stats["time_since_trip"] = elapsed
            stats["recovery_in"] = max(0, self.config.recovery_timeout - elapsed)

        if self.state.last_failure_time:
            stats["time_since_last_failure"] = time.time() - self.state.last_failure_time

        return stats


class ErnestProductionGuard:
    """Combined production guard with rate limiting + circuit breakers"""

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_graceful_degradation: bool = True
    ):
        self.rate_limiter = ErnestRateLimiter(rate_limit_config)
        self.circuit_breaker = ErnestCircuitBreaker(circuit_breaker_config)
        self.enable_graceful_degradation = enable_graceful_degradation
        self.logger = logging.getLogger(f"{__name__}.ErnestProductionGuard")

    async def guard_refinement(self, refinement_func, *args, **kwargs):
        """Guard a refinement operation with rate limiting + circuit breaker"""
        # Check rate limit
        allowed, reason = self.rate_limiter.check_refinement_allowed()
        if not allowed:
            if self.enable_graceful_degradation:
                self.logger.warning(f"Rate limit exceeded, skipping refinement: {reason}")
                return None  # Graceful degradation: return original
            raise RateLimitExceeded(reason)

        # Check circuit breaker
        if self.circuit_breaker.is_open():
            if self.enable_graceful_degradation:
                self.logger.warning("Circuit breaker OPEN, skipping refinement")
                return None  # Graceful degradation: return original
            raise CircuitBreakerOpen("Learning engine circuit breaker is OPEN")

        # Record refinement
        self.rate_limiter.record_refinement()

        # Execute with circuit breaker protection
        try:
            result = await refinement_func(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            if self.enable_graceful_degradation:
                self.logger.error(f"Refinement failed, returning None: {e}")
                return None
            raise

    async def guard_learning_update(self, update_func, *args, **kwargs):
        """Guard a learning update with rate limiting + circuit breaker"""
        # Check rate limit
        allowed, reason = self.rate_limiter.check_learning_update_allowed()
        if not allowed:
            self.logger.warning(f"Learning update rate limit exceeded: {reason}")
            return  # Silent skip for background learning

        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.logger.warning("Circuit breaker OPEN, skipping learning update")
            return

        # Record update
        self.rate_limiter.record_learning_update()

        # Execute with circuit breaker protection
        try:
            result = await update_func(*args, **kwargs)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            self.logger.error(f"Learning update failed: {e}")
            # Always silent skip for background learning
            return

    def get_health_status(self) -> Dict:
        """Get comprehensive health status"""
        rate_stats = self.rate_limiter.get_stats()
        circuit_stats = self.circuit_breaker.get_stats()

        # Determine overall health
        circuit_state = circuit_stats["state"]
        refinements_remaining = rate_stats["refinements_remaining"]

        if circuit_state == "open":
            health = "critical"
        elif circuit_state == "half_open":
            health = "degraded"
        elif refinements_remaining < 5:
            health = "warning"
        else:
            health = "healthy"

        return {
            "health": health,
            "rate_limiter": rate_stats,
            "circuit_breaker": circuit_stats,
            "graceful_degradation_enabled": self.enable_graceful_degradation
        }


# Exceptions
class RateLimitExceeded(Exception):
    """Rate limit exceeded"""
    pass


class CircuitBreakerOpen(Exception):
    """Circuit breaker is open"""
    pass


# Quick helpers
def create_production_guard(
    max_refinements_per_minute: int = 30,
    enable_graceful_degradation: bool = True
) -> ErnestProductionGuard:
    """Create production guard with sensible defaults"""
    rate_config = RateLimitConfig(max_refinements_per_minute=max_refinements_per_minute)
    circuit_config = CircuitBreakerConfig()
    return ErnestProductionGuard(rate_config, circuit_config, enable_graceful_degradation)
