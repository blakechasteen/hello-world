#!/usr/bin/env python3
"""
Retry Processor for Promptly Chains

Implements exponential backoff, circuit breaker pattern, and fallback strategies.
"""

import time
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from ..base import BaseChainStepProcessor


class BackoffStrategy(Enum):
    """Backoff strategies for retries"""
    CONSTANT = "constant"           # Fixed delay
    LINEAR = "linear"               # Linear increase
    EXPONENTIAL = "exponential"     # Exponential backoff
    FIBONACCI = "fibonacci"         # Fibonacci sequence
    JITTER = "jitter"              # Exponential with random jitter


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        window_size: int = 10
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.window_size = window_size

        self.state = CircuitState.CLOSED
        self.failures = deque(maxlen=window_size)
        self.successes = 0
        self.last_failure_time: Optional[datetime] = None

    def record_success(self) -> None:
        """Record successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Remove oldest failure if any
            if self.failures:
                self.failures.popleft()

    def record_failure(self) -> None:
        """Record failed execution"""
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self._open_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failures.append(datetime.now())
            if len(self.failures) >= self.failure_threshold:
                self._open_circuit()

    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self._half_open_circuit()
                    return True
            return False

        # HALF_OPEN state
        return True

    def _open_circuit(self) -> None:
        """Open the circuit (reject requests)"""
        self.state = CircuitState.OPEN
        self.successes = 0

    def _close_circuit(self) -> None:
        """Close the circuit (normal operation)"""
        self.state = CircuitState.CLOSED
        self.successes = 0
        self.failures.clear()

    def _half_open_circuit(self) -> None:
        """Half-open the circuit (testing recovery)"""
        self.state = CircuitState.HALF_OPEN
        self.successes = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit state"""
        return {
            "state": self.state.value,
            "failure_count": len(self.failures),
            "success_count": self.successes,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryProcessor(BaseChainStepProcessor):
    """
    Execute with retry logic and circuit breaker.

    Configuration format:
    {
        "type": "retry",
        "max_attempts": 3,
        "backoff_strategy": "constant|linear|exponential|fibonacci|jitter",
        "initial_delay": 1.0,       # seconds
        "max_delay": 60.0,          # seconds
        "backoff_multiplier": 2.0,  # for exponential
        "jitter_range": 0.1,        # for jitter (0.0-1.0)
        "retry_on": [...],          # conditions to retry (errors, patterns)
        "fallback": {...},          # fallback action on exhaustion
        "circuit_breaker": {
            "enabled": true,
            "failure_threshold": 5,
            "success_threshold": 2,
            "timeout": 60.0
        },
        "rate_limit": {
            "enabled": true,
            "requests_per_second": 10,
            "burst_size": 20
        },
        "action": {...}             # action to execute with retries
    }
    """

    def __init__(self, executor: Optional[Callable] = None):
        super().__init__(
            name="retry",
            description="Execute with retry logic and circuit breaker"
        )
        self._executor = executor
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiters: Dict[str, "RateLimiter"] = {}

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with retry logic.

        Args:
            step_input: Input data
            step_config: Retry configuration

        Returns:
            Result from successful execution or fallback
        """
        max_attempts = step_config.get("max_attempts", 3)
        backoff_strategy = step_config.get("backoff_strategy", "exponential")
        initial_delay = step_config.get("initial_delay", 1.0)
        max_delay = step_config.get("max_delay", 60.0)
        action = step_config.get("action", {})

        # Circuit breaker setup
        circuit_config = step_config.get("circuit_breaker", {})
        circuit_key = step_config.get("circuit_key", "default")

        if circuit_config.get("enabled", False):
            if circuit_key not in self._circuit_breakers:
                self._circuit_breakers[circuit_key] = CircuitBreaker(
                    failure_threshold=circuit_config.get("failure_threshold", 5),
                    success_threshold=circuit_config.get("success_threshold", 2),
                    timeout=circuit_config.get("timeout", 60.0),
                    window_size=circuit_config.get("window_size", 10)
                )

            circuit = self._circuit_breakers[circuit_key]

            # Check if circuit allows execution
            if not circuit.can_execute():
                return self._execute_fallback(step_input, step_config, "circuit_open")

        # Rate limiting
        rate_config = step_config.get("rate_limit", {})
        if rate_config.get("enabled", False):
            rate_key = step_config.get("rate_key", "default")
            if rate_key not in self._rate_limiters:
                self._rate_limiters[rate_key] = RateLimiter(
                    requests_per_second=rate_config.get("requests_per_second", 10),
                    burst_size=rate_config.get("burst_size", 20)
                )

            rate_limiter = self._rate_limiters[rate_key]
            if not rate_limiter.can_proceed():
                return self._execute_fallback(step_input, step_config, "rate_limited")

        # Retry loop
        attempts = 0
        last_error = None
        retry_history = []

        while attempts < max_attempts:
            try:
                attempts += 1

                # Calculate delay for this attempt
                if attempts > 1:
                    delay = self._calculate_delay(
                        attempt=attempts - 1,
                        strategy=backoff_strategy,
                        initial_delay=initial_delay,
                        max_delay=max_delay,
                        multiplier=step_config.get("backoff_multiplier", 2.0),
                        jitter_range=step_config.get("jitter_range", 0.1)
                    )
                    time.sleep(delay)
                    retry_history.append({"attempt": attempts, "delay": delay})

                # Execute action
                result = self._execute_action(action, step_input)

                # Check if result indicates failure
                if self._should_retry(result, step_config):
                    last_error = result.get("error", "unknown_error")
                    retry_history.append({
                        "attempt": attempts,
                        "error": last_error,
                        "retryable": True
                    })

                    # Record failure for circuit breaker
                    if circuit_config.get("enabled", False):
                        circuit.record_failure()

                    continue

                # Success!
                if circuit_config.get("enabled", False):
                    circuit.record_success()

                return {
                    **step_input,
                    "result": result,
                    "attempts": attempts,
                    "retry_history": retry_history,
                    "success": True
                }

            except Exception as e:
                last_error = str(e)
                retry_history.append({
                    "attempt": attempts,
                    "error": last_error,
                    "exception": True
                })

                # Record failure for circuit breaker
                if circuit_config.get("enabled", False):
                    circuit.record_failure()

                # Check if should retry this error
                if not self._should_retry_error(e, step_config):
                    break

        # All attempts exhausted - execute fallback
        return self._execute_fallback(
            step_input,
            step_config,
            last_error,
            attempts=attempts,
            retry_history=retry_history
        )

    def _calculate_delay(
        self,
        attempt: int,
        strategy: str,
        initial_delay: float,
        max_delay: float,
        multiplier: float,
        jitter_range: float
    ) -> float:
        """Calculate retry delay based on strategy"""
        import random

        if strategy == BackoffStrategy.CONSTANT.value:
            delay = initial_delay

        elif strategy == BackoffStrategy.LINEAR.value:
            delay = initial_delay * (attempt + 1)

        elif strategy == BackoffStrategy.EXPONENTIAL.value:
            delay = initial_delay * (multiplier ** attempt)

        elif strategy == BackoffStrategy.FIBONACCI.value:
            # Fibonacci sequence for delays
            fib = [1, 1]
            for i in range(2, attempt + 2):
                fib.append(fib[i-1] + fib[i-2])
            delay = initial_delay * fib[min(attempt + 1, len(fib) - 1)]

        elif strategy == BackoffStrategy.JITTER.value:
            # Exponential with random jitter
            base_delay = initial_delay * (multiplier ** attempt)
            jitter = random.uniform(-jitter_range, jitter_range) * base_delay
            delay = base_delay + jitter

        else:
            delay = initial_delay

        # Cap at max_delay
        return min(delay, max_delay)

    def _should_retry(self, result: Dict[str, Any], step_config: Dict[str, Any]) -> bool:
        """Check if result indicates a retryable failure"""
        retry_on = step_config.get("retry_on", [])

        if not retry_on:
            # Default: retry on error field
            return "error" in result and result["error"] is not None

        # Check retry conditions
        for condition in retry_on:
            if isinstance(condition, str):
                # Simple error pattern matching
                error = result.get("error", "")
                if condition in str(error):
                    return True
            elif isinstance(condition, dict):
                # Structured condition
                field = condition.get("field", "error")
                pattern = condition.get("pattern", "")
                value = result.get(field)

                if pattern and pattern in str(value):
                    return True

        return False

    def _should_retry_error(self, error: Exception, step_config: Dict[str, Any]) -> bool:
        """Check if exception is retryable"""
        retry_on = step_config.get("retry_on", [])

        if not retry_on:
            # Default: retry all exceptions
            return True

        # Check if error type matches retry conditions
        error_type = type(error).__name__
        error_msg = str(error)

        for condition in retry_on:
            if isinstance(condition, str):
                if condition in error_type or condition in error_msg:
                    return True

        return True  # Default to retry

    def _execute_action(self, action: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action"""
        if self._executor:
            return self._executor(action, data)
        else:
            # Simple execution
            return {"result": action.get("value"), "success": True}

    def _execute_fallback(
        self,
        step_input: Dict[str, Any],
        step_config: Dict[str, Any],
        error: Any,
        attempts: int = 0,
        retry_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute fallback action"""
        fallback = step_config.get("fallback", {})

        if fallback:
            fallback_result = self._execute_action(fallback, step_input)
            return {
                **step_input,
                "result": fallback_result,
                "fallback_used": True,
                "original_error": error,
                "attempts": attempts,
                "retry_history": retry_history or [],
                "success": False
            }
        else:
            # No fallback - return error
            return {
                **step_input,
                "result": None,
                "error": error,
                "attempts": attempts,
                "retry_history": retry_history or [],
                "success": False
            }

    def get_circuit_state(self, circuit_key: str = "default") -> Dict[str, Any]:
        """Get circuit breaker state"""
        if circuit_key in self._circuit_breakers:
            return self._circuit_breakers[circuit_key].get_state()
        return {"state": "not_initialized"}

    def reset_circuit(self, circuit_key: str = "default") -> None:
        """Reset circuit breaker"""
        if circuit_key in self._circuit_breakers:
            self._circuit_breakers[circuit_key]._close_circuit()


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests_per_second: float, burst_size: int):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()

    def can_proceed(self) -> bool:
        """Check if request can proceed"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

        # Check if token available
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True

        return False


# Export processors
__all__ = ["RetryProcessor", "CircuitBreaker", "RateLimiter", "BackoffStrategy", "CircuitState"]
