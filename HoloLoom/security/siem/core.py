"""
Core SIEM Integration for HoloLoom Security Pipeline

Provides async log forwarding with buffering, retry logic, and circuit breaker.

Created: 2025-11-15
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from enum import Enum

from .taxonomy import SecurityEvent


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # SIEM unavailable, using fallback
    HALF_OPEN = "half_open"  # Testing if SIEM recovered


@dataclass
class SIEMConfig:
    """Configuration for SIEM integration"""

    # Backend selection
    backend: str = "file"  # "splunk", "elk", "datadog", "file"

    # Buffering
    buffer_size: int = 1000  # Max events in buffer
    flush_interval: float = 10.0  # Seconds between flushes
    batch_size: int = 100  # Max events per batch

    # Retry logic
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay in seconds
    retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Circuit breaker
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2  # Successes needed to close circuit

    # Retention
    retention_days: int = 90
    hot_storage_days: int = 7  # Keep in fast storage

    # Fallback
    fallback_dir: Path = field(default_factory=lambda: Path("./security_logs"))
    enable_fallback: bool = True

    # Backend-specific configs
    splunk_config: Dict[str, Any] = field(default_factory=dict)
    elk_config: Dict[str, Any] = field(default_factory=dict)
    datadog_config: Dict[str, Any] = field(default_factory=dict)


class SIEMBackend(Protocol):
    """Protocol for SIEM backend implementations"""

    async def send_events(self, events: List[SecurityEvent]) -> bool:
        """
        Send events to SIEM backend

        Args:
            events: List of security events

        Returns:
            True if successful, False otherwise
        """
        ...

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query events from SIEM backend

        Args:
            start_time: Start of time range
            end_time: End of time range
            filters: Optional filters (category, severity, etc.)

        Returns:
            List of matching events
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if SIEM backend is healthy

        Returns:
            True if healthy, False otherwise
        """
        ...


class LogBuffer:
    """Thread-safe buffer for security events"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._dropped_count = 0

    async def add(self, event: SecurityEvent) -> None:
        """Add event to buffer"""
        async with self._lock:
            if len(self._buffer) >= self.max_size:
                self._dropped_count += 1
                logger.warning(f"Buffer full, dropping event (total dropped: {self._dropped_count})")
            self._buffer.append(event)

    async def get_batch(self, batch_size: int) -> List[SecurityEvent]:
        """Get a batch of events"""
        async with self._lock:
            batch = []
            for _ in range(min(batch_size, len(self._buffer))):
                if self._buffer:
                    batch.append(self._buffer.popleft())
            return batch

    async def size(self) -> int:
        """Get current buffer size"""
        async with self._lock:
            return len(self._buffer)

    async def clear(self) -> None:
        """Clear buffer"""
        async with self._lock:
            self._buffer.clear()


class CircuitBreaker:
    """Circuit breaker for SIEM backend failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

    def record_success(self) -> None:
        """Record successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()

    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_try_half_open():
                self._half_open_circuit()
                return True
            return False

        # HALF_OPEN
        return True

    def _open_circuit(self) -> None:
        """Open circuit (SIEM unavailable)"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.error("Circuit breaker OPEN - SIEM backend unavailable, using fallback")

    def _half_open_circuit(self) -> None:
        """Half-open circuit (testing recovery)"""
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker HALF_OPEN - testing SIEM backend recovery")

    def _close_circuit(self) -> None:
        """Close circuit (SIEM recovered)"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker CLOSED - SIEM backend recovered")

    def _should_try_half_open(self) -> bool:
        """Check if enough time has passed to try half-open"""
        if not self.last_failure_time:
            return True
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout


class FileFallbackBackend:
    """File-based fallback when SIEM unavailable"""

    def __init__(self, fallback_dir: Path):
        self.fallback_dir = fallback_dir
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

    async def write_events(self, events: List[SecurityEvent]) -> None:
        """Write events to JSON file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.fallback_dir / f"security_events_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump([e.to_dict() for e in events], f, indent=2)
            logger.info(f"Wrote {len(events)} events to fallback file: {filename}")
        except Exception as e:
            logger.error(f"Failed to write fallback file: {e}")


class SIEMIntegration:
    """
    Main SIEM integration orchestrator

    Provides async log forwarding with buffering, retry logic, and circuit breaker.
    """

    def __init__(self, config: SIEMConfig, backend: Optional[SIEMBackend] = None):
        self.config = config
        self.backend = backend
        self.buffer = LogBuffer(max_size=config.buffer_size)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            success_threshold=config.success_threshold,
        )
        self.fallback = FileFallbackBackend(config.fallback_dir) if config.enable_fallback else None

        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_sent": 0,
            "events_failed": 0,
            "events_dropped": 0,
            "backend_failures": 0,
            "fallback_writes": 0,
        }

    async def start(self) -> None:
        """Start background flushing task"""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"SIEM integration started (backend: {self.config.backend})")

    async def stop(self) -> None:
        """Stop background flushing and flush remaining events"""
        if not self._running:
            return

        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_buffer()
        logger.info("SIEM integration stopped")

    async def log_event(self, event: SecurityEvent) -> None:
        """
        Log a security event

        Args:
            event: Security event to log
        """
        await self.buffer.add(event)
        self.stats["events_logged"] += 1

    async def log_events(self, events: List[SecurityEvent]) -> None:
        """
        Log multiple security events

        Args:
            events: List of security events
        """
        for event in events:
            await self.buffer.add(event)
        self.stats["events_logged"] += len(events)

    async def _flush_loop(self) -> None:
        """Background loop for periodic flushing"""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def _flush_buffer(self) -> None:
        """Flush buffer to SIEM backend"""
        buffer_size = await self.buffer.size()
        if buffer_size == 0:
            return

        # Get batch of events
        batch = await self.buffer.get_batch(self.config.batch_size)
        if not batch:
            return

        # Try to send to backend
        success = await self._send_with_retry(batch)

        if success:
            self.stats["events_sent"] += len(batch)
        else:
            self.stats["events_failed"] += len(batch)

            # Use fallback if enabled
            if self.fallback:
                await self.fallback.write_events(batch)
                self.stats["fallback_writes"] += len(batch)

    async def _send_with_retry(self, events: List[SecurityEvent]) -> bool:
        """
        Send events with retry logic

        Args:
            events: Events to send

        Returns:
            True if successful, False otherwise
        """
        if not self.backend:
            logger.warning("No SIEM backend configured, using fallback")
            return False

        if not self.circuit_breaker.can_attempt():
            logger.debug("Circuit breaker OPEN, skipping send attempt")
            return False

        for attempt in range(self.config.max_retries):
            try:
                success = await self.backend.send_events(events)
                if success:
                    self.circuit_breaker.record_success()
                    return True
                else:
                    self.circuit_breaker.record_failure()
                    self.stats["backend_failures"] += 1

            except Exception as e:
                logger.error(f"SIEM backend error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                self.circuit_breaker.record_failure()
                self.stats["backend_failures"] += 1

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                await asyncio.sleep(delay)

        return False

    async def query(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query events from SIEM backend

        Args:
            start_time: Start of time range
            end_time: End of time range
            filters: Optional filters

        Returns:
            List of matching events
        """
        if not self.backend:
            logger.warning("No SIEM backend configured for querying")
            return []

        try:
            return await self.backend.query_events(start_time, end_time, filters)
        except Exception as e:
            logger.error(f"Failed to query SIEM backend: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """
        Check system health

        Returns:
            Health status dict
        """
        backend_healthy = False
        if self.backend:
            try:
                backend_healthy = await self.backend.health_check()
            except Exception as e:
                logger.error(f"Backend health check failed: {e}")

        return {
            "backend_healthy": backend_healthy,
            "circuit_state": self.circuit_breaker.state.value,
            "buffer_size": await self.buffer.size(),
            "buffer_max_size": self.config.buffer_size,
            "stats": self.stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            **self.stats,
            "circuit_state": self.circuit_breaker.state.value,
            "buffer_size": self.buffer._buffer.__len__(),
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
