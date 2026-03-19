"""
Background Learning Loop for CARTS
==================================

Async background learning loop that continuously updates learning systems
based on attack outcomes. Runs in a separate thread/task without blocking
the main red team orchestration.

Features:
- Configurable update intervals
- Graceful start/stop with proper cleanup
- Heat decay application
- Prior consolidation
- Pattern mining from logs

Philosophy: "Learn while you work."

Author: CARTS Team
Date: 2025-12-03
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================

class UpdateableProtocol(Protocol):
    """Protocol for components that can receive background updates."""

    def apply_decay(self) -> int:
        """Apply time-based decay. Returns count of affected items."""
        ...

    def save(self, path: Path) -> None:
        """Save state to file."""
        ...


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LearningEvent:
    """
    Record of a learning event for background processing.

    Attributes:
        timestamp: When the event occurred
        event_type: Type of event (attack_result, decay_tick, etc.)
        data: Event-specific data
    """
    timestamp: str
    event_type: str
    data: dict[str, Any]

    @classmethod
    def attack_result(
        cls,
        strategy: str,
        success: bool,
        severity: float = 0.0,
        payload_id: str | None = None,
        context: dict[str, str] | None = None,
    ) -> 'LearningEvent':
        """Create an attack result event."""
        return cls(
            timestamp=datetime.now().isoformat(),
            event_type='attack_result',
            data={
                'strategy': strategy,
                'success': success,
                'severity': severity,
                'payload_id': payload_id,
                'context': context,
            }
        )

    @classmethod
    def decay_tick(cls) -> 'LearningEvent':
        """Create a decay tick event."""
        return cls(
            timestamp=datetime.now().isoformat(),
            event_type='decay_tick',
            data={}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'data': self.data,
        }


@dataclass
class BackgroundLearnerConfig:
    """
    Configuration for background learner.

    Attributes:
        update_interval: Seconds between learning updates
        decay_interval: Seconds between decay applications
        save_interval: Seconds between state saves
        event_buffer_size: Max events to buffer before processing
        log_path: Path for event logging
        state_path: Path for state persistence
        enable_logging: Whether to log events to file
    """
    update_interval: float = 60.0
    decay_interval: float = 300.0  # 5 minutes
    save_interval: float = 600.0   # 10 minutes
    event_buffer_size: int = 100
    log_path: Path | None = None
    state_path: Path | None = None
    enable_logging: bool = True


@dataclass
class BackgroundLearnerStats:
    """
    Statistics from background learner.

    Attributes:
        total_events_processed: Total events processed
        total_update_cycles: Number of update cycles completed
        total_decay_cycles: Number of decay cycles completed
        last_update: Timestamp of last update
        last_decay: Timestamp of last decay
        last_save: Timestamp of last save
        is_running: Whether the learner is running
        uptime_seconds: Time since start
    """
    total_events_processed: int = 0
    total_update_cycles: int = 0
    total_decay_cycles: int = 0
    last_update: str | None = None
    last_decay: str | None = None
    last_save: str | None = None
    is_running: bool = False
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_events_processed': self.total_events_processed,
            'total_update_cycles': self.total_update_cycles,
            'total_decay_cycles': self.total_decay_cycles,
            'last_update': self.last_update,
            'last_decay': self.last_decay,
            'last_save': self.last_save,
            'is_running': self.is_running,
            'uptime_seconds': self.uptime_seconds,
        }


# =============================================================================
# Main Class
# =============================================================================

class BackgroundLearner:
    """
    Async background learning loop for CARTS.

    Runs continuous learning updates without blocking the main
    red team orchestration loop.

    Example:
        learner = BackgroundLearner(
            config=BackgroundLearnerConfig(update_interval=60.0),
            on_update=my_update_callback,
        )

        # Start background learning
        await learner.start()

        # Queue events for processing
        learner.queue_event(LearningEvent.attack_result("UNICODE_BYPASS", True, 0.8))

        # Stop gracefully
        await learner.stop()
    """

    def __init__(
        self,
        config: BackgroundLearnerConfig | None = None,
        on_update: Callable[[list[LearningEvent]], None] | None = None,
        on_decay: Callable[[], None] | None = None,
        on_save: Callable[[Path], None] | None = None,
        updateables: list[UpdateableProtocol] | None = None,
    ):
        """
        Initialize background learner.

        Args:
            config: Learner configuration
            on_update: Callback for update cycles (receives buffered events)
            on_decay: Callback for decay cycles
            on_save: Callback for save cycles
            updateables: Components implementing UpdateableProtocol
        """
        self.config = config or BackgroundLearnerConfig()
        self.on_update = on_update
        self.on_decay = on_decay
        self.on_save = on_save
        self.updateables = updateables or []

        self._event_buffer: list[LearningEvent] = []
        self._stats = BackgroundLearnerStats()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._start_time: datetime | None = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start background learning tasks."""
        if self._running:
            logger.warning("Background learner already running")
            return

        self._running = True
        self._stats.is_running = True
        self._start_time = datetime.now()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._update_loop()),
            asyncio.create_task(self._decay_loop()),
            asyncio.create_task(self._save_loop()),
        ]

        logger.info("Background learner started")

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop background learning gracefully.

        Args:
            timeout: Maximum seconds to wait for tasks to complete
        """
        if not self._running:
            return

        self._running = False
        self._stats.is_running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)

        self._tasks.clear()

        # Final save
        await self._do_save()

        logger.info("Background learner stopped")

    async def __aenter__(self) -> 'BackgroundLearner':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    # -------------------------------------------------------------------------
    # Event Queue
    # -------------------------------------------------------------------------

    def queue_event(self, event: LearningEvent) -> None:
        """
        Queue a learning event for background processing.

        Thread-safe event queuing.

        Args:
            event: Learning event to queue
        """
        self._event_buffer.append(event)

        # Log event if enabled
        if self.config.enable_logging and self.config.log_path:
            self._log_event(event)

        # Process immediately if buffer full
        if len(self._event_buffer) >= self.config.event_buffer_size:
            asyncio.create_task(self._process_buffer())

    def queue_attack_result(
        self,
        strategy: str,
        success: bool,
        severity: float = 0.0,
        payload_id: str | None = None,
        context: dict[str, str] | None = None,
    ) -> None:
        """
        Convenience method to queue an attack result.

        Args:
            strategy: Attack strategy name
            success: Whether the attack succeeded
            severity: Severity if successful
            payload_id: Optional payload identifier
            context: Optional context (target_type, vuln_class, defense_level)
        """
        event = LearningEvent.attack_result(
            strategy=strategy,
            success=success,
            severity=severity,
            payload_id=payload_id,
            context=context,
        )
        self.queue_event(event)

    # -------------------------------------------------------------------------
    # Background Loops
    # -------------------------------------------------------------------------

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                await self._process_buffer()
                self._stats.total_update_cycles += 1
                self._stats.last_update = datetime.now().isoformat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update loop error: {e}")

    async def _decay_loop(self) -> None:
        """Decay application loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.decay_interval)
                await self._do_decay()
                self._stats.total_decay_cycles += 1
                self._stats.last_decay = datetime.now().isoformat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decay loop error: {e}")

    async def _save_loop(self) -> None:
        """Periodic save loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.save_interval)
                await self._do_save()
                self._stats.last_save = datetime.now().isoformat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Save loop error: {e}")

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    async def _process_buffer(self) -> None:
        """Process buffered events."""
        async with self._lock:
            if not self._event_buffer:
                return

            # Take all events from buffer
            events = self._event_buffer.copy()
            self._event_buffer.clear()

        # Call update callback
        if self.on_update:
            try:
                self.on_update(events)
            except Exception as e:
                logger.error(f"Update callback error: {e}")

        self._stats.total_events_processed += len(events)

    async def _do_decay(self) -> None:
        """Apply decay to all updateable components."""
        total_affected = 0

        for updateable in self.updateables:
            try:
                affected = updateable.apply_decay()
                total_affected += affected
            except Exception as e:
                logger.error(f"Decay error for {updateable}: {e}")

        # Call decay callback
        if self.on_decay:
            try:
                self.on_decay()
            except Exception as e:
                logger.error(f"Decay callback error: {e}")

        logger.debug(f"Decay applied to {total_affected} items")

    async def _do_save(self) -> None:
        """Save state of all updateable components."""
        if not self.config.state_path:
            return

        for updateable in self.updateables:
            try:
                updateable.save(self.config.state_path)
            except Exception as e:
                logger.error(f"Save error for {updateable}: {e}")

        # Call save callback
        if self.on_save and self.config.state_path:
            try:
                self.on_save(self.config.state_path)
            except Exception as e:
                logger.error(f"Save callback error: {e}")

    def _log_event(self, event: LearningEvent) -> None:
        """Log event to file."""
        if not self.config.log_path:
            return

        try:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Event logging error: {e}")

    # -------------------------------------------------------------------------
    # Stats & Management
    # -------------------------------------------------------------------------

    def get_stats(self) -> BackgroundLearnerStats:
        """Get current statistics."""
        if self._start_time:
            self._stats.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        return self._stats

    def get_buffered_events(self) -> int:
        """Get number of buffered events."""
        return len(self._event_buffer)

    def is_running(self) -> bool:
        """Check if learner is running."""
        return self._running

    def add_updateable(self, updateable: UpdateableProtocol) -> None:
        """Add an updateable component."""
        self.updateables.append(updateable)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_background_learner(
    update_interval: float = 60.0,
    decay_interval: float = 300.0,
    save_interval: float = 600.0,
    log_path: str | None = None,
    state_path: str | None = None,
    on_update: Callable[[list[LearningEvent]], None] | None = None,
) -> BackgroundLearner:
    """
    Create a background learner with common configuration.

    Args:
        update_interval: Seconds between updates
        decay_interval: Seconds between decay applications
        save_interval: Seconds between saves
        log_path: Path for event logging
        state_path: Path for state persistence
        on_update: Callback for update cycles

    Returns:
        Configured BackgroundLearner
    """
    config = BackgroundLearnerConfig(
        update_interval=update_interval,
        decay_interval=decay_interval,
        save_interval=save_interval,
        log_path=Path(log_path) if log_path else None,
        state_path=Path(state_path) if state_path else None,
    )

    return BackgroundLearner(
        config=config,
        on_update=on_update,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    'LearningEvent',
    'BackgroundLearnerConfig',
    'BackgroundLearnerStats',

    # Protocols
    'UpdateableProtocol',

    # Main Class
    'BackgroundLearner',

    # Convenience Functions
    'create_background_learner',
]
