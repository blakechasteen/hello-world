"""
Event Replay - Event Sourcing Capability
=========================================

**Created**: December 3, 2025
**Purpose**: Replay stored events for recovery, debugging, and event sourcing

## Features

- Point-in-time replay (restore state to any timestamp)
- Selective replay (by topic, time range, correlation ID)
- Replay strategies (INSTANT, REALTIME, ACCELERATED)
- Integration with EventBroker for re-emission
- Progress tracking and callbacks

## Use Cases

1. **Disaster Recovery**: Replay events to rebuild state after failure
2. **Debugging**: Replay specific workflow to diagnose issues
3. **Testing**: Replay production events in test environment
4. **Auditing**: Review historical event sequences
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from skills.protocol import SkillEvent, EventType
from skills.event_bus.event_store import EventStore

logger = logging.getLogger(__name__)


# ============================================================================
# Replay Configuration
# ============================================================================

class ReplayStrategy(Enum):
    """How to pace event replay."""
    INSTANT = "instant"          # Replay all events as fast as possible
    REALTIME = "realtime"        # Replay at original timing
    ACCELERATED = "accelerated"  # Replay at N times speed


@dataclass
class ReplayConfig:
    """Configuration for event replay."""
    strategy: ReplayStrategy = ReplayStrategy.INSTANT
    speed_multiplier: float = 1.0    # For ACCELERATED strategy
    batch_size: int = 100            # Events per batch
    pause_between_batches_ms: int = 0
    emit_to_broker: bool = True      # Re-emit events to broker
    dry_run: bool = False            # Log but don't emit


@dataclass
class ReplayProgress:
    """Progress tracking for replay operations."""
    total_events: int = 0
    replayed_events: int = 0
    failed_events: int = 0
    skipped_events: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_event_timestamp: Optional[str] = None
    elapsed_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Get replay progress as percentage."""
        if self.total_events == 0:
            return 100.0
        return (self.replayed_events / self.total_events) * 100

    @property
    def is_complete(self) -> bool:
        """Check if replay is complete."""
        return self.replayed_events + self.failed_events + self.skipped_events >= self.total_events


# Type aliases
ReplayCallback = Callable[[SkillEvent, ReplayProgress], Awaitable[None]]
ReplayFilter = Callable[[SkillEvent], Awaitable[bool]]


# ============================================================================
# Event Replayer
# ============================================================================

class EventReplayer:
    """
    Replays events from EventStore.

    Supports:
    - Point-in-time replay
    - Selective replay by various criteria
    - Multiple replay strategies
    - Progress callbacks
    """

    def __init__(
        self,
        event_store: EventStore,
        broker: Optional[Any] = None,  # EventBroker
        config: Optional[ReplayConfig] = None
    ):
        """
        Initialize event replayer.

        Args:
            event_store: Source of events to replay
            broker: Optional EventBroker for re-emission
            config: Replay configuration
        """
        self.store = event_store
        self.broker = broker
        self.config = config or ReplayConfig()

        # State
        self._is_replaying = False
        self._should_stop = False
        self._progress = ReplayProgress()

        # Callbacks
        self._on_event_callbacks: List[ReplayCallback] = []
        self._on_complete_callbacks: List[Callable[[ReplayProgress], Awaitable[None]]] = []

        logger.info(f"EventReplayer created (strategy={self.config.strategy.value})")

    async def replay_time_range(
        self,
        start_time: str,
        end_time: Optional[str] = None,
        topic_pattern: Optional[str] = None,
        skill_name: Optional[str] = None,
        event_filter: Optional[ReplayFilter] = None
    ) -> ReplayProgress:
        """
        Replay events within a time range.

        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format), defaults to now
            topic_pattern: Optional topic filter (supports wildcards)
            skill_name: Optional skill name filter
            event_filter: Optional custom filter function

        Returns:
            ReplayProgress with results
        """
        if end_time is None:
            end_time = datetime.now().isoformat()

        # Count total events
        total = await self.store.count_events(
            topic_pattern=topic_pattern,
            skill_name=skill_name,
            start_time=start_time,
            end_time=end_time
        )

        self._progress = ReplayProgress(
            total_events=total,
            start_time=start_time,
            end_time=end_time
        )

        logger.info(f"Starting replay: {total} events from {start_time} to {end_time}")

        # Stream and replay events
        async for event in self.store.stream_events(
            start_time=start_time,
            end_time=end_time,
            batch_size=self.config.batch_size
        ):
            if self._should_stop:
                logger.info("Replay stopped by request")
                break

            # Apply topic filter
            if topic_pattern and not self._matches_topic(event.topic, topic_pattern):
                self._progress.skipped_events += 1
                continue

            # Apply skill filter
            if skill_name and event.skill_name != skill_name:
                self._progress.skipped_events += 1
                continue

            # Apply custom filter
            if event_filter:
                try:
                    if not await event_filter(event):
                        self._progress.skipped_events += 1
                        continue
                except Exception as e:
                    logger.warning(f"Filter error: {e}")
                    self._progress.skipped_events += 1
                    continue

            # Replay event
            await self._replay_event(event)

        # Complete
        await self._complete_replay()

        return self._progress

    async def replay_workflow(
        self,
        correlation_id: str,
        include_children: bool = True
    ) -> ReplayProgress:
        """
        Replay all events in a workflow chain.

        Args:
            correlation_id: Workflow correlation ID
            include_children: Include causally linked events

        Returns:
            ReplayProgress with results
        """
        # Get workflow events
        events = await self.store.get_workflow_events(
            correlation_id=correlation_id,
            include_children=include_children
        )

        self._progress = ReplayProgress(
            total_events=len(events)
        )

        logger.info(f"Replaying workflow {correlation_id}: {len(events)} events")

        for event in events:
            if self._should_stop:
                break
            await self._replay_event(event)

        await self._complete_replay()

        return self._progress

    async def replay_from_event(
        self,
        event_id: str,
        include_subsequent: bool = True,
        max_events: int = 1000
    ) -> ReplayProgress:
        """
        Replay from a specific event forward.

        Args:
            event_id: Starting event ID
            include_subsequent: Include events after this one
            max_events: Maximum events to replay

        Returns:
            ReplayProgress with results
        """
        # Get the starting event
        start_event = await self.store.get_event(event_id)
        if not start_event:
            logger.error(f"Event not found: {event_id}")
            return ReplayProgress(errors=[{"error": "Event not found", "event_id": event_id}])

        if include_subsequent:
            # Get events from this timestamp forward
            events = await self.store.query(
                start_time=start_event.timestamp,
                limit=max_events
            )
        else:
            events = [start_event]

        self._progress = ReplayProgress(
            total_events=len(events),
            start_time=start_event.timestamp
        )

        logger.info(f"Replaying from event {event_id}: {len(events)} events")

        for event in events:
            if self._should_stop:
                break
            await self._replay_event(event)

        await self._complete_replay()

        return self._progress

    async def replay_point_in_time(
        self,
        target_timestamp: str,
        topic_pattern: Optional[str] = None
    ) -> ReplayProgress:
        """
        Replay all events up to a specific point in time.

        Useful for restoring state to a specific moment.

        Args:
            target_timestamp: Target timestamp (ISO format)
            topic_pattern: Optional topic filter

        Returns:
            ReplayProgress with results
        """
        # Get earliest event timestamp
        earliest = await self.store.query(limit=1)
        if not earliest:
            logger.info("No events to replay")
            return ReplayProgress()

        start_time = earliest[0].timestamp

        return await self.replay_time_range(
            start_time=start_time,
            end_time=target_timestamp,
            topic_pattern=topic_pattern
        )

    def on_event(self, callback: ReplayCallback) -> None:
        """Register callback for each replayed event."""
        self._on_event_callbacks.append(callback)

    def on_complete(self, callback: Callable[[ReplayProgress], Awaitable[None]]) -> None:
        """Register callback for replay completion."""
        self._on_complete_callbacks.append(callback)

    def stop(self) -> None:
        """Request replay to stop."""
        self._should_stop = True
        logger.info("Replay stop requested")

    def get_progress(self) -> ReplayProgress:
        """Get current replay progress."""
        return self._progress

    async def _replay_event(self, event: SkillEvent) -> None:
        """Replay a single event."""
        replay_start = datetime.now()

        try:
            self._progress.current_event_timestamp = event.timestamp

            # Apply replay strategy timing
            if self.config.strategy == ReplayStrategy.REALTIME:
                await self._apply_realtime_delay(event)
            elif self.config.strategy == ReplayStrategy.ACCELERATED:
                await self._apply_accelerated_delay(event)
            # INSTANT: no delay

            # Emit to broker if configured
            if self.config.emit_to_broker and self.broker and not self.config.dry_run:
                await self.broker.emit(event)

            # Notify callbacks
            for callback in self._on_event_callbacks:
                try:
                    await callback(event, self._progress)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

            self._progress.replayed_events += 1

            if self.config.dry_run:
                logger.debug(f"[DRY RUN] Would replay: {event.event_id}")
            else:
                logger.debug(f"Replayed event: {event.event_id}")

        except Exception as e:
            self._progress.failed_events += 1
            self._progress.errors.append({
                "event_id": event.event_id,
                "error": str(e),
                "timestamp": event.timestamp
            })
            logger.error(f"Failed to replay event {event.event_id}: {e}")

        finally:
            elapsed = (datetime.now() - replay_start).total_seconds() * 1000
            self._progress.elapsed_ms += elapsed

    async def _apply_realtime_delay(self, event: SkillEvent) -> None:
        """Apply delay for realtime replay strategy."""
        if not hasattr(self, '_last_event_timestamp') or self._last_event_timestamp is None:
            self._last_event_timestamp = event.timestamp
            return

        try:
            last_ts = datetime.fromisoformat(self._last_event_timestamp)
            current_ts = datetime.fromisoformat(event.timestamp)
            delay_seconds = (current_ts - last_ts).total_seconds()

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            self._last_event_timestamp = event.timestamp
        except Exception:
            pass  # Skip delay on parse error

    async def _apply_accelerated_delay(self, event: SkillEvent) -> None:
        """Apply delay for accelerated replay strategy."""
        if not hasattr(self, '_last_event_timestamp') or self._last_event_timestamp is None:
            self._last_event_timestamp = event.timestamp
            return

        try:
            last_ts = datetime.fromisoformat(self._last_event_timestamp)
            current_ts = datetime.fromisoformat(event.timestamp)
            delay_seconds = (current_ts - last_ts).total_seconds()

            if delay_seconds > 0 and self.config.speed_multiplier > 0:
                await asyncio.sleep(delay_seconds / self.config.speed_multiplier)

            self._last_event_timestamp = event.timestamp
        except Exception:
            pass

    async def _complete_replay(self) -> None:
        """Complete the replay operation."""
        self._is_replaying = False

        logger.info(
            f"Replay complete: {self._progress.replayed_events} replayed, "
            f"{self._progress.failed_events} failed, "
            f"{self._progress.skipped_events} skipped"
        )

        # Notify completion callbacks
        for callback in self._on_complete_callbacks:
            try:
                await callback(self._progress)
            except Exception as e:
                logger.warning(f"Completion callback error: {e}")

    def _matches_topic(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports wildcards)."""
        # Convert glob pattern to simple matching
        if pattern == topic:
            return True

        pattern_parts = pattern.replace('#', '**').split('.')
        topic_parts = topic.split('.')

        return self._match_parts(pattern_parts, topic_parts)

    def _match_parts(self, pattern_parts: List[str], topic_parts: List[str]) -> bool:
        """Match topic parts against pattern parts."""
        if not pattern_parts:
            return not topic_parts

        if not topic_parts:
            return all(p in ('*', '**') for p in pattern_parts)

        if pattern_parts[0] == '**':
            # ** matches zero or more parts
            if len(pattern_parts) == 1:
                return True
            # Try matching rest of pattern from each position
            for i in range(len(topic_parts) + 1):
                if self._match_parts(pattern_parts[1:], topic_parts[i:]):
                    return True
            return False

        if pattern_parts[0] == '*':
            # * matches exactly one part
            return self._match_parts(pattern_parts[1:], topic_parts[1:])

        if pattern_parts[0] == topic_parts[0]:
            return self._match_parts(pattern_parts[1:], topic_parts[1:])

        return False


# ============================================================================
# Factory Function
# ============================================================================

def create_event_replayer(
    event_store: EventStore,
    broker: Optional[Any] = None,
    strategy: ReplayStrategy = ReplayStrategy.INSTANT,
    emit_to_broker: bool = True,
    dry_run: bool = False
) -> EventReplayer:
    """
    Create an event replayer instance.

    Args:
        event_store: Source of events
        broker: Optional EventBroker for re-emission
        strategy: Replay timing strategy
        emit_to_broker: Whether to re-emit events
        dry_run: Log but don't emit

    Returns:
        EventReplayer instance
    """
    config = ReplayConfig(
        strategy=strategy,
        emit_to_broker=emit_to_broker,
        dry_run=dry_run
    )
    return EventReplayer(event_store, broker, config)
