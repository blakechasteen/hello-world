"""
HoloLoom Integration - Memory Bridge
====================================

**Created**: December 3, 2025
**Purpose**: Bridge EventBus with HoloLoom's memory system

## Features

- Bidirectional event-memory synchronization
- Pattern detection events from memory operations
- Event storage as memories for long-term recall
- Cross-system correlation tracking
- Awareness graph integration

## Architecture

```
EventBus <──────────────────────> HoloLoom Memory
    │                                   │
    ├─ skill.completed ──────────────> experience()
    ├─ pattern.detected ─────────────> memory graph
    │                                   │
    └─<──── memory.stored <────────────┤
    └─<──── pattern.found <────────────┘
```

"""

import asyncio
import uuid
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from skills.protocol import SkillEvent, EventType
from skills.event_bus.broker import EventBroker

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class IntegrationMode(Enum):
    """How to integrate with HoloLoom."""
    PASSIVE = "passive"        # Only emit events, don't store memories
    ACTIVE = "active"          # Both emit events and store memories
    MEMORY_ONLY = "memory_only"  # Only store memories, don't emit events


@dataclass
class HoloLoomIntegrationConfig:
    """Configuration for HoloLoom integration."""
    mode: IntegrationMode = IntegrationMode.ACTIVE

    # Event → Memory settings
    store_completed_events: bool = True
    store_failed_events: bool = True
    store_pattern_events: bool = True
    memory_importance_threshold: float = 0.5

    # Memory → Event settings
    emit_memory_events: bool = True
    emit_pattern_events: bool = True
    emit_recall_events: bool = False  # Can be noisy

    # Correlation settings
    enable_correlation_tracking: bool = True
    correlation_ttl_hours: int = 24

    # Filtering
    topic_whitelist: Optional[List[str]] = None  # If set, only these topics
    topic_blacklist: List[str] = field(default_factory=list)


# ============================================================================
# Event Types for Memory Operations
# ============================================================================

MEMORY_STORED_TOPIC = "hololoom.memory.stored"
MEMORY_RECALLED_TOPIC = "hololoom.memory.recalled"
MEMORY_UPDATED_TOPIC = "hololoom.memory.updated"
PATTERN_DETECTED_TOPIC = "hololoom.pattern.detected"
AWARENESS_CHANGED_TOPIC = "hololoom.awareness.changed"
REFLECTION_COMPLETED_TOPIC = "hololoom.reflection.completed"


# ============================================================================
# Memory-Event Correlation
# ============================================================================

@dataclass
class CorrelationEntry:
    """Tracks correlation between events and memories."""
    correlation_id: str
    event_ids: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationTracker:
    """Tracks correlations between events and memories."""

    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        self._correlations: Dict[str, CorrelationEntry] = {}

    def add_event(self, correlation_id: str, event_id: str) -> None:
        """Add an event to a correlation."""
        if correlation_id not in self._correlations:
            self._correlations[correlation_id] = CorrelationEntry(
                correlation_id=correlation_id
            )

        entry = self._correlations[correlation_id]
        if event_id not in entry.event_ids:
            entry.event_ids.append(event_id)
        entry.last_activity = datetime.now().isoformat()

    def add_memory(self, correlation_id: str, memory_id: str) -> None:
        """Add a memory to a correlation."""
        if correlation_id not in self._correlations:
            self._correlations[correlation_id] = CorrelationEntry(
                correlation_id=correlation_id
            )

        entry = self._correlations[correlation_id]
        if memory_id not in entry.memory_ids:
            entry.memory_ids.append(memory_id)
        entry.last_activity = datetime.now().isoformat()

    def get_correlation(self, correlation_id: str) -> Optional[CorrelationEntry]:
        """Get correlation entry."""
        return self._correlations.get(correlation_id)

    def get_events_for_memory(self, memory_id: str) -> List[str]:
        """Get all event IDs correlated with a memory."""
        event_ids = []
        for entry in self._correlations.values():
            if memory_id in entry.memory_ids:
                event_ids.extend(entry.event_ids)
        return list(set(event_ids))

    def get_memories_for_event(self, event_id: str) -> List[str]:
        """Get all memory IDs correlated with an event."""
        memory_ids = []
        for entry in self._correlations.values():
            if event_id in entry.event_ids:
                memory_ids.extend(entry.memory_ids)
        return list(set(memory_ids))

    def prune_expired(self) -> int:
        """Remove expired correlations."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        cutoff_str = cutoff.isoformat()

        to_remove = [
            cid for cid, entry in self._correlations.items()
            if entry.last_activity < cutoff_str
        ]

        for cid in to_remove:
            del self._correlations[cid]

        return len(to_remove)


# ============================================================================
# HoloLoom Integration Bridge
# ============================================================================

class HoloLoomBridge:
    """
    Bridges EventBus with HoloLoom memory system.

    Enables:
    - Events → Memories: Important events stored for long-term recall
    - Memories → Events: Memory operations emit events for workflows
    - Cross-correlation: Track relationships between events and memories
    """

    def __init__(
        self,
        broker: EventBroker,
        hololoom: Optional[Any] = None,  # HoloLoom instance
        config: Optional[HoloLoomIntegrationConfig] = None
    ):
        """
        Initialize HoloLoom bridge.

        Args:
            broker: EventBroker for event emission/subscription
            hololoom: HoloLoom instance (optional, can be set later)
            config: Integration configuration
        """
        self.broker = broker
        self.hololoom = hololoom
        self.config = config or HoloLoomIntegrationConfig()

        # Correlation tracking
        self.correlation_tracker = CorrelationTracker(
            ttl_hours=self.config.correlation_ttl_hours
        ) if self.config.enable_correlation_tracking else None

        # Subscription IDs for cleanup
        self._subscription_ids: List[str] = []

        # Statistics
        self.stats = {
            'events_to_memory': 0,
            'memory_to_events': 0,
            'patterns_detected': 0,
            'correlations_tracked': 0
        }

        logger.info(f"HoloLoomBridge created (mode={self.config.mode.value})")

    def set_hololoom(self, hololoom: Any) -> None:
        """Set or update HoloLoom instance."""
        self.hololoom = hololoom
        logger.info("HoloLoom instance set")

    async def start(self) -> None:
        """Start the integration bridge."""
        if self.config.mode == IntegrationMode.MEMORY_ONLY:
            logger.info("Memory-only mode - no event subscriptions")
            return

        # Subscribe to events that should become memories
        if self.config.store_completed_events:
            sub_id = await self.broker.subscribe(
                topic_pattern="*.completed",
                handler=self._handle_completed_event,
                subscription_id=f"hololoom_completed_{uuid.uuid4().hex[:8]}"
            )
            self._subscription_ids.append(sub_id)

        if self.config.store_failed_events:
            sub_id = await self.broker.subscribe(
                topic_pattern="*.failed",
                handler=self._handle_failed_event,
                subscription_id=f"hololoom_failed_{uuid.uuid4().hex[:8]}"
            )
            self._subscription_ids.append(sub_id)

        if self.config.store_pattern_events:
            sub_id = await self.broker.subscribe(
                topic_pattern="pattern.*",
                handler=self._handle_pattern_event,
                subscription_id=f"hololoom_pattern_{uuid.uuid4().hex[:8]}"
            )
            self._subscription_ids.append(sub_id)

        logger.info(f"HoloLoomBridge started with {len(self._subscription_ids)} subscriptions")

    async def stop(self) -> None:
        """Stop the integration bridge."""
        for sub_id in self._subscription_ids:
            await self.broker.unsubscribe(sub_id)

        self._subscription_ids.clear()
        logger.info("HoloLoomBridge stopped")

    # ========================================================================
    # Event → Memory Handlers
    # ========================================================================

    async def _handle_completed_event(self, event: SkillEvent) -> None:
        """Store completed skill events as memories."""
        if not self._should_process_topic(event.topic):
            return

        if not self.hololoom:
            return

        try:
            # Create memory content from event
            content = self._event_to_memory_content(event)

            # Calculate importance (completed events are moderately important)
            importance = self._calculate_importance(event, base=0.6)

            if importance >= self.config.memory_importance_threshold:
                # Store as memory
                memory = await self.hololoom.experience(
                    content,
                    metadata={
                        'source': 'event_bus',
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'skill_name': event.skill_name,
                        'topic': event.topic,
                        'importance': importance
                    }
                )

                # Track correlation
                if self.correlation_tracker and event.correlation_id:
                    self.correlation_tracker.add_event(event.correlation_id, event.event_id)
                    if hasattr(memory, 'id'):
                        self.correlation_tracker.add_memory(event.correlation_id, memory.id)
                        self.stats['correlations_tracked'] += 1

                self.stats['events_to_memory'] += 1
                logger.debug(f"Stored completed event as memory: {event.event_id}")

        except Exception as e:
            logger.error(f"Failed to store completed event: {e}")

    async def _handle_failed_event(self, event: SkillEvent) -> None:
        """Store failed skill events as memories (for learning)."""
        if not self._should_process_topic(event.topic):
            return

        if not self.hololoom:
            return

        try:
            # Failed events are more important (learning opportunities)
            content = self._event_to_memory_content(event, include_error=True)
            importance = self._calculate_importance(event, base=0.8)

            if importance >= self.config.memory_importance_threshold:
                memory = await self.hololoom.experience(
                    content,
                    metadata={
                        'source': 'event_bus',
                        'event_id': event.event_id,
                        'event_type': 'SKILL_FAILED',
                        'skill_name': event.skill_name,
                        'topic': event.topic,
                        'importance': importance,
                        'is_failure': True
                    }
                )

                if self.correlation_tracker and event.correlation_id:
                    self.correlation_tracker.add_event(event.correlation_id, event.event_id)
                    if hasattr(memory, 'id'):
                        self.correlation_tracker.add_memory(event.correlation_id, memory.id)

                self.stats['events_to_memory'] += 1
                logger.debug(f"Stored failed event as memory: {event.event_id}")

        except Exception as e:
            logger.error(f"Failed to store failed event: {e}")

    async def _handle_pattern_event(self, event: SkillEvent) -> None:
        """Store pattern detection events as memories."""
        if not self._should_process_topic(event.topic):
            return

        if not self.hololoom:
            return

        try:
            # Patterns are high importance
            content = f"Pattern detected: {event.payload.get('pattern_type', 'unknown')}"
            if 'description' in event.payload:
                content += f"\n{event.payload['description']}"

            importance = self._calculate_importance(event, base=0.9)

            memory = await self.hololoom.experience(
                content,
                metadata={
                    'source': 'event_bus',
                    'event_id': event.event_id,
                    'event_type': 'PATTERN_DETECTED',
                    'pattern_type': event.payload.get('pattern_type'),
                    'importance': importance
                }
            )

            self.stats['patterns_detected'] += 1
            logger.debug(f"Stored pattern event as memory: {event.event_id}")

        except Exception as e:
            logger.error(f"Failed to store pattern event: {e}")

    # ========================================================================
    # Memory → Event Emission
    # ========================================================================

    async def emit_memory_stored(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Emit event when a memory is stored.

        Args:
            memory_id: ID of the stored memory
            content: Memory content summary
            metadata: Additional metadata
            correlation_id: Optional correlation ID

        Returns:
            Event ID
        """
        if self.config.mode == IntegrationMode.MEMORY_ONLY:
            return ""

        if not self.config.emit_memory_events:
            return ""

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CUSTOM,
            skill_name="hololoom",
            topic=MEMORY_STORED_TOPIC,
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            payload={
                'memory_id': memory_id,
                'content_preview': content[:200] if len(content) > 200 else content,
                **(metadata or {})
            }
        )

        await self.broker.emit(event)

        if self.correlation_tracker and correlation_id:
            self.correlation_tracker.add_event(correlation_id, event.event_id)
            self.correlation_tracker.add_memory(correlation_id, memory_id)

        self.stats['memory_to_events'] += 1
        return event.event_id

    async def emit_memory_recalled(
        self,
        query: str,
        memory_ids: List[str],
        relevance_scores: Optional[List[float]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Emit event when memories are recalled.

        Args:
            query: Recall query
            memory_ids: IDs of recalled memories
            relevance_scores: Optional relevance scores
            correlation_id: Optional correlation ID

        Returns:
            Event ID
        """
        if self.config.mode == IntegrationMode.MEMORY_ONLY:
            return ""

        if not self.config.emit_recall_events:
            return ""

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CUSTOM,
            skill_name="hololoom",
            topic=MEMORY_RECALLED_TOPIC,
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            payload={
                'query': query,
                'memory_ids': memory_ids,
                'count': len(memory_ids),
                'relevance_scores': relevance_scores
            }
        )

        await self.broker.emit(event)
        self.stats['memory_to_events'] += 1
        return event.event_id

    async def emit_pattern_detected(
        self,
        pattern_type: str,
        description: str,
        affected_memories: List[str],
        confidence: float,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Emit event when a pattern is detected in memories.

        Args:
            pattern_type: Type of pattern (recurring_topic, temporal_cluster, etc.)
            description: Human-readable description
            affected_memories: Memory IDs involved
            confidence: Detection confidence (0-1)
            correlation_id: Optional correlation ID

        Returns:
            Event ID
        """
        if self.config.mode == IntegrationMode.MEMORY_ONLY:
            return ""

        if not self.config.emit_pattern_events:
            return ""

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PATTERN_DETECTED,
            skill_name="hololoom",
            topic=PATTERN_DETECTED_TOPIC,
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            payload={
                'pattern_type': pattern_type,
                'description': description,
                'affected_memories': affected_memories,
                'confidence': confidence
            }
        )

        await self.broker.emit(event)
        self.stats['patterns_detected'] += 1
        return event.event_id

    async def emit_awareness_changed(
        self,
        active_nodes: int,
        coherence: float,
        activation_delta: Dict[str, float],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Emit event when awareness graph state changes significantly.

        Args:
            active_nodes: Number of currently active memory nodes
            coherence: Global coherence score
            activation_delta: Changes in activation levels
            correlation_id: Optional correlation ID

        Returns:
            Event ID
        """
        if self.config.mode == IntegrationMode.MEMORY_ONLY:
            return ""

        event = SkillEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CUSTOM,
            skill_name="hololoom",
            topic=AWARENESS_CHANGED_TOPIC,
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            payload={
                'active_nodes': active_nodes,
                'coherence': coherence,
                'activation_delta': activation_delta
            }
        )

        await self.broker.emit(event)
        return event.event_id

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _should_process_topic(self, topic: str) -> bool:
        """Check if topic should be processed."""
        if self.config.topic_whitelist:
            return any(
                topic.startswith(t.rstrip('*'))
                for t in self.config.topic_whitelist
            )

        for blacklisted in self.config.topic_blacklist:
            if topic.startswith(blacklisted.rstrip('*')):
                return False

        return True

    def _event_to_memory_content(
        self,
        event: SkillEvent,
        include_error: bool = False
    ) -> str:
        """Convert event to memory content string."""
        parts = [
            f"Skill: {event.skill_name}",
            f"Topic: {event.topic}",
            f"Time: {event.timestamp}"
        ]

        if event.payload:
            if 'result' in event.payload:
                parts.append(f"Result: {event.payload['result']}")
            if 'summary' in event.payload:
                parts.append(f"Summary: {event.payload['summary']}")
            if include_error and 'error' in event.payload:
                parts.append(f"Error: {event.payload['error']}")

        return "\n".join(parts)

    def _calculate_importance(self, event: SkillEvent, base: float = 0.5) -> float:
        """Calculate importance score for an event."""
        importance = base

        # Boost for explicit importance in payload
        if 'importance' in event.payload:
            importance = event.payload['importance']

        # Boost for certain event types
        if event.event_type == EventType.PATTERN_DETECTED:
            importance = min(1.0, importance + 0.2)
        elif event.event_type == EventType.SECURITY_ALERT:
            importance = min(1.0, importance + 0.3)

        return importance

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            **self.stats,
            'mode': self.config.mode.value,
            'subscriptions_active': len(self._subscription_ids)
        }

        if self.correlation_tracker:
            stats['correlations_count'] = len(self.correlation_tracker._correlations)

        return stats

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Factory Function
# ============================================================================

def create_hololoom_bridge(
    broker: EventBroker,
    hololoom: Optional[Any] = None,
    mode: IntegrationMode = IntegrationMode.ACTIVE,
    store_completed: bool = True,
    store_failed: bool = True,
    emit_memory_events: bool = True
) -> HoloLoomBridge:
    """
    Create a HoloLoom integration bridge.

    Args:
        broker: EventBroker instance
        hololoom: Optional HoloLoom instance
        mode: Integration mode
        store_completed: Store completed events as memories
        store_failed: Store failed events as memories
        emit_memory_events: Emit events for memory operations

    Returns:
        HoloLoomBridge instance
    """
    config = HoloLoomIntegrationConfig(
        mode=mode,
        store_completed_events=store_completed,
        store_failed_events=store_failed,
        emit_memory_events=emit_memory_events
    )
    return HoloLoomBridge(broker, hololoom, config)
