"""
Unified Forgetting Manager (Phase 2: Solved Memory)
=====================================================

Coordinates 5 scattered forgetting systems under a single interface:
1. KG Eviction (graph.py) - Hard limits and eviction strategies
2. Hot Pattern Decay (recursive/hot_patterns.py) - Usage-based decay
3. Memory Consolidation (consolidation.py) - Episodic → semantic
4. Lifecycle TTL (lifecycle_manager.py) - Time-based expiration
5. Activation Decay (activation_field.py) - Dynamic activation decay

Philosophy:
"Forgetting is a feature, not a bug" - lossy compression enables scaling.

Design Principles:
1. Single entry point for all forgetting operations
2. Configurable policies per memory tier
3. Observable: emit metrics and events for all forgetting operations
4. Safe: never lose PERMANENT memories
5. Gradual: prefer decay over immediate deletion

December 2025 - Phase 2 of "Solving Memory"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Forgetting Policies
# ============================================================================

class ForgetPolicy(Enum):
    """Forgetting strategies for different memory tiers."""
    NEVER = "never"              # Never forget (PERMANENT scope)
    DECAY = "decay"              # Gradual decay over time
    TTL = "ttl"                  # Hard time-to-live expiration
    LRU = "lru"                  # Least Recently Used eviction
    LFU = "lfu"                  # Least Frequently Used eviction
    IMPORTANCE = "importance"    # Multi-signal importance scoring
    CONSOLIDATE = "consolidate"  # Consolidate then optionally prune


class ForgetTrigger(Enum):
    """What triggers forgetting operations."""
    SCHEDULED = "scheduled"      # Background timer (e.g., every hour)
    CAPACITY = "capacity"        # When limits are exceeded
    MANUAL = "manual"            # Explicit request
    CONSOLIDATION = "consolidation"  # After memory consolidation


@dataclass
class ForgetEvent:
    """Record of a forgetting operation."""
    timestamp: datetime
    trigger: ForgetTrigger
    policy: ForgetPolicy
    subsystem: str  # Which subsystem performed the operation
    items_affected: int
    items_before: int
    items_after: int
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgetConfig:
    """Configuration for the unified forget manager."""

    # Global settings
    enable_scheduled_forgetting: bool = True
    scheduled_interval_minutes: int = 60  # Run every hour

    # KG Eviction (subsystem 1)
    enable_kg_eviction: bool = True
    kg_capacity_threshold: float = 0.9  # Trigger at 90% capacity
    kg_eviction_batch_size: int = 1000
    kg_eviction_policy: ForgetPolicy = ForgetPolicy.IMPORTANCE

    # Hot Pattern Decay (subsystem 2)
    enable_hot_decay: bool = True
    hot_decay_rate: float = 0.95  # 5% decay per hour
    hot_prune_threshold: float = 0.1  # Remove if heat < this

    # Lifecycle TTL (subsystem 3)
    enable_lifecycle_ttl: bool = True
    ttl_ephemeral_hours: int = 1
    ttl_temporary_days: int = 30

    # Activation Decay (subsystem 4)
    enable_activation_decay: bool = True
    activation_decay_rate: float = 0.5
    activation_decay_interval_seconds: float = 60.0

    # Consolidation (subsystem 5)
    enable_consolidation: bool = True
    consolidation_interval_minutes: int = 60
    prune_after_consolidation: bool = True

    # Safety
    preserve_permanent: bool = True  # Never forget PERMANENT memories
    min_retention_count: int = 100   # Always keep at least this many memories
    max_forget_per_cycle: int = 10000  # Limit per cycle to prevent mass deletion

    # Observability
    emit_events: bool = True
    log_operations: bool = True


# ============================================================================
# Unified Forget Manager
# ============================================================================

class ForgetManager:
    """
    Unified manager for all forgetting operations.

    Coordinates:
    - KG eviction (hard limits + strategies)
    - Hot pattern decay (usage-based)
    - Lifecycle TTL (time-based)
    - Activation decay (dynamic)
    - Consolidation cleanup (after consolidation)

    Usage:
        manager = ForgetManager(config, kg=kg, hot_tracker=tracker, ...)

        # Scheduled background forgetting
        await manager.start_background_loop()

        # Manual trigger
        result = await manager.forget_now(trigger=ForgetTrigger.MANUAL)

        # Subsystem-specific
        result = await manager.decay_hot_patterns()
    """

    def __init__(
        self,
        config: Optional[ForgetConfig] = None,
        kg: Optional[Any] = None,  # KG from graph.py
        hot_tracker: Optional[Any] = None,  # HotPatternTracker
        stream_manager: Optional[Any] = None,  # ContextStreamManager
        activation_field: Optional[Any] = None,  # ActivationField
        consolidator: Optional[Any] = None,  # MemoryConsolidator
        event_callback: Optional[Callable[[ForgetEvent], None]] = None
    ):
        """
        Initialize the unified forget manager.

        Args:
            config: Forgetting configuration
            kg: Knowledge graph (for KG eviction)
            hot_tracker: Hot pattern tracker (for decay)
            stream_manager: Context stream manager (for TTL)
            activation_field: Activation field (for decay)
            consolidator: Memory consolidator
            event_callback: Called for each forget event (optional)
        """
        self.config = config or ForgetConfig()

        # Subsystem references
        self._kg = kg
        self._hot_tracker = hot_tracker
        self._stream_manager = stream_manager
        self._activation_field = activation_field
        self._consolidator = consolidator

        # Event handling
        self._event_callback = event_callback
        self._event_history: List[ForgetEvent] = []
        self._max_history = 1000

        # Background task
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._total_items_forgotten = 0
        self._total_cycles = 0
        self._last_cycle_time: Optional[datetime] = None

        logger.info("ForgetManager initialized with config: %s", {
            'kg_eviction': config.enable_kg_eviction if config else True,
            'hot_decay': config.enable_hot_decay if config else True,
            'lifecycle_ttl': config.enable_lifecycle_ttl if config else True,
            'activation_decay': config.enable_activation_decay if config else True,
            'consolidation': config.enable_consolidation if config else True,
        })

    # ========================================================================
    # Subsystem Registration (for lazy initialization)
    # ========================================================================

    def register_kg(self, kg: Any) -> None:
        """Register knowledge graph for eviction."""
        self._kg = kg
        logger.debug("Registered KG for eviction")

    def register_hot_tracker(self, tracker: Any) -> None:
        """Register hot pattern tracker for decay."""
        self._hot_tracker = tracker
        logger.debug("Registered hot pattern tracker")

    def register_stream_manager(self, manager: Any) -> None:
        """Register context stream manager for TTL."""
        self._stream_manager = manager
        logger.debug("Registered stream manager for TTL")

    def register_activation_field(self, field: Any) -> None:
        """Register activation field for decay."""
        self._activation_field = field
        logger.debug("Registered activation field")

    def register_consolidator(self, consolidator: Any) -> None:
        """Register memory consolidator."""
        self._consolidator = consolidator
        logger.debug("Registered memory consolidator")

    # ========================================================================
    # Main Forgetting Interface
    # ========================================================================

    async def forget_now(
        self,
        trigger: ForgetTrigger = ForgetTrigger.MANUAL,
        subsystems: Optional[Set[str]] = None
    ) -> Dict[str, ForgetEvent]:
        """
        Execute forgetting across all (or specified) subsystems.

        Args:
            trigger: What triggered this operation
            subsystems: Which subsystems to run (None = all enabled)

        Returns:
            Dict mapping subsystem name to ForgetEvent
        """
        results = {}

        if subsystems is None:
            subsystems = {'kg', 'hot', 'lifecycle', 'activation', 'consolidation'}

        # 1. KG Eviction
        if 'kg' in subsystems and self.config.enable_kg_eviction and self._kg:
            event = await self._run_kg_eviction(trigger)
            if event:
                results['kg'] = event

        # 2. Hot Pattern Decay
        if 'hot' in subsystems and self.config.enable_hot_decay and self._hot_tracker:
            event = await self._run_hot_decay(trigger)
            if event:
                results['hot'] = event

        # 3. Lifecycle TTL
        if 'lifecycle' in subsystems and self.config.enable_lifecycle_ttl and self._stream_manager:
            event = await self._run_lifecycle_ttl(trigger)
            if event:
                results['lifecycle'] = event

        # 4. Activation Decay
        if 'activation' in subsystems and self.config.enable_activation_decay and self._activation_field:
            event = await self._run_activation_decay(trigger)
            if event:
                results['activation'] = event

        # 5. Consolidation Cleanup
        if 'consolidation' in subsystems and self.config.enable_consolidation and self._consolidator:
            event = await self._run_consolidation_cleanup(trigger)
            if event:
                results['consolidation'] = event

        # Update stats
        self._total_cycles += 1
        self._last_cycle_time = datetime.now()

        total_forgotten = sum(e.items_affected for e in results.values())
        self._total_items_forgotten += total_forgotten

        if self.config.log_operations and total_forgotten > 0:
            logger.info(
                "Forget cycle complete: %d items forgotten across %d subsystems",
                total_forgotten,
                len(results)
            )

        return results

    # ========================================================================
    # Subsystem-Specific Operations
    # ========================================================================

    async def _run_kg_eviction(self, trigger: ForgetTrigger) -> Optional[ForgetEvent]:
        """Run KG eviction if capacity exceeded."""
        if not self._kg:
            return None

        start_time = time.time()

        # Check if eviction needed
        try:
            node_count = self._kg.G.number_of_nodes()
            max_nodes = getattr(self._kg, '_max_nodes', 0)

            if max_nodes == 0:
                return None  # No limits configured

            capacity_ratio = node_count / max_nodes

            if capacity_ratio < self.config.kg_capacity_threshold:
                return None  # Not at capacity

            # Run eviction
            items_before = node_count
            eviction_result = self._kg.evict_to_capacity()
            items_after = self._kg.G.number_of_nodes()

            event = ForgetEvent(
                timestamp=datetime.now(),
                trigger=trigger,
                policy=ForgetPolicy(self._kg._eviction_strategy.value),
                subsystem='kg',
                items_affected=eviction_result.nodes_evicted,
                items_before=items_before,
                items_after=items_after,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'edges_removed': eviction_result.edges_removed,
                    'strategy_used': eviction_result.strategy_used.value,
                    'evicted_node_ids': eviction_result.evicted_node_ids[:10],  # First 10 for metadata
                }
            )

            self._emit_event(event)
            return event

        except Exception as e:
            logger.error("KG eviction failed: %s", e)
            return None

    async def _run_hot_decay(self, trigger: ForgetTrigger) -> Optional[ForgetEvent]:
        """Run hot pattern decay and pruning."""
        if not self._hot_tracker:
            return None

        start_time = time.time()

        try:
            items_before = len(self._hot_tracker.usage)

            # Apply decay
            self._hot_tracker._apply_decay_if_needed()

            # Prune cold patterns
            pruned = self._hot_tracker.prune_cold_patterns(
                min_heat=self.config.hot_prune_threshold
            )

            items_after = len(self._hot_tracker.usage)

            if pruned == 0:
                return None  # Nothing happened

            event = ForgetEvent(
                timestamp=datetime.now(),
                trigger=trigger,
                policy=ForgetPolicy.DECAY,
                subsystem='hot',
                items_affected=pruned,
                items_before=items_before,
                items_after=items_after,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'decay_rate': self.config.hot_decay_rate,
                    'prune_threshold': self.config.hot_prune_threshold,
                }
            )

            self._emit_event(event)
            return event

        except Exception as e:
            logger.error("Hot pattern decay failed: %s", e)
            return None

    async def _run_lifecycle_ttl(self, trigger: ForgetTrigger) -> Optional[ForgetEvent]:
        """Run TTL-based expiration across all streams."""
        if not self._stream_manager:
            return None

        start_time = time.time()

        try:
            total_before = 0
            total_after = 0
            total_pruned = 0

            # Prune each stream
            for stream in self._stream_manager.streams.values():
                before = len(stream.memories)
                total_before += before

                stream.prune_expired()

                after = len(stream.memories)
                total_after += after
                total_pruned += (before - after)

            if total_pruned == 0:
                return None

            event = ForgetEvent(
                timestamp=datetime.now(),
                trigger=trigger,
                policy=ForgetPolicy.TTL,
                subsystem='lifecycle',
                items_affected=total_pruned,
                items_before=total_before,
                items_after=total_after,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'streams_processed': len(self._stream_manager.streams),
                }
            )

            self._emit_event(event)
            return event

        except Exception as e:
            logger.error("Lifecycle TTL pruning failed: %s", e)
            return None

    async def _run_activation_decay(self, trigger: ForgetTrigger) -> Optional[ForgetEvent]:
        """Run activation decay on the activation field."""
        if not self._activation_field:
            return None

        start_time = time.time()

        try:
            items_before = len(self._activation_field.levels)

            # Apply decay
            self._activation_field.decay(rate=self.config.activation_decay_rate)

            # Count how many dropped below threshold (effectively forgotten)
            items_after = sum(
                1 for level in self._activation_field.levels.values()
                if level > 0.01  # Effectively zero
            )

            items_affected = items_before - items_after

            if items_affected == 0:
                return None

            event = ForgetEvent(
                timestamp=datetime.now(),
                trigger=trigger,
                policy=ForgetPolicy.DECAY,
                subsystem='activation',
                items_affected=items_affected,
                items_before=items_before,
                items_after=items_after,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'decay_rate': self.config.activation_decay_rate,
                }
            )

            self._emit_event(event)
            return event

        except Exception as e:
            logger.error("Activation decay failed: %s", e)
            return None

    async def _run_consolidation_cleanup(self, trigger: ForgetTrigger) -> Optional[ForgetEvent]:
        """Cleanup after consolidation (prune consolidated episodes)."""
        if not self._consolidator:
            return None

        if not self.config.prune_after_consolidation:
            return None

        start_time = time.time()

        try:
            # Get consolidation stats
            # Note: The actual consolidation runs separately; this just cleans up
            result = getattr(self._consolidator, '_last_consolidation_result', None)

            if not result:
                return None

            items_pruned = getattr(result, 'episodes_pruned', 0)

            if items_pruned == 0:
                return None

            event = ForgetEvent(
                timestamp=datetime.now(),
                trigger=ForgetTrigger.CONSOLIDATION,
                policy=ForgetPolicy.CONSOLIDATE,
                subsystem='consolidation',
                items_affected=items_pruned,
                items_before=getattr(result, 'input_episodes', 0),
                items_after=getattr(result, 'input_episodes', 0) - items_pruned,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={
                    'facts_created': getattr(result, 'output_facts', 0),
                    'strategy': getattr(result, 'strategy', 'unknown'),
                }
            )

            self._emit_event(event)
            return event

        except Exception as e:
            logger.error("Consolidation cleanup failed: %s", e)
            return None

    # ========================================================================
    # Background Loop
    # ========================================================================

    async def start_background_loop(self) -> None:
        """Start the background forgetting loop."""
        if self._running:
            logger.warning("Background loop already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info(
            "Started background forgetting loop (interval: %d minutes)",
            self.config.scheduled_interval_minutes
        )

    async def stop_background_loop(self) -> None:
        """Stop the background forgetting loop."""
        if not self._running:
            return

        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        logger.info("Stopped background forgetting loop")

    async def _background_loop(self) -> None:
        """Background loop that runs scheduled forgetting."""
        interval_seconds = self.config.scheduled_interval_minutes * 60

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)

                if self._running:
                    logger.debug("Running scheduled forgetting cycle")
                    await self.forget_now(trigger=ForgetTrigger.SCHEDULED)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in background forgetting loop: %s", e)
                # Don't crash the loop, just wait and retry
                await asyncio.sleep(60)

    # ========================================================================
    # Event Handling
    # ========================================================================

    def _emit_event(self, event: ForgetEvent) -> None:
        """Emit a forget event."""
        # Store in history
        self._event_history.append(event)

        # Trim history if too long
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Call callback if registered
        if self._event_callback and self.config.emit_events:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.error("Event callback failed: %s", e)

        # Log if configured
        if self.config.log_operations:
            logger.info(
                "Forget event: subsystem=%s policy=%s items=%d trigger=%s",
                event.subsystem,
                event.policy.value,
                event.items_affected,
                event.trigger.value
            )

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive forgetting statistics."""
        recent_events = self._event_history[-100:] if self._event_history else []

        by_subsystem = {}
        for event in recent_events:
            if event.subsystem not in by_subsystem:
                by_subsystem[event.subsystem] = {
                    'events': 0,
                    'items_forgotten': 0,
                    'total_duration_ms': 0.0,
                }
            by_subsystem[event.subsystem]['events'] += 1
            by_subsystem[event.subsystem]['items_forgotten'] += event.items_affected
            by_subsystem[event.subsystem]['total_duration_ms'] += event.duration_ms

        return {
            'total_cycles': self._total_cycles,
            'total_items_forgotten': self._total_items_forgotten,
            'last_cycle': self._last_cycle_time.isoformat() if self._last_cycle_time else None,
            'background_running': self._running,
            'subsystems_enabled': {
                'kg': self.config.enable_kg_eviction and self._kg is not None,
                'hot': self.config.enable_hot_decay and self._hot_tracker is not None,
                'lifecycle': self.config.enable_lifecycle_ttl and self._stream_manager is not None,
                'activation': self.config.enable_activation_decay and self._activation_field is not None,
                'consolidation': self.config.enable_consolidation and self._consolidator is not None,
            },
            'recent_events': len(recent_events),
            'by_subsystem': by_subsystem,
            'config': {
                'scheduled_interval_minutes': self.config.scheduled_interval_minutes,
                'kg_capacity_threshold': self.config.kg_capacity_threshold,
                'hot_decay_rate': self.config.hot_decay_rate,
                'preserve_permanent': self.config.preserve_permanent,
            }
        }

    def get_event_history(
        self,
        limit: int = 100,
        subsystem: Optional[str] = None
    ) -> List[ForgetEvent]:
        """Get recent forget events."""
        events = self._event_history

        if subsystem:
            events = [e for e in events if e.subsystem == subsystem]

        return events[-limit:]

    # ========================================================================
    # Async Context Manager
    # ========================================================================

    async def __aenter__(self) -> 'ForgetManager':
        """Start background loop on enter."""
        if self.config.enable_scheduled_forgetting:
            await self.start_background_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop background loop on exit."""
        await self.stop_background_loop()


# ============================================================================
# Factory Function
# ============================================================================

def create_forget_manager(
    config: Optional[ForgetConfig] = None,
    **kwargs
) -> ForgetManager:
    """
    Create a ForgetManager with default configuration.

    Args:
        config: Custom configuration (or None for defaults)
        **kwargs: Subsystem instances (kg, hot_tracker, etc.)

    Returns:
        Configured ForgetManager
    """
    return ForgetManager(config=config, **kwargs)
