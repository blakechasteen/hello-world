"""
DreamEngine Background Scheduler
=================================

Orchestrates when synthesis cycles run based on idle time, intervals, or thresholds.

**Philosophy:** Memory synthesis happens during "sleep" - when the system has
spare cycles and enough new experiences to learn from.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import time
from enum import Enum

from HoloLoom.synthesis.types import SynthesisResults
from HoloLoom.synthesis.pattern_synthesis import PatternSynthesizer
from HoloLoom.synthesis.contradiction_detection import ContradictionDetector
from HoloLoom.synthesis.gap_identification import GapIdentifier


# ============================================================================
# Configuration
# ============================================================================

class TriggerStrategy(Enum):
    """When to trigger synthesis cycles."""
    IDLE = "idle"  # During low activity
    PERIODIC = "periodic"  # Fixed time intervals
    THRESHOLD = "threshold"  # After N new memories
    MANUAL = "manual"  # Explicit trigger only


@dataclass
class SchedulerConfig:
    """Configuration for background scheduler."""
    enabled: bool = True
    trigger_strategy: TriggerStrategy = TriggerStrategy.PERIODIC

    # Idle strategy settings
    idle_threshold_seconds: float = 60.0  # Consider idle after 60s of no queries
    idle_check_interval: float = 10.0  # Check every 10s

    # Periodic strategy settings
    periodic_interval_seconds: float = 3600.0  # Run every hour

    # Threshold strategy settings
    threshold_new_memories: int = 100  # Run after 100 new memories
    threshold_new_queries: int = 50  # Or 50 new queries

    # Resource limits
    max_cycle_duration_ms: float = 5000.0  # Abort if cycle takes >5s
    min_memories_for_synthesis: int = 10  # Need at least 10 memories

    # Component enablement
    enable_pattern_synthesis: bool = True
    enable_contradiction_detection: bool = True
    enable_gap_identification: bool = True


# ============================================================================
# Background Scheduler
# ============================================================================

class BackgroundScheduler:
    """
    Orchestrates DreamEngine synthesis cycles in background.

    Runs pattern synthesis, contradiction detection, and gap identification
    based on configurable triggering strategies without blocking main queries.

    Example:
        scheduler = BackgroundScheduler(
            pattern_synthesizer=PatternSynthesizer(),
            contradiction_detector=ContradictionDetector(),
            gap_identifier=GapIdentifier()
        )

        await scheduler.start()
        # System runs synthesis in background

        await scheduler.stop()
    """

    def __init__(
        self,
        pattern_synthesizer: Optional[PatternSynthesizer] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        gap_identifier: Optional[GapIdentifier] = None,
        config: Optional[SchedulerConfig] = None,
        on_cycle_complete: Optional[Callable[[SynthesisResults], None]] = None
    ):
        """
        Initialize background scheduler.

        Args:
            pattern_synthesizer: Pattern synthesis component
            contradiction_detector: Contradiction detection component
            gap_identifier: Gap identification component
            config: Scheduler configuration
            on_cycle_complete: Callback when cycle completes
        """
        self.config = config or SchedulerConfig()

        # Components
        self.pattern_synthesizer = pattern_synthesizer or PatternSynthesizer()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.gap_identifier = gap_identifier or GapIdentifier()

        # Callback
        self.on_cycle_complete = on_cycle_complete

        # State tracking
        self.is_running = False
        self.last_query_time: Optional[datetime] = None
        self.last_synthesis_time: Optional[datetime] = None
        self.memories_since_last_synthesis = 0
        self.queries_since_last_synthesis = 0

        # Background task
        self._background_task: Optional[asyncio.Task] = None

        # Cycle history
        self.cycle_history: List[SynthesisResults] = []
        self.total_cycles = 0

    async def start(self):
        """Start background scheduler."""
        if self.is_running:
            return

        if not self.config.enabled:
            return

        self.is_running = True

        # Start background task based on strategy
        if self.config.trigger_strategy == TriggerStrategy.IDLE:
            self._background_task = asyncio.create_task(self._idle_loop())
        elif self.config.trigger_strategy == TriggerStrategy.PERIODIC:
            self._background_task = asyncio.create_task(self._periodic_loop())
        elif self.config.trigger_strategy == TriggerStrategy.THRESHOLD:
            self._background_task = asyncio.create_task(self._threshold_loop())
        # MANUAL strategy doesn't run background task

    async def stop(self):
        """Stop background scheduler."""
        self.is_running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

    def register_query(self):
        """Register that a query occurred (for idle detection)."""
        self.last_query_time = datetime.now()
        self.queries_since_last_synthesis += 1

    def register_memory(self):
        """Register that a memory was added (for threshold detection)."""
        self.memories_since_last_synthesis += 1

    async def trigger_synthesis(
        self,
        knowledge_graph: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Optional[SynthesisResults]:
        """
        Manually trigger a synthesis cycle.

        Args:
            knowledge_graph: Knowledge graph data (nodes, edges, node_counts)
            force: Force synthesis even if minimum thresholds not met

        Returns:
            Synthesis results or None if skipped
        """
        # Check minimum requirements
        if not force and self.memories_since_last_synthesis < self.config.min_memories_for_synthesis:
            return None

        start_time = time.perf_counter()

        # Run synthesis components
        patterns = []
        contradictions = []
        gaps = []

        try:
            # Pattern synthesis
            if self.config.enable_pattern_synthesis:
                patterns = self.pattern_synthesizer.synthesize_patterns()

            # Contradiction detection
            if self.config.enable_contradiction_detection:
                contradictions = self.contradiction_detector.detect_contradictions()

            # Gap identification
            if self.config.enable_gap_identification and knowledge_graph:
                gaps = self.gap_identifier.identify_gaps(knowledge_graph)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Create results
            results = SynthesisResults(
                patterns_synthesized=patterns,
                contradictions_detected=contradictions,
                gaps_identified=gaps,
                cycle_duration_ms=duration_ms,
                metadata={
                    'trigger_strategy': self.config.trigger_strategy.value,
                    'memories_processed': self.memories_since_last_synthesis,
                    'queries_processed': self.queries_since_last_synthesis
                }
            )

            # Update state
            self.last_synthesis_time = datetime.now()
            self.memories_since_last_synthesis = 0
            self.queries_since_last_synthesis = 0
            self.total_cycles += 1

            # Store in history (keep last 10)
            self.cycle_history.append(results)
            if len(self.cycle_history) > 10:
                self.cycle_history.pop(0)

            # Call callback
            if self.on_cycle_complete:
                self.on_cycle_complete(results)

            return results

        except Exception as e:
            # Log error but don't crash
            print(f"Synthesis cycle failed: {e}")
            return None

    async def _idle_loop(self):
        """Run synthesis cycles during idle periods."""
        while self.is_running:
            await asyncio.sleep(self.config.idle_check_interval)

            if not self.last_query_time:
                continue

            # Check if we're idle
            idle_duration = (datetime.now() - self.last_query_time).total_seconds()
            if idle_duration >= self.config.idle_threshold_seconds:
                # We're idle, run synthesis
                await self.trigger_synthesis()

    async def _periodic_loop(self):
        """Run synthesis cycles at fixed intervals."""
        while self.is_running:
            await asyncio.sleep(self.config.periodic_interval_seconds)

            # Run synthesis
            await self.trigger_synthesis()

    async def _threshold_loop(self):
        """Run synthesis when thresholds are met."""
        while self.is_running:
            await asyncio.sleep(10.0)  # Check every 10 seconds

            # Check thresholds
            if (self.memories_since_last_synthesis >= self.config.threshold_new_memories or
                self.queries_since_last_synthesis >= self.config.threshold_new_queries):
                # Threshold met, run synthesis
                await self.trigger_synthesis()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dict with statistics
        """
        return {
            'is_running': self.is_running,
            'trigger_strategy': self.config.trigger_strategy.value,
            'total_cycles': self.total_cycles,
            'last_synthesis_time': self.last_synthesis_time.isoformat() if self.last_synthesis_time else None,
            'memories_since_last_synthesis': self.memories_since_last_synthesis,
            'queries_since_last_synthesis': self.queries_since_last_synthesis,
            'recent_cycles': len(self.cycle_history),
            'avg_cycle_duration_ms': (
                sum(c.cycle_duration_ms for c in self.cycle_history) / len(self.cycle_history)
                if self.cycle_history else 0.0
            ),
            'total_patterns_synthesized': sum(
                len(c.patterns_synthesized) for c in self.cycle_history
            ),
            'total_contradictions_detected': sum(
                len(c.contradictions_detected) for c in self.cycle_history
            ),
            'total_gaps_identified': sum(
                len(c.gaps_identified) for c in self.cycle_history
            )
        }

    def get_last_results(self) -> Optional[SynthesisResults]:
        """
        Get results from last synthesis cycle.

        Returns:
            Last synthesis results or None if no cycles run yet
        """
        return self.cycle_history[-1] if self.cycle_history else None


# ============================================================================
# Standalone Functions
# ============================================================================

async def run_synthesis_cycle(
    pattern_synthesizer: PatternSynthesizer,
    contradiction_detector: ContradictionDetector,
    gap_identifier: GapIdentifier,
    knowledge_graph: Optional[Dict[str, Any]] = None
) -> SynthesisResults:
    """
    Run a complete synthesis cycle (standalone function).

    Args:
        pattern_synthesizer: Pattern synthesis component
        contradiction_detector: Contradiction detection component
        gap_identifier: Gap identification component
        knowledge_graph: Knowledge graph data

    Returns:
        Synthesis results

    Example:
        >>> synthesizer = PatternSynthesizer()
        >>> detector = ContradictionDetector()
        >>> identifier = GapIdentifier()
        >>> results = await run_synthesis_cycle(synthesizer, detector, identifier)
        >>> print(results.summary())
    """
    start_time = time.perf_counter()

    # Pattern synthesis
    patterns = pattern_synthesizer.synthesize_patterns()

    # Contradiction detection
    contradictions = contradiction_detector.detect_contradictions()

    # Gap identification
    gaps = []
    if knowledge_graph:
        gaps = gap_identifier.identify_gaps(knowledge_graph)

    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000

    return SynthesisResults(
        patterns_synthesized=patterns,
        contradictions_detected=contradictions,
        gaps_identified=gaps,
        cycle_duration_ms=duration_ms
    )
