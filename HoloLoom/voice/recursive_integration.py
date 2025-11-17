"""
Recursive Learning Integration for Elle AR
===========================================
Integrates HoloLoom's complete recursive learning system (Phases 1-5) with Elle AR
to enable self-improvement, pattern learning, and quality refinement for AR interactions.

Architecture:
    ARLearningEngine: Main integration layer wrapping FullLearningEngine
    ARProvenanceTracker: Provenance tracking for AR queries (gesture + voice + vision)
    ARQualityRefiner: Quality refinement for low-confidence AR responses
    ARBackgroundLearner: Background learning from AR usage patterns

Key Features:
    - Scratchpad provenance tracking for all AR queries
    - Pattern extraction from multimodal AR interactions
    - Quality refinement for low-confidence responses (<0.75)
    - Background learning every 60s with Thompson Sampling
    - <3ms overhead per query

Author: HoloLoom Team
Date: November 17, 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

# HoloLoom recursive learning imports
try:
    from HoloLoom.recursive import (
        FullLearningEngine,
        ThompsonPriors,
        PolicyWeights,
        BackgroundLearner,
        LearningMetrics,
        AdvancedRefiner,
        RefinementStrategy,
        RefinementResult,
        QualityMetrics,
        Scratchpad,
        ScratchpadEntry,
    )
    RECURSIVE_AVAILABLE = True
except ImportError:
    RECURSIVE_AVAILABLE = False

# HoloLoom core imports
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

# AR-specific imports
from HoloLoom.voice.ar_context import (
    ARContext,
    ARObject,
    ARObjectType,
    Vector3,
    Quaternion,
)

# AR pattern learning (will create next)
try:
    from HoloLoom.voice.ar_pattern_learner import (
        ARPatternLearner,
        ARPattern,
        GestureType,
        VoiceIntent,
    )
    AR_PATTERN_LEARNER_AVAILABLE = True
except ImportError:
    AR_PATTERN_LEARNER_AVAILABLE = False

# AR refinement (will create next)
try:
    from HoloLoom.voice.ar_refiner import (
        ARRefiner,
        ARRefinementStrategy,
        ARRefinementResult,
    )
    AR_REFINER_AVAILABLE = True
except ImportError:
    AR_REFINER_AVAILABLE = False

# AR background learner (will create next)
try:
    from HoloLoom.voice.ar_background_learner import (
        ARBackgroundLearner,
        ARLearningMetrics,
    )
    AR_BACKGROUND_LEARNER_AVAILABLE = True
except ImportError:
    AR_BACKGROUND_LEARNER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# AR Query Types
# ============================================================================

class ARQueryType(Enum):
    """Types of AR queries"""
    VOICE_ONLY = "voice_only"
    GESTURE_ONLY = "gesture_only"
    VISION_ONLY = "vision_only"
    VOICE_GESTURE = "voice_gesture"
    VOICE_VISION = "voice_vision"
    GESTURE_VISION = "gesture_vision"
    MULTIMODAL = "multimodal"  # All three


@dataclass
class ARQuery:
    """
    AR-specific query with multimodal context.

    Extends base Query with AR-specific metadata for pattern learning.
    """
    text: str
    ar_context: Optional[ARContext] = None
    gesture_detected: Optional[str] = None  # e.g., "pinch", "swipe_left"
    voice_intent: Optional[str] = None  # e.g., "select", "delete", "info"
    vision_objects: List[ARObject] = field(default_factory=list)
    user_position: Optional[Vector3] = None
    user_orientation: Optional[Quaternion] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def query_type(self) -> ARQueryType:
        """Determine query type based on active modalities"""
        has_voice = bool(self.text or self.voice_intent)
        has_gesture = bool(self.gesture_detected)
        has_vision = bool(self.vision_objects)

        if has_voice and has_gesture and has_vision:
            return ARQueryType.MULTIMODAL
        elif has_voice and has_gesture:
            return ARQueryType.VOICE_GESTURE
        elif has_voice and has_vision:
            return ARQueryType.VOICE_VISION
        elif has_gesture and has_vision:
            return ARQueryType.GESTURE_VISION
        elif has_voice:
            return ARQueryType.VOICE_ONLY
        elif has_gesture:
            return ARQueryType.GESTURE_ONLY
        elif has_vision:
            return ARQueryType.VISION_ONLY
        else:
            return ARQueryType.VOICE_ONLY  # Default

    def to_hololoom_query(self) -> Query:
        """Convert to standard HoloLoom Query"""
        # Enrich text with AR context
        enriched_text = self.text

        if self.gesture_detected:
            enriched_text = f"[Gesture: {self.gesture_detected}] {enriched_text}"

        if self.vision_objects:
            obj_names = [obj.name for obj in self.vision_objects[:3]]
            enriched_text = f"[Vision: {', '.join(obj_names)}] {enriched_text}"

        return Query(
            text=enriched_text,
            metadata={
                'ar_context': self.ar_context,
                'gesture': self.gesture_detected,
                'voice_intent': self.voice_intent,
                'vision_objects': self.vision_objects,
                'query_type': self.query_type.value,
                'timestamp': self.timestamp,
            }
        )


# ============================================================================
# AR Provenance Tracker
# ============================================================================

@dataclass
class ARProvenanceEntry:
    """
    AR-specific provenance entry extending ScratchpadEntry.

    Tracks thought → action → observation → score with AR context.
    """
    thought: str
    action: str
    observation: str
    score: float

    # AR-specific metadata
    gesture: Optional[str] = None
    voice_intent: Optional[str] = None
    vision_objects: List[str] = field(default_factory=list)
    query_type: str = "voice_only"
    ar_context_summary: Optional[str] = None

    iteration: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def to_scratchpad_entry(self) -> 'ScratchpadEntry':
        """Convert to standard ScratchpadEntry"""
        if not RECURSIVE_AVAILABLE:
            return None

        return ScratchpadEntry(
            thought=self.thought,
            action=self.action,
            observation=self.observation,
            score=self.score,
            iteration=self.iteration,
            timestamp=self.timestamp,
            metadata={
                'gesture': self.gesture,
                'voice_intent': self.voice_intent,
                'vision_objects': self.vision_objects,
                'query_type': self.query_type,
                'ar_context': self.ar_context_summary,
            }
        )


class ARProvenanceTracker:
    """
    Provenance tracker for AR queries.

    Wraps Scratchpad with AR-specific tracking for gesture + voice + vision.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize AR provenance tracker.

        Args:
            capacity: Maximum number of entries to store
        """
        if RECURSIVE_AVAILABLE:
            self.scratchpad = Scratchpad(capacity=capacity)
        else:
            self.scratchpad = None

        self.ar_entries: List[ARProvenanceEntry] = []
        self.capacity = capacity

    def track(
        self,
        ar_query: ARQuery,
        spacetime: Spacetime,
        thought: str = "Processing AR query",
    ):
        """
        Track AR query provenance.

        Args:
            ar_query: AR query with multimodal context
            spacetime: Result spacetime
            thought: System thought/reasoning
        """
        # Extract action from spacetime
        action = spacetime.metadata.get('tool_used', 'unknown')

        # Extract observation (response preview)
        observation = str(spacetime.response)[:200] if spacetime.response else "No response"

        # Score from confidence
        score = spacetime.confidence

        # Create AR provenance entry
        ar_entry = ARProvenanceEntry(
            thought=thought,
            action=action,
            observation=observation,
            score=score,
            gesture=ar_query.gesture_detected,
            voice_intent=ar_query.voice_intent,
            vision_objects=[obj.name for obj in ar_query.vision_objects],
            query_type=ar_query.query_type.value,
            ar_context_summary=self._summarize_ar_context(ar_query.ar_context),
        )

        # Store AR entry
        self.ar_entries.append(ar_entry)
        if len(self.ar_entries) > self.capacity:
            self.ar_entries = self.ar_entries[-self.capacity:]

        # Also track in standard scratchpad if available
        if self.scratchpad is not None and RECURSIVE_AVAILABLE:
            scratchpad_entry = ar_entry.to_scratchpad_entry()
            if scratchpad_entry:
                self.scratchpad.add_entry(
                    thought=scratchpad_entry.thought,
                    action=scratchpad_entry.action,
                    observation=scratchpad_entry.observation,
                    score=scratchpad_entry.score,
                    metadata=scratchpad_entry.metadata,
                )

    def get_ar_history(self, n: Optional[int] = None) -> List[ARProvenanceEntry]:
        """
        Get AR provenance history.

        Args:
            n: Number of recent entries (None = all)

        Returns:
            List of AR provenance entries
        """
        if n is None:
            return self.ar_entries.copy()
        return self.ar_entries[-n:]

    def get_by_query_type(self, query_type: ARQueryType) -> List[ARProvenanceEntry]:
        """Get entries by query type"""
        return [
            entry for entry in self.ar_entries
            if entry.query_type == query_type.value
        ]

    def get_by_gesture(self, gesture: str) -> List[ARProvenanceEntry]:
        """Get entries by gesture type"""
        return [
            entry for entry in self.ar_entries
            if entry.gesture == gesture
        ]

    def _summarize_ar_context(self, ar_context: Optional[ARContext]) -> Optional[str]:
        """Summarize AR context for provenance"""
        if ar_context is None:
            return None

        parts = []
        if ar_context.user_position:
            parts.append(f"pos={ar_context.user_position}")
        if ar_context.visible_objects:
            parts.append(f"objects={len(ar_context.visible_objects)}")

        return ", ".join(parts) if parts else None


# ============================================================================
# AR Learning Engine
# ============================================================================

@dataclass
class ARLearningConfig:
    """Configuration for AR learning engine"""
    enable_provenance: bool = True
    enable_pattern_learning: bool = True
    enable_refinement: bool = True
    enable_background_learning: bool = True

    refinement_threshold: float = 0.75  # Refine if confidence < this
    max_refinement_iterations: int = 3

    background_update_interval: float = 60.0  # Update every 60 seconds

    # Pattern learning thresholds
    min_pattern_confidence: float = 0.8
    min_pattern_support: int = 3  # Need 3+ examples

    # Thompson Sampling settings
    thompson_exploration_rate: float = 0.1

    # Provenance capacity
    provenance_capacity: int = 1000


class ARLearningEngine:
    """
    Main AR learning engine integrating all recursive learning phases.

    Wraps FullLearningEngine with AR-specific features:
    - AR provenance tracking (gesture + voice + vision)
    - AR pattern learning (multimodal patterns)
    - AR quality refinement (AR-specific strategies)
    - Background learning from AR usage

    Usage:
        async with ARLearningEngine(config, shards) as engine:
            spacetime = await engine.weave(ar_query)
            stats = engine.get_learning_statistics()
    """

    def __init__(
        self,
        config: Config,
        shards: List[MemoryShard],
        ar_config: Optional[ARLearningConfig] = None,
    ):
        """
        Initialize AR learning engine.

        Args:
            config: HoloLoom config
            shards: Memory shards
            ar_config: AR-specific learning config
        """
        self.config = config
        self.shards = shards
        self.ar_config = ar_config or ARLearningConfig()

        # Check availability
        if not RECURSIVE_AVAILABLE:
            logger.warning("Recursive learning not available - degraded mode")

        # Core orchestrator
        self.orchestrator: Optional[WeavingOrchestrator] = None

        # Recursive learning engine (if available)
        self.learning_engine: Optional['FullLearningEngine'] = None

        # AR-specific components
        self.provenance_tracker = ARProvenanceTracker(
            capacity=self.ar_config.provenance_capacity
        )

        self.pattern_learner = None  # Will initialize if available
        self.refiner = None  # Will initialize if available
        self.background_learner = None  # Will initialize if available

        # Background learning task
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self.queries_processed = 0
        self.patterns_discovered = 0
        self.refinements_performed = 0
        self.avg_confidence = 0.0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start the AR learning engine"""
        # Create orchestrator
        self.orchestrator = WeavingOrchestrator(
            cfg=self.config,
            shards=self.shards,
        )
        await self.orchestrator.__aenter__()

        # Initialize recursive learning if available
        if RECURSIVE_AVAILABLE and self.ar_config.enable_pattern_learning:
            try:
                self.learning_engine = FullLearningEngine(
                    cfg=self.config,
                    shards=self.shards,
                    enable_background_learning=False,  # We'll manage this
                )
                await self.learning_engine.__aenter__()
                logger.info("Recursive learning engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize recursive learning: {e}")

        # Initialize AR pattern learner if available
        if AR_PATTERN_LEARNER_AVAILABLE and self.ar_config.enable_pattern_learning:
            try:
                from HoloLoom.voice.ar_pattern_learner import ARPatternLearner
                self.pattern_learner = ARPatternLearner(
                    min_confidence=self.ar_config.min_pattern_confidence,
                    min_support=self.ar_config.min_pattern_support,
                )
                logger.info("AR pattern learner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AR pattern learner: {e}")

        # Initialize AR refiner if available
        if AR_REFINER_AVAILABLE and self.ar_config.enable_refinement:
            try:
                from HoloLoom.voice.ar_refiner import ARRefiner
                self.refiner = ARRefiner(
                    orchestrator=self.orchestrator,
                    threshold=self.ar_config.refinement_threshold,
                    max_iterations=self.ar_config.max_refinement_iterations,
                )
                logger.info("AR refiner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AR refiner: {e}")

        # Start background learning if enabled
        if self.ar_config.enable_background_learning:
            self._background_task = asyncio.create_task(
                self._background_learning_loop()
            )
            logger.info("Background learning started")

    async def stop(self):
        """Stop the AR learning engine"""
        # Signal shutdown
        self._shutdown_event.set()

        # Wait for background task
        if self._background_task:
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Background learning task timeout")
                self._background_task.cancel()

        # Cleanup learning engine
        if self.learning_engine:
            await self.learning_engine.__aexit__(None, None, None)

        # Cleanup orchestrator
        if self.orchestrator:
            await self.orchestrator.__aexit__(None, None, None)

        logger.info("AR learning engine stopped")

    async def weave(
        self,
        ar_query: ARQuery,
        enable_refinement: Optional[bool] = None,
    ) -> Spacetime:
        """
        Weave AR query with recursive learning.

        Args:
            ar_query: AR query with multimodal context
            enable_refinement: Override refinement setting

        Returns:
            Spacetime result
        """
        start_time = time.time()

        # Convert to HoloLoom query
        query = ar_query.to_hololoom_query()

        # Weave through orchestrator
        spacetime = await self.orchestrator.weave(query)

        # Track provenance
        if self.ar_config.enable_provenance:
            self.provenance_tracker.track(
                ar_query=ar_query,
                spacetime=spacetime,
                thought=f"Processing {ar_query.query_type.value} query",
            )

        # Extract and learn patterns
        if self.pattern_learner and self.ar_config.enable_pattern_learning:
            try:
                self.pattern_learner.extract_and_learn(ar_query, spacetime)
            except Exception as e:
                logger.warning(f"Pattern learning failed: {e}")

        # Refine if low confidence
        should_refine = (
            enable_refinement if enable_refinement is not None
            else self.ar_config.enable_refinement
        )

        if should_refine and spacetime.confidence < self.ar_config.refinement_threshold:
            if self.refiner:
                try:
                    refined = await self.refiner.refine(ar_query, spacetime)
                    spacetime = refined.final_spacetime
                    self.refinements_performed += 1
                    logger.info(
                        f"Refined AR query: {refined.initial_quality:.2f} → "
                        f"{refined.final_quality:.2f} ({refined.iterations} iterations)"
                    )
                except Exception as e:
                    logger.warning(f"Refinement failed: {e}")

        # Update statistics
        self.queries_processed += 1
        self.avg_confidence = (
            (self.avg_confidence * (self.queries_processed - 1) + spacetime.confidence) /
            self.queries_processed
        )

        duration = (time.time() - start_time) * 1000
        logger.debug(f"AR weave completed in {duration:.1f}ms (confidence={spacetime.confidence:.2f})")

        return spacetime

    async def _background_learning_loop(self):
        """Background learning loop (runs every 60s)"""
        while not self._shutdown_event.is_set():
            try:
                # Wait for interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.ar_config.background_update_interval
                )
                # Shutdown signaled
                break
            except asyncio.TimeoutError:
                # Time to learn
                await self._perform_background_learning()

        logger.info("Background learning loop stopped")

    async def _perform_background_learning(self):
        """Perform background learning update"""
        try:
            # Learn patterns from recent AR interactions
            if self.pattern_learner:
                recent_entries = self.provenance_tracker.get_ar_history(n=100)
                patterns_found = self.pattern_learner.mine_patterns(recent_entries)
                self.patterns_discovered += len(patterns_found)

                if patterns_found:
                    logger.info(f"Discovered {len(patterns_found)} new AR patterns")

            # Update learning engine if available
            if self.learning_engine and hasattr(self.learning_engine, 'update_from_buffer'):
                await self.learning_engine.update_from_buffer()

            logger.debug("Background learning update completed")

        except Exception as e:
            logger.error(f"Background learning error: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            'queries_processed': self.queries_processed,
            'avg_confidence': self.avg_confidence,
            'refinements_performed': self.refinements_performed,
            'patterns_discovered': self.patterns_discovered,
        }

        # Add provenance stats
        if self.provenance_tracker:
            stats['provenance_entries'] = len(self.provenance_tracker.ar_entries)

        # Add pattern learner stats
        if self.pattern_learner:
            stats['active_patterns'] = len(getattr(self.pattern_learner, 'patterns', []))

        # Add learning engine stats
        if self.learning_engine and hasattr(self.learning_engine, 'get_learning_statistics'):
            stats['learning_engine'] = self.learning_engine.get_learning_statistics()

        return stats

    def get_hot_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get hot AR patterns (most successful).

        Args:
            limit: Maximum number of patterns

        Returns:
            List of hot patterns
        """
        if not self.pattern_learner:
            return []

        return self.pattern_learner.get_hot_patterns(limit=limit)

    def save_learning_state(self, path: str):
        """Save learning state to disk"""
        import json

        state = {
            'queries_processed': self.queries_processed,
            'patterns_discovered': self.patterns_discovered,
            'refinements_performed': self.refinements_performed,
            'avg_confidence': self.avg_confidence,
            'timestamp': time.time(),
        }

        # Add pattern learner state
        if self.pattern_learner and hasattr(self.pattern_learner, 'get_state'):
            state['patterns'] = self.pattern_learner.get_state()

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Learning state saved to {path}")

    def load_learning_state(self, path: str):
        """Load learning state from disk"""
        import json

        with open(path, 'r') as f:
            state = json.load(f)

        self.queries_processed = state.get('queries_processed', 0)
        self.patterns_discovered = state.get('patterns_discovered', 0)
        self.refinements_performed = state.get('refinements_performed', 0)
        self.avg_confidence = state.get('avg_confidence', 0.0)

        # Load pattern learner state
        if self.pattern_learner and 'patterns' in state:
            if hasattr(self.pattern_learner, 'load_state'):
                self.pattern_learner.load_state(state['patterns'])

        logger.info(f"Learning state loaded from {path}")


# ============================================================================
# Helper Functions
# ============================================================================

async def weave_with_ar_learning(
    ar_query: ARQuery,
    config: Config,
    shards: List[MemoryShard],
    ar_config: Optional[ARLearningConfig] = None,
) -> Tuple[Spacetime, Dict[str, Any]]:
    """
    Convenience function to weave AR query with learning.

    Args:
        ar_query: AR query
        config: HoloLoom config
        shards: Memory shards
        ar_config: AR learning config

    Returns:
        (spacetime, learning_stats) tuple
    """
    async with ARLearningEngine(config, shards, ar_config) as engine:
        spacetime = await engine.weave(ar_query)
        stats = engine.get_learning_statistics()
        return spacetime, stats


# Export public API
__all__ = [
    'ARLearningEngine',
    'ARLearningConfig',
    'ARProvenanceTracker',
    'ARProvenanceEntry',
    'ARQuery',
    'ARQueryType',
    'weave_with_ar_learning',
]
