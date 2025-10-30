#!/usr/bin/env python3
"""
Hot Pattern Feedback - Phase 3 of Recursive Learning Vision
=============================================================
Tracks cache hot entries and feeds back to improve retrieval.

Architecture:
    HotPatternTracker: Tracks which knowledge elements are accessed most
    UsageAnalyzer: Analyzes usage patterns to identify valuable knowledge
    AdaptiveRetriever: Adjusts retrieval weights based on usage patterns

Philosophy:
    Usage reveals value. Knowledge that gets accessed frequently in successful
    queries is more valuable and should be prioritized in future retrieval.
    This creates a natural reinforcement loop: useful knowledge becomes easier
    to find, weak knowledge fades into the background.

Author: Claude Code
Date: 2025-10-29 (Phase 3 Implementation)
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import time
import math

# HoloLoom components
from HoloLoom.recursive.loop_integration import (
    LearningLoopEngine,
    LearningLoopConfig
)
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# Hot Pattern Tracking
# ============================================================================

@dataclass
class UsageRecord:
    """Record of knowledge element usage"""
    element_id: str  # Thread ID, shard ID, etc.
    element_type: str  # "thread", "shard", "motif"
    access_count: int = 0
    success_count: int = 0  # Times used in high-confidence queries
    total_confidence: float = 0.0
    avg_confidence: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    first_accessed: float = field(default_factory=time.time)

    def record_access(self, confidence: float):
        """Record a new access"""
        self.access_count += 1
        self.last_accessed = time.time()

        # Track if successful (high confidence)
        if confidence >= 0.75:
            self.success_count += 1

        # Update average confidence
        self.total_confidence += confidence
        self.avg_confidence = self.total_confidence / self.access_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)"""
        if self.access_count == 0:
            return 0.0
        return self.success_count / self.access_count

    @property
    def heat_score(self) -> float:
        """
        Calculate heat score combining frequency and quality.

        Heat = access_count * success_rate * avg_confidence

        High heat means: frequently accessed + high success rate + high confidence
        """
        return self.access_count * self.success_rate * self.avg_confidence

    @property
    def age_seconds(self) -> float:
        """Time since last access"""
        return time.time() - self.last_accessed


class HotPatternTracker:
    """
    Tracks which knowledge elements are accessed most (hot patterns).

    Features:
    - Access counting for threads, shards, motifs
    - Success rate tracking (high-confidence queries)
    - Heat score calculation (frequency + quality)
    - Decay for stale patterns
    """

    def __init__(
        self,
        decay_rate: float = 0.95,  # Heat decay per hour
        decay_interval: float = 3600.0  # Apply decay every hour
    ):
        """
        Initialize hot pattern tracker.

        Args:
            decay_rate: Multiplier for heat decay (0.95 = 5% loss per interval)
            decay_interval: Seconds between decay applications
        """
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval

        # Usage records by element ID
        self.usage: Dict[str, UsageRecord] = {}

        # Last decay time
        self.last_decay = time.time()

        # Statistics
        self.total_accesses = 0
        self.unique_elements = 0

        self.logger = logging.getLogger(f"{__name__}.HotPatternTracker")

    def record_usage(
        self,
        spacetime: Spacetime
    ):
        """
        Record usage from Spacetime trace.

        Tracks:
        - Threads activated
        - Motifs detected
        - Confidence achieved

        Args:
            spacetime: Woven fabric with trace
        """
        trace = spacetime.trace
        confidence = trace.tool_confidence

        # Record thread usage
        for thread_id in trace.threads_activated:
            if thread_id not in self.usage:
                self.usage[thread_id] = UsageRecord(
                    element_id=thread_id,
                    element_type="thread"
                )

            self.usage[thread_id].record_access(confidence)
            self.total_accesses += 1

        # Record motif usage
        for motif in trace.motifs_detected:
            motif_id = f"motif:{motif}"

            if motif_id not in self.usage:
                self.usage[motif_id] = UsageRecord(
                    element_id=motif_id,
                    element_type="motif"
                )

            self.usage[motif_id].record_access(confidence)
            self.total_accesses += 1

        self.unique_elements = len(self.usage)

        # Apply decay if needed
        self._apply_decay_if_needed()

    def get_hot_patterns(
        self,
        element_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[UsageRecord]:
        """
        Get hottest patterns (most valuable knowledge).

        Hot patterns have high heat scores (frequency * quality).

        Args:
            element_type: Filter by type ("thread", "motif", etc.)
            top_k: Return top K patterns

        Returns:
            List of usage records sorted by heat score
        """
        # Filter by type if specified
        if element_type:
            candidates = [
                r for r in self.usage.values()
                if r.element_type == element_type
            ]
        else:
            candidates = list(self.usage.values())

        # Sort by heat score
        candidates.sort(key=lambda r: r.heat_score, reverse=True)

        return candidates[:top_k]

    def get_cold_patterns(
        self,
        age_threshold: float = 3600.0,  # 1 hour
        top_k: int = 10
    ) -> List[UsageRecord]:
        """
        Get coldest patterns (stale knowledge).

        Cold patterns are:
        - Not accessed recently (age > threshold)
        - Low heat score

        Args:
            age_threshold: Min age in seconds to be considered cold
            top_k: Return top K cold patterns

        Returns:
            List of usage records sorted by age (oldest first)
        """
        candidates = [
            r for r in self.usage.values()
            if r.age_seconds > age_threshold
        ]

        # Sort by age (oldest first)
        candidates.sort(key=lambda r: r.age_seconds, reverse=True)

        return candidates[:top_k]

    def _apply_decay_if_needed(self):
        """Apply heat decay if interval elapsed"""
        now = time.time()
        elapsed = now - self.last_decay

        if elapsed >= self.decay_interval:
            # Apply exponential decay to all heat scores
            for record in self.usage.values():
                # Decay access count (heat proxy)
                record.access_count = int(record.access_count * self.decay_rate)
                record.success_count = int(record.success_count * self.decay_rate)

            self.last_decay = now
            self.logger.debug(
                f"Applied heat decay: {len(self.usage)} elements decayed"
            )

    def prune_cold_patterns(self, min_heat: float = 0.1) -> int:
        """
        Remove cold patterns with very low heat.

        Args:
            min_heat: Minimum heat score to keep

        Returns:
            Number of patterns pruned
        """
        to_remove = [
            element_id for element_id, record in self.usage.items()
            if record.heat_score < min_heat
        ]

        for element_id in to_remove:
            del self.usage[element_id]

        if to_remove:
            self.logger.info(f"Pruned {len(to_remove)} cold patterns")

        self.unique_elements = len(self.usage)

        return len(to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        if not self.usage:
            return {
                "total_accesses": 0,
                "unique_elements": 0,
                "avg_heat": 0.0,
                "hot_threads": 0,
                "hot_motifs": 0
            }

        # Calculate stats
        heat_scores = [r.heat_score for r in self.usage.values()]
        threads = [r for r in self.usage.values() if r.element_type == "thread"]
        motifs = [r for r in self.usage.values() if r.element_type == "motif"]

        return {
            "total_accesses": self.total_accesses,
            "unique_elements": self.unique_elements,
            "avg_heat": sum(heat_scores) / len(heat_scores),
            "max_heat": max(heat_scores),
            "hot_threads": len([t for t in threads if t.heat_score > 1.0]),
            "hot_motifs": len([m for m in motifs if m.heat_score > 1.0]),
            "elements_by_type": Counter(r.element_type for r in self.usage.values())
        }


# ============================================================================
# Adaptive Retrieval
# ============================================================================

@dataclass
class RetrievalWeights:
    """Dynamic retrieval weights based on usage"""
    thread_weights: Dict[str, float] = field(default_factory=dict)
    motif_weights: Dict[str, float] = field(default_factory=dict)
    base_weight: float = 1.0  # Default weight for unknown elements


class AdaptiveRetriever:
    """
    Adjusts retrieval weights based on hot patterns.

    Hot patterns (frequently successful) get boosted weights,
    making them easier to retrieve in future queries.
    """

    def __init__(
        self,
        hot_boost: float = 2.0,  # Boost factor for hot patterns
        cold_penalty: float = 0.5,  # Penalty factor for cold patterns
        heat_threshold: float = 1.0  # Min heat to be considered "hot"
    ):
        """
        Initialize adaptive retriever.

        Args:
            hot_boost: Weight multiplier for hot patterns
            cold_penalty: Weight multiplier for cold patterns
            heat_threshold: Minimum heat score to be considered hot
        """
        self.hot_boost = hot_boost
        self.cold_penalty = cold_penalty
        self.heat_threshold = heat_threshold

        self.weights = RetrievalWeights()

        self.logger = logging.getLogger(f"{__name__}.AdaptiveRetriever")

    def update_weights(self, tracker: HotPatternTracker):
        """
        Update retrieval weights from hot pattern tracker.

        Args:
            tracker: Hot pattern tracker with usage data
        """
        # Get all patterns
        all_patterns = list(tracker.usage.values())

        # Update thread weights
        for record in all_patterns:
            if record.element_type == "thread":
                # Calculate weight based on heat
                if record.heat_score >= self.heat_threshold:
                    weight = self.hot_boost * (record.heat_score / self.heat_threshold)
                else:
                    weight = self.cold_penalty

                self.weights.thread_weights[record.element_id] = weight

        # Update motif weights
        for record in all_patterns:
            if record.element_type == "motif":
                # Remove "motif:" prefix
                motif = record.element_id.replace("motif:", "")

                if record.heat_score >= self.heat_threshold:
                    weight = self.hot_boost * (record.heat_score / self.heat_threshold)
                else:
                    weight = self.cold_penalty

                self.weights.motif_weights[motif] = weight

        self.logger.info(
            f"Updated retrieval weights: {len(self.weights.thread_weights)} threads, "
            f"{len(self.weights.motif_weights)} motifs"
        )

    def get_thread_weight(self, thread_id: str) -> float:
        """Get weight for thread (1.0 if unknown)"""
        return self.weights.thread_weights.get(thread_id, self.weights.base_weight)

    def get_motif_weight(self, motif: str) -> float:
        """Get weight for motif (1.0 if unknown)"""
        return self.weights.motif_weights.get(motif, self.weights.base_weight)

    def boost_shards(
        self,
        shards: List[MemoryShard],
        query_motifs: List[str]
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Apply weights to shards based on hot patterns.

        Returns shards with computed relevance scores incorporating
        both semantic similarity and usage-based weights.

        Args:
            shards: Memory shards to weight
            query_motifs: Detected motifs in query

        Returns:
            List of (shard, relevance_score) tuples
        """
        weighted = []

        for shard in shards:
            # Base relevance (could come from semantic similarity)
            base_relevance = 1.0

            # Boost based on thread (shard ID)
            thread_weight = self.get_thread_weight(shard.id)

            # Boost based on motif overlap
            shard_tags = shard.metadata.get("tags", [])
            motif_weights = [
                self.get_motif_weight(motif)
                for motif in query_motifs
                if motif in shard_tags
            ]

            # Combine weights (geometric mean for balance)
            if motif_weights:
                avg_motif_weight = math.exp(
                    sum(math.log(w) for w in motif_weights) / len(motif_weights)
                )
            else:
                avg_motif_weight = 1.0

            # Final relevance
            relevance = base_relevance * thread_weight * avg_motif_weight

            weighted.append((shard, relevance))

        # Sort by relevance (descending)
        weighted.sort(key=lambda x: x[1], reverse=True)

        return weighted


# ============================================================================
# Hot Pattern Feedback Engine
# ============================================================================

@dataclass
class HotPatternConfig:
    """Configuration for hot pattern feedback"""
    enable_tracking: bool = True
    enable_adaptive_retrieval: bool = True
    update_weights_interval: int = 10  # Update weights every N queries
    decay_rate: float = 0.95
    hot_boost: float = 2.0


class HotPatternFeedbackEngine:
    """
    Wraps LearningLoopEngine with hot pattern feedback.

    Features:
    - Automatic usage tracking from queries
    - Hot pattern detection and weight updates
    - Adaptive retrieval based on usage

    Usage:
        config = Config.fast()
        hot_config = HotPatternConfig(enable_tracking=True)

        async with HotPatternFeedbackEngine(
            cfg=config,
            shards=shards,
            hot_config=hot_config
        ) as engine:
            # Process queries - hot patterns tracked automatically
            spacetime = await engine.weave(query)

            # Check hot patterns
            hot = engine.get_hot_patterns()
            stats = engine.get_hot_stats()
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory: Optional[Any] = None,
        loop_config: Optional[LearningLoopConfig] = None,
        hot_config: Optional[HotPatternConfig] = None
    ):
        """
        Initialize hot pattern feedback engine.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards (optional)
            memory: Dynamic memory backend (optional)
            loop_config: Learning loop configuration
            hot_config: Hot pattern configuration
        """
        self.cfg = cfg
        self.shards = shards or []
        self.memory = memory
        self.loop_config = loop_config or LearningLoopConfig()
        self.hot_config = hot_config or HotPatternConfig()

        # Create components
        self.learning_engine: Optional[LearningLoopEngine] = None
        self.hot_tracker = HotPatternTracker(
            decay_rate=self.hot_config.decay_rate
        )
        self.adaptive_retriever = AdaptiveRetriever(
            hot_boost=self.hot_config.hot_boost
        )

        self.queries_since_weight_update = 0

        self.logger = logging.getLogger(f"{__name__}.HotPatternFeedbackEngine")

    async def __aenter__(self):
        """Async context manager entry"""
        # Create learning engine
        self.learning_engine = LearningLoopEngine(
            cfg=self.cfg,
            shards=self.shards,
            memory=self.memory,
            loop_config=self.loop_config
        )
        await self.learning_engine.__aenter__()

        self.logger.info(
            f"HotPatternFeedbackEngine initialized (tracking={self.hot_config.enable_tracking})"
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.learning_engine:
            await self.learning_engine.__aexit__(exc_type, exc_val, exc_tb)

        # Final weight update
        if self.hot_config.enable_adaptive_retrieval:
            self.adaptive_retriever.update_weights(self.hot_tracker)

        self.logger.info(
            f"HotPatternFeedbackEngine closed (tracked {self.hot_tracker.total_accesses} accesses)"
        )

    async def weave(self, query: Query) -> Spacetime:
        """
        Process query with hot pattern tracking and adaptive retrieval.

        Steps:
        1. Weave query through learning engine
        2. Track usage in hot pattern tracker
        3. Update retrieval weights periodically

        Args:
            query: Query to process

        Returns:
            Spacetime from weaving
        """
        if not self.learning_engine:
            raise RuntimeError("Engine not initialized. Use async context manager.")

        # Weave query
        spacetime = await self.learning_engine.weave_and_learn(query)

        # Track usage
        if self.hot_config.enable_tracking:
            self.hot_tracker.record_usage(spacetime)

        # Update weights periodically
        self.queries_since_weight_update += 1
        if (
            self.hot_config.enable_adaptive_retrieval and
            self.queries_since_weight_update >= self.hot_config.update_weights_interval
        ):
            self.adaptive_retriever.update_weights(self.hot_tracker)
            self.queries_since_weight_update = 0

            self.logger.debug("Updated retrieval weights from hot patterns")

        return spacetime

    def get_hot_patterns(
        self,
        element_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[UsageRecord]:
        """Get hottest patterns"""
        return self.hot_tracker.get_hot_patterns(element_type, top_k)

    def get_hot_stats(self) -> Dict[str, Any]:
        """Get complete statistics"""
        learning_stats = self.learning_engine.get_learning_stats() if self.learning_engine else {}
        hot_stats = self.hot_tracker.get_statistics()

        return {
            **learning_stats,
            **hot_stats,
            "adaptive_retrieval_enabled": self.hot_config.enable_adaptive_retrieval,
            "weight_updates": self.queries_since_weight_update
        }

    def get_weighted_shards(
        self,
        query_motifs: List[str],
        top_k: int = 5
    ) -> List[Tuple[MemoryShard, float]]:
        """Get shards weighted by hot patterns"""
        weighted = self.adaptive_retriever.boost_shards(self.shards, query_motifs)
        return weighted[:top_k]


if __name__ == "__main__":
    print("HoloLoom Recursive Learning - Phase 3: Hot Pattern Feedback")
    print()
    print("Components:")
    print("  - HotPatternTracker: Track usage and calculate heat scores")
    print("  - AdaptiveRetriever: Adjust retrieval weights based on usage")
    print("  - HotPatternFeedbackEngine: Complete usage-based learning")
    print()
    print("Usage:")
    print("""
from HoloLoom.recursive import HotPatternFeedbackEngine, HotPatternConfig
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

config = Config.fast()
hot_config = HotPatternConfig(
    enable_tracking=True,
    enable_adaptive_retrieval=True
)

async with HotPatternFeedbackEngine(
    cfg=config,
    shards=shards,
    hot_config=hot_config
) as engine:
    # Process queries - hot patterns tracked automatically
    for query_text in queries:
        spacetime = await engine.weave(Query(text=query_text))

    # Check what's hot
    hot_threads = engine.get_hot_patterns(element_type="thread", top_k=5)
    hot_motifs = engine.get_hot_patterns(element_type="motif", top_k=5)

    # Get statistics
    stats = engine.get_hot_stats()
    print(f"Hot threads: {stats['hot_threads']}")
    print(f"Hot motifs: {stats['hot_motifs']}")
""")
