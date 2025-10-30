#!/usr/bin/env python3
"""
Loop Engine Integration - Phase 2 of Recursive Learning Vision
================================================================
Connects HoloLoom + Scratchpad with continuous learning loop engine.

Architecture:
    LearningLoopEngine: Wraps Scratchpad Orchestrator with continuous learning
    PatternExtractor: Extracts learnable patterns from Spacetime traces
    PatternLearner: Learns from accumulated patterns to improve future queries

Philosophy:
    The system learns from usage patterns by feeding successful weaving
    cycles back into pattern recognition. Hot patterns (frequently successful
    approaches) get reinforced, weak patterns get pruned.

Author: Claude Code
Date: 2025-10-29 (Phase 2 Implementation)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import time

# HoloLoom components
from HoloLoom.recursive.scratchpad_integration import (
    ScratchpadOrchestrator,
    ScratchpadConfig
)
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config

# Promptly components
from Promptly.promptly.recursive_loops import Scratchpad

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Extraction
# ============================================================================

@dataclass
class LearnedPattern:
    """A pattern learned from successful weaving cycles"""
    motifs: List[str]  # Detected motifs
    threads: List[str]  # Activated threads
    tool: str  # Tool selected
    adapter: str  # Adapter used
    confidence: float  # Confidence achieved
    query_type: str  # Classified query type (e.g., "factual", "procedural")
    occurrences: int = 1  # How many times seen
    avg_confidence: float = 0.0  # Average confidence when using this pattern
    last_seen: float = field(default_factory=time.time)

    def __hash__(self):
        """Hash by core pattern elements"""
        return hash((
            tuple(sorted(self.motifs[:3])),  # Top 3 motifs
            tuple(sorted(self.threads[:3])),  # Top 3 threads
            self.tool
        ))

    def update(self, new_confidence: float):
        """Update pattern with new occurrence"""
        self.occurrences += 1
        self.avg_confidence = (
            (self.avg_confidence * (self.occurrences - 1) + new_confidence) /
            self.occurrences
        )
        self.last_seen = time.time()


class PatternExtractor:
    """
    Extracts learnable patterns from Spacetime traces.

    Identifies:
    - Successful motif + thread combinations
    - Effective tool selections for query types
    - High-confidence adapter choices
    """

    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize pattern extractor.

        Args:
            confidence_threshold: Minimum confidence to consider pattern successful
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.PatternExtractor")

    def extract(self, spacetime: Spacetime) -> Optional[LearnedPattern]:
        """
        Extract pattern from successful weaving cycle.

        Only extracts patterns from high-confidence results,
        as these represent successful reasoning paths.

        Args:
            spacetime: Woven fabric to extract pattern from

        Returns:
            LearnedPattern if extraction successful, None otherwise
        """
        trace = spacetime.trace

        # Only learn from successful cycles
        if trace.tool_confidence < self.confidence_threshold:
            self.logger.debug(
                f"Skipping pattern extraction: confidence {trace.tool_confidence:.2f} < "
                f"{self.confidence_threshold}"
            )
            return None

        # Classify query type (simple heuristics)
        query_type = self._classify_query(spacetime.query_text)

        # Extract pattern
        pattern = LearnedPattern(
            motifs=trace.motifs_detected[:5],  # Top 5 motifs
            threads=trace.threads_activated[:5],  # Top 5 threads
            tool=trace.tool_selected,
            adapter=trace.policy_adapter,
            confidence=trace.tool_confidence,
            query_type=query_type,
            avg_confidence=trace.tool_confidence
        )

        self.logger.debug(
            f"Extracted pattern: {len(pattern.motifs)} motifs, {len(pattern.threads)} threads, "
            f"tool={pattern.tool}, type={query_type}"
        )

        return pattern

    def _classify_query(self, query_text: str) -> str:
        """
        Classify query type using simple heuristics.

        Types:
        - factual: "What is...", "Define..."
        - procedural: "How to...", "How does..."
        - analytical: "Why...", "Explain..."
        - comparative: "Compare...", "Difference..."
        - exploratory: "Tell me about...", "Describe..."
        """
        query_lower = query_text.lower()

        if query_lower.startswith(("what is", "what are", "define")):
            return "factual"
        elif query_lower.startswith(("how to", "how does", "how do")):
            return "procedural"
        elif query_lower.startswith(("why", "explain")):
            return "analytical"
        elif "compare" in query_lower or "difference" in query_lower:
            return "comparative"
        elif query_lower.startswith(("tell me", "describe")):
            return "exploratory"
        else:
            return "general"


# ============================================================================
# Pattern Learning
# ============================================================================

@dataclass
class LearningStats:
    """Statistics for pattern learning"""
    patterns_learned: int = 0
    patterns_updated: int = 0
    unique_patterns: int = 0
    hot_patterns_count: int = 0
    avg_pattern_confidence: float = 0.0
    queries_processed: int = 0
    learning_rate: float = 0.0  # Patterns per query


class PatternLearner:
    """
    Learns from accumulated patterns to improve future queries.

    Features:
    - Pattern accumulation and deduplication
    - Hot pattern detection (frequently successful)
    - Pattern pruning (remove stale/weak patterns)
    - Query suggestions based on learned patterns
    """

    def __init__(
        self,
        hot_threshold: int = 5,  # Min occurrences to be "hot"
        prune_age: float = 3600.0,  # Prune patterns older than 1 hour
        prune_confidence: float = 0.6  # Prune patterns with avg confidence < this
    ):
        """
        Initialize pattern learner.

        Args:
            hot_threshold: Minimum occurrences to consider pattern "hot"
            prune_age: Remove patterns not seen in this many seconds
            prune_confidence: Remove patterns with avg confidence below this
        """
        self.hot_threshold = hot_threshold
        self.prune_age = prune_age
        self.prune_confidence = prune_confidence

        # Pattern storage (pattern hash -> pattern)
        self.patterns: Dict[int, LearnedPattern] = {}

        # Index by query type for fast lookup
        self.patterns_by_type: Dict[str, Set[int]] = defaultdict(set)

        # Statistics
        self.stats = LearningStats()

        self.logger = logging.getLogger(f"{__name__}.PatternLearner")

    def learn(self, pattern: LearnedPattern):
        """
        Learn from a pattern.

        If pattern exists, update occurrence count and confidence.
        If new, add to learned patterns.

        Args:
            pattern: Pattern to learn from
        """
        pattern_hash = hash(pattern)

        if pattern_hash in self.patterns:
            # Update existing pattern
            existing = self.patterns[pattern_hash]
            existing.update(pattern.confidence)
            self.stats.patterns_updated += 1

            self.logger.debug(
                f"Updated pattern: {existing.occurrences} occurrences, "
                f"avg_confidence={existing.avg_confidence:.2f}"
            )
        else:
            # Learn new pattern
            self.patterns[pattern_hash] = pattern
            self.patterns_by_type[pattern.query_type].add(pattern_hash)
            self.stats.patterns_learned += 1

            self.logger.debug(
                f"Learned new pattern: type={pattern.query_type}, "
                f"motifs={len(pattern.motifs)}, threads={len(pattern.threads)}"
            )

        # Update statistics
        self.stats.unique_patterns = len(self.patterns)
        self.stats.queries_processed += 1
        self.stats.learning_rate = self.stats.patterns_learned / self.stats.queries_processed

        # Calculate average confidence
        if self.patterns:
            self.stats.avg_pattern_confidence = sum(
                p.avg_confidence for p in self.patterns.values()
            ) / len(self.patterns)

    def get_hot_patterns(self, query_type: Optional[str] = None) -> List[LearnedPattern]:
        """
        Get hot patterns (frequently successful).

        Hot patterns are those with:
        - High occurrence count (>= hot_threshold)
        - High average confidence
        - Recent usage

        Args:
            query_type: Filter by query type (optional)

        Returns:
            List of hot patterns sorted by occurrence count
        """
        # Filter patterns
        if query_type:
            pattern_hashes = self.patterns_by_type.get(query_type, set())
            candidates = [self.patterns[h] for h in pattern_hashes]
        else:
            candidates = list(self.patterns.values())

        # Filter hot patterns
        hot = [
            p for p in candidates
            if p.occurrences >= self.hot_threshold and
               p.avg_confidence >= self.prune_confidence
        ]

        # Sort by occurrences (descending)
        hot.sort(key=lambda p: p.occurrences, reverse=True)

        self.stats.hot_patterns_count = len(hot)

        return hot

    def prune_stale_patterns(self) -> int:
        """
        Remove stale and weak patterns.

        Prunes patterns that:
        - Haven't been seen in prune_age seconds
        - Have low average confidence (< prune_confidence)

        Returns:
            Number of patterns pruned
        """
        now = time.time()
        to_remove = []

        for pattern_hash, pattern in self.patterns.items():
            age = now - pattern.last_seen

            if age > self.prune_age or pattern.avg_confidence < self.prune_confidence:
                to_remove.append(pattern_hash)

        # Remove from main storage
        for pattern_hash in to_remove:
            pattern = self.patterns[pattern_hash]
            del self.patterns[pattern_hash]

            # Remove from type index
            self.patterns_by_type[pattern.query_type].discard(pattern_hash)

        if to_remove:
            self.logger.info(f"Pruned {len(to_remove)} stale/weak patterns")

        self.stats.unique_patterns = len(self.patterns)

        return len(to_remove)

    def suggest_improvements(self, query_text: str) -> Dict[str, Any]:
        """
        Suggest improvements for a query based on learned patterns.

        Args:
            query_text: Query to analyze

        Returns:
            Dict with suggestions
        """
        query_type = PatternExtractor()._classify_query(query_text)
        hot = self.get_hot_patterns(query_type)

        if not hot:
            return {
                "has_suggestions": False,
                "query_type": query_type,
                "message": "No patterns learned yet for this query type"
            }

        # Analyze hot patterns
        common_motifs = Counter()
        common_threads = Counter()
        common_tools = Counter()

        for pattern in hot:
            for motif in pattern.motifs:
                common_motifs[motif] += pattern.occurrences
            for thread in pattern.threads:
                common_threads[thread] += pattern.occurrences
            common_tools[pattern.tool] += pattern.occurrences

        return {
            "has_suggestions": True,
            "query_type": query_type,
            "hot_patterns_count": len(hot),
            "suggested_motifs": [m for m, _ in common_motifs.most_common(3)],
            "suggested_threads": [t for t, _ in common_threads.most_common(3)],
            "suggested_tool": common_tools.most_common(1)[0][0] if common_tools else None,
            "confidence_estimate": sum(p.avg_confidence for p in hot) / len(hot)
        }


# ============================================================================
# Learning Loop Engine
# ============================================================================

@dataclass
class LearningLoopConfig:
    """Configuration for learning loop"""
    enable_learning: bool = True
    auto_prune: bool = True
    prune_interval: int = 100  # Prune every N queries
    hot_threshold: int = 5
    confidence_threshold: float = 0.75


class LearningLoopEngine:
    """
    Wraps ScratchpadOrchestrator with continuous pattern learning.

    Features:
    - Automatic pattern extraction from successful queries
    - Pattern learning and hot pattern detection
    - Periodic pruning of stale patterns
    - Learning statistics tracking

    Usage:
        config = Config.fast()
        loop_config = LearningLoopConfig(enable_learning=True)

        async with LearningLoopEngine(
            cfg=config,
            shards=shards,
            loop_config=loop_config
        ) as engine:
            # Process queries - learning happens automatically
            spacetime = await engine.weave_and_learn(query)

            # Check learning progress
            stats = engine.get_learning_stats()
            hot_patterns = engine.get_hot_patterns()
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory: Optional[Any] = None,
        scratchpad_config: Optional[ScratchpadConfig] = None,
        loop_config: Optional[LearningLoopConfig] = None
    ):
        """
        Initialize learning loop engine.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards (optional)
            memory: Dynamic memory backend (optional)
            scratchpad_config: Scratchpad configuration
            loop_config: Learning loop configuration
        """
        self.cfg = cfg
        self.shards = shards or []
        self.memory = memory
        self.scratchpad_config = scratchpad_config or ScratchpadConfig()
        self.loop_config = loop_config or LearningLoopConfig()

        # Create components
        self.orchestrator: Optional[ScratchpadOrchestrator] = None
        self.pattern_extractor = PatternExtractor(
            confidence_threshold=self.loop_config.confidence_threshold
        )
        self.pattern_learner = PatternLearner(
            hot_threshold=self.loop_config.hot_threshold
        )

        self.queries_since_prune = 0

        self.logger = logging.getLogger(f"{__name__}.LearningLoopEngine")

    async def __aenter__(self):
        """Async context manager entry"""
        # Create orchestrator
        self.orchestrator = ScratchpadOrchestrator(
            cfg=self.cfg,
            shards=self.shards,
            memory=self.memory,
            scratchpad_config=self.scratchpad_config
        )
        await self.orchestrator.__aenter__()

        self.logger.info(
            f"LearningLoopEngine initialized (learning={self.loop_config.enable_learning})"
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

        # Final prune
        if self.loop_config.auto_prune:
            pruned = self.pattern_learner.prune_stale_patterns()
            if pruned > 0:
                self.logger.info(f"Final prune: removed {pruned} patterns")

        self.logger.info(
            f"LearningLoopEngine closed (learned {self.pattern_learner.stats.patterns_learned} patterns)"
        )

    async def weave(self, query: Query) -> Spacetime:
        """Standard weave interface (for compatibility)"""
        return await self.weave_and_learn(query)

    async def weave_and_learn(self, query: Query) -> Spacetime:
        """
        Process query with automatic pattern learning.

        Steps:
        1. Weave query through HoloLoom
        2. Extract pattern from result (if successful)
        3. Learn from pattern
        4. Auto-prune if needed

        Args:
            query: Query to process

        Returns:
            Spacetime from weaving
        """
        if not self.orchestrator:
            raise RuntimeError("Engine not initialized. Use async context manager.")

        # Weave query
        spacetime, scratchpad = await self.orchestrator.weave_with_provenance(query)

        # Learn from result
        if self.loop_config.enable_learning:
            pattern = self.pattern_extractor.extract(spacetime)

            if pattern:
                self.pattern_learner.learn(pattern)
                self.logger.debug(
                    f"Learned pattern from query (confidence={pattern.confidence:.2f})"
                )

        # Auto-prune check
        self.queries_since_prune += 1
        if (
            self.loop_config.auto_prune and
            self.queries_since_prune >= self.loop_config.prune_interval
        ):
            pruned = self.pattern_learner.prune_stale_patterns()
            self.queries_since_prune = 0

            if pruned > 0:
                self.logger.info(f"Auto-pruned {pruned} stale patterns")

        return spacetime

    def get_hot_patterns(self, query_type: Optional[str] = None) -> List[LearnedPattern]:
        """Get hot patterns (frequently successful)"""
        return self.pattern_learner.get_hot_patterns(query_type)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get complete learning statistics"""
        orchestrator_stats = self.orchestrator.get_statistics() if self.orchestrator else {}
        learning_stats = self.pattern_learner.stats

        return {
            # Orchestrator stats
            "queries_processed": orchestrator_stats.get("queries_processed", 0),
            "refinements_triggered": orchestrator_stats.get("refinements_triggered", 0),
            "avg_confidence": orchestrator_stats.get("avg_confidence", 0.0),

            # Learning stats
            "patterns_learned": learning_stats.patterns_learned,
            "patterns_updated": learning_stats.patterns_updated,
            "unique_patterns": learning_stats.unique_patterns,
            "hot_patterns_count": learning_stats.hot_patterns_count,
            "avg_pattern_confidence": learning_stats.avg_pattern_confidence,
            "learning_rate": learning_stats.learning_rate,

            # Query distribution by type
            "patterns_by_type": {
                qtype: len(hashes)
                for qtype, hashes in self.pattern_learner.patterns_by_type.items()
            }
        }

    def suggest_improvements(self, query_text: str) -> Dict[str, Any]:
        """Suggest improvements based on learned patterns"""
        return self.pattern_learner.suggest_improvements(query_text)


# ============================================================================
# Convenience Functions
# ============================================================================

async def weave_with_learning(
    query: Query,
    cfg: Config,
    shards: Optional[List[MemoryShard]] = None,
    enable_learning: bool = True
) -> Tuple[Spacetime, Dict[str, Any]]:
    """
    Convenience function for one-off weaving with learning.

    Args:
        query: Query to process
        cfg: HoloLoom configuration
        shards: Memory shards
        enable_learning: Enable pattern learning

    Returns:
        Tuple of (Spacetime, learning_stats)

    Usage:
        from HoloLoom.recursive import weave_with_learning

        spacetime, stats = await weave_with_learning(
            Query(text="How does Thompson Sampling work?"),
            Config.fast(),
            shards=shards
        )

        print(f"Learned {stats['patterns_learned']} patterns")
    """
    loop_config = LearningLoopConfig(enable_learning=enable_learning)

    async with LearningLoopEngine(
        cfg=cfg,
        shards=shards,
        loop_config=loop_config
    ) as engine:
        spacetime = await engine.weave_and_learn(query)
        stats = engine.get_learning_stats()

        return spacetime, stats


if __name__ == "__main__":
    print("HoloLoom Recursive Learning - Phase 2: Loop Engine Integration")
    print()
    print("Components:")
    print("  - PatternExtractor: Extract patterns from Spacetime traces")
    print("  - PatternLearner: Learn and maintain pattern library")
    print("  - LearningLoopEngine: Automatic learning from usage")
    print()
    print("Usage:")
    print("""
from HoloLoom.recursive import LearningLoopEngine, LearningLoopConfig
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

config = Config.fast()
loop_config = LearningLoopConfig(
    enable_learning=True,
    hot_threshold=5
)

async with LearningLoopEngine(
    cfg=config,
    shards=shards,
    loop_config=loop_config
) as engine:
    # Process multiple queries - learning happens automatically
    for query_text in queries:
        spacetime = await engine.weave_and_learn(Query(text=query_text))

    # Check what was learned
    stats = engine.get_learning_stats()
    hot_patterns = engine.get_hot_patterns()

    print(f"Learned {stats['patterns_learned']} patterns")
    print(f"Hot patterns: {stats['hot_patterns_count']}")
""")
