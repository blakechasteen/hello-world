#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Learning from Outcomes - Outcome Tracker

Tracks outcomes from context packing + LLM generation to learn optimal thresholds:

1. Quality metrics (coherence, completeness, relevance)
2. Budget efficiency (quality per token)
3. Latency characteristics
4. Importance threshold correlation

Enables data-driven tuning of:
- Compression thresholds (when to compress?)
- Importance cutoffs (what's "good enough"?)
- Budget allocation (are we over/under-allocating?)

Usage:
    tracker = OutcomeTracker()

    # Track outcome
    await tracker.track(
        query="What is X?",
        budget_used=5000,
        quality_score=0.92,
        latency_ms=150,
        compression_level="SUMMARY",
        importance_threshold=0.3
    )

    # Get insights
    insights = tracker.get_insights()
    print(f"Optimal importance threshold: {insights['optimal_importance_threshold']}")
    print(f"Recommended compression: {insights['recommended_compression']}")
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from collections import defaultdict


class CompressionLevel(str, Enum):
    """Compression levels from context_packer.py"""
    FULL = "full"
    DETAILED = "detailed"
    SUMMARY = "summary"
    MINIMAL = "minimal"


@dataclass
class Outcome:
    """
    Single outcome from context packing + LLM generation.

    Tracks all relevant metrics for learning.
    """

    # Query characteristics
    query: str
    query_complexity: str  # SIMPLE/MODERATE/COMPLEX/RESEARCH

    # Budget
    budget_allocated: int  # Tokens allocated
    budget_used: int       # Tokens actually used
    budget_efficiency: float  # quality / tokens (quality per token)

    # Quality metrics
    quality_overall: float       # 0.0-1.0
    quality_coherence: float     # 0.0-1.0
    quality_completeness: float  # 0.0-1.0
    quality_relevance: float     # 0.0-1.0

    # Performance metrics
    latency_ms: float           # Total latency
    latency_packing_ms: float   # Packing only
    latency_llm_ms: float       # LLM only

    # Packing strategy
    compression_level: CompressionLevel
    importance_threshold: float  # Minimum importance score
    memories_available: int      # Total memories available
    memories_included: int       # Memories actually included

    # Context utilization
    context_used_pct: float     # Percentage of context elements used in response

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    provider: str = "unknown"   # LLM provider
    model: str = "unknown"      # LLM model

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.budget_used > 0:
            self.budget_efficiency = self.quality_overall / self.budget_used
        else:
            self.budget_efficiency = 0.0


@dataclass
class ThresholdRecommendation:
    """
    Recommended threshold based on learning.

    Includes confidence and reasoning.
    """

    parameter: str               # "importance_threshold", "compression_level", etc.
    recommended_value: Any       # Recommended value
    current_value: Any           # Current value
    confidence: float            # 0.0-1.0 (how confident are we?)
    expected_improvement: float  # Expected quality improvement
    reasoning: str               # Human-readable explanation
    sample_size: int             # Number of samples this is based on


class OutcomeTracker:
    """
    Track outcomes from context packing + LLM generation.

    Learns optimal thresholds for:
    - Compression levels
    - Importance cutoffs
    - Budget allocation
    """

    def __init__(
        self,
        min_samples_for_recommendation: int = 10,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize outcome tracker.

        Args:
            min_samples_for_recommendation: Minimum samples needed for confident recommendations
            enable_persistence: Save outcomes to disk
            persistence_path: Path to save outcomes (if enabled)
        """
        self.min_samples = min_samples_for_recommendation
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path

        # Storage
        self.outcomes: List[Outcome] = []

        # Indexed lookups (for fast analysis)
        self.by_complexity: Dict[str, List[Outcome]] = defaultdict(list)
        self.by_compression: Dict[CompressionLevel, List[Outcome]] = defaultdict(list)
        self.by_provider: Dict[str, List[Outcome]] = defaultdict(list)

    async def track(
        self,
        query: str,
        query_complexity: str,
        budget_allocated: int,
        budget_used: int,
        quality_overall: float,
        quality_coherence: float = 0.0,
        quality_completeness: float = 0.0,
        quality_relevance: float = 0.0,
        latency_ms: float = 0.0,
        latency_packing_ms: float = 0.0,
        latency_llm_ms: float = 0.0,
        compression_level: CompressionLevel = CompressionLevel.DETAILED,
        importance_threshold: float = 0.0,
        memories_available: int = 0,
        memories_included: int = 0,
        context_used_pct: float = 0.0,
        provider: str = "unknown",
        model: str = "unknown"
    ) -> Outcome:
        """
        Track a single outcome.

        Args:
            query: Query text
            query_complexity: Complexity level (SIMPLE/MODERATE/COMPLEX/RESEARCH)
            budget_allocated: Tokens allocated
            budget_used: Tokens actually used
            quality_overall: Overall quality score (0.0-1.0)
            quality_coherence: Coherence score
            quality_completeness: Completeness score
            quality_relevance: Relevance score
            latency_ms: Total latency
            latency_packing_ms: Packing latency
            latency_llm_ms: LLM latency
            compression_level: Compression level used
            importance_threshold: Importance threshold used
            memories_available: Total memories available
            memories_included: Memories included in context
            context_used_pct: Percentage of context used in response
            provider: LLM provider
            model: LLM model

        Returns:
            Outcome object
        """
        outcome = Outcome(
            query=query,
            query_complexity=query_complexity,
            budget_allocated=budget_allocated,
            budget_used=budget_used,
            budget_efficiency=0.0,  # Calculated in __post_init__
            quality_overall=quality_overall,
            quality_coherence=quality_coherence,
            quality_completeness=quality_completeness,
            quality_relevance=quality_relevance,
            latency_ms=latency_ms,
            latency_packing_ms=latency_packing_ms,
            latency_llm_ms=latency_llm_ms,
            compression_level=compression_level,
            importance_threshold=importance_threshold,
            memories_available=memories_available,
            memories_included=memories_included,
            context_used_pct=context_used_pct,
            provider=provider,
            model=model
        )

        # Store
        self.outcomes.append(outcome)

        # Index
        self.by_complexity[query_complexity].append(outcome)
        self.by_compression[compression_level].append(outcome)
        self.by_provider[provider].append(outcome)

        # Persist (if enabled)
        if self.enable_persistence and self.persistence_path:
            await self._persist(outcome)

        return outcome

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            Statistics dict
        """
        if not self.outcomes:
            return {"total_outcomes": 0}

        qualities = [o.quality_overall for o in self.outcomes]
        efficiencies = [o.budget_efficiency for o in self.outcomes]
        latencies = [o.latency_ms for o in self.outcomes if o.latency_ms > 0]

        return {
            "total_outcomes": len(self.outcomes),
            "avg_quality": statistics.mean(qualities),
            "std_quality": statistics.stdev(qualities) if len(qualities) > 1 else 0.0,
            "avg_efficiency": statistics.mean(efficiencies),
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "complexity_distribution": {
                k: len(v) for k, v in self.by_complexity.items()
            },
            "compression_distribution": {
                k.value: len(v) for k, v in self.by_compression.items()
            }
        }

    def get_optimal_importance_threshold(
        self,
        complexity: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Get optimal importance threshold (threshold, expected_quality).

        Args:
            complexity: Filter by complexity level (optional)

        Returns:
            (optimal_threshold, expected_quality) or None if not enough data
        """
        outcomes = self.outcomes if complexity is None else self.by_complexity.get(complexity, [])

        if len(outcomes) < self.min_samples:
            return None

        # Group by threshold (rounded to nearest 0.1)
        threshold_groups: Dict[float, List[Outcome]] = defaultdict(list)
        for outcome in outcomes:
            rounded = round(outcome.importance_threshold, 1)
            threshold_groups[rounded].append(outcome)

        # Find threshold with highest average quality
        best_threshold = None
        best_quality = 0.0

        for threshold, group in threshold_groups.items():
            if len(group) < 3:  # Need at least 3 samples
                continue

            avg_quality = statistics.mean(o.quality_overall for o in group)
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_threshold = threshold

        if best_threshold is None:
            return None

        return (best_threshold, best_quality)

    def get_optimal_compression_level(
        self,
        complexity: Optional[str] = None
    ) -> Optional[Tuple[CompressionLevel, float]]:
        """
        Get optimal compression level (level, expected_quality).

        Args:
            complexity: Filter by complexity level (optional)

        Returns:
            (optimal_level, expected_quality) or None if not enough data
        """
        outcomes = self.outcomes if complexity is None else self.by_complexity.get(complexity, [])

        if len(outcomes) < self.min_samples:
            return None

        # Group by compression level
        level_groups: Dict[CompressionLevel, List[Outcome]] = defaultdict(list)
        for outcome in outcomes:
            level_groups[outcome.compression_level].append(outcome)

        # Find level with highest average quality
        best_level = None
        best_quality = 0.0

        for level, group in level_groups.items():
            if len(group) < 3:  # Need at least 3 samples
                continue

            avg_quality = statistics.mean(o.quality_overall for o in group)
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_level = level

        if best_level is None:
            return None

        return (best_level, best_quality)

    def get_budget_efficiency_insights(
        self,
        complexity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze budget efficiency (quality per token).

        Args:
            complexity: Filter by complexity level (optional)

        Returns:
            Insights dict with recommendations
        """
        outcomes = self.outcomes if complexity is None else self.by_complexity.get(complexity, [])

        if len(outcomes) < self.min_samples:
            return {"error": "Not enough data"}

        efficiencies = [o.budget_efficiency for o in outcomes]
        budgets = [o.budget_used for o in outcomes]
        qualities = [o.quality_overall for o in outcomes]

        # Find "sweet spot" budget (best efficiency)
        budget_ranges = [
            (0, 3000),
            (3000, 6000),
            (6000, 10000),
            (10000, 20000),
            (20000, float('inf'))
        ]

        range_efficiencies = {}
        for min_b, max_b in budget_ranges:
            in_range = [o for o in outcomes if min_b <= o.budget_used < max_b]
            if len(in_range) >= 3:
                avg_eff = statistics.mean(o.budget_efficiency for o in in_range)
                range_efficiencies[f"{min_b}-{max_b}"] = avg_eff

        return {
            "avg_efficiency": statistics.mean(efficiencies),
            "avg_budget_used": statistics.mean(budgets),
            "avg_quality": statistics.mean(qualities),
            "efficiency_by_budget_range": range_efficiencies,
            "sample_size": len(outcomes)
        }

    def get_recommendations(
        self,
        complexity: Optional[str] = None,
        current_importance_threshold: float = 0.0,
        current_compression_level: CompressionLevel = CompressionLevel.DETAILED
    ) -> List[ThresholdRecommendation]:
        """
        Get threshold recommendations based on learning.

        Args:
            complexity: Filter by complexity level
            current_importance_threshold: Current importance threshold
            current_compression_level: Current compression level

        Returns:
            List of recommendations
        """
        recommendations = []

        # Importance threshold recommendation
        optimal_threshold = self.get_optimal_importance_threshold(complexity)
        if optimal_threshold:
            threshold, expected_quality = optimal_threshold

            # Only recommend if significantly different
            if abs(threshold - current_importance_threshold) >= 0.1:
                outcomes = self.outcomes if complexity is None else self.by_complexity.get(complexity, [])

                recommendations.append(ThresholdRecommendation(
                    parameter="importance_threshold",
                    recommended_value=threshold,
                    current_value=current_importance_threshold,
                    confidence=min(len(outcomes) / (self.min_samples * 3), 1.0),
                    expected_improvement=expected_quality - 0.5,  # Assuming baseline of 0.5
                    reasoning=f"Based on {len(outcomes)} outcomes, threshold {threshold} achieves "
                              f"{expected_quality:.2f} average quality (current: {current_importance_threshold})",
                    sample_size=len(outcomes)
                ))

        # Compression level recommendation
        optimal_compression = self.get_optimal_compression_level(complexity)
        if optimal_compression:
            level, expected_quality = optimal_compression

            if level != current_compression_level:
                outcomes = self.outcomes if complexity is None else self.by_complexity.get(complexity, [])

                recommendations.append(ThresholdRecommendation(
                    parameter="compression_level",
                    recommended_value=level,
                    current_value=current_compression_level,
                    confidence=min(len(outcomes) / (self.min_samples * 3), 1.0),
                    expected_improvement=expected_quality - 0.5,
                    reasoning=f"Based on {len(outcomes)} outcomes, {level.value} compression achieves "
                              f"{expected_quality:.2f} average quality (current: {current_compression_level.value})",
                    sample_size=len(outcomes)
                ))

        return recommendations

    async def _persist(self, outcome: Outcome):
        """Persist outcome to disk (if enabled)"""
        # TODO: Implement persistence (JSONL format)
        pass

    def clear(self):
        """Clear all outcomes"""
        self.outcomes.clear()
        self.by_complexity.clear()
        self.by_compression.clear()
        self.by_provider.clear()
