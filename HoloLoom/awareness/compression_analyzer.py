#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Context Compression - Compression Analyzer

Measures compression effectiveness and provides insights.

Tracks:
1. Token savings (before/after)
2. Quality impact (did compression hurt quality?)
3. Compression patterns (what works best?)
4. ROI analysis (is compression worth the overhead?)

Usage:
    analyzer = CompressionAnalyzer()

    # Track compression
    analyzer.track_compression(
        original_tokens=5000,
        compressed_tokens=3500,
        quality_before=0.90,
        quality_after=0.88,
        strategy="merge_similar"
    )

    # Get insights
    insights = analyzer.get_insights()
    print(insights["overall_savings_pct"])  # e.g., 30%
    print(insights["quality_preservation"])  # e.g., 0.98
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompressionRecord:
    """Single compression operation record"""

    timestamp: float
    strategy: str  # "merge_similar", "keep_best", etc.
    original_tokens: int
    compressed_tokens: int
    token_savings: int
    compression_ratio: float  # compressed / original
    quality_before: Optional[float] = None  # 0.0-1.0
    quality_after: Optional[float] = None
    quality_delta: Optional[float] = None  # after - before
    latency_ms: Optional[float] = None  # Compression overhead
    duplicate_groups_found: int = 0
    elements_removed: int = 0
    elements_merged: int = 0

    @property
    def savings_pct(self) -> float:
        """Savings as percentage"""
        return (1 - self.compression_ratio) * 100

    @property
    def quality_preserved(self) -> bool:
        """Did quality stay above 95% of original?"""
        if self.quality_before is None or self.quality_after is None:
            return True  # Unknown, assume preserved
        return self.quality_after >= self.quality_before * 0.95


@dataclass
class CompressionInsights:
    """Insights from compression analysis"""

    # Overall statistics
    total_compressions: int
    total_tokens_saved: int
    avg_compression_ratio: float  # e.g., 0.7 = 30% savings
    avg_savings_pct: float  # e.g., 30.0

    # Quality impact
    avg_quality_before: Optional[float]
    avg_quality_after: Optional[float]
    avg_quality_delta: Optional[float]
    quality_preserved_pct: float  # % of compressions that preserved quality

    # Performance
    avg_latency_ms: Optional[float]
    total_latency_ms: float

    # Effectiveness by strategy
    strategy_performance: Dict[str, Dict[str, float]]  # strategy → metrics

    # Recommendations
    best_strategy: Optional[str]  # Strategy with best quality/savings tradeoff
    recommended_threshold: Optional[float]  # Similarity threshold recommendation

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"Compression Insights:",
            f"  Total Operations: {self.total_compressions}",
            f"  Tokens Saved: {self.total_tokens_saved:,} ({self.avg_savings_pct:.1f}%)",
            f"  Avg Compression: {self.avg_compression_ratio:.2f} ({(1-self.avg_compression_ratio)*100:.1f}% reduction)",
        ]

        if self.avg_quality_before is not None:
            lines.append(f"  Quality Impact: {self.avg_quality_before:.3f} → {self.avg_quality_after:.3f} ({self.avg_quality_delta:+.3f})")
            lines.append(f"  Quality Preserved: {self.quality_preserved_pct:.1f}%")

        if self.avg_latency_ms is not None:
            lines.append(f"  Avg Latency: {self.avg_latency_ms:.1f}ms")

        if self.best_strategy:
            lines.append(f"  Best Strategy: {self.best_strategy}")

        return '\n'.join(lines)


class CompressionAnalyzer:
    """
    Analyze compression effectiveness.

    Tracks compression operations and provides insights on:
    - Token savings
    - Quality impact
    - Strategy effectiveness
    - Performance overhead
    """

    def __init__(
        self,
        track_quality: bool = True,
        track_latency: bool = True
    ):
        """
        Initialize compression analyzer.

        Args:
            track_quality: Track quality impact of compression
            track_latency: Track compression latency overhead
        """
        self.track_quality = track_quality
        self.track_latency = track_latency

        # Storage
        self.records: List[CompressionRecord] = []

        # Indexed by strategy
        self.by_strategy: Dict[str, List[CompressionRecord]] = defaultdict(list)

    def track_compression(
        self,
        original_tokens: int,
        compressed_tokens: int,
        strategy: str = "unknown",
        quality_before: Optional[float] = None,
        quality_after: Optional[float] = None,
        latency_ms: Optional[float] = None,
        duplicate_groups_found: int = 0,
        elements_removed: int = 0,
        elements_merged: int = 0,
        timestamp: Optional[float] = None
    ) -> CompressionRecord:
        """
        Track a compression operation.

        Args:
            original_tokens: Tokens before compression
            compressed_tokens: Tokens after compression
            strategy: Compression strategy used
            quality_before: Quality score before compression (0.0-1.0)
            quality_after: Quality score after compression (0.0-1.0)
            latency_ms: Compression latency in milliseconds
            duplicate_groups_found: Number of duplicate groups
            elements_removed: Number of elements removed
            elements_merged: Number of elements merged
            timestamp: Unix timestamp (default: now)

        Returns:
            CompressionRecord
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        # Calculate metrics
        token_savings = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        quality_delta = None

        if quality_before is not None and quality_after is not None:
            quality_delta = quality_after - quality_before

        # Create record
        record = CompressionRecord(
            timestamp=timestamp,
            strategy=strategy,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_savings=token_savings,
            compression_ratio=compression_ratio,
            quality_before=quality_before,
            quality_after=quality_after,
            quality_delta=quality_delta,
            latency_ms=latency_ms,
            duplicate_groups_found=duplicate_groups_found,
            elements_removed=elements_removed,
            elements_merged=elements_merged
        )

        # Store
        self.records.append(record)
        self.by_strategy[strategy].append(record)

        return record

    def get_insights(
        self,
        strategy: Optional[str] = None,
        min_samples: int = 5
    ) -> CompressionInsights:
        """
        Get compression insights.

        Args:
            strategy: Filter by strategy (None = all)
            min_samples: Minimum samples for strategy analysis

        Returns:
            CompressionInsights
        """
        # Filter records
        records = self.records if strategy is None else self.by_strategy.get(strategy, [])

        if not records:
            return CompressionInsights(
                total_compressions=0,
                total_tokens_saved=0,
                avg_compression_ratio=1.0,
                avg_savings_pct=0.0,
                avg_quality_before=None,
                avg_quality_after=None,
                avg_quality_delta=None,
                quality_preserved_pct=0.0,
                avg_latency_ms=None,
                total_latency_ms=0.0,
                strategy_performance={},
                best_strategy=None,
                recommended_threshold=None
            )

        # Calculate overall statistics
        total_tokens_saved = sum(r.token_savings for r in records)
        avg_compression_ratio = statistics.mean(r.compression_ratio for r in records)
        avg_savings_pct = (1 - avg_compression_ratio) * 100

        # Quality statistics
        quality_before_values = [r.quality_before for r in records if r.quality_before is not None]
        quality_after_values = [r.quality_after for r in records if r.quality_after is not None]
        quality_delta_values = [r.quality_delta for r in records if r.quality_delta is not None]

        avg_quality_before = statistics.mean(quality_before_values) if quality_before_values else None
        avg_quality_after = statistics.mean(quality_after_values) if quality_after_values else None
        avg_quality_delta = statistics.mean(quality_delta_values) if quality_delta_values else None

        # Quality preservation rate
        quality_preserved_count = sum(1 for r in records if r.quality_preserved)
        quality_preserved_pct = (quality_preserved_count / len(records)) * 100

        # Latency statistics
        latency_values = [r.latency_ms for r in records if r.latency_ms is not None]
        avg_latency_ms = statistics.mean(latency_values) if latency_values else None
        total_latency_ms = sum(latency_values) if latency_values else 0.0

        # Strategy performance
        strategy_performance = {}

        for strat, strat_records in self.by_strategy.items():
            if len(strat_records) < min_samples:
                continue  # Not enough data

            strat_compression = statistics.mean(r.compression_ratio for r in strat_records)
            strat_quality_delta = statistics.mean(
                r.quality_delta for r in strat_records if r.quality_delta is not None
            ) if any(r.quality_delta is not None for r in strat_records) else None

            strat_preserved = sum(1 for r in strat_records if r.quality_preserved)
            strat_preserved_pct = (strat_preserved / len(strat_records)) * 100

            strategy_performance[strat] = {
                "compression_ratio": strat_compression,
                "savings_pct": (1 - strat_compression) * 100,
                "quality_delta": strat_quality_delta,
                "quality_preserved_pct": strat_preserved_pct,
                "sample_size": len(strat_records)
            }

        # Determine best strategy (highest savings with quality preserved)
        best_strategy = None
        best_score = -1.0

        for strat, metrics in strategy_performance.items():
            if metrics["quality_preserved_pct"] >= 95.0:  # Quality must be preserved
                # Score = savings × quality_preservation
                score = metrics["savings_pct"] * (metrics["quality_preserved_pct"] / 100)
                if score > best_score:
                    best_score = score
                    best_strategy = strat

        return CompressionInsights(
            total_compressions=len(records),
            total_tokens_saved=total_tokens_saved,
            avg_compression_ratio=avg_compression_ratio,
            avg_savings_pct=avg_savings_pct,
            avg_quality_before=avg_quality_before,
            avg_quality_after=avg_quality_after,
            avg_quality_delta=avg_quality_delta,
            quality_preserved_pct=quality_preserved_pct,
            avg_latency_ms=avg_latency_ms,
            total_latency_ms=total_latency_ms,
            strategy_performance=strategy_performance,
            best_strategy=best_strategy,
            recommended_threshold=None  # TODO: Analyze optimal similarity threshold
        )

    def get_recommendations(self) -> List[str]:
        """
        Get actionable recommendations.

        Returns:
            List of recommendation strings
        """
        insights = self.get_insights()
        recommendations = []

        # Recommendation 1: Overall effectiveness
        if insights.avg_savings_pct > 20:
            recommendations.append(
                f"✅ Compression is highly effective ({insights.avg_savings_pct:.1f}% savings). Continue using."
            )
        elif insights.avg_savings_pct > 10:
            recommendations.append(
                f"🟡 Compression provides moderate savings ({insights.avg_savings_pct:.1f}%). Consider increasing similarity threshold."
            )
        else:
            recommendations.append(
                f"⚠️ Compression provides minimal savings ({insights.avg_savings_pct:.1f}%). Consider disabling or adjusting thresholds."
            )

        # Recommendation 2: Quality impact
        if insights.avg_quality_delta is not None:
            if insights.avg_quality_delta < -0.05:  # >5% quality drop
                recommendations.append(
                    f"⚠️ Compression hurts quality ({insights.avg_quality_delta:+.3f}). Use more conservative settings."
                )
            elif insights.avg_quality_delta > 0:
                recommendations.append(
                    f"✅ Compression improves quality ({insights.avg_quality_delta:+.3f}). Likely removing noise."
                )

        # Recommendation 3: Strategy selection
        if insights.best_strategy:
            recommendations.append(
                f"✅ Best strategy: {insights.best_strategy} "
                f"({insights.strategy_performance[insights.best_strategy]['savings_pct']:.1f}% savings, "
                f"{insights.strategy_performance[insights.best_strategy]['quality_preserved_pct']:.1f}% quality preserved)"
            )

        # Recommendation 4: Latency overhead
        if insights.avg_latency_ms is not None:
            if insights.avg_latency_ms > 100:  # >100ms overhead
                recommendations.append(
                    f"⚠️ Compression adds significant latency ({insights.avg_latency_ms:.1f}ms). Consider async processing."
                )

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """Get raw statistics"""
        insights = self.get_insights()

        return {
            "total_compressions": insights.total_compressions,
            "total_tokens_saved": insights.total_tokens_saved,
            "avg_compression_ratio": insights.avg_compression_ratio,
            "avg_savings_pct": insights.avg_savings_pct,
            "quality_preserved_pct": insights.quality_preserved_pct,
            "strategy_performance": insights.strategy_performance
        }

    def reset_statistics(self):
        """Reset all tracking data"""
        self.records.clear()
        self.by_strategy.clear()
