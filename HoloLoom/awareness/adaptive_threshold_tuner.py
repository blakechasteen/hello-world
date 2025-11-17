#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Learning from Outcomes - Adaptive Threshold Tuner

Automatically tunes packing thresholds based on quality feedback.

Key Features:
1. Safe exploration (gradual changes, not big jumps)
2. Confidence-gated updates (only update with enough data)
3. Rollback on regression (automatically revert bad changes)
4. Per-complexity tuning (simple queries ≠ complex queries)

Usage:
    tuner = AdaptiveThresholdTuner(
        outcome_tracker=tracker,
        enable_auto_tuning=True
    )

    # Get current thresholds
    thresholds = tuner.get_thresholds(complexity="SIMPLE")

    # Use thresholds for packing
    packed = await packer.pack_context(
        importance_threshold=thresholds['importance_threshold'],
        compression_level=thresholds['compression_level']
    )

    # Track outcome
    await tracker.track(...)

    # Auto-tune (if enough data)
    await tuner.update()  # Checks if update needed
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import logging

from .outcome_tracker import OutcomeTracker, CompressionLevel, ThresholdRecommendation

logger = logging.getLogger(__name__)


class TuningStrategy(str, Enum):
    """Threshold tuning strategies"""
    CONSERVATIVE = "conservative"  # Slow, safe changes
    BALANCED = "balanced"          # Moderate changes
    AGGRESSIVE = "aggressive"      # Fast changes (risky)


@dataclass
class ThresholdSnapshot:
    """
    Snapshot of thresholds at a point in time.

    Used for rollback if quality degrades.
    """

    timestamp: datetime
    complexity: str

    # Thresholds
    importance_threshold: float
    compression_level: CompressionLevel

    # Performance at this configuration
    avg_quality: float
    avg_efficiency: float
    sample_size: int

    # Metadata
    reason: str  # Why these thresholds? (for debugging)


@dataclass
class TuningEvent:
    """
    Record of a threshold change.

    Tracks what changed, why, and the outcome.
    """

    timestamp: datetime
    complexity: str
    parameter: str               # "importance_threshold", "compression_level"
    old_value: Any
    new_value: Any
    reason: str                  # Why did we change it?
    confidence: float            # 0.0-1.0
    expected_improvement: float  # Expected quality gain
    actual_improvement: Optional[float] = None  # Actual quality gain (measured later)


class AdaptiveThresholdTuner:
    """
    Automatically tune packing thresholds based on outcomes.

    Safe, gradual tuning with rollback on regression.
    """

    def __init__(
        self,
        outcome_tracker: OutcomeTracker,
        enable_auto_tuning: bool = True,
        tuning_strategy: TuningStrategy = TuningStrategy.BALANCED,
        min_improvement_threshold: float = 0.02,  # Minimum quality improvement to accept change
        regression_threshold: float = 0.05,       # Quality drop to trigger rollback
        update_interval_seconds: int = 300        # Check for updates every 5 minutes
    ):
        """
        Initialize adaptive threshold tuner.

        Args:
            outcome_tracker: OutcomeTracker instance
            enable_auto_tuning: Enable automatic tuning
            tuning_strategy: Tuning strategy (conservative/balanced/aggressive)
            min_improvement_threshold: Minimum quality improvement to accept change
            regression_threshold: Quality drop threshold for rollback
            update_interval_seconds: Seconds between update checks
        """
        self.tracker = outcome_tracker
        self.enable_auto_tuning = enable_auto_tuning
        self.strategy = tuning_strategy
        self.min_improvement = min_improvement_threshold
        self.regression_threshold = regression_threshold
        self.update_interval = timedelta(seconds=update_interval_seconds)

        # Current thresholds (per complexity level)
        self.current_thresholds: Dict[str, Dict[str, Any]] = {
            "SIMPLE": {
                "importance_threshold": 0.3,
                "compression_level": CompressionLevel.SUMMARY
            },
            "MODERATE": {
                "importance_threshold": 0.4,
                "compression_level": CompressionLevel.DETAILED
            },
            "COMPLEX": {
                "importance_threshold": 0.5,
                "compression_level": CompressionLevel.DETAILED
            },
            "RESEARCH": {
                "importance_threshold": 0.6,
                "compression_level": CompressionLevel.FULL
            }
        }

        # History
        self.snapshots: Dict[str, List[ThresholdSnapshot]] = {
            "SIMPLE": [],
            "MODERATE": [],
            "COMPLEX": [],
            "RESEARCH": []
        }

        self.tuning_events: List[TuningEvent] = []

        # State
        self.last_update_time: Dict[str, datetime] = {}

    def get_thresholds(self, complexity: str) -> Dict[str, Any]:
        """
        Get current thresholds for complexity level.

        Args:
            complexity: Complexity level (SIMPLE/MODERATE/COMPLEX/RESEARCH)

        Returns:
            Dict with importance_threshold and compression_level
        """
        return self.current_thresholds.get(complexity, self.current_thresholds["MODERATE"])

    async def update(self, complexity: Optional[str] = None) -> List[TuningEvent]:
        """
        Check if update is needed and apply if so.

        Args:
            complexity: Specific complexity level to update (or None for all)

        Returns:
            List of tuning events (changes made)
        """
        if not self.enable_auto_tuning:
            return []

        events = []

        complexities = [complexity] if complexity else ["SIMPLE", "MODERATE", "COMPLEX", "RESEARCH"]

        for comp in complexities:
            # Check if enough time has passed
            if comp in self.last_update_time:
                elapsed = datetime.now() - self.last_update_time[comp]
                if elapsed < self.update_interval:
                    continue  # Too soon

            # Get recommendations
            current = self.current_thresholds[comp]
            recommendations = self.tracker.get_recommendations(
                complexity=comp,
                current_importance_threshold=current["importance_threshold"],
                current_compression_level=current["compression_level"]
            )

            # Apply recommendations (if confidence high enough)
            for rec in recommendations:
                event = await self._apply_recommendation(comp, rec)
                if event:
                    events.append(event)

            # Update last update time
            self.last_update_time[comp] = datetime.now()

        return events

    async def _apply_recommendation(
        self,
        complexity: str,
        recommendation: ThresholdRecommendation
    ) -> Optional[TuningEvent]:
        """
        Apply a threshold recommendation (if safe).

        Args:
            complexity: Complexity level
            recommendation: Threshold recommendation

        Returns:
            TuningEvent if applied, None otherwise
        """
        # Confidence threshold (based on strategy)
        confidence_thresholds = {
            TuningStrategy.CONSERVATIVE: 0.8,
            TuningStrategy.BALANCED: 0.6,
            TuningStrategy.AGGRESSIVE: 0.4
        }

        min_confidence = confidence_thresholds[self.strategy]

        if recommendation.confidence < min_confidence:
            logger.info(
                f"Skipping {recommendation.parameter} update for {complexity}: "
                f"confidence {recommendation.confidence:.2f} < {min_confidence:.2f}"
            )
            return None

        # Expected improvement threshold
        if recommendation.expected_improvement < self.min_improvement:
            logger.info(
                f"Skipping {recommendation.parameter} update for {complexity}: "
                f"expected improvement {recommendation.expected_improvement:.2f} < {self.min_improvement:.2f}"
            )
            return None

        # Create snapshot before change
        await self._create_snapshot(complexity)

        # Apply change
        current = self.current_thresholds[complexity]
        old_value = current[recommendation.parameter]

        current[recommendation.parameter] = recommendation.recommended_value

        # Record event
        event = TuningEvent(
            timestamp=datetime.now(),
            complexity=complexity,
            parameter=recommendation.parameter,
            old_value=old_value,
            new_value=recommendation.recommended_value,
            reason=recommendation.reasoning,
            confidence=recommendation.confidence,
            expected_improvement=recommendation.expected_improvement
        )

        self.tuning_events.append(event)

        logger.info(
            f"Updated {recommendation.parameter} for {complexity}: "
            f"{old_value} → {recommendation.recommended_value} "
            f"(expected improvement: +{recommendation.expected_improvement:.2f})"
        )

        return event

    async def _create_snapshot(self, complexity: str):
        """Create snapshot of current thresholds"""
        outcomes = self.tracker.by_complexity.get(complexity, [])

        if not outcomes:
            return

        snapshot = ThresholdSnapshot(
            timestamp=datetime.now(),
            complexity=complexity,
            importance_threshold=self.current_thresholds[complexity]["importance_threshold"],
            compression_level=self.current_thresholds[complexity]["compression_level"],
            avg_quality=statistics.mean(o.quality_overall for o in outcomes[-10:]),  # Last 10
            avg_efficiency=statistics.mean(o.budget_efficiency for o in outcomes[-10:]),
            sample_size=len(outcomes),
            reason="Pre-update snapshot"
        )

        self.snapshots[complexity].append(snapshot)

        # Keep only last 10 snapshots
        if len(self.snapshots[complexity]) > 10:
            self.snapshots[complexity] = self.snapshots[complexity][-10:]

    async def check_for_regression(
        self,
        complexity: str
    ) -> Optional[ThresholdSnapshot]:
        """
        Check if recent changes caused quality regression.

        Args:
            complexity: Complexity level to check

        Returns:
            Snapshot to rollback to (if regression detected), None otherwise
        """
        if len(self.snapshots[complexity]) < 2:
            return None  # Not enough history

        # Get recent outcomes
        outcomes = self.tracker.by_complexity.get(complexity, [])
        if len(outcomes) < 10:
            return None  # Not enough recent data

        # Compare current quality to last snapshot
        last_snapshot = self.snapshots[complexity][-1]
        recent_quality = statistics.mean(o.quality_overall for o in outcomes[-10:])

        quality_drop = last_snapshot.avg_quality - recent_quality

        if quality_drop > self.regression_threshold:
            logger.warning(
                f"Regression detected for {complexity}: "
                f"quality dropped {quality_drop:.2f} (threshold: {self.regression_threshold:.2f})"
            )
            return last_snapshot

        return None

    async def rollback(self, complexity: str, snapshot: ThresholdSnapshot):
        """
        Rollback to a previous snapshot.

        Args:
            complexity: Complexity level
            snapshot: Snapshot to rollback to
        """
        logger.info(
            f"Rolling back {complexity} to snapshot from {snapshot.timestamp} "
            f"(quality: {snapshot.avg_quality:.2f})"
        )

        # Restore thresholds
        self.current_thresholds[complexity]["importance_threshold"] = snapshot.importance_threshold
        self.current_thresholds[complexity]["compression_level"] = snapshot.compression_level

        # Record rollback event
        event = TuningEvent(
            timestamp=datetime.now(),
            complexity=complexity,
            parameter="rollback",
            old_value="current",
            new_value=snapshot.timestamp,
            reason=f"Quality regression detected (dropped {self.regression_threshold:.2f})",
            confidence=1.0,
            expected_improvement=0.0
        )

        self.tuning_events.append(event)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tuning statistics.

        Returns:
            Statistics dict
        """
        return {
            "auto_tuning_enabled": self.enable_auto_tuning,
            "strategy": self.strategy.value,
            "total_tuning_events": len(self.tuning_events),
            "current_thresholds": self.current_thresholds,
            "snapshots_per_complexity": {
                k: len(v) for k, v in self.snapshots.items()
            },
            "recent_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "complexity": e.complexity,
                    "parameter": e.parameter,
                    "old_value": str(e.old_value),
                    "new_value": str(e.new_value),
                    "confidence": e.confidence
                }
                for e in self.tuning_events[-5:]  # Last 5 events
            ]
        }

    def get_tuning_history(
        self,
        complexity: Optional[str] = None,
        parameter: Optional[str] = None
    ) -> List[TuningEvent]:
        """
        Get tuning history.

        Args:
            complexity: Filter by complexity level
            parameter: Filter by parameter

        Returns:
            List of tuning events
        """
        events = self.tuning_events

        if complexity:
            events = [e for e in events if e.complexity == complexity]

        if parameter:
            events = [e for e in events if e.parameter == parameter]

        return events

    def reset_thresholds(self, complexity: Optional[str] = None):
        """
        Reset thresholds to defaults.

        Args:
            complexity: Specific complexity to reset (or None for all)
        """
        defaults = {
            "SIMPLE": {
                "importance_threshold": 0.3,
                "compression_level": CompressionLevel.SUMMARY
            },
            "MODERATE": {
                "importance_threshold": 0.4,
                "compression_level": CompressionLevel.DETAILED
            },
            "COMPLEX": {
                "importance_threshold": 0.5,
                "compression_level": CompressionLevel.DETAILED
            },
            "RESEARCH": {
                "importance_threshold": 0.6,
                "compression_level": CompressionLevel.FULL
            }
        }

        if complexity:
            self.current_thresholds[complexity] = defaults[complexity].copy()
        else:
            self.current_thresholds = {k: v.copy() for k, v in defaults.items()}
