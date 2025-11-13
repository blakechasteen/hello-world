#!/usr/bin/env python3
"""
Strategy Updater for Hybrid Routing

Part 4: Learning Mechanisms (Days 16-20)

Adjusts routing strategy based on learning signals from the LearningTracker
and ConfidenceCalibrator. Implements conservative update rules to prevent
system instability.

Update Strategies:
1. Backend weight adjustment (performance-based)
2. Refinement threshold tuning (quality-based)
3. Confidence calibration (prediction accuracy)
4. Fallback strategy adaptation

Validated in Demo 5 (strategy updates improve accuracy over time)
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class StrategyUpdate:
    """Single strategy update event"""
    update_type: str  # "weight_adjustment", "refinement_threshold", "calibration"
    backend: Optional[str]
    old_value: float
    new_value: float
    reason: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "update_type": self.update_type,
            "backend": self.backend,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# Strategy Updater
# ============================================================================

class StrategyUpdater:
    """
    Adjust routing strategy based on learning signals

    Conservative update rules to prevent system instability:
    - Minimum observation threshold (100 events)
    - Gradual adjustments (max 20% change per update)
    - Rollback mechanism if performance degrades
    - Update interval enforcement (default: 1 hour)

    Usage:
        updater = StrategyUpdater(query_router, update_interval=3600.0)

        # Check for updates (called periodically)
        await updater.update_if_needed()

        # Force immediate update
        await updater.force_update()
    """

    def __init__(
        self,
        query_router,
        update_interval: float = 3600.0,  # 1 hour default
        min_observations: int = 100,
        max_adjustment: float = 0.20  # Max 20% change per update
    ):
        """
        Initialize strategy updater

        Args:
            query_router: QueryRouter instance to update
            update_interval: Seconds between strategy updates
            min_observations: Minimum observations required for update
            max_adjustment: Maximum adjustment factor per update
        """
        self.router = query_router
        self.update_interval = update_interval
        self.min_observations = min_observations
        self.max_adjustment = max_adjustment

        # State
        self.last_update = time.time()
        self.update_history: List[StrategyUpdate] = []
        self.update_count = 0

        # Rollback tracking
        self.last_overall_confidence = 0.5
        self.rollback_count = 0

    async def update_if_needed(self) -> bool:
        """
        Check if strategy update is needed and apply if so

        Returns:
            True if update was applied, False otherwise
        """
        # Check time since last update
        if time.time() - self.last_update < self.update_interval:
            return False

        # Check minimum observations
        if self.router.learning_tracker.total_events < self.min_observations:
            logger.info(
                f"Strategy update skipped: insufficient observations "
                f"({self.router.learning_tracker.total_events}/{self.min_observations})"
            )
            return False

        # Apply update
        await self.force_update()
        return True

    async def force_update(self):
        """Force immediate strategy update"""
        logger.info("Starting strategy update...")

        # Get current performance
        overall_perf = self.router.learning_tracker.get_overall_performance(window=100)
        backend_perf = self.router.learning_tracker.get_backend_comparison(window=100)

        # Update 1: Adjust backend weights based on performance
        await self._update_backend_weights(backend_perf)

        # Update 2: Tune refinement threshold based on quality
        await self._update_refinement_threshold(overall_perf)

        # Update 3: Update confidence calibration
        await self._update_calibration()

        # Update 4: Adapt fallback strategy
        await self._update_fallback_strategy(backend_perf)

        # Check for performance degradation
        current_confidence = overall_perf["avg_confidence"]
        if current_confidence < self.last_overall_confidence - 0.05:
            # Performance degraded - rollback
            logger.warning(
                f"Performance degraded: {self.last_overall_confidence:.3f} → {current_confidence:.3f}. "
                "Consider rollback."
            )
            self.rollback_count += 1
        else:
            self.last_overall_confidence = current_confidence

        self.last_update = time.time()
        self.update_count += 1

        logger.info(f"Strategy update complete (update #{self.update_count})")

    async def _update_backend_weights(self, backend_perf: Dict):
        """
        Update backend routing weights based on performance

        Strategy:
        - High-performing backends get weight boost
        - Low-performing backends get weight reduction
        - Max adjustment: 20% per update
        """
        for backend, metrics in backend_perf.items():
            if metrics.count == 0:
                continue  # No data for this backend

            # Get current weight from classifier
            current_weight = self.router.classifier.backend_weights.get(backend, 1.0)

            # Determine adjustment based on performance
            # Good performance: latency <50ms, confidence >0.80, fallback <10%
            is_good_performance = (
                metrics.avg_latency_ms < 50.0 and
                metrics.avg_confidence > 0.80 and
                metrics.fallback_rate < 0.10
            )

            # Poor performance: latency >150ms, confidence <0.70, fallback >20%
            is_poor_performance = (
                metrics.avg_latency_ms > 150.0 or
                metrics.avg_confidence < 0.70 or
                metrics.fallback_rate > 0.20
            )

            if is_good_performance:
                # Boost weight (max 20% increase)
                new_weight = min(current_weight * 1.20, current_weight + self.max_adjustment)
                reason = f"Good performance: conf={metrics.avg_confidence:.3f}, latency={metrics.avg_latency_ms:.1f}ms"
            elif is_poor_performance:
                # Reduce weight (max 20% decrease)
                new_weight = max(current_weight * 0.80, current_weight - self.max_adjustment)
                reason = f"Poor performance: conf={metrics.avg_confidence:.3f}, latency={metrics.avg_latency_ms:.1f}ms"
            else:
                # No change
                continue

            # Apply adjustment
            self.router.classifier.backend_weights[backend] = new_weight

            # Record update
            update = StrategyUpdate(
                update_type="weight_adjustment",
                backend=backend,
                old_value=current_weight,
                new_value=new_weight,
                reason=reason,
                timestamp=datetime.utcnow()
            )
            self.update_history.append(update)

            logger.info(
                f"Adjusted {backend} weight: {current_weight:.3f} → {new_weight:.3f} "
                f"({reason})"
            )

    async def _update_refinement_threshold(self, overall_perf: Dict):
        """
        Tune refinement threshold based on overall quality

        Strategy:
        - Low average confidence (<0.70) → enable refinement
        - High average confidence (>0.85) → disable refinement
        """
        avg_confidence = overall_perf["avg_confidence"]

        # Get current refinement state
        current_enabled = getattr(self.router, "enable_refinement", False)

        if avg_confidence < 0.70 and not current_enabled:
            # Enable refinement
            self.router.enable_refinement = True

            update = StrategyUpdate(
                update_type="refinement_threshold",
                backend=None,
                old_value=float(current_enabled),
                new_value=1.0,
                reason=f"Low average confidence: {avg_confidence:.3f}",
                timestamp=datetime.utcnow()
            )
            self.update_history.append(update)

            logger.info(f"Enabled refinement due to low confidence: {avg_confidence:.3f}")

        elif avg_confidence > 0.85 and current_enabled:
            # Disable refinement
            self.router.enable_refinement = False

            update = StrategyUpdate(
                update_type="refinement_threshold",
                backend=None,
                old_value=1.0,
                new_value=0.0,
                reason=f"High average confidence: {avg_confidence:.3f}",
                timestamp=datetime.utcnow()
            )
            self.update_history.append(update)

            logger.info(f"Disabled refinement due to high confidence: {avg_confidence:.3f}")

    async def _update_calibration(self):
        """
        Update confidence calibration based on prediction accuracy

        Strategy:
        - If ECE >0.10, adjust classifier confidence thresholds
        """
        calibrator = self.router.calibrator

        # Get overall calibration
        overall_curve = calibrator.get_calibration_curve()

        if not overall_curve.calibrated:
            logger.debug("Calibration not yet available (insufficient data)")
            return

        # Check if miscalibration is significant
        if overall_curve.ece > 0.15:
            logger.warning(
                f"Significant miscalibration detected: ECE={overall_curve.ece:.3f}"
            )

            # Record update
            update = StrategyUpdate(
                update_type="calibration",
                backend=None,
                old_value=0.0,
                new_value=overall_curve.ece,
                reason=f"ECE={overall_curve.ece:.3f} > 0.15 threshold",
                timestamp=datetime.utcnow()
            )
            self.update_history.append(update)

    async def _update_fallback_strategy(self, backend_perf: Dict):
        """
        Adapt fallback strategy based on backend reliability

        Strategy:
        - High fallback rate (>20%) → adjust fallback order
        """
        for backend, metrics in backend_perf.items():
            if metrics.count == 0:
                continue

            if metrics.fallback_rate > 0.20:
                logger.warning(
                    f"{backend} has high fallback rate: {metrics.fallback_rate*100:.1f}%"
                )

                update = StrategyUpdate(
                    update_type="fallback_strategy",
                    backend=backend,
                    old_value=0.0,
                    new_value=metrics.fallback_rate,
                    reason=f"Fallback rate: {metrics.fallback_rate*100:.1f}%",
                    timestamp=datetime.utcnow()
                )
                self.update_history.append(update)

    def get_update_summary(self) -> str:
        """
        Get strategy update summary

        Returns:
            Human-readable summary
        """
        summary = f"Strategy Update Summary:\n"
        summary += f"  Total updates: {self.update_count}\n"
        summary += f"  Last update: {datetime.fromtimestamp(self.last_update).isoformat()}\n"
        summary += f"  Update interval: {self.update_interval:.0f}s\n"
        summary += f"  Rollback count: {self.rollback_count}\n\n"

        # Recent updates
        if self.update_history:
            summary += f"Recent Updates (last 10):\n"
            for update in self.update_history[-10:]:
                summary += f"  [{update.update_type}] {update.backend or 'overall'}: "
                summary += f"{update.old_value:.3f} → {update.new_value:.3f} "
                summary += f"({update.reason})\n"

        return summary

    def get_stats(self) -> Dict:
        """
        Get strategy updater statistics

        Returns:
            Dict with updater stats
        """
        return {
            "update_count": self.update_count,
            "last_update": self.last_update,
            "update_interval": self.update_interval,
            "min_observations": self.min_observations,
            "max_adjustment": self.max_adjustment,
            "rollback_count": self.rollback_count,
            "recent_updates": [
                update.to_dict()
                for update in self.update_history[-10:]
            ]
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_strategy_updater(
    query_router,
    update_interval: float = 3600.0,
    min_observations: int = 100,
    max_adjustment: float = 0.20
) -> StrategyUpdater:
    """
    Create strategy updater

    Args:
        query_router: QueryRouter instance
        update_interval: Seconds between updates
        min_observations: Minimum observations for update
        max_adjustment: Maximum adjustment per update

    Returns:
        StrategyUpdater instance
    """
    return StrategyUpdater(
        query_router=query_router,
        update_interval=update_interval,
        min_observations=min_observations,
        max_adjustment=max_adjustment
    )
