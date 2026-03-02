"""
Conscience Calibrator with Thompson Sampling
=============================================

Bayesian calibration for conscience decision thresholds.

Uses Thompson Sampling to learn optimal risk thresholds per step type,
adapting based on actual outcomes (success/failure after conscience decisions).

Key Features:
- Per-step-type calibration (QUERY, VERIFICATION, RESEARCH, PLAN_STEP)
- Beta distribution priors for Bayesian updating
- Calibration confidence tracking
- Drift detection for threshold shifts
- Complete calibration history for auditing

Design Philosophy:
- "A calibrated conscience is a trustworthy conscience"
- Start with uniform priors (no prior assumptions)
- Update based on actual outcomes (success = correct decision)
- Track calibration uncertainty (α + β gives confidence)

Author: HoloLoom Team
Date: 2025-12-05
Phase: 2D - Thompson Sampling Integration
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

from hololoom.protocols.conscience import (
    ConscienceDecision,
    StepType,
    RiskLevel,
)

logger = logging.getLogger("hololoom.agentic.conscience_calibrator")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationPrior:
    """
    Beta distribution prior for a step type.

    Alpha = number of correct decisions (successes)
    Beta = number of incorrect decisions (failures)

    Mean = α / (α + β)
    Variance = αβ / ((α+β)²(α+β+1))
    """
    alpha: float = 1.0  # Successes (start with uniform prior)
    beta: float = 1.0   # Failures
    updates: int = 0    # Total updates
    last_update: Optional[datetime] = None

    @property
    def mean(self) -> float:
        """Expected probability of correct decision."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Uncertainty in the probability estimate."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    @property
    def confidence(self) -> float:
        """
        Confidence in the prior (0.0-1.0).

        Based on total observations - more data = more confidence.
        """
        # Confidence scales with observations (saturates around 100)
        total = self.alpha + self.beta - 2  # Subtract initial 1+1
        return min(1.0, total / 100.0)

    def sample(self) -> float:
        """Sample from the Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """
        Update prior based on outcome.

        Args:
            success: Whether the conscience decision was correct
            weight: Weight for this update (0.0-1.0)
        """
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.updates += 1
        self.last_update = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "updates": self.updates,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "mean": self.mean,
            "variance": self.variance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationPrior':
        """Create from dictionary."""
        prior = cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            updates=data.get("updates", 0),
        )
        if data.get("last_update"):
            prior.last_update = datetime.fromisoformat(data["last_update"])
        return prior


@dataclass
class CalibrationEvent:
    """Record of a calibration update."""
    timestamp: datetime
    step_type: StepType
    decision_allowed: bool
    actual_success: bool
    outcome_correct: bool
    confidence_before: float
    confidence_after: float
    prior_mean_before: float
    prior_mean_after: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "step_type": self.step_type.name,
            "decision_allowed": self.decision_allowed,
            "actual_success": self.actual_success,
            "outcome_correct": self.outcome_correct,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "prior_mean_before": self.prior_mean_before,
            "prior_mean_after": self.prior_mean_after,
            "metadata": self.metadata,
        }


@dataclass
class DriftAlert:
    """Alert when calibration drifts significantly."""
    timestamp: datetime
    step_type: StepType
    old_mean: float
    new_mean: float
    drift_magnitude: float
    direction: str  # "stricter" or "more_permissive"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "step_type": self.step_type.name,
            "old_mean": self.old_mean,
            "new_mean": self.new_mean,
            "drift_magnitude": self.drift_magnitude,
            "direction": self.direction,
        }


# =============================================================================
# Main Calibrator
# =============================================================================

class ConscienceCalibrator:
    """
    Thompson Sampling calibrator for conscience decisions.

    Maintains Beta distribution priors for each step type, updating
    based on actual outcomes to calibrate decision thresholds.

    Example:
        calibrator = ConscienceCalibrator()

        # Make decision
        decision = await adapter.gate_step(action, step_type, step_index)

        # Execute action
        result = await execute_action(action)

        # Calibrate based on outcome
        calibrator.update(decision, success=result.success)

        # Get calibrated threshold
        threshold = calibrator.get_threshold(StepType.VERIFICATION)
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        min_updates_for_calibration: int = 10,
        decay_rate: float = 0.0,  # 0.0 = no decay, positive = older updates matter less
    ):
        """
        Initialize the calibrator.

        Args:
            drift_threshold: Magnitude of change to trigger drift alert
            min_updates_for_calibration: Minimum updates before using calibrated threshold
            decay_rate: Decay rate for older observations (0.0 = no decay)
        """
        self._drift_threshold = drift_threshold
        self._min_updates = min_updates_for_calibration
        self._decay_rate = decay_rate

        # Initialize priors for each step type
        self._priors: Dict[StepType, CalibrationPrior] = {
            step_type: CalibrationPrior()
            for step_type in StepType
        }

        # Track calibration history
        self._history: List[CalibrationEvent] = []
        self._max_history = 1000  # Keep last 1000 events

        # Track drift alerts
        self._drift_alerts: List[DriftAlert] = []

        # Statistics
        self._total_updates = 0
        self._correct_predictions = 0
        self._false_positives = 0  # Blocked but would have succeeded
        self._false_negatives = 0  # Allowed but failed

        logger.info(
            f"ConscienceCalibrator initialized (drift_threshold={drift_threshold}, "
            f"min_updates={min_updates_for_calibration})"
        )

    # =========================================================================
    # Core API
    # =========================================================================

    def update(
        self,
        decision: ConscienceDecision,
        success: bool,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CalibrationEvent:
        """
        Update calibration based on a decision outcome.

        Args:
            decision: The conscience decision that was made
            success: Whether the actual action succeeded
            weight: Weight for this update (0.0-1.0)
            metadata: Optional metadata to record

        Returns:
            CalibrationEvent recording the update
        """
        step_type = decision.step_type
        prior = self._priors[step_type]

        # Record state before update
        mean_before = prior.mean
        confidence_before = prior.confidence

        # Determine if the decision was correct
        # Correct = (allowed and succeeded) OR (blocked and would have failed)
        # Since we can't know counterfactuals, we estimate:
        # - If allowed and succeeded: correct decision
        # - If allowed and failed: incorrect (should have blocked)
        # - If blocked: we assume it would have failed (correct block)
        #   (This is a conservative assumption - could be wrong)

        if decision.allowed:
            outcome_correct = success
            if not success:
                self._false_negatives += 1
        else:
            # Blocked - we assume blocking was correct
            # (Counterfactual: if we had allowed, it would have failed)
            outcome_correct = True
            # Note: This could be a false positive if action would have succeeded
            # We could track this separately if there's a way to verify

        # Update prior
        prior.update(outcome_correct, weight)

        # Track statistics
        self._total_updates += 1
        if outcome_correct:
            self._correct_predictions += 1

        # Check for drift
        mean_after = prior.mean
        drift = abs(mean_after - mean_before)

        if drift > self._drift_threshold and prior.updates > self._min_updates:
            direction = "stricter" if mean_after < mean_before else "more_permissive"
            alert = DriftAlert(
                timestamp=datetime.now(),
                step_type=step_type,
                old_mean=mean_before,
                new_mean=mean_after,
                drift_magnitude=drift,
                direction=direction,
            )
            self._drift_alerts.append(alert)
            logger.warning(
                f"[CALIBRATION DRIFT] {step_type.name}: "
                f"{mean_before:.3f} → {mean_after:.3f} ({direction})"
            )

        # Record event
        event = CalibrationEvent(
            timestamp=datetime.now(),
            step_type=step_type,
            decision_allowed=decision.allowed,
            actual_success=success,
            outcome_correct=outcome_correct,
            confidence_before=confidence_before,
            confidence_after=prior.confidence,
            prior_mean_before=mean_before,
            prior_mean_after=mean_after,
            metadata=metadata or {},
        )

        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.debug(
            f"[CALIBRATION] {step_type.name}: "
            f"decision={decision.allowed}, success={success}, "
            f"correct={outcome_correct}, mean={mean_after:.3f}"
        )

        return event

    def get_threshold(self, step_type: StepType) -> float:
        """
        Get the calibrated threshold for a step type.

        Returns the expected probability of correct decision,
        which can be used to adjust risk thresholds.

        Args:
            step_type: The step type to get threshold for

        Returns:
            Calibrated threshold (0.0-1.0)
        """
        prior = self._priors[step_type]

        if prior.updates < self._min_updates:
            # Not enough data - return default (0.5 = neutral)
            return 0.5

        return prior.mean

    def sample_threshold(self, step_type: StepType) -> float:
        """
        Sample a threshold from the posterior distribution.

        This implements Thompson Sampling - sampling from the
        posterior allows natural exploration of thresholds.

        Args:
            step_type: The step type to sample for

        Returns:
            Sampled threshold
        """
        prior = self._priors[step_type]
        return prior.sample()

    def get_confidence(self, step_type: StepType) -> float:
        """
        Get calibration confidence for a step type.

        Higher confidence = more data = more reliable threshold.

        Args:
            step_type: The step type to check

        Returns:
            Confidence (0.0-1.0)
        """
        return self._priors[step_type].confidence

    def should_use_calibration(self, step_type: StepType) -> bool:
        """
        Check if calibration should be used for a step type.

        Returns True if there's enough data to trust calibration.

        Args:
            step_type: The step type to check

        Returns:
            True if calibration is ready
        """
        return self._priors[step_type].updates >= self._min_updates

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive calibration statistics."""
        accuracy = (
            self._correct_predictions / max(self._total_updates, 1)
        )

        per_step_type = {}
        for step_type in StepType:
            prior = self._priors[step_type]
            per_step_type[step_type.name] = {
                "mean": prior.mean,
                "variance": prior.variance,
                "confidence": prior.confidence,
                "updates": prior.updates,
                "ready": prior.updates >= self._min_updates,
            }

        return {
            "total_updates": self._total_updates,
            "correct_predictions": self._correct_predictions,
            "accuracy": accuracy,
            "false_negatives": self._false_negatives,
            "drift_alerts": len(self._drift_alerts),
            "per_step_type": per_step_type,
            "drift_threshold": self._drift_threshold,
            "min_updates_for_calibration": self._min_updates,
        }

    def get_priors(self) -> Dict[str, Dict[str, Any]]:
        """Get all priors as dictionaries."""
        return {
            step_type.name: prior.to_dict()
            for step_type, prior in self._priors.items()
        }

    def get_drift_alerts(self, since: Optional[datetime] = None) -> List[DriftAlert]:
        """Get drift alerts, optionally filtered by time."""
        if since is None:
            return self._drift_alerts.copy()
        return [a for a in self._drift_alerts if a.timestamp >= since]

    def get_recent_history(self, limit: int = 100) -> List[CalibrationEvent]:
        """Get recent calibration history."""
        return self._history[-limit:]

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path) -> None:
        """Save calibration state to file."""
        state = {
            "priors": {
                step_type.name: prior.to_dict()
                for step_type, prior in self._priors.items()
            },
            "statistics": {
                "total_updates": self._total_updates,
                "correct_predictions": self._correct_predictions,
                "false_negatives": self._false_negatives,
                "false_positives": self._false_positives,
            },
            "config": {
                "drift_threshold": self._drift_threshold,
                "min_updates": self._min_updates,
                "decay_rate": self._decay_rate,
            },
            "saved_at": datetime.now().isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Calibration state saved to {path}")

    def load(self, path: Path) -> None:
        """Load calibration state from file."""
        path = Path(path)

        if not path.exists():
            logger.warning(f"Calibration file not found: {path}")
            return

        with open(path, 'r') as f:
            state = json.load(f)

        # Restore priors
        for step_type_name, prior_data in state.get("priors", {}).items():
            try:
                step_type = StepType[step_type_name]
                self._priors[step_type] = CalibrationPrior.from_dict(prior_data)
            except KeyError:
                logger.warning(f"Unknown step type: {step_type_name}")

        # Restore statistics
        stats = state.get("statistics", {})
        self._total_updates = stats.get("total_updates", 0)
        self._correct_predictions = stats.get("correct_predictions", 0)
        self._false_negatives = stats.get("false_negatives", 0)
        self._false_positives = stats.get("false_positives", 0)

        logger.info(f"Calibration state loaded from {path} ({self._total_updates} updates)")

    # =========================================================================
    # Integration with Conscience Adapter
    # =========================================================================

    def calibrate_decision(
        self,
        decision: ConscienceDecision,
        use_sampling: bool = False,
    ) -> ConscienceDecision:
        """
        Adjust a conscience decision based on calibration.

        This can adjust the risk level based on calibrated thresholds,
        potentially upgrading/downgrading the decision severity.

        Args:
            decision: The original conscience decision
            use_sampling: If True, sample threshold (exploration)
                         If False, use mean (exploitation)

        Returns:
            Potentially adjusted decision
        """
        step_type = decision.step_type

        if not self.should_use_calibration(step_type):
            # Not enough data - return original
            return decision

        # Get threshold
        if use_sampling:
            threshold = self.sample_threshold(step_type)
        else:
            threshold = self.get_threshold(step_type)

        # Adjust risk level based on threshold
        # Higher threshold = more confident in correct decisions = can be less strict
        # Lower threshold = less confident = should be stricter

        confidence = self.get_confidence(step_type)

        # Only adjust if we're confident in calibration
        if confidence < 0.5:
            return decision

        # Create modified decision with calibration metadata
        # Note: We don't change allowed/blocked, just add context
        decision.metadata["calibration"] = {
            "threshold": threshold,
            "confidence": confidence,
            "sampling": use_sampling,
        }

        return decision


# =============================================================================
# Factory Functions
# =============================================================================

def create_conscience_calibrator(
    drift_threshold: float = 0.1,
    min_updates: int = 10,
    decay_rate: float = 0.0,
    load_path: Optional[Path] = None,
) -> ConscienceCalibrator:
    """
    Create a ConscienceCalibrator with optional state loading.

    Args:
        drift_threshold: Threshold for drift alerts
        min_updates: Minimum updates before using calibration
        decay_rate: Decay rate for older observations
        load_path: Optional path to load previous state

    Returns:
        Configured ConscienceCalibrator
    """
    calibrator = ConscienceCalibrator(
        drift_threshold=drift_threshold,
        min_updates_for_calibration=min_updates,
        decay_rate=decay_rate,
    )

    if load_path:
        calibrator.load(load_path)

    return calibrator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class
    "ConscienceCalibrator",

    # Data classes
    "CalibrationPrior",
    "CalibrationEvent",
    "DriftAlert",

    # Factory
    "create_conscience_calibrator",
]
