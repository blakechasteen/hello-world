#!/usr/bin/env python3
"""
Confidence Calibration for Hybrid Routing

Part 4: Learning Mechanisms (Days 16-20)

Tracks predicted vs. actual confidence to compute calibration metrics
and provide adjustment factors for improving prediction accuracy.

Key Metrics:
- Expected Calibration Error (ECE): Avg difference between predicted and actual
- Calibration Curve: Predicted confidence vs. actual outcomes (binned)
- Adjustment Factors: Multipliers to correct systematic over/under-confidence

Validated in Demo 3 (ECE < 0.10)
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class CalibrationObservation:
    """Single calibration observation"""
    predicted_confidence: float
    actual_confidence: float
    backend: str
    error: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "predicted": self.predicted_confidence,
            "actual": self.actual_confidence,
            "backend": self.backend,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CalibrationCurve:
    """Calibration curve data"""
    calibrated: bool
    ece: float
    bin_centers: List[float]
    binned_actual: List[Optional[float]]
    binned_predicted: List[float]
    sample_size: int
    backend: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "calibrated": self.calibrated,
            "ece": self.ece,
            "bin_centers": self.bin_centers,
            "binned_actual": self.binned_actual,
            "binned_predicted": self.binned_predicted,
            "sample_size": self.sample_size,
            "backend": self.backend
        }


# ============================================================================
# Confidence Calibrator
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrate confidence predictions vs. actual outcomes

    Tracks predicted vs. actual confidence for each backend and computes
    calibration metrics (ECE) to assess prediction quality.

    Expected Calibration Error (ECE):
        ECE = (1/B) * sum(|predicted_i - actual_i|) for B bins
        Well-calibrated: ECE < 0.10

    Usage:
        calibrator = ConfidenceCalibrator()

        # Record observations
        calibrator.add_observation(
            predicted_confidence=0.85,
            actual_confidence=0.92,
            backend="sql"
        )

        # Get calibration curve
        curve = calibrator.get_calibration_curve("sql")
        print(f"ECE: {curve.ece:.3f}, Calibrated: {curve.calibrated}")

        # Get adjustment factor
        adjusted = calibrator.adjust_confidence(0.85, "sql")
    """

    def __init__(self, min_observations: int = 100):
        """
        Initialize calibrator

        Args:
            min_observations: Minimum observations required for calibration
        """
        self.calibration_history: List[CalibrationObservation] = []
        self.min_observations = min_observations

        # Statistics
        self.observation_count = 0
        self.backend_counts = {}

    def add_observation(
        self,
        predicted_confidence: float,
        actual_confidence: float,
        backend: str
    ):
        """
        Record prediction vs. outcome

        Args:
            predicted_confidence: Predicted confidence (0.0-1.0)
            actual_confidence: Actual confidence from result (0.0-1.0)
            backend: Backend used (sql, neo4j, qdrant)
        """
        error = abs(predicted_confidence - actual_confidence)

        obs = CalibrationObservation(
            predicted_confidence=predicted_confidence,
            actual_confidence=actual_confidence,
            backend=backend,
            error=error
        )

        self.calibration_history.append(obs)
        self.observation_count += 1

        # Update backend counts
        self.backend_counts[backend] = self.backend_counts.get(backend, 0) + 1

        logger.debug(
            f"Calibration observation: backend={backend}, "
            f"predicted={predicted_confidence:.3f}, actual={actual_confidence:.3f}, "
            f"error={error:.3f}"
        )

    def get_calibration_curve(
        self,
        backend: Optional[str] = None,
        num_bins: int = 10
    ) -> CalibrationCurve:
        """
        Compute calibration curve (predicted vs. actual)

        Args:
            backend: Specific backend (None for all)
            num_bins: Number of bins for calibration curve (default: 10)

        Returns:
            CalibrationCurve with ECE and binned data
        """
        # Filter history
        history = self.calibration_history
        if backend:
            history = [obs for obs in history if obs.backend == backend]

        # Check minimum observations
        if len(history) < self.min_observations:
            return CalibrationCurve(
                calibrated=False,
                ece=float('inf'),
                bin_centers=[],
                binned_actual=[],
                binned_predicted=[],
                sample_size=len(history),
                backend=backend
            )

        # Create bins [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        bins = [i / num_bins for i in range(num_bins + 1)]
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(num_bins)]

        # Bin observations
        binned_actual = []
        binned_predicted = []

        for i in range(num_bins):
            # Find observations in this bin
            in_bin = [
                obs for obs in history
                if bins[i] <= obs.predicted_confidence < bins[i+1]
            ]

            # Special case: last bin includes 1.0
            if i == num_bins - 1:
                in_bin = [
                    obs for obs in history
                    if bins[i] <= obs.predicted_confidence <= bins[i+1]
                ]

            if in_bin:
                # Average actual confidence in this bin
                avg_actual = sum(obs.actual_confidence for obs in in_bin) / len(in_bin)
                binned_actual.append(avg_actual)
                binned_predicted.append(bin_centers[i])
            else:
                binned_actual.append(None)
                binned_predicted.append(bin_centers[i])

        # Compute Expected Calibration Error (ECE)
        ece_values = [
            abs(pred - actual)
            for pred, actual in zip(binned_predicted, binned_actual)
            if actual is not None
        ]

        if ece_values:
            ece = sum(ece_values) / len(ece_values)
        else:
            ece = float('inf')

        # Use small epsilon for floating point comparison
        CALIBRATION_THRESHOLD = 0.10
        EPSILON = 1e-9

        return CalibrationCurve(
            calibrated=(ece < CALIBRATION_THRESHOLD + EPSILON),
            ece=ece,
            bin_centers=bin_centers,
            binned_actual=binned_actual,
            binned_predicted=binned_predicted,
            sample_size=len(history),
            backend=backend
        )

    def adjust_confidence(
        self,
        predicted: float,
        backend: str
    ) -> float:
        """
        Adjust confidence based on calibration

        Args:
            predicted: Predicted confidence
            backend: Backend used

        Returns:
            Adjusted confidence (calibrated)
        """
        # Get calibration curve
        curve = self.get_calibration_curve(backend)

        if not curve.calibrated:
            # Not enough data - no adjustment
            return predicted

        # Find bin
        bin_idx = int(predicted * len(curve.bin_centers))
        if bin_idx >= len(curve.bin_centers):
            bin_idx = len(curve.bin_centers) - 1

        # Get actual confidence for this bin
        actual = curve.binned_actual[bin_idx]

        if actual is None:
            # No data for this bin - no adjustment
            return predicted

        # Return calibrated confidence
        return actual

    def get_adjustment_factor(
        self,
        predicted: float,
        backend: str
    ) -> float:
        """
        Get calibration adjustment multiplier

        Args:
            predicted: Predicted confidence
            backend: Backend used

        Returns:
            Adjustment factor (actual / predicted)
        """
        adjusted = self.adjust_confidence(predicted, backend)

        if predicted > 0:
            return adjusted / predicted
        else:
            return 1.0

    def get_backend_calibration(self, backend: str) -> Dict:
        """
        Get calibration metrics for specific backend

        Args:
            backend: Backend name

        Returns:
            Dict with calibration metrics
        """
        curve = self.get_calibration_curve(backend)

        # Calculate average error
        backend_history = [
            obs for obs in self.calibration_history
            if obs.backend == backend
        ]

        if backend_history:
            avg_error = sum(obs.error for obs in backend_history) / len(backend_history)
            max_error = max(obs.error for obs in backend_history)
        else:
            avg_error = 0.0
            max_error = 0.0

        return {
            "backend": backend,
            "calibrated": curve.calibrated,
            "ece": curve.ece,
            "sample_size": curve.sample_size,
            "avg_error": avg_error,
            "max_error": max_error,
            "observations": len(backend_history)
        }

    def get_summary(self) -> str:
        """
        Get calibration summary

        Returns:
            Human-readable summary
        """
        summary = f"Confidence Calibration Summary:\n"
        summary += f"  Total observations: {self.observation_count}\n"
        summary += f"  Min observations for calibration: {self.min_observations}\n\n"

        # Overall calibration
        overall_curve = self.get_calibration_curve()
        summary += f"Overall:\n"
        summary += f"  Calibrated: {overall_curve.calibrated}\n"
        summary += f"  ECE: {overall_curve.ece:.3f} ({'<0.10 target' if overall_curve.calibrated else '>=0.10'})\n"
        summary += f"  Sample size: {overall_curve.sample_size}\n\n"

        # Per-backend calibration
        for backend in sorted(self.backend_counts.keys()):
            backend_metrics = self.get_backend_calibration(backend)
            summary += f"{backend.upper()}:\n"
            summary += f"  Calibrated: {backend_metrics['calibrated']}\n"
            summary += f"  ECE: {backend_metrics['ece']:.3f}\n"
            summary += f"  Avg error: {backend_metrics['avg_error']:.3f}\n"
            summary += f"  Sample size: {backend_metrics['sample_size']}\n\n"

        return summary

    def get_stats(self) -> Dict:
        """
        Get calibration statistics

        Returns:
            Dict with calibration stats
        """
        overall_curve = self.get_calibration_curve()

        return {
            "total_observations": self.observation_count,
            "min_observations": self.min_observations,
            "overall": {
                "calibrated": overall_curve.calibrated,
                "ece": overall_curve.ece,
                "sample_size": overall_curve.sample_size
            },
            "backends": {
                backend: self.get_backend_calibration(backend)
                for backend in self.backend_counts.keys()
            }
        }

    def reset_stats(self):
        """Reset calibration statistics (for testing)"""
        self.calibration_history = []
        self.observation_count = 0
        self.backend_counts = {}


# ============================================================================
# Helper Functions
# ============================================================================

def create_confidence_calibrator(
    min_observations: int = 100
) -> ConfidenceCalibrator:
    """
    Create confidence calibrator

    Args:
        min_observations: Minimum observations for calibration

    Returns:
        ConfidenceCalibrator instance
    """
    return ConfidenceCalibrator(min_observations=min_observations)
