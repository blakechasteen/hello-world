"""
Confidence Calibration Metrics
==============================

Metrics for evaluating how well reported confidence matches actual accuracy.

"A well-calibrated model reports 80% confidence when it's correct 80% of the time."

Metrics Implemented:
- ECE: Expected Calibration Error (average gap between confidence and accuracy)
- MCE: Maximum Calibration Error (worst bucket)
- Brier Score: Probabilistic accuracy (lower is better)
- Calibration Curve: (confidence, accuracy) pairs per bin
- Reliability Diagram: Data for visualization

Integration Points:
- Uses UncertaintyEnvelope from governance system
- Supports both binary and multi-class predictions
- Works with HoloLoom's confidence scores

Author: HoloLoom Architecture Team
Date: December 2025
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationMetrics:
    """
    Complete calibration metrics for a prediction system.

    Attributes:
        expected_calibration_error: ECE - weighted average of bin calibration errors
        maximum_calibration_error: MCE - worst bin calibration error
        brier_score: Mean squared error of probability forecasts (0-1)
        calibration_curve: List of (confidence, accuracy) pairs per bin
        reliability_diagram: Data for reliability diagram visualization
        total_predictions: Number of predictions evaluated
        num_bins: Number of bins used for calibration
        metadata: Additional evaluation metadata
    """
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    brier_score: float = 0.0
    calibration_curve: List[Tuple[float, float]] = field(default_factory=list)
    reliability_diagram: Dict[str, Any] = field(default_factory=dict)
    total_predictions: int = 0
    num_bins: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "calibration_curve": self.calibration_curve,
            "reliability_diagram": self.reliability_diagram,
            "total_predictions": self.total_predictions,
            "num_bins": self.num_bins,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary of calibration metrics."""
        lines = [
            f"Calibration Metrics ({self.total_predictions} predictions, {self.num_bins} bins)",
            f"  ECE: {self.expected_calibration_error:.4f}",
            f"  MCE: {self.maximum_calibration_error:.4f}",
            f"  Brier Score: {self.brier_score:.4f}",
            "",
            "Calibration Curve:",
        ]
        for conf, acc in self.calibration_curve:
            gap = abs(conf - acc)
            indicator = "✓" if gap < 0.1 else "⚠" if gap < 0.2 else "✗"
            lines.append(f"  {indicator} Conf={conf:.2f}: Acc={acc:.2f} (gap={gap:.2f})")

        return "\n".join(lines)

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if ECE is below threshold (default: 0.1)."""
        return self.expected_calibration_error < threshold

    def get_calibration_quality(self) -> str:
        """Get qualitative calibration quality."""
        ece = self.expected_calibration_error
        if ece < 0.05:
            return "excellent"
        elif ece < 0.1:
            return "good"
        elif ece < 0.15:
            return "fair"
        elif ece < 0.2:
            return "poor"
        else:
            return "very_poor"


@dataclass
class Prediction:
    """A single prediction with confidence and outcome."""
    confidence: float  # Reported confidence (0-1)
    correct: bool  # Whether prediction was correct
    predicted_class: Optional[int] = None
    true_class: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationBin:
    """Statistics for a single calibration bin."""
    bin_lower: float
    bin_upper: float
    bin_center: float
    count: int
    accuracy: float
    avg_confidence: float
    calibration_error: float  # |accuracy - avg_confidence|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "bin_center": self.bin_center,
            "count": self.count,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "calibration_error": self.calibration_error,
        }


# =============================================================================
# Individual Metric Functions
# =============================================================================

def compute_calibration_bins(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> List[CalibrationBin]:
    """
    Compute calibration statistics per bin.

    Bins predictions by confidence level and calculates accuracy per bin.

    Args:
        predictions: List of predictions with confidence and correctness
        num_bins: Number of equally-spaced bins

    Returns:
        List of CalibrationBin objects
    """
    if not predictions:
        return []

    # Initialize bins
    bins: List[List[Prediction]] = [[] for _ in range(num_bins)]

    # Assign predictions to bins
    for pred in predictions:
        # Handle edge case where confidence is exactly 1.0
        bin_idx = min(int(pred.confidence * num_bins), num_bins - 1)
        bins[bin_idx].append(pred)

    # Compute bin statistics
    result = []
    for i, bin_preds in enumerate(bins):
        bin_lower = i / num_bins
        bin_upper = (i + 1) / num_bins
        bin_center = (bin_lower + bin_upper) / 2

        if bin_preds:
            count = len(bin_preds)
            accuracy = sum(1 for p in bin_preds if p.correct) / count
            avg_confidence = sum(p.confidence for p in bin_preds) / count
            calibration_error = abs(accuracy - avg_confidence)
        else:
            count = 0
            accuracy = 0.0
            avg_confidence = bin_center
            calibration_error = 0.0

        result.append(CalibrationBin(
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            bin_center=bin_center,
            count=count,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            calibration_error=calibration_error,
        ))

    return result


def expected_calibration_error(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ (n_b / N) × |accuracy_b - confidence_b|

    Weighted average of absolute calibration error per bin.

    Args:
        predictions: List of predictions with confidence and correctness
        num_bins: Number of equally-spaced bins

    Returns:
        ECE score (0-1, lower is better)
    """
    if not predictions:
        return 0.0

    bins = compute_calibration_bins(predictions, num_bins)
    total_count = len(predictions)

    ece = 0.0
    for bin_stats in bins:
        weight = bin_stats.count / total_count
        ece += weight * bin_stats.calibration_error

    return ece


def maximum_calibration_error(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE).

    MCE = max_b |accuracy_b - confidence_b|

    Worst-case calibration error across all bins.

    Args:
        predictions: List of predictions with confidence and correctness
        num_bins: Number of equally-spaced bins

    Returns:
        MCE score (0-1, lower is better)
    """
    if not predictions:
        return 0.0

    bins = compute_calibration_bins(predictions, num_bins)

    # Only consider non-empty bins
    non_empty_bins = [b for b in bins if b.count > 0]
    if not non_empty_bins:
        return 0.0

    return max(b.calibration_error for b in non_empty_bins)


def brier_score(predictions: List[Prediction]) -> float:
    """
    Brier Score.

    BS = (1/N) Σ (confidence_i - correct_i)²

    Mean squared error of probability forecasts.
    For binary classification, correct_i is 0 or 1.

    Args:
        predictions: List of predictions with confidence and correctness

    Returns:
        Brier score (0-1, lower is better)
    """
    if not predictions:
        return 0.0

    squared_errors = [
        (pred.confidence - (1.0 if pred.correct else 0.0)) ** 2
        for pred in predictions
    ]

    return sum(squared_errors) / len(squared_errors)


def compute_calibration_curve(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> List[Tuple[float, float]]:
    """
    Compute calibration curve as (avg_confidence, accuracy) pairs.

    Args:
        predictions: List of predictions with confidence and correctness
        num_bins: Number of equally-spaced bins

    Returns:
        List of (avg_confidence, accuracy) tuples for each non-empty bin
    """
    bins = compute_calibration_bins(predictions, num_bins)

    return [
        (b.avg_confidence, b.accuracy)
        for b in bins
        if b.count > 0
    ]


def compute_reliability_diagram(
    predictions: List[Prediction],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute data for reliability diagram visualization.

    Returns:
        Dict with bin data for plotting:
        - bin_centers: List of bin center positions
        - accuracies: List of accuracy values per bin
        - confidences: List of average confidence per bin
        - counts: List of sample counts per bin
        - gaps: List of calibration gaps (positive = overconfident)
    """
    bins = compute_calibration_bins(predictions, num_bins)

    non_empty = [b for b in bins if b.count > 0]

    return {
        "bin_centers": [b.bin_center for b in non_empty],
        "accuracies": [b.accuracy for b in non_empty],
        "confidences": [b.avg_confidence for b in non_empty],
        "counts": [b.count for b in non_empty],
        "gaps": [b.avg_confidence - b.accuracy for b in non_empty],  # Positive = overconfident
        "all_bins": [b.to_dict() for b in bins],
    }


# =============================================================================
# Evaluator Class
# =============================================================================

class CalibrationEvaluator:
    """
    Comprehensive calibration evaluator for prediction systems.

    Supports:
    - Standard calibration metrics (ECE, MCE, Brier)
    - Multiple binning strategies
    - Reliability diagram generation
    - Integration with HoloLoom's confidence system
    """

    def __init__(
        self,
        num_bins: int = 10,
        min_samples_per_bin: int = 1,
    ):
        """
        Initialize calibration evaluator.

        Args:
            num_bins: Number of bins for calibration analysis
            min_samples_per_bin: Minimum samples required for bin to be counted
        """
        self.num_bins = num_bins
        self.min_samples_per_bin = min_samples_per_bin

    def evaluate(
        self,
        predictions: List[Prediction],
    ) -> CalibrationMetrics:
        """
        Evaluate calibration across all predictions.

        Args:
            predictions: List of predictions with confidence and correctness

        Returns:
            CalibrationMetrics with all computed metrics
        """
        if not predictions:
            return CalibrationMetrics()

        return CalibrationMetrics(
            expected_calibration_error=expected_calibration_error(predictions, self.num_bins),
            maximum_calibration_error=maximum_calibration_error(predictions, self.num_bins),
            brier_score=brier_score(predictions),
            calibration_curve=compute_calibration_curve(predictions, self.num_bins),
            reliability_diagram=compute_reliability_diagram(predictions, self.num_bins),
            total_predictions=len(predictions),
            num_bins=self.num_bins,
            metadata={
                "min_samples_per_bin": self.min_samples_per_bin,
            }
        )

    def evaluate_from_scores(
        self,
        confidences: List[float],
        correct: List[bool],
    ) -> CalibrationMetrics:
        """
        Evaluate calibration from raw scores.

        Convenience method when you have separate lists.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of correctness indicators (True/False)

        Returns:
            CalibrationMetrics
        """
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have same length")

        predictions = [
            Prediction(confidence=c, correct=cor)
            for c, cor in zip(confidences, correct)
        ]

        return self.evaluate(predictions)

    def evaluate_regression(
        self,
        predicted_values: List[float],
        true_values: List[float],
        confidence_intervals: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Evaluate calibration for regression with confidence intervals.

        Checks if true values fall within confidence intervals at the expected rate.

        Args:
            predicted_values: Predicted values
            true_values: True values
            confidence_intervals: (lower, upper) bounds for each prediction

        Returns:
            Dict with coverage statistics
        """
        if len(predicted_values) != len(true_values) != len(confidence_intervals):
            raise ValueError("All lists must have same length")

        covered = 0
        for true_val, (lower, upper) in zip(true_values, confidence_intervals):
            if lower <= true_val <= upper:
                covered += 1

        coverage = covered / len(true_values) if true_values else 0

        # Expected coverage is typically 0.95 for 95% CI
        # Gap indicates miscalibration
        return {
            "coverage": coverage,
            "expected_coverage": 0.95,  # Assuming 95% CI
            "coverage_gap": coverage - 0.95,
            "total_predictions": len(true_values),
        }

    def analyze_by_confidence_band(
        self,
        predictions: List[Prediction],
        bands: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze calibration within specific confidence bands.

        Useful for understanding calibration at different confidence levels.

        Args:
            predictions: List of predictions
            bands: List of (lower, upper) confidence bands

        Returns:
            Dict mapping band description to metrics
        """
        if bands is None:
            bands = [
                (0.0, 0.2),   # Very low confidence
                (0.2, 0.4),   # Low confidence
                (0.4, 0.6),   # Medium confidence
                (0.6, 0.8),   # High confidence
                (0.8, 1.0),   # Very high confidence
            ]

        results = {}
        for lower, upper in bands:
            band_name = f"{lower:.1f}-{upper:.1f}"
            band_preds = [
                p for p in predictions
                if lower <= p.confidence < upper
            ]

            if band_preds:
                accuracy = sum(1 for p in band_preds if p.correct) / len(band_preds)
                avg_conf = sum(p.confidence for p in band_preds) / len(band_preds)
                results[band_name] = {
                    "count": len(band_preds),
                    "accuracy": accuracy,
                    "avg_confidence": avg_conf,
                    "calibration_error": abs(accuracy - avg_conf),
                }
            else:
                results[band_name] = {
                    "count": 0,
                    "accuracy": 0.0,
                    "avg_confidence": 0.0,
                    "calibration_error": 0.0,
                }

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def create_prediction(
    confidence: float,
    correct: bool,
    **metadata
) -> Prediction:
    """
    Create a Prediction object.

    Args:
        confidence: Confidence score (0-1)
        correct: Whether prediction was correct
        **metadata: Additional metadata

    Returns:
        Prediction instance
    """
    return Prediction(
        confidence=confidence,
        correct=correct,
        metadata=metadata,
    )


def evaluate_simple(
    confidences: List[float],
    correct: List[bool],
    num_bins: int = 10,
) -> CalibrationMetrics:
    """
    Simple calibration evaluation.

    Args:
        confidences: List of confidence scores (0-1)
        correct: List of correctness indicators
        num_bins: Number of bins

    Returns:
        CalibrationMetrics
    """
    evaluator = CalibrationEvaluator(num_bins=num_bins)
    return evaluator.evaluate_from_scores(confidences, correct)


def is_overconfident(
    confidences: List[float],
    correct: List[bool],
    threshold: float = 0.8,
) -> bool:
    """
    Check if system is overconfident in high-confidence predictions.

    Args:
        confidences: Confidence scores
        correct: Correctness indicators
        threshold: Confidence threshold for "high confidence"

    Returns:
        True if accuracy is significantly below confidence for high-conf predictions
    """
    high_conf_preds = [
        (c, cor) for c, cor in zip(confidences, correct)
        if c >= threshold
    ]

    if not high_conf_preds:
        return False

    avg_conf = sum(c for c, _ in high_conf_preds) / len(high_conf_preds)
    accuracy = sum(1 for _, cor in high_conf_preds if cor) / len(high_conf_preds)

    # Overconfident if confidence > accuracy + gap
    return avg_conf > accuracy + 0.1


def is_underconfident(
    confidences: List[float],
    correct: List[bool],
    threshold: float = 0.4,
) -> bool:
    """
    Check if system is underconfident in low-confidence predictions.

    Args:
        confidences: Confidence scores
        correct: Correctness indicators
        threshold: Confidence threshold for "low confidence"

    Returns:
        True if accuracy is significantly above confidence for low-conf predictions
    """
    low_conf_preds = [
        (c, cor) for c, cor in zip(confidences, correct)
        if c <= threshold
    ]

    if not low_conf_preds:
        return False

    avg_conf = sum(c for c, _ in low_conf_preds) / len(low_conf_preds)
    accuracy = sum(1 for _, cor in low_conf_preds if cor) / len(low_conf_preds)

    # Underconfident if accuracy > confidence + gap
    return accuracy > avg_conf + 0.1
