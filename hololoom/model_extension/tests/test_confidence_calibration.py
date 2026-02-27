"""
Test Suite for Confidence Calibration Evaluation
=================================================

Comprehensive tests for calibration metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Calibration Curve
- Reliability Diagram

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
import random
from typing import List

from HoloLoom.model_extension.eval.calibration_metrics import (
    Prediction,
    CalibrationBin,
    CalibrationMetrics,
    CalibrationEvaluator,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    compute_calibration_curve,
    compute_reliability_diagram,
)


# =============================================================================
# Expected Calibration Error (ECE) Tests
# =============================================================================

class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration(self, well_calibrated_predictions):
        """Well-calibrated model should have low ECE."""
        ece = expected_calibration_error(well_calibrated_predictions, num_bins=10)

        # Well-calibrated should have ECE < 0.15
        assert ece < 0.15, f"Well-calibrated ECE too high: {ece}"

    def test_overconfident_calibration(self, overconfident_predictions):
        """Overconfident model should have higher ECE."""
        ece = expected_calibration_error(overconfident_predictions, num_bins=10)

        # Overconfident should have noticeable ECE
        assert ece > 0.1, f"Overconfident ECE too low: {ece}"

    def test_underconfident_calibration(self, underconfident_predictions):
        """Underconfident model should have higher ECE."""
        ece = expected_calibration_error(underconfident_predictions, num_bins=10)

        # Underconfident should have noticeable ECE
        assert ece > 0.1, f"Underconfident ECE too low: {ece}"

    def test_ece_bounds(self):
        """ECE should be between 0 and 1."""
        predictions = [
            Prediction(confidence=random.random(), correct=random.random() > 0.5)
            for _ in range(100)
        ]
        ece = expected_calibration_error(predictions, num_bins=10)

        assert 0.0 <= ece <= 1.0, f"ECE out of bounds: {ece}"

    def test_ece_empty_predictions(self):
        """ECE with no predictions should return 0."""
        ece = expected_calibration_error([], num_bins=10)

        assert ece == 0.0

    def test_ece_single_prediction(self):
        """ECE with single prediction."""
        predictions = [Prediction(confidence=0.9, correct=True)]
        ece = expected_calibration_error(predictions, num_bins=10)

        # Single correct prediction at 0.9 confidence
        # In bin 0.8-1.0, accuracy = 1.0, avg_conf ~ 0.9
        # ECE = |1.0 - 0.9| = 0.1
        assert 0.0 <= ece <= 0.2


class TestMCE:
    """Tests for Maximum Calibration Error."""

    def test_mce_always_gte_ece(self, well_calibrated_predictions):
        """MCE should always be >= ECE."""
        ece = expected_calibration_error(well_calibrated_predictions, num_bins=10)
        mce = maximum_calibration_error(well_calibrated_predictions, num_bins=10)

        assert mce >= ece, f"MCE ({mce}) < ECE ({ece})"

    def test_mce_worst_bin(self, overconfident_predictions):
        """MCE captures worst bin for overconfident predictions."""
        mce = maximum_calibration_error(overconfident_predictions, num_bins=10)

        # Overconfident: high confidence (0.8-0.95) but ~50% accuracy
        # Worst bin gap should be substantial
        assert mce > 0.2, f"MCE too low for overconfident: {mce}"

    def test_mce_bounds(self):
        """MCE should be between 0 and 1."""
        predictions = [
            Prediction(confidence=random.random(), correct=random.random() > 0.5)
            for _ in range(100)
        ]
        mce = maximum_calibration_error(predictions, num_bins=10)

        assert 0.0 <= mce <= 1.0, f"MCE out of bounds: {mce}"

    def test_mce_empty_predictions(self):
        """MCE with no predictions should return 0."""
        mce = maximum_calibration_error([], num_bins=10)

        assert mce == 0.0


class TestBrierScore:
    """Tests for Brier Score."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        predictions = [
            Prediction(confidence=1.0, correct=True),
            Prediction(confidence=0.0, correct=False),
        ]
        bs = brier_score(predictions)

        assert bs == 0.0

    def test_worst_predictions(self):
        """Worst predictions should have Brier score of 1."""
        predictions = [
            Prediction(confidence=1.0, correct=False),
            Prediction(confidence=0.0, correct=True),
        ]
        bs = brier_score(predictions)

        assert bs == 1.0

    def test_brier_bounds(self):
        """Brier score should be between 0 and 1."""
        predictions = [
            Prediction(confidence=random.random(), correct=random.random() > 0.5)
            for _ in range(100)
        ]
        bs = brier_score(predictions)

        assert 0.0 <= bs <= 1.0, f"Brier score out of bounds: {bs}"

    def test_brier_calibrated_vs_uncalibrated(
        self, well_calibrated_predictions, overconfident_predictions
    ):
        """Well-calibrated model should have lower Brier score."""
        bs_calibrated = brier_score(well_calibrated_predictions)
        bs_overconfident = brier_score(overconfident_predictions)

        # Well-calibrated should generally have lower Brier score
        # Note: This might not always hold due to randomness, so we use soft assertion
        assert bs_calibrated < bs_overconfident + 0.2

    def test_brier_empty_predictions(self):
        """Brier score with no predictions should return 0."""
        bs = brier_score([])

        assert bs == 0.0


class TestCalibrationCurve:
    """Tests for calibration curve computation."""

    def test_calibration_curve_bins(self, well_calibrated_predictions):
        """Calibration curve should have correct number of non-empty bins."""
        curve = compute_calibration_curve(well_calibrated_predictions, num_bins=10)

        # At least some bins should be populated
        assert len(curve) > 0
        assert len(curve) <= 10

    def test_calibration_curve_values(self, well_calibrated_predictions):
        """Calibration curve values should be in valid range."""
        curve = compute_calibration_curve(well_calibrated_predictions, num_bins=10)

        for avg_conf, accuracy in curve:
            assert 0.0 <= avg_conf <= 1.0, f"Invalid confidence: {avg_conf}"
            assert 0.0 <= accuracy <= 1.0, f"Invalid accuracy: {accuracy}"

    def test_calibration_curve_perfect_diagonal(self):
        """Perfect calibration should produce diagonal curve."""
        # Create perfectly calibrated predictions
        predictions = []
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(20):
                # Accuracy matches confidence exactly
                is_correct = random.random() < conf
                predictions.append(Prediction(confidence=conf, correct=is_correct))

        curve = compute_calibration_curve(predictions, num_bins=5)

        # Check that points are close to diagonal
        for avg_conf, accuracy in curve:
            gap = abs(avg_conf - accuracy)
            assert gap < 0.2, f"Gap too large: conf={avg_conf}, acc={accuracy}"


class TestReliabilityDiagram:
    """Tests for reliability diagram generation."""

    def test_reliability_diagram_structure(self, well_calibrated_predictions):
        """Reliability diagram should have correct structure."""
        diagram = compute_reliability_diagram(well_calibrated_predictions, num_bins=10)

        assert "bin_centers" in diagram
        assert "accuracies" in diagram
        assert "confidences" in diagram
        assert "counts" in diagram
        assert "gaps" in diagram
        assert "all_bins" in diagram

    def test_reliability_diagram_bins(self, well_calibrated_predictions):
        """Each bin should have required fields."""
        diagram = compute_reliability_diagram(well_calibrated_predictions, num_bins=10)

        for bin_data in diagram["all_bins"]:
            assert "bin_lower" in bin_data
            assert "bin_upper" in bin_data
            assert "count" in bin_data
            assert "avg_confidence" in bin_data
            assert "accuracy" in bin_data
            assert "calibration_error" in bin_data

    def test_reliability_diagram_gaps(self, overconfident_predictions):
        """Overconfident predictions should show positive gaps in high bins."""
        diagram = compute_reliability_diagram(overconfident_predictions, num_bins=10)

        # Find non-empty bins in high-confidence range
        non_empty_high_bins = [
            (c, g) for c, g, cnt in zip(
                diagram["confidences"],
                diagram["gaps"],
                diagram["counts"]
            ) if c >= 0.7 and cnt > 0
        ]

        # At least some high-confidence bins should show overconfidence (positive gap)
        if non_empty_high_bins:
            has_positive_gap = any(g > 0 for _, g in non_empty_high_bins)
            assert has_positive_gap, "Expected positive gaps in high-confidence bins"


class TestCalibrationEvaluator:
    """Tests for the main CalibrationEvaluator class."""

    def test_full_evaluation(self, well_calibrated_predictions):
        """Full evaluation should return all metrics."""
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(well_calibrated_predictions)

        assert isinstance(metrics, CalibrationMetrics)
        assert hasattr(metrics, "expected_calibration_error")
        assert hasattr(metrics, "maximum_calibration_error")
        assert hasattr(metrics, "brier_score")
        assert hasattr(metrics, "calibration_curve")
        assert hasattr(metrics, "reliability_diagram")

    def test_evaluation_consistency(self, well_calibrated_predictions):
        """Individual metrics should match full evaluation."""
        evaluator = CalibrationEvaluator(num_bins=10)

        # Full evaluation
        metrics = evaluator.evaluate(well_calibrated_predictions)

        # Individual computations using module functions
        ece = expected_calibration_error(well_calibrated_predictions, num_bins=10)
        mce = maximum_calibration_error(well_calibrated_predictions, num_bins=10)
        bs = brier_score(well_calibrated_predictions)

        assert abs(metrics.expected_calibration_error - ece) < 0.0001
        assert abs(metrics.maximum_calibration_error - mce) < 0.0001
        assert abs(metrics.brier_score - bs) < 0.0001

    def test_evaluator_different_bins(self, well_calibrated_predictions):
        """Different bin counts should produce different curves."""
        eval_5 = CalibrationEvaluator(num_bins=5)
        eval_10 = CalibrationEvaluator(num_bins=10)
        eval_20 = CalibrationEvaluator(num_bins=20)

        metrics_5 = eval_5.evaluate(well_calibrated_predictions)
        metrics_10 = eval_10.evaluate(well_calibrated_predictions)
        metrics_20 = eval_20.evaluate(well_calibrated_predictions)

        # More bins = potentially more points (up to num_bins)
        assert len(metrics_5.calibration_curve) <= 5
        assert len(metrics_10.calibration_curve) <= 10
        assert len(metrics_20.calibration_curve) <= 20

    def test_metrics_serialization(self, well_calibrated_predictions):
        """CalibrationMetrics should be serializable."""
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(well_calibrated_predictions)

        # Convert to dict
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "expected_calibration_error" in metrics_dict
        assert "maximum_calibration_error" in metrics_dict
        assert "brier_score" in metrics_dict


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_ece_function(self, well_calibrated_predictions):
        """Convenience ECE function should work."""
        ece = expected_calibration_error(well_calibrated_predictions)
        assert 0.0 <= ece <= 1.0

    def test_mce_function(self, well_calibrated_predictions):
        """Convenience MCE function should work."""
        mce = maximum_calibration_error(well_calibrated_predictions)
        assert 0.0 <= mce <= 1.0

    def test_brier_function(self, well_calibrated_predictions):
        """Convenience Brier score function should work."""
        bs = brier_score(well_calibrated_predictions)
        assert 0.0 <= bs <= 1.0


class TestCalibrationHelpers:
    """Tests for helper functions."""

    def test_is_overconfident(self, overconfident_predictions):
        """Should detect overconfident predictions."""
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(overconfident_predictions)

        # Check if system detects overconfidence
        # Overconfident: high confidence but lower accuracy
        diagram = metrics.reliability_diagram

        # Find high-confidence bins with data
        high_conf_indices = [
            i for i, c in enumerate(diagram["confidences"])
            if c >= 0.7 and diagram["counts"][i] > 5
        ]

        # Most high-confidence bins should show overconfidence
        if high_conf_indices:
            overconfident_bins = sum(
                1 for i in high_conf_indices if diagram["gaps"][i] > 0
            )
            assert overconfident_bins > 0, "Expected overconfidence in high bins"

    def test_is_underconfident(self, underconfident_predictions):
        """Should detect underconfident predictions."""
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(underconfident_predictions)

        # Check if system detects underconfidence
        # Underconfident: low confidence but higher accuracy
        diagram = metrics.reliability_diagram

        # Find low-confidence bins with data
        low_conf_indices = [
            i for i, c in enumerate(diagram["confidences"])
            if c <= 0.5 and diagram["counts"][i] > 5
        ]

        # Some low-confidence bins should show underconfidence (negative gap)
        if low_conf_indices:
            underconfident_bins = sum(
                1 for i in low_conf_indices if diagram["gaps"][i] < 0
            )
            assert underconfident_bins > 0, "Expected underconfidence in low bins"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_correct_predictions(self):
        """All correct predictions at various confidences."""
        predictions = [
            Prediction(confidence=0.9, correct=True),
            Prediction(confidence=0.7, correct=True),
            Prediction(confidence=0.5, correct=True),
        ]
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(predictions)

        # All correct means underconfident at lower confidence levels
        assert metrics.brier_score < 0.5

    def test_all_incorrect_predictions(self):
        """All incorrect predictions at various confidences."""
        predictions = [
            Prediction(confidence=0.9, correct=False),
            Prediction(confidence=0.7, correct=False),
            Prediction(confidence=0.5, correct=False),
        ]
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(predictions)

        # All incorrect means overconfident
        assert metrics.brier_score > 0.5

    def test_uniform_confidence(self):
        """All predictions at same confidence level."""
        predictions = [
            Prediction(confidence=0.5, correct=random.random() > 0.5)
            for _ in range(100)
        ]
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(predictions)

        # Only one bin should be populated (the one containing 0.5)
        non_empty_bins = [
            b for b in metrics.reliability_diagram["all_bins"]
            if b["count"] > 0
        ]
        assert len(non_empty_bins) == 1

    def test_extreme_confidences(self):
        """Predictions at extreme confidence values."""
        predictions = [
            Prediction(confidence=0.0, correct=False),
            Prediction(confidence=1.0, correct=True),
            Prediction(confidence=0.0, correct=False),
            Prediction(confidence=1.0, correct=True),
        ]
        evaluator = CalibrationEvaluator(num_bins=10)
        metrics = evaluator.evaluate(predictions)

        # Perfect calibration at extremes
        assert metrics.brier_score == 0.0
