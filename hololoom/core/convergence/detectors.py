"""
Convergence Detectors - Automatic Stopping Criteria
====================================================

Detects when recursive refinement has converged using multiple strategies.

Created: 2025-11-20
Author: HoloLoom Team
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from HoloLoom.core.protocols.recursive_reasoning import (
    RefinementStep,
    ConvergenceDetectorProtocol
)

logger = logging.getLogger(__name__)


# ============================================================================
# Base Detector
# ============================================================================

class ConvergenceDetector:
    """
    Base class for convergence detection.

    All detectors implement: detect(history) → (converged, reason)
    """

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """
        Detect convergence from refinement history.

        Args:
            history: List of refinement steps

        Returns:
            (converged, reason) tuple
        """
        raise NotImplementedError("Subclasses must implement detect()")

    def _get_confidences(self, history: List[RefinementStep]) -> List[float]:
        """Extract confidence scores from history."""
        return [step.confidence for step in history]

    def _get_improvements(self, history: List[RefinementStep]) -> List[float]:
        """Extract improvement deltas from history."""
        return [step.improvement for step in history]


# ============================================================================
# Confidence Threshold Detector
# ============================================================================

class ConfidenceThresholdDetector(ConvergenceDetector):
    """
    Converge when confidence >= threshold.

    Simple strategy: Stop when quality is "good enough".
    """

    def __init__(self, threshold: float = 0.85):
        """
        Initialize detector.

        Args:
            threshold: Confidence threshold (default: 0.85)
        """
        self.threshold = threshold
        logger.info(f"ConfidenceThresholdDetector initialized (threshold={threshold})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect if latest confidence >= threshold."""
        if not history:
            return False, "No history"

        latest_confidence = history[-1].confidence

        if latest_confidence >= self.threshold:
            return True, f"Confidence {latest_confidence:.2f} >= threshold {self.threshold}"

        return False, f"Confidence {latest_confidence:.2f} < threshold {self.threshold}"


# ============================================================================
# Improvement Delta Detector
# ============================================================================

class ImprovementDeltaDetector(ConvergenceDetector):
    """
    Converge when improvement < minimum delta.

    Strategy: Stop when marginal gains become negligible.
    """

    def __init__(self, min_delta: float = 0.05):
        """
        Initialize detector.

        Args:
            min_delta: Minimum improvement required (default: 0.05)
        """
        self.min_delta = min_delta
        logger.info(f"ImprovementDeltaDetector initialized (min_delta={min_delta})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect if latest improvement < min_delta."""
        if len(history) < 2:
            return False, "Insufficient history (need 2+ steps)"

        latest_improvement = history[-1].improvement

        if abs(latest_improvement) < self.min_delta:
            return True, f"Improvement {latest_improvement:+.3f} < min_delta {self.min_delta}"

        return False, f"Improvement {latest_improvement:+.3f} >= min_delta {self.min_delta}"


# ============================================================================
# Quality Plateau Detector
# ============================================================================

class QualityPlateauDetector(ConvergenceDetector):
    """
    Converge when quality plateaus (no improvement for N steps).

    Strategy: Detect when refinement is no longer helping.
    """

    def __init__(self, plateau_window: int = 3, min_delta: float = 0.02):
        """
        Initialize detector.

        Args:
            plateau_window: Number of steps to check for plateau (default: 3)
            min_delta: Minimum improvement to count as non-plateau (default: 0.02)
        """
        self.plateau_window = plateau_window
        self.min_delta = min_delta
        logger.info(f"QualityPlateauDetector initialized (window={plateau_window}, min_delta={min_delta})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect if quality has plateaued for N consecutive steps."""
        if len(history) < self.plateau_window:
            return False, f"Insufficient history (need {self.plateau_window}+ steps)"

        # Check last N improvements
        recent_improvements = [step.improvement for step in history[-self.plateau_window:]]

        # Count how many are below threshold
        below_threshold = sum(1 for imp in recent_improvements if abs(imp) < self.min_delta)

        if below_threshold >= self.plateau_window:
            return True, f"Quality plateaued for {self.plateau_window} steps (improvements < {self.min_delta})"

        return False, f"Quality still improving (plateau check: {below_threshold}/{self.plateau_window})"


# ============================================================================
# Max Iterations Detector
# ============================================================================

class MaxIterationsDetector(ConvergenceDetector):
    """
    Converge when max iterations reached.

    Fallback strategy: Hard limit to prevent infinite loops.
    """

    def __init__(self, max_iterations: int = 5):
        """
        Initialize detector.

        Args:
            max_iterations: Maximum iterations (default: 5)
        """
        self.max_iterations = max_iterations
        logger.info(f"MaxIterationsDetector initialized (max_iterations={max_iterations})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect if max iterations reached."""
        if len(history) >= self.max_iterations:
            return True, f"Max iterations {self.max_iterations} reached"

        return False, f"Iterations {len(history)}/{self.max_iterations}"


# ============================================================================
# High Variance Detector
# ============================================================================

class HighVarianceDetector(ConvergenceDetector):
    """
    Converge when confidence variance is too high (unstable refinement).

    Strategy: Stop if refinement is making things worse or highly inconsistent.
    """

    def __init__(self, variance_threshold: float = 0.15, window: int = 3):
        """
        Initialize detector.

        Args:
            variance_threshold: Maximum acceptable variance (default: 0.15)
            window: Window size for variance calculation (default: 3)
        """
        self.variance_threshold = variance_threshold
        self.window = window
        logger.info(f"HighVarianceDetector initialized (threshold={variance_threshold}, window={window})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect if confidence variance is too high."""
        if len(history) < self.window:
            return False, f"Insufficient history (need {self.window}+ steps)"

        # Calculate variance of recent confidences
        recent_confidences = [step.confidence for step in history[-self.window:]]
        variance = np.var(recent_confidences)

        if variance > self.variance_threshold:
            return True, f"High variance {variance:.3f} > threshold {self.variance_threshold} (unstable refinement)"

        return False, f"Variance {variance:.3f} <= threshold {self.variance_threshold}"


# ============================================================================
# Multi-Criteria Detector
# ============================================================================

class MultiCriteriaDetector(ConvergenceDetector):
    """
    Combine multiple detectors with voting or AND/OR logic.

    Most flexible strategy: Use multiple signals to decide convergence.
    """

    def __init__(
        self,
        detectors: List[ConvergenceDetector],
        logic: str = "OR",  # "OR", "AND", "MAJORITY"
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-criteria detector.

        Args:
            detectors: List of detector instances
            logic: Combination logic ("OR", "AND", "MAJORITY")
            weights: Optional weights for each detector (for MAJORITY voting)
        """
        self.detectors = detectors
        self.logic = logic.upper()
        self.weights = weights or [1.0] * len(detectors)

        if len(self.weights) != len(detectors):
            raise ValueError("Weights must match number of detectors")

        logger.info(f"MultiCriteriaDetector initialized ({len(detectors)} detectors, logic={self.logic})")

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """Detect using combined criteria."""
        results = []
        reasons = []

        # Run all detectors
        for i, detector in enumerate(self.detectors):
            converged, reason = detector.detect(history)
            results.append((converged, self.weights[i]))
            reasons.append(f"{detector.__class__.__name__}: {reason}")

        # Apply logic
        if self.logic == "OR":
            # Any detector triggers convergence
            converged = any(r[0] for r in results)
            combined_reason = " | ".join(reasons) if converged else "No convergence criteria met"

        elif self.logic == "AND":
            # All detectors must trigger convergence
            converged = all(r[0] for r in results)
            combined_reason = " & ".join(reasons) if converged else "Not all criteria met"

        elif self.logic == "MAJORITY":
            # Weighted voting (>50% vote for convergence)
            total_weight = sum(self.weights)
            converged_weight = sum(weight for converged, weight in results if converged)
            converged = converged_weight > (total_weight / 2)
            combined_reason = f"Weighted vote: {converged_weight:.1f}/{total_weight:.1f} ({' | '.join(reasons)})"

        else:
            raise ValueError(f"Unknown logic: {self.logic}")

        return converged, combined_reason


# ============================================================================
# Factory Functions
# ============================================================================

def create_confidence_detector(threshold: float = 0.85) -> ConfidenceThresholdDetector:
    """Create confidence threshold detector."""
    return ConfidenceThresholdDetector(threshold=threshold)


def create_improvement_detector(min_delta: float = 0.05) -> ImprovementDeltaDetector:
    """Create improvement delta detector."""
    return ImprovementDeltaDetector(min_delta=min_delta)


def create_plateau_detector(
    plateau_window: int = 3,
    min_delta: float = 0.02
) -> QualityPlateauDetector:
    """Create quality plateau detector."""
    return QualityPlateauDetector(plateau_window=plateau_window, min_delta=min_delta)


def create_multi_criteria_detector(
    confidence_threshold: float = 0.85,
    min_improvement: float = 0.05,
    plateau_window: int = 3,
    max_iterations: int = 5,
    logic: str = "OR"
) -> MultiCriteriaDetector:
    """
    Create multi-criteria detector with standard detectors.

    Args:
        confidence_threshold: Confidence threshold
        min_improvement: Minimum improvement delta
        plateau_window: Plateau detection window
        max_iterations: Maximum iterations
        logic: Combination logic ("OR", "AND", "MAJORITY")

    Returns:
        Configured MultiCriteriaDetector
    """
    detectors = [
        ConfidenceThresholdDetector(threshold=confidence_threshold),
        ImprovementDeltaDetector(min_delta=min_improvement),
        QualityPlateauDetector(plateau_window=plateau_window),
        MaxIterationsDetector(max_iterations=max_iterations),
    ]

    return MultiCriteriaDetector(detectors=detectors, logic=logic)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime

    print("="*80)
    print("Convergence Detection Demo")
    print("="*80 + "\n")

    # Simulate refinement history
    simulated_history = [
        RefinementStep(1, "Query 1", "Response 1", 0.65, 0.0, "initial", "Initial response", datetime.utcnow()),
        RefinementStep(2, "Query 2", "Response 2", 0.75, 0.10, "expand_search", "Expanded sources", datetime.utcnow()),
        RefinementStep(3, "Query 3", "Response 3", 0.82, 0.07, "rerank", "Reranked sources", datetime.utcnow()),
        RefinementStep(4, "Query 4", "Response 4", 0.86, 0.04, "verify", "Verified accuracy", datetime.utcnow()),
        RefinementStep(5, "Query 5", "Response 5", 0.87, 0.01, "alternate_mode", "Alternate reasoning", datetime.utcnow()),
    ]

    print("Simulated Refinement History:")
    print("  Iteration | Confidence | Improvement")
    print("  " + "-"*40)
    for step in simulated_history:
        print(f"     {step.iteration}      |   {step.confidence:.2f}     |   {step.improvement:+.3f}")

    print("\n" + "="*80)
    print("Testing Detectors:")
    print("="*80 + "\n")

    # Test individual detectors
    detectors = [
        ("Confidence Threshold (0.85)", ConfidenceThresholdDetector(0.85)),
        ("Improvement Delta (0.05)", ImprovementDeltaDetector(0.05)),
        ("Quality Plateau (window=3)", QualityPlateauDetector(plateau_window=3, min_delta=0.02)),
        ("Max Iterations (5)", MaxIterationsDetector(5)),
    ]

    for name, detector in detectors:
        converged, reason = detector.detect(simulated_history)
        status = "✓ CONVERGED" if converged else "✗ Continue"
        print(f"{name:30s} → {status}")
        print(f"  Reason: {reason}\n")

    # Test multi-criteria
    print("="*80)
    print("Multi-Criteria Detector:")
    print("="*80 + "\n")

    for logic in ["OR", "AND", "MAJORITY"]:
        multi_detector = create_multi_criteria_detector(
            confidence_threshold=0.85,
            min_improvement=0.05,
            plateau_window=3,
            max_iterations=5,
            logic=logic
        )

        converged, reason = multi_detector.detect(simulated_history)
        status = "✓ CONVERGED" if converged else "✗ Continue"
        print(f"{logic:10s} logic → {status}")
        print(f"  Reason: {reason}\n")

    print("✓ Demo complete!")
