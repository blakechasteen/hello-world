"""
Tests for convergence detectors (detectors.py).

Covers all detector classes, detection logic, edge cases, and factory functions.
"""

import sys
import types
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from unittest.mock import patch

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# RefinementStep stub — the real protocols module is missing this definition,
# so we define a compatible dataclass here and patch it into the import path
# before importing the detectors module.
# ---------------------------------------------------------------------------

@dataclass
class RefinementStep:
    """Minimal stub matching the positional signature used across the codebase."""
    iteration: int
    query: str
    response: str
    confidence: float
    improvement: float
    strategy_used: str
    reasoning: str
    timestamp: object = None  # datetime, but optional for tests


class _ConvergenceDetectorProtocol:
    """Stub protocol — unused by tests but required by the import."""
    pass


# Patch the missing names into the protocols module so detectors.py can import.
# Try loading the real module first to get all names; only create a minimal shim
# if the real module fails to import (missing deps like fabric.spacetime).
_real_mod = sys.modules.get("hololoom.protocols.recursive_reasoning") or \
            sys.modules.get("hololoom.core.protocols.recursive_reasoning")

if _real_mod is None:
    try:
        import hololoom.core.protocols.recursive_reasoning as _real_mod  # noqa: F811
    except (ImportError, Exception):
        _real_mod = None

if _real_mod is not None:
    if not hasattr(_real_mod, "RefinementStep"):
        _real_mod.RefinementStep = RefinementStep
    if not hasattr(_real_mod, "ConvergenceDetectorProtocol"):
        _real_mod.ConvergenceDetectorProtocol = _ConvergenceDetectorProtocol
    # Ensure both module paths point to the same module
    sys.modules.setdefault("hololoom.protocols.recursive_reasoning", _real_mod)
    sys.modules.setdefault("hololoom.core.protocols.recursive_reasoning", _real_mod)
else:
    # Real module failed to import — create a minimal shim
    _shim = types.ModuleType("hololoom.protocols.recursive_reasoning")
    _shim.RefinementStep = RefinementStep
    _shim.ConvergenceDetectorProtocol = _ConvergenceDetectorProtocol
    sys.modules["hololoom.protocols.recursive_reasoning"] = _shim
    sys.modules["hololoom.core.protocols.recursive_reasoning"] = _shim

# Re-import shimmed types from the protocol module so this test uses the same
# class objects as the code under test (avoids identity mismatches when another
# test file loaded its own shim first).
import hololoom.core.protocols.recursive_reasoning as _proto_mod2  # noqa: E402

RefinementStep = _proto_mod2.RefinementStep  # noqa: F811

# Now safe to import detectors
from hololoom.core.convergence.detectors import (  # noqa: E402
    ConvergenceDetector,
    ConfidenceThresholdDetector,
    ImprovementDeltaDetector,
    QualityPlateauDetector,
    MaxIterationsDetector,
    HighVarianceDetector,
    MultiCriteriaDetector,
    SemanticConvergenceDetector,
    create_confidence_detector,
    create_improvement_detector,
    create_plateau_detector,
    create_semantic_detector,
    create_multi_criteria_detector,
)


# ============================================================================
# Helpers
# ============================================================================

def _step(confidence: float, improvement: float = 0.0, iteration: int = 1) -> RefinementStep:
    """Convenience builder for a single RefinementStep."""
    return RefinementStep(
        iteration=iteration,
        query="q",
        response="r",
        confidence=confidence,
        improvement=improvement,
        strategy_used="test",
        reasoning="test step",
        timestamp=datetime.utcnow(),
    )


def _history(pairs: List[Tuple[float, float]]) -> List[RefinementStep]:
    """Build a history from (confidence, improvement) pairs."""
    return [_step(c, imp, i + 1) for i, (c, imp) in enumerate(pairs)]


# ============================================================================
# ConvergenceDetector (base class)
# ============================================================================

class TestConvergenceDetectorBase:
    def test_detect_raises_not_implemented(self):
        detector = ConvergenceDetector()
        with pytest.raises(NotImplementedError):
            detector.detect([])

    def test_get_confidences(self):
        detector = ConvergenceDetector()
        history = _history([(0.5, 0.0), (0.7, 0.2)])
        assert detector._get_confidences(history) == [0.5, 0.7]

    def test_get_improvements(self):
        detector = ConvergenceDetector()
        history = _history([(0.5, 0.0), (0.7, 0.2)])
        assert detector._get_improvements(history) == [0.0, 0.2]

    def test_get_confidences_empty(self):
        detector = ConvergenceDetector()
        assert detector._get_confidences([]) == []

    def test_get_improvements_empty(self):
        detector = ConvergenceDetector()
        assert detector._get_improvements([]) == []


# ============================================================================
# ConfidenceThresholdDetector
# ============================================================================

class TestConfidenceThresholdDetector:
    def test_empty_history_returns_false(self):
        d = ConfidenceThresholdDetector(threshold=0.85)
        converged, reason = d.detect([])
        assert converged is False
        assert "No history" in reason

    def test_below_threshold(self):
        d = ConfidenceThresholdDetector(threshold=0.85)
        converged, reason = d.detect([_step(0.70)])
        assert converged is False
        assert "0.70" in reason

    def test_at_threshold(self):
        d = ConfidenceThresholdDetector(threshold=0.85)
        converged, reason = d.detect([_step(0.85)])
        assert converged is True

    def test_above_threshold(self):
        d = ConfidenceThresholdDetector(threshold=0.85)
        converged, reason = d.detect([_step(0.95)])
        assert converged is True

    def test_only_latest_step_matters(self):
        """Early high confidence followed by low should NOT converge."""
        d = ConfidenceThresholdDetector(threshold=0.85)
        history = _history([(0.90, 0.0), (0.60, -0.30)])
        converged, _ = d.detect(history)
        assert converged is False

    def test_custom_threshold(self):
        d = ConfidenceThresholdDetector(threshold=0.5)
        converged, _ = d.detect([_step(0.51)])
        assert converged is True

    def test_threshold_zero(self):
        d = ConfidenceThresholdDetector(threshold=0.0)
        converged, _ = d.detect([_step(0.0)])
        assert converged is True

    def test_threshold_one(self):
        d = ConfidenceThresholdDetector(threshold=1.0)
        converged, _ = d.detect([_step(0.99)])
        assert converged is False


# ============================================================================
# ImprovementDeltaDetector
# ============================================================================

class TestImprovementDeltaDetector:
    def test_insufficient_history(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        converged, reason = d.detect([_step(0.5)])
        assert converged is False
        assert "Insufficient" in reason

    def test_empty_history(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        converged, reason = d.detect([])
        assert converged is False

    def test_improvement_above_delta(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.6, 0.10)])
        converged, _ = d.detect(history)
        assert converged is False

    def test_improvement_below_delta(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.52, 0.02)])
        converged, _ = d.detect(history)
        assert converged is True

    def test_zero_improvement(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.5, 0.0)])
        converged, _ = d.detect(history)
        assert converged is True

    def test_negative_improvement_below_delta(self):
        """Negative improvement with small absolute value should trigger."""
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.48, -0.02)])
        converged, _ = d.detect(history)
        assert converged is True

    def test_negative_improvement_above_delta(self):
        """Large negative improvement should NOT trigger (abs > delta)."""
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.3, -0.20)])
        converged, _ = d.detect(history)
        assert converged is False

    def test_only_latest_improvement_matters(self):
        d = ImprovementDeltaDetector(min_delta=0.05)
        history = _history([(0.5, 0.0), (0.6, 0.10), (0.61, 0.01)])
        converged, _ = d.detect(history)
        assert converged is True


# ============================================================================
# QualityPlateauDetector
# ============================================================================

class TestQualityPlateauDetector:
    def test_insufficient_history(self):
        d = QualityPlateauDetector(plateau_window=3, min_delta=0.02)
        history = _history([(0.5, 0.0), (0.6, 0.1)])
        converged, reason = d.detect(history)
        assert converged is False
        assert "Insufficient" in reason

    def test_no_plateau(self):
        d = QualityPlateauDetector(plateau_window=3, min_delta=0.02)
        history = _history([
            (0.5, 0.0),
            (0.6, 0.10),
            (0.7, 0.10),
        ])
        converged, _ = d.detect(history)
        assert converged is False

    def test_plateau_detected(self):
        d = QualityPlateauDetector(plateau_window=3, min_delta=0.02)
        history = _history([
            (0.80, 0.01),
            (0.81, 0.01),
            (0.81, 0.00),
        ])
        converged, reason = d.detect(history)
        assert converged is True
        assert "plateaued" in reason

    def test_partial_plateau_not_triggered(self):
        """Only 2 of 3 steps below threshold — not a plateau."""
        d = QualityPlateauDetector(plateau_window=3, min_delta=0.02)
        history = _history([
            (0.70, 0.01),
            (0.75, 0.05),
            (0.76, 0.01),
        ])
        converged, _ = d.detect(history)
        assert converged is False

    def test_plateau_window_one(self):
        d = QualityPlateauDetector(plateau_window=1, min_delta=0.02)
        history = _history([(0.80, 0.01)])
        converged, _ = d.detect(history)
        assert converged is True

    def test_long_history_only_checks_window(self):
        """Earlier large improvements shouldn't prevent plateau detection."""
        d = QualityPlateauDetector(plateau_window=3, min_delta=0.02)
        history = _history([
            (0.50, 0.0),
            (0.60, 0.10),
            (0.70, 0.10),
            (0.71, 0.01),
            (0.71, 0.00),
            (0.72, 0.01),
        ])
        converged, _ = d.detect(history)
        assert converged is True


# ============================================================================
# MaxIterationsDetector
# ============================================================================

class TestMaxIterationsDetector:
    def test_below_max(self):
        d = MaxIterationsDetector(max_iterations=5)
        history = _history([(0.5, 0.0)] * 3)
        converged, reason = d.detect(history)
        assert converged is False
        assert "3/5" in reason

    def test_at_max(self):
        d = MaxIterationsDetector(max_iterations=5)
        history = _history([(0.5, 0.0)] * 5)
        converged, reason = d.detect(history)
        assert converged is True
        assert "5" in reason

    def test_above_max(self):
        d = MaxIterationsDetector(max_iterations=5)
        history = _history([(0.5, 0.0)] * 7)
        converged, _ = d.detect(history)
        assert converged is True

    def test_empty_history(self):
        d = MaxIterationsDetector(max_iterations=5)
        converged, _ = d.detect([])
        assert converged is False

    def test_max_one(self):
        d = MaxIterationsDetector(max_iterations=1)
        converged, _ = d.detect([_step(0.5)])
        assert converged is True


# ============================================================================
# HighVarianceDetector
# ============================================================================

class TestHighVarianceDetector:
    def test_insufficient_history(self):
        d = HighVarianceDetector(variance_threshold=0.15, window=3)
        history = _history([(0.5, 0.0), (0.6, 0.1)])
        converged, reason = d.detect(history)
        assert converged is False
        assert "Insufficient" in reason

    def test_low_variance(self):
        d = HighVarianceDetector(variance_threshold=0.15, window=3)
        history = _history([
            (0.80, 0.0),
            (0.81, 0.01),
            (0.82, 0.01),
        ])
        converged, reason = d.detect(history)
        assert converged is False
        assert "Variance" in reason

    def test_high_variance(self):
        d = HighVarianceDetector(variance_threshold=0.01, window=3)
        history = _history([
            (0.30, 0.0),
            (0.90, 0.60),
            (0.40, -0.50),
        ])
        converged, reason = d.detect(history)
        assert converged is True
        assert "unstable" in reason

    def test_constant_confidence_zero_variance(self):
        d = HighVarianceDetector(variance_threshold=0.01, window=3)
        history = _history([(0.75, 0.0)] * 3)
        converged, _ = d.detect(history)
        assert converged is False

    def test_only_checks_window(self):
        """Old volatile data outside the window shouldn't matter."""
        d = HighVarianceDetector(variance_threshold=0.01, window=3)
        history = _history([
            (0.10, 0.0),   # old, outside window
            (0.90, 0.80),  # old, outside window
            (0.80, 0.0),
            (0.81, 0.01),
            (0.82, 0.01),
        ])
        converged, _ = d.detect(history)
        assert converged is False

    def test_empty_history(self):
        d = HighVarianceDetector(variance_threshold=0.15, window=3)
        converged, _ = d.detect([])
        assert converged is False


# ============================================================================
# MultiCriteriaDetector
# ============================================================================

class TestMultiCriteriaDetector:
    def _make_always_true(self):
        d = ConfidenceThresholdDetector(threshold=0.0)
        return d

    def _make_always_false(self):
        d = MaxIterationsDetector(max_iterations=9999)
        return d

    # --- OR logic ---

    def test_or_any_true(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true(), self._make_always_false()],
            logic="OR",
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is True

    def test_or_none_true(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_false(), self._make_always_false()],
            logic="OR",
        )
        converged, reason = mc.detect([_step(0.5)])
        assert converged is False
        assert "No convergence" in reason

    # --- AND logic ---

    def test_and_all_true(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true(), self._make_always_true()],
            logic="AND",
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is True

    def test_and_not_all_true(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true(), self._make_always_false()],
            logic="AND",
        )
        converged, reason = mc.detect([_step(0.5)])
        assert converged is False
        assert "Not all" in reason

    # --- MAJORITY logic ---

    def test_majority_above_half(self):
        mc = MultiCriteriaDetector(
            detectors=[
                self._make_always_true(),
                self._make_always_true(),
                self._make_always_false(),
            ],
            logic="MAJORITY",
        )
        converged, reason = mc.detect([_step(0.5)])
        assert converged is True
        assert "Weighted vote" in reason

    def test_majority_below_half(self):
        mc = MultiCriteriaDetector(
            detectors=[
                self._make_always_true(),
                self._make_always_false(),
                self._make_always_false(),
            ],
            logic="MAJORITY",
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is False

    def test_majority_exactly_half_not_converged(self):
        """Exactly 50% should NOT converge (needs strictly more than half)."""
        mc = MultiCriteriaDetector(
            detectors=[
                self._make_always_true(),
                self._make_always_false(),
            ],
            logic="MAJORITY",
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is False

    def test_majority_with_weights(self):
        """Heavy weight on the true detector tips the vote."""
        mc = MultiCriteriaDetector(
            detectors=[
                self._make_always_true(),
                self._make_always_false(),
            ],
            logic="MAJORITY",
            weights=[3.0, 1.0],
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is True

    def test_majority_weights_against(self):
        """Heavy weight on the false detector blocks convergence."""
        mc = MultiCriteriaDetector(
            detectors=[
                self._make_always_true(),
                self._make_always_false(),
            ],
            logic="MAJORITY",
            weights=[1.0, 3.0],
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is False

    # --- Edge cases ---

    def test_unknown_logic_raises(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true()],
            logic="INVALID",
        )
        with pytest.raises(ValueError, match="Unknown logic"):
            mc.detect([_step(0.5)])

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Weights must match"):
            MultiCriteriaDetector(
                detectors=[self._make_always_true()],
                weights=[1.0, 2.0],
            )

    def test_case_insensitive_logic(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true()],
            logic="or",
        )
        converged, _ = mc.detect([_step(0.5)])
        assert converged is True

    def test_default_weights_are_uniform(self):
        mc = MultiCriteriaDetector(
            detectors=[self._make_always_true(), self._make_always_false()],
            logic="MAJORITY",
        )
        assert mc.weights == [1.0, 1.0]

    def test_reason_includes_detector_names(self):
        mc = MultiCriteriaDetector(
            detectors=[ConfidenceThresholdDetector(0.5)],
            logic="OR",
        )
        _, reason = mc.detect([_step(0.6)])
        assert "ConfidenceThresholdDetector" in reason


# ============================================================================
# Factory functions
# ============================================================================

class TestFactoryFunctions:
    def test_create_confidence_detector(self):
        d = create_confidence_detector(threshold=0.9)
        assert isinstance(d, ConfidenceThresholdDetector)
        assert d.threshold == 0.9

    def test_create_confidence_detector_defaults(self):
        d = create_confidence_detector()
        assert d.threshold == 0.85

    def test_create_improvement_detector(self):
        d = create_improvement_detector(min_delta=0.1)
        assert isinstance(d, ImprovementDeltaDetector)
        assert d.min_delta == 0.1

    def test_create_improvement_detector_defaults(self):
        d = create_improvement_detector()
        assert d.min_delta == 0.05

    def test_create_plateau_detector(self):
        d = create_plateau_detector(plateau_window=5, min_delta=0.03)
        assert isinstance(d, QualityPlateauDetector)
        assert d.plateau_window == 5
        assert d.min_delta == 0.03

    def test_create_plateau_detector_defaults(self):
        d = create_plateau_detector()
        assert d.plateau_window == 3
        assert d.min_delta == 0.02

    def test_create_multi_criteria_detector(self):
        mc = create_multi_criteria_detector(
            confidence_threshold=0.9,
            min_improvement=0.1,
            plateau_window=4,
            max_iterations=10,
            logic="AND",
        )
        assert isinstance(mc, MultiCriteriaDetector)
        assert mc.logic == "AND"
        assert len(mc.detectors) == 4

    def test_create_multi_criteria_detector_defaults(self):
        mc = create_multi_criteria_detector()
        assert mc.logic == "OR"
        assert len(mc.detectors) == 4

    def test_create_multi_criteria_detector_functional(self):
        """End-to-end: the factory-built detector actually works."""
        mc = create_multi_criteria_detector(
            confidence_threshold=0.85,
            max_iterations=3,
            logic="OR",
        )
        history = _history([
            (0.50, 0.0),
            (0.60, 0.10),
            (0.70, 0.10),
        ])
        converged, _ = mc.detect(history)
        # max_iterations=3 should trigger
        assert converged is True


# ============================================================================
# SemanticConvergenceDetector
# ============================================================================

class TestSemanticConvergenceDetector:
    def _step_with_response(
        self, response: str, confidence: float = 0.7, improvement: float = 0.0, iteration: int = 1
    ) -> RefinementStep:
        return RefinementStep(
            iteration=iteration,
            query="q",
            response=response,
            confidence=confidence,
            improvement=improvement,
            strategy_used="test",
            reasoning="test step",
            timestamp=datetime.utcnow(),
        )

    def test_insufficient_history(self):
        d = SemanticConvergenceDetector(threshold=0.05, min_steps=2)
        converged, reason = d.detect([self._step_with_response("hello")])
        assert converged is False
        assert "Insufficient" in reason

    def test_empty_history(self):
        d = SemanticConvergenceDetector(threshold=0.05)
        converged, _ = d.detect([])
        assert converged is False

    def test_identical_responses_converge(self):
        """Identical responses should have JSD=0, triggering convergence."""
        d = SemanticConvergenceDetector(threshold=0.05)
        history = [
            self._step_with_response("The answer is 42.", iteration=1),
            self._step_with_response("The answer is 42.", iteration=2),
        ]
        converged, reason = d.detect(history)
        assert converged is True
        assert "JSD=0.0000" in reason

    def test_very_different_responses_do_not_converge(self):
        """Completely different responses should have high JSD."""
        d = SemanticConvergenceDetector(threshold=0.05)
        history = [
            self._step_with_response("The quick brown fox jumps over the lazy dog", iteration=1),
            self._step_with_response("12345 67890 !@#$% ^&*() QWERTY ZXCVBN", iteration=2),
        ]
        converged, _ = d.detect(history)
        assert converged is False

    def test_similar_responses_converge(self):
        """Slightly rephrased responses should have low JSD."""
        d = SemanticConvergenceDetector(threshold=0.3)
        history = [
            self._step_with_response(
                "Thompson Sampling uses Beta distributions to balance exploration and exploitation.",
                iteration=1,
            ),
            self._step_with_response(
                "Thompson Sampling uses Beta distributions for balancing exploration and exploitation.",
                iteration=2,
            ),
        ]
        converged, _ = d.detect(history)
        assert converged is True

    def test_custom_embedding_fn(self):
        """Test with a custom embedding function."""
        def dummy_embedding(text):
            # Hash-based deterministic embedding for testing
            h = hash(text) % 1000
            return np.array([h, h + 1, h + 2], dtype=np.float64)

        d = SemanticConvergenceDetector(threshold=0.05, embedding_fn=dummy_embedding)
        history = [
            self._step_with_response("response A", iteration=1),
            self._step_with_response("response A", iteration=2),
        ]
        converged, _ = d.detect(history)
        assert converged is True

    def test_custom_embedding_fn_different(self):
        """Different embeddings should not converge."""
        def spread_embedding(text):
            if "alpha" in text:
                return np.array([10.0, 0.0, 0.0])
            return np.array([0.0, 0.0, 10.0])

        d = SemanticConvergenceDetector(threshold=0.05, embedding_fn=spread_embedding)
        history = [
            self._step_with_response("alpha response", iteration=1),
            self._step_with_response("beta response", iteration=2),
        ]
        converged, _ = d.detect(history)
        assert converged is False

    def test_only_compares_last_two(self):
        """Should only compare the last two steps, not earlier history."""
        d = SemanticConvergenceDetector(threshold=0.05)
        history = [
            self._step_with_response("completely different text here", iteration=1),
            self._step_with_response("another unique response", iteration=2),
            self._step_with_response("the final answer is X", iteration=3),
            self._step_with_response("the final answer is X", iteration=4),
        ]
        converged, _ = d.detect(history)
        assert converged is True

    def test_empty_response_handled(self):
        d = SemanticConvergenceDetector(threshold=0.05)
        history = [
            self._step_with_response("", iteration=1),
            self._step_with_response("some text", iteration=2),
        ]
        converged, reason = d.detect(history)
        assert converged is False
        assert "Empty" in reason

    def test_jsd_bounded_zero_one(self):
        """JSD should always be in [0, 1] for base-2."""
        d = SemanticConvergenceDetector(threshold=2.0)  # high threshold, won't trigger
        for _ in range(10):
            text1 = f"random text {np.random.randint(0, 1000)}"
            text2 = f"other text {np.random.randint(0, 1000)}"
            history = [
                self._step_with_response(text1, iteration=1),
                self._step_with_response(text2, iteration=2),
            ]
            _, reason = d.detect(history)
            # Extract JSD value from reason string
            jsd_str = reason.split("JSD=")[1].split(" ")[0]
            jsd = float(jsd_str)
            assert 0.0 <= jsd <= 1.0, f"JSD={jsd} out of bounds"


class TestSemanticDetectorFactory:
    def test_create_semantic_detector(self):
        d = create_semantic_detector(threshold=0.1, min_steps=3)
        assert isinstance(d, SemanticConvergenceDetector)
        assert d.threshold == 0.1
        assert d.min_steps == 3

    def test_create_semantic_detector_defaults(self):
        d = create_semantic_detector()
        assert d.threshold == 0.05
        assert d.min_steps == 2

    def test_multi_criteria_with_semantic(self):
        """Semantic detector can be added to multi-criteria via factory."""
        mc = create_multi_criteria_detector(semantic_threshold=0.1)
        assert len(mc.detectors) == 5  # 4 standard + 1 semantic
        assert any(isinstance(d, SemanticConvergenceDetector) for d in mc.detectors)

    def test_multi_criteria_without_semantic(self):
        """Without semantic_threshold, no semantic detector added."""
        mc = create_multi_criteria_detector()
        assert len(mc.detectors) == 4
        assert not any(isinstance(d, SemanticConvergenceDetector) for d in mc.detectors)


# ============================================================================
# Integration scenarios
# ============================================================================

class TestIntegrationScenarios:
    def test_progressive_convergence(self):
        """Simulate a realistic refinement session that converges."""
        d = create_multi_criteria_detector(
            confidence_threshold=0.85,
            min_improvement=0.05,
            plateau_window=3,
            max_iterations=10,
            logic="OR",
        )
        history = _history([
            (0.65, 0.00),
            (0.75, 0.10),
            (0.82, 0.07),
            (0.86, 0.04),  # confidence >= 0.85 AND improvement < 0.05
        ])
        converged, reason = d.detect(history)
        assert converged is True

    def test_stalled_refinement(self):
        """Refinement that plateaus without reaching confidence threshold."""
        d = create_multi_criteria_detector(
            confidence_threshold=0.90,
            min_improvement=0.05,
            plateau_window=3,
            max_iterations=10,
            logic="OR",
        )
        history = _history([
            (0.60, 0.00),
            (0.65, 0.05),
            (0.66, 0.01),
            (0.66, 0.00),
            (0.67, 0.01),
        ])
        converged, _ = d.detect(history)
        # Should converge via plateau (last 3 improvements < 0.02)
        assert converged is True

    def test_unstable_refinement(self):
        """Refinement that oscillates wildly."""
        d = HighVarianceDetector(variance_threshold=0.05, window=3)
        history = _history([
            (0.30, 0.00),
            (0.90, 0.60),
            (0.40, -0.50),
        ])
        converged, _ = d.detect(history)
        assert converged is True
