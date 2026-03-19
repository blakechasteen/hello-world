"""
Tests for drift_detector.py — Information-theoretic hallucination detection.

Wire 2 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.drift_detector import (
    DriftDetector,
    DriftResult,
    DriftDetectorProtocol,
    create_drift_detector,
)


# ============================================================================
# DriftResult
# ============================================================================

class TestDriftResult:
    def test_fields(self):
        r = DriftResult(divergence=0.3, drifted=True, threshold=0.2, context_size=5, detail="test")
        assert r.divergence == 0.3
        assert r.drifted is True
        assert r.context_size == 5


# ============================================================================
# DriftDetector basics
# ============================================================================

class TestDriftDetectorBasics:
    def test_protocol_compliance(self):
        d = DriftDetector(threshold=0.3)
        assert isinstance(d, DriftDetectorProtocol)

    def test_empty_context_returns_no_drift(self):
        d = DriftDetector(threshold=0.3)
        result = d.check(np.array([1.0, 0.0, 0.0]), [])
        assert result.drifted is False
        assert result.context_size == 0
        assert "No context" in result.detail

    def test_dimension_mismatch_returns_no_drift(self):
        d = DriftDetector(threshold=0.3)
        response = np.array([1.0, 2.0, 3.0])
        context = [np.array([1.0, 2.0])]  # Different dimension
        result = d.check(response, context)
        assert result.drifted is False
        assert "mismatch" in result.detail


# ============================================================================
# DriftDetector — identical / similar / different embeddings
# ============================================================================

class TestDriftDetectorDivergence:
    def test_identical_embeddings_no_drift(self):
        d = DriftDetector(threshold=0.1)
        emb = np.array([1.0, 2.0, 3.0, 4.0])
        result = d.check(emb, [emb.copy()])
        assert result.drifted is False
        assert result.divergence == pytest.approx(0.0, abs=1e-9)

    def test_similar_embeddings_low_divergence(self):
        d = DriftDetector(threshold=0.5)
        response = np.array([1.0, 2.0, 3.0, 4.0])
        context = [np.array([1.1, 2.1, 2.9, 4.1])]
        result = d.check(response, context)
        assert result.drifted is False
        assert result.divergence < 0.5

    def test_opposite_embeddings_high_divergence(self):
        d = DriftDetector(threshold=0.1)
        response = np.array([10.0, 0.0, 0.0, 0.0])
        context = [np.array([0.0, 0.0, 0.0, 10.0])]
        result = d.check(response, context)
        assert result.drifted is True
        assert result.divergence > 0.1

    def test_multiple_context_embeddings_centroid(self):
        """Centroid of context should be used, not individual embeddings."""
        d = DriftDetector(threshold=0.3)
        response = np.array([2.0, 2.0, 2.0, 2.0])
        context = [
            np.array([1.0, 3.0, 1.0, 3.0]),
            np.array([3.0, 1.0, 3.0, 1.0]),
        ]
        # Centroid = [2, 2, 2, 2] — should be very close to response
        result = d.check(response, context)
        assert result.drifted is False
        assert result.divergence < 0.1

    def test_jsd_bounded_zero_one(self):
        """JSD should always be in [0, 1]."""
        d = DriftDetector(threshold=2.0)
        for _ in range(20):
            response = np.random.randn(16)
            context = [np.random.randn(16) for _ in range(3)]
            result = d.check(response, context)
            assert 0.0 <= result.divergence <= 1.0 + 1e-9


# ============================================================================
# Threshold behavior
# ============================================================================

class TestDriftThreshold:
    def test_strict_threshold_flags_more(self):
        response = np.array([5.0, 0.0, 0.0])
        context = [np.array([3.0, 1.0, 1.0])]

        strict = DriftDetector(threshold=0.01)
        permissive = DriftDetector(threshold=0.99)

        r_strict = strict.check(response, context)
        r_permissive = permissive.check(response, context)

        # Same divergence, different drift flags
        assert r_strict.divergence == pytest.approx(r_permissive.divergence)
        assert r_strict.drifted is True
        assert r_permissive.drifted is False

    def test_zero_threshold_always_drifts(self):
        """Threshold=0 means any non-zero divergence flags drift."""
        d = DriftDetector(threshold=0.0)
        response = np.array([1.0, 2.0])
        context = [np.array([1.0, 2.1])]
        result = d.check(response, context)
        assert result.drifted is True


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create_drift_detector(self):
        d = create_drift_detector(threshold=0.5)
        assert isinstance(d, DriftDetector)
        assert d.threshold == 0.5

    def test_create_drift_detector_default(self):
        d = create_drift_detector()
        assert d.threshold == 0.3
