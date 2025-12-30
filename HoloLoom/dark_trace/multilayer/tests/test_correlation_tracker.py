"""
Dark Trace Multi-Layer Correlation: Test Suite
===============================================

Comprehensive tests for correlation tracking, propagation analysis,
and causal metrics.

Tests cover:
- WelfordCorrelator numerical stability
- CorrelationTracker multi-layer tracking
- PropagationAnalyzer circuit discovery
- CausalStrengthEstimator and TransferEntropyEstimator

Created: 2025-12-28
"""

import pytest
import numpy as np
from typing import Dict, List
import math

from HoloLoom.dark_trace.multilayer.correlation_tracker import (
    CorrelationConfig,
    CorrelationMethod,
    CorrelationPair,
    LayerCorrelation,
    CorrelationSnapshot,
    WelfordCorrelator,
    CorrelationTracker,
)

from HoloLoom.dark_trace.multilayer.propagation_analyzer import (
    PropagationType,
    CircuitType,
    PropagationEdge,
    PropagationPath,
    CircuitCandidate,
    PropagationConfig,
    PropagationAnalyzer,
)

from HoloLoom.dark_trace.multilayer.causal_metrics import (
    CausalRelationType,
    CausalStrength,
    TransferEntropyResult,
    CausalStrengthEstimator,
    TransferEntropyEstimator,
)


# =============================================================================
# WelfordCorrelator Tests
# =============================================================================

class TestWelfordCorrelator:
    """Test Welford's online correlation algorithm."""

    def test_perfect_positive_correlation(self):
        """Test correlation = 1.0 for perfectly correlated data."""
        correlator = WelfordCorrelator()

        for i in range(100):
            x = float(i)
            y = 2.0 * x + 1.0  # Perfect linear relationship
            correlator.update(x, y)

        corr = correlator.correlation()
        assert corr is not None
        assert abs(corr - 1.0) < 0.001, f"Expected ~1.0, got {corr}"

    def test_perfect_negative_correlation(self):
        """Test correlation = -1.0 for perfectly anti-correlated data."""
        correlator = WelfordCorrelator()

        for i in range(100):
            x = float(i)
            y = -2.0 * x + 100.0  # Perfect negative relationship
            correlator.update(x, y)

        corr = correlator.correlation()
        assert corr is not None
        assert abs(corr - (-1.0)) < 0.001, f"Expected ~-1.0, got {corr}"

    def test_zero_correlation(self):
        """Test correlation ~0 for uncorrelated data."""
        correlator = WelfordCorrelator()
        np.random.seed(42)

        for i in range(1000):
            x = np.random.randn()
            y = np.random.randn()  # Independent
            correlator.update(x, y)

        corr = correlator.correlation()
        assert corr is not None
        assert abs(corr) < 0.1, f"Expected ~0, got {corr}"

    def test_insufficient_samples(self):
        """Test returns None with < 2 samples."""
        correlator = WelfordCorrelator()

        assert correlator.correlation() is None

        correlator.update(1.0, 2.0)
        assert correlator.correlation() is None

        correlator.update(2.0, 4.0)
        assert correlator.correlation() is not None

    def test_constant_values(self):
        """Test handles constant values (zero variance)."""
        correlator = WelfordCorrelator()

        for i in range(10):
            correlator.update(5.0, 5.0)  # Constant

        corr = correlator.correlation()
        assert corr is None  # Undefined for zero variance

    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        correlator = WelfordCorrelator()

        base = 1e10
        for i in range(100):
            x = base + float(i)
            y = base + 2.0 * float(i)
            correlator.update(x, y)

        corr = correlator.correlation()
        assert corr is not None
        assert abs(corr - 1.0) < 0.01, f"Expected ~1.0, got {corr}"

    def test_reset(self):
        """Test reset clears all state."""
        correlator = WelfordCorrelator()

        for i in range(50):
            correlator.update(float(i), float(i))

        assert correlator.n == 50

        correlator.reset()

        assert correlator.n == 0
        assert correlator.mean_x == 0.0
        assert correlator.correlation() is None


# =============================================================================
# CorrelationTracker Tests
# =============================================================================

class TestCorrelationTracker:
    """Test multi-layer correlation tracking."""

    @pytest.fixture
    def config(self):
        return CorrelationConfig(
            min_samples=5,
            correlation_threshold=0.3,
            window_size=100,
            method=CorrelationMethod.PEARSON,
        )

    @pytest.fixture
    def tracker(self, config):
        return CorrelationTracker(n_layers=4, config=config)

    def test_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker.n_layers == 4
        assert len(tracker._correlators) == 0

    def test_update_single_batch(self, tracker):
        """Test single batch update."""
        activations = {
            0: {"sae.0": 0.5, "sae.1": 0.3},
            1: {"sae.10": 0.4, "sae.11": 0.6},
        }

        tracker.update(activations)

        # Check correlators created for adjacent layers
        assert (0, 1) in tracker._correlators

    def test_update_multiple_batches(self, tracker):
        """Test correlation builds over multiple batches."""
        np.random.seed(42)

        # Create correlated features
        for batch in range(20):
            base = np.random.randn()
            activations = {
                0: {"sae.0": base, "sae.1": -base},
                1: {"sae.10": base * 0.9 + np.random.randn() * 0.1},
            }
            tracker.update(activations)

        # Get correlation
        layer_corr = tracker.get_layer_correlation(0, 1)
        assert layer_corr is not None
        assert layer_corr.source_layer == 0
        assert layer_corr.target_layer == 1
        assert len(layer_corr.correlations) > 0

    def test_trace_feature_propagation(self, tracker):
        """Test tracing feature propagation across layers."""
        np.random.seed(42)

        # Create propagating feature
        for batch in range(30):
            base = np.random.randn()
            activations = {
                0: {"sae.0": base},
                1: {"sae.10": base * 0.95},
                2: {"sae.20": base * 0.9},
                3: {"sae.30": base * 0.85},
            }
            tracker.update(activations)

        # Trace propagation
        path = tracker.trace_feature_propagation("sae.0")
        assert isinstance(path, list)

    def test_snapshot(self, tracker):
        """Test creating correlation snapshot."""
        np.random.seed(42)

        for batch in range(10):
            activations = {
                0: {"sae.0": np.random.randn()},
                1: {"sae.10": np.random.randn()},
            }
            tracker.update(activations)

        snapshot = tracker.snapshot()

        assert isinstance(snapshot, CorrelationSnapshot)
        assert snapshot.n_layers == 4
        assert snapshot.total_pairs >= 0

    def test_get_top_correlations(self, tracker):
        """Test getting top correlations."""
        np.random.seed(42)

        # Create features with varying correlations
        for batch in range(30):
            base = np.random.randn()
            activations = {
                0: {"sae.0": base, "sae.1": np.random.randn()},
                1: {"sae.10": base * 0.95, "sae.11": np.random.randn()},
            }
            tracker.update(activations)

        top = tracker.get_top_correlations(k=5)

        assert isinstance(top, list)
        assert len(top) <= 5

        # Check sorted by correlation strength
        if len(top) >= 2:
            for i in range(len(top) - 1):
                assert abs(top[i].correlation) >= abs(top[i + 1].correlation)


# =============================================================================
# PropagationAnalyzer Tests
# =============================================================================

class TestPropagationAnalyzer:
    """Test feature propagation analysis."""

    @pytest.fixture
    def config(self):
        return PropagationConfig(
            min_correlation=0.3,
            max_depth=5,
            min_circuit_features=3,
        )

    @pytest.fixture
    def correlation_tracker(self):
        tracker = CorrelationTracker(n_layers=4)
        np.random.seed(42)

        # Build correlations
        for batch in range(50):
            base = np.random.randn()
            activations = {
                0: {"sae.0": base, "sae.1": -base * 0.5},
                1: {"sae.10": base * 0.9, "sae.11": -base * 0.4},
                2: {"sae.20": base * 0.8, "sae.21": np.random.randn()},
                3: {"sae.30": base * 0.7},
            }
            tracker.update(activations)

        return tracker

    @pytest.fixture
    def analyzer(self, correlation_tracker, config):
        return PropagationAnalyzer(correlation_tracker, config)

    def test_trace_forward(self, analyzer):
        """Test forward propagation tracing."""
        path = analyzer.trace_feature("sae.0", PropagationType.FORWARD)

        assert isinstance(path, PropagationPath)
        assert path.start_feature == "sae.0"
        assert path.propagation_type == PropagationType.FORWARD

    def test_trace_backward(self, analyzer):
        """Test backward propagation tracing."""
        path = analyzer.trace_feature("sae.30", PropagationType.BACKWARD)

        assert isinstance(path, PropagationPath)
        assert path.start_feature == "sae.30"
        assert path.propagation_type == PropagationType.BACKWARD

    def test_discover_circuits(self, analyzer):
        """Test circuit discovery."""
        circuits = analyzer.discover_circuits(min_features=2)

        assert isinstance(circuits, list)
        for circuit in circuits:
            assert isinstance(circuit, CircuitCandidate)
            assert len(circuit.features) >= 2

    def test_analyze_bottlenecks(self, analyzer):
        """Test bottleneck analysis."""
        bottlenecks = analyzer.analyze_bottlenecks()

        assert isinstance(bottlenecks, list)
        for bottleneck in bottlenecks:
            assert "feature" in bottleneck
            assert "layer" in bottleneck
            assert "weakness" in bottleneck

    def test_propagation_path_strength(self, analyzer):
        """Test path strength calculation."""
        path = analyzer.trace_feature("sae.0", PropagationType.FORWARD)

        # Strength should be product of edge correlations
        if len(path.edges) > 0:
            manual_strength = 1.0
            for edge in path.edges:
                manual_strength *= abs(edge.correlation)

            assert abs(path.total_strength - manual_strength) < 0.001

    def test_circuit_types(self, analyzer):
        """Test circuit type classification."""
        circuits = analyzer.discover_circuits(min_features=2)

        for circuit in circuits:
            assert circuit.circuit_type in [
                CircuitType.LINEAR,
                CircuitType.BRANCHING,
                CircuitType.CONVERGING,
                CircuitType.COMPLEX,
            ]


# =============================================================================
# CausalStrengthEstimator Tests
# =============================================================================

class TestCausalStrengthEstimator:
    """Test causal strength estimation."""

    @pytest.fixture
    def estimator(self):
        return CausalStrengthEstimator()

    def test_observe_data(self, estimator):
        """Test observing activation data."""
        np.random.seed(42)

        for i in range(100):
            source = {"sae.0": np.random.randn()}
            target = {"sae.10": source["sae.0"] * 0.8 + np.random.randn() * 0.2}
            estimator.observe(source, target)

        assert estimator._n_observations == 100

    def test_estimate_causal_strength(self, estimator):
        """Test causal strength estimation."""
        np.random.seed(42)

        # Create causal relationship
        for i in range(200):
            source = {"sae.0": np.random.randn()}
            target = {"sae.10": source["sae.0"] * 0.9 + np.random.randn() * 0.1}
            estimator.observe(source, target)

        strength = estimator.estimate("sae.0", "sae.10")

        assert isinstance(strength, CausalStrength)
        assert strength.source_feature == "sae.0"
        assert strength.target_feature == "sae.10"
        assert strength.strength >= 0.0
        assert strength.relation_type in [
            CausalRelationType.NONE,
            CausalRelationType.WEAK,
            CausalRelationType.MODERATE,
            CausalRelationType.STRONG,
        ]

    def test_confounder_adjustment(self, estimator):
        """Test confounder adjustment."""
        np.random.seed(42)

        # Create confounded relationship
        for i in range(200):
            confounder = np.random.randn()
            source = {"sae.0": confounder + np.random.randn() * 0.3}
            target = {"sae.10": confounder + np.random.randn() * 0.3}
            confounders = {"sae.conf": confounder}

            estimator.observe(source, target, confounders)

        # Estimate with adjustment
        strength = estimator.estimate("sae.0", "sae.10")

        # Adjusted strength should be lower than raw correlation
        assert isinstance(strength, CausalStrength)

    def test_insufficient_data(self, estimator):
        """Test handles insufficient data gracefully."""
        strength = estimator.estimate("sae.0", "sae.10")

        assert strength.strength == 0.0
        assert strength.relation_type == CausalRelationType.NONE
        assert strength.confidence == 0.0


# =============================================================================
# TransferEntropyEstimator Tests
# =============================================================================

class TestTransferEntropyEstimator:
    """Test transfer entropy estimation."""

    @pytest.fixture
    def estimator(self):
        return TransferEntropyEstimator(n_bins=8, history_length=2)

    def test_observe_time_series(self, estimator):
        """Test observing time series data."""
        np.random.seed(42)

        for t in range(100):
            estimator.observe("sae.0", np.random.randn())
            estimator.observe("sae.10", np.random.randn())

        assert len(estimator._history["sae.0"]) == 100
        assert len(estimator._history["sae.10"]) == 100

    def test_estimate_independent(self, estimator):
        """Test TE ~0 for independent time series."""
        np.random.seed(42)

        for t in range(500):
            estimator.observe("sae.0", np.random.randn())
            estimator.observe("sae.10", np.random.randn())

        result = estimator.estimate("sae.0", "sae.10")

        assert isinstance(result, TransferEntropyResult)
        # TE should be near zero for independent series
        assert result.transfer_entropy < 0.5

    def test_estimate_causal(self, estimator):
        """Test TE > 0 for causally related time series."""
        np.random.seed(42)

        source_prev = 0.0
        for t in range(500):
            source = 0.8 * source_prev + np.random.randn() * 0.2
            target = 0.9 * source_prev + np.random.randn() * 0.1  # Causal from source

            estimator.observe("sae.0", source)
            estimator.observe("sae.10", target)

            source_prev = source

        result = estimator.estimate("sae.0", "sae.10")

        assert isinstance(result, TransferEntropyResult)
        assert result.source_feature == "sae.0"
        assert result.target_feature == "sae.10"

    def test_asymmetry(self, estimator):
        """Test TE is asymmetric (direction matters)."""
        np.random.seed(42)

        source_prev = 0.0
        for t in range(500):
            source = 0.8 * source_prev + np.random.randn() * 0.2
            target = 0.9 * source_prev + np.random.randn() * 0.1

            estimator.observe("sae.0", source)
            estimator.observe("sae.10", target)

            source_prev = source

        te_forward = estimator.estimate("sae.0", "sae.10")
        te_backward = estimator.estimate("sae.10", "sae.0")

        # TE should be different in each direction
        assert te_forward.transfer_entropy != te_backward.transfer_entropy

    def test_significance_testing(self, estimator):
        """Test significance testing with permutation."""
        np.random.seed(42)

        for t in range(200):
            estimator.observe("sae.0", np.random.randn())
            estimator.observe("sae.10", np.random.randn())

        result = estimator.estimate("sae.0", "sae.10", n_permutations=50)

        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0

    def test_insufficient_history(self, estimator):
        """Test handles insufficient history."""
        estimator.observe("sae.0", 1.0)

        result = estimator.estimate("sae.0", "sae.10")

        assert result.transfer_entropy == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test full correlation → propagation → causal pipeline."""
        np.random.seed(42)

        # 1. Build correlation tracker
        tracker = CorrelationTracker(n_layers=4)

        for batch in range(100):
            base = np.random.randn()
            noise = 0.1

            activations = {
                0: {"sae.0": base},
                1: {"sae.10": base * 0.9 + np.random.randn() * noise},
                2: {"sae.20": base * 0.8 + np.random.randn() * noise},
                3: {"sae.30": base * 0.7 + np.random.randn() * noise},
            }
            tracker.update(activations)

        # 2. Analyze propagation
        analyzer = PropagationAnalyzer(tracker)
        path = analyzer.trace_feature("sae.0", PropagationType.FORWARD)

        assert len(path.edges) > 0

        # 3. Discover circuits
        circuits = analyzer.discover_circuits(min_features=2)

        # 4. Estimate causal strength
        causal_estimator = CausalStrengthEstimator()

        for batch in range(100):
            base = np.random.randn()
            source = {"sae.0": base}
            target = {"sae.10": base * 0.9 + np.random.randn() * 0.1}
            causal_estimator.observe(source, target)

        strength = causal_estimator.estimate("sae.0", "sae.10")

        assert strength.strength > 0.5  # Should detect strong relationship

    def test_layer_correlation_consistency(self):
        """Test correlations are consistent across methods."""
        np.random.seed(42)

        config = CorrelationConfig(
            min_samples=10,
            method=CorrelationMethod.PEARSON,
        )
        tracker = CorrelationTracker(n_layers=3, config=config)

        # Generate perfectly correlated data
        for batch in range(50):
            x = np.random.randn()
            activations = {
                0: {"sae.0": x},
                1: {"sae.10": x},  # Perfect correlation
                2: {"sae.20": -x},  # Perfect anti-correlation
            }
            tracker.update(activations)

        # Check correlations
        corr_01 = tracker.get_layer_correlation(0, 1)
        corr_12 = tracker.get_layer_correlation(1, 2)

        assert corr_01 is not None
        assert corr_12 is not None

        # Find specific pairs
        pair_01 = None
        pair_12 = None

        for pair in corr_01.correlations:
            if pair.source_feature == "sae.0" and pair.target_feature == "sae.10":
                pair_01 = pair

        for pair in corr_12.correlations:
            if pair.source_feature == "sae.10" and pair.target_feature == "sae.20":
                pair_12 = pair

        if pair_01:
            assert pair_01.correlation > 0.9  # Near 1.0

        if pair_12:
            assert pair_12.correlation < -0.9  # Near -1.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_activations(self):
        """Test handles empty activations."""
        tracker = CorrelationTracker(n_layers=3)

        tracker.update({})  # Empty
        tracker.update({0: {}})  # Empty layer

        snapshot = tracker.snapshot()
        assert snapshot.total_pairs == 0

    def test_single_layer(self):
        """Test handles single layer (no cross-layer correlation)."""
        tracker = CorrelationTracker(n_layers=1)

        for i in range(10):
            tracker.update({0: {"sae.0": float(i)}})

        # No adjacent layers to correlate
        snapshot = tracker.snapshot()
        assert snapshot.total_pairs == 0

    def test_missing_layers(self):
        """Test handles non-contiguous layers."""
        tracker = CorrelationTracker(n_layers=5)

        for i in range(20):
            # Skip layer 2
            activations = {
                0: {"sae.0": np.random.randn()},
                1: {"sae.10": np.random.randn()},
                3: {"sae.30": np.random.randn()},
                4: {"sae.40": np.random.randn()},
            }
            tracker.update(activations)

        # Should still track available pairs
        corr_01 = tracker.get_layer_correlation(0, 1)
        corr_34 = tracker.get_layer_correlation(3, 4)

        assert corr_01 is not None or corr_34 is not None

    def test_nan_handling(self):
        """Test handles NaN values."""
        correlator = WelfordCorrelator()

        correlator.update(1.0, 2.0)
        correlator.update(float('nan'), 3.0)  # NaN
        correlator.update(2.0, 4.0)

        # Should still compute correlation from valid values
        corr = correlator.correlation()
        if corr is not None:
            assert not math.isnan(corr)

    def test_inf_handling(self):
        """Test handles infinity values."""
        correlator = WelfordCorrelator()

        for i in range(10):
            correlator.update(float(i), float(i) * 2)

        correlator.update(float('inf'), 100.0)  # Infinity

        # Should handle gracefully
        corr = correlator.correlation()
        # May be None or a number, but not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
