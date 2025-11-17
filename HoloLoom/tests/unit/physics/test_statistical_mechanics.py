"""
Tests for Physics - Statistical Mechanics

Tests emergent memory consolidation using statistical physics.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 5
Target: +8 tests for statistical mechanics
"""

import pytest
import numpy as np
from HoloLoom.physics.statistical_mechanics import (
    StatisticalMechanicsEngine,
    EnsembleState,
    PartitionFunction
)


class TestStatisticalMechanicsInitialization:
    """Test suite for StatisticalMechanicsEngine initialization."""

    def test_engine_initialization(self):
        """Test engine initialization with default parameters."""
        engine = StatisticalMechanicsEngine(
            beta=1.0,  # Inverse temperature
            ensemble_size=100
        )

        assert engine.beta == 1.0
        assert engine.ensemble_size == 100

    def test_ensemble_state_creation(self):
        """Test EnsembleState object creation."""
        state = EnsembleState(
            energies=[0.1, 0.2, 0.3],
            probabilities=[0.5, 0.3, 0.2]
        )

        assert len(state.energies) == 3
        assert len(state.probabilities) == 3
        assert sum(state.probabilities) == pytest.approx(1.0, abs=0.01)


class TestBoltzmannDistribution:
    """Test suite for Boltzmann distribution calculations."""

    def test_boltzmann_probability(self):
        """Test Boltzmann probability calculation."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        energy = 0.5
        partition_func = 10.0

        prob = engine.boltzmann_probability(energy, partition_func)

        # P(E) = exp(-beta * E) / Z
        expected = np.exp(-1.0 * energy) / partition_func
        assert prob == pytest.approx(expected, abs=0.01)

    def test_partition_function(self):
        """Test partition function calculation."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        energies = [0.0, 0.5, 1.0, 1.5]

        Z = engine.calculate_partition_function(energies)

        # Z = sum(exp(-beta * E_i))
        expected = sum(np.exp(-1.0 * E) for E in energies)
        assert Z == pytest.approx(expected, abs=0.01)


class TestEnsembleOperations:
    """Test suite for ensemble state operations."""

    def test_sample_from_ensemble(self):
        """Test sampling states from ensemble."""
        engine = StatisticalMechanicsEngine(beta=1.0, ensemble_size=100)

        energies = [0.0, 1.0, 2.0]  # Low, medium, high energy states

        # Sample many times
        samples = []
        for _ in range(100):
            state_idx = engine.sample_state(energies)
            samples.append(state_idx)

        # Should favor low energy states at beta=1.0
        # State 0 (E=0.0) should be sampled most often
        assert samples.count(0) > samples.count(2)

    def test_ensemble_average_energy(self):
        """Test ensemble average energy calculation."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        energies = [0.0, 1.0, 2.0, 3.0]

        avg_energy = engine.calculate_average_energy(energies)

        # Average should be weighted by Boltzmann probabilities
        assert 0.0 <= avg_energy <= max(energies)


class TestMemoryConsolidation:
    """Test suite for memory consolidation operations."""

    def test_consolidate_memories(self):
        """Test memory consolidation based on energy landscape."""
        engine = StatisticalMechanicsEngine(beta=2.0)  # Low temperature

        # Simulate memory energies (lower = more important)
        memory_energies = {
            "memory_1": 0.1,  # Very important
            "memory_2": 0.5,  # Moderately important
            "memory_3": 2.0   # Less important
        }

        consolidated = engine.consolidate(memory_energies, keep_fraction=0.5)

        # Should keep low-energy (important) memories
        assert "memory_1" in consolidated
        assert len(consolidated) <= len(memory_energies)

    def test_forgetting_curve(self):
        """Test that less important memories are more likely to be forgotten."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        # Repeat consolidation many times
        important_kept = 0
        unimportant_kept = 0

        for _ in range(50):
            energies = {"important": 0.1, "unimportant": 2.0}
            consolidated = engine.consolidate(energies, keep_fraction=0.5)

            if "important" in consolidated:
                important_kept += 1
            if "unimportant" in consolidated:
                unimportant_kept += 1

        # Important memory should be kept more often
        assert important_kept > unimportant_kept


class TestEntropyCalculation:
    """Test suite for entropy calculations."""

    def test_entropy_formula(self):
        """Test entropy S = -sum(p_i * log(p_i)) calculation."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        probabilities = [0.25, 0.25, 0.25, 0.25]  # Maximum entropy (uniform)

        entropy = engine.calculate_entropy(probabilities)

        # Maximum entropy for 4 states = log(4) = 1.386
        assert entropy > 1.0

    def test_low_entropy_state(self):
        """Test entropy of concentrated distribution."""
        engine = StatisticalMechanicsEngine(beta=1.0)

        # Low entropy: one dominant state
        concentrated = [0.9, 0.05, 0.03, 0.02]

        # High entropy: uniform distribution
        uniform = [0.25, 0.25, 0.25, 0.25]

        entropy_low = engine.calculate_entropy(concentrated)
        entropy_high = engine.calculate_entropy(uniform)

        # Uniform distribution should have higher entropy
        assert entropy_high > entropy_low


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_zero_temperature_limit(self):
        """Test behavior at very low temperature (beta → ∞)."""
        engine = StatisticalMechanicsEngine(beta=100.0)  # Very low temperature

        energies = [0.0, 1.0, 2.0]

        # At T→0, should deterministically choose lowest energy
        samples = [engine.sample_state(energies) for _ in range(20)]

        # Should always (or almost always) choose state 0
        assert samples.count(0) >= 18  # At least 90%

    def test_high_temperature_limit(self):
        """Test behavior at very high temperature (beta → 0)."""
        engine = StatisticalMechanicsEngine(beta=0.01)  # Very high temperature

        energies = [0.0, 1.0, 2.0]

        # At T→∞, should sample all states roughly equally
        samples = [engine.sample_state(energies) for _ in range(100)]

        # Each state should be sampled at least a few times
        for state_idx in range(len(energies)):
            assert samples.count(state_idx) > 10  # At least 10% for each
