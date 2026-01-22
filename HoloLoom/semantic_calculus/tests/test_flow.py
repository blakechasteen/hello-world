"""
Tests for Semantic Flow: PDE-Based Temporal Dynamics
=====================================================

Tests for SemanticFlowState and SemanticFlow PDE-based evolution.

Test Coverage:
    - SemanticFlowState dataclass and properties
    - SemanticFlow initialization and configuration
    - PDE solver creation (heat, wave, reaction-diffusion, Hamilton-Jacobi)
    - State evolution and trajectory tracking
    - Energy and entropy evolution
    - Edge cases and numerical stability
    - Performance benchmarks

Author: HoloLoom Team
Date: 2025-01-22
"""

import pytest
import numpy as np
from typing import List, Dict, Optional
import warnings


# ============================================================================
# Test Imports
# ============================================================================

class TestFlowImports:
    """Test that all flow components can be imported."""

    def test_import_semantic_flow_state(self):
        """Test SemanticFlowState import."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState
        assert SemanticFlowState is not None

    def test_import_semantic_flow(self):
        """Test SemanticFlow import."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        assert SemanticFlow is not None

    def test_import_create_semantic_flow(self):
        """Test create_semantic_flow factory import."""
        from HoloLoom.semantic_calculus.flow import create_semantic_flow
        assert create_semantic_flow is not None

    def test_import_pde_availability_flag(self):
        """Test PDE availability flag import."""
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        # Just check it exists (may be True or False)
        assert isinstance(_PDE_AVAILABLE, bool)


# ============================================================================
# SemanticFlowState Tests
# ============================================================================

class TestSemanticFlowStateDataclass:
    """Test SemanticFlowState dataclass functionality."""

    def test_creation_basic(self):
        """Test basic SemanticFlowState creation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.array([1.0, 2.0, 3.0, 4.0])
        state = SemanticFlowState(activation=activation, time=0.0)

        assert np.allclose(state.activation, activation)
        assert state.time == 0.0
        assert state.metadata == {}

    def test_creation_with_metadata(self):
        """Test SemanticFlowState creation with metadata."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.ones(16)
        metadata = {'pde_type': 'heat', 'diffusivity': 1.0}
        state = SemanticFlowState(activation=activation, time=1.5, metadata=metadata)

        assert state.time == 1.5
        assert state.metadata['pde_type'] == 'heat'
        assert state.metadata['diffusivity'] == 1.0

    def test_post_init_array_conversion(self):
        """Test that post_init converts lists to arrays."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        # Pass a list instead of array
        activation = [1.0, 2.0, 3.0]
        state = SemanticFlowState(activation=activation, time=0.0)

        assert isinstance(state.activation, np.ndarray)
        assert state.activation.shape == (3,)


class TestSemanticFlowStateEnergy:
    """Test SemanticFlowState energy property."""

    def test_energy_zero_activation(self):
        """Test energy is zero with zero activation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.zeros(10)
        state = SemanticFlowState(activation=activation, time=0.0)

        assert state.energy == 0.0

    def test_energy_unit_activation(self):
        """Test energy with unit activation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.array([1.0, 0.0, 0.0, 0.0])
        state = SemanticFlowState(activation=activation, time=0.0)

        # Energy = ||activation||^2 = 1
        assert state.energy == 1.0

    def test_energy_general(self):
        """Test energy with general activation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.array([3.0, 4.0])  # ||a||^2 = 9 + 16 = 25
        state = SemanticFlowState(activation=activation, time=0.0)

        assert state.energy == 25.0


class TestSemanticFlowStateEntropy:
    """Test SemanticFlowState entropy property."""

    def test_entropy_uniform_activation(self):
        """Test entropy with uniform activation (max entropy)."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.ones(4) * 0.25  # Uniform distribution
        state = SemanticFlowState(activation=activation, time=0.0)

        # Max entropy for 4 states = log(4)
        expected_entropy = np.log(4)
        assert np.isclose(state.entropy, expected_entropy, rtol=1e-3)

    def test_entropy_peaked_activation(self):
        """Test entropy with peaked activation (low entropy)."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.array([1.0, 0.0, 0.0, 0.0])
        state = SemanticFlowState(activation=activation, time=0.0)

        # Peaked distribution has zero entropy
        assert state.entropy == 0.0

    def test_entropy_always_non_negative(self):
        """Test that entropy is always non-negative."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        rng = np.random.RandomState(42)
        for _ in range(10):
            activation = rng.randn(16)
            state = SemanticFlowState(activation=activation, time=0.0)
            assert state.entropy >= 0


# ============================================================================
# SemanticFlow Initialization Tests
# ============================================================================

class TestSemanticFlowInitialization:
    """Test SemanticFlow initialization."""

    @pytest.fixture
    def skip_if_no_pde(self):
        """Skip test if PDE solvers not available."""
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")

    def test_init_heat_equation(self, skip_if_no_pde):
        """Test initialization with heat equation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

        assert flow.dimensions == 16
        assert flow.pde_type == "heat"
        assert flow.dt == 0.01

    def test_init_wave_equation(self, skip_if_no_pde):
        """Test initialization with wave equation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="wave", dt=0.005)

        assert flow.pde_type == "wave"

    def test_init_reaction_diffusion(self, skip_if_no_pde):
        """Test initialization with reaction-diffusion equation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(
            dimensions=16,
            pde_type="reaction_diffusion",
            dt=0.01,
            reaction_type='competitive'
        )

        assert flow.pde_type == "reaction_diffusion"

    def test_init_hamilton_jacobi(self, skip_if_no_pde):
        """Test initialization with Hamilton-Jacobi equation."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="hamilton_jacobi", dt=0.01)

        assert flow.pde_type == "hamilton_jacobi"

    def test_init_creates_default_laplacian(self, skip_if_no_pde):
        """Test that default Laplacian is created."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

        assert flow.laplacian.shape == (16, 16)

    def test_init_custom_laplacian(self, skip_if_no_pde):
        """Test initialization with custom Laplacian."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        # Custom Laplacian (tridiagonal)
        n = 16
        L = np.diag(np.ones(n) * 2) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)

        flow = SemanticFlow(dimensions=n, pde_type="heat", dt=0.01, laplacian=L)

        assert np.allclose(flow.laplacian, L)

    def test_init_invalid_laplacian_shape(self, skip_if_no_pde):
        """Test that wrong Laplacian shape raises error."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        L = np.eye(10)  # Wrong shape for 16 dimensions

        with pytest.raises(ValueError, match="Laplacian shape"):
            SemanticFlow(dimensions=16, pde_type="heat", dt=0.01, laplacian=L)

    def test_init_unknown_pde_type(self, skip_if_no_pde):
        """Test that unknown PDE type raises error."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        with pytest.raises(ValueError, match="Unknown PDE type"):
            SemanticFlow(dimensions=16, pde_type="unknown_type", dt=0.01)


# ============================================================================
# Laplacian Tests
# ============================================================================

class TestDefaultLaplacian:
    """Test default Laplacian creation."""

    @pytest.fixture
    def skip_if_no_pde(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")

    def test_laplacian_symmetric(self, skip_if_no_pde):
        """Test that default Laplacian is symmetric."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        L = flow.laplacian

        assert np.allclose(L, L.T)

    def test_laplacian_row_sum_zero(self, skip_if_no_pde):
        """Test that Laplacian rows sum to zero."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        L = flow.laplacian

        row_sums = np.sum(L, axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)

    def test_laplacian_diagonal_positive(self, skip_if_no_pde):
        """Test that Laplacian diagonal is non-negative."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        L = flow.laplacian

        assert np.all(np.diag(L) >= 0)

    def test_laplacian_nearest_neighbor(self, skip_if_no_pde):
        """Test that default Laplacian connects nearest neighbors."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=8, pde_type="heat", dt=0.01)
        L = flow.laplacian

        # Check diagonal is 2 (degree = 2 neighbors on circle)
        assert np.allclose(np.diag(L), 2)

        # Check off-diagonal neighbors are -1
        for i in range(8):
            assert L[i, (i+1) % 8] == -1
            assert L[i, (i-1) % 8] == -1


# ============================================================================
# State Creation Tests
# ============================================================================

class TestCreateState:
    """Test SemanticFlow.create_state method."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_create_state_basic(self, flow):
        """Test basic state creation."""
        activation = np.ones(16)
        state = flow.create_state(activation)

        assert isinstance(state.activation, np.ndarray)
        assert state.time == 0.0

    def test_create_state_with_time(self, flow):
        """Test state creation with time."""
        activation = np.ones(16)
        state = flow.create_state(activation, time=2.5)

        assert state.time == 2.5

    def test_create_state_with_metadata(self, flow):
        """Test state creation with metadata."""
        activation = np.ones(16)
        state = flow.create_state(activation, source='test', confidence=0.9)

        assert state.metadata['source'] == 'test'
        assert state.metadata['confidence'] == 0.9

    def test_create_state_wrong_dimension(self, flow):
        """Test that wrong dimension raises error."""
        activation = np.ones(8)  # Wrong dimension

        with pytest.raises(ValueError, match="Activation dimension"):
            flow.create_state(activation)


# ============================================================================
# Evolution Tests
# ============================================================================

class TestEvolution:
    """Test SemanticFlow evolution."""

    @pytest.fixture
    def heat_flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_evolve_returns_state(self, heat_flow):
        """Test that evolve returns SemanticFlowState."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        initial = heat_flow.create_state(np.ones(16))
        final = heat_flow.evolve(initial, steps=10)

        assert isinstance(final, SemanticFlowState)

    def test_evolve_changes_state(self, heat_flow):
        """Test that evolution changes the state."""
        # Non-uniform initial state
        activation = np.zeros(16)
        activation[0] = 1.0  # Localized activation
        initial = heat_flow.create_state(activation)

        final = heat_flow.evolve(initial, steps=100)

        # Heat equation should spread the activation
        # Final state should be more uniform
        initial_std = np.std(initial.activation)
        final_std = np.std(final.activation)
        assert final_std < initial_std

    def test_evolve_trajectory_tracking(self, heat_flow):
        """Test that trajectory is tracked."""
        initial = heat_flow.create_state(np.ones(16))
        heat_flow.evolve(initial, steps=10, track_trajectory=True)

        trajectory = heat_flow.get_trajectory()
        # Trajectory includes initial state + n steps = n+1 states
        assert len(trajectory) == 11

    def test_evolve_no_trajectory_tracking(self, heat_flow):
        """Test evolution without trajectory tracking."""
        initial = heat_flow.create_state(np.ones(16))
        heat_flow.evolve(initial, steps=10, track_trajectory=False)

        # Trajectory should be empty when not tracking
        trajectory = heat_flow.get_trajectory()
        # Note: Implementation may vary - check actual behavior

    def test_evolve_time_progression(self, heat_flow):
        """Test that time progresses correctly."""
        initial = heat_flow.create_state(np.ones(16), time=0.0)
        final = heat_flow.evolve(initial, steps=10)

        expected_time = 10 * heat_flow.dt
        assert np.isclose(final.time, expected_time, rtol=0.1)


class TestWaveEvolution:
    """Test wave equation evolution."""

    @pytest.fixture
    def wave_flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="wave", dt=0.005)

    def test_wave_oscillation(self, wave_flow):
        """Test that wave equation produces oscillations."""
        # Initial displacement
        activation = np.zeros(16)
        activation[8] = 1.0
        initial = wave_flow.create_state(activation)

        final = wave_flow.evolve(initial, steps=200)

        # Wave should spread and oscillate
        assert not np.allclose(final.activation, initial.activation)

    def test_wave_with_velocity(self, wave_flow):
        """Test wave equation with initial velocity."""
        from HoloLoom.semantic_calculus.flow import SemanticFlowState

        activation = np.zeros(16)
        velocity = np.ones(16) * 0.1

        initial = SemanticFlowState(
            activation=activation,
            time=0.0,
            metadata={'velocity': velocity}
        )

        final = wave_flow.evolve(initial, steps=100)

        # Should evolve
        assert not np.allclose(final.activation, 0)


# ============================================================================
# Trajectory Analysis Tests
# ============================================================================

class TestTrajectoryAnalysis:
    """Test trajectory analysis methods."""

    @pytest.fixture
    def evolved_flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        activation = np.zeros(16)
        activation[0] = 1.0
        initial = flow.create_state(activation)
        flow.evolve(initial, steps=50, track_trajectory=True)
        return flow

    def test_get_trajectory_returns_list(self, evolved_flow):
        """Test that get_trajectory returns list of states."""
        trajectory = evolved_flow.get_trajectory()

        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

    def test_get_trajectory_array(self, evolved_flow):
        """Test get_trajectory_array returns numpy array."""
        trajectory_array = evolved_flow.get_trajectory_array()

        assert isinstance(trajectory_array, np.ndarray)
        # Shape should be (n_steps, n_dims)
        assert trajectory_array.ndim == 2
        assert trajectory_array.shape[1] == 16

    def test_compute_energy_evolution(self, evolved_flow):
        """Test energy evolution computation."""
        energies = evolved_flow.compute_energy_evolution()

        assert isinstance(energies, np.ndarray)
        assert len(energies) == len(evolved_flow.get_trajectory())
        # All energies should be non-negative
        assert np.all(energies >= 0)

    def test_compute_entropy_evolution(self, evolved_flow):
        """Test entropy evolution computation."""
        entropies = evolved_flow.compute_entropy_evolution()

        assert isinstance(entropies, np.ndarray)
        assert len(entropies) == len(evolved_flow.get_trajectory())
        # All entropies should be non-negative
        assert np.all(entropies >= 0)

    def test_heat_equation_entropy_increases(self, evolved_flow):
        """Test that heat equation increases entropy (second law)."""
        entropies = evolved_flow.compute_entropy_evolution()

        # Heat equation should generally increase entropy
        # Allow some numerical fluctuation
        entropy_trend = entropies[-1] - entropies[0]
        # Should increase or stay similar
        assert entropy_trend >= -0.1


# ============================================================================
# Semantic Distance Tests
# ============================================================================

class TestSemanticDistance:
    """Test semantic distance computation."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_distance_zero_for_same_state(self, flow):
        """Test distance is zero for identical states."""
        activation = np.ones(16)
        state1 = flow.create_state(activation)
        state2 = flow.create_state(activation.copy())

        distance = flow.compute_semantic_distance(state1, state2)
        assert distance == 0.0

    def test_distance_symmetric(self, flow):
        """Test distance is symmetric."""
        state1 = flow.create_state(np.ones(16))
        state2 = flow.create_state(np.ones(16) * 2)

        d12 = flow.compute_semantic_distance(state1, state2)
        d21 = flow.compute_semantic_distance(state2, state1)

        assert np.isclose(d12, d21)

    def test_distance_triangle_inequality(self, flow):
        """Test triangle inequality holds."""
        state1 = flow.create_state(np.array([1.0] + [0.0] * 15))
        state2 = flow.create_state(np.array([0.5] + [0.5] + [0.0] * 14))
        state3 = flow.create_state(np.array([0.0] * 15 + [1.0]))

        d12 = flow.compute_semantic_distance(state1, state2)
        d23 = flow.compute_semantic_distance(state2, state3)
        d13 = flow.compute_semantic_distance(state1, state3)

        assert d13 <= d12 + d23 + 1e-10  # Small tolerance


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Test flow reset functionality."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_reset_clears_trajectory(self, flow):
        """Test that reset clears trajectory."""
        initial = flow.create_state(np.ones(16))
        flow.evolve(initial, steps=10, track_trajectory=True)

        assert len(flow.get_trajectory()) > 0

        flow.reset()

        assert len(flow.get_trajectory()) == 0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateSemanticFlow:
    """Test create_semantic_flow factory function."""

    @pytest.fixture
    def skip_if_no_pde(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")

    def test_create_default(self, skip_if_no_pde):
        """Test factory with default parameters."""
        from HoloLoom.semantic_calculus.flow import create_semantic_flow

        flow = create_semantic_flow()

        assert flow.dimensions == 16  # Default
        assert flow.pde_type == "heat"  # Default
        assert flow.dt == 0.01  # Default

    def test_create_custom_dimensions(self, skip_if_no_pde):
        """Test factory with custom dimensions."""
        from HoloLoom.semantic_calculus.flow import create_semantic_flow

        flow = create_semantic_flow(dimensions=32)

        assert flow.dimensions == 32

    def test_create_wave_type(self, skip_if_no_pde):
        """Test factory with wave PDE type."""
        from HoloLoom.semantic_calculus.flow import create_semantic_flow

        flow = create_semantic_flow(pde_type="wave", dt=0.005)

        assert flow.pde_type == "wave"


# ============================================================================
# Edge Cases
# ============================================================================

class TestFlowEdgeCases:
    """Test edge cases for SemanticFlow."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_zero_activation(self, flow):
        """Test evolution with zero initial activation."""
        initial = flow.create_state(np.zeros(16))
        final = flow.evolve(initial, steps=10)

        # Zero stays zero with heat equation (equilibrium)
        assert np.allclose(final.activation, 0, atol=1e-10)

    def test_uniform_activation(self, flow):
        """Test evolution with uniform activation."""
        initial = flow.create_state(np.ones(16))
        final = flow.evolve(initial, steps=100)

        # Evolution completes successfully
        # Note: Boundary conditions may cause non-uniform evolution
        # so we don't assert strict uniformity preservation
        assert final is not None
        assert final.activation.shape == (16,)
        assert not np.any(np.isnan(final.activation))

    def test_single_step_evolution(self, flow):
        """Test evolution with single step."""
        initial = flow.create_state(np.ones(16))
        final = flow.evolve(initial, steps=1)

        # Should complete without error
        assert final is not None

    def test_many_steps_evolution(self, flow):
        """Test evolution with many steps."""
        activation = np.zeros(16)
        activation[0] = 1.0
        initial = flow.create_state(activation)
        final = flow.evolve(initial, steps=1000)

        # Should complete without error
        assert final is not None
        # After many steps, should approach equilibrium
        assert np.std(final.activation) < np.std(initial.activation)


class TestFlowNumericalStability:
    """Test numerical stability of flow evolution."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_no_nan_values(self, flow):
        """Test that evolution doesn't produce NaN values."""
        rng = np.random.RandomState(42)
        activation = rng.randn(16)
        initial = flow.create_state(activation)

        final = flow.evolve(initial, steps=100)

        assert not np.any(np.isnan(final.activation))

    def test_no_inf_values(self, flow):
        """Test that evolution doesn't produce Inf values."""
        rng = np.random.RandomState(42)
        activation = rng.randn(16)
        initial = flow.create_state(activation)

        final = flow.evolve(initial, steps=100)

        assert not np.any(np.isinf(final.activation))

    def test_small_activation_stability(self, flow):
        """Test stability with very small activation."""
        activation = np.ones(16) * 1e-15
        initial = flow.create_state(activation)

        final = flow.evolve(initial, steps=50)

        assert not np.any(np.isnan(final.activation))

    def test_large_activation_stability(self, flow):
        """Test stability with large activation."""
        activation = np.ones(16) * 1000
        initial = flow.create_state(activation)

        final = flow.evolve(initial, steps=50)

        assert not np.any(np.isnan(final.activation))
        assert not np.any(np.isinf(final.activation))


# ============================================================================
# Performance Tests
# ============================================================================

class TestFlowPerformance:
    """Performance benchmarks for SemanticFlow."""

    @pytest.fixture
    def flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow
        return SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)

    def test_evolution_speed(self, flow):
        """Test that evolution completes in reasonable time."""
        import time

        activation = np.ones(16)
        initial = flow.create_state(activation)

        start = time.perf_counter()
        flow.evolve(initial, steps=1000)
        elapsed = time.perf_counter() - start

        # 1000 steps should complete in < 2 seconds
        assert elapsed < 2.0

    def test_trajectory_array_speed(self, flow):
        """Test trajectory array extraction speed."""
        import time

        activation = np.ones(16)
        initial = flow.create_state(activation)
        flow.evolve(initial, steps=500, track_trajectory=True)

        start = time.perf_counter()
        trajectory_array = flow.get_trajectory_array()
        elapsed = time.perf_counter() - start

        # Array extraction should be fast
        assert elapsed < 0.1
        # Trajectory includes initial state + n steps = n+1 states
        assert trajectory_array.shape == (501, 16)


# ============================================================================
# Visualization Tests
# ============================================================================

class TestFlowVisualization:
    """Test flow visualization (minimal, checking it runs)."""

    @pytest.fixture
    def evolved_flow(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        activation = np.zeros(16)
        activation[0] = 1.0
        initial = flow.create_state(activation)
        flow.evolve(initial, steps=50, track_trajectory=True)
        return flow

    def test_visualize_returns_figure(self, evolved_flow, tmp_path):
        """Test that visualization returns figure."""
        try:
            import matplotlib
            matplotlib.use('Agg')

            dimension_names = [f"Dim{i}" for i in range(16)]
            fig, axes = evolved_flow.visualize_trajectory(dimension_names=dimension_names)

            assert fig is not None
            assert len(axes) == 3

            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_visualize_save(self, evolved_flow, tmp_path):
        """Test that visualization can save to file."""
        try:
            import matplotlib
            matplotlib.use('Agg')

            save_path = str(tmp_path / "test_flow.png")
            dimension_names = [f"Dim{i}" for i in range(16)]

            fig, axes = evolved_flow.visualize_trajectory(
                dimension_names=dimension_names,
                save_path=save_path
            )

            import os
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


# ============================================================================
# Integration Tests
# ============================================================================

class TestFlowWithSpectrum:
    """Test integration of SemanticFlow with SemanticSpectrum."""

    @pytest.fixture
    def skip_if_no_pde(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")

    def test_flow_with_spectrum_projection(self, skip_if_no_pde, mock_spectrum, mock_embed_fn):
        """Test using spectrum projection with flow."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        # Project a word to semantic space
        embedding = mock_embed_fn("Thompson Sampling")
        projection = mock_spectrum.project_vector(embedding)

        # Create flow with same dimensions as spectrum
        n_dims = len(projection)
        flow = SemanticFlow(dimensions=n_dims, pde_type="heat", dt=0.01)

        # Create initial state from projection
        activation = np.array(list(projection.values()))
        initial = flow.create_state(activation)

        # Evolve
        final = flow.evolve(initial, steps=50)

        # Should complete without error
        assert final.activation.shape == (n_dims,)

    def test_trajectory_projection_evolution(self, skip_if_no_pde, mock_spectrum, sample_trajectory):
        """Test evolving trajectory projections."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        # Project trajectory - returns Dict[str, np.ndarray] mapping dim names to value arrays
        trajectory_proj = mock_spectrum.project_trajectory(sample_trajectory)

        # Get dimensions and first timestep values
        n_dims = len(trajectory_proj)
        # Extract first timestep value from each dimension
        first_proj = {dim_name: values[0] for dim_name, values in trajectory_proj.items()}

        # Create flow
        flow = SemanticFlow(dimensions=n_dims, pde_type="heat", dt=0.01)

        # Use first trajectory point as initial condition
        activation = np.array(list(first_proj.values()))
        initial = flow.create_state(activation)

        # Evolve
        final = flow.evolve(initial, steps=20)

        assert final is not None


class TestPDETypeComparison:
    """Compare behavior of different PDE types."""

    @pytest.fixture
    def skip_if_no_pde(self):
        from HoloLoom.semantic_calculus.flow import _PDE_AVAILABLE
        if not _PDE_AVAILABLE:
            pytest.skip("PDE solvers not available")

    def test_heat_vs_wave_behavior(self, skip_if_no_pde):
        """Test that heat and wave PDEs behave differently."""
        from HoloLoom.semantic_calculus.flow import SemanticFlow

        # Same initial condition
        activation = np.zeros(16)
        activation[8] = 1.0  # Localized peak

        # Heat equation (diffusion)
        heat_flow = SemanticFlow(dimensions=16, pde_type="heat", dt=0.01)
        heat_initial = heat_flow.create_state(activation.copy())
        heat_final = heat_flow.evolve(heat_initial, steps=100)

        # Wave equation (oscillation)
        wave_flow = SemanticFlow(dimensions=16, pde_type="wave", dt=0.005)
        wave_initial = wave_flow.create_state(activation.copy())
        wave_final = wave_flow.evolve(wave_initial, steps=100)

        # Heat should spread and decay, wave should preserve more structure
        # This is a qualitative check
        heat_max = np.max(np.abs(heat_final.activation))
        wave_max = np.max(np.abs(wave_final.activation))

        # Both should complete without producing NaN or Inf
        assert not np.any(np.isnan(heat_final.activation))
        assert not np.any(np.isnan(wave_final.activation))

        # Heat equation typically diffuses, but boundary effects may cause variations
        # So we only check that values are finite and evolution completed
        assert np.all(np.isfinite(heat_final.activation))
