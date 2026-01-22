"""
Tests for Geometric Integration in Semantic Dimension Space
============================================================

Tests for SemanticState, GeometricIntegrator, and MultiScaleGeometricFlow.

Test Coverage:
    - SemanticState dataclass and properties
    - GeometricIntegrator initialization and methods
    - Symplectic integration (Stormer-Verlet)
    - Trajectory integration with energy conservation
    - MultiScaleGeometricFlow resonance detection
    - Edge cases and numerical stability
    - Performance benchmarks

Author: HoloLoom Team
Date: 2025-01-22
"""

import pytest
import numpy as np
from typing import Callable, List, Dict
import warnings


# ============================================================================
# Test Imports
# ============================================================================

class TestIntegratorImports:
    """Test that all integrator components can be imported."""

    def test_import_semantic_state(self):
        """Test SemanticState import."""
        from HoloLoom.semantic_calculus.integrator import SemanticState
        assert SemanticState is not None

    def test_import_geometric_integrator(self):
        """Test GeometricIntegrator import."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator
        assert GeometricIntegrator is not None

    def test_import_multi_scale_geometric_flow(self):
        """Test MultiScaleGeometricFlow import."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow
        assert MultiScaleGeometricFlow is not None

    def test_import_visualize_function(self):
        """Test visualize_geometric_flow import."""
        from HoloLoom.semantic_calculus.integrator import visualize_geometric_flow
        assert visualize_geometric_flow is not None

    def test_import_force_field_function(self):
        """Test compute_semantic_force_field import."""
        from HoloLoom.semantic_calculus.integrator import compute_semantic_force_field
        assert compute_semantic_force_field is not None


# ============================================================================
# SemanticState Tests
# ============================================================================

class TestSemanticStateDataclass:
    """Test SemanticState dataclass functionality."""

    def test_creation_basic(self):
        """Test basic SemanticState creation."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.array([1.0, 2.0, 3.0])
        p = np.array([0.1, 0.2, 0.3])
        state = SemanticState(q=q, p=p, t=0.0)

        assert np.allclose(state.q, q)
        assert np.allclose(state.p, p)
        assert state.t == 0.0

    def test_creation_with_time(self):
        """Test SemanticState creation with explicit time."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.zeros(5)
        p = np.ones(5)
        state = SemanticState(q=q, p=p, t=2.5)

        assert state.t == 2.5

    def test_creation_16d(self):
        """Test SemanticState creation with 16D vectors (standard semantic space)."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        rng = np.random.RandomState(42)
        q = rng.randn(16)
        p = rng.randn(16)
        state = SemanticState(q=q, p=p, t=1.0)

        assert state.q.shape == (16,)
        assert state.p.shape == (16,)


class TestSemanticStateKineticEnergy:
    """Test SemanticState kinetic energy computation."""

    def test_kinetic_energy_zero_momentum(self):
        """Test kinetic energy is zero when momentum is zero."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.array([1.0, 2.0, 3.0])
        p = np.zeros(3)
        state = SemanticState(q=q, p=p)

        # Note: kinetic_energy is a property with mass argument
        # but the signature shows it as a property - this may be a bug in source
        # Test what the source code does
        ke = state.kinetic_energy
        assert isinstance(ke, (int, float))
        assert ke == 0.0

    def test_kinetic_energy_unit_momentum(self):
        """Test kinetic energy with unit momentum."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        # p = [1,0,0] -> ||p||^2 = 1 -> T = 0.5 * 1 / 1 = 0.5
        q = np.zeros(3)
        p = np.array([1.0, 0.0, 0.0])
        state = SemanticState(q=q, p=p)

        ke = state.kinetic_energy
        assert np.isclose(ke, 0.5, rtol=1e-10)

    def test_kinetic_energy_general(self):
        """Test kinetic energy with general momentum."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.zeros(3)
        p = np.array([3.0, 4.0, 0.0])  # ||p||^2 = 25 -> T = 0.5 * 25 = 12.5
        state = SemanticState(q=q, p=p)

        ke = state.kinetic_energy
        assert np.isclose(ke, 12.5, rtol=1e-10)


class TestSemanticStateHamiltonian:
    """Test SemanticState Hamiltonian computation."""

    def test_hamiltonian_zero_potential(self):
        """Test Hamiltonian equals kinetic energy when potential is zero."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 0.0, 0.0])
        state = SemanticState(q=q, p=p)

        def zero_potential(q):
            return 0.0

        H = state.hamiltonian(zero_potential, mass=1.0)
        assert np.isclose(H, 0.5, rtol=1e-10)

    def test_hamiltonian_harmonic_potential(self):
        """Test Hamiltonian with harmonic potential V = 0.5 * k * ||q||^2."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        k = 1.0  # spring constant
        q = np.array([1.0, 0.0, 0.0])  # ||q||^2 = 1 -> V = 0.5
        p = np.array([1.0, 0.0, 0.0])  # ||p||^2 = 1 -> T = 0.5
        state = SemanticState(q=q, p=p)

        def harmonic_potential(q):
            return 0.5 * k * np.dot(q, q)

        H = state.hamiltonian(harmonic_potential, mass=1.0)
        assert np.isclose(H, 1.0, rtol=1e-10)  # T + V = 0.5 + 0.5 = 1.0

    def test_hamiltonian_with_mass(self):
        """Test Hamiltonian with non-unit mass."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.zeros(3)
        p = np.array([2.0, 0.0, 0.0])  # ||p||^2 = 4
        state = SemanticState(q=q, p=p)

        def zero_potential(q):
            return 0.0

        # T = 0.5 * ||p||^2 / m = 0.5 * 4 / 2 = 1.0
        H = state.hamiltonian(zero_potential, mass=2.0)
        assert np.isclose(H, 1.0, rtol=1e-10)


# ============================================================================
# GeometricIntegrator Tests
# ============================================================================

class TestGeometricIntegratorInitialization:
    """Test GeometricIntegrator initialization."""

    def test_init_basic(self, mock_integrator):
        """Test basic initialization from fixture."""
        assert mock_integrator is not None
        assert hasattr(mock_integrator, 'P')
        assert hasattr(mock_integrator, 'P_T')
        assert hasattr(mock_integrator, 'mass')

    def test_init_projection_shape(self):
        """Test projection matrix shape is preserved."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        rng = np.random.RandomState(42)
        P = rng.randn(16, 384)
        integrator = GeometricIntegrator(P, mass=1.0)

        assert integrator.P.shape == (16, 384)
        assert integrator.P_T.shape == (384, 16)
        assert integrator.n_dims == 16

    def test_init_custom_mass(self):
        """Test initialization with custom mass."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        rng = np.random.RandomState(42)
        P = rng.randn(16, 384)
        integrator = GeometricIntegrator(P, mass=2.5)

        assert integrator.mass == 2.5

    def test_init_cache_enabled(self):
        """Test initialization with projection cache enabled."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        rng = np.random.RandomState(42)
        P = rng.randn(16, 384)
        integrator = GeometricIntegrator(P, enable_projection_cache=True)

        assert integrator._proj_cache is not None

    def test_init_cache_disabled(self):
        """Test initialization with projection cache disabled."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        rng = np.random.RandomState(42)
        P = rng.randn(16, 384)
        integrator = GeometricIntegrator(P, enable_projection_cache=False)

        assert integrator._proj_cache is None


class TestGeometricIntegratorProjection:
    """Test GeometricIntegrator projection methods."""

    def test_project_to_semantic(self, mock_integrator, random_vector_384):
        """Test projection from full space to semantic space."""
        q_sem = mock_integrator.project_to_semantic(random_vector_384)

        assert q_sem.shape == (16,)  # mock_integrator uses 16D
        assert not np.allclose(q_sem, 0)  # Should not be all zeros

    def test_lift_to_full(self, mock_integrator, random_vector_16):
        """Test lifting from semantic space to full space."""
        q_full = mock_integrator.lift_to_full(random_vector_16)

        assert q_full.shape == (384,)

    def test_project_lift_roundtrip(self, mock_integrator, random_vector_384):
        """Test that project then lift gives approximate reconstruction."""
        q_sem = mock_integrator.project_to_semantic(random_vector_384)
        q_reconstructed = mock_integrator.lift_to_full(q_sem)

        # Reconstruction should be in the subspace spanned by P.T
        # Not exactly equal to original, but captures projected component
        assert q_reconstructed.shape == random_vector_384.shape

    def test_project_gradient(self, mock_integrator, random_vector_384):
        """Test gradient projection."""
        grad_full = random_vector_384
        grad_sem = mock_integrator.project_gradient(grad_full)

        assert grad_sem.shape == (16,)


class TestStormerVerletStep:
    """Test Stormer-Verlet symplectic integration step."""

    def test_single_step_harmonic(self, mock_integrator):
        """Test single Stormer-Verlet step with harmonic potential."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        # Initial state
        q = np.zeros(16)
        q[0] = 1.0  # Displacement in first dimension
        p = np.zeros(16)
        state = SemanticState(q=q, p=p, t=0.0)

        # Harmonic potential gradient: ∇V = k*q
        k = 1.0

        def gradient_fn(q):
            return k * q

        dt = 0.01
        new_state = mock_integrator.stormer_verlet_step(state, gradient_fn, dt)

        # State should have evolved
        assert new_state.t == pytest.approx(dt, rel=1e-10)
        assert not np.allclose(new_state.q, q)
        assert not np.allclose(new_state.p, p)

    def test_step_preserves_dimension(self, mock_integrator):
        """Test that step preserves dimensionality."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        rng = np.random.RandomState(42)
        q = rng.randn(16)
        p = rng.randn(16)
        state = SemanticState(q=q, p=p, t=0.0)

        def zero_gradient(q):
            return np.zeros_like(q)

        new_state = mock_integrator.stormer_verlet_step(state, zero_gradient, dt=0.01)

        assert new_state.q.shape == q.shape
        assert new_state.p.shape == p.shape

    def test_step_free_particle(self, mock_integrator):
        """Test that free particle (zero potential) moves linearly."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        q = np.zeros(16)
        p = np.ones(16)  # Constant velocity
        state = SemanticState(q=q, p=p, t=0.0)

        def zero_gradient(q):
            return np.zeros_like(q)

        dt = 0.1
        new_state = mock_integrator.stormer_verlet_step(state, zero_gradient, dt)

        # Free particle: q(t+dt) = q(t) + dt * p / m
        expected_q = q + dt * p / mock_integrator.mass
        assert np.allclose(new_state.q, expected_q, rtol=1e-6)
        # Momentum should be unchanged (no forces)
        assert np.allclose(new_state.p, p, rtol=1e-6)


class TestIntegrateTrajectory:
    """Test trajectory integration."""

    def test_trajectory_length(self, mock_integrator, random_vector_384):
        """Test that trajectory has correct number of states."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.1 * q  # Simple gradient

        n_steps = 10
        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=n_steps
        )

        # Should have n_steps + 1 states (initial + n_steps)
        assert len(states) == n_steps + 1

    def test_trajectory_time_progression(self, mock_integrator, random_vector_384):
        """Test that time progresses correctly in trajectory."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return np.zeros_like(q)

        dt = 0.1
        n_steps = 5
        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=dt, n_steps=n_steps
        )

        # Check time progression
        for i, state in enumerate(states):
            expected_time = i * dt
            assert np.isclose(state.t, expected_time, rtol=1e-10)

    def test_trajectory_initial_state(self, mock_integrator, random_vector_384):
        """Test that trajectory starts with correct initial state."""
        q0_full = random_vector_384
        p0_full = random_vector_384 * 0.1

        def gradient_fn(q):
            return np.zeros_like(q)

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=5
        )

        # First state should be projection of initial conditions
        q0_sem = mock_integrator.project_to_semantic(q0_full)
        p0_sem = mock_integrator.project_to_semantic(p0_full)

        assert np.allclose(states[0].q, q0_sem, rtol=1e-10)
        assert np.allclose(states[0].p, p0_sem, rtol=1e-10)


class TestEnergyConservation:
    """Test symplectic energy conservation."""

    def test_energy_conservation_harmonic(self, mock_integrator):
        """Test energy conservation in harmonic oscillator."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        # Create harmonic oscillator
        k = 1.0

        def potential(q):
            return 0.5 * k * np.dot(q, q)

        def gradient_fn(q):
            return k * q

        # Initial conditions (some energy)
        rng = np.random.RandomState(42)
        q0_full = rng.randn(384)
        q0_full = q0_full / np.linalg.norm(q0_full)
        p0_full = rng.randn(384) * 0.5

        # Integrate
        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=100
        )

        # Compute energy drift
        drift = mock_integrator.compute_energy_drift(states, potential)

        # Energy should be well conserved (relative drift < 10%)
        assert abs(drift['drift']) < 0.1
        # Standard deviation should be small relative to mean
        assert drift['std'] / drift['mean'] < 0.1

    def test_energy_drift_returns_dict(self, mock_integrator, random_vector_384):
        """Test that compute_energy_drift returns proper dictionary."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.1 * q

        def potential(q):
            return 0.05 * np.dot(q, q)

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=10
        )

        drift = mock_integrator.compute_energy_drift(states, potential)

        assert 'energies' in drift
        assert 'mean' in drift
        assert 'std' in drift
        assert 'drift' in drift
        assert 'conservation_quality' in drift

    def test_conservation_quality_bounds(self, mock_integrator):
        """Test that conservation quality is in reasonable bounds."""
        rng = np.random.RandomState(42)
        q0_full = rng.randn(384)
        p0_full = rng.randn(384) * 0.1

        def gradient_fn(q):
            return 0.01 * q

        def potential(q):
            return 0.005 * np.dot(q, q)

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=50
        )

        drift = mock_integrator.compute_energy_drift(states, potential)

        # Conservation quality should be between 0 and 1 for good integrator
        assert 0 <= drift['conservation_quality'] <= 1.5  # Allow some tolerance


# ============================================================================
# MultiScaleGeometricFlow Tests
# ============================================================================

class TestMultiScaleGeometricFlowInitialization:
    """Test MultiScaleGeometricFlow initialization."""

    def test_init_basic(self):
        """Test basic initialization with multiple scales."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            192: rng.randn(16, 192),
            384: rng.randn(16, 384)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        assert len(flow.scales) == 3
        assert 96 in flow.integrators
        assert 192 in flow.integrators
        assert 384 in flow.integrators

    def test_init_with_masses(self):
        """Test initialization with custom masses."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            384: rng.randn(16, 384)
        }
        masses = {96: 0.5, 384: 2.0}

        flow = MultiScaleGeometricFlow(projection_matrices, masses=masses)

        assert flow.integrators[96].mass == 0.5
        assert flow.integrators[384].mass == 2.0

    def test_init_scales_sorted(self):
        """Test that scales are stored in sorted order."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow

        rng = np.random.RandomState(42)
        projection_matrices = {
            384: rng.randn(16, 384),
            96: rng.randn(16, 96),
            192: rng.randn(16, 192)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        assert flow.scales == [96, 192, 384]


class TestMultiScaleResonance:
    """Test multi-scale resonance computation."""

    def test_compute_resonance_basic(self):
        """Test resonance computation with simple states."""
        from HoloLoom.semantic_calculus.integrator import (
            MultiScaleGeometricFlow, SemanticState
        )

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            192: rng.randn(16, 192)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        # Create synchronized states (high resonance)
        states_96 = [
            SemanticState(q=rng.randn(16), p=rng.randn(16), t=i * 0.1)
            for i in range(5)
        ]
        states_192 = [
            SemanticState(q=rng.randn(16), p=rng.randn(16), t=i * 0.1)
            for i in range(5)
        ]

        states_by_scale = {96: states_96, 192: states_192}
        resonance = flow.compute_resonance(states_by_scale)

        assert resonance.shape == (5,)
        assert np.all(resonance >= 0)

    def test_resonance_identical_states(self):
        """Test resonance is high for identical states."""
        from HoloLoom.semantic_calculus.integrator import (
            MultiScaleGeometricFlow, SemanticState
        )

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            192: rng.randn(16, 192)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        # Create identical states across scales
        shared_q = rng.randn(16)
        shared_q = shared_q / np.linalg.norm(shared_q)

        states_96 = [
            SemanticState(q=shared_q.copy(), p=rng.randn(16), t=i * 0.1)
            for i in range(5)
        ]
        states_192 = [
            SemanticState(q=shared_q.copy(), p=rng.randn(16), t=i * 0.1)
            for i in range(5)
        ]

        states_by_scale = {96: states_96, 192: states_192}
        resonance = flow.compute_resonance(states_by_scale)

        # Identical normalized positions should give high correlation
        assert np.all(resonance > 0.9)


class TestMultiScaleIntegration:
    """Test multi-scale integration."""

    def test_integrate_all_scales(self):
        """Test integration across all scales."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            192: rng.randn(16, 192)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        # Initial conditions for each scale
        q0_full_by_scale = {
            96: rng.randn(96),
            192: rng.randn(192)
        }
        p0_full_by_scale = {
            96: rng.randn(96) * 0.1,
            192: rng.randn(192) * 0.1
        }

        # Gradient functions
        def gradient_fn_96(q):
            return 0.01 * q

        def gradient_fn_192(q):
            return 0.01 * q

        gradient_fns = {96: gradient_fn_96, 192: gradient_fn_192}

        result = flow.integrate_all_scales(
            q0_full_by_scale, p0_full_by_scale,
            gradient_fns, dt=0.01, n_steps=10
        )

        assert 'states_by_scale' in result
        assert 'resonance' in result
        assert 96 in result['states_by_scale']
        assert 192 in result['states_by_scale']
        assert len(result['states_by_scale'][96]) == 11  # n_steps + 1
        assert len(result['states_by_scale'][192]) == 11


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestComputeSemanticForceField:
    """Test semantic force field computation."""

    def test_force_field_basic(self):
        """Test basic force field computation."""
        from HoloLoom.semantic_calculus.integrator import compute_semantic_force_field

        rng = np.random.RandomState(42)
        P = rng.randn(16, 384)

        def gradient_fn(q):
            return 0.1 * q

        # Sample points in semantic space
        sample_points = rng.randn(5, 16)

        forces = compute_semantic_force_field(P, gradient_fn, sample_points)

        assert forces.shape == (5, 16)

    def test_force_field_shape(self):
        """Test that force field has correct shape."""
        from HoloLoom.semantic_calculus.integrator import compute_semantic_force_field

        rng = np.random.RandomState(42)
        n_dims = 8
        emb_dim = 256
        n_points = 10

        P = rng.randn(n_dims, emb_dim)

        def gradient_fn(q):
            return np.zeros_like(q)

        sample_points = rng.randn(n_points, n_dims)
        forces = compute_semantic_force_field(P, gradient_fn, sample_points)

        assert forces.shape == (n_points, n_dims)

    def test_force_field_direction(self):
        """Test that force points opposite to gradient (F = -nabla V)."""
        from HoloLoom.semantic_calculus.integrator import compute_semantic_force_field

        rng = np.random.RandomState(42)
        P = np.eye(16, 384)  # Simple identity-like projection

        # Gradient in one direction
        def gradient_fn(q):
            grad = np.zeros_like(q)
            grad[0] = 1.0  # Positive gradient in first direction
            return grad

        sample_points = rng.randn(1, 16)
        forces = compute_semantic_force_field(P, gradient_fn, sample_points)

        # Force should be in opposite direction to gradient
        assert forces[0, 0] < 0  # Negative force where gradient is positive


class TestVisualizeGeometricFlow:
    """Test visualization function (minimal tests, mainly checking it runs)."""

    def test_visualize_returns_figure(self, mock_integrator):
        """Test that visualization returns figure and axes."""
        pytest.importorskip("matplotlib")
        from HoloLoom.semantic_calculus.integrator import (
            visualize_geometric_flow, SemanticState
        )

        rng = np.random.RandomState(42)

        # Create some states
        states = [
            SemanticState(q=rng.randn(16), p=rng.randn(16), t=i * 0.1)
            for i in range(20)
        ]

        dimension_names = [f"Dim{i}" for i in range(16)]

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            fig, axes = visualize_geometric_flow(states, dimension_names)

            assert fig is not None
            assert len(axes) == 3

            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_visualize_handles_save(self, mock_integrator, tmp_path):
        """Test that visualization can save to file."""
        pytest.importorskip("matplotlib")
        from HoloLoom.semantic_calculus.integrator import (
            visualize_geometric_flow, SemanticState
        )

        rng = np.random.RandomState(42)

        states = [
            SemanticState(q=rng.randn(16), p=rng.randn(16), t=i * 0.1)
            for i in range(10)
        ]

        dimension_names = [f"Dim{i}" for i in range(16)]
        save_path = str(tmp_path / "test_flow.png")

        try:
            import matplotlib
            matplotlib.use('Agg')
            fig, axes = visualize_geometric_flow(states, dimension_names, save_path=save_path)

            import os
            assert os.path.exists(save_path)

            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


# ============================================================================
# Edge Cases
# ============================================================================

class TestIntegratorEdgeCases:
    """Test edge cases for integrator."""

    def test_zero_initial_momentum(self, mock_integrator, random_vector_384):
        """Test integration with zero initial momentum."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.1 * q

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=5
        )

        assert len(states) == 6
        # Should still evolve due to gradient
        assert not np.allclose(states[0].q, states[-1].q)

    def test_zero_gradient(self, mock_integrator, random_vector_384):
        """Test integration with zero gradient (free particle)."""
        q0_full = random_vector_384
        p0_full = np.ones(384)

        def zero_gradient(q):
            return np.zeros_like(q)

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, zero_gradient, dt=0.01, n_steps=5
        )

        # Momentum should be unchanged
        assert np.allclose(states[0].p, states[-1].p, rtol=1e-6)

    def test_single_step_integration(self, mock_integrator, random_vector_384):
        """Test integration with n_steps=1."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.1 * q

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=1
        )

        assert len(states) == 2  # Initial + 1 step

    def test_small_timestep(self, mock_integrator, random_vector_384):
        """Test integration with very small timestep."""
        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.1 * q

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=1e-6, n_steps=10
        )

        # Should still work without numerical issues
        assert len(states) == 11
        assert not np.any(np.isnan(states[-1].q))
        assert not np.any(np.isnan(states[-1].p))

    def test_large_timestep_warning(self, mock_integrator, random_vector_384):
        """Test that large timestep might cause energy drift."""
        q0_full = random_vector_384
        p0_full = random_vector_384 * 0.5

        k = 1.0

        def gradient_fn(q):
            return k * q

        def potential(q):
            return 0.5 * k * np.dot(q, q)

        # Large timestep
        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.5, n_steps=20
        )

        drift = mock_integrator.compute_energy_drift(states, potential)

        # Large timestep may have worse conservation
        # Just check it doesn't crash
        assert 'drift' in drift


class TestIntegratorNumericalStability:
    """Test numerical stability."""

    def test_normalized_vectors(self, mock_integrator):
        """Test with normalized vectors."""
        rng = np.random.RandomState(42)
        q0_full = rng.randn(384)
        q0_full = q0_full / np.linalg.norm(q0_full)
        p0_full = rng.randn(384) * 0.1

        def gradient_fn(q):
            return 0.01 * q

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=100
        )

        # Check no NaN or Inf
        for state in states:
            assert not np.any(np.isnan(state.q))
            assert not np.any(np.isnan(state.p))
            assert not np.any(np.isinf(state.q))
            assert not np.any(np.isinf(state.p))

    def test_small_vectors(self, mock_integrator):
        """Test with very small vectors."""
        q0_full = np.ones(384) * 1e-10
        p0_full = np.ones(384) * 1e-10

        def gradient_fn(q):
            return 0.01 * q

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=10
        )

        # Should handle small values
        for state in states:
            assert not np.any(np.isnan(state.q))
            assert not np.any(np.isnan(state.p))

    def test_large_values(self, mock_integrator):
        """Test with large values."""
        rng = np.random.RandomState(42)
        q0_full = rng.randn(384) * 100
        p0_full = rng.randn(384) * 10

        def gradient_fn(q):
            return 0.0001 * q  # Small gradient to prevent explosion

        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.001, n_steps=10
        )

        # Should handle large values with appropriate timestep
        for state in states:
            assert not np.any(np.isnan(state.q))
            assert not np.any(np.isinf(state.q))


# ============================================================================
# Performance Tests
# ============================================================================

class TestIntegratorPerformance:
    """Performance benchmarks for integrator."""

    def test_integration_speed(self, mock_integrator, random_vector_384):
        """Test that integration completes in reasonable time."""
        import time

        q0_full = random_vector_384
        p0_full = np.zeros(384)

        def gradient_fn(q):
            return 0.01 * q

        start = time.perf_counter()
        states = mock_integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=1000
        )
        elapsed = time.perf_counter() - start

        # Should complete 1000 steps in < 1 second
        assert elapsed < 1.0
        assert len(states) == 1001

    def test_multi_scale_integration_speed(self):
        """Test multi-scale integration performance."""
        from HoloLoom.semantic_calculus.integrator import MultiScaleGeometricFlow
        import time

        rng = np.random.RandomState(42)
        projection_matrices = {
            96: rng.randn(16, 96),
            192: rng.randn(16, 192),
            384: rng.randn(16, 384)
        }

        flow = MultiScaleGeometricFlow(projection_matrices)

        q0_full_by_scale = {
            scale: rng.randn(scale) for scale in [96, 192, 384]
        }
        p0_full_by_scale = {
            scale: rng.randn(scale) * 0.1 for scale in [96, 192, 384]
        }

        def make_gradient(scale):
            def gradient(q):
                return 0.01 * q
            return gradient

        gradient_fns = {scale: make_gradient(scale) for scale in [96, 192, 384]}

        start = time.perf_counter()
        result = flow.integrate_all_scales(
            q0_full_by_scale, p0_full_by_scale,
            gradient_fns, dt=0.01, n_steps=100
        )
        elapsed = time.perf_counter() - start

        # Multi-scale integration for 100 steps should be < 0.5 seconds
        assert elapsed < 0.5
        assert 'states_by_scale' in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegratorWithSpectrum:
    """Test integration of GeometricIntegrator with SemanticSpectrum."""

    def test_spectrum_projection_compatibility(self, mock_spectrum, mock_embed_fn):
        """Test that spectrum projections work with integrator."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        # Get learned axes from spectrum
        # The spectrum's internal axes can serve as projection basis
        rng = np.random.RandomState(42)

        # Create a projection matrix from dimension axes
        n_dims = len(mock_spectrum.dimensions)
        emb_dim = 384

        # Build projection matrix from learned axes
        P = np.zeros((n_dims, emb_dim))
        for i, dim in enumerate(mock_spectrum.dimensions):
            if dim.axis is not None:
                P[i, :] = dim.axis
            else:
                P[i, :] = rng.randn(emb_dim)
                P[i, :] = P[i, :] / (np.linalg.norm(P[i, :]) + 1e-10)

        integrator = GeometricIntegrator(P, mass=1.0)

        # Test projection
        test_vec = mock_embed_fn("test")
        projected = integrator.project_to_semantic(test_vec)

        assert projected.shape == (n_dims,)

    def test_integrate_semantic_trajectory(self, mock_spectrum, mock_embed_fn):
        """Test integrating trajectory in semantic space."""
        from HoloLoom.semantic_calculus.integrator import GeometricIntegrator

        rng = np.random.RandomState(42)
        n_dims = len(mock_spectrum.dimensions)
        emb_dim = 384

        # Create orthonormal projection
        P = rng.randn(n_dims, emb_dim)
        P, _ = np.linalg.qr(P.T)
        P = P.T[:n_dims, :]

        integrator = GeometricIntegrator(P, mass=1.0)

        # Initial conditions from embedding
        q0_full = mock_embed_fn("Thompson Sampling")
        p0_full = mock_embed_fn("exploration") * 0.1

        # Simple harmonic potential
        def gradient_fn(q):
            return 0.1 * q

        states = integrator.integrate_trajectory(
            q0_full, p0_full, gradient_fn, dt=0.01, n_steps=50
        )

        assert len(states) == 51
        # Check dimensions match
        assert states[0].q.shape == (n_dims,)


class TestSymplecticProperties:
    """Test symplectic properties of the integrator."""

    def test_phase_space_volume_preservation(self, mock_integrator):
        """Test that phase space volume is approximately preserved."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        # Create a cloud of initial conditions
        rng = np.random.RandomState(42)
        n_particles = 10

        initial_states = []
        for i in range(n_particles):
            q = rng.randn(16) * 0.1
            p = rng.randn(16) * 0.1
            initial_states.append(SemanticState(q=q, p=p, t=0.0))

        # Harmonic gradient
        def gradient_fn(q):
            return 0.1 * q

        # Evolve each particle
        final_states = []
        for state in initial_states:
            dt = 0.01
            current = state
            for _ in range(100):
                current = mock_integrator.stormer_verlet_step(
                    current, gradient_fn, dt
                )
            final_states.append(current)

        # Compute approximate phase space "volume" (using std of positions)
        initial_q_std = np.std([s.q for s in initial_states])
        initial_p_std = np.std([s.p for s in initial_states])
        initial_volume = initial_q_std * initial_p_std

        final_q_std = np.std([s.q for s in final_states])
        final_p_std = np.std([s.p for s in final_states])
        final_volume = final_q_std * final_p_std

        # Volume should be approximately preserved (within 50% for this simple test)
        volume_ratio = final_volume / initial_volume
        assert 0.5 < volume_ratio < 2.0

    def test_time_reversibility(self, mock_integrator):
        """Test that Stormer-Verlet is time-reversible."""
        from HoloLoom.semantic_calculus.integrator import SemanticState

        rng = np.random.RandomState(42)
        q0 = rng.randn(16)
        p0 = rng.randn(16)
        state = SemanticState(q=q0.copy(), p=p0.copy(), t=0.0)

        def gradient_fn(q):
            return 0.1 * q

        dt = 0.01
        n_steps = 50

        # Forward integration
        for _ in range(n_steps):
            state = mock_integrator.stormer_verlet_step(state, gradient_fn, dt)

        # Reverse momentum
        state = SemanticState(q=state.q, p=-state.p, t=state.t)

        # Backward integration
        for _ in range(n_steps):
            state = mock_integrator.stormer_verlet_step(state, gradient_fn, dt)

        # Reverse momentum again
        state = SemanticState(q=state.q, p=-state.p, t=state.t)

        # Should approximately return to initial state
        assert np.allclose(state.q, q0, rtol=1e-3, atol=1e-5)
        assert np.allclose(state.p, p0, rtol=1e-3, atol=1e-5)
