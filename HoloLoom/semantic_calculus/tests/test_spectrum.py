"""
Tests for SemanticSpectrum
==========================

Comprehensive tests for SemanticSpectrum class including:
    - Initialization and configuration
    - Axis learning (learn_axes)
    - Vector projection (project_vector)
    - Trajectory projection (project_trajectory)
    - Velocity and acceleration computation
    - Semantic force analysis
    - Temporal dynamics (PDE-based evolution)
    - Dominant dimension detection

Run tests:
    pytest HoloLoom/semantic_calculus/tests/test_spectrum.py -v
"""

import pytest
import numpy as np
from typing import List, Dict


# ============================================================================
# Test Imports
# ============================================================================

class TestSpectrumImports:
    """Test SemanticSpectrum imports."""

    def test_import_semantic_spectrum(self):
        """Test SemanticSpectrum import."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum
        assert SemanticSpectrum is not None

    def test_import_from_init(self):
        """Test import from package init."""
        try:
            from HoloLoom.semantic_calculus import SemanticSpectrum
            assert SemanticSpectrum is not None
        except ImportError:
            pytest.skip("SemanticSpectrum not exported from package init")


# ============================================================================
# Initialization Tests
# ============================================================================

class TestSpectrumInitialization:
    """Test SemanticSpectrum initialization."""

    def test_init_default_dimensions(self):
        """Test initialization with default dimensions (STANDARD_DIMENSIONS)."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum()

        assert spectrum is not None
        assert len(spectrum.dimensions) == 16  # Standard dimensions

    def test_init_custom_dimensions(self, sample_dimensions_list):
        """Test initialization with custom dimensions."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)

        assert len(spectrum.dimensions) == len(sample_dimensions_list)

    def test_init_extended_dimensions(self, extended_dimensions):
        """Test initialization with all 228 extended dimensions."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=extended_dimensions)

        assert len(spectrum.dimensions) == 228

    def test_init_empty_dimensions(self):
        """Test initialization with empty dimension list."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=[])

        assert len(spectrum.dimensions) == 0

    def test_init_single_dimension(self, sample_dimension):
        """Test initialization with single dimension."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=[sample_dimension])

        assert len(spectrum.dimensions) == 1


# ============================================================================
# Axis Learning Tests
# ============================================================================

class TestSpectrumAxisLearning:
    """Test SemanticSpectrum.learn_axes() functionality."""

    def test_learn_axes_basic(self, mock_embed_fn, sample_dimensions_list):
        """Test basic axis learning."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)
        spectrum.learn_axes(mock_embed_fn)

        # Check all dimensions have learned axes
        for dim in spectrum.dimensions:
            assert dim.axis is not None
            assert dim.axis.shape == (384,)

    def test_learn_axes_normalized(self, mock_embed_fn, sample_dimensions_list):
        """Test that learned axes are normalized."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)
        spectrum.learn_axes(mock_embed_fn)

        for dim in spectrum.dimensions:
            norm = np.linalg.norm(dim.axis)
            assert np.isclose(norm, 1.0, atol=1e-6), f"Axis for {dim.name} not normalized"

    def test_learn_axes_standard(self, mock_embed_fn, standard_dimensions):
        """Test learning all 16 standard axes."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=standard_dimensions)
        spectrum.learn_axes(mock_embed_fn)

        learned_count = sum(1 for dim in spectrum.dimensions if dim.axis is not None)
        assert learned_count == 16

    def test_learn_axes_partial(self, mock_embed_fn):
        """Test that learn_axes works with partially learned dimensions."""
        from HoloLoom.semantic_calculus.dimensions import (
            SemanticSpectrum,
            SemanticDimension
        )

        # Create dimensions, one with pre-learned axis
        dim1 = SemanticDimension(
            name="Dim1",
            positive_exemplars=["good"],
            negative_exemplars=["bad"]
        )
        dim2 = SemanticDimension(
            name="Dim2",
            positive_exemplars=["happy"],
            negative_exemplars=["sad"],
            axis=np.random.randn(384)  # Pre-set axis
        )

        spectrum = SemanticSpectrum(dimensions=[dim1, dim2])
        spectrum.learn_axes(mock_embed_fn)

        # Both should have axes now
        assert dim1.axis is not None
        assert dim2.axis is not None

    def test_learn_axes_deterministic(self, mock_embed_fn, sample_dimensions_list):
        """Test that axis learning is deterministic."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum1 = SemanticSpectrum(dimensions=sample_dimensions_list)
        spectrum1.learn_axes(mock_embed_fn)

        # Reset axes
        for dim in sample_dimensions_list:
            dim.axis = None

        spectrum2 = SemanticSpectrum(dimensions=sample_dimensions_list)
        spectrum2.learn_axes(mock_embed_fn)

        # Compare axes
        for dim1, dim2 in zip(spectrum1.dimensions, spectrum2.dimensions):
            assert np.allclose(dim1.axis, dim2.axis)


# ============================================================================
# Vector Projection Tests
# ============================================================================

class TestSpectrumVectorProjection:
    """Test SemanticSpectrum.project_vector() functionality."""

    def test_project_vector_basic(self, mock_spectrum, mock_embed_fn):
        """Test basic vector projection."""
        vec = mock_embed_fn("test query")
        projection = mock_spectrum.project_vector(vec)

        assert isinstance(projection, dict)
        assert len(projection) == len(mock_spectrum.dimensions)

    def test_project_vector_keys(self, mock_spectrum, mock_embed_fn):
        """Test that projection has correct dimension keys."""
        vec = mock_embed_fn("test")
        projection = mock_spectrum.project_vector(vec)

        expected_keys = {dim.name for dim in mock_spectrum.dimensions}
        actual_keys = set(projection.keys())

        assert expected_keys == actual_keys

    def test_project_vector_values_type(self, mock_spectrum, mock_embed_fn):
        """Test that projection values are floats."""
        vec = mock_embed_fn("test")
        projection = mock_spectrum.project_vector(vec)

        for dim_name, value in projection.items():
            assert isinstance(value, (int, float)), f"{dim_name} has invalid type"
            assert not np.isnan(value), f"{dim_name} is NaN"

    def test_project_vector_range(self, mock_spectrum, mock_embed_fn):
        """Test projection values are in reasonable range."""
        # Project 50 different vectors
        for i in range(50):
            vec = mock_embed_fn(f"word_{i}")
            projection = mock_spectrum.project_vector(vec)

            for dim_name, value in projection.items():
                # Projections should be in [-2, 2] range for normalized vectors
                assert -3 <= value <= 3, f"{dim_name}: {value} out of range"

    def test_project_zero_vector(self, mock_spectrum):
        """Test projection of zero vector."""
        zero_vec = np.zeros(384)
        projection = mock_spectrum.project_vector(zero_vec)

        # All projections should be 0 or near-0
        for dim_name, value in projection.items():
            assert np.isclose(value, 0.0, atol=1e-6), f"{dim_name} non-zero for zero input"

    def test_project_axis_aligned_vector(self, mock_spectrum):
        """Test projection of vector aligned with first axis."""
        # Get first dimension's axis
        first_dim = mock_spectrum.dimensions[0]

        projection = mock_spectrum.project_vector(first_dim.axis)

        # Projection onto self should be 1.0
        assert np.isclose(projection[first_dim.name], 1.0, atol=1e-6)

    def test_project_consistency(self, mock_spectrum, mock_embed_fn):
        """Test that same vector gives same projection."""
        vec = mock_embed_fn("consistent test")

        proj1 = mock_spectrum.project_vector(vec)
        proj2 = mock_spectrum.project_vector(vec)

        for dim_name in proj1:
            assert np.isclose(proj1[dim_name], proj2[dim_name])


# ============================================================================
# Trajectory Projection Tests
# ============================================================================

class TestSpectrumTrajectoryProjection:
    """Test SemanticSpectrum.project_trajectory() functionality."""

    def test_project_trajectory_basic(self, mock_spectrum, sample_trajectory):
        """Test basic trajectory projection."""
        projections = mock_spectrum.project_trajectory(sample_trajectory)

        assert isinstance(projections, dict)
        assert len(projections) == len(mock_spectrum.dimensions)

    def test_project_trajectory_shape(self, mock_spectrum, sample_trajectory):
        """Test trajectory projection shape."""
        projections = mock_spectrum.project_trajectory(sample_trajectory)

        n_steps = len(sample_trajectory)
        for dim_name, values in projections.items():
            assert isinstance(values, np.ndarray)
            assert len(values) == n_steps, f"{dim_name} has wrong length"

    def test_project_trajectory_values(self, mock_spectrum, sample_trajectory):
        """Test trajectory projection values are valid."""
        projections = mock_spectrum.project_trajectory(sample_trajectory)

        for dim_name, values in projections.items():
            assert not np.any(np.isnan(values)), f"NaN in {dim_name}"
            assert not np.any(np.isinf(values)), f"Inf in {dim_name}"

    def test_project_trajectory_consistency_with_vector(
        self, mock_spectrum, sample_trajectory
    ):
        """Test trajectory projection matches individual vector projections."""
        traj_proj = mock_spectrum.project_trajectory(sample_trajectory)

        # Check first and last positions
        for idx in [0, -1]:
            vec_proj = mock_spectrum.project_vector(sample_trajectory[idx])

            for dim_name in vec_proj:
                expected = vec_proj[dim_name]
                actual = traj_proj[dim_name][idx]
                assert np.isclose(expected, actual, atol=1e-6), \
                    f"Mismatch at idx {idx} for {dim_name}"

    def test_project_single_step_trajectory(self, mock_spectrum, mock_embed_fn):
        """Test trajectory with single step."""
        single_step = mock_embed_fn("test").reshape(1, -1)
        projections = mock_spectrum.project_trajectory(single_step)

        for dim_name, values in projections.items():
            assert len(values) == 1

    def test_project_long_trajectory(self, mock_spectrum, large_trajectory):
        """Test projection of long trajectory (performance)."""
        import time

        start = time.time()
        projections = mock_spectrum.project_trajectory(large_trajectory[:100])
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Trajectory projection took {elapsed:.2f}s"


# ============================================================================
# Velocity Computation Tests
# ============================================================================

class TestSpectrumVelocity:
    """Test SemanticSpectrum.compute_spectrum_velocity() functionality."""

    def test_compute_velocity_basic(self, mock_spectrum, sample_trajectory):
        """Test basic velocity computation."""
        velocity = mock_spectrum.compute_spectrum_velocity(sample_trajectory)

        assert isinstance(velocity, dict)
        assert len(velocity) == len(mock_spectrum.dimensions)

    def test_compute_velocity_shape(self, mock_spectrum, sample_trajectory):
        """Test velocity shape matches trajectory."""
        velocity = mock_spectrum.compute_spectrum_velocity(sample_trajectory)

        n_steps = len(sample_trajectory)
        for dim_name, values in velocity.items():
            assert len(values) == n_steps, f"{dim_name} velocity has wrong length"

    def test_compute_velocity_dt(self, mock_spectrum, sample_trajectory):
        """Test velocity with different dt values."""
        vel_dt1 = mock_spectrum.compute_spectrum_velocity(sample_trajectory, dt=1.0)
        vel_dt2 = mock_spectrum.compute_spectrum_velocity(sample_trajectory, dt=2.0)

        # Velocity with dt=2 should be half of dt=1 (scaled by time)
        for dim_name in vel_dt1:
            # Check ratio at middle point (avoid boundary effects)
            mid = len(sample_trajectory) // 2
            ratio = vel_dt1[dim_name][mid] / (vel_dt2[dim_name][mid] + 1e-10)
            # Allow some tolerance due to finite differences
            assert 1.5 < ratio < 2.5 or abs(vel_dt2[dim_name][mid]) < 1e-6

    def test_compute_velocity_stationary(self, mock_spectrum, mock_embed_fn):
        """Test velocity for stationary trajectory (same point repeated)."""
        # Create stationary trajectory
        point = mock_embed_fn("stationary")
        stationary = np.tile(point, (5, 1))

        velocity = mock_spectrum.compute_spectrum_velocity(stationary)

        # Velocity should be near zero (except boundaries due to finite diff)
        for dim_name, values in velocity.items():
            # Check middle points
            mid_velocities = values[1:-1]
            assert np.allclose(mid_velocities, 0, atol=1e-6), \
                f"Non-zero velocity for stationary {dim_name}"

    def test_compute_velocity_linear(self, mock_spectrum):
        """Test velocity for linear trajectory."""
        rng = np.random.RandomState(42)

        # Create linear trajectory
        start = rng.randn(384)
        start = start / np.linalg.norm(start)
        direction = rng.randn(384)
        direction = direction / np.linalg.norm(direction) * 0.01  # Small steps

        trajectory = np.array([start + i * direction for i in range(10)])

        velocity = mock_spectrum.compute_spectrum_velocity(trajectory, dt=1.0)

        # Velocity should be approximately constant (linear motion)
        for dim_name, values in velocity.items():
            mid_velocities = values[1:-1]
            if np.std(mid_velocities) > 1e-6:  # Only if non-trivial velocity
                std_ratio = np.std(mid_velocities) / (np.mean(np.abs(mid_velocities)) + 1e-10)
                assert std_ratio < 0.5, f"Non-constant velocity for linear {dim_name}"


# ============================================================================
# Acceleration Computation Tests
# ============================================================================

class TestSpectrumAcceleration:
    """Test SemanticSpectrum.compute_spectrum_acceleration() functionality."""

    def test_compute_acceleration_basic(self, mock_spectrum, sample_trajectory):
        """Test basic acceleration computation."""
        accel = mock_spectrum.compute_spectrum_acceleration(sample_trajectory)

        assert isinstance(accel, dict)
        assert len(accel) == len(mock_spectrum.dimensions)

    def test_compute_acceleration_shape(self, mock_spectrum, sample_trajectory):
        """Test acceleration shape matches trajectory."""
        accel = mock_spectrum.compute_spectrum_acceleration(sample_trajectory)

        n_steps = len(sample_trajectory)
        for dim_name, values in accel.items():
            assert len(values) == n_steps, f"{dim_name} acceleration has wrong length"

    def test_compute_acceleration_constant_velocity(self, mock_spectrum):
        """Test acceleration for constant velocity (should be near zero)."""
        rng = np.random.RandomState(42)

        # Linear trajectory = constant velocity = zero acceleration
        start = rng.randn(384)
        start = start / np.linalg.norm(start)
        direction = rng.randn(384)
        direction = direction / np.linalg.norm(direction) * 0.01

        trajectory = np.array([start + i * direction for i in range(10)])

        accel = mock_spectrum.compute_spectrum_acceleration(trajectory, dt=1.0)

        # Acceleration should be near zero (except boundary effects)
        for dim_name, values in accel.items():
            mid_accels = values[2:-2]  # Avoid boundaries
            assert np.allclose(mid_accels, 0, atol=0.1), \
                f"Non-zero acceleration for constant velocity {dim_name}"


# ============================================================================
# Semantic Force Analysis Tests
# ============================================================================

class TestSemanticForceAnalysis:
    """Test SemanticSpectrum.analyze_semantic_forces() functionality."""

    def test_analyze_forces_basic(self, mock_spectrum, sample_trajectory):
        """Test basic force analysis."""
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory)

        assert isinstance(analysis, dict)

    def test_analyze_forces_keys(self, mock_spectrum, sample_trajectory):
        """Test that analysis contains expected keys."""
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory)

        expected_keys = [
            'projections', 'velocities', 'accelerations',
            'dominant_velocity', 'dominant_force', 'distances'
        ]

        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"

    def test_analyze_forces_dominant_velocity(self, mock_spectrum, sample_trajectory):
        """Test dominant velocity extraction."""
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory)

        dominant_vel = analysis['dominant_velocity']

        # Should be a list of tuples (dimension_name, magnitude)
        assert isinstance(dominant_vel, list)
        if len(dominant_vel) > 0:
            assert isinstance(dominant_vel[0], tuple)
            assert len(dominant_vel[0]) == 2

    def test_analyze_forces_ordering(self, mock_spectrum, sample_trajectory):
        """Test that dominant dimensions are ordered by magnitude."""
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory)

        dominant_vel = analysis['dominant_velocity']

        # Check ordering (highest first)
        for i in range(len(dominant_vel) - 1):
            assert abs(dominant_vel[i][1]) >= abs(dominant_vel[i + 1][1])


# ============================================================================
# Dominant Dimension Tests
# ============================================================================

class TestDominantDimensions:
    """Test SemanticSpectrum.get_dominant_dimensions() functionality."""

    def test_get_dominant_basic(self, mock_spectrum, mock_embed_fn):
        """Test basic dominant dimension extraction."""
        vec = mock_embed_fn("test")
        projection = mock_spectrum.project_vector(vec)

        dominant = mock_spectrum.get_dominant_dimensions(projection, top_k=3)

        assert isinstance(dominant, list)
        assert len(dominant) <= 3

    def test_get_dominant_ordering(self, mock_spectrum, mock_embed_fn):
        """Test that dominant dimensions are ordered correctly."""
        vec = mock_embed_fn("strongly positive valence")
        projection = mock_spectrum.project_vector(vec)

        dominant = mock_spectrum.get_dominant_dimensions(projection, top_k=5)

        # Verify ordering by absolute value
        for i in range(len(dominant) - 1):
            assert abs(dominant[i][1]) >= abs(dominant[i + 1][1])

    def test_get_dominant_top_k_limit(self, mock_spectrum, mock_embed_fn):
        """Test top_k parameter limits results."""
        vec = mock_embed_fn("test")
        projection = mock_spectrum.project_vector(vec)

        for k in [1, 2, 3, 5]:
            dominant = mock_spectrum.get_dominant_dimensions(projection, top_k=k)
            assert len(dominant) <= k


# ============================================================================
# Temporal Dynamics Tests
# ============================================================================

class TestTemporalDynamics:
    """Test SemanticSpectrum temporal dynamics functionality."""

    def test_enable_temporal_dynamics(self, mock_spectrum):
        """Test enabling temporal dynamics."""
        try:
            mock_spectrum.enable_temporal_dynamics(pde_type="heat", dt=0.01)

            # The attribute is _semantic_flow (private), not temporal_flow
            assert hasattr(mock_spectrum, '_semantic_flow')
            assert mock_spectrum._semantic_flow is not None
        except (ImportError, AttributeError):
            pytest.skip("Temporal dynamics require PDE solvers")

    def test_evolve_basic(self, mock_spectrum, mock_embed_fn):
        """Test basic evolution."""
        try:
            mock_spectrum.enable_temporal_dynamics(pde_type="heat", dt=0.01)

            # Create initial projection
            vec = mock_embed_fn("initial state")
            initial = mock_spectrum.project_vector(vec)

            evolved = mock_spectrum.evolve(initial, steps=10)

            assert isinstance(evolved, dict)
            assert len(evolved) == len(initial)
        except (ImportError, AttributeError):
            pytest.skip("Temporal dynamics require PDE solvers")

    def test_evolve_changes_state(self, mock_spectrum, mock_embed_fn):
        """Test that evolution changes state over time."""
        try:
            mock_spectrum.enable_temporal_dynamics(pde_type="heat", dt=0.1)

            vec = mock_embed_fn("high energy state")
            initial = mock_spectrum.project_vector(vec)

            evolved = mock_spectrum.evolve(initial, steps=100)

            # At least some dimensions should change
            changed = sum(
                1 for dim in initial
                if not np.isclose(initial[dim], evolved[dim], atol=0.01)
            )

            assert changed > 0, "No dimensions changed during evolution"
        except (ImportError, AttributeError):
            pytest.skip("Temporal dynamics require PDE solvers")


# ============================================================================
# Edge Cases
# ============================================================================

class TestSpectrumEdgeCases:
    """Test edge cases for SemanticSpectrum."""

    def test_empty_spectrum(self, mock_embed_fn):
        """Test operations on empty spectrum."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=[])
        # Must call learn_axes() first, even for empty spectrum
        spectrum.learn_axes(mock_embed_fn)

        vec = np.random.randn(384)
        projection = spectrum.project_vector(vec)

        assert projection == {}  # Empty dict for empty spectrum

    def test_single_dimension_spectrum(self, sample_dimension, mock_embed_fn):
        """Test spectrum with single dimension."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=[sample_dimension])
        spectrum.learn_axes(mock_embed_fn)

        vec = mock_embed_fn("test")
        projection = spectrum.project_vector(vec)

        assert len(projection) == 1

    def test_projection_with_unlearned_axes(self, sample_dimensions_list):
        """Test projection before axes are learned."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)

        vec = np.random.randn(384)

        # Behavior depends on implementation - might return zeros or raise
        try:
            projection = spectrum.project_vector(vec)
            # If it works, projections should be 0 or handle gracefully
            for value in projection.values():
                assert value == 0 or np.isnan(value)
        except (ValueError, TypeError, AttributeError):
            # Expected if axes must be learned first
            pass

    def test_trajectory_single_point(self, mock_spectrum, mock_embed_fn):
        """Test trajectory with single point."""
        single = mock_embed_fn("single").reshape(1, -1)

        proj = mock_spectrum.project_trajectory(single)
        for values in proj.values():
            assert len(values) == 1

    def test_very_high_dimensional_projection(self, mock_embed_fn):
        """Test with 244 dimensions."""
        from HoloLoom.semantic_calculus.dimensions import (
            SemanticSpectrum,
            EXTENDED_244_DIMENSIONS
        )

        spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS[:50])
        spectrum.learn_axes(mock_embed_fn)

        vec = mock_embed_fn("test")
        projection = spectrum.project_vector(vec)

        assert len(projection) == 50


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestSpectrumNumericalStability:
    """Test numerical stability of SemanticSpectrum."""

    def test_projection_small_input(self, mock_spectrum):
        """Test projection with very small input values."""
        small_vec = np.ones(384) * 1e-15

        projection = mock_spectrum.project_vector(small_vec)

        for value in projection.values():
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_projection_large_input(self, mock_spectrum):
        """Test projection with very large input values."""
        large_vec = np.ones(384) * 1e10

        projection = mock_spectrum.project_vector(large_vec)

        for value in projection.values():
            assert not np.isnan(value)

    def test_velocity_numerical_precision(self, mock_spectrum):
        """Test velocity computation numerical precision."""
        rng = np.random.RandomState(42)

        # Nearly identical consecutive points
        base = rng.randn(384)
        base = base / np.linalg.norm(base)

        trajectory = np.array([base + 1e-10 * rng.randn(384) for _ in range(5)])

        velocity = mock_spectrum.compute_spectrum_velocity(trajectory, dt=1.0)

        # Should not have NaN despite tiny differences
        for values in velocity.values():
            assert not np.any(np.isnan(values))


# ============================================================================
# Performance Tests
# ============================================================================

class TestSpectrumPerformance:
    """Performance tests for SemanticSpectrum."""

    def test_projection_speed(self, mock_spectrum, mock_embed_fn):
        """Test vector projection speed."""
        import time

        vec = mock_embed_fn("test")

        start = time.time()
        for _ in range(1000):
            mock_spectrum.project_vector(vec)
        elapsed = time.time() - start

        # Should complete 1000 projections in under 1 second
        assert elapsed < 1.0, f"1000 projections took {elapsed:.2f}s"

    def test_trajectory_projection_speed(self, mock_spectrum, large_trajectory):
        """Test trajectory projection speed."""
        import time

        start = time.time()
        mock_spectrum.project_trajectory(large_trajectory[:100])
        elapsed = time.time() - start

        # Should project 100 points in under 2 seconds
        assert elapsed < 2.0, f"100-point trajectory took {elapsed:.2f}s"

    def test_force_analysis_speed(self, mock_spectrum, sample_trajectory):
        """Test semantic force analysis speed."""
        import time

        start = time.time()
        for _ in range(100):
            mock_spectrum.analyze_semantic_forces(sample_trajectory)
        elapsed = time.time() - start

        # Should complete 100 analyses in under 5 seconds
        assert elapsed < 5.0, f"100 force analyses took {elapsed:.2f}s"


# ============================================================================
# Integration Tests
# ============================================================================

class TestSpectrumIntegration:
    """Integration tests for SemanticSpectrum with other components."""

    def test_spectrum_with_trajectory_analysis(
        self, mock_spectrum, sample_trajectory_with_shift
    ):
        """Test full trajectory analysis workflow."""
        # Project trajectory
        projections = mock_spectrum.project_trajectory(sample_trajectory_with_shift)

        # Compute velocities
        velocities = mock_spectrum.compute_spectrum_velocity(
            sample_trajectory_with_shift, dt=1.0
        )

        # Compute accelerations
        accelerations = mock_spectrum.compute_spectrum_acceleration(
            sample_trajectory_with_shift, dt=1.0
        )

        # Full analysis
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory_with_shift)

        # Verify consistency
        assert len(projections) == len(velocities) == len(accelerations)
        assert 'dominant_velocity' in analysis

    def test_spectrum_semantic_shift_detection(
        self, mock_spectrum, sample_trajectory_with_shift
    ):
        """Test that spectrum can detect semantic shifts."""
        analysis = mock_spectrum.analyze_semantic_forces(sample_trajectory_with_shift)

        # Get velocities at shift point (around index 5)
        dominant_vel = analysis['dominant_velocity']

        # Should detect some activity (non-zero velocities)
        assert len(dominant_vel) > 0

    def test_spectrum_with_flow_calculus(self, mock_embed_fn, standard_dimensions):
        """Test spectrum integration with flow calculus."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        try:
            from HoloLoom.semantic_calculus.flow_calculus import SemanticFlowCalculus
        except ImportError:
            pytest.skip("Flow calculus not available")

        spectrum = SemanticSpectrum(dimensions=standard_dimensions)
        spectrum.learn_axes(mock_embed_fn)

        # Should work together without errors
        vec = mock_embed_fn("test integration")
        proj = spectrum.project_vector(vec)

        assert len(proj) == 16


# ============================================================================
# Shift Detection Tests
# ============================================================================

class TestShiftDetection:
    """Test semantic shift detection capabilities."""

    def test_detect_topic_change(self, mock_embed_fn_with_semantic_structure):
        """Test detection of topic/semantic shift."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum()
        spectrum.learn_axes(mock_embed_fn_with_semantic_structure)

        # Create trajectory with topic change
        warmth_words = ["warm", "friendly", "kind", "loving", "caring"]
        cold_words = ["cold", "distant", "hostile", "unfriendly", "aloof"]

        embeddings = (
            [mock_embed_fn_with_semantic_structure(w) for w in warmth_words] +
            [mock_embed_fn_with_semantic_structure(w) for w in cold_words]
        )
        trajectory = np.array(embeddings)

        # Analyze
        analysis = spectrum.analyze_semantic_forces(trajectory, dt=1.0)

        # Should detect change at transition point
        velocities = analysis['velocities']

        # Find maximum velocity magnitude at transition (around index 5)
        warmth_vel = velocities.get('Warmth', np.zeros(len(trajectory)))

        # Velocity should be highest around transition
        mid_velocity = abs(warmth_vel[4]) + abs(warmth_vel[5])

        # Should be larger than early/late velocities
        early_velocity = abs(warmth_vel[1]) + abs(warmth_vel[2])

        assert mid_velocity > early_velocity * 0.5 or mid_velocity > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
