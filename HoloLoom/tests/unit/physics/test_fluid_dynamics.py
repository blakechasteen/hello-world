"""
Tests for Physics - Fluid Dynamics

Tests Navier-Stokes context packing for optimal information flow.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 5
Target: +8 tests for fluid dynamics
"""

import pytest
import numpy as np
from HoloLoom.physics.fluid_dynamics import (
    FluidDynamicsEngine,
    VelocityField,
    PressureField,
    FlowPattern
)


class TestFluidDynamicsInitialization:
    """Test suite for FluidDynamicsEngine initialization."""

    def test_engine_initialization(self):
        """Test engine initialization with default parameters."""
        engine = FluidDynamicsEngine(
            viscosity=0.1,
            density=1.0
        )

        assert engine.viscosity == 0.1
        assert engine.density == 1.0

    def test_velocity_field_creation(self):
        """Test VelocityField object creation."""
        field = VelocityField(
            vx=np.array([1.0, 2.0, 3.0]),
            vy=np.array([0.5, 1.0, 1.5])
        )

        assert len(field.vx) == 3
        assert len(field.vy) == 3


class TestNavierStokesSimulation:
    """Test suite for Navier-Stokes equation simulation."""

    def test_velocity_field_evolution(self):
        """Test velocity field evolves according to Navier-Stokes."""
        engine = FluidDynamicsEngine(viscosity=0.1)

        initial_velocity = VelocityField(
            vx=np.ones(10),
            vy=np.zeros(10)
        )

        # Evolve one timestep
        next_velocity = engine.evolve(initial_velocity, dt=0.01)

        # Velocity should change due to viscosity and pressure
        assert not np.allclose(next_velocity.vx, initial_velocity.vx)

    def test_viscosity_damping(self):
        """Test viscosity damps velocity over time."""
        engine = FluidDynamicsEngine(viscosity=0.5)  # High viscosity

        # Start with high velocity
        velocity = VelocityField(
            vx=np.ones(10) * 10.0,
            vy=np.zeros(10)
        )

        # Evolve multiple timesteps
        for _ in range(10):
            velocity = engine.evolve(velocity, dt=0.1)

        # Velocity should decrease due to viscosity
        assert np.mean(velocity.vx) < 10.0


class TestPressureField:
    """Test suite for pressure field calculations."""

    def test_pressure_from_velocity(self):
        """Test pressure field calculation from velocity divergence."""
        engine = FluidDynamicsEngine()

        velocity = VelocityField(
            vx=np.array([1.0, 2.0, 3.0, 4.0]),
            vy=np.array([0.0, 0.0, 0.0, 0.0])
        )

        pressure = engine.calculate_pressure(velocity)

        # Pressure should be calculated
        assert isinstance(pressure, PressureField)
        assert len(pressure.values) > 0

    def test_incompressibility(self):
        """Test incompressibility condition (div v = 0)."""
        engine = FluidDynamicsEngine()

        # Create incompressible flow
        velocity = VelocityField(
            vx=np.array([1.0, 1.0, 1.0]),  # Uniform flow
            vy=np.array([0.0, 0.0, 0.0])
        )

        divergence = engine.calculate_divergence(velocity)

        # Uniform flow should have zero divergence
        assert np.abs(np.mean(divergence)) < 0.1


class TestFlowPatterns:
    """Test suite for flow pattern detection."""

    def test_detect_vortex(self):
        """Test vortex pattern detection."""
        engine = FluidDynamicsEngine()

        # Create circular flow (vortex)
        theta = np.linspace(0, 2*np.pi, 100)
        vx = -np.sin(theta)  # Circular velocity
        vy = np.cos(theta)

        velocity = VelocityField(vx=vx, vy=vy)

        pattern = engine.detect_flow_pattern(velocity)

        # Should detect some kind of pattern
        assert isinstance(pattern, FlowPattern)

    def test_laminar_vs_turbulent(self):
        """Test distinction between laminar and turbulent flow."""
        engine = FluidDynamicsEngine()

        # Laminar flow: smooth, uniform
        laminar = VelocityField(
            vx=np.ones(100),
            vy=np.zeros(100)
        )

        # Turbulent flow: chaotic, random
        turbulent = VelocityField(
            vx=np.random.randn(100),
            vy=np.random.randn(100)
        )

        laminar_pattern = engine.detect_flow_pattern(laminar)
        turbulent_pattern = engine.detect_flow_pattern(turbulent)

        # Patterns should be different
        assert laminar_pattern.type != turbulent_pattern.type or \
               laminar_pattern.intensity != turbulent_pattern.intensity


class TestContextPacking:
    """Test suite for context packing optimization."""

    def test_pack_information(self):
        """Test packing information into limited context window."""
        engine = FluidDynamicsEngine()

        # Simulate information items with importance scores
        items = [
            ("item_1", 0.9),  # High importance
            ("item_2", 0.7),
            ("item_3", 0.5),
            ("item_4", 0.2)   # Low importance
        ]

        # Pack into limited space
        packed = engine.pack_context(items, capacity=2)

        # Should prioritize high-importance items
        assert len(packed) <= 2
        assert "item_1" in packed  # Highest importance should be included

    def test_flow_based_priority(self):
        """Test that packing prioritizes items in main flow."""
        engine = FluidDynamicsEngine()

        # Items with flow velocities (speed indicates importance/relevance)
        items_with_flow = [
            ("fast_item", 10.0),   # High velocity = important
            ("slow_item", 1.0)     # Low velocity = less important
        ]

        packed = engine.pack_context(items_with_flow, capacity=1)

        # Should pack the high-velocity (important) item
        assert "fast_item" in packed


class TestStreamlines:
    """Test suite for streamline calculations."""

    def test_compute_streamlines(self):
        """Test streamline computation from velocity field."""
        engine = FluidDynamicsEngine()

        velocity = VelocityField(
            vx=np.ones(50),
            vy=np.zeros(50)
        )

        streamlines = engine.compute_streamlines(velocity, num_lines=5)

        # Should compute streamlines
        assert len(streamlines) == 5
        assert all(len(line) > 0 for line in streamlines)


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_zero_viscosity(self):
        """Test behavior with zero viscosity (inviscid flow)."""
        engine = FluidDynamicsEngine(viscosity=0.0)

        velocity = VelocityField(
            vx=np.ones(10) * 5.0,
            vy=np.zeros(10)
        )

        # Evolve with zero viscosity
        next_velocity = engine.evolve(velocity, dt=0.1)

        # With zero viscosity and no pressure gradients, velocity should persist
        # (or change only due to pressure/advection)
        assert isinstance(next_velocity, VelocityField)

    def test_static_field(self):
        """Test behavior with zero velocity everywhere."""
        engine = FluidDynamicsEngine()

        static = VelocityField(
            vx=np.zeros(10),
            vy=np.zeros(10)
        )

        next_velocity = engine.evolve(static, dt=0.1)

        # Static field should remain static
        assert np.allclose(next_velocity.vx, 0.0, atol=1e-6)
        assert np.allclose(next_velocity.vy, 0.0, atol=1e-6)
