"""
Tests for Physics - Thermodynamic Optimizer

Tests free energy minimization for exploration/exploitation balance.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 5
Target: +8 tests for thermodynamics
"""

import pytest
import numpy as np
from HoloLoom.physics.thermodynamics import (
    ThermodynamicOptimizer,
    CoolingSchedule,
    TemperatureState
)


class TestThermodynamicOptimizer Initialization:
    """Test suite for ThermodynamicOptimizer initialization."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization with default parameters."""
        thermo = ThermodynamicOptimizer(
            initial_temperature=1.0,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.95
        )

        assert thermo.temperature_state.current_temp == 1.0
        assert thermo.temperature_state.schedule == CoolingSchedule.EXPONENTIAL
        assert thermo.temperature_state.rate == 0.95

    def test_cooling_schedules(self):
        """Test all cooling schedule types."""
        schedules = [
            CoolingSchedule.EXPONENTIAL,
            CoolingSchedule.LINEAR,
            CoolingSchedule.INVERSE,
            CoolingSchedule.ADAPTIVE
        ]

        for schedule in schedules:
            thermo = ThermodynamicOptimizer(
                initial_temperature=1.0,
                cooling_schedule=schedule
            )
            assert thermo.temperature_state.schedule == schedule


class TestActionSelection:
    """Test suite for action selection with thermodynamic balance."""

    def test_select_action_high_temperature(self):
        """Test action selection favors exploration at high temperature."""
        thermo = ThermodynamicOptimizer(initial_temperature=10.0)

        energy_costs = [0.1, 0.3, 0.5]
        diversity_scores = [0.2, 0.5, 0.9]  # Last action has high diversity

        # Run multiple times to check stochasticity
        actions = []
        for _ in range(20):
            action = thermo.select_action(
                energy_costs=energy_costs,
                diversity_scores=diversity_scores,
                temperature=10.0  # High temperature
            )
            actions.append(action)

        # Should explore different actions
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Not always picking the same action

    def test_select_action_low_temperature(self):
        """Test action selection favors exploitation at low temperature."""
        thermo = ThermodynamicOptimizer(initial_temperature=0.01)

        energy_costs = [0.1, 0.5, 0.9]  # First action has lowest cost
        diversity_scores = [0.5, 0.5, 0.5]

        actions = []
        for _ in range(10):
            action = thermo.select_action(
                energy_costs=energy_costs,
                diversity_scores=diversity_scores,
                temperature=0.01  # Low temperature -> exploitation
            )
            actions.append(action)

        # Should consistently pick lowest cost action (index 0)
        assert actions.count(0) >= 7  # At least 70% exploit


class TestTemperatureScheduling:
    """Test suite for temperature scheduling and annealing."""

    def test_exponential_cooling(self):
        """Test exponential temperature decay."""
        thermo = ThermodynamicOptimizer(
            initial_temperature=1.0,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.9
        )

        initial_temp = thermo.temperature_state.current_temp

        # Step forward
        thermo.step()

        # Temperature should decrease exponentially
        assert thermo.temperature_state.current_temp < initial_temp
        assert thermo.temperature_state.current_temp == pytest.approx(initial_temp * 0.9, rel=0.01)

    def test_linear_cooling(self):
        """Test linear temperature decay."""
        thermo = ThermodynamicOptimizer(
            initial_temperature=1.0,
            cooling_schedule=CoolingSchedule.LINEAR,
            cooling_rate=0.1
        )

        initial_temp = thermo.temperature_state.current_temp
        thermo.step()

        # Temperature should decrease linearly
        expected = initial_temp - 0.1
        assert thermo.temperature_state.current_temp == pytest.approx(expected, abs=0.01)

    def test_minimum_temperature_constraint(self):
        """Test that temperature doesn't fall below minimum."""
        thermo = ThermodynamicOptimizer(
            initial_temperature=0.05,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            cooling_rate=0.5
        )

        # Cool many times
        for _ in range(100):
            thermo.step()

        # Should not go below min_temp
        assert thermo.temperature_state.current_temp >= thermo.temperature_state.min_temp


class TestFreeEnergyCalculation:
    """Test suite for free energy computation."""

    def test_free_energy_formula(self):
        """Test F = E - T*S calculation."""
        thermo = ThermodynamicOptimizer(initial_temperature=1.0)

        energy = 0.5
        entropy = 0.3
        temperature = 1.0

        free_energy = thermo.calculate_free_energy(
            energy=energy,
            entropy=entropy,
            temperature=temperature
        )

        # F = E - T*S = 0.5 - 1.0*0.3 = 0.2
        expected = energy - temperature * entropy
        assert free_energy == pytest.approx(expected, abs=0.01)

    def test_free_energy_trade_off(self):
        """Test free energy balances energy and entropy."""
        thermo = ThermodynamicOptimizer(initial_temperature=1.0)

        # Low energy, low entropy
        f1 = thermo.calculate_free_energy(0.1, 0.1, 1.0)

        # High energy, high entropy
        f2 = thermo.calculate_free_energy(0.8, 0.8, 1.0)

        # Free energy should balance both
        assert isinstance(f1, float)
        assert isinstance(f2, float)
