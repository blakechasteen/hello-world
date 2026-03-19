"""
Tests for free_energy.py — Active inference bridge over spring dynamics.

Wire 9 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.free_energy import (
    FreeEnergyBridge,
    FreeEnergyState,
    ActionValue,
    create_free_energy_bridge,
)


class TestFreeEnergyComputation:
    def test_zero_energy_at_equilibrium(self):
        """No spring tension = no surprise = zero potential energy."""
        bridge = FreeEnergyBridge()
        state = bridge.compute_free_energy(
            node_activations={"A": 0.5, "B": 0.5},
            node_velocities={"A": 0.0, "B": 0.0},
            edge_potentials=[("A", "B", 0.0)],  # No tension
        )
        assert state.potential_energy == 0.0
        assert state.kinetic_energy == 0.0
        assert state.surprise == 0.0

    def test_tension_creates_surprise(self):
        """Spring tension between nodes creates prediction error."""
        bridge = FreeEnergyBridge()
        state = bridge.compute_free_energy(
            node_activations={"A": 1.0, "B": 0.0},
            node_velocities={"A": 0.0, "B": 0.0},
            edge_potentials=[("A", "B", 0.4)],  # ½k(1.0-0.0)² = 0.4
        )
        assert state.potential_energy == 0.4
        assert state.surprise == 0.4  # Surprise ≈ potential energy

    def test_velocity_creates_kinetic_energy(self):
        bridge = FreeEnergyBridge()
        state = bridge.compute_free_energy(
            node_activations={"A": 0.5},
            node_velocities={"A": 2.0},
            edge_potentials=[],
        )
        assert state.kinetic_energy == pytest.approx(2.0)  # ½ * 2² = 2

    def test_complexity_increases_with_non_uniform_activation(self):
        """Concentrated activation = higher complexity (KL from uniform)."""
        bridge = FreeEnergyBridge()
        # Uniform activation
        state_uniform = bridge.compute_free_energy(
            node_activations={"A": 0.5, "B": 0.5, "C": 0.5},
            node_velocities={"A": 0.0, "B": 0.0, "C": 0.0},
            edge_potentials=[],
        )
        # Concentrated activation
        state_concentrated = bridge.compute_free_energy(
            node_activations={"A": 1.0, "B": 0.0, "C": 0.0},
            node_velocities={"A": 0.0, "B": 0.0, "C": 0.0},
            edge_potentials=[],
        )
        assert state_concentrated.complexity > state_uniform.complexity

    def test_total_is_potential_plus_complexity(self):
        bridge = FreeEnergyBridge()
        state = bridge.compute_free_energy(
            node_activations={"A": 0.8, "B": 0.2},
            node_velocities={"A": 0.0, "B": 0.0},
            edge_potentials=[("A", "B", 0.3)],
        )
        assert state.total_free_energy == pytest.approx(
            state.potential_energy + state.complexity
        )

    def test_prior_precision_scales_complexity(self):
        """Higher prior precision = more costly to deviate from uniform."""
        low_prec = FreeEnergyBridge(prior_precision=0.1)
        high_prec = FreeEnergyBridge(prior_precision=10.0)

        activations = {"A": 1.0, "B": 0.0}
        velocities = {"A": 0.0, "B": 0.0}

        s_low = low_prec.compute_free_energy(activations, velocities, [])
        s_high = high_prec.compute_free_energy(activations, velocities, [])

        assert s_high.complexity > s_low.complexity

    def test_empty_activations(self):
        bridge = FreeEnergyBridge()
        state = bridge.compute_free_energy({}, {}, [])
        assert state.total_free_energy == 0.0

    def test_entropy_higher_for_uniform(self):
        bridge = FreeEnergyBridge()
        s_uniform = bridge.compute_free_energy(
            {"A": 1.0, "B": 1.0, "C": 1.0}, {"A": 0, "B": 0, "C": 0}, []
        )
        s_concentrated = bridge.compute_free_energy(
            {"A": 10.0, "B": 0.01, "C": 0.01}, {"A": 0, "B": 0, "C": 0}, []
        )
        assert s_uniform.entropy > s_concentrated.entropy


class TestActionRanking:
    def test_high_tension_node_ranked_first(self):
        """Node with most spring tension should be most epistemically valuable."""
        bridge = FreeEnergyBridge()
        actions = bridge.rank_actions(
            candidates=["A", "B", "C"],
            node_activations={"A": 0.0, "B": 0.0, "C": 0.0},
            neighbor_potentials={"A": 0.1, "B": 0.9, "C": 0.3},
        )
        assert actions[0].node_id == "B"  # Most tension
        assert actions[0].epistemic_value > actions[1].epistemic_value

    def test_target_node_gets_pragmatic_bonus(self):
        bridge = FreeEnergyBridge()
        actions = bridge.rank_actions(
            candidates=["A", "B"],
            node_activations={"A": 0.0, "B": 0.0},
            neighbor_potentials={"A": 0.5, "B": 0.5},
            target_nodes=["B"],
        )
        # B gets pragmatic bonus, should rank first
        assert actions[0].node_id == "B"
        assert actions[0].pragmatic_value == 1.0
        assert actions[1].pragmatic_value == 0.0

    def test_already_active_node_lower_energy_reduction(self):
        bridge = FreeEnergyBridge()
        actions = bridge.rank_actions(
            candidates=["A", "B"],
            node_activations={"A": 0.9, "B": 0.1},
            neighbor_potentials={"A": 0.5, "B": 0.5},
        )
        # B has more room to activate (0.1 vs 0.9)
        a_reduction = next(a for a in actions if a.node_id == "A").energy_reduction
        b_reduction = next(a for a in actions if a.node_id == "B").energy_reduction
        assert b_reduction > a_reduction

    def test_empty_candidates(self):
        bridge = FreeEnergyBridge()
        actions = bridge.rank_actions([], {}, {})
        assert actions == []


class TestFactory:
    def test_create_bridge(self):
        bridge = create_free_energy_bridge(prior_precision=2.0)
        assert isinstance(bridge, FreeEnergyBridge)
        assert bridge.prior_precision == 2.0

    def test_defaults(self):
        bridge = create_free_energy_bridge()
        assert bridge.prior_precision == 1.0
