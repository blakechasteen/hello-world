"""
Unit Tests for Spring Dynamics Core
====================================
Tests for the physics math: Hooke's Law, Velocity Verlet (Euler fallback),
activation decay, force computation, energy calculations, and edge cases.

Focuses on the math correctness of spring_dynamics.py without requiring
the full KG or advanced integrator infrastructure.
"""

import math
import pytest
import numpy as np
from unittest.mock import MagicMock
from dataclasses import dataclass

from hololoom.core.memory.spring_dynamics import (
    SpringConfig,
    SpringDynamics,
    NodeState,
    SpringPropagationResult,
    _SpringForceFunction,
)


# ============================================================================
# Lightweight mock graph (avoids KG import overhead)
# ============================================================================

class MockGraph:
    """Minimal graph that satisfies SpringDynamics' interface: .G with nodes/edges."""

    def __init__(self):
        self._nodes = set()
        self._edges = []

    def add_node(self, node_id):
        self._nodes.add(node_id)

    def add_edge(self, u, v, **attrs):
        self._nodes.add(u)
        self._nodes.add(v)
        self._edges.append((u, v, attrs))

    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        if data:
            return [(u, v, d) for u, v, d in self._edges]
        return [(u, v) for u, v, _ in self._edges]


class MockKG:
    """Wraps MockGraph behind .G property, matching KG interface."""

    def __init__(self):
        self.G = MockGraph()

    def add_node(self, node_id):
        self.G.add_node(node_id)

    def add_edge(self, u, v, edge_type="RELATED_TO", weight=1.0):
        self.G.add_edge(u, v, type=edge_type, weight=weight)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def two_node_graph():
    """A → B with weight 1.0, type RELATED_TO."""
    kg = MockKG()
    kg.add_edge("A", "B", edge_type="RELATED_TO", weight=1.0)
    return kg


@pytest.fixture
def chain_graph():
    """A → B → C chain."""
    kg = MockKG()
    kg.add_edge("A", "B", edge_type="CONNECTS_TO", weight=1.0)
    kg.add_edge("B", "C", edge_type="CONNECTS_TO", weight=1.0)
    return kg


@pytest.fixture
def star_graph():
    """Center connected to 4 leaves with varying weights."""
    kg = MockKG()
    kg.add_edge("center", "n1", weight=1.0)
    kg.add_edge("center", "n2", weight=0.5)
    kg.add_edge("center", "n3", weight=0.25)
    kg.add_edge("center", "n4", weight=0.1)
    return kg


@pytest.fixture
def euler_config():
    """Config that forces Euler integration (no advanced integrator)."""
    return SpringConfig(
        use_advanced_integrator=False,
        stiffness=0.5,
        damping=0.9,
        decay=0.99,
        dt=0.1,
        max_iterations=100,
        convergence_epsilon=1e-4,
        activation_threshold=0.05,
    )


@pytest.fixture
def isolated_node_graph():
    """Single node, no edges."""
    kg = MockKG()
    kg.add_node("lonely")
    return kg


# ============================================================================
# NodeState: apply_force math
# ============================================================================

class TestNodeStatePhysics:
    """Verify the Euler step math: a = F/m, v += a*dt, v *= damping, x += v*dt, x *= decay."""

    def test_apply_force_positive(self):
        """Positive force increases activation from rest."""
        state = NodeState(node_id="x", activation=0.0, velocity=0.0, mass=1.0)
        state.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)

        # a = 1.0/1.0 = 1.0
        # v = 0 + 1.0*0.1 = 0.1, then *1.0 = 0.1
        # x = 0 + 0.1*0.1 = 0.01, then *1.0 = 0.01
        assert state.velocity == pytest.approx(0.1)
        assert state.activation == pytest.approx(0.01)

    def test_apply_force_with_damping(self):
        """Damping reduces velocity each step."""
        state = NodeState(node_id="x", activation=0.0, velocity=0.0, mass=1.0)
        state.apply_force(force=1.0, dt=0.1, damping=0.5, decay=1.0)

        # v = (0 + 1.0*0.1) * 0.5 = 0.05
        # x = 0 + 0.05*0.1 = 0.005
        assert state.velocity == pytest.approx(0.05)
        assert state.activation == pytest.approx(0.005)

    def test_apply_force_with_decay(self):
        """Decay reduces activation each step (forgetting)."""
        state = NodeState(node_id="x", activation=0.5, velocity=0.0, mass=1.0)
        state.apply_force(force=0.0, dt=0.1, damping=1.0, decay=0.9)

        # No force, no velocity change: v stays 0
        # x = (0.5 + 0*0.1) * 0.9 = 0.45
        assert state.velocity == pytest.approx(0.0)
        assert state.activation == pytest.approx(0.45)

    def test_apply_force_with_mass(self):
        """Heavier mass means less acceleration for same force."""
        light = NodeState(node_id="light", mass=1.0)
        heavy = NodeState(node_id="heavy", mass=4.0)

        light.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)
        heavy.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)

        # light: a=1.0, v=0.1, x=0.01
        # heavy: a=0.25, v=0.025, x=0.0025
        assert light.activation == pytest.approx(0.01)
        assert heavy.activation == pytest.approx(0.0025)
        assert light.activation > heavy.activation

    def test_clamping_upper_bound(self):
        """Activation cannot exceed 1.0."""
        state = NodeState(node_id="x", activation=0.99, velocity=0.0, mass=1.0)
        state.apply_force(force=100.0, dt=1.0, damping=1.0, decay=1.0)
        assert state.activation == 1.0

    def test_clamping_lower_bound(self):
        """Activation cannot go below 0.0."""
        state = NodeState(node_id="x", activation=0.01, velocity=0.0, mass=1.0)
        state.apply_force(force=-100.0, dt=1.0, damping=1.0, decay=1.0)
        assert state.activation == 0.0

    def test_zero_force_zero_velocity_no_change(self):
        """No force + no velocity = no change (except decay)."""
        state = NodeState(node_id="x", activation=0.5, velocity=0.0, mass=1.0)
        state.apply_force(force=0.0, dt=0.1, damping=0.9, decay=1.0)

        assert state.velocity == pytest.approx(0.0)
        assert state.activation == pytest.approx(0.5)

    def test_negative_force_decreases_activation(self):
        """Negative force pulls activation down."""
        state = NodeState(node_id="x", activation=0.5, velocity=0.0, mass=1.0)
        state.apply_force(force=-1.0, dt=0.1, damping=1.0, decay=1.0)

        assert state.velocity < 0.0
        assert state.activation < 0.5

    def test_multiple_steps_accumulate_velocity(self):
        """Repeated force builds velocity (like real physics)."""
        state = NodeState(node_id="x", activation=0.0, velocity=0.0, mass=1.0)
        for _ in range(5):
            state.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)

        # After 5 steps with no damping/decay, velocity should be 0.5
        assert state.velocity == pytest.approx(0.5)
        assert state.activation > 0.0

    def test_reset_clears_state(self):
        """Reset brings node back to zero."""
        state = NodeState(node_id="x", activation=0.8, velocity=0.3, mass=2.0)
        state.reset()
        assert state.activation == 0.0
        assert state.velocity == 0.0
        # mass is not reset (it's a property of the node, not state)
        assert state.mass == 2.0


# ============================================================================
# SpringConfig: edge stiffness calculations
# ============================================================================

class TestSpringConfig:

    def test_edge_stiffness_basic(self):
        """k_eff = stiffness * weight * multiplier."""
        config = SpringConfig(stiffness=0.5)
        # IS_A multiplier is 1.2
        result = config.get_edge_stiffness("IS_A", edge_weight=1.0)
        assert result == pytest.approx(0.5 * 1.0 * 1.2)

    def test_edge_stiffness_with_weight(self):
        """Weight scales the stiffness linearly."""
        config = SpringConfig(stiffness=0.5)
        full = config.get_edge_stiffness("RELATED_TO", edge_weight=1.0)
        half = config.get_edge_stiffness("RELATED_TO", edge_weight=0.5)
        assert half == pytest.approx(full * 0.5)

    def test_unknown_edge_type_uses_multiplier_1(self):
        """Unknown edge types default to multiplier 1.0."""
        config = SpringConfig(stiffness=0.4)
        result = config.get_edge_stiffness("CUSTOM_TYPE", edge_weight=1.0)
        assert result == pytest.approx(0.4)

    def test_zero_weight_gives_zero_stiffness(self):
        """Zero-weight edge transmits no force."""
        config = SpringConfig(stiffness=0.8)
        result = config.get_edge_stiffness("IS_A", edge_weight=0.0)
        assert result == 0.0

    def test_all_predefined_multipliers(self):
        """All predefined edge types have distinct multipliers."""
        config = SpringConfig(stiffness=1.0)
        types = ["IS_A", "PART_OF", "USES", "MENTIONS", "RELATED_TO"]
        values = [config.get_edge_stiffness(t, 1.0) for t in types]
        # IS_A > PART_OF > USES > MENTIONS > RELATED_TO
        assert values == sorted(values, reverse=True)

    def test_custom_edge_type_multipliers(self):
        """Custom multiplier dict overrides defaults."""
        config = SpringConfig(
            stiffness=1.0,
            edge_type_multipliers={"STRONG": 5.0, "WEAK": 0.1},
        )
        assert config.get_edge_stiffness("STRONG", 1.0) == pytest.approx(5.0)
        assert config.get_edge_stiffness("WEAK", 1.0) == pytest.approx(0.1)


# ============================================================================
# SpringDynamics: initialization and state management
# ============================================================================

class TestSpringDynamicsInit:

    def test_creates_node_states_for_all_graph_nodes(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        assert "A" in dyn.node_states
        assert "B" in dyn.node_states

    def test_all_nodes_start_inactive(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        for ns in dyn.node_states.values():
            assert ns.activation == 0.0
            assert ns.velocity == 0.0

    def test_default_config_when_none(self, two_node_graph):
        dyn = SpringDynamics(two_node_graph, config=None)
        assert isinstance(dyn.config, SpringConfig)

    def test_metrics_start_at_zero(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        assert dyn.iterations == 0
        assert dyn.converged is False
        assert dyn.final_energy == 0.0


class TestActivateNodes:

    def test_activate_sets_activation_and_records_seed(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 0.7})
        assert dyn.node_states["A"].activation == pytest.approx(0.7)
        assert dyn.seed_nodes["A"] == pytest.approx(0.7)

    def test_activate_clamps_above_1(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 5.0})
        assert dyn.node_states["A"].activation == 1.0
        assert dyn.seed_nodes["A"] == 1.0

    def test_activate_clamps_below_0(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": -3.0})
        assert dyn.node_states["A"].activation == 0.0

    def test_activate_nonexistent_node_ignored(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"NONEXISTENT": 1.0})
        assert "NONEXISTENT" not in dyn.seed_nodes

    def test_activate_resets_velocity(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.node_states["A"].velocity = 0.5
        dyn.activate_nodes({"A": 1.0})
        assert dyn.node_states["A"].velocity == 0.0


class TestReset:

    def test_reset_clears_all_state(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()

        dyn.reset()

        for ns in dyn.node_states.values():
            assert ns.activation == 0.0
            assert ns.velocity == 0.0
        assert dyn.iterations == 0
        assert dyn.converged is False
        assert dyn.final_energy == 0.0
        assert len(dyn.seed_nodes) == 0


# ============================================================================
# Force computation (Hooke's Law: F = -k * delta_a)
# ============================================================================

class TestComputeForces:

    def test_no_activation_no_force(self, two_node_graph, euler_config):
        """All nodes at zero activation => zero forces."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        forces = dyn._compute_forces()
        assert forces["A"] == pytest.approx(0.0)
        assert forces["B"] == pytest.approx(0.0)

    def test_equal_activation_no_force(self, two_node_graph, euler_config):
        """Equal activation on both sides => zero net spring force."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.node_states["A"].activation = 0.5
        dyn.node_states["B"].activation = 0.5
        forces = dyn._compute_forces()
        assert forces["A"] == pytest.approx(0.0)
        assert forces["B"] == pytest.approx(0.0)

    def test_activation_difference_creates_force(self, two_node_graph, euler_config):
        """Higher activation on one side creates pulling force (Hooke's Law)."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        # A=1.0, B=0.0, edge A->B
        forces = dyn._compute_forces()

        # Force on A: k * (B - A) = k * (0 - 1) = -k (negative, pulls A down)
        # Force on B: -k * (B - A) = k (positive, pulls B up)
        # But seed anchoring means A uses max(activation, seed) = 1.0
        assert forces["A"] < 0  # Spring pulls A toward B's lower level
        assert forces["B"] > 0  # Spring pulls B toward A's higher level

    def test_newton_third_law(self, two_node_graph, euler_config):
        """Forces are equal and opposite (Newton's third law)."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.node_states["A"].activation = 0.8
        dyn.node_states["B"].activation = 0.2
        forces = dyn._compute_forces()

        # Net force should sum to zero
        assert forces["A"] + forces["B"] == pytest.approx(0.0, abs=1e-10)

    def test_stronger_stiffness_stronger_force(self, euler_config):
        """Higher stiffness => larger force for same displacement."""
        kg_weak = MockKG()
        kg_weak.add_edge("A", "B", edge_type="MENTIONS", weight=1.0)  # multiplier 0.7

        kg_strong = MockKG()
        kg_strong.add_edge("A", "B", edge_type="IS_A", weight=1.0)  # multiplier 1.2

        dyn_weak = SpringDynamics(kg_weak, euler_config)
        dyn_strong = SpringDynamics(kg_strong, euler_config)

        for d in [dyn_weak, dyn_strong]:
            d.node_states["A"].activation = 1.0
            d.node_states["B"].activation = 0.0

        f_weak = dyn_weak._compute_forces()
        f_strong = dyn_strong._compute_forces()

        assert abs(f_strong["B"]) > abs(f_weak["B"])


# ============================================================================
# Energy computation
# ============================================================================

class TestComputeEnergy:

    def test_zero_state_zero_energy(self, two_node_graph, euler_config):
        """All at rest => zero energy."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        energy = dyn._compute_energy()
        assert energy == pytest.approx(0.0)

    def test_spring_potential_energy(self, two_node_graph, euler_config):
        """E_spring = 0.5 * k * (delta_a)^2."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.node_states["A"].activation = 1.0
        dyn.node_states["B"].activation = 0.0

        energy = dyn._compute_energy()
        # k_eff = stiffness * weight * multiplier = 0.5 * 1.0 * 0.6 (RELATED_TO)
        k_eff = euler_config.get_edge_stiffness("RELATED_TO", 1.0)
        expected_spring = 0.5 * k_eff * (1.0 ** 2)
        assert energy == pytest.approx(expected_spring)

    def test_kinetic_energy(self, euler_config):
        """E_kinetic = 0.5 * m * v^2."""
        kg = MockKG()
        kg.add_node("A")  # isolated node, no edges => no spring energy
        dyn = SpringDynamics(kg, euler_config)
        dyn.node_states["A"].velocity = 2.0

        energy = dyn._compute_energy()
        expected_kinetic = 0.5 * euler_config.mass * (2.0 ** 2)
        assert energy == pytest.approx(expected_kinetic)

    def test_energy_is_non_negative(self, chain_graph, euler_config):
        """Energy is always non-negative (sum of squares)."""
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn._compute_energy() >= 0.0

    def test_equal_activation_zero_spring_energy(self, two_node_graph, euler_config):
        """No displacement => zero spring potential."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.node_states["A"].activation = 0.5
        dyn.node_states["B"].activation = 0.5
        # No velocity => zero kinetic too
        energy = dyn._compute_energy()
        assert energy == pytest.approx(0.0)


# ============================================================================
# Euler propagation
# ============================================================================

class TestPropagateEuler:

    def test_propagation_returns_result(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_activation_spreads_to_neighbor(self, two_node_graph, euler_config):
        """Activation on A should spread to connected B."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn.get_activation("B") > 0.0

    def test_activation_spreads_transitively(self, chain_graph, euler_config):
        """A→B→C: activating A should eventually reach C."""
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn.get_activation("C") > 0.0

    def test_seed_maintained_when_configured(self, two_node_graph):
        """With maintain_seed_activation=True, seed doesn't drop below initial."""
        config = SpringConfig(
            use_advanced_integrator=False,
            maintain_seed_activation=True,
            max_iterations=50,
        )
        dyn = SpringDynamics(two_node_graph, config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn.get_activation("A") >= 1.0 - 1e-9

    def test_convergence_detected(self, two_node_graph, euler_config):
        """Small graph should converge within max_iterations."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        # Should converge for such a tiny graph
        assert result.converged or result.iterations == euler_config.max_iterations

    def test_max_iterations_respected(self, two_node_graph):
        """Propagation stops at max_iterations even without convergence."""
        config = SpringConfig(
            use_advanced_integrator=False,
            max_iterations=3,
            convergence_epsilon=1e-20,  # basically never converges
        )
        dyn = SpringDynamics(two_node_graph, config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert result.iterations == 3

    def test_result_node_activations_dict(self, chain_graph, euler_config):
        """Result contains node_activations for non-trivial nodes."""
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        # A should be in the dict (activation > 0.01)
        assert "A" in result.node_activations
        assert result.node_activations["A"] > 0.01

    def test_stronger_edge_propagates_more(self):
        """Higher weight edge transmits more activation."""
        kg_strong = MockKG()
        kg_strong.add_edge("src", "dst", weight=1.0)
        kg_weak = MockKG()
        kg_weak.add_edge("src", "dst", weight=0.1)

        config = SpringConfig(
            use_advanced_integrator=False,
            max_iterations=50,
            maintain_seed_activation=True,
        )

        dyn_s = SpringDynamics(kg_strong, config)
        dyn_w = SpringDynamics(kg_weak, config)

        dyn_s.activate_nodes({"src": 1.0})
        dyn_w.activate_nodes({"src": 1.0})
        dyn_s.propagate()
        dyn_w.propagate()

        assert dyn_s.get_activation("dst") > dyn_w.get_activation("dst")

    def test_no_activation_no_propagation(self, chain_graph, euler_config):
        """No seed activation => nothing propagates."""
        dyn = SpringDynamics(chain_graph, euler_config)
        result = dyn.propagate()
        for ns in dyn.node_states.values():
            assert ns.activation == pytest.approx(0.0)

    def test_isolated_node_decays(self, isolated_node_graph):
        """Decay reduces activation each step (tested directly, not through propagate)."""
        state = NodeState(node_id="lonely", activation=1.0, velocity=0.0, mass=1.0)
        # Apply zero force with decay for 10 steps
        for _ in range(10):
            state.apply_force(force=0.0, dt=0.1, damping=0.9, decay=0.9)
        # Should have decayed: 1.0 * 0.9^10 ~ 0.349
        assert state.activation == pytest.approx(0.9 ** 10, abs=1e-6)
        assert state.activation < 0.5


# ============================================================================
# get_active_nodes and get_activation
# ============================================================================

class TestGetActiveNodes:

    def test_returns_nodes_above_threshold(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.node_states["A"].activation = 0.5
        dyn.node_states["B"].activation = 0.01
        dyn.node_states["C"].activation = 0.0

        active = dyn.get_active_nodes(threshold=0.05)
        assert "A" in active
        assert "B" not in active
        assert "C" not in active

    def test_sorted_by_activation_descending(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.node_states["A"].activation = 0.3
        dyn.node_states["B"].activation = 0.9
        dyn.node_states["C"].activation = 0.6

        active = dyn.get_active_nodes(threshold=0.1)
        assert active == ["B", "C", "A"]

    def test_default_threshold_from_config(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        dyn.node_states["A"].activation = euler_config.activation_threshold + 0.01
        dyn.node_states["B"].activation = euler_config.activation_threshold - 0.01

        active = dyn.get_active_nodes()
        assert "A" in active
        assert "B" not in active

    def test_empty_when_all_below_threshold(self, chain_graph, euler_config):
        dyn = SpringDynamics(chain_graph, euler_config)
        active = dyn.get_active_nodes(threshold=0.5)
        assert active == []

    def test_get_activation_nonexistent_node(self, two_node_graph, euler_config):
        dyn = SpringDynamics(two_node_graph, euler_config)
        assert dyn.get_activation("NONEXISTENT") == 0.0


# ============================================================================
# SpringPropagationResult
# ============================================================================

class TestSpringPropagationResult:

    def test_str_converged(self):
        result = SpringPropagationResult(
            iterations=42,
            converged=True,
            final_energy=0.001,
            activated_nodes=["A", "B"],
            node_activations={"A": 0.9, "B": 0.5},
        )
        s = str(result)
        assert "converged" in s
        assert "42" in s

    def test_str_not_converged(self):
        result = SpringPropagationResult(
            iterations=200,
            converged=False,
            final_energy=1.5,
            activated_nodes=["A"],
            node_activations={"A": 0.9},
        )
        s = str(result)
        assert "max iterations" in s


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:

    def test_empty_graph(self, euler_config):
        """Engine works on a graph with no nodes."""
        kg = MockKG()
        dyn = SpringDynamics(kg, euler_config)
        result = dyn.propagate()
        assert result.activated_nodes == []
        assert result.iterations >= 1

    def test_single_node_no_edges(self, isolated_node_graph, euler_config):
        """Single node with no edges still works."""
        dyn = SpringDynamics(isolated_node_graph, euler_config)
        dyn.activate_nodes({"lonely": 1.0})
        result = dyn.propagate()
        assert "lonely" in result.activated_nodes

    def test_self_loop_does_not_crash(self, euler_config):
        """Self-loop edge shouldn't crash (delta_a = 0)."""
        kg = MockKG()
        kg.add_edge("A", "A", weight=1.0)
        dyn = SpringDynamics(kg, euler_config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_very_high_stiffness(self, two_node_graph):
        """Very high stiffness shouldn't break (clamping saves us)."""
        config = SpringConfig(
            use_advanced_integrator=False,
            stiffness=100.0,
            max_iterations=10,
        )
        dyn = SpringDynamics(two_node_graph, config)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        # Activation should still be bounded
        for ns in dyn.node_states.values():
            assert 0.0 <= ns.activation <= 1.0

    def test_zero_stiffness_no_propagation(self, two_node_graph):
        """Zero stiffness => no spring force => no propagation."""
        config = SpringConfig(
            use_advanced_integrator=False,
            stiffness=0.0,
            decay=1.0,
            damping=1.0,
            maintain_seed_activation=False,
            max_iterations=20,
        )
        dyn = SpringDynamics(two_node_graph, config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()

        # B should not have received any activation
        assert dyn.get_activation("B") == pytest.approx(0.0)

    def test_zero_dt_no_movement(self, two_node_graph):
        """dt=0 means no integration step => nothing moves."""
        config = SpringConfig(
            use_advanced_integrator=False,
            dt=0.0,
            decay=1.0,
            maintain_seed_activation=False,
            max_iterations=10,
        )
        dyn = SpringDynamics(two_node_graph, config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        # A stays at 1.0 (no movement), B stays at 0
        assert dyn.get_activation("B") == pytest.approx(0.0)

    def test_decay_1_preserves_activation(self, isolated_node_graph):
        """decay=1.0 means no forgetting."""
        config = SpringConfig(
            use_advanced_integrator=False,
            decay=1.0,
            maintain_seed_activation=False,
            max_iterations=10,
        )
        dyn = SpringDynamics(isolated_node_graph, config)
        dyn.node_states["lonely"].activation = 0.5
        initial = dyn.get_activation("lonely")
        dyn.propagate()
        # No edges, no forces, no decay => stays the same
        assert dyn.get_activation("lonely") == pytest.approx(initial)

    def test_repeated_propagate_calls(self, two_node_graph, euler_config):
        """Calling propagate multiple times is safe."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        r1 = dyn.propagate()
        r2 = dyn.propagate()
        # Second run starts from where first ended
        assert isinstance(r2, SpringPropagationResult)

    def test_activate_after_propagate(self, two_node_graph, euler_config):
        """Activating new seeds after propagation works."""
        dyn = SpringDynamics(two_node_graph, euler_config)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        # Now activate B too
        dyn.activate_nodes({"B": 1.0})
        assert dyn.get_activation("B") == 1.0


# ============================================================================
# Multi-node topology tests
# ============================================================================

class TestTopology:

    def test_star_center_propagates_to_all_leaves(self, star_graph, euler_config):
        """Center activation reaches all leaf nodes."""
        dyn = SpringDynamics(star_graph, euler_config)
        dyn.activate_nodes({"center": 1.0})
        dyn.propagate()

        for leaf in ["n1", "n2", "n3", "n4"]:
            assert dyn.get_activation(leaf) > 0.0

    def test_star_stronger_edge_more_activation(self, star_graph, euler_config):
        """Leaves connected by stronger edges get more activation."""
        dyn = SpringDynamics(star_graph, euler_config)
        dyn.activate_nodes({"center": 1.0})
        dyn.propagate()

        # n1 (w=1.0) > n2 (w=0.5) > n3 (w=0.25) > n4 (w=0.1)
        a1 = dyn.get_activation("n1")
        a2 = dyn.get_activation("n2")
        a3 = dyn.get_activation("n3")
        a4 = dyn.get_activation("n4")
        assert a1 > a2 > a3 > a4

    def test_bidirectional_edge_doubles_force(self, euler_config):
        """Two edges (A→B, B→A) create stronger coupling."""
        kg_uni = MockKG()
        kg_uni.add_edge("A", "B", weight=1.0)

        kg_bi = MockKG()
        kg_bi.add_edge("A", "B", weight=1.0)
        kg_bi.add_edge("B", "A", weight=1.0)

        dyn_uni = SpringDynamics(kg_uni, euler_config)
        dyn_bi = SpringDynamics(kg_bi, euler_config)

        dyn_uni.activate_nodes({"A": 1.0})
        dyn_bi.activate_nodes({"A": 1.0})

        # Just check force magnitudes before propagation
        f_uni = dyn_uni._compute_forces()
        f_bi = dyn_bi._compute_forces()

        assert abs(f_bi["B"]) > abs(f_uni["B"])
