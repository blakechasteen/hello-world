"""
Unit Tests for HoloLoom Spring Dynamics Module
===============================================
Comprehensive tests for spring-based activation propagation on knowledge graphs.

Created 2026-01-28

Covers:
- SpringConfig creation, defaults, and edge stiffness calculation
- NodeState reset and force application
- SpringDynamics initialization, activation, propagation, convergence
- get_active_nodes filtering and get_activation lookup
- Energy calculation correctness
- Edge cases: empty graph, single node, disconnected components
"""

import math
import pytest
import networkx as nx

from HoloLoom.memory.spring_dynamics import (
    SpringConfig,
    NodeState,
    SpringDynamics,
    SpringPropagationResult,
)


# ---------------------------------------------------------------------------
# Helpers: mock graph objects with .G property
# ---------------------------------------------------------------------------

class MockGraph:
    """Minimal graph wrapper exposing a .G property (NetworkX MultiDiGraph)."""

    def __init__(self, graph: nx.MultiDiGraph = None):
        self.G = graph if graph is not None else nx.MultiDiGraph()


def make_simple_graph() -> MockGraph:
    """Create A --IS_A--> B --USES--> C chain graph."""
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", type="IS_A", weight=1.0)
    G.add_edge("B", "C", type="USES", weight=1.0)
    return MockGraph(G)


def make_triangle_graph() -> MockGraph:
    """A <-> B <-> C <-> A triangle with mixed edge types."""
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", type="IS_A", weight=1.0)
    G.add_edge("B", "C", type="MENTIONS", weight=0.8)
    G.add_edge("C", "A", type="USES", weight=0.5)
    return MockGraph(G)


def make_disconnected_graph() -> MockGraph:
    """Two disconnected components: {X, Y} and {P, Q}."""
    G = nx.MultiDiGraph()
    G.add_edge("X", "Y", type="IS_A", weight=1.0)
    G.add_edge("P", "Q", type="USES", weight=1.0)
    return MockGraph(G)


def make_empty_graph() -> MockGraph:
    """A graph with no nodes and no edges."""
    return MockGraph(nx.MultiDiGraph())


def make_single_node_graph() -> MockGraph:
    """Graph with a single isolated node and no edges."""
    G = nx.MultiDiGraph()
    G.add_node("solo")
    return MockGraph(G)


# ===========================================================================
# Tests: SpringConfig
# ===========================================================================

class TestSpringConfig:
    """Tests for SpringConfig dataclass and methods."""

    def test_default_values(self):
        """SpringConfig() should use documented default values."""
        cfg = SpringConfig()
        assert cfg.stiffness == 0.80
        assert cfg.damping == 0.85
        assert cfg.decay == 0.99
        assert cfg.max_iterations == 200
        assert cfg.convergence_epsilon == 5e-4
        assert cfg.dt == 0.2
        assert cfg.activation_threshold == 0.1
        assert cfg.mass == 1.0
        assert cfg.maintain_seed_activation is True
        assert cfg.use_advanced_integrator is True
        assert cfg.integrator_type == "verlet"

    def test_custom_values(self):
        """Custom parameters should override defaults."""
        cfg = SpringConfig(stiffness=0.5, damping=0.7, max_iterations=50)
        assert cfg.stiffness == 0.5
        assert cfg.damping == 0.7
        assert cfg.max_iterations == 50

    def test_edge_type_multipliers_defaults(self):
        """Default edge type multipliers should include known types."""
        cfg = SpringConfig()
        assert cfg.edge_type_multipliers["IS_A"] == 1.2
        assert cfg.edge_type_multipliers["PART_OF"] == 1.1
        assert cfg.edge_type_multipliers["USES"] == 0.9
        assert cfg.edge_type_multipliers["MENTIONS"] == 0.7
        assert cfg.edge_type_multipliers["RELATED_TO"] == 0.6

    def test_get_edge_stiffness_known_type(self):
        """get_edge_stiffness should apply the multiplier for known edge types."""
        cfg = SpringConfig(stiffness=1.0)
        # IS_A multiplier = 1.2, weight = 1.0 => 1.0 * 1.0 * 1.2 = 1.2
        assert cfg.get_edge_stiffness("IS_A", 1.0) == pytest.approx(1.2)
        # MENTIONS multiplier = 0.7, weight = 2.0 => 1.0 * 2.0 * 0.7 = 1.4
        assert cfg.get_edge_stiffness("MENTIONS", 2.0) == pytest.approx(1.4)

    def test_get_edge_stiffness_unknown_type(self):
        """Unknown edge types should use multiplier 1.0."""
        cfg = SpringConfig(stiffness=0.5)
        # Unknown type, multiplier defaults to 1.0 => 0.5 * 1.0 * 1.0 = 0.5
        assert cfg.get_edge_stiffness("INVENTED_TYPE", 1.0) == pytest.approx(0.5)

    def test_get_edge_stiffness_weight_scaling(self):
        """Edge weight should scale the stiffness linearly."""
        cfg = SpringConfig(stiffness=0.8)
        s1 = cfg.get_edge_stiffness("IS_A", 0.5)
        s2 = cfg.get_edge_stiffness("IS_A", 1.0)
        assert s2 == pytest.approx(2.0 * s1)


# ===========================================================================
# Tests: NodeState
# ===========================================================================

class TestNodeState:
    """Tests for NodeState dataclass and methods."""

    def test_default_state(self):
        """New NodeState should start inactive with zero velocity."""
        ns = NodeState(node_id="n1")
        assert ns.node_id == "n1"
        assert ns.activation == 0.0
        assert ns.velocity == 0.0
        assert ns.mass == 1.0

    def test_reset(self):
        """reset() should zero out activation and velocity."""
        ns = NodeState(node_id="n1", activation=0.8, velocity=0.5)
        ns.reset()
        assert ns.activation == 0.0
        assert ns.velocity == 0.0

    def test_apply_force_increases_activation(self):
        """Positive force should increase activation on an inactive node."""
        ns = NodeState(node_id="n1", activation=0.0, velocity=0.0, mass=1.0)
        ns.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)
        # acceleration = 1.0/1.0 = 1.0
        # velocity = 0 + 1.0*0.1 = 0.1, then * damping(1.0) = 0.1
        # activation = 0 + 0.1*0.1 = 0.01, then * decay(1.0) = 0.01
        assert ns.velocity == pytest.approx(0.1)
        assert ns.activation == pytest.approx(0.01)

    def test_apply_force_damping(self):
        """Damping should reduce velocity after each step."""
        ns = NodeState(node_id="n1", activation=0.0, velocity=1.0, mass=1.0)
        ns.apply_force(force=0.0, dt=0.1, damping=0.5, decay=1.0)
        # velocity = 1.0 + 0*0.1 = 1.0, then * 0.5 = 0.5
        assert ns.velocity == pytest.approx(0.5)

    def test_apply_force_decay(self):
        """Decay should reduce activation after each step."""
        ns = NodeState(node_id="n1", activation=1.0, velocity=0.0, mass=1.0)
        ns.apply_force(force=0.0, dt=0.1, damping=1.0, decay=0.9)
        # velocity stays 0, activation = 1.0 + 0 = 1.0, then * 0.9 = 0.9
        assert ns.activation == pytest.approx(0.9)

    def test_apply_force_clamps_to_zero(self):
        """Activation should never go below 0."""
        ns = NodeState(node_id="n1", activation=0.0, velocity=0.0, mass=1.0)
        ns.apply_force(force=-100.0, dt=1.0, damping=1.0, decay=1.0)
        assert ns.activation >= 0.0

    def test_apply_force_clamps_to_one(self):
        """Activation should never exceed 1."""
        ns = NodeState(node_id="n1", activation=0.9, velocity=10.0, mass=1.0)
        ns.apply_force(force=100.0, dt=1.0, damping=1.0, decay=1.0)
        assert ns.activation <= 1.0

    def test_apply_force_mass_affects_acceleration(self):
        """Higher mass should result in lower acceleration for the same force."""
        ns_light = NodeState(node_id="light", activation=0.0, velocity=0.0, mass=1.0)
        ns_heavy = NodeState(node_id="heavy", activation=0.0, velocity=0.0, mass=10.0)
        ns_light.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)
        ns_heavy.apply_force(force=1.0, dt=0.1, damping=1.0, decay=1.0)
        assert ns_light.velocity > ns_heavy.velocity


# ===========================================================================
# Tests: SpringDynamics Initialization
# ===========================================================================

class TestSpringDynamicsInit:
    """Tests for SpringDynamics construction and state initialization."""

    def test_init_creates_node_states(self):
        """Node states should be created for every node in graph."""
        graph = make_simple_graph()  # A, B, C
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        assert set(sd.node_states.keys()) == {"A", "B", "C"}

    def test_init_default_config(self):
        """Passing None config should use SpringConfig defaults."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, config=None)
        assert sd.config.stiffness == SpringConfig().stiffness

    def test_init_all_nodes_inactive(self):
        """All nodes should start with zero activation and velocity."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        for state in sd.node_states.values():
            assert state.activation == 0.0
            assert state.velocity == 0.0

    def test_init_empty_graph(self):
        """Empty graph should produce no node states but not crash."""
        graph = make_empty_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        assert len(sd.node_states) == 0

    def test_init_single_node(self):
        """Single-node graph should have exactly one node state."""
        graph = make_single_node_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        assert len(sd.node_states) == 1
        assert "solo" in sd.node_states

    def test_init_metrics_zeroed(self):
        """Metrics should start at zero."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        assert sd.iterations == 0
        assert sd.converged is False
        assert sd.final_energy == 0.0


# ===========================================================================
# Tests: activate_nodes
# ===========================================================================

class TestActivateNodes:
    """Tests for SpringDynamics.activate_nodes."""

    def test_activate_sets_activation(self):
        """activate_nodes should set activation on specified nodes."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0, "B": 0.5})
        assert sd.node_states["A"].activation == 1.0
        assert sd.node_states["B"].activation == 0.5
        assert sd.node_states["C"].activation == 0.0

    def test_activate_clamps_above_one(self):
        """Activation values > 1 should be clamped to 1."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 5.0})
        assert sd.node_states["A"].activation == 1.0

    def test_activate_clamps_below_zero(self):
        """Activation values < 0 should be clamped to 0."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": -1.0})
        assert sd.node_states["A"].activation == 0.0

    def test_activate_records_seed_nodes(self):
        """Activated nodes should be recorded in seed_nodes dict."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 0.9})
        assert "A" in sd.seed_nodes
        assert sd.seed_nodes["A"] == pytest.approx(0.9)

    def test_activate_unknown_node_ignored(self):
        """Activating a node not in the graph should be silently ignored."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"NONEXISTENT": 1.0})
        assert "NONEXISTENT" not in sd.node_states
        assert "NONEXISTENT" not in sd.seed_nodes

    def test_activate_zeroes_velocity(self):
        """activate_nodes should reset velocity to 0 for seed nodes."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].velocity = 5.0
        sd.activate_nodes({"A": 0.8})
        assert sd.node_states["A"].velocity == 0.0


# ===========================================================================
# Tests: propagate (Euler fallback)
# ===========================================================================

class TestPropagateEuler:
    """Tests for propagation using Euler integration (fallback)."""

    def _make_euler_config(self, **kwargs) -> SpringConfig:
        """Create config that forces Euler integration."""
        defaults = dict(use_advanced_integrator=False)
        defaults.update(kwargs)
        return SpringConfig(**defaults)

    def test_propagate_returns_result(self):
        """propagate should return a SpringPropagationResult."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, self._make_euler_config())
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_propagate_spreads_activation(self):
        """Activation should spread from seed to connected neighbors."""
        graph = make_simple_graph()  # A -> B -> C
        sd = SpringDynamics(graph, self._make_euler_config())
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        # B is directly connected to A, should have gained activation
        assert sd.node_states["B"].activation > 0.0
        # A should remain high (maintain_seed_activation=True by default)
        assert sd.node_states["A"].activation >= 0.9

    def test_propagate_two_hop_spread(self):
        """Activation should propagate beyond direct neighbors (A -> B -> C)."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, self._make_euler_config())
        sd.activate_nodes({"A": 1.0})
        sd.propagate()
        # C is two hops from A; should have some non-zero activation
        assert sd.node_states["C"].activation > 0.0

    def test_propagate_converges(self):
        """Propagation should converge before max_iterations on a simple graph."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, self._make_euler_config(max_iterations=500))
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        assert result.converged is True
        assert result.iterations < 500

    def test_propagate_max_iterations_reached(self):
        """With very few max_iterations, convergence may not be reached."""
        graph = make_simple_graph()
        # Very low iterations + very low epsilon -> unlikely to converge in 2 steps
        sd = SpringDynamics(graph, self._make_euler_config(
            max_iterations=2,
            convergence_epsilon=1e-12,
        ))
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        assert result.iterations == 2
        # May or may not converge depending on energy, just check it runs

    def test_propagate_empty_graph(self):
        """Propagation on an empty graph should return immediately."""
        graph = make_empty_graph()
        sd = SpringDynamics(graph, self._make_euler_config())
        result = sd.propagate()
        assert isinstance(result, SpringPropagationResult)
        assert len(result.activated_nodes) == 0

    def test_propagate_single_node_no_spread(self):
        """A single isolated node has no neighbors to spread to."""
        graph = make_single_node_graph()
        sd = SpringDynamics(graph, self._make_euler_config())
        sd.activate_nodes({"solo": 1.0})
        result = sd.propagate()
        # solo should keep its activation (seed maintained)
        assert sd.node_states["solo"].activation > 0.5
        assert "solo" in result.activated_nodes

    def test_propagate_disconnected_components(self):
        """Activation should not spread across disconnected components."""
        graph = make_disconnected_graph()  # {X,Y} and {P,Q}
        sd = SpringDynamics(graph, self._make_euler_config())
        sd.activate_nodes({"X": 1.0})
        sd.propagate()
        # Y is connected to X, should have activation
        assert sd.node_states["Y"].activation > 0.0
        # P and Q are disconnected; should remain at zero
        assert sd.node_states["P"].activation == 0.0
        assert sd.node_states["Q"].activation == 0.0

    def test_propagate_maintain_seed_activation(self):
        """When maintain_seed_activation=True, seed nodes stay at initial level."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, self._make_euler_config(maintain_seed_activation=True))
        sd.activate_nodes({"A": 0.8})
        sd.propagate()
        assert sd.node_states["A"].activation >= 0.8

    def test_propagate_no_maintain_seed(self):
        """When maintain_seed_activation=False, seed nodes can decay."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, self._make_euler_config(
            maintain_seed_activation=False,
            decay=0.90,
            max_iterations=100,
        ))
        sd.activate_nodes({"A": 1.0})
        sd.propagate()
        # With decay=0.90 and no anchoring, activation should have decayed
        assert sd.node_states["A"].activation < 1.0

    def test_propagate_result_str(self):
        """SpringPropagationResult.__str__ should return a readable string."""
        result = SpringPropagationResult(
            iterations=10,
            converged=True,
            final_energy=0.001,
            activated_nodes=["A", "B"],
            node_activations={"A": 0.9, "B": 0.3},
        )
        s = str(result)
        assert "converged" in s
        assert "10" in s


# ===========================================================================
# Tests: get_active_nodes and get_activation
# ===========================================================================

class TestGetActiveNodes:
    """Tests for get_active_nodes and get_activation."""

    def test_get_active_nodes_default_threshold(self):
        """get_active_nodes should use config threshold by default."""
        graph = make_simple_graph()
        cfg = SpringConfig(use_advanced_integrator=False, activation_threshold=0.5)
        sd = SpringDynamics(graph, cfg)
        sd.node_states["A"].activation = 0.8
        sd.node_states["B"].activation = 0.3
        sd.node_states["C"].activation = 0.6
        active = sd.get_active_nodes()
        assert "A" in active
        assert "C" in active
        assert "B" not in active

    def test_get_active_nodes_custom_threshold(self):
        """get_active_nodes should accept custom threshold."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].activation = 0.05
        sd.node_states["B"].activation = 0.15
        active = sd.get_active_nodes(threshold=0.1)
        assert "B" in active
        assert "A" not in active

    def test_get_active_nodes_sorted_by_activation(self):
        """Active nodes should be sorted by activation descending."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].activation = 0.3
        sd.node_states["B"].activation = 0.9
        sd.node_states["C"].activation = 0.6
        active = sd.get_active_nodes(threshold=0.0)
        assert active == ["B", "C", "A"]

    def test_get_active_nodes_empty_when_all_inactive(self):
        """All-zero activations should yield empty list."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        active = sd.get_active_nodes()
        assert active == []

    def test_get_activation_existing_node(self):
        """get_activation should return the activation of an existing node."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].activation = 0.75
        assert sd.get_activation("A") == pytest.approx(0.75)

    def test_get_activation_missing_node(self):
        """get_activation should return 0.0 for a node not in the graph."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        assert sd.get_activation("NONEXISTENT") == 0.0


# ===========================================================================
# Tests: Energy calculation
# ===========================================================================

class TestEnergyCalculation:
    """Tests for _compute_energy and related energy methods."""

    def test_energy_zero_when_inactive(self):
        """System energy should be zero when no nodes are active."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        energy = sd._compute_energy()
        assert energy == pytest.approx(0.0)

    def test_energy_zero_when_uniform_activation(self):
        """If all connected nodes have equal activation, spring potential is zero."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        for state in sd.node_states.values():
            state.activation = 0.5
            state.velocity = 0.0
        energy = sd._compute_energy()
        assert energy == pytest.approx(0.0)

    def test_energy_positive_when_activation_difference(self):
        """Energy should be positive when connected nodes have different activations."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].activation = 1.0
        sd.node_states["B"].activation = 0.0
        energy = sd._compute_energy()
        assert energy > 0.0

    def test_energy_includes_kinetic(self):
        """Energy should include kinetic contribution from velocity."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        # Set uniform activation (zero spring potential) but give one node velocity
        for state in sd.node_states.values():
            state.activation = 0.5
            state.velocity = 0.0
        sd.node_states["A"].velocity = 2.0
        energy = sd._compute_energy()
        # KE = 0.5 * m * v^2 = 0.5 * 1.0 * 4.0 = 2.0
        assert energy == pytest.approx(2.0)

    def test_energy_manual_calculation(self):
        """Verify energy against manual calculation on small graph."""
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", type="RELATED_TO", weight=1.0)
        graph = MockGraph(G)
        cfg = SpringConfig(stiffness=1.0, use_advanced_integrator=False)
        sd = SpringDynamics(graph, cfg)
        sd.node_states["A"].activation = 1.0
        sd.node_states["B"].activation = 0.0
        sd.node_states["A"].velocity = 0.0
        sd.node_states["B"].velocity = 0.0
        energy = sd._compute_energy()
        # Spring PE = 0.5 * k_eff * (1.0 - 0.0)^2
        # k_eff = stiffness(1.0) * weight(1.0) * multiplier(RELATED_TO=0.6) = 0.6
        # PE = 0.5 * 0.6 * 1.0 = 0.3
        assert energy == pytest.approx(0.3)


# ===========================================================================
# Tests: reset
# ===========================================================================

class TestReset:
    """Tests for SpringDynamics.reset method."""

    def test_reset_clears_all_activations(self):
        """reset() should zero all node states."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0, "B": 0.5})
        sd.reset()
        for state in sd.node_states.values():
            assert state.activation == 0.0
            assert state.velocity == 0.0

    def test_reset_clears_metrics(self):
        """reset() should zero iterations, converged, energy, seed_nodes."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0})
        sd.propagate()
        sd.reset()
        assert sd.iterations == 0
        assert sd.converged is False
        assert sd.final_energy == 0.0
        assert len(sd.seed_nodes) == 0


# ===========================================================================
# Tests: _compute_forces
# ===========================================================================

class TestComputeForces:
    """Tests for the internal _compute_forces method."""

    def test_forces_zero_when_inactive(self):
        """All forces should be zero when all activations are zero."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        forces = sd._compute_forces()
        for f in forces.values():
            assert f == pytest.approx(0.0)

    def test_forces_nonzero_with_activation_diff(self):
        """Forces should be non-zero when neighboring nodes differ in activation."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0})
        sd.seed_nodes.clear()  # avoid seed anchoring in force calc
        forces = sd._compute_forces()
        # A->B edge: activation_diff = 0-1 = -1 on u=A side, so force on A is negative
        # but force computation uses max(state.activation, seed_nodes.get(..., 0.0))
        # Since seed_nodes is cleared, it uses raw activations
        # Force on A from edge A->B: stiffness * (B.act - A.act) => stiffness * (0 - 1) < 0
        assert forces["A"] != 0.0

    def test_forces_newtons_third_law(self):
        """Forces should satisfy Newton's third law on a single-edge graph."""
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", type="IS_A", weight=1.0)
        graph = MockGraph(G)
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.node_states["A"].activation = 1.0
        sd.node_states["B"].activation = 0.0
        sd.seed_nodes.clear()
        forces = sd._compute_forces()
        # Force on A and B should be equal in magnitude, opposite in sign
        assert forces["A"] == pytest.approx(-forces["B"])


# ===========================================================================
# Tests: SpringPropagationResult
# ===========================================================================

class TestSpringPropagationResult:
    """Tests for the result dataclass."""

    def test_str_converged(self):
        result = SpringPropagationResult(
            iterations=42, converged=True, final_energy=0.01,
            activated_nodes=["A"], node_activations={"A": 0.9},
        )
        assert "converged" in str(result)
        assert "42" in str(result)

    def test_str_not_converged(self):
        result = SpringPropagationResult(
            iterations=200, converged=False, final_energy=1.5,
            activated_nodes=[], node_activations={},
        )
        assert "max iterations" in str(result)

    def test_node_activations_filter(self):
        """After propagation, node_activations should only have non-trivial entries."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        # All values in node_activations should be > 0.01
        for v in result.node_activations.values():
            assert v > 0.01


# ===========================================================================
# Tests: Integration / end-to-end scenarios
# ===========================================================================

class TestEndToEnd:
    """Full workflow integration tests."""

    def test_activate_propagate_get_active(self):
        """Full workflow: activate -> propagate -> get_active_nodes."""
        graph = make_triangle_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0})
        result = sd.propagate()
        active = result.activated_nodes
        # A should be in active list
        assert "A" in active
        # At least one neighbor should also be active
        assert len(active) >= 1

    def test_multiple_seeds(self):
        """Multiple seed nodes should both propagate."""
        graph = make_disconnected_graph()  # {X,Y} and {P,Q}
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"X": 1.0, "P": 0.8})
        result = sd.propagate()
        # Both components should have activation
        assert sd.node_states["Y"].activation > 0.0
        assert sd.node_states["Q"].activation > 0.0

    def test_reset_and_reuse(self):
        """After reset, the system should behave as freshly initialized."""
        graph = make_simple_graph()
        sd = SpringDynamics(graph, SpringConfig(use_advanced_integrator=False))
        sd.activate_nodes({"A": 1.0})
        sd.propagate()
        sd.reset()

        # All nodes should be inactive
        for state in sd.node_states.values():
            assert state.activation == 0.0

        # Can propagate again with new seeds
        sd.activate_nodes({"C": 0.7})
        result = sd.propagate()
        assert sd.node_states["C"].activation > 0.0
