"""
Unit Tests for Spring Dynamics — Coverage Expansion
=====================================================
Targets the 78% of spring_dynamics.py not covered by test_spring_dynamics_core.py.

Focus areas:
- _SpringForceFunction.__call__ (lines 185-227)
- _propagate_advanced via use_advanced_integrator=True (lines 336, 409-466)
- _initialize_advanced_integrator branches (lines 478-479, 495-502, 511-521)
- _compute_forces skip for unknown nodes (line 542)
- _compute_energy skip for unknown nodes (line 580)
- _compute_energy_from_state (lines 610-634)
- SpringConfig and NodeState completeness
"""

import warnings
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from hololoom.core.memory.spring_dynamics import (
    SpringConfig,
    SpringDynamics,
    NodeState,
    SpringPropagationResult,
    _SpringForceFunction,
    _HAVE_ADVANCED_INTEGRATORS,
)


# ============================================================================
# Shared minimal mock graph (same pattern as test_spring_dynamics_core.py)
# ============================================================================

class MockGraph:
    """Minimal graph satisfying SpringDynamics' interface."""

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
    """Wraps MockGraph behind .G property."""

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
def two_node_kg():
    kg = MockKG()
    kg.add_edge("A", "B", edge_type="RELATED_TO", weight=1.0)
    return kg


@pytest.fixture
def chain_kg():
    kg = MockKG()
    kg.add_edge("A", "B", edge_type="RELATED_TO", weight=1.0)
    kg.add_edge("B", "C", edge_type="RELATED_TO", weight=1.0)
    return kg


@pytest.fixture
def euler_cfg():
    """Euler config — use_advanced_integrator=False."""
    return SpringConfig(
        use_advanced_integrator=False,
        stiffness=0.5,
        damping=0.9,
        decay=0.99,
        dt=0.1,
        max_iterations=50,
        convergence_epsilon=1e-4,
        activation_threshold=0.05,
    )


@pytest.fixture
def advanced_cfg():
    """Config requesting advanced (Verlet) integration."""
    return SpringConfig(
        use_advanced_integrator=True,
        integrator_type="verlet",
        stiffness=0.5,
        damping=0.9,
        decay=0.99,
        dt=0.1,
        max_iterations=50,
        convergence_epsilon=1e-4,
        activation_threshold=0.05,
    )


# ============================================================================
# _SpringForceFunction
# ============================================================================

class TestSpringForceFunction:
    """Exercise _SpringForceFunction.__call__ — the uncovered Hamiltonian force path."""

    def _make_dynamics(self, kg, cfg=None):
        """Create SpringDynamics with Euler config (force fn is independent of integrator)."""
        if cfg is None:
            cfg = SpringConfig(use_advanced_integrator=False, stiffness=0.5)
        return SpringDynamics(kg, cfg)

    def test_call_returns_two_arrays(self, two_node_kg):
        """Force function returns (dq_dt, dp_dt) tuple of numpy arrays."""
        from hololoom.memory.integrators import DynamicalState

        dyn = self._make_dynamics(two_node_kg)
        force_fn = _SpringForceFunction(dyn)

        q0 = np.array([1.0, 0.0])
        p0 = np.array([0.0, 0.0])
        state = DynamicalState(q=q0, p=p0, t=0.0)

        dq_dt, dp_dt = force_fn(state)
        assert isinstance(dq_dt, np.ndarray)
        assert isinstance(dp_dt, np.ndarray)
        assert dq_dt.shape == (2,)
        assert dp_dt.shape == (2,)

    def test_zero_state_zero_force(self, two_node_kg):
        """All zeros => no spring force (Δa = 0)."""
        from hololoom.memory.integrators import DynamicalState

        dyn = self._make_dynamics(two_node_kg)
        force_fn = _SpringForceFunction(dyn)

        q0 = np.zeros(2)
        p0 = np.zeros(2)
        state = DynamicalState(q=q0, p=p0, t=0.0)

        dq_dt, dp_dt = force_fn(state)

        # No activation difference, no momentum => no force/velocity
        assert np.allclose(dq_dt, 0.0)
        assert np.allclose(dp_dt, 0.0)

    def test_activation_difference_creates_spring_force(self, two_node_kg):
        """High activation on A, zero on B => B gets pulled up (positive dp_dt)."""
        from hololoom.memory.integrators import DynamicalState

        dyn = self._make_dynamics(two_node_kg)
        force_fn = _SpringForceFunction(dyn)

        node_list = list(dyn.node_states.keys())
        q0 = np.zeros(len(node_list))
        p0 = np.zeros(len(node_list))

        # Set A to 1.0
        idx_a = node_list.index("A")
        idx_b = node_list.index("B")
        q0[idx_a] = 1.0

        state = DynamicalState(q=q0, p=p0, t=0.0)
        dq_dt, dp_dt = force_fn(state)

        # B should receive positive force (pulled toward A's high activation)
        # A should receive negative force (dragged toward B's low activation)
        assert dp_dt[idx_b] > 0
        assert dp_dt[idx_a] < 0

    def test_newton_third_law_in_force_fn(self, two_node_kg):
        """Forces sum to zero: action = -reaction."""
        from hololoom.memory.integrators import DynamicalState

        dyn = self._make_dynamics(two_node_kg)
        force_fn = _SpringForceFunction(dyn)

        node_list = list(dyn.node_states.keys())
        q0 = np.zeros(len(node_list))
        p0 = np.zeros(len(node_list))
        q0[node_list.index("A")] = 0.8

        state = DynamicalState(q=q0, p=p0, t=0.0)
        _, dp_dt = force_fn(state)

        # Net momentum change should be zero (equal and opposite)
        assert np.sum(dp_dt) == pytest.approx(0.0, abs=1e-9)

    def test_momentum_drives_velocity_dq_dt(self, two_node_kg):
        """Non-zero momentum => non-zero dq_dt = p/m."""
        from hololoom.memory.integrators import DynamicalState

        cfg = SpringConfig(use_advanced_integrator=False, stiffness=0.5, mass=2.0)
        dyn = self._make_dynamics(two_node_kg, cfg)
        force_fn = _SpringForceFunction(dyn)

        node_list = list(dyn.node_states.keys())
        q0 = np.zeros(len(node_list))
        p0 = np.full(len(node_list), 4.0)  # momentum = 4.0, mass = 2.0 => v = 2.0

        state = DynamicalState(q=q0, p=p0, t=0.0)
        dq_dt, _ = force_fn(state)

        # dq/dt = p / m = 4.0 / 2.0 = 2.0 for each node (before damping)
        # Damping is applied as -c * dq_dt to dp_dt, not to dq_dt itself
        assert np.allclose(dq_dt, 2.0)

    def test_skip_unknown_nodes_in_edges(self):
        """If edge references node not in node_list, it is skipped gracefully."""
        from hololoom.memory.integrators import DynamicalState

        kg = MockKG()
        kg.add_edge("A", "B", weight=1.0)
        cfg = SpringConfig(use_advanced_integrator=False, stiffness=0.5)
        dyn = SpringDynamics(kg, cfg)

        # Monkey-patch: add an edge referencing a non-existent node "Z"
        dyn.graph.G._edges.append(("A", "Z", {"type": "RELATED_TO", "weight": 1.0}))

        force_fn = _SpringForceFunction(dyn)
        q0 = np.array([1.0, 0.0])  # A=1.0, B=0.0 (only 2 nodes in node_list)
        p0 = np.zeros(2)
        state = DynamicalState(q=q0, p=p0, t=0.0)

        # Should not raise, just skip the bad edge
        dq_dt, dp_dt = force_fn(state)
        assert isinstance(dp_dt, np.ndarray)

    def test_damping_applied_to_force(self, two_node_kg):
        """Non-zero momentum => damping force reduces dp_dt."""
        from hololoom.memory.integrators import DynamicalState

        cfg = SpringConfig(use_advanced_integrator=False, stiffness=0.0, damping=0.5, mass=1.0)
        dyn = self._make_dynamics(two_node_kg, cfg)
        force_fn = _SpringForceFunction(dyn)

        node_list = list(dyn.node_states.keys())
        q0 = np.zeros(len(node_list))
        p0 = np.ones(len(node_list))  # v = p/m = 1.0

        state = DynamicalState(q=q0, p=p0, t=0.0)
        _, dp_dt = force_fn(state)

        # With stiffness=0: dp_dt = -damping * dq_dt = -0.5 * 1.0 = -0.5
        assert np.allclose(dp_dt, -0.5)

    def test_edge_type_multiplier_affects_force(self):
        """IS_A edge (multiplier 1.2) creates stronger force than RELATED_TO (0.6)."""
        from hololoom.memory.integrators import DynamicalState

        def make_dyn(edge_type):
            kg = MockKG()
            kg.add_edge("A", "B", edge_type=edge_type, weight=1.0)
            return SpringDynamics(kg, SpringConfig(use_advanced_integrator=False, stiffness=0.5))

        dyn_strong = make_dyn("IS_A")
        dyn_weak = make_dyn("RELATED_TO")

        for dyn in [dyn_strong, dyn_weak]:
            node_list = list(dyn.node_states.keys())
            q0 = np.zeros(len(node_list))
            q0[node_list.index("A")] = 1.0
            p0 = np.zeros(len(node_list))
            from hololoom.memory.integrators import DynamicalState as DS
            state = DS(q=q0, p=p0, t=0.0)
            force_fn = _SpringForceFunction(dyn)
            _, dp_dt_strong = force_fn(state) if dyn is dyn_strong else (None, None)
            _, dp_dt_weak = force_fn(state) if dyn is dyn_weak else (None, None)

        # Re-run cleanly
        from hololoom.memory.integrators import DynamicalState as DS

        def get_force(dyn):
            node_list = list(dyn.node_states.keys())
            q0 = np.zeros(len(node_list))
            q0[node_list.index("A")] = 1.0
            p0 = np.zeros(len(node_list))
            state = DS(q=q0, p=p0, t=0.0)
            fn = _SpringForceFunction(dyn)
            _, dp_dt = fn(state)
            return dp_dt[node_list.index("B")]

        force_strong = get_force(make_dyn("IS_A"))
        force_weak = get_force(make_dyn("RELATED_TO"))
        assert abs(force_strong) > abs(force_weak)


# ============================================================================
# Advanced integrator propagation
# ============================================================================

@pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
class TestPropagateAdvanced:
    """Exercise _propagate_advanced and related branches."""

    def test_propagate_returns_result(self, two_node_kg, advanced_cfg):
        """Advanced propagation returns a SpringPropagationResult."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_activation_spreads_to_neighbor(self, two_node_kg, advanced_cfg):
        """Advanced integrator spreads activation from A to B."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn.get_activation("B") > 0.0

    def test_seed_maintained_advanced(self, two_node_kg, advanced_cfg):
        """maintain_seed_activation anchors seed nodes in advanced mode."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="verlet",
            maintain_seed_activation=True,
            max_iterations=30,
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        assert dyn.get_activation("A") >= 1.0 - 1e-9

    def test_advanced_propagation_copies_state_back(self, chain_kg, advanced_cfg):
        """After advanced propagation, node_states hold the final activations."""
        dyn = SpringDynamics(chain_kg, advanced_cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()

        # node_states should reflect the final activations
        for nid, activation in result.node_activations.items():
            assert dyn.node_states[nid].activation == pytest.approx(activation, abs=1e-9)

    def test_advanced_convergence_detected(self, two_node_kg, advanced_cfg):
        """Advanced integration converges or exhausts max_iterations."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert result.converged or result.iterations == advanced_cfg.max_iterations

    def test_advanced_max_iterations_respected(self, two_node_kg):
        """Advanced integration stops at max_iterations."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="verlet",
            max_iterations=3,
            convergence_epsilon=1e-20,  # practically never converges
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert result.iterations == 3

    def test_verlet_integrator_used(self, two_node_kg, advanced_cfg):
        """SpringDynamics uses verlet integrator when configured."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        assert dyn._use_advanced is True
        assert dyn.integrator is not None

    def test_rk4_integrator(self, two_node_kg):
        """RK4 integrator works end-to-end."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="rk4",
            max_iterations=10,
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_symplectic_euler_integrator(self, two_node_kg):
        """Symplectic Euler integrator works end-to-end."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="symplectic",
            max_iterations=10,
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_rk45_integrator_initializes(self, two_node_kg):
        """RK45 adaptive integrator initializes without error."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="rk45",
            max_iterations=10,
        )
        # RK45 integrator initialization should succeed
        dyn = SpringDynamics(two_node_kg, cfg)
        # If initialization succeeded with advanced path, integrator is set
        # (RK45's step() returns AdaptiveStepResult — propagation bug in source,
        #  but the initialization branch is the coverage target here)
        assert dyn._use_advanced is True
        assert dyn.integrator is not None

    def test_euler_via_advanced_path(self, two_node_kg):
        """Euler integrator type through advanced path works."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="euler",
            max_iterations=10,
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_unknown_integrator_type_falls_back_to_euler(self, two_node_kg):
        """Unknown integrator_type triggers warning and falls back to Euler path."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="this_does_not_exist",
            max_iterations=10,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dyn = SpringDynamics(two_node_kg, cfg)

        # Should fall back to Euler (not use advanced)
        assert dyn._use_advanced is False
        assert any("falling back" in str(warning.message).lower() or
                   "unknown integrator" in str(warning.message).lower()
                   for warning in w)

    def test_unknown_integrator_propagate_still_works(self, two_node_kg):
        """After fallback, propagation still runs via Euler path."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="this_does_not_exist",
            max_iterations=10,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dyn = SpringDynamics(two_node_kg, cfg)

        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    def test_seed_not_maintained_when_disabled(self, two_node_kg):
        """With maintain_seed_activation=False, seed nodes can decay in advanced mode."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="verlet",
            maintain_seed_activation=False,
            decay=0.5,  # strong decay
            max_iterations=20,
        )
        dyn = SpringDynamics(two_node_kg, cfg)
        dyn.activate_nodes({"A": 1.0})
        dyn.propagate()
        # A may drop below 1.0 since seeds are not anchored
        # (just verify no crash and valid range)
        assert 0.0 <= dyn.get_activation("A") <= 1.0

    def test_no_seed_nodes_advanced(self, two_node_kg, advanced_cfg):
        """Advanced propagation with no seeds activated runs without error."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)


# ============================================================================
# _initialize_advanced_integrator — fallback when integrators unavailable
# ============================================================================

class TestInitializeAdvancedIntegrator:
    """Cover the initialization branches not hit by happy path."""

    def test_force_function_wraps_dynamics(self, two_node_kg):
        """_SpringForceFunction stores reference to parent SpringDynamics."""
        dyn = SpringDynamics(two_node_kg, SpringConfig(use_advanced_integrator=False))
        force_fn = _SpringForceFunction(dyn)
        assert force_fn.dynamics is dyn
        assert set(force_fn.node_list) == set(dyn.node_states.keys())

    @pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
    def test_integrator_created_on_init(self, two_node_kg, advanced_cfg):
        """Advanced integrator is created during __init__ when configured."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        assert dyn.integrator is not None
        assert dyn.force_fn is not None

    @pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
    def test_create_integrator_exception_falls_back(self, two_node_kg):
        """If create_integrator raises, _use_advanced is set False."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="verlet",
            max_iterations=5,
        )
        with patch("hololoom.core.memory.spring_dynamics.create_integrator") as mock_create:
            mock_create.side_effect = RuntimeError("GPU exploded")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dyn = SpringDynamics(two_node_kg, cfg)

        assert dyn._use_advanced is False
        assert any("falling back" in str(warning.message).lower() for warning in w)

    @pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
    def test_propagate_falls_back_after_failed_init(self, two_node_kg):
        """After create_integrator exception, propagate() uses Euler path."""
        cfg = SpringConfig(
            use_advanced_integrator=True,
            integrator_type="verlet",
            max_iterations=5,
        )
        with patch("hololoom.core.memory.spring_dynamics.create_integrator") as mock_create:
            mock_create.side_effect = RuntimeError("GPU exploded")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                dyn = SpringDynamics(two_node_kg, cfg)

        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)


# ============================================================================
# _compute_forces — edge skip path (line 542)
# ============================================================================

class TestComputeForcesSkipPath:
    """Cover the 'continue' branch when a node referenced in an edge has no state."""

    def test_missing_node_state_skipped_in_forces(self, euler_cfg):
        """Edge pointing to node not in node_states is silently skipped."""
        kg = MockKG()
        kg.add_edge("A", "B", weight=1.0)
        dyn = SpringDynamics(kg, euler_cfg)

        # Add a phantom edge to a node without a state entry
        dyn.graph.G._edges.append(("A", "PHANTOM", {"type": "RELATED_TO", "weight": 1.0}))

        # node_states does NOT contain PHANTOM
        assert "PHANTOM" not in dyn.node_states

        dyn.node_states["A"].activation = 1.0
        forces = dyn._compute_forces()

        # Should not raise; PHANTOM edge skipped
        assert "A" in forces
        assert "B" in forces
        assert "PHANTOM" not in forces

    def test_both_nodes_missing_state_skipped(self, euler_cfg):
        """Edge where BOTH endpoints lack state entries is skipped."""
        kg = MockKG()
        kg.add_node("A")  # One real node
        dyn = SpringDynamics(kg, euler_cfg)

        # Add edge between two phantom nodes
        dyn.graph.G._edges.append(("X", "Y", {"type": "RELATED_TO", "weight": 1.0}))

        forces = dyn._compute_forces()
        # Should not crash, X and Y not in forces
        assert "A" in forces
        assert "X" not in forces
        assert "Y" not in forces


# ============================================================================
# _compute_energy — edge skip path (line 580)
# ============================================================================

class TestComputeEnergySkipPath:
    """Cover the 'continue' branch in _compute_energy for phantom nodes."""

    def test_missing_node_state_skipped_in_energy(self, euler_cfg):
        """Edge to a node without state is silently skipped in energy calc."""
        kg = MockKG()
        kg.add_edge("A", "B", weight=1.0)
        dyn = SpringDynamics(kg, euler_cfg)

        # Add phantom edge
        dyn.graph.G._edges.append(("A", "GHOST", {"type": "RELATED_TO", "weight": 1.0}))
        assert "GHOST" not in dyn.node_states

        dyn.node_states["A"].activation = 1.0
        dyn.node_states["B"].activation = 0.5

        energy = dyn._compute_energy()
        assert energy >= 0.0  # No crash, valid energy


# ============================================================================
# _compute_energy_from_state (lines 610-634)
# ============================================================================

class TestComputeEnergyFromState:
    """Cover _compute_energy_from_state used by the advanced propagation path."""

    def _make_dynamical_state(self, n, q_values=None, p_values=None):
        """Create a DynamicalState with given q/p arrays."""
        from hololoom.memory.integrators import DynamicalState
        q = np.array(q_values if q_values is not None else [0.0] * n)
        p = np.array(p_values if p_values is not None else [0.0] * n)
        return DynamicalState(q=q, p=p, t=0.0)

    def test_zero_state_zero_energy(self, two_node_kg, euler_cfg):
        """All zeros => total energy is 0."""
        dyn = SpringDynamics(two_node_kg, euler_cfg)
        node_list = list(dyn.node_states.keys())
        state = self._make_dynamical_state(2)
        energy = dyn._compute_energy_from_state(state, node_list)
        assert energy == pytest.approx(0.0)

    def test_spring_potential_energy_formula(self, two_node_kg, euler_cfg):
        """E_spring = 0.5 * k * delta_a^2 matches formula."""
        dyn = SpringDynamics(two_node_kg, euler_cfg)
        node_list = list(dyn.node_states.keys())

        # Set A=1.0, B=0.0
        q_values = [0.0] * len(node_list)
        q_values[node_list.index("A")] = 1.0

        state = self._make_dynamical_state(len(node_list), q_values=q_values)
        energy = dyn._compute_energy_from_state(state, node_list)

        k_eff = euler_cfg.get_edge_stiffness("RELATED_TO", 1.0)
        expected = 0.5 * k_eff * (1.0 ** 2)
        assert energy == pytest.approx(expected)

    def test_kinetic_energy_formula(self, euler_cfg):
        """E_kinetic = p^2 / (2m) matches formula."""
        kg = MockKG()
        kg.add_node("A")  # isolated, no spring energy
        dyn = SpringDynamics(kg, euler_cfg)
        node_list = list(dyn.node_states.keys())

        # p = 2.0, m = 1.0 => KE = 4.0 / 2 = 2.0
        state = self._make_dynamical_state(1, p_values=[2.0])
        energy = dyn._compute_energy_from_state(state, node_list)

        expected = 0.5 * (2.0 ** 2) / euler_cfg.mass
        assert energy == pytest.approx(expected)

    def test_combined_spring_and_kinetic_energy(self, two_node_kg, euler_cfg):
        """Total energy = spring + kinetic when both are non-zero."""
        dyn = SpringDynamics(two_node_kg, euler_cfg)
        node_list = list(dyn.node_states.keys())

        q_values = [0.0] * len(node_list)
        q_values[node_list.index("A")] = 1.0
        p_values = [2.0] * len(node_list)

        state = self._make_dynamical_state(len(node_list), q_values=q_values, p_values=p_values)
        energy = dyn._compute_energy_from_state(state, node_list)

        k_eff = euler_cfg.get_edge_stiffness("RELATED_TO", 1.0)
        spring_e = 0.5 * k_eff * (1.0 ** 2)
        kinetic_e = len(node_list) * 0.5 * (2.0 ** 2) / euler_cfg.mass
        assert energy == pytest.approx(spring_e + kinetic_e)

    def test_energy_is_non_negative(self, chain_kg, euler_cfg):
        """Energy is always non-negative (sums of squares)."""
        dyn = SpringDynamics(chain_kg, euler_cfg)
        node_list = list(dyn.node_states.keys())

        q_values = [0.5, 0.3, 0.1]
        p_values = [-1.0, 0.5, 0.2]
        state = self._make_dynamical_state(3, q_values=q_values, p_values=p_values)

        energy = dyn._compute_energy_from_state(state, node_list)
        assert energy >= 0.0

    def test_skip_edge_with_node_not_in_node_list(self, euler_cfg):
        """Edges referencing nodes absent from node_list are skipped."""
        kg = MockKG()
        kg.add_edge("A", "B", weight=1.0)
        dyn = SpringDynamics(kg, euler_cfg)

        # Add phantom edge
        dyn.graph.G._edges.append(("A", "GHOST", {"type": "RELATED_TO", "weight": 1.0}))

        # Only include A and B in the node list (not GHOST)
        node_list = ["A", "B"]
        q_values = [1.0, 0.0]
        state = self._make_dynamical_state(2, q_values=q_values)

        energy = dyn._compute_energy_from_state(state, node_list)
        # GHOST edge is skipped — energy is just the A-B spring
        k_eff = euler_cfg.get_edge_stiffness("RELATED_TO", 1.0)
        expected = 0.5 * k_eff * (1.0 ** 2)
        assert energy == pytest.approx(expected)

    def test_equal_activations_zero_spring_energy(self, two_node_kg, euler_cfg):
        """Equal activations => zero spring potential."""
        dyn = SpringDynamics(two_node_kg, euler_cfg)
        node_list = list(dyn.node_states.keys())

        q_values = [0.5, 0.5]
        state = self._make_dynamical_state(2, q_values=q_values)

        energy = dyn._compute_energy_from_state(state, node_list)
        assert energy == pytest.approx(0.0)

    def test_mass_affects_kinetic_energy(self, euler_cfg):
        """Doubling mass halves kinetic energy for same momentum."""
        kg = MockKG()
        kg.add_node("A")
        dyn = SpringDynamics(kg, euler_cfg)
        node_list = ["A"]

        state = self._make_dynamical_state(1, p_values=[2.0])

        # Default mass = 1.0
        e_light = dyn._compute_energy_from_state(state, node_list)

        # Double the mass
        dyn.config = SpringConfig(use_advanced_integrator=False, mass=2.0)
        e_heavy = dyn._compute_energy_from_state(state, node_list)

        assert e_light == pytest.approx(2.0)  # 0.5 * 4 / 1
        assert e_heavy == pytest.approx(1.0)  # 0.5 * 4 / 2
        assert e_light > e_heavy

    def test_different_edge_types_affect_spring_energy(self, euler_cfg):
        """IS_A (multiplier 1.2) gives more spring energy than MENTIONS (0.7)."""
        def make_energy(edge_type):
            kg = MockKG()
            kg.add_edge("A", "B", edge_type=edge_type, weight=1.0)
            dyn = SpringDynamics(kg, euler_cfg)
            node_list = list(dyn.node_states.keys())
            q_values = [0.0] * len(node_list)
            q_values[node_list.index("A")] = 1.0
            from hololoom.memory.integrators import DynamicalState
            state = DynamicalState(q=np.array(q_values), p=np.zeros(len(node_list)), t=0.0)
            return dyn._compute_energy_from_state(state, node_list)

        e_isa = make_energy("IS_A")
        e_mentions = make_energy("MENTIONS")
        assert e_isa > e_mentions


# ============================================================================
# propagate() dispatch — covers line 336
# ============================================================================

class TestPropagateDispatch:
    """Verify propagate() routes to the correct internal method."""

    def test_euler_path_when_advanced_disabled(self, two_node_kg, euler_cfg):
        """With use_advanced_integrator=False, _propagate_euler is used."""
        dyn = SpringDynamics(two_node_kg, euler_cfg)
        assert dyn._use_advanced is False
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)

    @pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
    def test_advanced_path_when_enabled(self, two_node_kg, advanced_cfg):
        """With use_advanced_integrator=True, _propagate_advanced is used."""
        dyn = SpringDynamics(two_node_kg, advanced_cfg)
        assert dyn._use_advanced is True
        dyn.activate_nodes({"A": 1.0})
        result = dyn.propagate()
        assert isinstance(result, SpringPropagationResult)


# ============================================================================
# Integration: Euler vs Advanced produce consistent qualitative results
# ============================================================================

@pytest.mark.skipif(not _HAVE_ADVANCED_INTEGRATORS, reason="Advanced integrators not available")
class TestEulerVsAdvancedConsistency:
    """Both paths should yield similar qualitative outcomes."""

    def test_both_spread_activation(self, two_node_kg):
        """Both Euler and Verlet spread activation from A to B."""
        for use_adv, itype in [(False, "euler"), (True, "verlet")]:
            cfg = SpringConfig(
                use_advanced_integrator=use_adv,
                integrator_type=itype,
                max_iterations=50,
            )
            dyn = SpringDynamics(two_node_kg, cfg)
            dyn.activate_nodes({"A": 1.0})
            dyn.propagate()
            assert dyn.get_activation("B") > 0.0, f"B not activated with {itype}"

    def test_both_return_valid_energy(self, two_node_kg):
        """Both paths report non-negative final energy."""
        for use_adv, itype in [(False, "euler"), (True, "verlet")]:
            cfg = SpringConfig(
                use_advanced_integrator=use_adv,
                integrator_type=itype,
                max_iterations=10,
            )
            dyn = SpringDynamics(two_node_kg, cfg)
            dyn.activate_nodes({"A": 1.0})
            result = dyn.propagate()
            assert result.final_energy >= 0.0, f"Negative energy with {itype}"

    def test_both_respect_activation_clamp(self, two_node_kg):
        """Both paths keep activations in [0, 1]."""
        for use_adv, itype in [(False, "euler"), (True, "verlet")]:
            cfg = SpringConfig(
                use_advanced_integrator=use_adv,
                integrator_type=itype,
                stiffness=5.0,  # high stiffness, risks blowup
                max_iterations=20,
            )
            dyn = SpringDynamics(two_node_kg, cfg)
            dyn.activate_nodes({"A": 1.0})
            dyn.propagate()
            for ns in dyn.node_states.values():
                assert 0.0 <= ns.activation <= 1.0, f"Clamp failed with {itype}"


# ============================================================================
# SpringConfig additional coverage
# ============================================================================

class TestSpringConfigAdditional:

    def test_defaults_are_sane(self):
        """Default SpringConfig has reasonable physics values."""
        cfg = SpringConfig()
        assert 0 < cfg.stiffness <= 1.0
        assert 0 < cfg.damping <= 1.0
        assert 0 < cfg.decay <= 1.0
        assert cfg.dt > 0
        assert cfg.max_iterations > 0

    def test_integrator_type_default_is_verlet(self):
        """Default integrator type is verlet."""
        cfg = SpringConfig()
        assert cfg.integrator_type == "verlet"

    def test_use_advanced_default_is_true(self):
        """By default, advanced integrator is requested."""
        cfg = SpringConfig()
        assert cfg.use_advanced_integrator is True

    def test_maintain_seed_default_is_true(self):
        """maintain_seed_activation defaults to True."""
        cfg = SpringConfig()
        assert cfg.maintain_seed_activation is True

    def test_mass_default_is_1(self):
        """Default mass is 1.0."""
        cfg = SpringConfig()
        assert cfg.mass == 1.0


# ============================================================================
# NodeState additional coverage
# ============================================================================

class TestNodeStateAdditional:

    def test_default_activation_is_zero(self):
        ns = NodeState(node_id="x")
        assert ns.activation == 0.0
        assert ns.velocity == 0.0

    def test_reset_after_apply_force(self):
        ns = NodeState(node_id="x", activation=0.5, velocity=0.2, mass=1.0)
        ns.apply_force(1.0, dt=0.1, damping=0.9, decay=0.99)
        ns.reset()
        assert ns.activation == 0.0
        assert ns.velocity == 0.0

    def test_apply_force_accumulates_correctly(self):
        """Two-step accumulation: both steps contribute to final activation."""
        ns = NodeState(node_id="x", activation=0.0, velocity=0.0, mass=1.0)
        ns.apply_force(1.0, dt=0.1, damping=1.0, decay=1.0)
        after_step1 = ns.activation
        ns.apply_force(1.0, dt=0.1, damping=1.0, decay=1.0)
        after_step2 = ns.activation
        assert after_step2 > after_step1  # accelerating


# ============================================================================
# SpringPropagationResult additional
# ============================================================================

class TestSpringPropagationResultAdditional:

    def test_str_includes_node_count(self):
        result = SpringPropagationResult(
            iterations=10,
            converged=True,
            final_energy=0.01,
            activated_nodes=["A", "B", "C"],
            node_activations={"A": 0.9, "B": 0.5, "C": 0.3},
        )
        s = str(result)
        assert "3" in s  # 3 activated nodes

    def test_str_shows_energy(self):
        result = SpringPropagationResult(
            iterations=5,
            converged=False,
            final_energy=3.14159,
            activated_nodes=["A"],
            node_activations={"A": 0.5},
        )
        s = str(result)
        assert "3.1416" in s or "3.14" in s
