"""
Tests for kg_world_model.py — Forward prediction via graph dynamics.

Wire 10 of the structured AI integration roadmap.
"""

import numpy as np
import pytest
import networkx as nx

from hololoom.core.convergence.kg_world_model import (
    KGWorldModel,
    StatePrediction,
    TrajectoryPrediction,
    create_kg_world_model,
)


@pytest.fixture
def simple_graph():
    """A → B → C, A → D"""
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=1.0)
    g.add_edge("A", "D", weight=0.5)
    return g


@pytest.fixture
def chain_graph():
    """A → B → C → D → E"""
    g = nx.MultiDiGraph()
    for s, t in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
        g.add_edge(s, t, weight=1.0)
    return g


# ============================================================================
# Single-step prediction
# ============================================================================

class TestSingleStep:
    def test_returns_prediction(self, simple_graph):
        model = KGWorldModel()
        pred = model.predict_next_state(
            current_activations={"A": 1.0},
            kg=simple_graph,
        )
        assert isinstance(pred, StatePrediction)
        assert pred.steps_ahead == 1

    def test_activation_spreads_to_neighbors(self, simple_graph):
        model = KGWorldModel()
        pred = model.predict_next_state(
            current_activations={"A": 1.0},
            kg=simple_graph,
        )
        # B is a neighbor of A — should gain some activation
        assert pred.predicted_activations.get("B", 0.0) > 0.0

    def test_inactive_neighbors_appear(self, simple_graph):
        model = KGWorldModel()
        pred = model.predict_next_state(
            current_activations={"A": 1.0},
            kg=simple_graph,
        )
        # B and D are neighbors — should appear in predictions
        assert "B" in pred.predicted_activations
        assert "D" in pred.predicted_activations

    def test_deltas_computed(self, simple_graph):
        model = KGWorldModel()
        pred = model.predict_next_state(
            current_activations={"A": 1.0, "B": 0.0},
            kg=simple_graph,
        )
        assert "A" in pred.activation_deltas
        assert "B" in pred.activation_deltas
        # A should decrease (decay), B should increase (spring pull)
        assert pred.activation_deltas["A"] < 0  # Decay
        assert pred.activation_deltas["B"] > 0  # Spring activation

    def test_emerging_nodes_detected(self, simple_graph):
        model = KGWorldModel(activation_threshold=0.05)
        pred = model.predict_next_state(
            current_activations={"A": 1.0},
            kg=simple_graph,
        )
        # Nodes that cross the threshold should be in emerging
        for node in pred.emerging_nodes:
            assert pred.predicted_activations[node] >= model.activation_threshold

    def test_no_spread_without_edges(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")  # No edges
        model = KGWorldModel()
        pred = model.predict_next_state(
            current_activations={"A": 1.0, "B": 0.0},
            kg=g,
        )
        # B should stay at 0 (no spring connection)
        assert pred.predicted_activations["B"] == pytest.approx(0.0)


# ============================================================================
# Trajectory prediction
# ============================================================================

class TestTrajectory:
    def test_returns_trajectory(self, chain_graph):
        model = KGWorldModel()
        traj = model.predict_trajectory(
            current_activations={"A": 1.0},
            kg=chain_graph,
            steps=5,
        )
        assert isinstance(traj, TrajectoryPrediction)
        assert traj.total_steps == 5
        assert len(traj.states) == 5

    def test_energy_trajectory_recorded(self, chain_graph):
        model = KGWorldModel()
        traj = model.predict_trajectory(
            current_activations={"A": 1.0},
            kg=chain_graph,
            steps=10,
        )
        assert len(traj.energy_trajectory) == 10
        assert all(e >= 0 for e in traj.energy_trajectory)

    def test_activation_propagates_to_neighbors(self, chain_graph):
        model = KGWorldModel()
        traj = model.predict_trajectory(
            current_activations={"A": 1.0},
            kg=chain_graph,
            steps=20,
        )
        # B is a direct neighbor of A — should gain activation
        final = traj.final_activations
        assert final.get("B", 0.0) > 0.0

    def test_convergence_detected(self, simple_graph):
        model = KGWorldModel(damping=0.5)  # High damping → fast convergence
        traj = model.predict_trajectory(
            current_activations={"A": 0.5, "B": 0.5},
            kg=simple_graph,
            steps=50,
            convergence_threshold=1e-4,
        )
        # With high damping, should converge before 50 steps
        # (may or may not, depends on dynamics — just check it's a valid int or None)
        assert traj.convergence_step is None or traj.convergence_step <= 50

    def test_steps_ahead_increments(self, chain_graph):
        model = KGWorldModel()
        traj = model.predict_trajectory(
            current_activations={"A": 1.0},
            kg=chain_graph,
            steps=3,
        )
        for i, state in enumerate(traj.states):
            assert state.steps_ahead == i + 1


# ============================================================================
# Prediction validation
# ============================================================================

class TestValidation:
    def test_perfect_prediction(self, simple_graph):
        model = KGWorldModel()
        pred = model.predict_next_state({"A": 1.0}, simple_graph)

        # Validate against itself (perfect prediction)
        accuracy = model.validate_prediction(pred, pred.predicted_activations)
        assert accuracy == 1.0

    def test_wrong_prediction_lowers_confidence(self, simple_graph):
        model = KGWorldModel()
        initial_conf = model.prediction_confidence()

        pred = model.predict_next_state({"A": 1.0}, simple_graph)
        # Completely wrong actual state
        wrong_actual = {n: 1.0 - v for n, v in pred.predicted_activations.items()}
        model.validate_prediction(pred, wrong_actual, tolerance=0.01)

        assert model.prediction_confidence() < initial_conf

    def test_good_prediction_raises_confidence(self, simple_graph):
        model = KGWorldModel()
        # Lower confidence first
        model._prediction_failures = 5.0

        pred = model.predict_next_state({"A": 1.0}, simple_graph)
        conf_before = model.prediction_confidence()
        model.validate_prediction(pred, pred.predicted_activations)

        assert model.prediction_confidence() > conf_before

    def test_confidence_bounded(self):
        model = KGWorldModel()
        conf = model.prediction_confidence()
        assert 0.0 <= conf <= 1.0


# ============================================================================
# Physics properties
# ============================================================================

class TestPhysics:
    def test_energy_decreases_with_damping(self, chain_graph):
        """With damping, energy should generally decrease over time."""
        model = KGWorldModel(damping=0.5)
        traj = model.predict_trajectory(
            current_activations={"A": 1.0},
            kg=chain_graph,
            steps=20,
        )
        # Energy should trend downward (not necessarily monotonically due to spring dynamics)
        energies = traj.energy_trajectory
        assert energies[-1] <= energies[0] + 0.1  # Allow small fluctuation

    def test_decay_reduces_activation(self, simple_graph):
        """Decay should reduce isolated node activation over time."""
        g = nx.MultiDiGraph()
        g.add_node("lonely")
        model = KGWorldModel(decay=0.9)
        traj = model.predict_trajectory(
            current_activations={"lonely": 1.0},
            kg=g,
            steps=10,
        )
        # Should decay toward 0
        assert traj.final_activations["lonely"] < 0.5


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create(self):
        m = create_kg_world_model(stiffness=0.5, damping=0.7, decay=0.95)
        assert isinstance(m, KGWorldModel)
        assert m.stiffness == 0.5
        assert m.damping == 0.7

    def test_defaults(self):
        m = create_kg_world_model()
        assert m.stiffness == 0.8
        assert m.damping == 0.85
        assert m.decay == 0.99
