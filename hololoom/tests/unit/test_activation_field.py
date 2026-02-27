"""
Unit tests for ActivationField.

Verifies:
1. Region activation works correctly
2. Activation spreads through graph
3. Threshold filtering works
4. Decay process works
5. Metrics are accurate
"""

import pytest
import numpy as np
import networkx as nx
from HoloLoom.memory.activation_field import ActivationField


def test_activation_field_creation():
    """Test activation field can be created."""
    field = ActivationField()

    assert len(field.levels) == 0
    assert len(field.spatial_index) == 0
    assert field.n_active() == 0
    assert field.density() == 0.0


def test_activate_region_single_node():
    """Test activating a single node at exact query position."""
    field = ActivationField()

    # Add node at origin
    center = np.zeros(244)
    field.update_spatial_index("node1", center)

    # Activate region at origin
    activated = field.activate_region(
        center=center,
        radius=0.5,
        node_ids=["node1"]
    )

    assert "node1" in activated
    assert field.levels["node1"] == 1.0  # At center, full activation


def test_activate_region_distance_decay():
    """Test activation decays with distance."""
    field = ActivationField()

    # Add nodes at different distances
    center = np.zeros(244)
    near = np.ones(244) * 0.1  # Close
    far = np.ones(244) * 0.4   # Farther

    field.update_spatial_index("center", center)
    field.update_spatial_index("near", near)
    field.update_spatial_index("far", far)

    # Activate with radius 0.5
    activated = field.activate_region(
        center=center,
        radius=0.5,
        node_ids=["center", "near", "far"]
    )

    # All should be activated (within radius)
    assert len(activated) == 3

    # Activation should decay with distance
    assert field.levels["center"] > field.levels["near"]
    assert field.levels["near"] > field.levels["far"]


def test_activate_region_respects_radius():
    """Test nodes outside radius are not activated."""
    field = ActivationField()

    center = np.zeros(244)
    near = np.ones(244) * 0.1
    far = np.ones(244) * 2.0  # Far outside radius

    field.update_spatial_index("center", center)
    field.update_spatial_index("near", near)
    field.update_spatial_index("far", far)

    # Activate with small radius
    activated = field.activate_region(
        center=center,
        radius=0.5,
        node_ids=["center", "near", "far"]
    )

    # Only center and near should be activated
    assert "center" in activated
    assert "near" in activated
    assert "far" not in activated


def test_spread_via_graph():
    """Test activation spreads through graph connections."""
    field = ActivationField()

    # Create simple graph: A → B → C
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", strength=0.8)
    graph.add_edge("B", "C", strength=0.6)

    # Add positions (not used for spreading, but needed for spatial index)
    field.update_spatial_index("A", np.zeros(244))
    field.update_spatial_index("B", np.ones(244) * 0.1)
    field.update_spatial_index("C", np.ones(244) * 0.2)

    # Initial activation at A
    field.levels["A"] = 1.0

    # Spread with 1 iteration
    field.spread_via_graph(graph, iterations=1, decay_factor=0.5)

    # B should be activated (A → B)
    assert "B" in field.levels
    assert field.levels["B"] > 0.0

    # C should NOT be activated yet (need 2 iterations)
    assert "C" not in field.levels or field.levels["C"] < 0.05


def test_spread_via_graph_multiple_iterations():
    """Test activation spreads multiple hops."""
    field = ActivationField()

    # Create graph: A → B → C
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", strength=0.8)
    graph.add_edge("B", "C", strength=0.8)

    field.update_spatial_index("A", np.zeros(244))
    field.update_spatial_index("B", np.ones(244) * 0.1)
    field.update_spatial_index("C", np.ones(244) * 0.2)

    # Initial activation at A
    field.levels["A"] = 1.0

    # Spread with 2 iterations
    field.spread_via_graph(graph, iterations=2, decay_factor=0.5)

    # C should now be activated (A → B → C)
    assert "C" in field.levels
    assert field.levels["C"] > 0.0

    # Activation should decay across hops
    assert field.levels["A"] > field.levels["B"] > field.levels["C"]


def test_above_threshold():
    """Test threshold filtering works."""
    field = ActivationField()

    # Set various activation levels
    field.levels = {
        "high": 0.9,
        "medium": 0.5,
        "low": 0.2,
        "verylow": 0.05
    }

    # Filter with threshold 0.3
    above = field.above_threshold(0.3)

    # Should only include high and medium
    assert "high" in above
    assert "medium" in above
    assert "low" not in above
    assert "verylow" not in above

    # Should be sorted by activation (highest first)
    assert above[0] == "high"
    assert above[1] == "medium"


def test_top_k():
    """Test top K selection works."""
    field = ActivationField()

    # Set activation levels
    field.levels = {
        "node1": 0.9,
        "node2": 0.7,
        "node3": 0.5,
        "node4": 0.3,
        "node5": 0.1
    }

    # Get top 3
    top = field.top_k(3)

    assert len(top) == 3
    assert top[0] == ("node1", 0.9)
    assert top[1] == ("node2", 0.7)
    assert top[2] == ("node3", 0.5)


def test_decay():
    """Test activation decay works."""
    field = ActivationField()

    # Set initial activation
    field.levels = {
        "node1": 1.0,
        "node2": 0.5
    }

    # Decay by 50%
    field.decay(rate=0.5)

    # Activation should be halved
    assert field.levels["node1"] == pytest.approx(0.5)
    assert field.levels["node2"] == pytest.approx(0.25)


def test_decay_removes_weak_activation():
    """Test decay removes very weak activations."""
    field = ActivationField()

    # Set very weak activation
    field.levels = {
        "node1": 0.02,
        "node2": 0.5
    }

    # Decay
    field.decay(rate=0.5)

    # Very weak activation should be removed
    assert "node1" not in field.levels
    # Strong activation should remain
    assert "node2" in field.levels


def test_metrics():
    """Test activation field metrics are accurate."""
    field = ActivationField()

    # Add nodes
    field.update_spatial_index("node1", np.zeros(244))
    field.update_spatial_index("node2", np.ones(244))
    field.update_spatial_index("node3", np.ones(244) * 2)

    # Set activations
    field.levels = {
        "node1": 0.9,
        "node2": 0.5,
        "node3": 0.05  # Below default threshold
    }

    # Check metrics
    assert field.n_active(threshold=0.1) == 2  # node1, node2
    assert field.density() == pytest.approx(2/3)  # 2 active of 3 total
    assert field.total_activation() == pytest.approx(1.45)


def test_clear():
    """Test clearing activation works."""
    field = ActivationField()

    field.levels = {"node1": 0.9, "node2": 0.5}

    field.clear()

    assert len(field.levels) == 0
    assert len(field.prev_levels) == 2  # Previous state preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])