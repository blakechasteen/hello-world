"""
Tests for Spring-Based Memory Activation
=========================================
Tests for physics-driven spreading activation in knowledge graphs.

Run:
    pytest tests/test_spring_activation.py -v

Author: HoloLoom Test Team
Date: 2025-10-29
"""

import pytest
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_dynamics import (
    SpringConfig,
    SpringDynamics,
    NodeState,
    SpringPropagationResult
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_graph():
    """Create a simple 3-node graph for testing."""
    kg = KG()

    # Create a chain: A → B → C
    kg.add_edge(KGEdge("A", "B", "CONNECTS_TO", weight=1.0))
    kg.add_edge(KGEdge("B", "C", "CONNECTS_TO", weight=1.0))

    return kg


@pytest.fixture
def complex_graph():
    """Create a more complex graph for testing."""
    kg = KG()

    # Star pattern: Center → Node1, Node2, Node3
    kg.add_edge(KGEdge("center", "node1", "LINKS_TO", weight=1.0))
    kg.add_edge(KGEdge("center", "node2", "LINKS_TO", weight=0.8))
    kg.add_edge(KGEdge("center", "node3", "LINKS_TO", weight=0.6))

    # Some cross-links
    kg.add_edge(KGEdge("node1", "node2", "RELATES_TO", weight=0.5))

    return kg


# ============================================================================
# NodeState Tests
# ============================================================================

def test_node_state_initialization():
    """Test that NodeState initializes correctly."""
    state = NodeState(node_id="test_node")

    assert state.node_id == "test_node"
    assert state.activation == 0.0
    assert state.velocity == 0.0
    assert state.mass == 1.0


def test_node_state_reset():
    """Test that reset() clears activation and velocity."""
    state = NodeState(node_id="test", activation=0.8, velocity=0.3)

    state.reset()

    assert state.activation == 0.0
    assert state.velocity == 0.0


def test_node_state_apply_force():
    """Test that applying force updates activation."""
    state = NodeState(node_id="test")

    # Apply positive force (should increase activation)
    state.apply_force(force=10.0, dt=0.1, damping=0.9, decay=0.99)

    assert state.activation > 0.0  # Should have increased
    assert state.velocity > 0.0  # Should have positive velocity


def test_node_state_clamping():
    """Test that activation is clamped to [0, 1]."""
    state = NodeState(node_id="test")

    # Apply huge force (would go above 1.0)
    for _ in range(100):
        state.apply_force(force=100.0, dt=0.1, damping=0.5, decay=1.0)

    assert 0.0 <= state.activation <= 1.0  # Should be clamped


# ============================================================================
# SpringConfig Tests
# ============================================================================

def test_spring_config_defaults():
    """Test that SpringConfig has reasonable defaults."""
    config = SpringConfig()

    assert 0.0 < config.stiffness < 1.0
    assert 0.0 < config.damping < 1.0
    assert 0.0 < config.decay <= 1.0
    assert config.max_iterations > 0
    assert config.activation_threshold > 0.0


def test_spring_config_edge_stiffness():
    """Test that edge type multipliers work."""
    config = SpringConfig(stiffness=0.2)

    # IS_A relationships should be stronger
    is_a_stiffness = config.get_edge_stiffness("IS_A", edge_weight=1.0)
    generic_stiffness = config.get_edge_stiffness("RELATED_TO", edge_weight=1.0)

    assert is_a_stiffness > generic_stiffness


# ============================================================================
# SpringDynamics Tests
# ============================================================================

def test_spring_dynamics_initialization(simple_graph):
    """Test that SpringDynamics initializes node states."""
    dynamics = SpringDynamics(simple_graph)

    # Should have created states for all nodes
    assert "A" in dynamics.node_states
    assert "B" in dynamics.node_states
    assert "C" in dynamics.node_states

    # All should start at 0 activation
    assert dynamics.get_activation("A") == 0.0
    assert dynamics.get_activation("B") == 0.0
    assert dynamics.get_activation("C") == 0.0


def test_spring_dynamics_activate_nodes(simple_graph):
    """Test activating seed nodes."""
    dynamics = SpringDynamics(simple_graph)

    dynamics.activate_nodes({"A": 1.0, "B": 0.5})

    assert dynamics.get_activation("A") == 1.0
    assert dynamics.get_activation("B") == 0.5
    assert dynamics.get_activation("C") == 0.0


def test_spring_dynamics_propagation(simple_graph):
    """Test that activation propagates through springs."""
    dynamics = SpringDynamics(simple_graph, SpringConfig(max_iterations=50))

    # Activate node A
    dynamics.activate_nodes({"A": 1.0})

    # Propagate
    result = dynamics.propagate()

    # Node A should still be activated (though may have decayed slightly)
    assert dynamics.get_activation("A") > 0.5

    # Node B should have gained activation (connected to A)
    assert dynamics.get_activation("B") > 0.1

    # Node C should have some activation (connected to B)
    # This tests multi-hop propagation!
    assert dynamics.get_activation("C") > 0.0


def test_spring_dynamics_convergence(simple_graph):
    """Test that propagation converges."""
    dynamics = SpringDynamics(simple_graph)

    dynamics.activate_nodes({"A": 1.0})

    result = dynamics.propagate()

    # Should converge (or hit max iterations)
    assert result.converged or result.iterations == dynamics.config.max_iterations

    # Energy should be positive but finite
    assert result.final_energy >= 0.0
    assert result.final_energy < float('inf')


def test_spring_dynamics_reset(simple_graph):
    """Test that reset() clears all activations."""
    dynamics = SpringDynamics(simple_graph)

    # Activate and propagate
    dynamics.activate_nodes({"A": 1.0})
    dynamics.propagate()

    # Reset
    dynamics.reset()

    # All activations should be zero
    assert dynamics.get_activation("A") == 0.0
    assert dynamics.get_activation("B") == 0.0
    assert dynamics.get_activation("C") == 0.0


def test_spring_dynamics_get_active_nodes(complex_graph):
    """Test getting active nodes above threshold."""
    dynamics = SpringDynamics(complex_graph, SpringConfig(activation_threshold=0.2))

    # Activate center node
    dynamics.activate_nodes({"center": 1.0})

    # Propagate
    dynamics.propagate()

    # Get active nodes
    active = dynamics.get_active_nodes()

    # Center should definitely be active
    assert "center" in active

    # At least some neighbors should be active
    assert len(active) > 1

    # Results should be sorted by activation (highest first)
    activations = [dynamics.get_activation(nid) for nid in active]
    assert activations == sorted(activations, reverse=True)


def test_spring_dynamics_multi_seed(complex_graph):
    """Test activating multiple seeds simultaneously."""
    dynamics = SpringDynamics(complex_graph)

    # Activate multiple nodes
    dynamics.activate_nodes({
        "center": 1.0,
        "node1": 0.8,
        "node3": 0.6
    })

    result = dynamics.propagate()

    # All seeds should remain active
    assert dynamics.get_activation("center") > 0.5
    assert dynamics.get_activation("node1") > 0.4
    assert dynamics.get_activation("node3") > 0.3

    # Propagation should have activated neighbors
    assert dynamics.get_activation("node2") > 0.0


def test_spring_dynamics_edge_weight_affects_propagation():
    """Test that edge weights affect propagation strength."""
    kg = KG()

    # Create two paths with different weights
    kg.add_edge(KGEdge("source", "strong_target", "STRONG", weight=1.0))
    kg.add_edge(KGEdge("source", "weak_target", "WEAK", weight=0.2))

    dynamics = SpringDynamics(kg, SpringConfig(max_iterations=30))

    dynamics.activate_nodes({"source": 1.0})
    dynamics.propagate()

    # Strong edge should transmit more activation
    strong_activation = dynamics.get_activation("strong_target")
    weak_activation = dynamics.get_activation("weak_target")

    assert strong_activation > weak_activation


# ============================================================================
# Integration Tests
# ============================================================================

def test_spring_activation_transitive_relationships():
    """
    Test that spring activation finds transitive relationships.

    This is the KEY ADVANTAGE over static retrieval!
    """
    kg = KG()

    # Create a chain: Query → Concept1 → Concept2 → Concept3
    kg.add_edge(KGEdge("query", "concept1", "DIRECTLY_RELATED", weight=1.0))
    kg.add_edge(KGEdge("concept1", "concept2", "RELATED_TO", weight=0.8))
    kg.add_edge(KGEdge("concept2", "concept3", "IMPLIES", weight=0.7))

    dynamics = SpringDynamics(kg, SpringConfig(
        max_iterations=100,
        activation_threshold=0.05  # Low threshold to catch transitive
    ))

    # Activate only the query
    dynamics.activate_nodes({"query": 1.0})

    result = dynamics.propagate()

    # Direct connection should be strongly activated
    assert dynamics.get_activation("concept1") > 0.5

    # 1-hop transitive should be activated
    assert dynamics.get_activation("concept2") > 0.2

    # 2-hop transitive should be activated (this is the magic!)
    assert dynamics.get_activation("concept3") > 0.05

    # Get active nodes
    active = dynamics.get_active_nodes(threshold=0.05)

    # All concepts should be retrieved (transitive!)
    assert "concept1" in active
    assert "concept2" in active
    assert "concept3" in active


# ============================================================================
# Performance Tests
# ============================================================================

def test_spring_dynamics_convergence_speed():
    """Test that convergence happens reasonably quickly."""
    kg = KG()

    # Create small network
    for i in range(10):
        kg.add_edge(KGEdge(f"node{i}", f"node{(i+1) % 10}", "CONNECTS", weight=0.8))

    dynamics = SpringDynamics(kg, SpringConfig(max_iterations=200))

    dynamics.activate_nodes({"node0": 1.0})

    result = dynamics.propagate()

    # Should converge in reasonable time (not hit max iterations)
    assert result.converged
    assert result.iterations < 200  # Usually converges in 50-100 for small graphs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
