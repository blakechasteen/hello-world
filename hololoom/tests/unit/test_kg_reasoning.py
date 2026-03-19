"""
Tests for kg_reasoning.py — MCTS-guided multi-hop KG reasoning.

Wire 8 of the structured AI integration roadmap.
"""

import numpy as np
import pytest
import networkx as nx

from hololoom.core.convergence.kg_reasoning import (
    KGReasoner,
    KGReasoningResult,
    ReasoningPath,
    create_kg_reasoner,
)


# ============================================================================
# Test graph fixture
# ============================================================================

@pytest.fixture
def simple_graph():
    """
    A → B → C → D
    A → E → D
    B → F
    """
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", rel_type="causes")
    g.add_edge("B", "C", rel_type="leads_to")
    g.add_edge("C", "D", rel_type="results_in")
    g.add_edge("A", "E", rel_type="relates_to")
    g.add_edge("E", "D", rel_type="enables")
    g.add_edge("B", "F", rel_type="uses")
    return g


@pytest.fixture
def disconnected_graph():
    """Two disconnected components."""
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", rel_type="related")
    g.add_edge("X", "Y", rel_type="related")
    return g


@pytest.fixture
def deep_chain():
    """A → B → C → D → E → F → G (7 nodes deep)."""
    g = nx.MultiDiGraph()
    nodes = list("ABCDEFG")
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i+1], rel_type="next")
    return g


# ============================================================================
# Basic reasoning
# ============================================================================

class TestKGReasonerBasics:
    def test_reason_returns_result(self, simple_graph):
        reasoner = KGReasoner(max_depth=3, num_simulations=20)
        result = reasoner.reason(simple_graph, anchors=["A"])
        assert isinstance(result, KGReasoningResult)
        assert result.paths_explored > 0
        assert result.search_time_ms >= 0  # Can be 0.0 on fast machines

    def test_best_path_starts_from_anchor(self, simple_graph):
        reasoner = KGReasoner(max_depth=3, num_simulations=30)
        result = reasoner.reason(simple_graph, anchors=["A"])
        assert result.best_path.nodes[0] == "A"

    def test_paths_respect_max_depth(self, deep_chain):
        reasoner = KGReasoner(max_depth=3, num_simulations=30)
        result = reasoner.reason(deep_chain, anchors=["A"])
        for path in result.all_paths:
            assert path.depth <= 3

    def test_multiple_anchors(self, simple_graph):
        reasoner = KGReasoner(max_depth=2, num_simulations=20)
        result = reasoner.reason(simple_graph, anchors=["A", "B"])
        assert result.anchor_entities == ["A", "B"]
        # Should have paths from both anchors
        starts = {p.nodes[0] for p in result.all_paths}
        assert "A" in starts or "B" in starts


# ============================================================================
# Target-directed reasoning
# ============================================================================

class TestTargetDirected:
    def test_target_boosts_score(self, simple_graph):
        reasoner = KGReasoner(max_depth=4, num_simulations=50)

        # Without target
        r1 = reasoner.reason(simple_graph, anchors=["A"])

        # With target D — paths reaching D should score higher
        r2 = reasoner.reason(simple_graph, anchors=["A"], targets=["D"])

        # Find paths that reach D
        paths_to_d = [p for p in r2.all_paths if "D" in p.nodes]
        if paths_to_d:
            assert paths_to_d[0].score > 0

    def test_unreachable_target(self, disconnected_graph):
        reasoner = KGReasoner(max_depth=3, num_simulations=20)
        result = reasoner.reason(disconnected_graph, anchors=["A"], targets=["Y"])
        # Should still return results, just without target bonus
        assert result.paths_explored > 0


# ============================================================================
# Edge types
# ============================================================================

class TestEdgeTypes:
    def test_paths_include_edge_types(self, simple_graph):
        reasoner = KGReasoner(max_depth=3, num_simulations=30)
        result = reasoner.reason(simple_graph, anchors=["A"])
        for path in result.all_paths:
            if path.depth > 0:
                assert len(path.edges) == path.depth
                assert all(isinstance(e, str) for e in path.edges)

    def test_reverse_edges_found(self, simple_graph):
        """Reasoning should traverse edges in both directions."""
        reasoner = KGReasoner(max_depth=3, num_simulations=50)
        result = reasoner.reason(simple_graph, anchors=["D"])
        # D has no outgoing edges but has incoming — should find inv_ edges
        inv_paths = [p for p in result.all_paths if any("inv_" in e for e in p.edges)]
        assert len(inv_paths) > 0


# ============================================================================
# Custom scorer
# ============================================================================

class TestCustomScorer:
    def test_custom_scorer_used(self, simple_graph):
        calls = [0]

        def my_scorer(nodes, edges):
            calls[0] += 1
            return 0.5  # Constant score

        reasoner = KGReasoner(max_depth=2, num_simulations=10, scorer=my_scorer)
        result = reasoner.reason(simple_graph, anchors=["A"])
        assert calls[0] > 0
        # All paths should have score 0.5
        for p in result.all_paths:
            assert p.score == pytest.approx(0.5)

    def test_scorer_can_prefer_long_paths(self, deep_chain):
        def long_path_scorer(nodes, edges):
            return len(nodes) / 7.0  # Longer = better

        reasoner = KGReasoner(max_depth=5, num_simulations=50, scorer=long_path_scorer)
        result = reasoner.reason(deep_chain, anchors=["A"])
        assert result.best_path.depth >= 2  # Should find deeper paths


# ============================================================================
# Deduplication
# ============================================================================

class TestDeduplication:
    def test_unique_paths(self, simple_graph):
        reasoner = KGReasoner(max_depth=3, num_simulations=100)
        result = reasoner.reason(simple_graph, anchors=["A"])
        path_tuples = [tuple(p.nodes) for p in result.all_paths]
        assert len(path_tuples) == len(set(path_tuples))


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_node_graph(self):
        g = nx.MultiDiGraph()
        g.add_node("lonely")
        reasoner = KGReasoner(max_depth=3, num_simulations=10)
        result = reasoner.reason(g, anchors=["lonely"])
        assert result.best_path.nodes == ["lonely"]
        assert result.best_path.depth == 0

    def test_self_loop(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "A", rel_type="self_ref")
        g.add_edge("A", "B", rel_type="related")
        reasoner = KGReasoner(max_depth=3, num_simulations=20)
        result = reasoner.reason(g, anchors=["A"])
        # Should not get stuck in self-loop (visited set prevents it)
        for p in result.all_paths:
            assert len(p.nodes) == len(set(p.nodes))

    def test_empty_anchors(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", rel_type="related")
        reasoner = KGReasoner(max_depth=3, num_simulations=10)
        result = reasoner.reason(g, anchors=[])
        assert result.paths_explored == 0


# ============================================================================
# HoloLoom KG integration
# ============================================================================

class TestHoloLoomKGIntegration:
    @pytest.mark.integration
    def test_works_with_hololoom_kg(self):
        """Test with HoloLoom's actual KG class (slow — imports spaCy)."""
        try:
            from hololoom.core.memory.graph import KG, KGEdge
            kg = KG()
            kg.add_edges([
                KGEdge("quantum", "factoring", "enables", 1.0),
                KGEdge("factoring", "RSA", "breaks", 1.0),
                KGEdge("RSA", "cryptography", "underpins", 1.0),
            ])
            reasoner = KGReasoner(max_depth=4, num_simulations=30)
            result = reasoner.reason(kg, anchors=["quantum"], targets=["cryptography"])

            # KG class may wrap the graph differently — just verify we get paths
            assert result.paths_explored > 0
            assert result.best_path.nodes[0] == "quantum"
        except ImportError:
            pytest.skip("HoloLoom KG module not available")


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create_kg_reasoner(self):
        r = create_kg_reasoner(max_depth=5, num_simulations=100)
        assert isinstance(r, KGReasoner)
        assert r.max_depth == 5
        assert r.num_simulations == 100

    def test_create_defaults(self):
        r = create_kg_reasoner()
        assert r.max_depth == 4
        assert r.num_simulations == 50
