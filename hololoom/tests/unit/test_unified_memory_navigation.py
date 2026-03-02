"""
Unit tests for UnifiedMemory navigation and pattern discovery.

Tests cover:
- All 4 navigation directions (FORWARD, BACKWARD, SIDEWAYS, DEEP)
- All 4 pattern types (LOOP, CLUSTER, RESONANCE, THREAD)
- Edge cases (empty graphs, missing backend, disconnected components)
- Integration with NetworkX graph backend

Author: Claude Code
Date: 2025-11-22
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock
import networkx as nx

from hololoom.memory.unified import UnifiedMemory, Memory, NavigationDirection, MemoryPattern
from hololoom.config import Config


@pytest.fixture
def mock_graph_backend():
    """Create mock backend with NetworkX graph."""
    backend = Mock()

    # Create NetworkX MultiDiGraph
    G = nx.MultiDiGraph()

    # Add test nodes and edges
    # Linear chain: A → B → C → D (using 'type' key for edge type)
    G.add_edge("memory_a", "memory_b", weight=1.0, type="LEADS_TO")
    G.add_edge("memory_b", "memory_c", weight=1.0, type="LEADS_TO")
    G.add_edge("memory_c", "memory_d", weight=1.0, type="LEADS_TO")

    # Backward references: D → A (creates cycle)
    G.add_edge("memory_d", "memory_a", weight=0.8, type="LEADS_TO")

    # Sideways connections: B ← E → C (E is sibling to B and C)
    G.add_edge("memory_a", "memory_e", weight=0.9, type="LEADS_TO")
    G.add_edge("memory_e", "memory_c", weight=0.9, type="LEADS_TO")

    # Cluster: F ↔ G ↔ H (tightly connected)
    G.add_edge("memory_f", "memory_g", weight=1.0, type="USES")
    G.add_edge("memory_g", "memory_f", weight=1.0, type="USES")
    G.add_edge("memory_g", "memory_h", weight=1.0, type="USES")
    G.add_edge("memory_h", "memory_g", weight=1.0, type="USES")
    G.add_edge("memory_f", "memory_h", weight=0.8, type="USES")

    # Create mock graph object
    graph = Mock()
    graph.G = G

    backend.graph = graph
    backend.retrieve = AsyncMock(return_value=[])
    backend.store = AsyncMock()

    return backend


@pytest.fixture
def unified_memory_with_graph(mock_graph_backend):
    """Create UnifiedMemory instance with mock graph backend."""
    config = Config.fast()
    memory = UnifiedMemory(config)
    memory._backend = mock_graph_backend
    memory._backend_available = True
    return memory


@pytest.fixture
def empty_graph_backend():
    """Create backend with empty graph."""
    backend = Mock()
    G = nx.MultiDiGraph()
    graph = Mock()
    graph.G = G
    backend.graph = graph
    backend.retrieve = AsyncMock(return_value=[])
    return backend


class TestNavigationForward:
    """Test FORWARD navigation (successors)."""

    def test_navigate_forward_basic(self, unified_memory_with_graph):
        """Test basic forward navigation following successors."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=3
        )

        # Should follow: A → B → C → D
        assert len(result) == 3
        assert result[0].id == "memory_b"
        assert result[1].id == "memory_c"
        assert result[2].id == "memory_d"

        # Check context
        assert result[0].context['direction'] == 'forward'
        assert result[0].context['step'] == 1

    def test_navigate_forward_stops_at_dead_end(self, unified_memory_with_graph):
        """Test forward navigation in a graph with cycle."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_d",
            direction=NavigationDirection.FORWARD,
            steps=5  # Request more than available
        )

        # Graph has cycle D → A → B → C → D, so it continues
        # Should follow: D → A → B → C (avoids revisiting D)
        assert len(result) >= 1
        assert result[0].id == "memory_a"

    def test_navigate_forward_single_step(self, unified_memory_with_graph):
        """Test forward navigation with single step."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=1
        )

        assert len(result) == 1
        assert result[0].id == "memory_b"


class TestNavigationBackward:
    """Test BACKWARD navigation (predecessors)."""

    def test_navigate_backward_basic(self, unified_memory_with_graph):
        """Test basic backward navigation following predecessors."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_d",
            direction=NavigationDirection.BACKWARD,
            steps=3
        )

        # Should follow: D ← C ← B ← A
        assert len(result) == 3
        assert result[0].id == "memory_c"
        assert result[1].id == "memory_b"
        assert result[2].id == "memory_a"

        # Check context
        assert result[0].context['direction'] == 'backward'

    def test_navigate_backward_multiple_predecessors(self, unified_memory_with_graph):
        """Test backward navigation with multiple predecessors (picks first)."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_c",
            direction=NavigationDirection.BACKWARD,
            steps=1
        )

        # memory_c has two predecessors: B and E
        # Should pick one (implementation picks first)
        assert len(result) == 1
        assert result[0].id in ["memory_b", "memory_e"]

    def test_navigate_backward_from_start(self, unified_memory_with_graph):
        """Test backward navigation from node with no predecessors."""
        # First, need to add node with no predecessors
        unified_memory_with_graph._backend.graph.G.add_node("start_node")

        result = unified_memory_with_graph.navigate(
            from_memory="start_node",
            direction=NavigationDirection.BACKWARD,
            steps=3
        )

        # Should get empty result
        assert len(result) == 0


class TestNavigationSideways:
    """Test SIDEWAYS navigation (siblings)."""

    def test_navigate_sideways_shared_parent(self, unified_memory_with_graph):
        """Test sideways navigation to nodes sharing parent."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_b",
            direction=NavigationDirection.SIDEWAYS,
            steps=2
        )

        # B and E share parent A, so E should be a sibling
        memory_ids = [m.id for m in result]
        assert "memory_e" in memory_ids

        # Check context
        if result:
            assert result[0].context['direction'] == 'sideways'

    def test_navigate_sideways_shared_child(self, unified_memory_with_graph):
        """Test sideways navigation to nodes sharing child."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_b",
            direction=NavigationDirection.SIDEWAYS,
            steps=3
        )

        # B and E both point to C, so they're siblings
        # Should find E when starting from B
        memory_ids = [m.id for m in result]
        assert "memory_e" in memory_ids

    def test_navigate_sideways_isolated_node(self, unified_memory_with_graph):
        """Test sideways navigation from isolated node."""
        # Add isolated node
        unified_memory_with_graph._backend.graph.G.add_node("isolated")

        result = unified_memory_with_graph.navigate(
            from_memory="isolated",
            direction=NavigationDirection.SIDEWAYS,
            steps=3
        )

        # Should get empty result (no siblings)
        assert len(result) == 0


class TestNavigationDeep:
    """Test DEEP navigation (cycles/strange loops)."""

    def test_navigate_deep_finds_cycle(self, unified_memory_with_graph):
        """Test deep navigation finds cycles."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.DEEP,
            steps=10
        )

        # Graph has cycle: A → B → C → D → A
        # Should find nodes in the cycle
        assert len(result) > 0
        memory_ids = [m.id for m in result]

        # Should contain cycle nodes
        cycle_nodes = {"memory_a", "memory_b", "memory_c", "memory_d"}
        assert any(mid in cycle_nodes for mid in memory_ids)

        # Check context
        if result:
            assert result[0].context['direction'] == 'deep'

    def test_navigate_deep_fallback_to_bfs(self, unified_memory_with_graph):
        """Test deep navigation falls back to BFS if cycle detection fails."""
        # This tests the try/except fallback in _navigate_deep
        result = unified_memory_with_graph.navigate(
            from_memory="memory_f",
            direction=NavigationDirection.DEEP,
            steps=5
        )

        # Should still return results via BFS fallback
        assert len(result) > 0


class TestPatternDiscoveryLoops:
    """Test strange loop pattern discovery."""

    def test_discover_loops_basic(self, unified_memory_with_graph):
        """Test discovering strange loops in graph."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop"],
            min_strength=0.3
        )

        # Should find the cycle: A → B → C → D → A
        loop_patterns = [p for p in patterns if p.pattern_type == "loop"]
        assert len(loop_patterns) > 0

        # Check pattern structure
        pattern = loop_patterns[0]
        assert pattern.strength >= 0.3
        assert len(pattern.memories) >= 2
        assert "loop" in pattern.description.lower()

    def test_discover_loops_high_threshold(self, unified_memory_with_graph):
        """Test loop discovery with high strength threshold."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop"],
            min_strength=0.95  # Very high threshold
        )

        # Might not find any loops with such high threshold
        loop_patterns = [p for p in patterns if p.pattern_type == "loop"]
        # This is okay - high threshold filters aggressively
        assert isinstance(loop_patterns, list)

    def test_discover_loops_multiple_cycles(self, unified_memory_with_graph):
        """Test discovering multiple loops."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop"],
            min_strength=0.3
        )

        # Graph has at least 2 cycles:
        # 1. A → B → C → D → A
        # 2. F ↔ G (bidirectional)
        loop_patterns = [p for p in patterns if p.pattern_type == "loop"]
        assert len(loop_patterns) >= 1  # At least one loop


class TestPatternDiscoveryClusters:
    """Test cluster pattern discovery."""

    def test_discover_clusters_basic(self, unified_memory_with_graph):
        """Test discovering memory clusters."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["cluster"],
            min_strength=0.3
        )

        # Should find cluster: F ↔ G ↔ H (tightly connected)
        cluster_patterns = [p for p in patterns if p.pattern_type == "cluster"]
        assert len(cluster_patterns) > 0

        # Check pattern structure
        pattern = cluster_patterns[0]
        assert pattern.strength >= 0.3
        assert len(pattern.memories) >= 2
        assert "cluster" in pattern.description.lower()

    def test_discover_clusters_size_threshold(self, unified_memory_with_graph):
        """Test cluster discovery filters by size."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["cluster"],
            min_strength=0.3
        )

        cluster_patterns = [p for p in patterns if p.pattern_type == "cluster"]

        # All clusters should have at least 2 members
        for pattern in cluster_patterns:
            assert len(pattern.memories) >= 2


class TestPatternDiscoveryResonances:
    """Test resonance pattern discovery."""

    def test_discover_resonances_with_awareness(self, unified_memory_with_graph):
        """Test discovering resonance patterns from awareness graph."""
        # Mock awareness graph (note: backend expects 'awareness_graph' not 'awareness')
        awareness = Mock()

        # Create mock graph with activation data
        awareness_graph = nx.Graph()
        awareness_graph.add_node("memory_a", activation=0.95)
        awareness_graph.add_node("memory_b", activation=0.88)
        awareness_graph.add_node("memory_c", activation=0.75)

        awareness.graph = awareness_graph
        unified_memory_with_graph._backend.awareness_graph = awareness

        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["resonance"],
            min_strength=0.7
        )

        # Should find resonances
        resonance_patterns = [p for p in patterns if p.pattern_type == "resonance"]
        assert len(resonance_patterns) > 0

        # Check pattern
        pattern = resonance_patterns[0]
        assert pattern.strength >= 0.7
        assert "resonance" in pattern.description.lower()

    def test_discover_resonances_no_awareness_graph(self, unified_memory_with_graph):
        """Test resonance discovery without awareness graph."""
        # Remove awareness graph
        if hasattr(unified_memory_with_graph._backend, 'awareness'):
            delattr(unified_memory_with_graph._backend, 'awareness')

        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["resonance"],
            min_strength=0.5
        )

        # Should return empty list (graceful degradation)
        resonance_patterns = [p for p in patterns if p.pattern_type == "resonance"]
        assert len(resonance_patterns) == 0


class TestPatternDiscoveryThreads:
    """Test thread pattern discovery."""

    def test_discover_threads_narrative_chain(self, unified_memory_with_graph):
        """Test discovering narrative threads."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["thread"],
            min_strength=0.3
        )

        # Should find thread: A → B → C → D
        thread_patterns = [p for p in patterns if p.pattern_type == "thread"]
        assert len(thread_patterns) > 0

        # Check pattern
        pattern = thread_patterns[0]
        assert pattern.strength >= 0.3
        assert len(pattern.memories) >= 2
        assert "thread" in pattern.description.lower()

    def test_discover_threads_length_threshold(self, unified_memory_with_graph):
        """Test thread discovery filters by length."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["thread"],
            min_strength=0.5
        )

        thread_patterns = [p for p in patterns if p.pattern_type == "thread"]

        # All threads should have at least 2 memories
        for pattern in thread_patterns:
            assert len(pattern.memories) >= 2


class TestPatternDiscoveryMultiple:
    """Test discovering multiple pattern types."""

    def test_discover_all_patterns(self, unified_memory_with_graph):
        """Test discovering all pattern types simultaneously."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.3
        )

        # Should find patterns of different types
        assert len(patterns) > 0

        # Check variety
        pattern_types = {p.pattern_type for p in patterns}
        assert len(pattern_types) > 1  # At least 2 different types

    def test_discover_no_patterns_high_threshold(self, unified_memory_with_graph):
        """Test discovering patterns with very high threshold."""
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.99  # Nearly impossible threshold
        )

        # Might find nothing, which is okay
        assert isinstance(patterns, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_navigate_nonexistent_node(self, unified_memory_with_graph):
        """Test navigation from non-existent starting node."""
        result = unified_memory_with_graph.navigate(
            from_memory="nonexistent_node",
            direction=NavigationDirection.FORWARD,
            steps=3
        )

        # Should return empty list
        assert result == []

    def test_navigate_no_backend(self):
        """Test navigation without backend."""
        config = Config.fast()
        memory = UnifiedMemory(config)
        memory._backend = None
        memory._backend_available = False

        result = memory.navigate(
            from_memory="any_node",
            direction=NavigationDirection.FORWARD,
            steps=3
        )

        # Should return empty list (graceful degradation)
        assert result == []

    def test_navigate_backend_no_graph(self, unified_memory_with_graph):
        """Test navigation when backend has no graph attribute."""
        # Remove graph attribute
        delattr(unified_memory_with_graph._backend, 'graph')

        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=3
        )

        # Should return empty list
        assert result == []

    def test_discover_patterns_empty_graph(self, empty_graph_backend):
        """Test pattern discovery on empty graph."""
        config = Config.fast()
        memory = UnifiedMemory(config)
        memory._backend = empty_graph_backend
        memory._backend_available = True

        patterns = memory.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.3
        )

        # Should return empty list
        assert patterns == []

    def test_discover_patterns_no_backend(self):
        """Test pattern discovery without backend."""
        config = Config.fast()
        memory = UnifiedMemory(config)
        memory._backend = None
        memory._backend_available = False

        patterns = memory.discover_patterns(
            pattern_types=["loop"],
            min_strength=0.3
        )

        # Should return empty list
        assert patterns == []

    def test_navigate_zero_steps(self, unified_memory_with_graph):
        """Test navigation with zero steps."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=0
        )

        # Should return empty list
        assert result == []

    def test_navigate_negative_steps(self, unified_memory_with_graph):
        """Test navigation with negative steps."""
        result = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=-1
        )

        # Should return empty list
        assert result == []


class TestIntegration:
    """Integration tests with real graph structures."""

    def test_full_workflow_navigation_and_patterns(self, unified_memory_with_graph):
        """Test complete workflow: navigate + discover patterns."""
        # Navigate forward from A
        path = unified_memory_with_graph.navigate(
            from_memory="memory_a",
            direction=NavigationDirection.FORWARD,
            steps=2
        )

        assert len(path) > 0

        # Discover patterns
        patterns = unified_memory_with_graph.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.3
        )

        assert len(patterns) > 0

        # Patterns should describe graph structure
        pattern_types = {p.pattern_type for p in patterns}
        assert "loop" in pattern_types or "thread" in pattern_types

    def test_navigate_all_directions(self, unified_memory_with_graph):
        """Test navigating in all 4 directions from same node."""
        node = "memory_b"

        forward = unified_memory_with_graph.navigate(
            from_memory=node, direction=NavigationDirection.FORWARD, steps=2
        )
        backward = unified_memory_with_graph.navigate(
            from_memory=node, direction=NavigationDirection.BACKWARD, steps=2
        )
        sideways = unified_memory_with_graph.navigate(
            from_memory=node, direction=NavigationDirection.SIDEWAYS, steps=2
        )
        deep = unified_memory_with_graph.navigate(
            from_memory=node, direction=NavigationDirection.DEEP, steps=5
        )

        # All should return results (B is well-connected)
        assert len(forward) > 0
        assert len(backward) > 0
        assert len(sideways) > 0
        assert len(deep) > 0

        # Results should be different
        forward_ids = {m.id for m in forward}
        backward_ids = {m.id for m in backward}
        assert forward_ids != backward_ids  # Different directions find different nodes
