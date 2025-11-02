"""
Unit tests for memory/graph.py - NetworkX knowledge graph.

Tests:
1. KG (KnowledgeGraph) initialization
2. add_edge() with different edge types
3. add_edges() batch operation
4. get_neighbors() retrieval with directions
5. get_subgraph() extraction with expansion
6. Path finding between entities
7. Edge type queries
8. Temporal connections
9. Graph statistics
10. Entity indexing
"""

import pytest
from datetime import datetime, UTC
from HoloLoom.memory.graph import KG, KGEdge
import networkx as nx


class TestKGInitialization:
    """Test KG initialization."""

    def test_create_empty_graph(self):
        """Empty graph should have zero nodes and edges."""
        kg = KG()

        assert kg is not None
        assert kg.G is not None
        assert isinstance(kg.G, nx.MultiDiGraph)
        assert kg.G.number_of_nodes() == 0
        assert kg.G.number_of_edges() == 0

    def test_entity_index_initialized(self):
        """Entity index should be initialized as empty dict."""
        kg = KG()

        assert hasattr(kg, '_entity_index')
        assert isinstance(kg._entity_index, dict)
        assert len(kg._entity_index) == 0


class TestEdgeOperations:
    """Test edge addition and management."""

    def test_add_single_edge(self):
        """Adding edge should create nodes and edge."""
        kg = KG()
        edge = KGEdge("A", "B", "IS_A", 1.0)

        kg.add_edge(edge)

        assert kg.G.number_of_nodes() == 2
        assert kg.G.number_of_edges() == 1
        assert kg.G.has_edge("A", "B")

    def test_add_edge_creates_nodes(self):
        """add_edge should automatically create nodes."""
        kg = KG()
        edge = KGEdge("entity1", "entity2", "RELATES_TO", 0.8)

        kg.add_edge(edge)

        assert "entity1" in kg.G
        assert "entity2" in kg.G

    def test_add_edge_with_metadata(self):
        """Edge metadata should be stored."""
        kg = KG()
        edge = KGEdge(
            "src", "dst", "CUSTOM",
            weight=0.9,
            span_id="span_123",
            metadata={"key": "value"}
        )

        kg.add_edge(edge)

        # Get edge data
        edges = list(kg.G.edges("src", "dst", data=True))
        assert len(edges) == 1
        _, _, data = edges[0]
        assert data["type"] == "CUSTOM"
        assert data["weight"] == 0.9
        assert data["span_id"] == "span_123"
        assert data["key"] == "value"

    def test_add_edges_batch(self):
        """add_edges should bulk add edges."""
        kg = KG()
        edges = [
            KGEdge("A", "B", "IS_A", 1.0),
            KGEdge("B", "C", "USES", 0.8),
            KGEdge("C", "D", "MENTIONS", 0.6)
        ]

        kg.add_edges(edges)

        assert kg.G.number_of_nodes() == 4
        assert kg.G.number_of_edges() == 3

    def test_multi_edges_between_same_nodes(self):
        """MultiDiGraph should support multiple edges between same nodes."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("A", "B", "USES", 0.8))

        assert kg.G.number_of_edges() == 2
        assert len(list(kg.G.edges("A", "B"))) == 2

    def test_entity_index_updated(self):
        """Entity index should be updated on edge addition."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))

        assert "A" in kg._entity_index
        assert "B" in kg._entity_index
        assert "B" in kg._entity_index["A"]
        assert "A" in kg._entity_index["B"]


class TestNeighborRetrieval:
    """Test neighbor queries."""

    def test_get_neighbors_outgoing(self):
        """get_neighbors with direction='out' should return successors."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("A", "C", "USES", 0.9))

        neighbors = kg.get_neighbors("A", direction="out")

        assert "B" in neighbors
        assert "C" in neighbors
        assert len(neighbors) == 2

    def test_get_neighbors_incoming(self):
        """get_neighbors with direction='in' should return predecessors."""
        kg = KG()
        kg.add_edge(KGEdge("A", "C", "IS_A", 1.0))
        kg.add_edge(KGEdge("B", "C", "USES", 0.9))

        neighbors = kg.get_neighbors("C", direction="in")

        assert "A" in neighbors
        assert "B" in neighbors
        assert len(neighbors) == 2

    def test_get_neighbors_both(self):
        """get_neighbors with direction='both' should return all neighbors."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("C", "B", "USES", 0.9))
        kg.add_edge(KGEdge("B", "D", "MENTIONS", 0.8))

        neighbors = kg.get_neighbors("B", direction="both")

        assert "A" in neighbors
        assert "C" in neighbors
        assert "D" in neighbors
        assert len(neighbors) == 3

    def test_get_neighbors_nonexistent(self):
        """get_neighbors for nonexistent entity should return empty set."""
        kg = KG()

        neighbors = kg.get_neighbors("nonexistent")

        assert len(neighbors) == 0
        assert isinstance(neighbors, set)

    def test_get_neighbors_multi_hop(self):
        """get_neighbors with max_hops>1 should traverse multiple edges."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("B", "C", "IS_A", 1.0))
        kg.add_edge(KGEdge("C", "D", "IS_A", 1.0))

        # 1-hop: just B
        neighbors_1 = kg.get_neighbors("A", max_hops=1)
        assert neighbors_1 == {"B"}

        # 2-hop: B and C
        neighbors_2 = kg.get_neighbors("A", max_hops=2)
        assert neighbors_2 == {"B", "C"}

        # 3-hop: B, C, and D
        neighbors_3 = kg.get_neighbors("A", max_hops=3)
        assert neighbors_3 == {"B", "C", "D"}


class TestSubgraphExtraction:
    """Test subgraph extraction."""

    def test_subgraph_for_single_entity(self):
        """subgraph_for_entities with single entity."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("B", "C", "USES", 0.8))

        subgraph = kg.subgraph_for_entities(["A"], expand=False)

        assert subgraph.number_of_nodes() == 1
        assert "A" in subgraph

    def test_subgraph_with_expansion(self):
        """subgraph_for_entities with expand=True should include neighbors."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("A", "C", "USES", 0.9))
        kg.add_edge(KGEdge("B", "D", "MENTIONS", 0.8))

        subgraph = kg.subgraph_for_entities(["A"], expand=True, max_hops=1)

        assert "A" in subgraph
        assert "B" in subgraph
        assert "C" in subgraph
        assert subgraph.number_of_nodes() == 3

    def test_subgraph_for_multiple_entities(self):
        """subgraph_for_entities with multiple entities."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("C", "D", "USES", 0.9))

        subgraph = kg.subgraph_for_entities(["A", "C"], expand=False)

        assert subgraph.number_of_nodes() == 2
        assert "A" in subgraph
        assert "C" in subgraph

    def test_subgraph_for_nonexistent_entities(self):
        """subgraph_for_entities with nonexistent entities should return empty."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))

        subgraph = kg.subgraph_for_entities(["X", "Y"], expand=False)

        assert subgraph.number_of_nodes() == 0


class TestPathFinding:
    """Test path finding between entities."""

    def test_get_paths_direct(self):
        """get_paths should find direct path."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))

        paths = kg.get_paths("A", "B")

        assert len(paths) > 0
        assert ["A", "B"] in paths

    def test_get_paths_multi_hop(self):
        """get_paths should find multi-hop paths."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("B", "C", "USES", 0.9))

        paths = kg.get_paths("A", "C", max_length=3)

        assert len(paths) > 0
        assert ["A", "B", "C"] in paths

    def test_get_paths_no_connection(self):
        """get_paths with no connection should return empty."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("C", "D", "USES", 0.9))

        paths = kg.get_paths("A", "D")

        assert len(paths) == 0

    def test_get_paths_nonexistent_nodes(self):
        """get_paths with nonexistent nodes should return empty."""
        kg = KG()

        paths = kg.get_paths("X", "Y")

        assert len(paths) == 0


class TestEdgeTypeQueries:
    """Test edge type-specific queries."""

    def test_get_edge_types(self):
        """get_edge_types should return all edge types between nodes."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("A", "B", "USES", 0.8))

        edge_types = kg.get_edge_types("A", "B")

        assert "IS_A" in edge_types
        assert "USES" in edge_types
        assert len(edge_types) == 2

    def test_get_edge_types_no_edge(self):
        """get_edge_types with no edge should return empty."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))

        edge_types = kg.get_edge_types("A", "C")

        assert len(edge_types) == 0

    def test_get_related_by_type_outgoing(self):
        """get_related_by_type should filter by edge type."""
        kg = KG()
        kg.add_edge(KGEdge("Python", "language", "IS_A", 1.0))
        kg.add_edge(KGEdge("Python", "Django", "USES", 0.9))
        kg.add_edge(KGEdge("Python", "Flask", "USES", 0.8))

        related = kg.get_related_by_type("Python", "USES", direction="out")

        assert "Django" in related
        assert "Flask" in related
        assert "language" not in related

    def test_get_related_by_type_incoming(self):
        """get_related_by_type with direction='in' should get predecessors."""
        kg = KG()
        kg.add_edge(KGEdge("BERT", "transformer", "IS_A", 1.0))
        kg.add_edge(KGEdge("GPT", "transformer", "IS_A", 1.0))

        related = kg.get_related_by_type("transformer", "IS_A", direction="in")

        assert "BERT" in related
        assert "GPT" in related

    def test_get_related_by_type_nonexistent(self):
        """get_related_by_type for nonexistent entity should return empty."""
        kg = KG()

        related = kg.get_related_by_type("X", "IS_A", direction="out")

        assert len(related) == 0


class TestTemporalConnections:
    """Test temporal edge connections."""

    def test_connect_entity_to_time(self):
        """connect_entity_to_time should create time bucket node."""
        kg = KG()
        timestamp = datetime(2024, 1, 31, 20, 30, 0, tzinfo=UTC)

        thread_id = kg.connect_entity_to_time("event1", timestamp)

        assert thread_id.startswith("time::")
        assert thread_id in kg.G
        assert kg.G.has_edge("event1", thread_id)

    def test_connect_entity_to_time_reuses_bucket(self):
        """connect_entity_to_time should reuse existing time bucket."""
        kg = KG()
        timestamp1 = datetime(2024, 1, 31, 20, 30, 0, tzinfo=UTC)
        timestamp2 = datetime(2024, 1, 31, 20, 45, 0, tzinfo=UTC)  # Same bucket

        thread_id1 = kg.connect_entity_to_time("event1", timestamp1)
        thread_id2 = kg.connect_entity_to_time("event2", timestamp2)

        # Should reuse same bucket
        assert thread_id1 == thread_id2
        assert kg.G.has_edge("event1", thread_id1)
        assert kg.G.has_edge("event2", thread_id1)

    def test_connect_entity_to_time_custom_edge_type(self):
        """connect_entity_to_time should accept custom edge type."""
        kg = KG()
        timestamp = datetime(2024, 1, 31, 20, 30, 0, tzinfo=UTC)

        thread_id = kg.connect_entity_to_time(
            "event1",
            timestamp,
            edge_type="OCCURRED_AT",
            weight=0.9
        )

        edges = list(kg.G.edges("event1", thread_id, data=True))
        assert len(edges) == 1
        _, _, data = edges[0]
        assert data["type"] == "OCCURRED_AT"
        assert data["weight"] == 0.9


class TestGraphStatistics:
    """Test graph statistics."""

    def test_stats_empty_graph(self):
        """stats() on empty graph should return zeros."""
        kg = KG()

        stats = kg.stats()

        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0
        assert stats["avg_degree"] == 0
        assert stats["is_connected"] is False

    def test_stats_simple_graph(self):
        """stats() should return correct counts."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("B", "C", "USES", 0.9))

        stats = kg.stats()

        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 2
        assert stats["avg_degree"] > 0
        assert stats["is_connected"] is True

    def test_stats_disconnected_graph(self):
        """stats() should detect disconnected components."""
        kg = KG()
        kg.add_edge(KGEdge("A", "B", "IS_A", 1.0))
        kg.add_edge(KGEdge("C", "D", "USES", 0.9))

        stats = kg.stats()

        assert stats["num_nodes"] == 4
        assert stats["num_edges"] == 2
        assert stats["is_connected"] is False


class TestKGEdgeDataClass:
    """Test KGEdge data class."""

    def test_kgedge_creation(self):
        """KGEdge should create with all fields."""
        edge = KGEdge(
            src="A",
            dst="B",
            type="IS_A",
            weight=0.9,
            span_id="span_123",
            metadata={"key": "value"}
        )

        assert edge.src == "A"
        assert edge.dst == "B"
        assert edge.type == "IS_A"
        assert edge.weight == 0.9
        assert edge.span_id == "span_123"
        assert edge.metadata["key"] == "value"

    def test_kgedge_defaults(self):
        """KGEdge should use default values."""
        edge = KGEdge("A", "B", "IS_A")

        assert edge.weight == 1.0
        assert edge.span_id is None
        assert edge.metadata == {}

    def test_kgedge_to_dict(self):
        """KGEdge.to_dict() should serialize."""
        edge = KGEdge("A", "B", "IS_A", 0.8, "span_1", {"k": "v"})

        data = edge.to_dict()

        assert data["src"] == "A"
        assert data["dst"] == "B"
        assert data["type"] == "IS_A"
        assert data["weight"] == 0.8
        assert data["span_id"] == "span_1"
        assert data["metadata"]["k"] == "v"

    def test_kgedge_from_dict(self):
        """KGEdge.from_dict() should deserialize."""
        data = {
            "src": "A",
            "dst": "B",
            "type": "IS_A",
            "weight": 0.7,
            "span_id": "span_2",
            "metadata": {"key": "val"}
        }

        edge = KGEdge.from_dict(data)

        assert edge.src == "A"
        assert edge.dst == "B"
        assert edge.type == "IS_A"
        assert edge.weight == 0.7
        assert edge.span_id == "span_2"
        assert edge.metadata["key"] == "val"

    def test_kgedge_roundtrip(self):
        """KGEdge should roundtrip through dict."""
        original = KGEdge("A", "B", "IS_A", 0.6, "span", {"meta": "data"})

        data = original.to_dict()
        reconstructed = KGEdge.from_dict(data)

        assert reconstructed.src == original.src
        assert reconstructed.dst == original.dst
        assert reconstructed.type == original.type
        assert reconstructed.weight == original.weight
        assert reconstructed.span_id == original.span_id
        assert reconstructed.metadata == original.metadata


# Total assertions: 80+
# Test classes: 10
# Coverage: Initialization, edges, neighbors, subgraphs, paths, edge types, temporal, stats, serialization
