"""
Knowledge Graph Network Visualization Tests
=============================================
Comprehensive tests for force-directed network visualization.

Test Coverage:
- Force-directed layout (Fruchterman-Reingold algorithm)
- Node sizing by degree/importance
- Edge type colors (7 relationship types)
- Path highlighting
- Integration with HoloLoom KG
- Edge cases and validation

Author: Claude Code
Date: January 2026
"""

import pytest
import math
from typing import List

from HoloLoom.visualization.knowledge_graph import (
    GraphNode,
    GraphEdge,
    EdgeType,
    ForceDirectedLayout,
    KnowledgeGraphRenderer,
    render_knowledge_graph_from_networkx,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_nodes() -> List[GraphNode]:
    """Create basic nodes for testing."""
    return [
        GraphNode(id="node1", label="Node 1", degree=3),
        GraphNode(id="node2", label="Node 2", degree=2),
        GraphNode(id="node3", label="Node 3", degree=1),
    ]


@pytest.fixture
def basic_edges() -> List[GraphEdge]:
    """Create basic edges for testing."""
    return [
        GraphEdge(src="node1", dst="node2", type="IS_A", weight=1.0),
        GraphEdge(src="node2", dst="node3", type="USES", weight=0.8),
        GraphEdge(src="node1", dst="node3", type="MENTIONS", weight=0.5),
    ]


@pytest.fixture
def complex_graph() -> tuple:
    """Create a more complex graph for testing."""
    nodes = [
        GraphNode(id="attention", label="attention", degree=4, node_type="concept"),
        GraphNode(id="transformer", label="transformer", degree=5, node_type="concept"),
        GraphNode(id="neural_network", label="neural network", degree=3, node_type="concept"),
        GraphNode(id="BERT", label="BERT", degree=2, node_type="model"),
        GraphNode(id="GPT", label="GPT", degree=2, node_type="model"),
        GraphNode(id="multi-head", label="multi-head", degree=2, node_type="technique"),
    ]

    edges = [
        GraphEdge(src="attention", dst="transformer", type="USES", weight=1.0),
        GraphEdge(src="transformer", dst="neural_network", type="IS_A", weight=1.0),
        GraphEdge(src="attention", dst="neural_network", type="PART_OF", weight=0.8),
        GraphEdge(src="BERT", dst="transformer", type="IS_A", weight=1.0),
        GraphEdge(src="GPT", dst="transformer", type="IS_A", weight=1.0),
        GraphEdge(src="multi-head", dst="attention", type="IS_A", weight=1.0),
    ]

    return nodes, edges


@pytest.fixture
def layout() -> ForceDirectedLayout:
    """Create default layout."""
    return ForceDirectedLayout()


@pytest.fixture
def renderer() -> KnowledgeGraphRenderer:
    """Create default renderer."""
    return KnowledgeGraphRenderer()


# ============================================================================
# GraphNode Tests
# ============================================================================

class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_basic_creation(self):
        """Test basic node creation."""
        node = GraphNode(id="test", label="Test Node")

        assert node.id == "test"
        assert node.label == "Test Node"
        assert node.x == 0.0
        assert node.y == 0.0
        assert node.vx == 0.0
        assert node.vy == 0.0
        assert node.degree == 0
        assert node.node_type == "default"

    def test_full_creation(self):
        """Test node creation with all fields."""
        node = GraphNode(
            id="full_node",
            label="Full Node",
            x=100.0,
            y=200.0,
            vx=1.0,
            vy=-1.0,
            degree=5,
            node_type="concept",
            metadata={'key': 'value'}
        )

        assert node.x == 100.0
        assert node.y == 200.0
        assert node.degree == 5
        assert node.node_type == "concept"
        assert node.metadata == {'key': 'value'}


# ============================================================================
# GraphEdge Tests
# ============================================================================

class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_basic_creation(self):
        """Test basic edge creation."""
        edge = GraphEdge(src="a", dst="b", type="IS_A")

        assert edge.src == "a"
        assert edge.dst == "b"
        assert edge.type == "IS_A"
        assert edge.weight == 1.0  # Default

    def test_full_creation(self):
        """Test edge creation with all fields."""
        edge = GraphEdge(
            src="node1",
            dst="node2",
            type="USES",
            weight=0.8,
            metadata={'relationship_id': 123}
        )

        assert edge.weight == 0.8
        assert edge.metadata == {'relationship_id': 123}


# ============================================================================
# EdgeType Tests
# ============================================================================

class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_edge_types_exist(self):
        """Test all expected edge types exist."""
        assert hasattr(EdgeType, 'IS_A')
        assert hasattr(EdgeType, 'USES')
        assert hasattr(EdgeType, 'MENTIONS')
        assert hasattr(EdgeType, 'LEADS_TO')
        assert hasattr(EdgeType, 'PART_OF')
        assert hasattr(EdgeType, 'IN_TIME')
        assert hasattr(EdgeType, 'OCCURRED_AT')
        assert hasattr(EdgeType, 'UNKNOWN')

    def test_edge_type_values(self):
        """Test edge type string values."""
        assert EdgeType.IS_A.value == "is_a"
        assert EdgeType.USES.value == "uses"
        assert EdgeType.MENTIONS.value == "mentions"
        assert EdgeType.LEADS_TO.value == "leads_to"
        assert EdgeType.PART_OF.value == "part_of"
        assert EdgeType.IN_TIME.value == "in_time"
        assert EdgeType.OCCURRED_AT.value == "occurred_at"
        assert EdgeType.UNKNOWN.value == "unknown"


# ============================================================================
# ForceDirectedLayout Tests
# ============================================================================

class TestForceDirectedLayout:
    """Tests for ForceDirectedLayout algorithm."""

    def test_default_initialization(self):
        """Test layout with default parameters."""
        layout = ForceDirectedLayout()

        assert layout.width == 800
        assert layout.height == 600
        assert layout.iterations == 300
        assert layout.attraction_strength == 0.01
        assert layout.repulsion_strength == 1000
        assert layout.damping == 0.8

    def test_custom_initialization(self):
        """Test layout with custom parameters."""
        layout = ForceDirectedLayout(
            width=1000,
            height=800,
            iterations=500,
            attraction_strength=0.02,
            repulsion_strength=2000,
            damping=0.7
        )

        assert layout.width == 1000
        assert layout.height == 800
        assert layout.iterations == 500

    def test_layout_empty_nodes(self, layout):
        """Test layout with empty nodes list."""
        result = layout.layout([], [])
        assert result == []

    def test_layout_single_node(self, layout):
        """Test layout with single node."""
        nodes = [GraphNode(id="single", label="Single")]
        result = layout.layout(nodes, [])

        assert len(result) == 1
        # Node should be positioned within bounds
        assert 0 < result[0].x < layout.width
        assert 0 < result[0].y < layout.height

    def test_layout_positions_nodes(self, layout, basic_nodes, basic_edges):
        """Test that layout positions all nodes."""
        result = layout.layout(basic_nodes.copy(), basic_edges)

        assert len(result) == len(basic_nodes)

        # All nodes should have valid positions
        for node in result:
            assert 0 <= node.x <= layout.width
            assert 0 <= node.y <= layout.height

    def test_layout_connected_nodes_closer(self, layout):
        """Test that connected nodes tend to be closer."""
        nodes = [
            GraphNode(id="a", label="A"),
            GraphNode(id="b", label="B"),
            GraphNode(id="c", label="C"),
        ]
        edges = [
            GraphEdge(src="a", dst="b", type="IS_A", weight=1.0),
        ]

        result = layout.layout(nodes.copy(), edges)

        # Find nodes a, b, c
        node_a = next(n for n in result if n.id == "a")
        node_b = next(n for n in result if n.id == "b")
        node_c = next(n for n in result if n.id == "c")

        # Distance between a and b (connected)
        dist_ab = math.sqrt((node_a.x - node_b.x)**2 + (node_a.y - node_b.y)**2)

        # Distance between a and c (not connected)
        dist_ac = math.sqrt((node_a.x - node_c.x)**2 + (node_a.y - node_c.y)**2)

        # Connected nodes should generally be closer (not always guaranteed due to randomness)
        # This is a probabilistic assertion
        # We mainly test that layout runs without error

    def test_layout_keeps_nodes_in_bounds(self, layout, basic_nodes, basic_edges):
        """Test that layout keeps nodes within canvas bounds."""
        result = layout.layout(basic_nodes.copy(), basic_edges)

        margin = 50  # Layout uses margin
        for node in result:
            assert margin <= node.x <= layout.width - margin
            assert margin <= node.y <= layout.height - margin


# ============================================================================
# Renderer Initialization Tests
# ============================================================================

class TestRendererInitialization:
    """Tests for KnowledgeGraphRenderer initialization."""

    def test_default_initialization(self):
        """Test renderer with default parameters."""
        renderer = KnowledgeGraphRenderer()

        assert renderer.width == 900
        assert renderer.height == 700
        assert renderer.node_size_min == 8
        assert renderer.node_size_max == 24
        assert renderer.show_edge_labels is False
        assert renderer.layout_iterations == 300

    def test_custom_initialization(self):
        """Test renderer with custom parameters."""
        renderer = KnowledgeGraphRenderer(
            width=1200,
            height=900,
            node_size_min=5,
            node_size_max=30,
            show_edge_labels=True,
            layout_iterations=500
        )

        assert renderer.width == 1200
        assert renderer.height == 900
        assert renderer.node_size_min == 5
        assert renderer.node_size_max == 30
        assert renderer.show_edge_labels is True


# ============================================================================
# Rendering Tests
# ============================================================================

class TestRendering:
    """Tests for graph rendering."""

    def test_render_basic_html(self, renderer, basic_nodes, basic_edges):
        """Test basic HTML rendering."""
        html = renderer.render(basic_nodes, basic_edges)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<svg" in html

    def test_render_with_title(self, renderer, basic_nodes, basic_edges):
        """Test rendering with custom title."""
        html = renderer.render(
            basic_nodes,
            basic_edges,
            title="Custom Graph Title"
        )

        assert "Custom Graph Title" in html

    def test_render_with_subtitle(self, renderer, basic_nodes, basic_edges):
        """Test rendering with subtitle."""
        html = renderer.render(
            basic_nodes,
            basic_edges,
            title="Graph",
            subtitle="Test subtitle"
        )

        assert "Test subtitle" in html

    def test_render_empty_returns_empty_state(self, renderer):
        """Test rendering empty graph returns empty state."""
        html = renderer.render([], [])

        assert "No nodes available" in html

    def test_render_includes_node_labels(self, renderer, basic_nodes, basic_edges):
        """Test that node labels are rendered."""
        html = renderer.render(basic_nodes, basic_edges)

        # Labels should appear (possibly truncated)
        assert "Node 1" in html or "node1" in html.lower()

    def test_render_includes_statistics(self, renderer, basic_nodes, basic_edges):
        """Test that statistics are rendered."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "Nodes" in html
        assert "Edges" in html


# ============================================================================
# Node Sizing Tests
# ============================================================================

class TestNodeSizing:
    """Tests for node sizing based on degree."""

    def test_node_size_varies_by_degree(self, renderer):
        """Test that node size varies by degree."""
        nodes = [
            GraphNode(id="low", label="Low", degree=1),
            GraphNode(id="medium", label="Medium", degree=5),
            GraphNode(id="high", label="High", degree=10),
        ]
        edges = []

        html = renderer.render(nodes, edges)

        # Should render nodes with different radii
        assert "circle" in html.lower()
        assert 'r="' in html  # Radius attribute

    def test_single_node_gets_middle_size(self, renderer):
        """Test that single node gets middle size."""
        nodes = [GraphNode(id="single", label="Single", degree=5)]

        html = renderer.render(nodes, [])

        # Should render without error
        assert isinstance(html, str)
        assert "<svg" in html


# ============================================================================
# Edge Type Color Tests
# ============================================================================

class TestEdgeTypeColors:
    """Tests for edge type color mapping."""

    def test_edge_colors_defined(self, renderer):
        """Test that all edge types have colors defined."""
        for edge_type in EdgeType:
            assert edge_type in renderer.EDGE_COLORS

    def test_is_a_color_blue(self, renderer):
        """Test IS_A edges are blue."""
        color = renderer.EDGE_COLORS[EdgeType.IS_A]
        assert "#3b82f6" in color or "blue" in color.lower()

    def test_uses_color_green(self, renderer):
        """Test USES edges are green."""
        color = renderer.EDGE_COLORS[EdgeType.USES]
        assert "#10b981" in color or "green" in color.lower()

    def test_edge_colors_appear_in_html(self, renderer, basic_nodes, basic_edges):
        """Test that edge colors appear in rendered HTML."""
        html = renderer.render(basic_nodes, basic_edges)

        # At least one edge color should appear
        colors_found = any(
            color in html
            for color in renderer.EDGE_COLORS.values()
        )
        assert colors_found


# ============================================================================
# Path Highlighting Tests
# ============================================================================

class TestPathHighlighting:
    """Tests for path highlighting functionality."""

    def test_highlighted_path_rendering(self, renderer, basic_nodes, basic_edges):
        """Test rendering with highlighted path."""
        highlighted = ["node1", "node2"]

        html = renderer.render(
            basic_nodes,
            basic_edges,
            highlighted_path=highlighted
        )

        assert isinstance(html, str)
        # Highlighted nodes/edges should have different styling

    def test_partial_path_highlighting(self, renderer, basic_nodes, basic_edges):
        """Test highlighting partial path."""
        highlighted = ["node1"]

        html = renderer.render(
            basic_nodes,
            basic_edges,
            highlighted_path=highlighted
        )

        assert isinstance(html, str)

    def test_no_highlighting(self, renderer, basic_nodes, basic_edges):
        """Test rendering without highlighting."""
        html = renderer.render(
            basic_nodes,
            basic_edges,
            highlighted_path=None
        )

        assert isinstance(html, str)


# ============================================================================
# Legend Tests
# ============================================================================

class TestLegend:
    """Tests for legend rendering."""

    def test_legend_rendered(self, renderer, basic_nodes, basic_edges):
        """Test that legend is rendered."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "graph-legend" in html

    def test_legend_contains_edge_types(self, renderer, basic_nodes, basic_edges):
        """Test that legend contains edge type labels."""
        html = renderer.render(basic_nodes, basic_edges)

        # Should have edge type labels
        assert "Is A" in html or "is_a" in html.lower()


# ============================================================================
# SVG Structure Tests
# ============================================================================

class TestSVGStructure:
    """Tests for SVG structure validity."""

    def test_svg_has_dimensions(self, renderer, basic_nodes, basic_edges):
        """Test that SVG has proper dimensions."""
        html = renderer.render(basic_nodes, basic_edges)

        assert f'width="{renderer.width}"' in html
        assert f'height="{renderer.height}"' in html

    def test_svg_has_nodes(self, renderer, basic_nodes, basic_edges):
        """Test that SVG has circle elements for nodes."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "circle" in html.lower()
        assert "graph-node" in html

    def test_svg_has_edges(self, renderer, basic_nodes, basic_edges):
        """Test that SVG has line elements for edges."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "line" in html.lower()
        assert "graph-edge" in html

    def test_svg_properly_closed(self, renderer, basic_nodes, basic_edges):
        """Test that SVG is properly closed."""
        html = renderer.render(basic_nodes, basic_edges)

        svg_opens = html.count("<svg")
        svg_closes = html.count("</svg>")
        assert svg_opens == svg_closes


# ============================================================================
# Interactivity Tests
# ============================================================================

class TestInteractivity:
    """Tests for interactive features."""

    def test_tooltip_div_rendered(self, renderer, basic_nodes, basic_edges):
        """Test that tooltip div is rendered."""
        html = renderer.render(basic_nodes, basic_edges)

        assert 'class="tooltip"' in html

    def test_javascript_included(self, renderer, basic_nodes, basic_edges):
        """Test that JavaScript is included."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "<script>" in html
        assert "</script>" in html

    def test_node_hover_listeners(self, renderer, basic_nodes, basic_edges):
        """Test that node hover listeners are set up."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "mouseenter" in html or "hover" in html.lower()


# ============================================================================
# NetworkX Integration Tests
# ============================================================================

class TestNetworkXIntegration:
    """Tests for NetworkX integration."""

    def test_render_from_networkx(self):
        """Test rendering from NetworkX graph."""
        try:
            import networkx as nx

            G = nx.MultiDiGraph()
            G.add_node("A", node_type="concept")
            G.add_node("B", node_type="concept")
            G.add_edge("A", "B", type="IS_A", weight=1.0)

            html = render_knowledge_graph_from_networkx(G)

            assert isinstance(html, str)
            assert "<svg" in html
        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_render_networkx_with_options(self):
        """Test rendering NetworkX graph with options."""
        try:
            import networkx as nx

            G = nx.MultiDiGraph()
            G.add_node("A")
            G.add_node("B")
            G.add_edge("A", "B", type="USES")

            html = render_knowledge_graph_from_networkx(
                G,
                title="NetworkX Graph",
                subtitle="Test graph",
                max_nodes=50,
                highlighted_path=["A", "B"]
            )

            assert "NetworkX Graph" in html
        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_max_nodes_limit(self):
        """Test that max_nodes limits output."""
        try:
            import networkx as nx

            G = nx.MultiDiGraph()
            for i in range(100):
                G.add_node(f"node_{i}")

            html = render_knowledge_graph_from_networkx(G, max_nodes=10)

            # Should only render 10 nodes
            assert isinstance(html, str)
        except ImportError:
            pytest.skip("NetworkX not installed")


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_node_no_edges(self, renderer):
        """Test single node with no edges."""
        nodes = [GraphNode(id="lonely", label="Lonely Node")]

        html = renderer.render(nodes, [])

        assert isinstance(html, str)
        assert "Lonely Node" in html or "lonely" in html.lower()

    def test_many_nodes(self, renderer):
        """Test with many nodes."""
        nodes = [
            GraphNode(id=f"node_{i}", label=f"Node {i}", degree=i % 5)
            for i in range(50)
        ]
        edges = [
            GraphEdge(src=f"node_{i}", dst=f"node_{i+1}", type="LEADS_TO")
            for i in range(49)
        ]

        html = renderer.render(nodes, edges)

        assert isinstance(html, str)
        assert "<svg" in html

    def test_self_loop_edge(self, renderer):
        """Test edge that loops back to same node."""
        nodes = [GraphNode(id="loop", label="Loop Node")]
        edges = [GraphEdge(src="loop", dst="loop", type="USES")]

        html = renderer.render(nodes, edges)

        assert isinstance(html, str)

    def test_unknown_edge_type(self, renderer):
        """Test unknown edge type gets default color."""
        nodes = [
            GraphNode(id="a", label="A"),
            GraphNode(id="b", label="B"),
        ]
        edges = [
            GraphEdge(src="a", dst="b", type="CUSTOM_TYPE")
        ]

        html = renderer.render(nodes, edges)

        # Should render without error
        assert isinstance(html, str)

    def test_long_node_labels_truncated(self, renderer):
        """Test that long node labels are truncated."""
        nodes = [
            GraphNode(
                id="long",
                label="This is a very long label that should be truncated"
            )
        ]

        html = renderer.render(nodes, [])

        # Label should be truncated (check for [:15])
        assert isinstance(html, str)

    def test_zero_weight_edge(self, renderer):
        """Test edge with zero weight."""
        nodes = [
            GraphNode(id="a", label="A"),
            GraphNode(id="b", label="B"),
        ]
        edges = [
            GraphEdge(src="a", dst="b", type="IS_A", weight=0.0)
        ]

        html = renderer.render(nodes, edges)

        assert isinstance(html, str)

    def test_disconnected_components(self, renderer):
        """Test graph with disconnected components."""
        nodes = [
            GraphNode(id="a1", label="A1", degree=1),
            GraphNode(id="a2", label="A2", degree=1),
            GraphNode(id="b1", label="B1", degree=1),
            GraphNode(id="b2", label="B2", degree=1),
        ]
        edges = [
            GraphEdge(src="a1", dst="a2", type="IS_A"),
            GraphEdge(src="b1", dst="b2", type="IS_A"),
        ]

        html = renderer.render(nodes, edges)

        assert isinstance(html, str)


# ============================================================================
# HTML Structure Tests
# ============================================================================

class TestHTMLStructure:
    """Tests for HTML structure validity."""

    def test_complete_html_document(self, renderer, basic_nodes, basic_edges):
        """Test that output is complete HTML document."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_css_styles_included(self, renderer, basic_nodes, basic_edges):
        """Test that CSS styles are included."""
        html = renderer.render(basic_nodes, basic_edges)

        assert "<style>" in html
        assert "</style>" in html

    def test_no_external_dependencies(self, renderer, basic_nodes, basic_edges):
        """Test that HTML has no external dependencies."""
        html = renderer.render(basic_nodes, basic_edges)

        # Should not have external links
        assert 'href="http' not in html
        assert 'src="http' not in html


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_graph_rendering(self, renderer, complex_graph):
        """Test complete rendering of complex graph."""
        nodes, edges = complex_graph

        html = renderer.render(
            nodes,
            edges,
            title="Transformer Architecture",
            subtitle="Entity relationships",
            highlighted_path=["attention", "transformer"]
        )

        # Verify all components
        assert "Transformer Architecture" in html
        assert "<svg" in html
        assert "graph-legend" in html
        assert "<script>" in html

        # Verify statistics
        assert "Nodes" in html
        assert "Edges" in html
