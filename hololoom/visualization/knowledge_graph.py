"""
HoloLoom Knowledge Graph Network Visualization
==============================================
Tufte-style network visualization for YarnGraph (KG) relationships.

Features:
- Force-directed layout (Fruchterman-Reingold algorithm)
- Node sizing by degree/importance
- Typed edges with semantic colors
- Path highlighting
- Interactive tooltips
- Zero external dependencies (pure HTML/CSS/SVG)

Tufte Principles Applied:
- Maximize data-ink ratio: Minimal decoration, focus on relationships
- Direct labeling: Node names inline, edge types on hover
- Data density: Show full graph structure efficiently
- Meaning first: Relationship types clearly distinguished by color

Integration:
- Direct integration with HoloLoom.memory.graph.KG
- Accepts NetworkX MultiDiGraph
- Programmatic API for automated rendering

Author: HoloLoom Team
Date: 2025-10-29
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import math
import random


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GraphNode:
    """Node in the knowledge graph visualization."""
    id: str
    label: str
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0  # Velocity for force simulation
    vy: float = 0.0
    degree: int = 0
    node_type: str = "default"
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in the knowledge graph visualization."""
    src: str
    dst: str
    type: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class EdgeType(Enum):
    """Common edge types with semantic colors."""
    IS_A = "is_a"           # Taxonomy (blue)
    USES = "uses"           # Functional (green)
    MENTIONS = "mentions"   # Reference (gray)
    LEADS_TO = "leads_to"   # Causal (orange)
    PART_OF = "part_of"     # Composition (purple)
    IN_TIME = "in_time"     # Temporal (cyan)
    OCCURRED_AT = "occurred_at"  # Event (teal)
    UNKNOWN = "unknown"     # Fallback (lightgray)


# ============================================================================
# Force-Directed Layout Algorithm
# ============================================================================

class ForceDirectedLayout:
    """
    Fruchterman-Reingold force-directed layout algorithm.

    Pure Python implementation for positioning graph nodes.

    Physics Model:
    - Repulsion: All nodes repel each other (inverse square law)
    - Attraction: Connected nodes attract (spring force)
    - Cooling: Gradually reduce movement over iterations

    Parameters:
    - iterations: Number of simulation steps (default 300)
    - attraction_strength: Spring constant for edges (default 0.01)
    - repulsion_strength: Charge strength for nodes (default 1000)
    - damping: Velocity damping factor (default 0.8)
    """

    def __init__(
        self,
        width: float = 800,
        height: float = 600,
        iterations: int = 300,
        attraction_strength: float = 0.01,
        repulsion_strength: float = 1000,
        damping: float = 0.8
    ):
        self.width = width
        self.height = height
        self.iterations = iterations
        self.attraction_strength = attraction_strength
        self.repulsion_strength = repulsion_strength
        self.damping = damping

    def layout(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[GraphNode]:
        """
        Compute force-directed layout for nodes.

        Args:
            nodes: List of GraphNode objects
            edges: List of GraphEdge objects

        Returns:
            List of GraphNode objects with updated positions
        """
        if not nodes:
            return []

        # Initialize random positions
        for node in nodes:
            node.x = random.uniform(0.2 * self.width, 0.8 * self.width)
            node.y = random.uniform(0.2 * self.height, 0.8 * self.height)
            node.vx = 0.0
            node.vy = 0.0

        # Build adjacency for fast edge lookup
        adjacency: Dict[str, Set[str]] = {}
        for edge in edges:
            if edge.src not in adjacency:
                adjacency[edge.src] = set()
            adjacency[edge.src].add(edge.dst)

        # Simulation loop
        for iteration in range(self.iterations):
            # Calculate cooling factor (linearly decrease)
            cooling = 1.0 - (iteration / self.iterations)

            # Reset forces
            for node in nodes:
                node.vx *= self.damping
                node.vy *= self.damping

            # Repulsion forces (all pairs)
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i+1:]:
                    dx = node_b.x - node_a.x
                    dy = node_b.y - node_a.y
                    distance_sq = max(dx*dx + dy*dy, 1.0)  # Avoid division by zero
                    distance = math.sqrt(distance_sq)

                    # Repulsion force (inverse square)
                    force = self.repulsion_strength / distance_sq
                    fx = (dx / distance) * force
                    fy = (dy / distance) * force

                    node_a.vx -= fx
                    node_a.vy -= fy
                    node_b.vx += fx
                    node_b.vy += fy

            # Attraction forces (edges)
            for edge in edges:
                src_node = next((n for n in nodes if n.id == edge.src), None)
                dst_node = next((n for n in nodes if n.id == edge.dst), None)

                if src_node and dst_node:
                    dx = dst_node.x - src_node.x
                    dy = dst_node.y - src_node.y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance > 0:
                        # Spring force (linear)
                        force = self.attraction_strength * distance * edge.weight
                        fx = (dx / distance) * force
                        fy = (dy / distance) * force

                        src_node.vx += fx
                        src_node.vy += fy
                        dst_node.vx -= fx
                        dst_node.vy -= fy

            # Update positions
            for node in nodes:
                node.x += node.vx * cooling
                node.y += node.vy * cooling

                # Keep within bounds (with margin)
                margin = 50
                node.x = max(margin, min(self.width - margin, node.x))
                node.y = max(margin, min(self.height - margin, node.y))

        return nodes


# ============================================================================
# Knowledge Graph Network Renderer
# ============================================================================

class KnowledgeGraphRenderer:
    """
    Render knowledge graph networks with force-directed layout.

    Tufte-Inspired Design:
    - High data-ink ratio: Minimal decoration, maximum information
    - Semantic colors: Edge types clearly distinguished
    - Direct labeling: Node names inline (no legend lookup)
    - Interactive tooltips: Details on demand

    Features:
    - Force-directed layout for natural clustering
    - Node sizing by degree (importance)
    - Edge type filtering
    - Path highlighting
    - Responsive sizing

    Thread Safety:
    - Stateless rendering (no shared mutable state)
    - Safe for concurrent calls

    Usage:
        renderer = KnowledgeGraphRenderer()
        html = renderer.render(nodes, edges, title="My Knowledge Graph")
    """

    # Edge type color mapping (semantic colors)
    EDGE_COLORS = {
        EdgeType.IS_A: "#3b82f6",        # Blue (taxonomy)
        EdgeType.USES: "#10b981",         # Green (functional)
        EdgeType.MENTIONS: "#6b7280",     # Gray (reference)
        EdgeType.LEADS_TO: "#f59e0b",     # Orange (causal)
        EdgeType.PART_OF: "#8b5cf6",      # Purple (composition)
        EdgeType.IN_TIME: "#06b6d4",      # Cyan (temporal)
        EdgeType.OCCURRED_AT: "#14b8a6",  # Teal (event)
        EdgeType.UNKNOWN: "#d1d5db"       # Light gray (fallback)
    }

    def __init__(
        self,
        width: int = 900,
        height: int = 700,
        node_size_min: int = 8,
        node_size_max: int = 24,
        show_edge_labels: bool = False,
        layout_iterations: int = 300
    ):
        """
        Initialize renderer.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            node_size_min: Minimum node radius
            node_size_max: Maximum node radius
            show_edge_labels: Show edge type labels inline
            layout_iterations: Force simulation iterations
        """
        self.width = width
        self.height = height
        self.node_size_min = node_size_min
        self.node_size_max = node_size_max
        self.show_edge_labels = show_edge_labels
        self.layout_iterations = layout_iterations

    def render(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        title: str = "Knowledge Graph Network",
        subtitle: Optional[str] = None,
        highlighted_path: Optional[List[str]] = None
    ) -> str:
        """
        Render knowledge graph network to HTML.

        Args:
            nodes: List of GraphNode objects
            edges: List of GraphEdge objects
            title: Chart title
            subtitle: Optional subtitle
            highlighted_path: Optional path to highlight (list of node IDs)

        Returns:
            Complete HTML document as string
        """
        if not nodes:
            return self._render_empty(title)

        # Compute layout
        layout = ForceDirectedLayout(
            width=self.width,
            height=self.height,
            iterations=self.layout_iterations
        )
        positioned_nodes = layout.layout(nodes.copy(), edges)

        # Build HTML
        html_parts = [
            self._render_html_head(title),
            self._render_styles(),
            f'<body>',
            f'<div class="graph-container">',
            f'<div class="graph-header">',
            f'<h2>{title}</h2>',
        ]

        if subtitle:
            html_parts.append(f'<p class="graph-subtitle">{subtitle}</p>')

        # Statistics
        stats_html = self._render_statistics(positioned_nodes, edges)
        html_parts.append(stats_html)

        html_parts.append(f'</div>')  # Close header

        # Legend
        legend_html = self._render_legend()
        html_parts.append(legend_html)

        # SVG Network
        svg_html = self._render_svg_network(
            positioned_nodes,
            edges,
            highlighted_path
        )
        html_parts.append(svg_html)

        html_parts.extend([
            f'</div>',  # Close container
            f'</body>',
            f'</html>'
        ])

        return '\n'.join(html_parts)

    def _render_html_head(self, title: str) -> str:
        """Render HTML head with metadata."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>'''

    def _render_styles(self) -> str:
        """Render CSS styles (Tufte-inspired)."""
        return '''
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #ffffff;
        color: #1f2937;
        padding: 40px 20px;
    }

    .graph-container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 30px;
    }

    .graph-header {
        margin-bottom: 20px;
    }

    .graph-header h2 {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }

    .graph-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 12px;
    }

    .graph-stats {
        display: flex;
        gap: 24px;
        padding: 12px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .stat-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .stat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #9ca3af;
        font-weight: 500;
    }

    .stat-value {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
    }

    .graph-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        padding: 12px;
        background: #f9fafb;
        border-radius: 4px;
        margin-bottom: 20px;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #4b5563;
    }

    .legend-line {
        width: 20px;
        height: 2px;
    }

    .graph-svg {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        background: #fafafa;
    }

    .graph-node {
        cursor: pointer;
        transition: all 0.2s;
    }

    .graph-node:hover {
        stroke-width: 2.5px;
    }

    .graph-edge {
        pointer-events: none;
    }

    .graph-edge.highlighted {
        stroke-width: 3px;
        opacity: 1.0;
    }

    .node-label {
        font-size: 11px;
        font-weight: 500;
        pointer-events: none;
        user-select: none;
    }

    .edge-label {
        font-size: 9px;
        fill: #6b7280;
        pointer-events: none;
        user-select: none;
    }

    /* Tooltip */
    .tooltip {
        position: absolute;
        background: rgba(17, 24, 39, 0.95);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s;
        z-index: 1000;
        max-width: 250px;
    }

    .tooltip.visible {
        opacity: 1;
    }

    .tooltip-title {
        font-weight: 600;
        margin-bottom: 4px;
        font-size: 13px;
    }

    .tooltip-detail {
        font-size: 11px;
        color: #d1d5db;
        margin: 2px 0;
    }
</style>
</head>'''

    def _render_statistics(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> str:
        """Render graph statistics."""
        num_nodes = len(nodes)
        num_edges = len(edges)
        avg_degree = sum(n.degree for n in nodes) / max(1, num_nodes)

        # Count edge types
        edge_types = {}
        for edge in edges:
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

        return f'''
        <div class="graph-stats">
            <div class="stat-item">
                <span class="stat-label">Nodes</span>
                <span class="stat-value">{num_nodes}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Edges</span>
                <span class="stat-value">{num_edges}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Avg Degree</span>
                <span class="stat-value">{avg_degree:.1f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Edge Types</span>
                <span class="stat-value">{len(edge_types)}</span>
            </div>
        </div>'''

    def _render_legend(self) -> str:
        """Render edge type legend."""
        legend_items = []
        for edge_type, color in self.EDGE_COLORS.items():
            if edge_type != EdgeType.UNKNOWN:
                label = edge_type.value.replace('_', ' ').title()
                legend_items.append(
                    f'<div class="legend-item">'
                    f'<div class="legend-line" style="background: {color};"></div>'
                    f'<span>{label}</span>'
                    f'</div>'
                )

        return f'''
        <div class="graph-legend">
            {' '.join(legend_items)}
        </div>'''

    def _render_svg_network(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        highlighted_path: Optional[List[str]] = None
    ) -> str:
        """Render SVG network visualization."""
        # Calculate node sizes based on degree
        if nodes:
            max_degree = max(n.degree for n in nodes)
            min_degree = min(n.degree for n in nodes)
            degree_range = max(max_degree - min_degree, 1)

        svg_parts = [
            f'<svg class="graph-svg" width="{self.width}" height="{self.height}" '
            f'xmlns="http://www.w3.org/2000/svg">',
        ]

        # Render edges first (so they're behind nodes)
        highlighted_set = set(highlighted_path) if highlighted_path else set()

        for edge in edges:
            src_node = next((n for n in nodes if n.id == edge.src), None)
            dst_node = next((n for n in nodes if n.id == edge.dst), None)

            if src_node and dst_node:
                # Determine if edge is highlighted
                is_highlighted = (
                    edge.src in highlighted_set and
                    edge.dst in highlighted_set
                )

                edge_svg = self._render_edge(
                    src_node,
                    dst_node,
                    edge,
                    is_highlighted
                )
                svg_parts.append(edge_svg)

        # Render nodes
        for node in nodes:
            # Calculate node size
            if degree_range > 0:
                size_factor = (node.degree - min_degree) / degree_range
            else:
                size_factor = 0.5

            radius = self.node_size_min + size_factor * (self.node_size_max - self.node_size_min)

            is_highlighted = node.id in highlighted_set
            node_svg = self._render_node(node, radius, is_highlighted)
            svg_parts.append(node_svg)

        svg_parts.append('</svg>')

        # Add tooltip div
        svg_parts.append('<div class="tooltip" id="tooltip"></div>')

        # Add interactivity script
        svg_parts.append(self._render_script(nodes, edges))

        return '\n'.join(svg_parts)

    def _render_edge(
        self,
        src: GraphNode,
        dst: GraphNode,
        edge: GraphEdge,
        highlighted: bool
    ) -> str:
        """Render single edge."""
        # Get edge color based on type
        edge_type_enum = self._normalize_edge_type(edge.type)
        color = self.EDGE_COLORS.get(edge_type_enum, self.EDGE_COLORS[EdgeType.UNKNOWN])

        # Edge styling
        opacity = 1.0 if highlighted else 0.4
        stroke_width = 3 if highlighted else 1.5
        highlight_class = "highlighted" if highlighted else ""

        # Draw arrow
        dx = dst.x - src.x
        dy = dst.y - src.y
        length = math.sqrt(dx*dx + dy*dy)

        if length > 0:
            # Shorten line to not overlap with nodes
            offset = 15  # Node radius buffer
            ratio = (length - offset) / length
            end_x = src.x + dx * ratio
            end_y = src.y + dy * ratio
        else:
            end_x = dst.x
            end_y = dst.y

        edge_svg = f'''
    <line class="graph-edge {highlight_class}"
          x1="{src.x}" y1="{src.y}"
          x2="{end_x}" y2="{end_y}"
          stroke="{color}"
          stroke-width="{stroke_width}"
          opacity="{opacity}"
          marker-end="url(#arrowhead-{edge_type_enum.value})" />'''

        # Optional edge label
        if self.show_edge_labels:
            mid_x = (src.x + dst.x) / 2
            mid_y = (src.y + dst.y) / 2
            label = edge.type.replace('_', ' ')
            edge_svg += f'''
    <text class="edge-label" x="{mid_x}" y="{mid_y}" text-anchor="middle">{label}</text>'''

        return edge_svg

    def _render_node(
        self,
        node: GraphNode,
        radius: float,
        highlighted: bool
    ) -> str:
        """Render single node."""
        # Node styling
        fill_color = "#3b82f6" if highlighted else "#6366f1"
        stroke_color = "#1e40af" if highlighted else "#4f46e5"
        stroke_width = 2.5 if highlighted else 1.5

        node_svg = f'''
    <circle class="graph-node"
            data-node-id="{node.id}"
            cx="{node.x}" cy="{node.y}" r="{radius}"
            fill="{fill_color}"
            stroke="{stroke_color}"
            stroke-width="{stroke_width}" />
    <text class="node-label"
          x="{node.x}" y="{node.y + radius + 12}"
          text-anchor="middle">{node.label[:15]}</text>'''

        return node_svg

    def _render_script(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> str:
        """Render JavaScript for interactivity."""
        # Build node data for tooltips
        node_data_js = []
        for node in nodes:
            node_data_js.append(f'''
        {{
            id: "{node.id}",
            label: "{node.label}",
            degree: {node.degree},
            type: "{node.node_type}"
        }}''')

        return f'''
<script>
(function() {{
    const nodes = [{','.join(node_data_js)}];
    const tooltip = document.getElementById('tooltip');

    // Add hover listeners to nodes
    document.querySelectorAll('.graph-node').forEach(node => {{
        node.addEventListener('mouseenter', function(e) {{
            const nodeId = this.getAttribute('data-node-id');
            const nodeData = nodes.find(n => n.id === nodeId);

            if (nodeData) {{
                tooltip.innerHTML = `
                    <div class="tooltip-title">${{nodeData.label}}</div>
                    <div class="tooltip-detail">Degree: ${{nodeData.degree}}</div>
                    <div class="tooltip-detail">Type: ${{nodeData.type}}</div>
                `;
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
                tooltip.classList.add('visible');
            }}
        }});

        node.addEventListener('mouseleave', function() {{
            tooltip.classList.remove('visible');
        }});

        node.addEventListener('mousemove', function(e) {{
            tooltip.style.left = e.pageX + 10 + 'px';
            tooltip.style.top = e.pageY + 10 + 'px';
        }});
    }});
}})();
</script>'''

    def _render_empty(self, title: str) -> str:
        """Render empty state."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #f9fafb;
        }}
        .empty-state {{
            text-align: center;
            color: #6b7280;
        }}
        .empty-state h2 {{
            font-size: 24px;
            margin-bottom: 12px;
        }}
        .empty-state p {{
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="empty-state">
        <h2>{title}</h2>
        <p>No nodes available to display</p>
    </div>
</body>
</html>'''

    def _normalize_edge_type(self, type_str: str) -> EdgeType:
        """Normalize edge type string to enum."""
        type_lower = type_str.lower()
        for edge_type in EdgeType:
            if edge_type.value == type_lower:
                return edge_type
        return EdgeType.UNKNOWN


# ============================================================================
# Convenience Functions
# ============================================================================

def render_knowledge_graph_from_kg(
    kg,  # HoloLoom.memory.graph.KG
    title: str = "Knowledge Graph Network",
    subtitle: Optional[str] = None,
    max_nodes: int = 50,
    highlighted_path: Optional[List[str]] = None
) -> str:
    """
    Render knowledge graph directly from HoloLoom KG object.

    Primary programmatic API for automated tool calling.

    Args:
        kg: HoloLoom.memory.graph.KG instance
        title: Chart title
        subtitle: Optional subtitle
        max_nodes: Maximum nodes to render (for large graphs)
        highlighted_path: Optional path to highlight

    Returns:
        Complete HTML document as string

    Example:
        from HoloLoom.memory.graph import KG
        from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg

        kg = KG()
        # ... add edges ...
        html = render_knowledge_graph_from_kg(kg, title="My Domain Model")

        with open('graph.html', 'w') as f:
            f.write(html)
    """
    # Extract nodes and edges from KG
    nodes = []
    edges = []

    # Get all nodes (limit to max_nodes)
    all_nodes = list(kg.G.nodes())[:max_nodes]

    for node_id in all_nodes:
        node_data = kg.G.nodes.get(node_id, {})
        degree = kg.G.degree(node_id)

        graph_node = GraphNode(
            id=node_id,
            label=node_id,
            degree=degree,
            node_type=node_data.get('node_type', 'default'),
            metadata=node_data
        )
        nodes.append(graph_node)

    # Get all edges between included nodes
    node_id_set = set(all_nodes)
    for src, dst, key, data in kg.G.edges(keys=True, data=True):
        if src in node_id_set and dst in node_id_set:
            edge_type = data.get('type', 'unknown')
            weight = data.get('weight', 1.0)

            graph_edge = GraphEdge(
                src=src,
                dst=dst,
                type=edge_type,
                weight=weight,
                metadata=data
            )
            edges.append(graph_edge)

    # Render
    renderer = KnowledgeGraphRenderer()
    return renderer.render(
        nodes,
        edges,
        title=title,
        subtitle=subtitle,
        highlighted_path=highlighted_path
    )


def render_knowledge_graph_from_networkx(
    graph,  # nx.MultiDiGraph
    title: str = "Knowledge Graph Network",
    subtitle: Optional[str] = None,
    max_nodes: int = 50,
    highlighted_path: Optional[List[str]] = None
) -> str:
    """
    Render knowledge graph from NetworkX MultiDiGraph.

    Alternative API for direct NetworkX integration.

    Args:
        graph: NetworkX MultiDiGraph
        title: Chart title
        subtitle: Optional subtitle
        max_nodes: Maximum nodes to render
        highlighted_path: Optional path to highlight

    Returns:
        Complete HTML document as string
    """
    nodes = []
    edges = []

    # Extract nodes (limit to max_nodes)
    all_nodes = list(graph.nodes())[:max_nodes]

    for node_id in all_nodes:
        node_data = graph.nodes.get(node_id, {})
        degree = graph.degree(node_id)

        graph_node = GraphNode(
            id=node_id,
            label=str(node_id),
            degree=degree,
            node_type=node_data.get('node_type', 'default'),
            metadata=node_data
        )
        nodes.append(graph_node)

    # Extract edges
    node_id_set = set(all_nodes)
    for src, dst, key, data in graph.edges(keys=True, data=True):
        if src in node_id_set and dst in node_id_set:
            edge_type = data.get('type', 'unknown')
            weight = data.get('weight', 1.0)

            graph_edge = GraphEdge(
                src=src,
                dst=dst,
                type=edge_type,
                weight=weight,
                metadata=data
            )
            edges.append(graph_edge)

    # Render
    renderer = KnowledgeGraphRenderer()
    return renderer.render(
        nodes,
        edges,
        title=title,
        subtitle=subtitle,
        highlighted_path=highlighted_path
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=== Knowledge Graph Network Demo ===\n")

    # Create sample graph
    nodes = [
        GraphNode("attention", "attention", degree=4, node_type="concept"),
        GraphNode("transformer", "transformer", degree=5, node_type="concept"),
        GraphNode("neural_network", "neural network", degree=3, node_type="concept"),
        GraphNode("BERT", "BERT", degree=2, node_type="model"),
        GraphNode("GPT", "GPT", degree=2, node_type="model"),
        GraphNode("multi-head", "multi-head", degree=2, node_type="technique"),
        GraphNode("self-attention", "self-attention", degree=2, node_type="technique"),
    ]

    edges = [
        GraphEdge("attention", "transformer", "USES", 1.0),
        GraphEdge("transformer", "neural_network", "IS_A", 1.0),
        GraphEdge("attention", "neural_network", "PART_OF", 0.8),
        GraphEdge("BERT", "transformer", "IS_A", 1.0),
        GraphEdge("GPT", "transformer", "IS_A", 1.0),
        GraphEdge("multi-head", "attention", "IS_A", 1.0),
        GraphEdge("self-attention", "attention", "IS_A", 1.0),
    ]

    renderer = KnowledgeGraphRenderer()
    html = renderer.render(
        nodes,
        edges,
        title="Transformer Architecture Knowledge Graph",
        subtitle="Entity relationships in neural network domain"
    )

    output_path = "demo_knowledge_graph.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Demo generated: {output_path}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print("\nOpen in browser to view interactive network!")
