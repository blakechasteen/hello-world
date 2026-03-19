"""
Interactive Circuit Explorer for Dark Trace

Visualizes neural circuit graphs discovered through activation patching
and causal tracing. Supports interactive exploration with node highlighting,
path tracing, and importance-based layout.

Author: HoloLoom Team
Created: December 2025
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(Enum):
    """Types of circuit nodes."""
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"
    INPUT = "input"
    OUTPUT = "output"
    SAE_FEATURE = "sae_feature"
    SEMANTIC = "semantic"


@dataclass
class CircuitNode:
    """A node in the circuit graph."""
    id: str
    label: str
    node_type: NodeType
    layer: int
    importance: float = 0.0
    activation: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Get node color based on type."""
        colors = {
            NodeType.ATTENTION: "#e94560",
            NodeType.MLP: "#0f9b0f",
            NodeType.RESIDUAL: "#1a73e8",
            NodeType.INPUT: "#9c27b0",
            NodeType.OUTPUT: "#ff9800",
            NodeType.SAE_FEATURE: "#00bcd4",
            NodeType.SEMANTIC: "#e91e63",
        }
        return colors.get(self.node_type, "#888888")


@dataclass
class CircuitEdge:
    """An edge in the circuit graph."""
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "causal"
    metadata: dict[str, Any] = field(default_factory=dict)


class CircuitExplorer:
    """
    Interactive circuit explorer visualization.

    Uses force-directed layout with layer constraints for
    readable neural circuit visualizations.
    """

    def __init__(
        self,
        width: int = 900,
        height: int = 600,
        node_radius: int = 20,
        layer_spacing: int = 150,
    ):
        self.width = width
        self.height = height
        self.node_radius = node_radius
        self.layer_spacing = layer_spacing
        self.nodes: list[CircuitNode] = []
        self.edges: list[CircuitEdge] = []

    def add_node(self, node: CircuitNode) -> None:
        """Add a node to the circuit."""
        self.nodes.append(node)

    def add_edge(self, edge: CircuitEdge) -> None:
        """Add an edge to the circuit."""
        self.edges.append(edge)

    def add_nodes(self, nodes: list[CircuitNode]) -> None:
        """Add multiple nodes."""
        self.nodes.extend(nodes)

    def add_edges(self, edges: list[CircuitEdge]) -> None:
        """Add multiple edges."""
        self.edges.extend(edges)

    def _compute_layout(self) -> dict[str, tuple[float, float]]:
        """Compute node positions using layer-constrained force layout."""
        positions: dict[str, tuple[float, float]] = {}

        # Group nodes by layer
        layers: dict[int, list[CircuitNode]] = {}
        for node in self.nodes:
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)

        # Compute positions
        n_layers = max(layers.keys()) - min(layers.keys()) + 1 if layers else 1
        min_layer = min(layers.keys()) if layers else 0

        for layer_idx, layer_nodes in layers.items():
            # X position based on layer
            x = 80 + (layer_idx - min_layer) * self.layer_spacing

            # Y positions distributed vertically
            n_nodes = len(layer_nodes)
            for i, node in enumerate(layer_nodes):
                y = (i + 1) * self.height / (n_nodes + 1)
                positions[node.id] = (x, y)

        return positions

    def render(
        self,
        title: str = "Circuit Explorer",
        highlighted_path: list[str] | None = None,
        show_importance: bool = True,
    ) -> str:
        """Render the circuit as interactive HTML/SVG."""
        positions = self._compute_layout()
        highlighted_set = set(highlighted_path or [])

        # Build SVG elements
        edges_svg = self._render_edges(positions, highlighted_set)
        nodes_svg = self._render_nodes(positions, highlighted_set, show_importance)

        # Legend
        legend_svg = self._render_legend()

        # JavaScript for interactivity
        script = self._render_script()

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #e94560;
            --border: #333;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent);
        }}
        .circuit-container {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
        .node {{
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .node:hover {{
            transform: scale(1.1);
        }}
        .node-label {{
            font-size: 10px;
            pointer-events: none;
        }}
        .edge {{
            stroke-opacity: 0.6;
            transition: stroke-opacity 0.2s;
        }}
        .edge.highlighted {{
            stroke-opacity: 1;
            stroke-width: 3;
        }}
        .tooltip {{
            position: absolute;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(22, 33, 62, 0.9);
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .info-panel {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    <div class="circuit-container" style="position: relative;">
        <svg width="{self.width}" height="{self.height}" id="circuit-svg">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7"
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#888"/>
                </marker>
                <marker id="arrowhead-highlight" markerWidth="10" markerHeight="7"
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="{self.nodes[0].color if self.nodes else '#e94560'}"/>
                </marker>
            </defs>
            <g class="edges">
                {edges_svg}
            </g>
            <g class="nodes">
                {nodes_svg}
            </g>
        </svg>
        {legend_svg}
        <div class="tooltip" id="tooltip"></div>
    </div>
    <div class="info-panel" id="info-panel">
        <strong>Circuit Summary</strong><br>
        Nodes: {len(self.nodes)} | Edges: {len(self.edges)} |
        Layers: {len(set(n.layer for n in self.nodes))}
    </div>
</div>
{script}
</body>
</html>
"""

    def _render_edges(
        self,
        positions: dict[str, tuple[float, float]],
        highlighted: set[str]
    ) -> str:
        """Render edge SVG elements."""
        edges_svg = []

        for edge in self.edges:
            if edge.source not in positions or edge.target not in positions:
                continue

            x1, y1 = positions[edge.source]
            x2, y2 = positions[edge.target]

            # Shorten edge to not overlap with nodes
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx /= length
                dy /= length

            x1 += dx * self.node_radius
            y1 += dy * self.node_radius
            x2 -= dx * (self.node_radius + 10)  # Extra space for arrowhead
            y2 -= dy * (self.node_radius + 10)

            is_highlighted = edge.source in highlighted and edge.target in highlighted
            highlight_class = "highlighted" if is_highlighted else ""
            stroke = "#e94560" if is_highlighted else "#888"
            marker = "arrowhead-highlight" if is_highlighted else "arrowhead"
            opacity = min(1.0, edge.weight)

            edges_svg.append(
                f'<line class="edge {highlight_class}" '
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="{1 + edge.weight}" '
                f'stroke-opacity="{opacity}" '
                f'marker-end="url(#{marker})" '
                f'data-source="{edge.source}" data-target="{edge.target}" '
                f'data-weight="{edge.weight:.3f}"/>'
            )

        return "\n                ".join(edges_svg)

    def _render_nodes(
        self,
        positions: dict[str, tuple[float, float]],
        highlighted: set[str],
        show_importance: bool
    ) -> str:
        """Render node SVG elements."""
        nodes_svg = []

        for node in self.nodes:
            if node.id not in positions:
                continue

            x, y = positions[node.id]
            is_highlighted = node.id in highlighted

            # Adjust radius based on importance if enabled
            radius = self.node_radius
            if show_importance and node.importance > 0:
                radius = self.node_radius * (0.8 + 0.4 * node.importance)

            stroke = "#fff" if is_highlighted else node.color
            stroke_width = 3 if is_highlighted else 1

            # Node metadata for tooltip
            metadata_json = json.dumps({
                "id": node.id,
                "label": node.label,
                "type": node.node_type.value,
                "layer": node.layer,
                "importance": f"{node.importance:.3f}",
                "activation": f"{node.activation:.3f}",
            })

            nodes_svg.append(f"""
                <g class="node" data-node='{metadata_json}' transform="translate({x}, {y})">
                    <circle r="{radius}" fill="{node.color}"
                            stroke="{stroke}" stroke-width="{stroke_width}"/>
                    <text class="node-label" text-anchor="middle" dy="4" fill="#fff">
                        {node.label[:6]}
                    </text>
                </g>
""")

        return "".join(nodes_svg)

    def _render_legend(self) -> str:
        """Render legend."""
        items = []
        for node_type in NodeType:
            node = CircuitNode(id="", label="", node_type=node_type, layer=0)
            items.append(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background: {node.color}"></div>'
                f'{node_type.value.replace("_", " ").title()}'
                f'</div>'
            )

        return f"""
        <div class="legend">
            <strong>Node Types</strong>
            {"".join(items)}
        </div>
"""

    def _render_script(self) -> str:
        """Render JavaScript for interactivity."""
        return """
<script>
    const tooltip = document.getElementById('tooltip');
    const infoPanel = document.getElementById('info-panel');

    document.querySelectorAll('.node').forEach(node => {
        node.addEventListener('mouseenter', (e) => {
            const data = JSON.parse(node.dataset.node);
            tooltip.innerHTML = `
                <strong>${data.label}</strong><br>
                Type: ${data.type}<br>
                Layer: ${data.layer}<br>
                Importance: ${data.importance}<br>
                Activation: ${data.activation}
            `;
            tooltip.style.opacity = 1;
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';

            // Highlight connected edges
            document.querySelectorAll('.edge').forEach(edge => {
                if (edge.dataset.source === data.id || edge.dataset.target === data.id) {
                    edge.classList.add('highlighted');
                }
            });
        });

        node.addEventListener('mouseleave', () => {
            tooltip.style.opacity = 0;
            document.querySelectorAll('.edge').forEach(edge => {
                edge.classList.remove('highlighted');
            });
        });

        node.addEventListener('click', (e) => {
            const data = JSON.parse(node.dataset.node);
            infoPanel.innerHTML = `
                <strong>Selected Node: ${data.label}</strong><br>
                ID: ${data.id}<br>
                Type: ${data.type}<br>
                Layer: ${data.layer}<br>
                Importance: ${data.importance}<br>
                Activation: ${data.activation}
            `;
        });
    });

    document.addEventListener('mousemove', (e) => {
        if (tooltip.style.opacity === '1') {
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';
        }
    });
</script>
"""

    def to_json(self) -> str:
        """Export circuit as JSON."""
        return json.dumps({
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "type": n.node_type.value,
                    "layer": n.layer,
                    "importance": n.importance,
                    "activation": n.activation,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "type": e.edge_type,
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
        }, indent=2)


def render_circuit_graph(
    nodes: list[CircuitNode],
    edges: list[CircuitEdge],
    title: str = "Circuit Explorer",
    highlighted_path: list[str] | None = None,
    width: int = 900,
    height: int = 600,
) -> str:
    """Convenience function to render a circuit graph."""
    explorer = CircuitExplorer(width=width, height=height)
    explorer.add_nodes(nodes)
    explorer.add_edges(edges)
    return explorer.render(title=title, highlighted_path=highlighted_path)
