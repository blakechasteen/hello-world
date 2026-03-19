"""
Dark Trace Circuit Visualizer: Interactive Circuit Visualization

This module provides visualization capabilities for circuit graphs,
enabling interactive exploration of mechanistic interpretability results.

Research Basis:
- Circuit visualization best practices from Anthropic
- Graph layout algorithms (force-directed, hierarchical)
- Interactive web-based visualization (HTML/SVG)

Key Features:
- Multiple output formats (HTML, SVG, DOT)
- Hierarchical layer-based layout
- Safety-critical path highlighting
- Vulnerability markers
- Interactive tooltips
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
)

from hololoom.dark_trace.circuits.graph import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
    CircuitPath,
    EdgeType,
    NodeType,
)


class LayoutAlgorithm(Enum):
    """Supported layout algorithms for circuit visualization."""

    HIERARCHICAL = "hierarchical"  # Layer-based top-to-bottom
    FORCE_DIRECTED = "force_directed"  # Spring-based physics
    RADIAL = "radial"  # Circular with layers as rings
    ORTHOGONAL = "orthogonal"  # Grid-based with right angles


class OutputFormat(Enum):
    """Supported output formats for visualization."""

    HTML = "html"  # Interactive HTML with SVG
    SVG = "svg"  # Static SVG
    DOT = "dot"  # Graphviz DOT format
    JSON = "json"  # JSON for custom rendering


class ColorScheme(Enum):
    """Color schemes for visualization."""

    DEFAULT = "default"
    SAFETY = "safety"  # Highlights safety-relevant features
    ACTIVATION = "activation"  # Color by activation level
    IMPORTANCE = "importance"  # Color by importance score
    LAYER = "layer"  # Color by layer index


@dataclass
class CircuitVisualizationConfig:
    """Configuration for circuit visualization.

    Attributes:
        layout: Layout algorithm to use
        output_format: Output format for visualization
        color_scheme: Color scheme for nodes/edges
        width: Canvas width in pixels
        height: Canvas height in pixels
        node_radius: Base radius for nodes
        edge_width: Base width for edges
        show_labels: Whether to show node labels
        show_weights: Whether to show edge weights
        highlight_safety_paths: Highlight safety-critical paths
        highlight_vulnerabilities: Highlight vulnerability nodes
        interactive: Enable interactive features (tooltips, zoom)
        animation_duration: Animation duration in ms (0 for none)
        font_size: Font size for labels
        layer_spacing: Vertical spacing between layers
        node_spacing: Horizontal spacing between nodes
    """

    layout: LayoutAlgorithm = LayoutAlgorithm.HIERARCHICAL
    output_format: OutputFormat = OutputFormat.HTML
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    width: int = 1200
    height: int = 800
    node_radius: float = 20.0
    edge_width: float = 2.0
    show_labels: bool = True
    show_weights: bool = True
    highlight_safety_paths: bool = True
    highlight_vulnerabilities: bool = True
    interactive: bool = True
    animation_duration: int = 300
    font_size: int = 12
    layer_spacing: int = 100
    node_spacing: int = 80

    @classmethod
    def minimal(cls) -> "CircuitVisualizationConfig":
        """Minimal configuration for quick visualization."""
        return cls(
            show_labels=False,
            show_weights=False,
            interactive=False,
            animation_duration=0,
        )

    @classmethod
    def research(cls) -> "CircuitVisualizationConfig":
        """Research configuration with full detail."""
        return cls(
            width=1600,
            height=1000,
            show_labels=True,
            show_weights=True,
            highlight_safety_paths=True,
            highlight_vulnerabilities=True,
            interactive=True,
        )

    @classmethod
    def safety(cls) -> "CircuitVisualizationConfig":
        """Safety-focused configuration highlighting risks."""
        return cls(
            color_scheme=ColorScheme.SAFETY,
            highlight_safety_paths=True,
            highlight_vulnerabilities=True,
            node_radius=25.0,
            edge_width=3.0,
        )


@dataclass
class NodePosition:
    """Position of a node in the visualization."""

    x: float
    y: float
    node_id: str


@dataclass
class RenderedNode:
    """A node rendered for visualization."""

    node_id: str
    x: float
    y: float
    radius: float
    color: str
    stroke_color: str
    stroke_width: float
    label: str
    tooltip: str
    is_highlighted: bool = False
    is_vulnerable: bool = False
    opacity: float = 1.0


@dataclass
class RenderedEdge:
    """An edge rendered for visualization."""

    source_x: float
    source_y: float
    target_x: float
    target_y: float
    width: float
    color: str
    is_curved: bool = False
    control_x: float | None = None
    control_y: float | None = None
    label: str | None = None
    is_highlighted: bool = False
    opacity: float = 1.0
    dash_array: str | None = None


@dataclass
class CircuitDiagram:
    """Rendered circuit diagram ready for output.

    Contains all rendered elements and can be exported
    to various formats.
    """

    nodes: list[RenderedNode]
    edges: list[RenderedEdge]
    width: int
    height: int
    title: str = "Circuit Diagram"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save diagram to file.

        Format determined by file extension:
        - .html: Interactive HTML
        - .svg: Static SVG
        - .dot: Graphviz DOT
        - .json: JSON data
        """
        ext = path.rsplit(".", 1)[-1].lower()

        if ext == "html":
            content = self.to_html()
        elif ext == "svg":
            content = self.to_svg()
        elif ext == "dot":
            content = self.to_dot()
        elif ext == "json":
            content = self.to_json()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def to_html(self) -> str:
        """Export as interactive HTML."""
        svg_content = self._render_svg_content()

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        .container {{
            max-width: {self.width + 40}px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .description {{
            color: #888;
            margin-bottom: 20px;
        }}
        .diagram-container {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
        .node {{
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .node:hover {{
            opacity: 0.8;
        }}
        .edge {{
            transition: opacity 0.2s;
        }}
        .tooltip {{
            position: absolute;
            background: #0f0f23;
            color: #eee;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            border: 1px solid #333;
        }}
        .tooltip.visible {{
            opacity: 1;
        }}
        .legend {{
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .metadata {{
            margin-top: 20px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p class="description">{self.description}</p>
        <div class="diagram-container">
            <svg width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                    </marker>
                    <marker id="arrowhead-highlighted" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#00d4ff"/>
                    </marker>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                {svg_content}
            </svg>
        </div>
        <div class="legend">
            {self._render_legend()}
        </div>
        <div class="metadata">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
            Nodes: {len(self.nodes)} |
            Edges: {len(self.edges)}
        </div>
    </div>
    <div class="tooltip" id="tooltip"></div>
    <script>
        const tooltip = document.getElementById('tooltip');

        document.querySelectorAll('.node').forEach(node => {{
            node.addEventListener('mouseenter', (e) => {{
                const text = node.getAttribute('data-tooltip');
                if (text) {{
                    tooltip.innerHTML = text;
                    tooltip.classList.add('visible');
                }}
            }});

            node.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            }});

            node.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('visible');
            }});
        }});
    </script>
</body>
</html>"""

    def to_svg(self) -> str:
        """Export as static SVG."""
        svg_content = self._render_svg_content()

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
        </marker>
    </defs>
    <rect width="100%" height="100%" fill="#16213e"/>
    {svg_content}
</svg>"""

    def to_dot(self) -> str:
        """Export as Graphviz DOT format."""
        lines = [
            f'digraph "{self.title}" {{',
            '    rankdir=TB;',
            '    node [shape=circle, style=filled];',
            '    edge [arrowsize=0.8];',
            '',
        ]

        # Add nodes
        for node in self.nodes:
            attrs = [
                f'fillcolor="{node.color}"',
                f'label="{node.label}"',
                f'tooltip="{node.tooltip}"',
            ]
            if node.is_highlighted:
                attrs.append('penwidth=3')
            lines.append(f'    "{node.node_id}" [{", ".join(attrs)}];')

        lines.append('')

        # Add edges - need to track source/target from rendered positions
        # This is a simplified version
        for i, edge in enumerate(self.edges):
            attrs = [f'color="{edge.color}"']
            if edge.label:
                attrs.append(f'label="{edge.label}"')
            if edge.is_highlighted:
                attrs.append('penwidth=3')
            if edge.dash_array:
                attrs.append('style=dashed')
            # Note: DOT format needs actual node IDs, not positions
            lines.append(f'    // Edge {i}: style=[{", ".join(attrs)}]')

        lines.append('}')

        return '\n'.join(lines)

    def to_json(self) -> str:
        """Export as JSON data."""
        data = {
            "title": self.title,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "nodes": [
                {
                    "id": n.node_id,
                    "x": n.x,
                    "y": n.y,
                    "radius": n.radius,
                    "color": n.color,
                    "label": n.label,
                    "is_highlighted": n.is_highlighted,
                    "is_vulnerable": n.is_vulnerable,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_x": e.source_x,
                    "source_y": e.source_y,
                    "target_x": e.target_x,
                    "target_y": e.target_y,
                    "width": e.width,
                    "color": e.color,
                    "label": e.label,
                    "is_highlighted": e.is_highlighted,
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=2)

    def _render_svg_content(self) -> str:
        """Render SVG content (edges and nodes)."""
        parts = []

        # Render edges first (behind nodes)
        parts.append('<g class="edges">')
        for edge in self.edges:
            parts.append(self._render_edge_svg(edge))
        parts.append('</g>')

        # Render nodes
        parts.append('<g class="nodes">')
        for node in self.nodes:
            parts.append(self._render_node_svg(node))
        parts.append('</g>')

        return '\n'.join(parts)

    def _render_node_svg(self, node: RenderedNode) -> str:
        """Render a single node as SVG."""
        filter_attr = 'filter="url(#glow)"' if node.is_highlighted else ''

        parts = [
            f'<g class="node" data-tooltip="{node.tooltip}" transform="translate({node.x}, {node.y})">',
            f'  <circle r="{node.radius}" fill="{node.color}" stroke="{node.stroke_color}" '
            f'stroke-width="{node.stroke_width}" opacity="{node.opacity}" {filter_attr}/>',
        ]

        if node.label:
            parts.append(
                f'  <text text-anchor="middle" dy="0.35em" fill="#fff" '
                f'font-size="10" font-weight="500">{node.label[:8]}</text>'
            )

        if node.is_vulnerable:
            parts.append(
                f'  <circle r="{node.radius + 5}" fill="none" stroke="#ff4444" '
                f'stroke-width="2" stroke-dasharray="4,2"/>'
            )

        parts.append('</g>')

        return '\n'.join(parts)

    def _render_edge_svg(self, edge: RenderedEdge) -> str:
        """Render a single edge as SVG."""
        marker = "arrowhead-highlighted" if edge.is_highlighted else "arrowhead"
        dash = f'stroke-dasharray="{edge.dash_array}"' if edge.dash_array else ''

        if edge.is_curved and edge.control_x is not None:
            path = (
                f'M {edge.source_x} {edge.source_y} '
                f'Q {edge.control_x} {edge.control_y} '
                f'{edge.target_x} {edge.target_y}'
            )
        else:
            path = f'M {edge.source_x} {edge.source_y} L {edge.target_x} {edge.target_y}'

        parts = [
            '<g class="edge">',
            f'  <path d="{path}" fill="none" stroke="{edge.color}" '
            f'stroke-width="{edge.width}" opacity="{edge.opacity}" '
            f'marker-end="url(#{marker})" {dash}/>',
        ]

        if edge.label:
            mid_x = (edge.source_x + edge.target_x) / 2
            mid_y = (edge.source_y + edge.target_y) / 2
            parts.append(
                f'  <text x="{mid_x}" y="{mid_y - 8}" text-anchor="middle" '
                f'fill="#888" font-size="10">{edge.label}</text>'
            )

        parts.append('</g>')

        return '\n'.join(parts)

    def _render_legend(self) -> str:
        """Render the legend HTML."""
        items = [
            ("#4ecdc4", "Input"),
            ("#ff6b6b", "Output"),
            ("#45b7d1", "Attention"),
            ("#96ceb4", "MLP"),
            ("#ffeaa7", "SAE Feature"),
        ]

        parts = []
        for color, label in items:
            parts.append(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background: {color}"></div>'
                f'<span>{label}</span>'
                f'</div>'
            )

        return '\n'.join(parts)


class CircuitVisualizer:
    """Visualizer for circuit graphs.

    Creates visual representations of circuits for
    mechanistic interpretability analysis.

    Example:
        graph = CircuitGraph.from_trace(trace)
        viz = CircuitVisualizer(graph)
        diagram = viz.create_diagram(title="My Circuit")
        diagram.save("circuit.html")
    """

    # Color palettes for different node types
    NODE_COLORS: dict[NodeType, str] = {
        NodeType.INPUT: "#4ecdc4",
        NodeType.OUTPUT: "#ff6b6b",
        NodeType.EMBEDDING: "#a29bfe",
        NodeType.ATTENTION_HEAD: "#45b7d1",
        NodeType.ATTENTION_PATTERN: "#74b9ff",
        NodeType.MLP_NEURON: "#96ceb4",
        NodeType.MLP_LAYER: "#55efc4",
        NodeType.SAE_FEATURE: "#ffeaa7",
        NodeType.SAE_ENCODER: "#fdcb6e",
        NodeType.SAE_DECODER: "#f39c12",
        NodeType.RESIDUAL: "#dfe6e9",
        NodeType.LAYER_NORM: "#b2bec3",
    }

    # Colors for edge types
    EDGE_COLORS: dict[EdgeType, str] = {
        EdgeType.CAUSAL_STRONG: "#00d4ff",
        EdgeType.CAUSAL_WEAK: "#74b9ff",
        EdgeType.CORRELATIONAL: "#a29bfe",
        EdgeType.FEEDFORWARD: "#55efc4",
        EdgeType.ATTENTION: "#ffeaa7",
        EdgeType.RESIDUAL: "#dfe6e9",
        EdgeType.SKIP: "#fd79a8",
        EdgeType.INHIBITORY: "#ff7675",
        EdgeType.INFORMATION_FLOW: "#81ecec",
        EdgeType.COMPOSITION: "#fab1a0",
        EdgeType.SUPERPOSITION: "#e17055",
    }

    def __init__(
        self,
        graph: CircuitGraph,
        config: CircuitVisualizationConfig | None = None,
    ):
        """Initialize visualizer.

        Args:
            graph: Circuit graph to visualize
            config: Visualization configuration
        """
        self.graph = graph
        self.config = config or CircuitVisualizationConfig()
        self._positions: dict[str, NodePosition] = {}
        self._highlighted_nodes: set[str] = set()
        self._highlighted_edges: set[tuple[str, str]] = set()
        self._vulnerable_nodes: set[str] = set()

    def create_diagram(
        self,
        title: str = "Circuit Diagram",
        description: str = "",
        safety_paths: list[CircuitPath] | None = None,
        vulnerable_nodes: list[str] | None = None,
    ) -> CircuitDiagram:
        """Create a circuit diagram.

        Args:
            title: Diagram title
            description: Diagram description
            safety_paths: Paths to highlight as safety-critical
            vulnerable_nodes: Nodes to mark as vulnerable

        Returns:
            Rendered circuit diagram
        """
        # Set highlighted elements
        if self.config.highlight_safety_paths and safety_paths:
            for path in safety_paths:
                for node in path.nodes:
                    self._highlighted_nodes.add(node.node_id)
                for edge in path.edges:
                    self._highlighted_edges.add((edge.source_id, edge.target_id))

        if self.config.highlight_vulnerabilities and vulnerable_nodes:
            self._vulnerable_nodes = set(vulnerable_nodes)

        # Calculate positions
        self._calculate_layout()

        # Render nodes and edges
        rendered_nodes = self._render_nodes()
        rendered_edges = self._render_edges()

        return CircuitDiagram(
            nodes=rendered_nodes,
            edges=rendered_edges,
            width=self.config.width,
            height=self.config.height,
            title=title,
            description=description,
            metadata={
                "node_count": len(self.graph._nodes),
                "edge_count": len(self.graph._edges),
                "layout": self.config.layout.value,
                "color_scheme": self.config.color_scheme.value,
            },
        )

    def highlight_path(self, path: CircuitPath) -> None:
        """Highlight a specific path in the visualization."""
        for node in path.nodes:
            self._highlighted_nodes.add(node.node_id)
        for edge in path.edges:
            self._highlighted_edges.add((edge.source_id, edge.target_id))

    def mark_vulnerable(self, node_ids: list[str]) -> None:
        """Mark nodes as vulnerable."""
        self._vulnerable_nodes.update(node_ids)

    def _calculate_layout(self) -> None:
        """Calculate node positions based on layout algorithm."""
        if self.config.layout == LayoutAlgorithm.HIERARCHICAL:
            self._hierarchical_layout()
        elif self.config.layout == LayoutAlgorithm.FORCE_DIRECTED:
            self._force_directed_layout()
        elif self.config.layout == LayoutAlgorithm.RADIAL:
            self._radial_layout()
        else:
            self._hierarchical_layout()  # Default

    def _hierarchical_layout(self) -> None:
        """Calculate hierarchical layer-based layout."""
        # Group nodes by layer
        layers: dict[int, list[CircuitNode]] = {}
        for node in self.graph._nodes.values():
            layer = node.layer_idx
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        if not layers:
            return

        # Calculate positions
        num_layers = max(layers.keys()) - min(layers.keys()) + 1
        layer_height = (self.config.height - 100) / max(num_layers, 1)

        min_layer = min(layers.keys())

        for layer_idx, nodes in sorted(layers.items()):
            # Vertical position based on layer
            y = 50 + (layer_idx - min_layer) * layer_height

            # Horizontal distribution
            num_nodes = len(nodes)
            if num_nodes == 1:
                positions = [self.config.width / 2]
            else:
                spacing = min(
                    self.config.node_spacing,
                    (self.config.width - 100) / (num_nodes - 1)
                )
                start_x = (self.config.width - spacing * (num_nodes - 1)) / 2
                positions = [start_x + i * spacing for i in range(num_nodes)]

            for i, node in enumerate(nodes):
                self._positions[node.node_id] = NodePosition(
                    x=positions[i],
                    y=y,
                    node_id=node.node_id,
                )

    def _force_directed_layout(self) -> None:
        """Calculate force-directed layout using simple physics simulation."""
        nodes = list(self.graph._nodes.values())
        if not nodes:
            return

        # Initialize random positions
        import random
        random.seed(42)

        positions: dict[str, tuple[float, float]] = {}
        for node in nodes:
            positions[node.node_id] = (
                random.uniform(100, self.config.width - 100),
                random.uniform(100, self.config.height - 100),
            )

        # Build adjacency for attraction forces
        adjacency: dict[str, set[str]] = {n.node_id: set() for n in nodes}
        for edge in self.graph._edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        # Simulation parameters
        iterations = 100
        repulsion = 5000
        attraction = 0.01
        damping = 0.9

        velocities: dict[str, tuple[float, float]] = {
            n.node_id: (0.0, 0.0) for n in nodes
        }

        for _ in range(iterations):
            forces: dict[str, tuple[float, float]] = {
                n.node_id: (0.0, 0.0) for n in nodes
            }

            # Repulsion between all pairs
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    dx = positions[n1.node_id][0] - positions[n2.node_id][0]
                    dy = positions[n1.node_id][1] - positions[n2.node_id][1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.1

                    force = repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist

                    forces[n1.node_id] = (
                        forces[n1.node_id][0] + fx,
                        forces[n1.node_id][1] + fy,
                    )
                    forces[n2.node_id] = (
                        forces[n2.node_id][0] - fx,
                        forces[n2.node_id][1] - fy,
                    )

            # Attraction along edges
            for node in nodes:
                for neighbor_id in adjacency[node.node_id]:
                    dx = positions[neighbor_id][0] - positions[node.node_id][0]
                    dy = positions[neighbor_id][1] - positions[node.node_id][1]

                    forces[node.node_id] = (
                        forces[node.node_id][0] + attraction * dx,
                        forces[node.node_id][1] + attraction * dy,
                    )

            # Update positions
            for node in nodes:
                vx = (velocities[node.node_id][0] + forces[node.node_id][0]) * damping
                vy = (velocities[node.node_id][1] + forces[node.node_id][1]) * damping
                velocities[node.node_id] = (vx, vy)

                new_x = positions[node.node_id][0] + vx
                new_y = positions[node.node_id][1] + vy

                # Boundary constraints
                new_x = max(50, min(self.config.width - 50, new_x))
                new_y = max(50, min(self.config.height - 50, new_y))

                positions[node.node_id] = (new_x, new_y)

        # Store final positions
        for node_id, (x, y) in positions.items():
            self._positions[node_id] = NodePosition(x=x, y=y, node_id=node_id)

    def _radial_layout(self) -> None:
        """Calculate radial layout with layers as concentric rings."""
        # Group nodes by layer
        layers: dict[int, list[CircuitNode]] = {}
        for node in self.graph._nodes.values():
            layer = node.layer_idx
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        if not layers:
            return

        center_x = self.config.width / 2
        center_y = self.config.height / 2

        num_layers = len(layers)
        max_radius = min(center_x, center_y) - 50
        radius_step = max_radius / max(num_layers, 1)

        sorted_layers = sorted(layers.keys())

        for ring_idx, layer_idx in enumerate(sorted_layers):
            nodes = layers[layer_idx]
            radius = (ring_idx + 1) * radius_step

            num_nodes = len(nodes)
            angle_step = 2 * math.pi / max(num_nodes, 1)

            for i, node in enumerate(nodes):
                angle = i * angle_step - math.pi / 2
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                self._positions[node.node_id] = NodePosition(
                    x=x,
                    y=y,
                    node_id=node.node_id,
                )

    def _render_nodes(self) -> list[RenderedNode]:
        """Render all nodes."""
        rendered = []

        for node_id, node in self.graph._nodes.items():
            if node_id not in self._positions:
                continue

            pos = self._positions[node_id]
            color = self._get_node_color(node)
            is_highlighted = node_id in self._highlighted_nodes
            is_vulnerable = node_id in self._vulnerable_nodes

            # Adjust radius based on importance
            radius = self.config.node_radius
            if is_highlighted:
                radius *= 1.3
            if node.importance > 0.8:
                radius *= 1.2

            # Create tooltip content
            tooltip = self._create_tooltip(node)

            rendered.append(RenderedNode(
                node_id=node_id,
                x=pos.x,
                y=pos.y,
                radius=radius,
                color=color,
                stroke_color="#fff" if is_highlighted else "#333",
                stroke_width=3 if is_highlighted else 1,
                label=self._get_node_label(node) if self.config.show_labels else "",
                tooltip=tooltip,
                is_highlighted=is_highlighted,
                is_vulnerable=is_vulnerable,
                opacity=1.0 if is_highlighted else 0.9,
            ))

        return rendered

    def _render_edges(self) -> list[RenderedEdge]:
        """Render all edges."""
        rendered = []

        for edge in self.graph._edges:
            if edge.source_id not in self._positions:
                continue
            if edge.target_id not in self._positions:
                continue

            source_pos = self._positions[edge.source_id]
            target_pos = self._positions[edge.target_id]

            is_highlighted = (edge.source_id, edge.target_id) in self._highlighted_edges
            color = self._get_edge_color(edge)

            # Adjust width based on weight
            width = self.config.edge_width
            if is_highlighted:
                width *= 2
            width *= max(0.5, min(2.0, edge.weight))

            # Calculate edge points (with offset for node radius)
            dx = target_pos.x - source_pos.x
            dy = target_pos.y - source_pos.y
            dist = math.sqrt(dx * dx + dy * dy) + 0.1

            # Get node radii for offset calculation
            source_node = self.graph._nodes.get(edge.source_id)
            target_node = self.graph._nodes.get(edge.target_id)
            source_radius = self.config.node_radius
            target_radius = self.config.node_radius

            offset_x = dx / dist
            offset_y = dy / dist

            source_x = source_pos.x + offset_x * source_radius
            source_y = source_pos.y + offset_y * source_radius
            target_x = target_pos.x - offset_x * (target_radius + 10)
            target_y = target_pos.y - offset_y * (target_radius + 10)

            # Label for weight if enabled
            label = None
            if self.config.show_weights and edge.weight != 1.0:
                label = f"{edge.weight:.2f}"

            # Dash array for non-causal edges
            dash_array = None
            if not edge.is_causal:
                dash_array = "5,3"

            rendered.append(RenderedEdge(
                source_x=source_x,
                source_y=source_y,
                target_x=target_x,
                target_y=target_y,
                width=width,
                color=color,
                is_curved=False,
                label=label,
                is_highlighted=is_highlighted,
                opacity=1.0 if is_highlighted else 0.6,
                dash_array=dash_array,
            ))

        return rendered

    def _get_node_color(self, node: CircuitNode) -> str:
        """Get color for a node based on configuration."""
        if self.config.color_scheme == ColorScheme.ACTIVATION:
            # Gradient from blue (low) to red (high)
            r = int(255 * node.activation)
            b = int(255 * (1 - node.activation))
            return f"rgb({r}, 100, {b})"

        elif self.config.color_scheme == ColorScheme.IMPORTANCE:
            # Gradient from gray (low) to gold (high)
            intensity = int(200 + 55 * node.importance)
            return f"rgb({intensity}, {int(intensity * 0.8)}, {int(intensity * 0.4)})"

        elif self.config.color_scheme == ColorScheme.LAYER:
            # Hue based on layer
            hue = (node.layer_idx * 30) % 360
            return f"hsl({hue}, 70%, 60%)"

        elif self.config.color_scheme == ColorScheme.SAFETY:
            # Red for safety-relevant, normal otherwise
            if node.node_id in self._highlighted_nodes:
                return "#ff4757"
            if node.node_id in self._vulnerable_nodes:
                return "#ffa502"
            return self.NODE_COLORS.get(node.node_type, "#636e72")

        else:
            return self.NODE_COLORS.get(node.node_type, "#636e72")

    def _get_edge_color(self, edge: CircuitEdge) -> str:
        """Get color for an edge."""
        if (edge.source_id, edge.target_id) in self._highlighted_edges:
            return "#00d4ff"
        return self.EDGE_COLORS.get(edge.edge_type, "#636e72")

    def _get_node_label(self, node: CircuitNode) -> str:
        """Get display label for a node."""
        if node.feature_idx is not None:
            return f"F{node.feature_idx}"

        type_abbrev = {
            NodeType.INPUT: "IN",
            NodeType.OUTPUT: "OUT",
            NodeType.EMBEDDING: "EMB",
            NodeType.ATTENTION_HEAD: "ATT",
            NodeType.ATTENTION_PATTERN: "AP",
            NodeType.MLP_NEURON: "MLP",
            NodeType.MLP_LAYER: "ML",
            NodeType.SAE_FEATURE: "SAE",
            NodeType.SAE_ENCODER: "ENC",
            NodeType.SAE_DECODER: "DEC",
            NodeType.RESIDUAL: "RES",
            NodeType.LAYER_NORM: "LN",
        }

        abbrev = type_abbrev.get(node.node_type, "?")
        return f"{abbrev}{node.layer_idx}"

    def _create_tooltip(self, node: CircuitNode) -> str:
        """Create tooltip content for a node."""
        lines = [
            f"<b>{node.node_id}</b>",
            f"Type: {node.node_type.value}",
            f"Layer: {node.layer_idx}",
            f"Activation: {node.activation:.3f}",
            f"Importance: {node.importance:.3f}",
        ]

        if node.feature_idx is not None:
            lines.append(f"Feature: {node.feature_idx}")

        if node.metadata:
            for key, value in list(node.metadata.items())[:3]:
                lines.append(f"{key}: {value}")

        return "<br>".join(lines)


# Convenience functions

def visualize_circuit(
    graph: CircuitGraph,
    output_path: str,
    title: str = "Circuit Diagram",
    **config_kwargs,
) -> CircuitDiagram:
    """Convenience function to visualize a circuit graph.

    Args:
        graph: Circuit graph to visualize
        output_path: Path to save output
        title: Diagram title
        **config_kwargs: Configuration options

    Returns:
        The created diagram
    """
    config = CircuitVisualizationConfig(**config_kwargs)
    visualizer = CircuitVisualizer(graph, config)
    diagram = visualizer.create_diagram(title=title)
    diagram.save(output_path)
    return diagram


def quick_visualize(
    graph: CircuitGraph,
    output_path: str = "circuit.html",
) -> CircuitDiagram:
    """Quick visualization with defaults.

    Args:
        graph: Circuit graph to visualize
        output_path: Output file path

    Returns:
        The created diagram
    """
    return visualize_circuit(graph, output_path)
