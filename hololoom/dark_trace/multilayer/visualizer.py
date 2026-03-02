"""
Hierarchy Visualization for Multi-Layer SAE Analysis

Creates visualizations for:
- Feature hierarchies across layers
- Information flow diagrams
- Feature evolution plots
- Abstraction ladders

Design Philosophy (Edward Tufte):
- Maximize data-ink ratio
- Show the data directly
- Minimize chartjunk
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json


class VisualizationFormat(Enum):
    """Output format for visualizations."""
    HTML = "html"
    SVG = "svg"
    JSON = "json"  # For custom rendering
    MERMAID = "mermaid"  # For documentation


class ColorScheme(Enum):
    """Color schemes for visualizations."""
    DEFAULT = "default"  # Blue-green gradient
    WARM = "warm"  # Red-orange gradient
    COOL = "cool"  # Blue-purple gradient
    SEMANTIC = "semantic"  # Based on abstraction level
    CONTRAST = "contrast"  # High contrast for accessibility


@dataclass
class VisualizationConfig:
    """Configuration for hierarchy visualization."""

    # Output settings
    format: VisualizationFormat = VisualizationFormat.HTML
    width: int = 1200
    height: int = 800

    # Visual style
    color_scheme: ColorScheme = ColorScheme.SEMANTIC
    show_labels: bool = True
    show_tooltips: bool = True
    animate_transitions: bool = True

    # Layout
    layer_spacing: int = 120  # Vertical space between layers
    node_spacing: int = 40  # Horizontal space between nodes
    node_min_size: int = 8
    node_max_size: int = 32

    # Filtering
    min_activation: float = 0.1  # Minimum activation to show
    max_nodes_per_layer: int = 50  # Limit nodes for readability
    min_edge_weight: float = 0.3  # Minimum edge weight to show

    # Flow visualization
    show_flow_direction: bool = True
    flow_arrow_size: int = 8
    edge_opacity: float = 0.6

    # Evolution plots
    show_confidence_bands: bool = True
    sparkline_height: int = 40

    @classmethod
    def default(cls) -> "VisualizationConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def compact(cls) -> "VisualizationConfig":
        """Compact visualization for embedding."""
        return cls(
            width=600,
            height=400,
            layer_spacing=80,
            node_spacing=25,
            max_nodes_per_layer=30,
            show_tooltips=False,
            animate_transitions=False
        )

    @classmethod
    def detailed(cls) -> "VisualizationConfig":
        """Detailed visualization for analysis."""
        return cls(
            width=1600,
            height=1000,
            layer_spacing=150,
            max_nodes_per_layer=100,
            min_edge_weight=0.1,
            show_confidence_bands=True
        )


@dataclass
class HierarchyDiagram:
    """Rendered hierarchy diagram."""

    # Content
    content: str = ""  # HTML/SVG/JSON content
    format: VisualizationFormat = VisualizationFormat.HTML

    # Metadata
    n_layers: int = 0
    n_nodes: int = 0
    n_edges: int = 0

    # Dimensions
    width: int = 0
    height: int = 0

    # Interactive elements
    has_tooltips: bool = False
    has_zoom: bool = False
    has_pan: bool = False

    def save(self, path: str) -> None:
        """Save diagram to file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "format": self.format.value,
            "n_layers": self.n_layers,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "width": self.width,
            "height": self.height,
            "has_tooltips": self.has_tooltips,
            "has_zoom": self.has_zoom,
            "has_pan": self.has_pan
        }


@dataclass
class FlowDiagram:
    """Rendered information flow diagram."""

    content: str = ""
    format: VisualizationFormat = VisualizationFormat.HTML

    # Flow statistics
    total_flow: float = 0.0
    bottleneck_layer: Optional[int] = None
    compression_ratio: float = 1.0

    # Dimensions
    width: int = 0
    height: int = 0

    def save(self, path: str) -> None:
        """Save diagram to file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "format": self.format.value,
            "total_flow": self.total_flow,
            "bottleneck_layer": self.bottleneck_layer,
            "compression_ratio": self.compression_ratio,
            "width": self.width,
            "height": self.height
        }


@dataclass
class EvolutionPlot:
    """Feature evolution plot across layers."""

    content: str = ""
    format: VisualizationFormat = VisualizationFormat.HTML

    # Feature info
    feature_id: str = ""
    emergence_layer: int = 0
    peak_layer: int = 0

    # Trajectory
    layer_activations: Dict[int, float] = field(default_factory=dict)

    # Dimensions
    width: int = 0
    height: int = 0

    def save(self, path: str) -> None:
        """Save plot to file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "format": self.format.value,
            "feature_id": self.feature_id,
            "emergence_layer": self.emergence_layer,
            "peak_layer": self.peak_layer,
            "layer_activations": self.layer_activations,
            "width": self.width,
            "height": self.height
        }


class HierarchyVisualizer:
    """
    Visualizes feature hierarchies and information flow.

    Creates HTML/SVG visualizations following Tufte principles:
    - High data-ink ratio
    - Direct labeling
    - Minimal decoration

    Usage:
        hierarchy_analyzer = FeatureHierarchyAnalyzer(manager)
        flow_analyzer = InformationFlowAnalyzer(manager)

        visualizer = HierarchyVisualizer(
            hierarchy_analyzer,
            flow_analyzer,
            config=VisualizationConfig.default()
        )

        diagram = visualizer.create_hierarchy_diagram()
        diagram.save("hierarchy.html")

        flow = visualizer.create_flow_diagram()
        evolution = visualizer.create_evolution_plot("L3.42")
    """

    # Color palettes for different schemes
    COLOR_PALETTES = {
        ColorScheme.DEFAULT: {
            "RAW": "#94a3b8",     # Slate
            "LOW": "#60a5fa",     # Blue
            "MID": "#34d399",     # Green
            "HIGH": "#fbbf24",    # Amber
            "ABSTRACT": "#f472b6" # Pink
        },
        ColorScheme.SEMANTIC: {
            "RAW": "#6b7280",
            "LOW": "#3b82f6",
            "MID": "#10b981",
            "HIGH": "#f59e0b",
            "ABSTRACT": "#8b5cf6"
        },
        ColorScheme.WARM: {
            "RAW": "#fca5a5",
            "LOW": "#f87171",
            "MID": "#fb923c",
            "HIGH": "#fbbf24",
            "ABSTRACT": "#fcd34d"
        },
        ColorScheme.COOL: {
            "RAW": "#a5b4fc",
            "LOW": "#818cf8",
            "MID": "#6366f1",
            "HIGH": "#8b5cf6",
            "ABSTRACT": "#a855f7"
        },
        ColorScheme.CONTRAST: {
            "RAW": "#000000",
            "LOW": "#0066cc",
            "MID": "#00aa00",
            "HIGH": "#cc6600",
            "ABSTRACT": "#cc0066"
        }
    }

    def __init__(
        self,
        hierarchy_analyzer: Any = None,  # FeatureHierarchyAnalyzer
        flow_analyzer: Any = None,  # InformationFlowAnalyzer
        config: Optional[VisualizationConfig] = None
    ):
        """
        Initialize visualizer.

        Args:
            hierarchy_analyzer: Optional FeatureHierarchyAnalyzer
            flow_analyzer: Optional InformationFlowAnalyzer
            config: Visualization configuration
        """
        self.hierarchy_analyzer = hierarchy_analyzer
        self.flow_analyzer = flow_analyzer
        self.config = config or VisualizationConfig.default()

    def create_hierarchy_diagram(
        self,
        nodes: Optional[List[Any]] = None,  # List[HierarchyNode]
        edges: Optional[List[Any]] = None,  # List[HierarchyEdge]
        test_inputs: Optional[Any] = None,
        title: str = "Feature Hierarchy"
    ) -> HierarchyDiagram:
        """
        Create hierarchy diagram showing features across layers.

        Args:
            nodes: Pre-computed hierarchy nodes (or None to compute)
            edges: Pre-computed hierarchy edges
            test_inputs: Test inputs if nodes/edges not provided
            title: Diagram title

        Returns:
            HierarchyDiagram with rendered content
        """
        # Get nodes and edges if not provided
        if nodes is None or edges is None:
            if self.hierarchy_analyzer is not None and test_inputs is not None:
                nodes, edges = self.hierarchy_analyzer.build_hierarchy_graph(
                    test_inputs,
                    top_k_per_layer=self.config.max_nodes_per_layer
                )
            else:
                nodes, edges = [], []

        # Filter by minimum activation
        if nodes:
            nodes = [n for n in nodes if n.activation >= self.config.min_activation]

        # Filter edges by minimum weight
        if edges:
            edges = [e for e in edges if e.weight >= self.config.min_edge_weight]

        # Render based on format
        if self.config.format == VisualizationFormat.HTML:
            content = self._render_hierarchy_html(nodes, edges, title)
        elif self.config.format == VisualizationFormat.SVG:
            content = self._render_hierarchy_svg(nodes, edges, title)
        elif self.config.format == VisualizationFormat.MERMAID:
            content = self._render_hierarchy_mermaid(nodes, edges, title)
        else:  # JSON
            content = self._render_hierarchy_json(nodes, edges, title)

        # Collect layer info
        layers = set()
        for node in nodes:
            if hasattr(node, 'layer_idx'):
                layers.add(node.layer_idx)

        return HierarchyDiagram(
            content=content,
            format=self.config.format,
            n_layers=len(layers),
            n_nodes=len(nodes),
            n_edges=len(edges),
            width=self.config.width,
            height=self.config.height,
            has_tooltips=self.config.show_tooltips,
            has_zoom=True,
            has_pan=True
        )

    def create_flow_diagram(
        self,
        flow_result: Optional[Any] = None,  # FlowAnalysisResult
        test_inputs: Optional[Any] = None,
        title: str = "Information Flow"
    ) -> FlowDiagram:
        """
        Create information flow diagram.

        Args:
            flow_result: Pre-computed flow analysis (or None to compute)
            test_inputs: Test inputs if flow_result not provided
            title: Diagram title

        Returns:
            FlowDiagram with rendered content
        """
        # Get flow result if not provided
        if flow_result is None:
            if self.flow_analyzer is not None and test_inputs is not None:
                flow_result = self.flow_analyzer.analyze(test_inputs)
            else:
                # Create empty result
                from hololoom.dark_trace.multilayer.flow import FlowAnalysisResult
                flow_result = FlowAnalysisResult()

        # Render based on format
        if self.config.format == VisualizationFormat.HTML:
            content = self._render_flow_html(flow_result, title)
        elif self.config.format == VisualizationFormat.SVG:
            content = self._render_flow_svg(flow_result, title)
        elif self.config.format == VisualizationFormat.MERMAID:
            content = self._render_flow_mermaid(flow_result, title)
        else:
            content = self._render_flow_json(flow_result, title)

        return FlowDiagram(
            content=content,
            format=self.config.format,
            total_flow=flow_result.total_information_preserved,
            bottleneck_layer=flow_result.bottleneck_layer,
            compression_ratio=flow_result.overall_compression_ratio,
            width=self.config.width,
            height=self.config.height
        )

    def create_evolution_plot(
        self,
        feature_id: str,
        evolution: Optional[Any] = None,  # FeatureEvolution
        test_inputs: Optional[Any] = None,
        title: Optional[str] = None
    ) -> EvolutionPlot:
        """
        Create feature evolution plot.

        Args:
            feature_id: Feature to plot
            evolution: Pre-computed evolution (or None to compute)
            test_inputs: Test inputs if evolution not provided
            title: Optional plot title

        Returns:
            EvolutionPlot with rendered content
        """
        if title is None:
            title = f"Feature Evolution: {feature_id}"

        # Get evolution if not provided
        if evolution is None:
            if self.hierarchy_analyzer is not None and test_inputs is not None:
                evolution = self.hierarchy_analyzer.track_feature_evolution(
                    feature_id, test_inputs
                )
            else:
                # Create empty evolution
                from hololoom.dark_trace.multilayer.hierarchy import FeatureEvolution
                evolution = FeatureEvolution(
                    feature_id=feature_id,
                    source_layer=0,
                    emergence_layer=0,
                    peak_layer=0,
                    layer_activations={},
                    predecessors=[],
                    successors=[],
                    semantic_drift=0.0
                )

        # Render based on format
        if self.config.format == VisualizationFormat.HTML:
            content = self._render_evolution_html(evolution, title)
        elif self.config.format == VisualizationFormat.SVG:
            content = self._render_evolution_svg(evolution, title)
        else:
            content = self._render_evolution_json(evolution, title)

        return EvolutionPlot(
            content=content,
            format=self.config.format,
            feature_id=feature_id,
            emergence_layer=evolution.emergence_layer,
            peak_layer=evolution.peak_layer,
            layer_activations=dict(evolution.layer_activations),
            width=self.config.width,
            height=self.config.sparkline_height * 4
        )

    def _get_color(self, abstraction_level: str) -> str:
        """Get color for abstraction level."""
        palette = self.COLOR_PALETTES.get(
            self.config.color_scheme,
            self.COLOR_PALETTES[ColorScheme.DEFAULT]
        )
        return palette.get(abstraction_level.upper(), "#94a3b8")

    def _render_hierarchy_html(
        self,
        nodes: List[Any],
        edges: List[Any],
        title: str
    ) -> str:
        """Render hierarchy as interactive HTML."""
        # Group nodes by layer
        layers: Dict[int, List[Any]] = {}
        for node in nodes:
            layer = node.layer_idx if hasattr(node, 'layer_idx') else 0
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        # Build SVG content
        svg_width = self.config.width
        svg_height = self.config.height

        # Calculate positions
        node_positions: Dict[str, Tuple[int, int]] = {}
        sorted_layers = sorted(layers.keys())

        for layer_idx in sorted_layers:
            layer_nodes = layers[layer_idx]
            y = 80 + layer_idx * self.config.layer_spacing

            # Center nodes horizontally
            total_width = len(layer_nodes) * self.config.node_spacing
            start_x = (svg_width - total_width) // 2

            for i, node in enumerate(layer_nodes):
                x = start_x + i * self.config.node_spacing
                node_id = node.feature_id if hasattr(node, 'feature_id') else str(i)
                node_positions[node_id] = (x, y)

        # Build SVG
        svg_lines = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            '  <defs>',
            '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '      <polygon points="0 0, 10 3.5, 0 7" fill="#666" />',
            '    </marker>',
            '  </defs>',
            f'  <text x="{svg_width//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>'
        ]

        # Draw edges
        svg_lines.append('  <g class="edges">')
        for edge in edges:
            source_id = edge.source_id if hasattr(edge, 'source_id') else None
            target_id = edge.target_id if hasattr(edge, 'target_id') else None

            if source_id in node_positions and target_id in node_positions:
                x1, y1 = node_positions[source_id]
                x2, y2 = node_positions[target_id]
                weight = edge.weight if hasattr(edge, 'weight') else 0.5
                opacity = min(1.0, weight + 0.3)

                svg_lines.append(
                    f'    <line x1="{x1}" y1="{y1+10}" x2="{x2}" y2="{y2-10}" '
                    f'stroke="#666" stroke-width="{max(1, weight * 3)}" '
                    f'opacity="{opacity}" marker-end="url(#arrowhead)" />'
                )
        svg_lines.append('  </g>')

        # Draw nodes
        svg_lines.append('  <g class="nodes">')
        for node in nodes:
            node_id = node.feature_id if hasattr(node, 'feature_id') else "?"
            if node_id not in node_positions:
                continue

            x, y = node_positions[node_id]
            activation = node.activation if hasattr(node, 'activation') else 0.5
            level = node.abstraction_level.value if hasattr(node, 'abstraction_level') else "MID"
            color = self._get_color(level)

            # Scale node size by activation
            size = self.config.node_min_size + (
                self.config.node_max_size - self.config.node_min_size
            ) * activation

            svg_lines.append(
                f'    <circle cx="{x}" cy="{y}" r="{size/2}" '
                f'fill="{color}" stroke="#333" stroke-width="1">'
            )
            if self.config.show_tooltips:
                label = node.label if hasattr(node, 'label') else node_id
                svg_lines.append(f'      <title>{label} (activation: {activation:.2f})</title>')
            svg_lines.append('    </circle>')

            if self.config.show_labels and len(nodes) < 100:
                short_label = node_id[-6:] if len(node_id) > 6 else node_id
                svg_lines.append(
                    f'    <text x="{x}" y="{y + size/2 + 12}" text-anchor="middle" '
                    f'font-size="9" fill="#333">{short_label}</text>'
                )

        svg_lines.append('  </g>')

        # Layer labels
        svg_lines.append('  <g class="layer-labels">')
        for layer_idx in sorted_layers:
            y = 80 + layer_idx * self.config.layer_spacing
            svg_lines.append(
                f'    <text x="30" y="{y + 5}" font-size="12" fill="#666">Layer {layer_idx}</text>'
            )
        svg_lines.append('  </g>')

        svg_lines.append('</svg>')

        # Wrap in HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .container {{ max-width: {svg_width}px; margin: 0 auto; }}
        svg {{ border: 1px solid #e5e7eb; border-radius: 8px; }}
        .nodes circle {{ cursor: pointer; transition: opacity 0.2s; }}
        .nodes circle:hover {{ opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        {chr(10).join(svg_lines)}
    </div>
</body>
</html>"""

        return html

    def _render_hierarchy_svg(
        self,
        nodes: List[Any],
        edges: List[Any],
        title: str
    ) -> str:
        """Render hierarchy as standalone SVG."""
        # Simplified version - extract SVG from HTML
        html = self._render_hierarchy_html(nodes, edges, title)
        # Extract SVG tag
        start = html.find('<svg')
        end = html.find('</svg>') + 6
        if start >= 0 and end > start:
            return html[start:end]
        return '<svg></svg>'

    def _render_hierarchy_mermaid(
        self,
        nodes: List[Any],
        edges: List[Any],
        title: str
    ) -> str:
        """Render hierarchy as Mermaid diagram."""
        lines = [f"graph TD", f"    %% {title}"]

        # Add nodes with styling
        for node in nodes[:50]:  # Limit for readability
            node_id = node.feature_id if hasattr(node, 'feature_id') else "node"
            # Sanitize ID for Mermaid
            safe_id = node_id.replace(".", "_").replace("-", "_")
            label = node.label if hasattr(node, 'label') else node_id
            short_label = label[:20] if len(label) > 20 else label
            lines.append(f'    {safe_id}["{short_label}"]')

        # Add edges
        for edge in edges[:100]:  # Limit
            source = edge.source_id if hasattr(edge, 'source_id') else "src"
            target = edge.target_id if hasattr(edge, 'target_id') else "tgt"
            source_safe = source.replace(".", "_").replace("-", "_")
            target_safe = target.replace(".", "_").replace("-", "_")
            weight = edge.weight if hasattr(edge, 'weight') else 0.5

            if weight > 0.7:
                lines.append(f'    {source_safe} ==> {target_safe}')
            elif weight > 0.4:
                lines.append(f'    {source_safe} --> {target_safe}')
            else:
                lines.append(f'    {source_safe} -.-> {target_safe}')

        return "\n".join(lines)

    def _render_hierarchy_json(
        self,
        nodes: List[Any],
        edges: List[Any],
        title: str
    ) -> str:
        """Render hierarchy as JSON for custom visualization."""
        data = {
            "title": title,
            "nodes": [],
            "edges": [],
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "layer_spacing": self.config.layer_spacing
            }
        }

        for node in nodes:
            data["nodes"].append({
                "id": node.feature_id if hasattr(node, 'feature_id') else "?",
                "layer": node.layer_idx if hasattr(node, 'layer_idx') else 0,
                "label": node.label if hasattr(node, 'label') else "",
                "activation": node.activation if hasattr(node, 'activation') else 0,
                "level": node.abstraction_level.value if hasattr(node, 'abstraction_level') else "mid"
            })

        for edge in edges:
            data["edges"].append({
                "source": edge.source_id if hasattr(edge, 'source_id') else "",
                "target": edge.target_id if hasattr(edge, 'target_id') else "",
                "weight": edge.weight if hasattr(edge, 'weight') else 0,
                "type": edge.relation_type.value if hasattr(edge, 'relation_type') else "unknown"
            })

        return json.dumps(data, indent=2)

    def _render_flow_html(
        self,
        flow_result: Any,
        title: str
    ) -> str:
        """Render flow diagram as HTML."""
        svg_width = self.config.width
        svg_height = self.config.height

        svg_lines = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            f'  <text x="{svg_width//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>'
        ]

        # Draw layers as boxes with flow between them
        layer_flows = flow_result.layer_flows if hasattr(flow_result, 'layer_flows') else {}
        cross_flows = flow_result.cross_layer_flows if hasattr(flow_result, 'cross_layer_flows') else {}

        sorted_layers = sorted(layer_flows.keys())
        n_layers = len(sorted_layers)

        if n_layers > 0:
            box_width = (svg_width - 200) // n_layers
            box_height = 60

            for i, layer_idx in enumerate(sorted_layers):
                x = 100 + i * box_width
                y = svg_height // 2 - box_height // 2

                flow = layer_flows[layer_idx]
                capacity = flow.capacity_used if hasattr(flow, 'capacity_used') else 0.5

                # Color based on capacity
                if capacity > 0.9:
                    color = "#ef4444"  # Red - saturated
                elif capacity > 0.7:
                    color = "#f59e0b"  # Amber - high
                elif capacity > 0.4:
                    color = "#10b981"  # Green - normal
                else:
                    color = "#6b7280"  # Gray - low

                svg_lines.append(
                    f'    <rect x="{x}" y="{y}" width="{box_width - 20}" height="{box_height}" '
                    f'fill="{color}" opacity="0.8" rx="5" />'
                )
                svg_lines.append(
                    f'    <text x="{x + box_width//2 - 10}" y="{y + 25}" text-anchor="middle" '
                    f'fill="white" font-size="12" font-weight="bold">Layer {layer_idx}</text>'
                )
                svg_lines.append(
                    f'    <text x="{x + box_width//2 - 10}" y="{y + 45}" text-anchor="middle" '
                    f'fill="white" font-size="10">{capacity:.0%} capacity</text>'
                )

            # Draw flow arrows
            for (src, tgt), cross_flow in cross_flows.items():
                if src in sorted_layers and tgt in sorted_layers:
                    src_i = sorted_layers.index(src)
                    tgt_i = sorted_layers.index(tgt)

                    x1 = 100 + src_i * box_width + box_width - 20
                    x2 = 100 + tgt_i * box_width
                    y_center = svg_height // 2

                    strength = cross_flow.flow_strength if hasattr(cross_flow, 'flow_strength') else 0.5
                    width = max(1, strength * 10)

                    svg_lines.append(
                        f'    <line x1="{x1}" y1="{y_center}" x2="{x2}" y2="{y_center}" '
                        f'stroke="#3b82f6" stroke-width="{width}" opacity="0.6" />'
                    )

        # Bottleneck indicator
        bottleneck = flow_result.bottleneck_layer if hasattr(flow_result, 'bottleneck_layer') else None
        if bottleneck is not None and bottleneck in sorted_layers:
            i = sorted_layers.index(bottleneck)
            x = 100 + i * box_width + box_width // 2 - 10
            y = svg_height // 2 + box_height // 2 + 20
            svg_lines.append(
                f'    <text x="{x}" y="{y}" text-anchor="middle" fill="#ef4444" '
                f'font-size="14">⚠ Bottleneck</text>'
            )

        svg_lines.append('</svg>')

        # Wrap in HTML
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    </style>
</head>
<body>
    {chr(10).join(svg_lines)}
</body>
</html>"""

    def _render_flow_svg(self, flow_result: Any, title: str) -> str:
        """Extract SVG from flow HTML."""
        html = self._render_flow_html(flow_result, title)
        start = html.find('<svg')
        end = html.find('</svg>') + 6
        if start >= 0 and end > start:
            return html[start:end]
        return '<svg></svg>'

    def _render_flow_mermaid(self, flow_result: Any, title: str) -> str:
        """Render flow as Mermaid diagram."""
        lines = [f"graph LR", f"    %% {title}"]

        layer_flows = flow_result.layer_flows if hasattr(flow_result, 'layer_flows') else {}
        cross_flows = flow_result.cross_layer_flows if hasattr(flow_result, 'cross_layer_flows') else {}

        for layer_idx, flow in layer_flows.items():
            capacity = flow.capacity_used if hasattr(flow, 'capacity_used') else 0.5
            lines.append(f'    L{layer_idx}["Layer {layer_idx}<br/>{capacity:.0%}"]')

        for (src, tgt), cross_flow in cross_flows.items():
            strength = cross_flow.flow_strength if hasattr(cross_flow, 'flow_strength') else 0.5
            if strength > 0.7:
                lines.append(f'    L{src} ==> L{tgt}')
            elif strength > 0.3:
                lines.append(f'    L{src} --> L{tgt}')
            else:
                lines.append(f'    L{src} -.-> L{tgt}')

        return "\n".join(lines)

    def _render_flow_json(self, flow_result: Any, title: str) -> str:
        """Render flow as JSON."""
        return json.dumps(flow_result.to_dict() if hasattr(flow_result, 'to_dict') else {}, indent=2)

    def _render_evolution_html(self, evolution: Any, title: str) -> str:
        """Render evolution plot as HTML with sparkline."""
        activations = evolution.layer_activations if hasattr(evolution, 'layer_activations') else {}

        if not activations:
            return f"<div>No activation data for {title}</div>"

        # Build sparkline
        sorted_layers = sorted(activations.keys())
        values = [activations[l] for l in sorted_layers]
        max_val = max(values) if values else 1

        # SVG dimensions
        width = 400
        height = self.config.sparkline_height * 3

        # Calculate points
        points = []
        for i, val in enumerate(values):
            x = 50 + i * (width - 100) / max(1, len(values) - 1)
            y = height - 20 - (val / max_val) * (height - 40)
            points.append(f"{x},{y}")

        path_d = "M " + " L ".join(points)

        emergence = evolution.emergence_layer if hasattr(evolution, 'emergence_layer') else 0
        peak = evolution.peak_layer if hasattr(evolution, 'peak_layer') else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .stats {{ margin-top: 10px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h3>{title}</h3>
    <svg width="{width}" height="{height}">
        <path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2" />
        <!-- Dots at data points -->
        {"".join([f'<circle cx="{50 + i * (width - 100) / max(1, len(values) - 1)}" cy="{height - 20 - (v / max_val) * (height - 40)}" r="3" fill="#3b82f6" />' for i, v in enumerate(values)])}
        <!-- Peak marker -->
        <text x="{50 + sorted_layers.index(peak) * (width - 100) / max(1, len(sorted_layers) - 1)}" y="{height - 5}" text-anchor="middle" font-size="10" fill="#10b981">peak</text>
    </svg>
    <div class="stats">
        Emergence: Layer {emergence} | Peak: Layer {peak} | Layers: {', '.join(map(str, sorted_layers))}
    </div>
</body>
</html>"""

        return html

    def _render_evolution_svg(self, evolution: Any, title: str) -> str:
        """Extract SVG from evolution HTML."""
        html = self._render_evolution_html(evolution, title)
        start = html.find('<svg')
        end = html.find('</svg>') + 6
        if start >= 0 and end > start:
            return html[start:end]
        return '<svg></svg>'

    def _render_evolution_json(self, evolution: Any, title: str) -> str:
        """Render evolution as JSON."""
        return json.dumps(evolution.to_dict() if hasattr(evolution, 'to_dict') else {}, indent=2)


# Convenience functions
def create_hierarchy_visualization(
    hierarchy_analyzer: Any,
    test_inputs: Any,
    config: Optional[VisualizationConfig] = None,
    output_path: Optional[str] = None
) -> HierarchyDiagram:
    """
    Create and optionally save a hierarchy visualization.

    Args:
        hierarchy_analyzer: FeatureHierarchyAnalyzer
        test_inputs: Test inputs to analyze
        config: Optional visualization config
        output_path: Optional path to save HTML

    Returns:
        HierarchyDiagram
    """
    visualizer = HierarchyVisualizer(
        hierarchy_analyzer=hierarchy_analyzer,
        config=config
    )

    diagram = visualizer.create_hierarchy_diagram(test_inputs=test_inputs)

    if output_path:
        diagram.save(output_path)

    return diagram


def create_flow_visualization(
    flow_analyzer: Any,
    test_inputs: Any,
    config: Optional[VisualizationConfig] = None,
    output_path: Optional[str] = None
) -> FlowDiagram:
    """
    Create and optionally save a flow visualization.

    Args:
        flow_analyzer: InformationFlowAnalyzer
        test_inputs: Test inputs to analyze
        config: Optional visualization config
        output_path: Optional path to save HTML

    Returns:
        FlowDiagram
    """
    visualizer = HierarchyVisualizer(
        flow_analyzer=flow_analyzer,
        config=config
    )

    diagram = visualizer.create_flow_diagram(test_inputs=test_inputs)

    if output_path:
        diagram.save(output_path)

    return diagram
