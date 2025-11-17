#!/usr/bin/env python3
"""
Chain Visualization Tools

Provides tools for visualizing chain execution graphs, dependencies, and traces.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in execution graph"""
    INPUT = "input"
    PROCESSOR = "processor"
    OUTPUT = "output"
    DECISION = "decision"
    PARALLEL = "parallel"
    LOOP = "loop"


@dataclass
class GraphNode:
    """Node in execution graph"""
    id: str
    name: str
    node_type: NodeType
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    """Edge between nodes"""
    source: str
    target: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = None


class ChainVisualizer:
    """Visualize chain execution graphs"""

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

    def build_graph_from_chain(self, chain_def: Dict[str, Any]) -> None:
        """Build execution graph from chain definition"""
        # Clear existing graph
        self.nodes = []
        self.edges = []

        # Add input node
        self.add_node("input", "Input", NodeType.INPUT, {})

        # Add step nodes
        steps = chain_def.get("steps", [])
        for step in steps:
            node_type = self._get_node_type(step.processor_type)

            self.add_node(
                step.name,
                step.name,
                node_type,
                {
                    "processor_type": step.processor_type,
                    "config": step.config
                }
            )

            # Add edges for dependencies
            if step.dependencies:
                for dep in step.dependencies:
                    self.add_edge(dep, step.name)
            else:
                # Connect to input if no dependencies
                self.add_edge("input", step.name)

        # Add output node
        last_steps = [s.name for s in steps if not any(s.name in step.dependencies for step in steps)]
        self.add_node("output", "Output", NodeType.OUTPUT, {})

        for step_name in last_steps:
            self.add_edge(step_name, "output")

    def add_node(self, node_id: str, name: str, node_type: NodeType, metadata: Dict[str, Any]) -> None:
        """Add a node to the graph"""
        node = GraphNode(id=node_id, name=name, node_type=node_type, metadata=metadata)
        self.nodes.append(node)

    def add_edge(self, source: str, target: str, label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an edge to the graph"""
        edge = GraphEdge(source=source, target=target, label=label, metadata=metadata or {})
        self.edges.append(edge)

    def to_mermaid(self) -> str:
        """Export graph to Mermaid diagram format"""
        lines = ["flowchart TD"]

        # Define nodes
        for node in self.nodes:
            shape = self._get_mermaid_shape(node.node_type)
            lines.append(f"    {node.id}{shape[0]}{node.name}{shape[1]}")

        # Define edges
        for edge in self.edges:
            if edge.label:
                lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} --> {edge.target}")

        # Add styling
        lines.extend([
            "",
            "    classDef inputNode fill:#e1f5e1,stroke:#4caf50",
            "    classDef processorNode fill:#e3f2fd,stroke:#2196f3",
            "    classDef outputNode fill:#fff3e0,stroke:#ff9800",
            "    classDef decisionNode fill:#f3e5f5,stroke:#9c27b0",
            "    classDef parallelNode fill:#fce4ec,stroke:#e91e63",
            "    classDef loopNode fill:#e0f2f1,stroke:#009688",
        ])

        # Apply styles to nodes
        for node in self.nodes:
            class_name = self._get_mermaid_class(node.node_type)
            lines.append(f"    class {node.id} {class_name}")

        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Export graph to Graphviz DOT format"""
        lines = ["digraph Chain {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=rounded];")
        lines.append("")

        # Define nodes
        for node in self.nodes:
            style = self._get_graphviz_style(node.node_type)
            lines.append(f'    {node.id} [label="{node.name}", {style}];')

        lines.append("")

        # Define edges
        for edge in self.edges:
            if edge.label:
                lines.append(f'    {edge.source} -> {edge.target} [label="{edge.label}"];')
            else:
                lines.append(f'    {edge.source} -> {edge.target};')

        lines.append("}")

        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Export graph to ASCII art representation"""
        # Simple tree-based ASCII representation
        lines = []
        visited: Set[str] = set()

        def render_node(node_id: str, indent: int = 0) -> None:
            if node_id in visited:
                return

            visited.add(node_id)
            node = next((n for n in self.nodes if n.id == node_id), None)
            if not node:
                return

            # Render this node
            prefix = "  " * indent
            symbol = self._get_ascii_symbol(node.node_type)
            lines.append(f"{prefix}{symbol} {node.name}")

            # Render children
            children = [e.target for e in self.edges if e.source == node_id]
            for child in children:
                render_node(child, indent + 1)

        # Start from input node
        render_node("input")

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Export graph to JSON format"""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "metadata": node.metadata
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ]
        }

    def _get_node_type(self, processor_type: str) -> NodeType:
        """Map processor type to node type"""
        mapping = {
            "conditional": NodeType.DECISION,
            "parallel": NodeType.PARALLEL,
            "loop": NodeType.LOOP,
        }
        return mapping.get(processor_type, NodeType.PROCESSOR)

    def _get_mermaid_shape(self, node_type: NodeType) -> tuple:
        """Get Mermaid shape for node type"""
        shapes = {
            NodeType.INPUT: ("[", "]"),
            NodeType.OUTPUT: ("[", "]"),
            NodeType.PROCESSOR: ("[", "]"),
            NodeType.DECISION: ("{", "}"),
            NodeType.PARALLEL: ("[[", "]]"),
            NodeType.LOOP: ("((", "))"),
        }
        return shapes.get(node_type, ("[", "]"))

    def _get_mermaid_class(self, node_type: NodeType) -> str:
        """Get Mermaid CSS class for node type"""
        classes = {
            NodeType.INPUT: "inputNode",
            NodeType.OUTPUT: "outputNode",
            NodeType.PROCESSOR: "processorNode",
            NodeType.DECISION: "decisionNode",
            NodeType.PARALLEL: "parallelNode",
            NodeType.LOOP: "loopNode",
        }
        return classes.get(node_type, "processorNode")

    def _get_graphviz_style(self, node_type: NodeType) -> str:
        """Get Graphviz style for node type"""
        styles = {
            NodeType.INPUT: 'shape=ellipse, style="filled", fillcolor="#e1f5e1"',
            NodeType.OUTPUT: 'shape=ellipse, style="filled", fillcolor="#fff3e0"',
            NodeType.PROCESSOR: 'shape=box, style="rounded,filled", fillcolor="#e3f2fd"',
            NodeType.DECISION: 'shape=diamond, style="filled", fillcolor="#f3e5f5"',
            NodeType.PARALLEL: 'shape=box, style="rounded,filled", fillcolor="#fce4ec"',
            NodeType.LOOP: 'shape=box, style="rounded,filled", fillcolor="#e0f2f1"',
        }
        return styles.get(node_type, 'shape=box')

    def _get_ascii_symbol(self, node_type: NodeType) -> str:
        """Get ASCII symbol for node type"""
        symbols = {
            NodeType.INPUT: "▶",
            NodeType.OUTPUT: "■",
            NodeType.PROCESSOR: "□",
            NodeType.DECISION: "◇",
            NodeType.PARALLEL: "⊞",
            NodeType.LOOP: "↻",
        }
        return symbols.get(node_type, "□")


class ExecutionTraceVisualizer:
    """Visualize execution traces"""

    def __init__(self):
        pass

    def trace_to_timeline(self, trace: List[Dict[str, Any]]) -> str:
        """Convert execution trace to ASCII timeline"""
        lines = []
        lines.append("Execution Timeline")
        lines.append("=" * 60)
        lines.append("")

        for i, entry in enumerate(trace, 1):
            step_name = entry.get("step", "unknown")
            status = entry.get("status", "unknown")
            step_type = entry.get("type", "")

            # Format status symbol
            status_symbol = {
                "completed": "✓",
                "failed": "✗",
                "skipped": "○",
                "pending": "◯"
            }.get(status, "?")

            # Format line
            line = f"{i:3d}. {status_symbol} {step_name}"
            if step_type:
                line += f" ({step_type})"

            lines.append(line)

            # Add error details if failed
            if status == "failed" and "error" in entry:
                lines.append(f"     Error: {entry['error']}")

            # Add result summary if completed
            if status == "completed" and "result" in entry:
                result = entry["result"]
                if isinstance(result, dict) and "success" in result:
                    lines.append(f"     Success: {result['success']}")

        return "\n".join(lines)

    def trace_to_html(self, trace: List[Dict[str, Any]]) -> str:
        """Convert execution trace to HTML"""
        html_parts = []
        html_parts.append("<html><head><style>")
        html_parts.append("""
            body { font-family: Arial, sans-serif; padding: 20px; }
            .trace-container { max-width: 800px; margin: 0 auto; }
            .trace-entry { border-left: 3px solid #ddd; padding: 10px; margin: 10px 0; }
            .trace-entry.completed { border-color: #4caf50; background: #f1f8f4; }
            .trace-entry.failed { border-color: #f44336; background: #fef1f0; }
            .trace-entry.skipped { border-color: #ff9800; background: #fff8f0; }
            .trace-header { font-weight: bold; margin-bottom: 5px; }
            .trace-details { font-size: 0.9em; color: #666; }
            .trace-error { color: #d32f2f; margin-top: 5px; }
        """)
        html_parts.append("</style></head><body>")
        html_parts.append('<div class="trace-container">')
        html_parts.append('<h1>Execution Trace</h1>')

        for i, entry in enumerate(trace, 1):
            step_name = entry.get("step", "unknown")
            status = entry.get("status", "unknown")
            step_type = entry.get("type", "")

            html_parts.append(f'<div class="trace-entry {status}">')
            html_parts.append(f'<div class="trace-header">{i}. {step_name}')
            if step_type:
                html_parts.append(f' <span style="color: #666;">({step_type})</span>')
            html_parts.append('</div>')

            html_parts.append(f'<div class="trace-details">Status: {status}</div>')

            if status == "failed" and "error" in entry:
                html_parts.append(f'<div class="trace-error">Error: {entry["error"]}</div>')

            html_parts.append('</div>')

        html_parts.append('</div>')
        html_parts.append('</body></html>')

        return "\n".join(html_parts)

    def trace_to_json(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert execution trace to structured JSON"""
        return {
            "total_steps": len(trace),
            "completed": sum(1 for e in trace if e.get("status") == "completed"),
            "failed": sum(1 for e in trace if e.get("status") == "failed"),
            "skipped": sum(1 for e in trace if e.get("status") == "skipped"),
            "trace": trace
        }


# Convenience functions
def visualize_chain(chain_def: Dict[str, Any], format: str = "mermaid") -> str:
    """Visualize a chain definition"""
    visualizer = ChainVisualizer()
    visualizer.build_graph_from_chain(chain_def)

    if format == "mermaid":
        return visualizer.to_mermaid()
    elif format == "graphviz":
        return visualizer.to_graphviz()
    elif format == "ascii":
        return visualizer.to_ascii()
    elif format == "json":
        import json
        return json.dumps(visualizer.to_json(), indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def visualize_trace(trace: List[Dict[str, Any]], format: str = "timeline") -> str:
    """Visualize an execution trace"""
    visualizer = ExecutionTraceVisualizer()

    if format == "timeline":
        return visualizer.trace_to_timeline(trace)
    elif format == "html":
        return visualizer.trace_to_html(trace)
    elif format == "json":
        import json
        return json.dumps(visualizer.trace_to_json(trace), indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


__all__ = [
    "ChainVisualizer",
    "ExecutionTraceVisualizer",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "visualize_chain",
    "visualize_trace"
]
