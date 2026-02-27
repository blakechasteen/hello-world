"""
HoloLoom TUI Knowledge Graph Display
November 2025

ASCII visualization of knowledge graph in terminal.
Renders nodes, edges, and relationships as text art.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class GraphNode:
    """Node in the graph."""
    id: str
    label: str
    node_type: str = "entity"
    activation: float = 0.0
    degree: int = 0


@dataclass
class GraphEdge:
    """Edge in the graph."""
    source: str
    target: str
    edge_type: str = "RELATES_TO"
    weight: float = 1.0


class GraphDisplay:
    """
    ASCII knowledge graph visualization.

    Supports multiple display modes:
    - Tree view (hierarchical)
    - Table view (list format)
    - ASCII art (box drawing)
    - Network view (connections)
    """

    # Edge type symbols and colors
    EDGE_SYMBOLS = {
        "IS_A": "─▷",
        "USES": "──•",
        "MENTIONS": "···",
        "LEADS_TO": "──→",
        "PART_OF": "─⊂",
        "IN_TIME": "─◷",
        "OCCURRED_AT": "─@",
        "RELATES_TO": "───"
    }

    EDGE_COLORS = {
        "IS_A": "blue",
        "USES": "green",
        "MENTIONS": "dim",
        "LEADS_TO": "yellow",
        "PART_OF": "magenta",
        "IN_TIME": "cyan",
        "OCCURRED_AT": "cyan",
        "RELATES_TO": "white"
    }

    def __init__(self, console: Optional['Console'] = None):
        """Initialize graph display."""
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required")

        self.console = console or Console()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node_id: str, label: str = None,
                 node_type: str = "entity", activation: float = 0.0):
        """Add a node to the graph."""
        self.nodes[node_id] = GraphNode(
            id=node_id,
            label=label or node_id,
            node_type=node_type,
            activation=activation
        )

    def add_edge(self, source: str, target: str,
                 edge_type: str = "RELATES_TO", weight: float = 1.0):
        """Add an edge to the graph."""
        self.edges.append(GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight
        ))
        # Update degrees
        if source in self.nodes:
            self.nodes[source].degree += 1
        if target in self.nodes:
            self.nodes[target].degree += 1

    def from_kg(self, kg):
        """Import from hololoom KG object."""
        # Get all nodes
        if hasattr(kg, 'get_all_nodes'):
            for node_id in kg.get_all_nodes():
                self.add_node(node_id)

        # Get all edges
        if hasattr(kg, 'get_all_edges'):
            for edge in kg.get_all_edges():
                self.add_edge(
                    edge.source,
                    edge.target,
                    edge.edge_type,
                    getattr(edge, 'weight', 1.0)
                )

    def clear(self):
        """Clear the graph."""
        self.nodes.clear()
        self.edges.clear()

    def _get_adjacency(self) -> Dict[str, List[Tuple[str, str]]]:
        """Build adjacency list."""
        adj: Dict[str, List[Tuple[str, str]]] = {n: [] for n in self.nodes}
        for edge in self.edges:
            if edge.source in adj:
                adj[edge.source].append((edge.target, edge.edge_type))
        return adj

    def render_tree(self, root: str = None, max_depth: int = 3) -> Tree:
        """Render graph as Rich Tree (hierarchical view)."""
        adj = self._get_adjacency()

        # Find root if not specified (node with most outgoing edges)
        if root is None:
            if not self.nodes:
                return Tree("[dim]Empty graph[/dim]")
            root = max(self.nodes.keys(), key=lambda n: len(adj.get(n, [])))

        # Build tree recursively
        visited: Set[str] = set()

        def build_tree(node_id: str, depth: int = 0) -> Tree:
            if depth > max_depth or node_id in visited:
                if node_id in visited:
                    return Tree(f"[dim]↩ {node_id}[/dim]")
                return Tree("[dim]...[/dim]")

            visited.add(node_id)
            node = self.nodes.get(node_id)

            # Format node
            if node:
                activation_bar = self._activation_bar(node.activation)
                label = f"[bold]{node.label}[/bold] {activation_bar}"
            else:
                label = f"[dim]{node_id}[/dim]"

            tree = Tree(label)

            # Add children
            for target, edge_type in adj.get(node_id, []):
                symbol = self.EDGE_SYMBOLS.get(edge_type, "───")
                color = self.EDGE_COLORS.get(edge_type, "white")
                child_tree = build_tree(target, depth + 1)
                tree.add(Text.assemble(
                    (f"{symbol} ", color),
                    child_tree.label
                ))
                for sub in child_tree.children:
                    tree.children[-1].add(sub.label)

            return tree

        return build_tree(root)

    def _activation_bar(self, activation: float, width: int = 5) -> str:
        """Create activation level indicator."""
        if activation <= 0:
            return "[dim]○○○○○[/dim]"

        filled = int(activation * width)
        if activation >= 0.8:
            color = "green"
        elif activation >= 0.5:
            color = "yellow"
        else:
            color = "red"

        bar = "●" * filled + "○" * (width - filled)
        return f"[{color}]{bar}[/{color}]"

    def render_table(self, sort_by: str = "degree") -> Table:
        """Render graph as Rich Table."""
        table = Table(title="Knowledge Graph Nodes")

        table.add_column("Node", style="bold")
        table.add_column("Type", style="cyan")
        table.add_column("Degree", justify="right")
        table.add_column("Activation", justify="center")

        # Sort nodes
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: getattr(n, sort_by, 0),
            reverse=True
        )

        for node in sorted_nodes[:20]:  # Limit to 20
            table.add_row(
                node.label,
                node.node_type,
                str(node.degree),
                self._activation_bar(node.activation)
            )

        return table

    def render_edges_table(self) -> Table:
        """Render edges as Rich Table."""
        table = Table(title="Graph Edges")

        table.add_column("Source", style="bold")
        table.add_column("Relation", style="cyan")
        table.add_column("Target", style="bold")
        table.add_column("Weight", justify="right")

        for edge in self.edges[:30]:  # Limit to 30
            color = self.EDGE_COLORS.get(edge.edge_type, "white")
            table.add_row(
                edge.source,
                f"[{color}]{edge.edge_type}[/{color}]",
                edge.target,
                f"{edge.weight:.2f}"
            )

        return table

    def render_ascii_network(self, width: int = 60, height: int = 20) -> str:
        """Render graph as ASCII art network."""
        if not self.nodes:
            return "Empty graph"

        # Simple force-directed layout
        positions = self._simple_layout(width, height)

        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw edges first
        for edge in self.edges:
            if edge.source in positions and edge.target in positions:
                x1, y1 = positions[edge.source]
                x2, y2 = positions[edge.target]
                self._draw_line(canvas, x1, y1, x2, y2)

        # Draw nodes
        for node_id, (x, y) in positions.items():
            node = self.nodes[node_id]
            # Use first 3 chars of label
            label = node.label[:3].upper()
            for i, char in enumerate(label):
                if 0 <= x + i < width and 0 <= y < height:
                    canvas[y][x + i] = char

        # Convert to string
        return '\n'.join(''.join(row) for row in canvas)

    def _simple_layout(self, width: int, height: int) -> Dict[str, Tuple[int, int]]:
        """Simple circular layout for nodes."""
        positions = {}
        n = len(self.nodes)
        if n == 0:
            return positions

        cx, cy = width // 2, height // 2
        radius = min(width, height) // 2 - 3

        for i, node_id in enumerate(self.nodes):
            angle = 2 * math.pi * i / n
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle) * 0.5)  # Flatten for terminal aspect ratio
            positions[node_id] = (max(0, min(width - 4, x)), max(0, min(height - 1, y)))

        return positions

    def _draw_line(self, canvas: List[List[str]], x1: int, y1: int, x2: int, y2: int):
        """Draw a line on ASCII canvas using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        if dx > dy:
            err = dx / 2
            while x != x2:
                if 0 <= x < len(canvas[0]) and 0 <= y < len(canvas):
                    if canvas[y][x] == ' ':
                        canvas[y][x] = '·'
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y2:
                if 0 <= x < len(canvas[0]) and 0 <= y < len(canvas):
                    if canvas[y][x] == ' ':
                        canvas[y][x] = '·'
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def render_panel(self, mode: str = "tree", **kwargs) -> Panel:
        """Render graph as Rich Panel."""
        if mode == "tree":
            content = self.render_tree(**kwargs)
        elif mode == "table":
            content = self.render_table(**kwargs)
        elif mode == "edges":
            content = self.render_edges_table()
        elif mode == "ascii":
            content = Text(self.render_ascii_network(**kwargs))
        else:
            content = self.render_table()

        return Panel(
            content,
            title="[bold blue]Knowledge Graph[/bold blue]",
            border_style="blue"
        )

    def render_compact(self, max_nodes: int = 5) -> Text:
        """Render compact summary."""
        text = Text()

        # Node count
        text.append(f"Nodes: {len(self.nodes)} ", style="bold")
        text.append(f"Edges: {len(self.edges)}\n", style="bold")

        # Top nodes by degree
        top_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.degree,
            reverse=True
        )[:max_nodes]

        for node in top_nodes:
            text.append(f"  • {node.label}", style="cyan")
            text.append(f" ({node.degree} connections)\n", style="dim")

        return text

    def print(self, mode: str = "tree", **kwargs):
        """Print graph to console."""
        self.console.print(self.render_panel(mode=mode, **kwargs))


class SourceAttributionDisplay:
    """
    Display source attribution from memory retrieval.

    Shows which memory nodes contributed to the response.
    """

    def __init__(self, console: Optional['Console'] = None):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required")
        self.console = console or Console()

    def render_sources(self, sources: List[Dict]) -> Panel:
        """
        Render source attribution.

        Args:
            sources: List of dicts with 'id', 'content', 'relevance', 'edge_type'
        """
        table = Table(show_header=True, header_style="bold")
        table.add_column("Rel%", justify="right", width=5)
        table.add_column("Source", style="cyan")
        table.add_column("Content", max_width=40)

        for src in sorted(sources, key=lambda s: s.get('relevance', 0), reverse=True):
            rel = src.get('relevance', 0) * 100
            rel_style = "green" if rel >= 70 else "yellow" if rel >= 40 else "dim"

            content = src.get('content', '')[:40]
            if len(src.get('content', '')) > 40:
                content += "..."

            table.add_row(
                f"[{rel_style}]{rel:.0f}%[/{rel_style}]",
                src.get('id', 'unknown'),
                content
            )

        return Panel(
            table,
            title=f"[bold]Sources ({len(sources)} memories)[/bold]",
            border_style="green"
        )

    def print_sources(self, sources: List[Dict]):
        """Print sources to console."""
        self.console.print(self.render_sources(sources))


# Quick test
if __name__ == "__main__":
    print("Testing Graph Display...")

    display = GraphDisplay()

    # Add sample nodes and edges
    display.add_node("thompson", "Thompson Sampling", activation=0.9)
    display.add_node("bayesian", "Bayesian Methods", activation=0.7)
    display.add_node("exploration", "Exploration", activation=0.6)
    display.add_node("exploitation", "Exploitation", activation=0.5)
    display.add_node("bandit", "Multi-Armed Bandit", activation=0.8)

    display.add_edge("thompson", "bayesian", "IS_A")
    display.add_edge("thompson", "exploration", "USES")
    display.add_edge("thompson", "exploitation", "USES")
    display.add_edge("bandit", "thompson", "USES")
    display.add_edge("exploration", "exploitation", "LEADS_TO")

    print("\n=== Tree View ===")
    display.print(mode="tree")

    print("\n=== Table View ===")
    display.print(mode="table")

    print("\n=== ASCII Network ===")
    display.print(mode="ascii")

    print("\n=== Source Attribution ===")
    sources = [
        {"id": "mem_001", "content": "Thompson Sampling balances exploration and exploitation", "relevance": 0.92},
        {"id": "mem_002", "content": "Bayesian approach to decision making", "relevance": 0.78},
        {"id": "mem_003", "content": "Multi-armed bandit algorithms", "relevance": 0.65}
    ]
    source_display = SourceAttributionDisplay()
    source_display.print_sources(sources)
