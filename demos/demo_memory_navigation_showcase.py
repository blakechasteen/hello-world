#!/usr/bin/env python3
"""
Memory Navigation Showcase - Phase 2 Demo

Demonstrates HoloLoom's spatial memory navigation and pattern discovery.
Navigate memories in 4 directions and discover emergent patterns.

This demo works WITHOUT heavy ML dependencies - uses networkx for graphs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time

try:
    import networkx as nx
except ImportError:
    print("Warning: networkx not available, using minimal fallback")
    nx = None

# ============================================================================
# MEMORY GRAPH STRUCTURES
# ============================================================================

class NavigationDirection(Enum):
    """Spatial metaphors for memory navigation."""
    FORWARD = "forward"      # What comes next (successors)
    BACKWARD = "backward"    # What came before (predecessors)
    SIDEWAYS = "sideways"    # Related concepts (siblings)
    DEEP = "deep"            # Holistic connections (cycles)


class EdgeType(Enum):
    """Semantic edge types in knowledge graph."""
    IS_A = "IS_A"           # Taxonomy (cat IS_A animal)
    USES = "USES"           # Functional (model USES attention)
    MENTIONS = "MENTIONS"   # Reference
    LEADS_TO = "LEADS_TO"   # Causal/temporal
    PART_OF = "PART_OF"     # Composition
    SIMILAR_TO = "SIMILAR_TO"  # Semantic similarity


@dataclass
class MemoryNode:
    """A single memory in the knowledge graph."""
    id: str
    content: str
    timestamp: float = 0.0
    activation: float = 0.5
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryPattern:
    """An emergent pattern discovered in memories."""
    pattern_type: str
    description: str
    memories: List[str]
    strength: float
    metadata: Dict = field(default_factory=dict)


class MemoryGraph:
    """
    Knowledge graph for memory navigation.

    Supports 4-direction navigation and pattern discovery.
    """

    def __init__(self):
        self._nodes = {}
        self._edges = defaultdict(list)
        self.nodes: Dict[str, MemoryNode] = {}

        if nx:
            self.G = nx.MultiDiGraph()
        else:
            self.G = None

    def add_memory(self, node: MemoryNode) -> None:
        """Add a memory node."""
        self.nodes[node.id] = node
        if self.G is not None:
            self.G.add_node(node.id, **{
                'content': node.content,
                'activation': node.activation,
                'timestamp': node.timestamp
            })

    def add_edge(self, source: str, target: str, edge_type: EdgeType, weight: float = 1.0) -> None:
        """Add a directed edge between memories."""
        if self.G is not None:
            self.G.add_edge(source, target, type=edge_type.value, weight=weight)
        else:
            self._edges[source].append((target, edge_type, weight))

    def navigate(self, from_id: str, direction: NavigationDirection, steps: int = 5) -> List[str]:
        """Navigate through memory space."""
        if self.G is None:
            return []

        if from_id not in self.G.nodes():
            return []

        if direction == NavigationDirection.FORWARD:
            return self._navigate_forward(from_id, steps)
        elif direction == NavigationDirection.BACKWARD:
            return self._navigate_backward(from_id, steps)
        elif direction == NavigationDirection.SIDEWAYS:
            return self._navigate_sideways(from_id, steps)
        elif direction == NavigationDirection.DEEP:
            return self._navigate_deep(from_id, steps)
        return []

    def _navigate_forward(self, from_id: str, steps: int) -> List[str]:
        """Follow successors (what comes next)."""
        path = []
        current = from_id
        visited = {from_id}

        for _ in range(steps):
            successors = list(self.G.successors(current))
            unvisited = [s for s in successors if s not in visited]
            if not unvisited:
                break
            next_node = unvisited[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def _navigate_backward(self, from_id: str, steps: int) -> List[str]:
        """Follow predecessors (what came before)."""
        path = []
        current = from_id
        visited = {from_id}

        for _ in range(steps):
            predecessors = list(self.G.predecessors(current))
            unvisited = [p for p in predecessors if p not in visited]
            if not unvisited:
                break
            prev_node = unvisited[0]
            path.append(prev_node)
            visited.add(prev_node)
            current = prev_node

        return path

    def _navigate_sideways(self, from_id: str, steps: int) -> List[str]:
        """Find siblings (shared parents/children)."""
        siblings = set()

        # Find nodes that share parents
        for parent in self.G.predecessors(from_id):
            for sibling in self.G.successors(parent):
                if sibling != from_id:
                    siblings.add(sibling)

        # Find nodes that share children
        for child in self.G.successors(from_id):
            for sibling in self.G.predecessors(child):
                if sibling != from_id:
                    siblings.add(sibling)

        return list(siblings)[:steps]

    def _navigate_deep(self, from_id: str, steps: int) -> List[str]:
        """Find cycles and deep connections."""
        # BFS to find nodes at exactly 'steps' distance
        visited = {from_id}
        current_level = [from_id]
        deep_nodes = []

        for depth in range(steps):
            next_level = []
            for node in current_level:
                for neighbor in list(self.G.successors(node)) + list(self.G.predecessors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        if depth == steps - 1:
                            deep_nodes.append(neighbor)
            current_level = next_level
            if not current_level:
                break

        return deep_nodes

    def discover_patterns(self, min_strength: float = 0.3) -> List[MemoryPattern]:
        """Discover emergent patterns in memory graph."""
        patterns = []

        # Find loops (cycles)
        patterns.extend(self._find_loops(min_strength))

        # Find clusters
        patterns.extend(self._find_clusters(min_strength))

        # Find threads (long paths)
        patterns.extend(self._find_threads(min_strength))

        # Find resonances (highly connected nodes)
        patterns.extend(self._find_resonances(min_strength))

        return sorted(patterns, key=lambda p: p.strength, reverse=True)

    def _find_loops(self, min_strength: float) -> List[MemoryPattern]:
        """Find cycles in the graph (strange loops)."""
        patterns = []
        if self.G is None:
            return patterns

        try:
            cycles = list(nx.simple_cycles(self.G))
            for cycle in cycles[:5]:  # Limit to 5 loops
                if len(cycle) >= 2:
                    strength = 1.0 / len(cycle)  # Shorter cycles = stronger
                    if strength >= min_strength:
                        patterns.append(MemoryPattern(
                            pattern_type="loop",
                            description=f"Cycle of {len(cycle)} memories",
                            memories=cycle,
                            strength=strength
                        ))
        except:
            pass

        return patterns

    def _find_clusters(self, min_strength: float) -> List[MemoryPattern]:
        """Find densely connected clusters."""
        patterns = []
        if self.G is None:
            return patterns

        try:
            # Use connected components on undirected view
            undirected = self.G.to_undirected()
            components = list(nx.connected_components(undirected))

            for component in components:
                if len(component) >= 3:
                    # Strength based on density
                    subgraph = self.G.subgraph(component)
                    density = nx.density(subgraph) if len(component) > 1 else 0
                    if density >= min_strength:
                        patterns.append(MemoryPattern(
                            pattern_type="cluster",
                            description=f"Cluster of {len(component)} memories",
                            memories=list(component),
                            strength=density
                        ))
        except:
            pass

        return patterns

    def _find_threads(self, min_strength: float) -> List[MemoryPattern]:
        """Find long sequential paths (narrative threads)."""
        patterns = []
        if self.G is None:
            return patterns

        try:
            # Find longest paths from root nodes
            roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

            for root in roots[:3]:
                try:
                    # DFS to find longest path
                    longest = []
                    for target in self.G.nodes():
                        if target != root:
                            try:
                                path = nx.shortest_path(self.G, root, target)
                                if len(path) > len(longest):
                                    longest = path
                            except nx.NetworkXNoPath:
                                pass

                    if len(longest) >= 3:
                        strength = len(longest) / 10.0  # Longer = stronger
                        if strength >= min_strength:
                            patterns.append(MemoryPattern(
                                pattern_type="thread",
                                description=f"Thread of {len(longest)} memories",
                                memories=longest,
                                strength=min(strength, 1.0)
                            ))
                except:
                    pass
        except:
            pass

        return patterns

    def _find_resonances(self, min_strength: float) -> List[MemoryPattern]:
        """Find highly activated/connected memories."""
        patterns = []
        if self.G is None:
            return patterns

        # Find hub nodes (high degree centrality)
        try:
            centrality = nx.degree_centrality(self.G)
            hubs = [(n, c) for n, c in centrality.items() if c >= min_strength]
            hubs.sort(key=lambda x: x[1], reverse=True)

            for node, cent in hubs[:3]:
                neighbors = list(self.G.successors(node)) + list(self.G.predecessors(node))
                patterns.append(MemoryPattern(
                    pattern_type="resonance",
                    description=f"Hub '{node}' with {len(neighbors)} connections",
                    memories=[node] + neighbors[:5],
                    strength=cent
                ))
        except:
            pass

        return patterns


def render_graph_ascii(graph: MemoryGraph, width: int = 60, height: int = 15) -> str:
    """Render graph as ASCII art."""
    if not graph.nodes:
        return "  (empty graph)"

    lines = []
    node_list = list(graph.nodes.keys())

    # Simple layout: arrange in levels
    if graph.G is not None:
        try:
            # Topological order if DAG
            order = list(nx.topological_sort(graph.G))
        except:
            order = node_list
    else:
        order = node_list

    # Create level assignment
    levels = defaultdict(list)
    node_level = {}

    for i, node in enumerate(order):
        level = 0
        if graph.G is not None:
            for pred in graph.G.predecessors(node):
                if pred in node_level:
                    level = max(level, node_level[pred] + 1)
        node_level[node] = level
        levels[level].append(node)

    # Render
    max_level = max(levels.keys()) if levels else 0

    for level in range(max_level + 1):
        nodes_at_level = levels[level]
        spacing = width // (len(nodes_at_level) + 1)

        # Node line
        line = ""
        positions = {}
        for i, node in enumerate(nodes_at_level):
            pos = spacing * (i + 1)
            positions[node] = pos
            label = node[:8]
            line += " " * (pos - len(line)) + f"[{label}]"

        lines.append(line)

        # Edge lines to next level
        if level < max_level:
            edge_line = " " * width
            for node in nodes_at_level:
                if graph.G is not None:
                    for succ in graph.G.successors(node):
                        if succ in levels[level + 1]:
                            start = positions.get(node, 0) + 4
                            edge_line = edge_line[:start] + "│" + edge_line[start+1:]
            lines.append(edge_line)

            arrow_line = " " * width
            for node in nodes_at_level:
                if graph.G is not None:
                    for succ in graph.G.successors(node):
                        if succ in levels[level + 1]:
                            start = positions.get(node, 0) + 4
                            arrow_line = arrow_line[:start] + "↓" + arrow_line[start+1:]
            lines.append(arrow_line)

    return "\n".join(["  " + line for line in lines])


def build_sample_graph() -> MemoryGraph:
    """Build a sample knowledge graph for demos."""
    graph = MemoryGraph()

    # Add memories about machine learning
    memories = [
        ("ml", "Machine Learning"),
        ("supervised", "Supervised Learning"),
        ("unsupervised", "Unsupervised Learning"),
        ("neural_net", "Neural Networks"),
        ("decision_tree", "Decision Trees"),
        ("clustering", "Clustering"),
        ("deep_learning", "Deep Learning"),
        ("transformer", "Transformer Architecture"),
        ("attention", "Attention Mechanism"),
        ("bert", "BERT Model"),
        ("gpt", "GPT Model"),
        ("thompson", "Thompson Sampling"),
        ("bandit", "Multi-Armed Bandit"),
        ("exploration", "Exploration Strategy"),
    ]

    for id, content in memories:
        graph.add_memory(MemoryNode(id=id, content=content, activation=0.5 + np.random.random() * 0.5))

    # Add edges (knowledge relationships)
    edges = [
        ("ml", "supervised", EdgeType.IS_A),
        ("ml", "unsupervised", EdgeType.IS_A),
        ("supervised", "neural_net", EdgeType.USES),
        ("supervised", "decision_tree", EdgeType.USES),
        ("unsupervised", "clustering", EdgeType.USES),
        ("neural_net", "deep_learning", EdgeType.LEADS_TO),
        ("deep_learning", "transformer", EdgeType.USES),
        ("transformer", "attention", EdgeType.USES),
        ("transformer", "bert", EdgeType.LEADS_TO),
        ("transformer", "gpt", EdgeType.LEADS_TO),
        ("thompson", "bandit", EdgeType.USES),
        ("bandit", "exploration", EdgeType.USES),
        ("exploration", "thompson", EdgeType.LEADS_TO),  # Creates a loop!
        ("ml", "thompson", EdgeType.USES),
    ]

    for source, target, edge_type in edges:
        graph.add_edge(source, target, edge_type)

    return graph


def demo_graph_construction():
    """Demo 1: Building a knowledge graph."""
    print("\n" + "=" * 60)
    print("Demo 1: Knowledge Graph Construction")
    print("=" * 60)

    graph = build_sample_graph()

    print(f"\n  Built graph with {len(graph.nodes)} memories")
    print(f"  Edge types: IS_A, USES, LEADS_TO")

    print("\n  Sample Memories:")
    for i, (id, node) in enumerate(list(graph.nodes.items())[:6]):
        print(f"    • {id}: \"{node.content}\" (activation={node.activation:.2f})")

    print("\n  Graph Structure (ASCII):")
    print(render_graph_ascii(graph))

    if graph.G is not None:
        print(f"\n  Graph Statistics:")
        print(f"    Nodes: {graph.G.number_of_nodes()}")
        print(f"    Edges: {graph.G.number_of_edges()}")
        try:
            print(f"    Density: {nx.density(graph.G):.3f}")
        except:
            pass


def demo_navigation_directions():
    """Demo 2: Navigate in 4 directions."""
    print("\n" + "=" * 60)
    print("Demo 2: Spatial Navigation - 4 Directions")
    print("=" * 60)

    graph = build_sample_graph()

    print("\n  Starting from: 'transformer' (Transformer Architecture)")
    print("  " + "─" * 50)

    directions = [
        (NavigationDirection.FORWARD, "What comes next?"),
        (NavigationDirection.BACKWARD, "What came before?"),
        (NavigationDirection.SIDEWAYS, "What's related?"),
        (NavigationDirection.DEEP, "What's deeply connected?"),
    ]

    for direction, question in directions:
        path = graph.navigate("transformer", direction, steps=3)

        print(f"\n  {direction.value.upper():10s} - \"{question}\"")
        if path:
            arrow = "→" if direction == NavigationDirection.FORWARD else "←" if direction == NavigationDirection.BACKWARD else "↔" if direction == NavigationDirection.SIDEWAYS else "⟳"
            print(f"    transformer {arrow} " + f" {arrow} ".join(path))
        else:
            print(f"    (no path found)")


def demo_pattern_discovery():
    """Demo 3: Discover emergent patterns."""
    print("\n" + "=" * 60)
    print("Demo 3: Pattern Discovery - Emergent Structures")
    print("=" * 60)

    graph = build_sample_graph()

    print("\n  Searching for patterns in knowledge graph...")

    patterns = graph.discover_patterns(min_strength=0.1)

    pattern_types = defaultdict(list)
    for p in patterns:
        pattern_types[p.pattern_type].append(p)

    for ptype, plist in pattern_types.items():
        print(f"\n  {ptype.upper()} patterns found: {len(plist)}")

        emoji = {"loop": "🔄", "cluster": "🎯", "thread": "📜", "resonance": "✨"}.get(ptype, "•")

        for p in plist[:2]:
            print(f"    {emoji} {p.description} (strength={p.strength:.2f})")
            print(f"       Memories: {' → '.join(p.memories[:5])}")


def demo_exploration_scenario():
    """Demo 4: Real exploration scenario."""
    print("\n" + "=" * 60)
    print("Demo 4: Exploration Scenario - Learning Journey")
    print("=" * 60)

    graph = build_sample_graph()

    print("\n  Scenario: User asks 'Tell me about Thompson Sampling'")
    print("  " + "─" * 50)

    start = "thompson"
    print(f"\n  1. Start at: '{start}' ({graph.nodes[start].content})")

    # Navigate backward to find foundations
    backward = graph.navigate(start, NavigationDirection.BACKWARD, steps=3)
    print(f"\n  2. BACKWARD (foundations):")
    if backward:
        for node in backward:
            content = graph.nodes.get(node, MemoryNode(node, node)).content
            print(f"       ← {node}: {content}")
    else:
        print("       (reached root)")

    # Navigate forward to find applications
    forward = graph.navigate(start, NavigationDirection.FORWARD, steps=3)
    print(f"\n  3. FORWARD (applications):")
    if forward:
        for node in forward:
            content = graph.nodes.get(node, MemoryNode(node, node)).content
            print(f"       → {node}: {content}")
    else:
        print("       (reached leaf)")

    # Navigate sideways to find alternatives
    sideways = graph.navigate(start, NavigationDirection.SIDEWAYS, steps=3)
    print(f"\n  4. SIDEWAYS (alternatives):")
    if sideways:
        for node in sideways:
            content = graph.nodes.get(node, MemoryNode(node, node)).content
            print(f"       ↔ {node}: {content}")
    else:
        print("       (no siblings)")

    # Check for loops
    deep = graph.navigate(start, NavigationDirection.DEEP, steps=3)
    print(f"\n  5. DEEP (distant connections):")
    if deep:
        for node in deep:
            content = graph.nodes.get(node, MemoryNode(node, node)).content
            print(f"       ⟳ {node}: {content}")
    else:
        print("       (no deep connections)")


def demo_activation_spreading():
    """Demo 5: Simulate activation spreading."""
    print("\n" + "=" * 60)
    print("Demo 5: Activation Spreading - Memory Retrieval")
    print("=" * 60)

    graph = build_sample_graph()

    print("\n  Query activates 'transformer' node")
    print("  Activation spreads through connections...")

    # Initialize activations
    activations = {node: 0.0 for node in graph.nodes}
    activations["transformer"] = 1.0

    print("\n  Round 0 (initial):")
    print(f"    transformer: ████████████████████ 1.00")

    # Spread for 3 rounds
    decay = 0.5

    for round_num in range(1, 4):
        new_activations = activations.copy()

        if graph.G is not None:
            for node, act in activations.items():
                if act > 0.01:
                    # Spread to neighbors
                    for neighbor in list(graph.G.successors(node)) + list(graph.G.predecessors(node)):
                        spread = act * decay
                        new_activations[neighbor] = max(new_activations[neighbor], spread)

        activations = new_activations

        # Show top activated
        sorted_act = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n  Round {round_num} (decay={decay}):")
        for node, act in sorted_act:
            if act > 0.01:
                bar_len = int(act * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {node:15s}: {bar} {act:.2f}")


def demo_path_finding():
    """Demo 6: Find paths between concepts."""
    print("\n" + "=" * 60)
    print("Demo 6: Path Finding - Connecting Concepts")
    print("=" * 60)

    graph = build_sample_graph()

    pairs = [
        ("ml", "gpt"),
        ("thompson", "attention"),
        ("supervised", "clustering"),
    ]

    print("\n  Finding shortest paths between concepts:")

    for source, target in pairs:
        print(f"\n  {source} → {target}:")

        if graph.G is not None and nx is not None:
            try:
                path = nx.shortest_path(graph.G, source, target)
                path_str = " → ".join(path)
                print(f"    Path: {path_str}")
                print(f"    Length: {len(path) - 1} hops")
            except nx.NetworkXNoPath:
                print(f"    No direct path found")
            except Exception as e:
                print(f"    Error: {type(e).__name__}")
        else:
            print(f"    (networkx not available)")


def main():
    """Run all memory navigation demonstrations."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " MEMORY NAVIGATION SHOWCASE ".center(58) + "║")
    print("║" + " Spatial Navigation & Pattern Discovery ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if not nx:
        print("\n  Warning: networkx not installed. Some features limited.")

    start = time.time()

    demo_graph_construction()
    demo_navigation_directions()
    demo_pattern_discovery()
    demo_exploration_scenario()
    demo_activation_spreading()
    demo_path_finding()

    elapsed = time.time() - start

    print("\n" + "═" * 60)
    print(f"  Completed 6 demonstrations in {elapsed:.2f}s")
    print("  Directions: FORWARD, BACKWARD, SIDEWAYS, DEEP")
    print("  Patterns: loops, clusters, threads, resonances")
    print("  Dependencies: networkx (graph operations)")
    print("═" * 60)


if __name__ == "__main__":
    main()
