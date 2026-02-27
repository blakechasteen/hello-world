"""
Circuit Graph: Graph-Based Circuit Representation

Provides graph-based representation of computational circuits discovered
through tracing, enabling analysis, manipulation, and visualization.

Research Basis:
- ACDC (Conmy et al., 2023): Graph-based circuit representation
- IOI Circuit (Wang et al., 2022): Circuit structure analysis
- Causal Scrubbing: Graph-based causal validation

Key Features:
- Node types for different circuit components
- Edge types with causal/correlational distinction
- Path finding and critical path analysis
- Subgraph extraction and manipulation
- Integration with CircuitTrace from tracer module
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
import heapq


class NodeType(Enum):
    """Types of nodes in a circuit graph."""

    # Input/Output
    INPUT = "input"
    OUTPUT = "output"

    # Model components
    EMBEDDING = "embedding"
    ATTENTION_HEAD = "attention_head"
    ATTENTION_PATTERN = "attention_pattern"
    MLP_NEURON = "mlp_neuron"
    MLP_LAYER = "mlp_layer"
    RESIDUAL = "residual"
    LAYER_NORM = "layer_norm"

    # SAE components
    SAE_FEATURE = "sae_feature"
    SAE_ENCODER = "sae_encoder"
    SAE_DECODER = "sae_decoder"

    # Abstract
    CONCEPT = "concept"
    PATTERN = "pattern"
    AGGREGATION = "aggregation"


class EdgeType(Enum):
    """Types of edges in a circuit graph."""

    # Causal edges (verified through patching)
    CAUSAL_STRONG = "causal_strong"      # High confidence causal
    CAUSAL_WEAK = "causal_weak"          # Low confidence causal

    # Correlational edges (statistical)
    CORRELATIONAL = "correlational"       # Correlated but not causal

    # Structural edges (architectural)
    FEEDFORWARD = "feedforward"           # Standard forward connection
    ATTENTION = "attention"               # Attention-based connection
    RESIDUAL = "residual"                 # Residual stream connection
    SKIP = "skip"                         # Skip connection

    # Information flow
    INFORMATION_FLOW = "information_flow" # Information transfer
    INHIBITORY = "inhibitory"             # Inhibitory connection

    # Composition
    COMPOSITION = "composition"           # Feature composition
    SUPERPOSITION = "superposition"       # Superposition relationship


@dataclass
class CircuitNode:
    """A node in the circuit graph."""

    node_id: str
    node_type: NodeType
    layer_idx: int
    position: Optional[int] = None  # Token position if applicable

    # Activation information
    activation: float = 0.0
    importance: float = 0.0

    # SAE-specific
    feature_idx: Optional[int] = None
    feature_label: Optional[str] = None

    # Attention-specific
    head_idx: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitNode):
            return False
        return self.node_id == other.node_id

    @property
    def is_sae_feature(self) -> bool:
        """Check if this node represents an SAE feature."""
        return self.node_type == NodeType.SAE_FEATURE

    @property
    def is_attention(self) -> bool:
        """Check if this node is attention-related."""
        return self.node_type in {NodeType.ATTENTION_HEAD, NodeType.ATTENTION_PATTERN}

    @property
    def is_mlp(self) -> bool:
        """Check if this node is MLP-related."""
        return self.node_type in {NodeType.MLP_NEURON, NodeType.MLP_LAYER}


@dataclass
class CircuitEdge:
    """An edge in the circuit graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0

    # Causal verification
    is_causal: bool = False
    causal_effect: float = 0.0  # Effect size from patching

    # Information flow
    mutual_information: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitEdge):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )

    @property
    def is_causal_edge(self) -> bool:
        """Check if this is a verified causal edge."""
        return self.edge_type in {EdgeType.CAUSAL_STRONG, EdgeType.CAUSAL_WEAK}


@dataclass
class CircuitPath:
    """A path through the circuit graph."""

    nodes: List[CircuitNode]
    edges: List[CircuitEdge]

    # Path metrics
    total_weight: float = 0.0
    min_activation: float = 0.0
    path_importance: float = 0.0

    # Causal verification
    is_causal_path: bool = False
    causal_effect: float = 0.0

    def __post_init__(self):
        """Compute path metrics."""
        if self.edges:
            self.total_weight = sum(e.weight for e in self.edges)
        if self.nodes:
            activations = [n.activation for n in self.nodes if n.activation > 0]
            self.min_activation = min(activations) if activations else 0.0
            importances = [n.importance for n in self.nodes]
            self.path_importance = sum(importances) / len(importances) if importances else 0.0
        # Check if all edges are causal
        self.is_causal_path = all(e.is_causal for e in self.edges) if self.edges else False
        if self.is_causal_path:
            effects = [e.causal_effect for e in self.edges]
            self.causal_effect = min(effects) if effects else 0.0  # Bottleneck effect

    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self.edges)

    @property
    def source(self) -> Optional[CircuitNode]:
        """First node in the path."""
        return self.nodes[0] if self.nodes else None

    @property
    def target(self) -> Optional[CircuitNode]:
        """Last node in the path."""
        return self.nodes[-1] if self.nodes else None

    def contains_node(self, node_id: str) -> bool:
        """Check if path contains a node."""
        return any(n.node_id == node_id for n in self.nodes)

    def get_node_types(self) -> List[NodeType]:
        """Get sequence of node types along path."""
        return [n.node_type for n in self.nodes]


class CircuitGraph:
    """
    Graph-based representation of a computational circuit.

    Provides:
    - Node and edge management
    - Path finding (shortest, critical, all paths)
    - Subgraph extraction
    - Graph metrics and analysis
    - Integration with CircuitTrace

    Example:
        graph = CircuitGraph()

        # Add nodes
        graph.add_node(CircuitNode("input_0", NodeType.INPUT, layer_idx=0))
        graph.add_node(CircuitNode("attn_0_1", NodeType.ATTENTION_HEAD, layer_idx=0, head_idx=1))

        # Add edges
        graph.add_edge(CircuitEdge("input_0", "attn_0_1", EdgeType.FEEDFORWARD))

        # Find paths
        path = graph.get_shortest_path("input_0", "output_0")
        critical = graph.get_critical_path()
    """

    def __init__(self):
        """Initialize empty circuit graph."""
        self._nodes: Dict[str, CircuitNode] = {}
        self._edges: Dict[Tuple[str, str], List[CircuitEdge]] = {}
        self._adjacency: Dict[str, Set[str]] = {}  # Forward adjacency
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # Backward adjacency

    # ==================== Node Operations ====================

    def add_node(self, node: CircuitNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = set()
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = set()

    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return

        # Remove edges
        for target in list(self._adjacency.get(node_id, [])):
            self._remove_edge_internal(node_id, target)
        for source in list(self._reverse_adjacency.get(node_id, [])):
            self._remove_edge_internal(source, node_id)

        # Remove node
        del self._nodes[node_id]
        self._adjacency.pop(node_id, None)
        self._reverse_adjacency.pop(node_id, None)

    def get_nodes(
        self,
        node_type: Optional[NodeType] = None,
        layer_idx: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[CircuitNode]:
        """Get nodes with optional filtering."""
        nodes = list(self._nodes.values())

        if node_type is not None:
            nodes = [n for n in nodes if n.node_type == node_type]
        if layer_idx is not None:
            nodes = [n for n in nodes if n.layer_idx == layer_idx]
        if min_importance > 0:
            nodes = [n for n in nodes if n.importance >= min_importance]

        return nodes

    @property
    def nodes(self) -> List[CircuitNode]:
        """All nodes in the graph."""
        return list(self._nodes.values())

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    # ==================== Edge Operations ====================

    def add_edge(self, edge: CircuitEdge) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not in graph")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not in graph")

        key = (edge.source_id, edge.target_id)
        if key not in self._edges:
            self._edges[key] = []
        self._edges[key].append(edge)

        self._adjacency[edge.source_id].add(edge.target_id)
        self._reverse_adjacency[edge.target_id].add(edge.source_id)

    def get_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[EdgeType] = None
    ) -> Optional[CircuitEdge]:
        """Get an edge between two nodes."""
        key = (source_id, target_id)
        edges = self._edges.get(key, [])

        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges[0] if edges else None

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
        causal_only: bool = False
    ) -> List[CircuitEdge]:
        """Get edges with optional filtering."""
        all_edges: List[CircuitEdge] = []
        for edge_list in self._edges.values():
            all_edges.extend(edge_list)

        if source_id is not None:
            all_edges = [e for e in all_edges if e.source_id == source_id]
        if target_id is not None:
            all_edges = [e for e in all_edges if e.target_id == target_id]
        if edge_type is not None:
            all_edges = [e for e in all_edges if e.edge_type == edge_type]
        if causal_only:
            all_edges = [e for e in all_edges if e.is_causal]

        return all_edges

    def _remove_edge_internal(self, source_id: str, target_id: str) -> None:
        """Internal edge removal."""
        key = (source_id, target_id)
        if key in self._edges:
            del self._edges[key]
        self._adjacency.get(source_id, set()).discard(target_id)
        self._reverse_adjacency.get(target_id, set()).discard(source_id)

    def remove_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[EdgeType] = None
    ) -> None:
        """Remove edge(s) between two nodes."""
        key = (source_id, target_id)
        if key not in self._edges:
            return

        if edge_type is None:
            # Remove all edges
            self._remove_edge_internal(source_id, target_id)
        else:
            # Remove specific edge type
            self._edges[key] = [e for e in self._edges[key] if e.edge_type != edge_type]
            if not self._edges[key]:
                self._remove_edge_internal(source_id, target_id)

    @property
    def edges(self) -> List[CircuitEdge]:
        """All edges in the graph."""
        all_edges: List[CircuitEdge] = []
        for edge_list in self._edges.values():
            all_edges.extend(edge_list)
        return all_edges

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return sum(len(edges) for edges in self._edges.values())

    # ==================== Adjacency Operations ====================

    def get_successors(self, node_id: str) -> List[CircuitNode]:
        """Get nodes that this node connects to."""
        successor_ids = self._adjacency.get(node_id, set())
        return [self._nodes[nid] for nid in successor_ids if nid in self._nodes]

    def get_predecessors(self, node_id: str) -> List[CircuitNode]:
        """Get nodes that connect to this node."""
        predecessor_ids = self._reverse_adjacency.get(node_id, set())
        return [self._nodes[nid] for nid in predecessor_ids if nid in self._nodes]

    def get_neighbors(self, node_id: str) -> List[CircuitNode]:
        """Get all connected nodes (both directions)."""
        neighbor_ids = self._adjacency.get(node_id, set()) | self._reverse_adjacency.get(node_id, set())
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    # ==================== Path Finding ====================

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight_key: str = "weight"
    ) -> Optional[CircuitPath]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            weight_key: Edge attribute to use as weight

        Returns:
            CircuitPath if path exists, None otherwise
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        # Dijkstra's algorithm
        distances: Dict[str, float] = {source_id: 0}
        predecessors: Dict[str, Tuple[str, CircuitEdge]] = {}
        heap = [(0, source_id)]
        visited: Set[str] = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current == target_id:
                break

            for neighbor_id in self._adjacency.get(current, []):
                if neighbor_id in visited:
                    continue

                # Get edge weight
                edges = self._edges.get((current, neighbor_id), [])
                if not edges:
                    continue
                edge = edges[0]  # Use first edge
                edge_weight = getattr(edge, weight_key, 1.0) if hasattr(edge, weight_key) else edge.weight

                new_dist = dist + edge_weight
                if neighbor_id not in distances or new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = (current, edge)
                    heapq.heappush(heap, (new_dist, neighbor_id))

        # Reconstruct path
        if target_id not in predecessors and source_id != target_id:
            return None

        path_nodes: List[CircuitNode] = []
        path_edges: List[CircuitEdge] = []
        current = target_id

        while current != source_id:
            path_nodes.append(self._nodes[current])
            pred, edge = predecessors[current]
            path_edges.append(edge)
            current = pred

        path_nodes.append(self._nodes[source_id])
        path_nodes.reverse()
        path_edges.reverse()

        return CircuitPath(nodes=path_nodes, edges=path_edges)

    def get_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 10,
        causal_only: bool = False
    ) -> List[CircuitPath]:
        """
        Find all paths between two nodes (up to max length).

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            max_length: Maximum path length
            causal_only: Only include causal edges

        Returns:
            List of CircuitPath objects
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        paths: List[CircuitPath] = []

        def dfs(
            current: str,
            path_nodes: List[CircuitNode],
            path_edges: List[CircuitEdge],
            visited: Set[str]
        ):
            if len(path_edges) > max_length:
                return

            if current == target_id:
                paths.append(CircuitPath(nodes=list(path_nodes), edges=list(path_edges)))
                return

            for neighbor_id in self._adjacency.get(current, []):
                if neighbor_id in visited:
                    continue

                edges = self._edges.get((current, neighbor_id), [])
                if causal_only:
                    edges = [e for e in edges if e.is_causal]
                if not edges:
                    continue

                edge = edges[0]
                neighbor_node = self._nodes[neighbor_id]

                path_nodes.append(neighbor_node)
                path_edges.append(edge)
                visited.add(neighbor_id)

                dfs(neighbor_id, path_nodes, path_edges, visited)

                path_nodes.pop()
                path_edges.pop()
                visited.remove(neighbor_id)

        source_node = self._nodes[source_id]
        dfs(source_id, [source_node], [], {source_id})

        return paths

    def get_critical_path(self) -> Optional[CircuitPath]:
        """
        Find the critical path (highest importance path from input to output).

        Returns:
            CircuitPath with highest total importance
        """
        # Find input and output nodes
        input_nodes = self.get_nodes(node_type=NodeType.INPUT)
        output_nodes = self.get_nodes(node_type=NodeType.OUTPUT)

        if not input_nodes or not output_nodes:
            # Fall back to nodes with no predecessors/successors
            input_nodes = [n for n in self.nodes if not self.get_predecessors(n.node_id)]
            output_nodes = [n for n in self.nodes if not self.get_successors(n.node_id)]

        if not input_nodes or not output_nodes:
            return None

        best_path: Optional[CircuitPath] = None
        best_importance = -float('inf')

        for input_node in input_nodes:
            for output_node in output_nodes:
                paths = self.get_all_paths(input_node.node_id, output_node.node_id, max_length=20)
                for path in paths:
                    if path.path_importance > best_importance:
                        best_importance = path.path_importance
                        best_path = path

        return best_path

    # ==================== Subgraph Operations ====================

    def extract_subgraph(
        self,
        node_ids: Set[str],
        include_connecting_edges: bool = True
    ) -> "CircuitGraph":
        """
        Extract a subgraph containing specified nodes.

        Args:
            node_ids: Set of node IDs to include
            include_connecting_edges: Include edges between selected nodes

        Returns:
            New CircuitGraph containing the subgraph
        """
        subgraph = CircuitGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                subgraph.add_node(self._nodes[node_id])

        # Add edges
        if include_connecting_edges:
            for (source_id, target_id), edges in self._edges.items():
                if source_id in node_ids and target_id in node_ids:
                    for edge in edges:
                        subgraph.add_edge(edge)

        return subgraph

    def get_layer_subgraph(self, layer_idx: int) -> "CircuitGraph":
        """Extract subgraph for a specific layer."""
        layer_nodes = self.get_nodes(layer_idx=layer_idx)
        node_ids = {n.node_id for n in layer_nodes}
        return self.extract_subgraph(node_ids)

    def get_causal_subgraph(self) -> "CircuitGraph":
        """Extract subgraph containing only causal edges."""
        subgraph = CircuitGraph()

        # Add all nodes
        for node in self.nodes:
            subgraph.add_node(node)

        # Add only causal edges
        for edge in self.get_edges(causal_only=True):
            subgraph.add_edge(edge)

        return subgraph

    # ==================== Graph Metrics ====================

    def get_density(self) -> float:
        """Compute graph density (edges / possible edges)."""
        n = self.n_nodes
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)  # Directed graph
        return self.n_edges / max_edges

    def get_in_degree(self, node_id: str) -> int:
        """Get in-degree of a node."""
        return len(self._reverse_adjacency.get(node_id, set()))

    def get_out_degree(self, node_id: str) -> int:
        """Get out-degree of a node."""
        return len(self._adjacency.get(node_id, set()))

    def get_degree_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get degree distribution for all nodes."""
        distribution = {}
        for node_id in self._nodes:
            distribution[node_id] = {
                "in_degree": self.get_in_degree(node_id),
                "out_degree": self.get_out_degree(node_id),
                "total_degree": self.get_in_degree(node_id) + self.get_out_degree(node_id)
            }
        return distribution

    def get_hub_nodes(self, top_k: int = 5) -> List[CircuitNode]:
        """Get nodes with highest connectivity (hubs)."""
        degree_dist = self.get_degree_distribution()
        sorted_nodes = sorted(
            self._nodes.keys(),
            key=lambda nid: degree_dist[nid]["total_degree"],
            reverse=True
        )
        return [self._nodes[nid] for nid in sorted_nodes[:top_k]]

    # ==================== Conversion ====================

    @classmethod
    def from_trace(cls, trace: Any) -> "CircuitGraph":
        """
        Create CircuitGraph from a CircuitTrace.

        Args:
            trace: CircuitTrace object from tracer module

        Returns:
            CircuitGraph representation
        """
        from hololoom.dark_trace.circuits.tracer import CircuitTrace, TraceNode, TraceEdge

        if not isinstance(trace, CircuitTrace):
            raise TypeError(f"Expected CircuitTrace, got {type(trace)}")

        graph = cls()

        # Convert trace nodes to graph nodes
        for trace_node in trace.nodes:
            # Map component type to node type
            node_type_map = {
                "embedding": NodeType.EMBEDDING,
                "attention": NodeType.ATTENTION_HEAD,
                "mlp": NodeType.MLP_LAYER,
                "sae_feature": NodeType.SAE_FEATURE,
            }
            node_type = node_type_map.get(
                trace_node.component_type.value,
                NodeType.CONCEPT
            )

            circuit_node = CircuitNode(
                node_id=trace_node.node_id,
                node_type=node_type,
                layer_idx=trace_node.layer_idx,
                activation=trace_node.activation,
                importance=trace_node.importance,
                metadata=trace_node.metadata
            )
            graph.add_node(circuit_node)

        # Convert trace edges to graph edges
        for trace_edge in trace.edges:
            edge_type = EdgeType.CAUSAL_STRONG if trace_edge.is_causal else EdgeType.CORRELATIONAL

            circuit_edge = CircuitEdge(
                source_id=trace_edge.source_id,
                target_id=trace_edge.target_id,
                edge_type=edge_type,
                weight=trace_edge.weight,
                is_causal=trace_edge.is_causal,
                metadata=trace_edge.metadata
            )
            graph.add_edge(circuit_edge)

        return graph

    def to_networkx(self) -> Any:
        """
        Convert to NetworkX DiGraph for advanced analysis.

        Returns:
            networkx.DiGraph (or raises if networkx not available)
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required for to_networkx()")

        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(
                node.node_id,
                node_type=node.node_type.value,
                layer_idx=node.layer_idx,
                activation=node.activation,
                importance=node.importance
            )

        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                is_causal=edge.is_causal
            )

        return G

    # ==================== Iteration ====================

    def iter_nodes(self) -> Iterator[CircuitNode]:
        """Iterate over nodes."""
        return iter(self._nodes.values())

    def iter_edges(self) -> Iterator[CircuitEdge]:
        """Iterate over edges."""
        for edge_list in self._edges.values():
            yield from edge_list

    def iter_layers(self) -> Iterator[Tuple[int, List[CircuitNode]]]:
        """Iterate over layers and their nodes."""
        layer_nodes: Dict[int, List[CircuitNode]] = {}
        for node in self.nodes:
            if node.layer_idx not in layer_nodes:
                layer_nodes[node.layer_idx] = []
            layer_nodes[node.layer_idx].append(node)

        for layer_idx in sorted(layer_nodes.keys()):
            yield layer_idx, layer_nodes[layer_idx]

    def __repr__(self) -> str:
        return f"CircuitGraph(nodes={self.n_nodes}, edges={self.n_edges})"
