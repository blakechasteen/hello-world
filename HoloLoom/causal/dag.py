"""
Causal DAG Implementation

Core data structures for Pearl-style causal models.
Implements directed acyclic graphs with d-separation for conditional independence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
import networkx as nx


class NodeType(Enum):
    """Type of causal node."""
    OBSERVABLE = "observable"      # Can be directly observed
    LATENT = "latent"              # Hidden/unobserved variable
    INTERVENTION = "intervention"  # Set by do() operator
    DECISION = "decision"          # Agent decision node


@dataclass
class CausalNode:
    """
    Variable in causal graph.

    Attributes:
        name: Unique identifier
        node_type: Observable, latent, intervention, or decision
        domain: Possible values (discrete) or range (continuous)
        description: Human-readable description
        metadata: Additional information
    """
    name: str
    node_type: NodeType = NodeType.OBSERVABLE
    domain: Optional[List[Any]] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"CausalNode({self.name}, {self.node_type.value})"


@dataclass
class CausalEdge:
    """
    Causal relationship between nodes.

    Attributes:
        source: Cause node
        target: Effect node
        strength: Causal strength (0-1, or unbounded)
        mechanism: Description of causal mechanism
        confidence: How certain we are about this edge (0-1)
        metadata: Additional information
    """
    source: str
    target: str
    strength: float = 1.0
    mechanism: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"CausalEdge({self.source} → {self.target}, strength={self.strength:.2f})"


class CausalDAG:
    """
    Directed Acyclic Graph for causal models.

    Supports:
    - d-separation for conditional independence
    - Topological ordering
    - Parent/child/ancestor/descendant queries
    - Markov blanket computation
    - Graph manipulation (add/remove nodes/edges)

    Based on Pearl's causal hierarchy:
    - Level 1: Association (observational)
    - Level 2: Intervention (do-calculus)
    - Level 3: Counterfactuals (twin networks)
    """

    def __init__(self):
        """Initialize empty causal DAG."""
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.graph = nx.DiGraph()

    def add_node(self, node: CausalNode) -> None:
        """Add node to DAG."""
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists")

        self.nodes[node.name] = node
        self.graph.add_node(node.name, **{
            'node_type': node.node_type,
            'description': node.description,
            'metadata': node.metadata
        })

    def add_edge(self, edge: CausalEdge) -> None:
        """
        Add causal edge to DAG.

        Validates that:
        - Both nodes exist
        - Adding edge doesn't create cycle
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} doesn't exist")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} doesn't exist")

        # Check for cycles
        self.graph.add_edge(edge.source, edge.target)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(edge.source, edge.target)
            raise ValueError(f"Adding edge {edge.source} → {edge.target} would create cycle")

        self.edges[(edge.source, edge.target)] = edge
        # Store edge metadata in networkx
        self.graph.edges[edge.source, edge.target]['strength'] = edge.strength
        self.graph.edges[edge.source, edge.target]['mechanism'] = edge.mechanism
        self.graph.edges[edge.source, edge.target]['confidence'] = edge.confidence

    def remove_edge(self, source: str, target: str) -> None:
        """Remove causal edge."""
        if (source, target) in self.edges:
            del self.edges[(source, target)]
            self.graph.remove_edge(source, target)

    def parents(self, node: str) -> Set[str]:
        """Get direct parents (causes) of node."""
        return set(self.graph.predecessors(node))

    def children(self, node: str) -> Set[str]:
        """Get direct children (effects) of node."""
        return set(self.graph.successors(node))

    def ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (transitive causes) of node."""
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> Set[str]:
        """Get all descendants (transitive effects) of node."""
        return nx.descendants(self.graph, node)

    def markov_blanket(self, node: str) -> Set[str]:
        """
        Get Markov blanket of node.

        Markov blanket = parents + children + co-parents (parents of children).
        This is the minimal set that d-separates node from rest of graph.
        """
        parents = self.parents(node)
        children = self.children(node)

        # Co-parents: parents of node's children (excluding node itself)
        co_parents = set()
        for child in children:
            co_parents.update(self.parents(child) - {node})

        return parents | children | co_parents

    def topological_order(self) -> List[str]:
        """
        Get topological ordering of nodes.

        Returns nodes in order where all parents come before children.
        Useful for sampling from causal model.
        """
        return list(nx.topological_sort(self.graph))

    def is_d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.

        d-separation is the graphical criterion for conditional independence:
        X ⊥ Y | Z  iff  X and Y are d-separated by Z

        Args:
            X: First set of nodes
            Y: Second set of nodes
            Z: Conditioning set

        Returns:
            True if X and Y are d-separated by Z
        """
        return nx.d_separated(self.graph, X, Y, Z)

    def get_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Get all paths from source to target.

        Useful for understanding causal pathways.
        """
        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []

    def find_colliders(self) -> List[str]:
        """
        Find all colliders in the graph.

        A collider is a node with multiple parents (X → Z ← Y).
        Colliders block association but can be opened by conditioning.
        """
        colliders = []
        for node in self.nodes:
            if len(self.parents(node)) >= 2:
                colliders.append(node)
        return colliders

    def find_confounders(self, X: str, Y: str) -> Set[str]:
        """
        Find confounders between X and Y.

        A confounder Z has paths to both X and Y (Z → X, Z → Y).
        Confounders create spurious associations.
        """
        X_ancestors = self.ancestors(X) | {X}
        Y_ancestors = self.ancestors(Y) | {Y}

        # Common ancestors are potential confounders
        return X_ancestors & Y_ancestors - {X, Y}

    def find_mediators(self, X: str, Y: str) -> Set[str]:
        """
        Find mediators between X and Y.

        A mediator M is on a directed path X → M → Y.
        Mediators transmit causal effects.
        """
        mediators = set()

        # Find all paths from X to Y
        paths = self.get_paths(X, Y)

        # Extract intermediate nodes
        for path in paths:
            mediators.update(path[1:-1])  # Exclude X and Y themselves

        return mediators

    def backdoor_paths(self, X: str, Y: str) -> List[List[str]]:
        """
        Find backdoor paths from X to Y.

        Backdoor path: Path from X to Y that starts with an arrow INTO X.
        These paths create confounding bias.
        """
        backdoor = []

        # Get parents of X (arrows into X)
        for parent in self.parents(X):
            # Find paths from parent to Y that don't go through X
            subgraph = self.graph.copy()
            subgraph.remove_node(X)

            try:
                for path in nx.all_simple_paths(subgraph, parent, Y):
                    backdoor.append([X] + path)
            except nx.NetworkXNoPath:
                continue

        return backdoor

    def satisfies_backdoor_criterion(self, X: str, Y: str, Z: Set[str]) -> bool:
        """
        Check if Z satisfies backdoor criterion for causal effect of X on Y.

        Backdoor criterion:
        1. Z blocks all backdoor paths from X to Y
        2. Z contains no descendants of X

        If satisfied, P(Y|do(X=x)) = Σ_z P(Y|X=x,Z=z)P(Z=z)
        """
        # Check condition 2: Z contains no descendants of X
        X_descendants = self.descendants(X)
        if Z & X_descendants:
            return False

        # Check condition 1: Z d-separates X from Y via backdoor paths
        # For each backdoor path, check if it's blocked by Z
        backdoors = self.backdoor_paths(X, Y)

        for path in backdoors:
            # Check if this path is blocked by Z
            # A path is blocked if any non-collider is in Z,
            # or if a collider is NOT in Z (and no descendant of collider is in Z)
            blocked = False

            for i in range(len(path) - 1):
                node = path[i]

                # Check if node is a collider on this path
                if i > 0 and i < len(path) - 1:
                    prev_node = path[i - 1]
                    next_node = path[i + 1]

                    # Collider: both arrows point into node
                    is_collider = (
                        (prev_node in self.children(node) or node in self.parents(prev_node)) and
                        (next_node in self.children(node) or node in self.parents(next_node))
                    )

                    if is_collider:
                        # Collider blocks path unless in Z (or descendant in Z)
                        node_and_descendants = {node} | self.descendants(node)
                        if not (node_and_descendants & Z):
                            blocked = True
                            break
                    else:
                        # Non-collider blocks path if in Z
                        if node in Z:
                            blocked = True
                            break

            if not blocked:
                return False  # Found unblocked backdoor path

        return True  # All backdoor paths blocked

    def frontdoor_paths(self, X: str, Y: str) -> List[List[str]]:
        """
        Find frontdoor paths from X to Y.

        Frontdoor path: Directed path X → M → Y where M intercepts all
        directed paths from X to Y.
        """
        # Find all directed paths from X to Y
        all_paths = self.get_paths(X, Y)

        if not all_paths:
            return []

        # Find nodes that appear in ALL paths (necessary mediators)
        path_sets = [set(path[1:-1]) for path in all_paths]  # Exclude X and Y
        necessary_mediators = set.intersection(*path_sets) if path_sets else set()

        # Frontdoor paths are those through necessary mediators
        frontdoor = []
        for mediator in necessary_mediators:
            # Check if there's a path X → M and M → Y
            X_to_M = self.get_paths(X, mediator)
            M_to_Y = self.get_paths(mediator, Y)

            if X_to_M and M_to_Y:
                # Combine paths
                for p1 in X_to_M:
                    for p2 in M_to_Y:
                        frontdoor.append(p1 + p2[1:])

        return frontdoor

    def satisfies_frontdoor_criterion(self, X: str, Y: str, Z: Set[str]) -> bool:
        """
        Check if Z satisfies frontdoor criterion for causal effect of X on Y.

        Frontdoor criterion:
        1. Z intercepts all directed paths from X to Y
        2. No backdoor paths from X to Z
        3. X blocks all backdoor paths from Z to Y

        Useful when there are unobserved confounders between X and Y.
        """
        # Condition 1: Z intercepts all directed paths X → Y
        all_paths = self.get_paths(X, Y)
        for path in all_paths:
            if not (set(path[1:-1]) & Z):
                return False  # Found path not intercepted by Z

        # Condition 2: No unblocked backdoor paths from X to Z
        for z in Z:
            backdoors = self.backdoor_paths(X, z)
            if backdoors:
                return False

        # Condition 3: X blocks all backdoor paths from Z to Y
        for z in Z:
            if not self.is_d_separated({z}, {Y}, {X}):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            'nodes': [
                {
                    'name': n.name,
                    'node_type': n.node_type.value,
                    'domain': n.domain,
                    'description': n.description,
                    'metadata': n.metadata
                }
                for n in self.nodes.values()
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'strength': e.strength,
                    'mechanism': e.mechanism,
                    'confidence': e.confidence,
                    'metadata': e.metadata
                }
                for e in self.edges.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalDAG':
        """Deserialize DAG from dictionary."""
        dag = cls()

        # Add nodes
        for node_data in data['nodes']:
            node = CausalNode(
                name=node_data['name'],
                node_type=NodeType(node_data['node_type']),
                domain=node_data.get('domain'),
                description=node_data.get('description', ''),
                metadata=node_data.get('metadata', {})
            )
            dag.add_node(node)

        # Add edges
        for edge_data in data['edges']:
            edge = CausalEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                strength=edge_data.get('strength', 1.0),
                mechanism=edge_data.get('mechanism', ''),
                confidence=edge_data.get('confidence', 1.0),
                metadata=edge_data.get('metadata', {})
            )
            dag.add_edge(edge)

        return dag

    def __repr__(self):
        return f"CausalDAG({len(self.nodes)} nodes, {len(self.edges)} edges)"
