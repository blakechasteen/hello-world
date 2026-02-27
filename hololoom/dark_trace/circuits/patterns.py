"""
Circuit Pattern Detection: Known Patterns and Anomaly Detection

Provides pattern matching for identifying known circuit structures
and detecting anomalous circuits that may indicate issues.

Research Basis:
- IOI Circuit (Wang et al., 2022): Canonical indirect object identification circuit
- Induction Heads (Olsson et al., 2022): In-context learning circuits
- Greater-Than Circuit (Hanna et al., 2023): Numerical comparison circuits
- Docstring Circuit: Code understanding patterns

Key Features:
- Library of known circuit patterns
- Pattern matching with structural similarity
- Anomaly detection for unusual circuits
- Pattern composition and decomposition
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import re


class PatternCategory(Enum):
    """Categories of known circuit patterns."""

    # Language understanding
    SYNTACTIC = "syntactic"           # Grammar/syntax processing
    SEMANTIC = "semantic"             # Meaning/semantics processing
    FACTUAL = "factual"               # Factual recall

    # Reasoning
    LOGICAL = "logical"               # Logical reasoning
    MATHEMATICAL = "mathematical"     # Mathematical operations
    ANALOGICAL = "analogical"         # Analogy/comparison

    # Copying/Retrieval
    INDUCTION = "induction"           # In-context learning
    COPYING = "copying"               # Direct copying
    RETRIEVAL = "retrieval"           # Memory retrieval

    # Safety-relevant
    DECEPTIVE = "deceptive"           # Potential deception
    SYCOPHANTIC = "sycophantic"       # Sycophancy patterns
    REFUSAL = "refusal"               # Refusal patterns

    # Meta
    COMPOSITION = "composition"       # Composed of sub-patterns
    UNKNOWN = "unknown"               # Unclassified


class AnomalyType(Enum):
    """Types of circuit anomalies."""

    # Structural anomalies
    DISCONNECTED = "disconnected"           # Isolated components
    EXCESSIVE_DEPTH = "excessive_depth"     # Unusually deep paths
    BOTTLENECK = "bottleneck"               # Information bottleneck
    REDUNDANT = "redundant"                 # Redundant pathways

    # Activation anomalies
    DEAD_FEATURES = "dead_features"         # Features never activate
    SATURATED = "saturated"                 # Always-on features
    POLYSEMANTIC = "polysemantic"           # Feature fires for unrelated inputs

    # Causal anomalies
    ACAUSAL = "acausal"                     # Correlational but not causal
    CIRCULAR = "circular"                   # Circular dependencies

    # Safety anomalies
    BYPASS = "bypass"                       # Safety bypass
    HIDDEN_PATH = "hidden_path"             # Hidden information flow


@dataclass
class PatternSignature:
    """
    Structural signature for pattern matching.

    Describes expected structure without concrete node IDs.
    """

    # Node type sequence (e.g., [ATTENTION_HEAD, MLP_LAYER, SAE_FEATURE])
    node_type_sequence: List[str]

    # Edge type constraints
    edge_type_constraints: Dict[Tuple[int, int], str] = field(default_factory=dict)

    # Layer constraints
    layer_span: Optional[Tuple[int, int]] = None  # (min_layer, max_layer)
    layer_progression: str = "forward"  # forward, backward, any

    # Activation constraints
    min_activation_threshold: float = 0.0
    activation_pattern: Optional[str] = None  # regex for activation sequence

    # Structural constraints
    min_nodes: int = 1
    max_nodes: int = 100
    allow_branching: bool = True
    allow_merging: bool = True


@dataclass
class KnownPattern:
    """
    A known circuit pattern (e.g., IOI circuit, induction head).

    Represents a canonical circuit structure that has been
    identified and characterized in interpretability research.
    """

    pattern_id: str
    name: str
    description: str
    category: PatternCategory

    # Pattern structure
    signature: PatternSignature

    # Canonical example
    canonical_nodes: List[str] = field(default_factory=list)
    canonical_edges: List[Tuple[str, str]] = field(default_factory=list)

    # Research reference
    paper_reference: Optional[str] = None
    discovery_date: Optional[str] = None

    # Pattern characteristics
    is_safety_relevant: bool = False
    typical_layers: Optional[Tuple[int, int]] = None
    expected_features: List[str] = field(default_factory=list)

    # Matching configuration
    similarity_threshold: float = 0.7
    required_components: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.pattern_id)


@dataclass
class PatternMatch:
    """Result of pattern matching."""

    pattern: KnownPattern
    matched_nodes: List[str]
    matched_edges: List[Tuple[str, str]]

    # Match quality
    similarity_score: float
    structural_match: float  # How well structure matches
    activation_match: float  # How well activations match

    # Alignment
    node_alignment: Dict[str, str] = field(default_factory=dict)  # pattern_node -> graph_node

    # Confidence
    confidence: float = 0.0
    is_partial_match: bool = False
    missing_components: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute confidence."""
        self.confidence = (
            0.5 * self.structural_match +
            0.3 * self.activation_match +
            0.2 * (1.0 - len(self.missing_components) / max(len(self.pattern.canonical_nodes), 1))
        )

    @property
    def is_strong_match(self) -> bool:
        """Check if this is a high-confidence match."""
        return self.confidence >= self.pattern.similarity_threshold and not self.is_partial_match


@dataclass
class Anomaly:
    """Detected circuit anomaly."""

    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    description: str

    # Location
    affected_nodes: List[str] = field(default_factory=list)
    affected_edges: List[Tuple[str, str]] = field(default_factory=list)
    layer_range: Optional[Tuple[int, int]] = None

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        """Check if anomaly is critical severity."""
        return self.severity >= 0.8

    @property
    def is_safety_relevant(self) -> bool:
        """Check if anomaly is safety-relevant."""
        return self.anomaly_type in {
            AnomalyType.BYPASS,
            AnomalyType.HIDDEN_PATH,
            AnomalyType.DECEPTIVE if hasattr(AnomalyType, 'DECEPTIVE') else None
        }


class PatternLibrary:
    """
    Library of known circuit patterns.

    Contains canonical patterns identified in interpretability research,
    including IOI circuits, induction heads, and safety-relevant patterns.
    """

    def __init__(self):
        """Initialize with built-in patterns."""
        self._patterns: Dict[str, KnownPattern] = {}
        self._load_builtin_patterns()

    def _load_builtin_patterns(self):
        """Load built-in known patterns."""

        # IOI Circuit (Indirect Object Identification)
        ioi_pattern = KnownPattern(
            pattern_id="ioi_circuit",
            name="Indirect Object Identification Circuit",
            description=(
                "Circuit that identifies the indirect object in sentences like "
                "'When Mary and John went to the store, John gave a drink to'. "
                "Involves name mover heads and S-inhibition heads."
            ),
            category=PatternCategory.SYNTACTIC,
            signature=PatternSignature(
                node_type_sequence=["attention_head", "attention_head", "mlp_layer"],
                layer_span=(7, 11),
                layer_progression="forward",
                min_nodes=5,
                max_nodes=20
            ),
            paper_reference="Wang et al., 2022 - Interpretability in the Wild",
            is_safety_relevant=False,
            typical_layers=(7, 11),
            expected_features=["name_mover", "s_inhibition", "duplicate_token"]
        )
        self.add_pattern(ioi_pattern)

        # Induction Head
        induction_pattern = KnownPattern(
            pattern_id="induction_head",
            name="Induction Head Circuit",
            description=(
                "Circuit for in-context learning that copies patterns from earlier "
                "in the context. Composed of previous token head + induction head."
            ),
            category=PatternCategory.INDUCTION,
            signature=PatternSignature(
                node_type_sequence=["attention_head", "attention_head"],
                layer_span=(1, 8),
                layer_progression="forward",
                min_nodes=2,
                max_nodes=10
            ),
            paper_reference="Olsson et al., 2022 - In-context Learning and Induction Heads",
            is_safety_relevant=False,
            typical_layers=(1, 8),
            expected_features=["previous_token", "induction"]
        )
        self.add_pattern(induction_pattern)

        # Greater-Than Circuit
        greater_than_pattern = KnownPattern(
            pattern_id="greater_than",
            name="Greater-Than Circuit",
            description=(
                "Circuit for numerical comparison, determining which of two "
                "numbers is larger. Involves specific attention and MLP patterns."
            ),
            category=PatternCategory.MATHEMATICAL,
            signature=PatternSignature(
                node_type_sequence=["embedding", "attention_head", "mlp_layer", "mlp_layer"],
                layer_span=(0, 12),
                layer_progression="forward",
                min_nodes=4
            ),
            paper_reference="Hanna et al., 2023 - How does GPT-2 compute greater-than?",
            is_safety_relevant=False,
            typical_layers=(0, 12),
            expected_features=["number_embedding", "comparison"]
        )
        self.add_pattern(greater_than_pattern)

        # Copy Suppression Head
        copy_suppression_pattern = KnownPattern(
            pattern_id="copy_suppression",
            name="Copy Suppression Head",
            description=(
                "Attention head that suppresses direct copying of tokens, "
                "important for generating novel text rather than repetition."
            ),
            category=PatternCategory.COPYING,
            signature=PatternSignature(
                node_type_sequence=["attention_head"],
                layer_span=(8, 12),
                layer_progression="any",
                min_nodes=1,
                max_nodes=5
            ),
            is_safety_relevant=False,
            typical_layers=(8, 12)
        )
        self.add_pattern(copy_suppression_pattern)

        # Sycophancy Pattern (Safety-Relevant)
        sycophancy_pattern = KnownPattern(
            pattern_id="sycophancy",
            name="Sycophancy Circuit",
            description=(
                "Circuit that may cause model to agree with user statements "
                "even when incorrect. Safety-relevant for alignment."
            ),
            category=PatternCategory.SYCOPHANTIC,
            signature=PatternSignature(
                node_type_sequence=["attention_head", "sae_feature", "mlp_layer"],
                layer_span=(5, 15),
                layer_progression="forward"
            ),
            is_safety_relevant=True,
            expected_features=["user_sentiment", "agreement", "validation"]
        )
        self.add_pattern(sycophancy_pattern)

        # Refusal Circuit (Safety-Relevant)
        refusal_pattern = KnownPattern(
            pattern_id="refusal",
            name="Refusal Circuit",
            description=(
                "Circuit responsible for refusing harmful requests. "
                "Critical for safety - bypass of this circuit is concerning."
            ),
            category=PatternCategory.REFUSAL,
            signature=PatternSignature(
                node_type_sequence=["attention_head", "sae_feature", "mlp_layer"],
                layer_span=(10, 20),
                layer_progression="forward"
            ),
            is_safety_relevant=True,
            expected_features=["harm_detection", "refusal_trigger", "safety_output"]
        )
        self.add_pattern(refusal_pattern)

    def add_pattern(self, pattern: KnownPattern) -> None:
        """Add a pattern to the library."""
        self._patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[KnownPattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_patterns_by_category(self, category: PatternCategory) -> List[KnownPattern]:
        """Get all patterns in a category."""
        return [p for p in self._patterns.values() if p.category == category]

    def get_safety_relevant_patterns(self) -> List[KnownPattern]:
        """Get all safety-relevant patterns."""
        return [p for p in self._patterns.values() if p.is_safety_relevant]

    @property
    def patterns(self) -> List[KnownPattern]:
        """All patterns in the library."""
        return list(self._patterns.values())

    def __len__(self) -> int:
        return len(self._patterns)


class CircuitPatternMatcher:
    """
    Matches circuit graphs against known patterns.

    Provides structural pattern matching to identify known
    circuit types in traced circuits.

    Example:
        matcher = CircuitPatternMatcher()

        # Match against all patterns
        matches = matcher.find_patterns(circuit_graph)

        # Match against specific pattern
        match = matcher.match_pattern(circuit_graph, "ioi_circuit")
    """

    def __init__(self, library: Optional[PatternLibrary] = None):
        """
        Initialize pattern matcher.

        Args:
            library: Pattern library (uses default if not provided)
        """
        self.library = library or PatternLibrary()

    def find_patterns(
        self,
        graph: Any,  # CircuitGraph
        min_confidence: float = 0.5,
        categories: Optional[List[PatternCategory]] = None
    ) -> List[PatternMatch]:
        """
        Find all matching patterns in a circuit graph.

        Args:
            graph: CircuitGraph to analyze
            min_confidence: Minimum match confidence
            categories: Optional category filter

        Returns:
            List of PatternMatch objects, sorted by confidence
        """
        matches = []

        patterns_to_check = self.library.patterns
        if categories:
            patterns_to_check = [p for p in patterns_to_check if p.category in categories]

        for pattern in patterns_to_check:
            match = self._match_single_pattern(graph, pattern)
            if match and match.confidence >= min_confidence:
                matches.append(match)

        # Sort by confidence (descending)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def match_pattern(
        self,
        graph: Any,  # CircuitGraph
        pattern_id: str
    ) -> Optional[PatternMatch]:
        """
        Match against a specific pattern.

        Args:
            graph: CircuitGraph to analyze
            pattern_id: Pattern ID to match against

        Returns:
            PatternMatch if found, None otherwise
        """
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            return None

        return self._match_single_pattern(graph, pattern)

    def _match_single_pattern(
        self,
        graph: Any,  # CircuitGraph
        pattern: KnownPattern
    ) -> Optional[PatternMatch]:
        """Match graph against a single pattern."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph

        if not isinstance(graph, CircuitGraph):
            return None

        # Check basic structural constraints
        signature = pattern.signature
        if graph.n_nodes < signature.min_nodes or graph.n_nodes > signature.max_nodes:
            return None

        # Find candidate subgraphs
        matched_nodes: List[str] = []
        matched_edges: List[Tuple[str, str]] = []
        node_alignment: Dict[str, str] = {}

        # Check layer span
        if signature.layer_span:
            min_layer, max_layer = signature.layer_span
            layer_nodes = [n for n in graph.nodes if min_layer <= n.layer_idx <= max_layer]
            if not layer_nodes:
                return None
            matched_nodes = [n.node_id for n in layer_nodes]

        # Check node type sequence
        if signature.node_type_sequence:
            type_counts = {}
            for node in graph.nodes:
                node_type_str = node.node_type.value
                type_counts[node_type_str] = type_counts.get(node_type_str, 0) + 1

            # Check if required types exist
            required_types = set(signature.node_type_sequence)
            if not required_types.issubset(set(type_counts.keys())):
                # Partial match
                missing = required_types - set(type_counts.keys())
                return PatternMatch(
                    pattern=pattern,
                    matched_nodes=matched_nodes,
                    matched_edges=matched_edges,
                    similarity_score=0.5,
                    structural_match=0.5,
                    activation_match=0.5,
                    is_partial_match=True,
                    missing_components=list(missing)
                )

        # Compute structural similarity
        structural_match = self._compute_structural_similarity(graph, pattern)

        # Compute activation match
        activation_match = self._compute_activation_match(graph, pattern)

        # Collect matched edges
        if matched_nodes:
            matched_node_set = set(matched_nodes)
            for edge in graph.edges:
                if edge.source_id in matched_node_set and edge.target_id in matched_node_set:
                    matched_edges.append((edge.source_id, edge.target_id))

        return PatternMatch(
            pattern=pattern,
            matched_nodes=matched_nodes if matched_nodes else [n.node_id for n in graph.nodes],
            matched_edges=matched_edges,
            similarity_score=(structural_match + activation_match) / 2,
            structural_match=structural_match,
            activation_match=activation_match,
            node_alignment=node_alignment
        )

    def _compute_structural_similarity(
        self,
        graph: Any,  # CircuitGraph
        pattern: KnownPattern
    ) -> float:
        """Compute structural similarity score."""
        signature = pattern.signature
        score = 1.0

        # Check layer span alignment
        if signature.layer_span:
            min_layer, max_layer = signature.layer_span
            graph_layers = {n.layer_idx for n in graph.nodes}
            if graph_layers:
                graph_min = min(graph_layers)
                graph_max = max(graph_layers)
                # Penalize if outside expected range
                if graph_min < min_layer or graph_max > max_layer:
                    score *= 0.8

        # Check layer progression
        if signature.layer_progression == "forward":
            # Check that paths generally go forward in layers
            for edge in graph.edges:
                source = graph.get_node(edge.source_id)
                target = graph.get_node(edge.target_id)
                if source and target and source.layer_idx > target.layer_idx:
                    score *= 0.9  # Penalize backward edges

        # Check branching/merging constraints
        if not signature.allow_branching:
            for node in graph.nodes:
                if graph.get_out_degree(node.node_id) > 1:
                    score *= 0.9

        if not signature.allow_merging:
            for node in graph.nodes:
                if graph.get_in_degree(node.node_id) > 1:
                    score *= 0.9

        return max(0.0, min(1.0, score))

    def _compute_activation_match(
        self,
        graph: Any,  # CircuitGraph
        pattern: KnownPattern
    ) -> float:
        """Compute activation pattern match score."""
        signature = pattern.signature

        if signature.min_activation_threshold > 0:
            # Check that activations meet threshold
            activations = [n.activation for n in graph.nodes]
            if activations:
                above_threshold = sum(1 for a in activations if a >= signature.min_activation_threshold)
                score = above_threshold / len(activations)
                return score

        # Default: assume activations match
        return 0.8


class AnomalyDetector:
    """
    Detects anomalies in circuit graphs.

    Identifies structural, activation, and safety-relevant
    anomalies that may indicate issues with the circuit.

    Example:
        detector = AnomalyDetector()
        anomalies = detector.detect(circuit_graph)

        critical = [a for a in anomalies if a.is_critical]
        safety = [a for a in anomalies if a.is_safety_relevant]
    """

    def __init__(
        self,
        dead_feature_threshold: float = 0.01,
        saturation_threshold: float = 0.99,
        bottleneck_ratio: float = 0.1,
        max_depth: int = 50
    ):
        """
        Initialize anomaly detector.

        Args:
            dead_feature_threshold: Activation below this is "dead"
            saturation_threshold: Activation above this is "saturated"
            bottleneck_ratio: Ratio triggering bottleneck detection
            max_depth: Maximum expected path depth
        """
        self.dead_feature_threshold = dead_feature_threshold
        self.saturation_threshold = saturation_threshold
        self.bottleneck_ratio = bottleneck_ratio
        self.max_depth = max_depth

    def detect(self, graph: Any) -> List[Anomaly]:  # CircuitGraph
        """
        Detect all anomalies in a circuit graph.

        Args:
            graph: CircuitGraph to analyze

        Returns:
            List of detected Anomaly objects
        """
        anomalies: List[Anomaly] = []

        # Structural anomalies
        anomalies.extend(self._detect_disconnected(graph))
        anomalies.extend(self._detect_excessive_depth(graph))
        anomalies.extend(self._detect_bottlenecks(graph))
        anomalies.extend(self._detect_circular(graph))

        # Activation anomalies
        anomalies.extend(self._detect_dead_features(graph))
        anomalies.extend(self._detect_saturated(graph))

        # Safety anomalies
        anomalies.extend(self._detect_hidden_paths(graph))

        return anomalies

    def _detect_disconnected(self, graph: Any) -> List[Anomaly]:
        """Detect disconnected components."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []

        # Find nodes with no edges
        isolated = []
        for node in graph.nodes:
            if graph.get_in_degree(node.node_id) == 0 and graph.get_out_degree(node.node_id) == 0:
                isolated.append(node.node_id)

        if isolated:
            severity = len(isolated) / max(graph.n_nodes, 1)
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.DISCONNECTED,
                severity=min(severity, 1.0),
                description=f"Found {len(isolated)} isolated nodes with no connections",
                affected_nodes=isolated,
                recommendations=["Check if isolated nodes should be connected to the circuit"]
            ))

        return anomalies

    def _detect_excessive_depth(self, graph: Any) -> List[Anomaly]:
        """Detect unusually deep paths."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []

        # Find longest path using DFS
        max_path_length = 0
        for node in graph.nodes:
            path_length = self._get_max_path_length(graph, node.node_id, set())
            max_path_length = max(max_path_length, path_length)

        if max_path_length > self.max_depth:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.EXCESSIVE_DEPTH,
                severity=min(max_path_length / self.max_depth, 1.0),
                description=f"Maximum path depth ({max_path_length}) exceeds threshold ({self.max_depth})",
                evidence={"max_depth": max_path_length},
                recommendations=["Consider if circuit is computing unnecessary intermediate steps"]
            ))

        return anomalies

    def _get_max_path_length(self, graph: Any, node_id: str, visited: Set[str]) -> int:
        """Get maximum path length from a node using DFS."""
        if node_id in visited:
            return 0

        visited.add(node_id)
        max_length = 0

        successors = graph.get_successors(node_id)
        for succ in successors:
            length = 1 + self._get_max_path_length(graph, succ.node_id, visited.copy())
            max_length = max(max_length, length)

        return max_length

    def _detect_bottlenecks(self, graph: Any) -> List[Anomaly]:
        """Detect information bottlenecks."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []

        # Find nodes where many edges converge to few
        for node in graph.nodes:
            in_degree = graph.get_in_degree(node.node_id)
            out_degree = graph.get_out_degree(node.node_id)

            # Bottleneck: many inputs, few outputs
            if in_degree > 5 and out_degree <= 1:
                severity = (in_degree - out_degree) / in_degree
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BOTTLENECK,
                    severity=severity,
                    description=f"Node {node.node_id} is a bottleneck: {in_degree} inputs, {out_degree} outputs",
                    affected_nodes=[node.node_id],
                    evidence={"in_degree": in_degree, "out_degree": out_degree},
                    recommendations=["Check if information loss occurs at this node"]
                ))

        return anomalies

    def _detect_circular(self, graph: Any) -> List[Anomaly]:
        """Detect circular dependencies."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []

        # Simple cycle detection using DFS
        def has_cycle_from(start: str, current: str, visited: Set[str], path: Set[str]) -> bool:
            if current in path:
                return True
            if current in visited:
                return False

            visited.add(current)
            path.add(current)

            for succ in graph.get_successors(current):
                if has_cycle_from(start, succ.node_id, visited, path):
                    return True

            path.remove(current)
            return False

        # Check for cycles
        visited: Set[str] = set()
        for node in graph.nodes:
            if node.node_id not in visited:
                if has_cycle_from(node.node_id, node.node_id, visited, set()):
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.CIRCULAR,
                        severity=0.7,
                        description=f"Circular dependency detected involving {node.node_id}",
                        affected_nodes=[node.node_id],
                        recommendations=["Circular dependencies may indicate feedback loops - verify if intentional"]
                    ))

        return anomalies

    def _detect_dead_features(self, graph: Any) -> List[Anomaly]:
        """Detect features that never activate."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph, NodeType

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []
        dead_features = []

        for node in graph.nodes:
            if node.node_type == NodeType.SAE_FEATURE:
                if node.activation < self.dead_feature_threshold:
                    dead_features.append(node.node_id)

        if dead_features:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.DEAD_FEATURES,
                severity=len(dead_features) / max(graph.n_nodes, 1),
                description=f"Found {len(dead_features)} dead features (activation < {self.dead_feature_threshold})",
                affected_nodes=dead_features,
                recommendations=["Dead features may indicate over-sparse SAE or irrelevant features"]
            ))

        return anomalies

    def _detect_saturated(self, graph: Any) -> List[Anomaly]:
        """Detect always-on features."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph, NodeType

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []
        saturated = []

        for node in graph.nodes:
            if node.node_type == NodeType.SAE_FEATURE:
                if node.activation > self.saturation_threshold:
                    saturated.append(node.node_id)

        if saturated:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.SATURATED,
                severity=len(saturated) / max(graph.n_nodes, 1),
                description=f"Found {len(saturated)} saturated features (activation > {self.saturation_threshold})",
                affected_nodes=saturated,
                recommendations=["Saturated features may be polysemantic or represent universal concepts"]
            ))

        return anomalies

    def _detect_hidden_paths(self, graph: Any) -> List[Anomaly]:
        """Detect potential hidden information paths (safety-relevant)."""
        from hololoom.dark_trace.circuits.graph import CircuitGraph, EdgeType

        if not isinstance(graph, CircuitGraph):
            return []

        anomalies = []

        # Look for paths that bypass expected route
        # This is a simplified check - in practice would need more sophisticated analysis

        # Find edges with very low weight that still carry information
        hidden_edges = []
        for edge in graph.edges:
            if edge.weight < 0.1 and edge.mutual_information > 0.5:
                hidden_edges.append((edge.source_id, edge.target_id))

        if hidden_edges:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.HIDDEN_PATH,
                severity=0.6,
                description=f"Found {len(hidden_edges)} potential hidden information paths",
                affected_edges=hidden_edges,
                evidence={"low_weight_high_mi": len(hidden_edges)},
                recommendations=[
                    "Hidden paths may indicate information flow not captured by main circuit",
                    "Investigate if these paths could be exploited to bypass safety measures"
                ]
            ))

        return anomalies
