"""
Truth Maintenance System — Justification tracking and belief retraction.

Classical AI TMS adapted for graph-native operation:
  - Every PPR-retrieved node gets a justification record (WHY it was retrieved)
  - Justifications are stored as JUSTIFIED_BY edges in the graph
  - When contradictions are detected, the weaker belief is retracted
  - Retraction propagates via ACT-R activation (self-correcting)

Implements the KnowledgeSource protocol at two phases:
  POST_NAVIGATE:   Record justifications (trace seed → node paths)
  POST_CONFIDENCE: Propagate retractions when contradictions exist

Graph-native: justifications ARE edges, retractions ARE activation
decreases, explanations ARE path traversals. No parallel data structures.

No external dependencies.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .confidence import find_contradicting_pairs
from .knowledge_source import Phase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TMSConfig:
    """Configuration for the Truth Maintenance System."""
    max_trace_depth: int = 3          # Max hops for seed-to-node tracing
    retraction_penalty: float = 0.3   # Activation decrease for retracted nodes
    min_justification_strength: float = 0.1  # Below this = weakly justified
    max_justifications_per_node: int = 5     # Limit justification records per node
    store_edges: bool = True          # Store JUSTIFIED_BY edges in graph


@dataclass
class JustificationRecord:
    """Why a belief (retrieved node) is held.

    Traces the path from a seed node to the retrieved belief
    through the graph. Strength is derived from seed weight
    and path length (shorter paths = stronger justification).
    """
    belief_id: str
    seed_id: str
    path: List[str]       # Node IDs from seed to belief
    ppr_score: float
    seed_weight: float
    query_id: str

    @property
    def strength(self) -> float:
        """Justification strength: seed_weight / path_length."""
        return self.seed_weight / max(1, len(self.path))


@dataclass
class Retraction:
    """Record of a retracted belief with explanation."""
    retracted_id: str
    retained_id: str
    retracted_strength: float
    retained_strength: float
    reason: str
    explanation: str   # Human-readable justification chain


# =============================================================================
# Truth Maintenance System
# =============================================================================

class TruthMaintenanceSystem:
    """Belief justification tracking and retraction propagation.

    Implements KnowledgeSource protocol.

    POST_NAVIGATE: For each PPR-retrieved node, trace which seeds can
    reach it via graph edges. Record the path as a JustificationRecord
    and optionally store JUSTIFIED_BY edges. Post all justifications
    to blackboard.flags["justifications"].

    POST_CONFIDENCE: When contradictions are detected (from confidence
    estimator), find contradicting node pairs. Compare justification
    strength. Retract the weaker belief by decreasing its activation
    (ACT-R composition). Post retraction info to blackboard for FLAG.

    Usage:
        tms = TruthMaintenanceSystem(activation_adapter=adapter)
        registry.register(tms)
        # TMS automatically contributes at POST_NAVIGATE and POST_CONFIDENCE
    """

    def __init__(
        self,
        activation_adapter=None,
        config: Optional[TMSConfig] = None,
    ):
        self._activation = activation_adapter  # Optional ACT-R integration
        self._config = config or TMSConfig()
        # Query-scoped justification index (rebuilt each query)
        self._justifications: Dict[str, List[JustificationRecord]] = {}

    @property
    def name(self) -> str:
        return "tms"

    @property
    def phases(self) -> tuple:
        return (Phase.POST_NAVIGATE, Phase.POST_CONFIDENCE)

    def activation_level(self, blackboard: Any) -> float:
        """Always active — justification recording is lightweight."""
        return 1.0

    def contribute(self, blackboard: Any, graph: Any, context: Dict[str, Any]) -> None:
        """Route to the appropriate phase handler."""
        # POST_NAVIGATE: has nav_result but no confidence
        if "nav_result" in context and "confidence" not in context:
            self._record_justifications(blackboard, graph, context["nav_result"])
        # POST_CONFIDENCE: has both confidence and nav_result
        elif "confidence" in context and "nav_result" in context:
            self._check_retractions(blackboard, graph, context)

    # =========================================================================
    # Phase 1: Record Justifications (POST_NAVIGATE)
    # =========================================================================

    def _record_justifications(self, blackboard, graph, nav_result) -> None:
        """Record WHY each retrieved node was retrieved.

        For each node in nav_result.ranked_nodes:
            1. Find which seed(s) can reach it via graph edges (BFS)
            2. Record the path as a JustificationRecord
            3. Optionally store JUSTIFIED_BY edge in graph
            4. Post to blackboard.flags["justifications"]
        """
        seed_ids = {s.node_id: s for s in nav_result.seed_nodes}
        justifications: Dict[str, List[JustificationRecord]] = {}

        query_id = ""
        if blackboard is not None:
            query_id = getattr(blackboard, "query_id", "")

        for node_id, ppr_score in nav_result.ranked_nodes:
            records = []

            # If node IS a seed, it's self-justified
            if node_id in seed_ids:
                seed = seed_ids[node_id]
                records.append(JustificationRecord(
                    belief_id=node_id,
                    seed_id=node_id,
                    path=[node_id],
                    ppr_score=ppr_score,
                    seed_weight=seed.weight,
                    query_id=query_id,
                ))
            else:
                # Trace which seeds can reach this node
                for seed in nav_result.seed_nodes:
                    path = self._find_path(
                        graph, seed.node_id, node_id,
                        max_depth=self._config.max_trace_depth,
                    )
                    if path:
                        records.append(JustificationRecord(
                            belief_id=node_id,
                            seed_id=seed.node_id,
                            path=path,
                            ppr_score=ppr_score,
                            seed_weight=seed.weight,
                            query_id=query_id,
                        ))
                        if len(records) >= self._config.max_justifications_per_node:
                            break

            if records:
                justifications[node_id] = records
                # Store JUSTIFIED_BY edges in graph
                if self._config.store_edges:
                    for rec in records:
                        self._store_justification_edge(graph, rec)

        self._justifications = justifications

        if blackboard is not None:
            blackboard.post_flag("justifications", justifications)

        logger.debug(
            "TMS: recorded justifications for %d/%d retrieved nodes",
            len(justifications), len(nav_result.ranked_nodes),
        )

    def _find_path(
        self, graph, source: str, target: str, max_depth: int,
    ) -> Optional[List[str]]:
        """BFS from source to target, return path if found within max_depth."""
        if source == target:
            return [source]

        if not graph.has_node(source) or not graph.has_node(target):
            return None

        visited: Set[str] = {source}
        queue: deque = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                break

            try:
                neighbors = graph.neighbors(current)
            except Exception:
                continue

            for neighbor_id, _, _ in neighbors:
                if neighbor_id == target:
                    return path + [neighbor_id]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def _store_justification_edge(self, graph, record: JustificationRecord) -> None:
        """Store a JUSTIFIED_BY edge in the graph.

        belief_node → seed_node via JUSTIFIED_BY
        """
        bus = getattr(graph, '_bus', None)
        if bus is None or not hasattr(bus, '_edges'):
            return

        # Only store if both nodes exist
        if not graph.has_node(record.belief_id) or not graph.has_node(record.seed_id):
            return

        edges = bus._edges.get(record.belief_id, [])
        # Avoid duplicate edges
        for target, rel in edges:
            if target == record.seed_id and rel == "JUSTIFIED_BY":
                return

        if record.belief_id not in bus._edges:
            bus._edges[record.belief_id] = []
        bus._edges[record.belief_id].append((record.seed_id, "JUSTIFIED_BY"))

    # =========================================================================
    # Phase 2: Check Retractions (POST_CONFIDENCE)
    # =========================================================================

    def _check_retractions(self, blackboard, graph, context) -> None:
        """When contradictions detected, retract the weakest belief.

        1. Read contradiction_count from confidence result
        2. Find contradicting node pairs (reuse negation detection logic)
        3. Compare justification strength for each pair
        4. Retract weaker belief:
           a. Decrease its activation (ACT-R composition)
           b. Post retraction info to blackboard.flags["retractions"]
        """
        confidence = context["confidence"]
        nav_result = context["nav_result"]

        # Only act when contradictions exist
        if confidence.narrative.contradiction_count == 0:
            return

        # Find contradicting pairs (graph-structural entity anchoring)
        contradicting = self._find_contradicting_pairs(nav_result, graph)
        if not contradicting:
            return

        retractions: List[Retraction] = []
        for node_a, node_b in contradicting:
            strength_a = self._node_strength(node_a)
            strength_b = self._node_strength(node_b)

            # Retract the weaker one
            if strength_a < strength_b:
                weaker, stronger = node_a, node_b
                w_str, s_str = strength_a, strength_b
            else:
                weaker, stronger = node_b, node_a
                w_str, s_str = strength_b, strength_a

            # Decrease activation of retracted node (ACT-R composition)
            if self._activation is not None:
                current = self._activation.levels.get(weaker, 0.0)
                self._activation.levels[weaker] = max(
                    0.0, current - self._config.retraction_penalty
                )

            retractions.append(Retraction(
                retracted_id=weaker,
                retained_id=stronger,
                retracted_strength=w_str,
                retained_strength=s_str,
                reason=(
                    f"Weaker justification "
                    f"({w_str:.2f} vs {s_str:.2f})"
                ),
                explanation=self._explain(weaker),
            ))

        if retractions and blackboard is not None:
            blackboard.post_flag("retractions", [
                {
                    "retracted": r.retracted_id,
                    "retained": r.retained_id,
                    "reason": r.reason,
                    "explanation": r.explanation,
                }
                for r in retractions
            ])
            blackboard.post_flag("tms_active", True)

        logger.info(
            "TMS: %d retractions from %d contradictions",
            len(retractions), len(contradicting),
        )

    def _find_contradicting_pairs(
        self, nav_result, graph=None,
    ) -> List[Tuple[str, str]]:
        """Find contradicting node pairs via entity-anchored semantic analysis.

        Delegates to the shared find_contradicting_pairs() function which
        checks: shared entity neighbor + negation asymmetry + Jaccard overlap.
        """
        return find_contradicting_pairs(
            nav_result.ranked_nodes,
            nav_result.node_data,
            graph=graph,
        )

    def _node_strength(self, node_id: str) -> float:
        """Total justification strength for a node."""
        records = self._justifications.get(node_id, [])
        if not records:
            return 0.0
        return sum(r.strength for r in records)

    # =========================================================================
    # Explanation
    # =========================================================================

    def _explain(self, node_id: str) -> str:
        """Build human-readable explanation of why a node was retrieved."""
        records = self._justifications.get(node_id, [])
        if not records:
            return "No justification recorded."
        chains = []
        for r in records:
            path_str = " → ".join(r.path)
            chains.append(
                f"Seed '{r.seed_id}' (weight={r.seed_weight:.2f}) "
                f"via [{path_str}]"
            )
        return "; ".join(chains)

    def explain_node(self, node_id: str) -> Dict[str, Any]:
        """Full justification report for a node (for debugging/UI)."""
        records = self._justifications.get(node_id, [])
        return {
            "node_id": node_id,
            "justified": len(records) > 0,
            "total_strength": self._node_strength(node_id),
            "justifications": [
                {
                    "seed_id": r.seed_id,
                    "path": r.path,
                    "strength": r.strength,
                    "ppr_score": r.ppr_score,
                }
                for r in records
            ],
            "explanation": self._explain(node_id),
        }

    def report(self) -> Dict[str, Any]:
        """Diagnostic info about current TMS state."""
        total_justified = sum(
            1 for records in self._justifications.values() if records
        )
        total_records = sum(
            len(records) for records in self._justifications.values()
        )
        return {
            "nodes_justified": total_justified,
            "total_justification_records": total_records,
            "config": {
                "max_trace_depth": self._config.max_trace_depth,
                "retraction_penalty": self._config.retraction_penalty,
                "store_edges": self._config.store_edges,
            },
        }
