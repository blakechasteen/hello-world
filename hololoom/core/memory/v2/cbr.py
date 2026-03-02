"""
Case-Based Reasoning — Reuse past successful query experiences.

Classical AI CBR adapted for graph-native operation:
  - After each query (POST_LOOP), store the (query, features, reward, response)
    triple as a "case" node in the graph with SIMILAR_CASE edges
  - Before each query (POST_NAVIGATE), find similar past cases by feature
    distance and inject their solutions as retrieval hints

The CBR cycle: Retrieve → Reuse → Revise → Retain
  - Retrieve: find cases with similar context_features (Euclidean distance)
  - Reuse:    post case solutions to blackboard for context enrichment
  - Revise:   reward feedback updates case quality scores
  - Retain:   successful queries become new cases automatically

Implements the KnowledgeSource protocol at two phases:
  POST_NAVIGATE: Retrieve similar cases, post hints to blackboard
  POST_LOOP:     Retain successful query as new case

Graph-native: cases are stored as graph nodes with SIMILAR_CASE edges.

No external dependencies.
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_source import Phase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CBRConfig:
    """Configuration for Case-Based Reasoning."""
    max_cases: int = 200                # Hard cap on stored cases
    max_retrieve: int = 3               # Top-K cases to retrieve
    min_similarity: float = 0.7         # Minimum similarity to retrieve
    min_reward_to_retain: float = 0.5   # Only retain successful cases
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "query_length": 1.0,
        "seed_count": 1.5,              # Seed structure matters most
        "ppr_entropy": 1.2,
        "entity_ratio": 1.0,
        "wm_overlap": 0.3,             # Low weight — state-dependent
        "confidence": 0.5,             # Medium weight — varies
    })
    store_in_graph: bool = True


@dataclass
class Case:
    """A stored (problem, solution, outcome) triple."""
    case_id: str
    query: str
    features: Dict[str, float]
    preset_id: str
    reward: float
    response_summary: str           # First 200 chars of response
    shell_count: int
    node_id: Optional[str] = None   # Graph node ID (if stored)
    retrieval_count: int = 0        # How often this case was retrieved
    avg_reuse_reward: float = 0.0   # Average reward when this case was reused

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query[:80],
            "features": self.features,
            "preset_id": self.preset_id,
            "reward": self.reward,
            "shell_count": self.shell_count,
            "retrieval_count": self.retrieval_count,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        """Full serialization for persistence."""
        return {
            "case_id": self.case_id,
            "query": self.query,
            "features": self.features,
            "preset_id": self.preset_id,
            "reward": self.reward,
            "response_summary": self.response_summary,
            "shell_count": self.shell_count,
            "node_id": self.node_id,
            "retrieval_count": self.retrieval_count,
            "avg_reuse_reward": self.avg_reuse_reward,
        }

    @classmethod
    def from_snapshot(cls, d: Dict[str, Any]) -> "Case":
        """Reconstruct Case from snapshot dict."""
        return cls(
            case_id=d["case_id"],
            query=d["query"],
            features=d.get("features", {}),
            preset_id=d.get("preset_id", "balanced"),
            reward=d.get("reward", 0.0),
            response_summary=d.get("response_summary", ""),
            shell_count=d.get("shell_count", 1),
            node_id=d.get("node_id"),
            retrieval_count=d.get("retrieval_count", 0),
            avg_reuse_reward=d.get("avg_reuse_reward", 0.0),
        )


# =============================================================================
# Case-Based Reasoning System
# =============================================================================

class CBRSystem:
    """Case-Based Reasoning via feature-distance retrieval.

    Implements KnowledgeSource protocol.

    POST_NAVIGATE: After navigation, compute feature distance between
    the current query's context_features and stored cases. Post the
    top-K most similar case solutions to blackboard.flags["cbr_hints"]
    for the formatter/LLM to use as additional context.

    POST_LOOP: After the loop completes with sufficient reward, store
    the query as a new case with SIMILAR_CASE edges to existing
    similar cases.

    Usage:
        cbr = CBRSystem()
        registry.register(cbr)
        # CBR automatically retrieves and retains cases
    """

    def __init__(self, config: Optional[CBRConfig] = None):
        self._config = config or CBRConfig()
        self._cases: List[Case] = []

    @property
    def name(self) -> str:
        return "cbr"

    @property
    def phases(self) -> tuple:
        return (Phase.POST_NAVIGATE, Phase.POST_LOOP)

    def activation_level(self, blackboard: Any) -> float:
        """Higher activation when cases exist to retrieve from."""
        if self._cases:
            return 0.7
        return 0.1  # Still active to start retaining

    def contribute(self, blackboard: Any, graph: Any, context: Dict[str, Any]) -> None:
        """Route to appropriate phase handler."""
        if "nav_result" in context and "confidence" not in context:
            # POST_NAVIGATE: retrieve similar cases
            self._retrieve_cases(blackboard, context)
        elif "loop_result" in context:
            # POST_LOOP: retain successful query as case
            self._retain_case(blackboard, graph, context)

    # =========================================================================
    # POST_NAVIGATE: Retrieve Similar Cases
    # =========================================================================

    def _retrieve_cases(self, blackboard, context) -> None:
        """Find similar past cases and post hints to blackboard."""
        if blackboard is None or not self._cases:
            return

        features = blackboard.context_features()
        scored: List[Tuple[float, Case]] = []

        for case in self._cases:
            sim = self._similarity(features, case.features)
            if sim >= self._config.min_similarity:
                scored.append((sim, case))

        scored.sort(key=lambda x: (-x[0], -x[1].reward))
        top_cases = scored[:self._config.max_retrieve]

        if top_cases:
            hints = []
            for sim, case in top_cases:
                case.retrieval_count += 1
                hints.append({
                    "case_id": case.case_id,
                    "similarity": round(sim, 3),
                    "query": case.query[:100],
                    "reward": case.reward,
                    "response_hint": case.response_summary,
                    "preset_id": case.preset_id,
                })

            blackboard.post_flag("cbr_hints", hints)
            logger.info(
                "CBR: retrieved %d similar cases (best sim=%.2f)",
                len(hints), top_cases[0][0],
            )

    def _similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Weighted Euclidean distance converted to similarity [0, 1]."""
        weights = self._config.feature_weights
        sq_sum = 0.0
        w_sum = 0.0

        for fname, w in weights.items():
            va = a.get(fname, 0.0)
            vb = b.get(fname, 0.0)
            sq_sum += w * (va - vb) ** 2
            w_sum += w

        if w_sum == 0:
            return 0.0

        distance = math.sqrt(sq_sum / w_sum)
        # Convert distance to similarity: 1.0 = identical, 0.0 = maximally different
        return max(0.0, 1.0 - distance)

    # =========================================================================
    # POST_LOOP: Retain Successful Cases
    # =========================================================================

    def _retain_case(self, blackboard, graph, context) -> None:
        """Store the completed query as a new case if successful."""
        if blackboard is None:
            return

        loop_result = context["loop_result"]
        reward = loop_result.final_confidence.combined
        if reward < self._config.min_reward_to_retain:
            return

        features = blackboard.context_features()

        # Determine preset used
        chunk_info = blackboard.flags.get("chunk_action")
        preset_used = "balanced"
        if chunk_info:
            preset_used = chunk_info.get("preset_id", "balanced")

        # Get response summary
        response = ""
        shells = getattr(loop_result, 'shells_executed', None) or getattr(loop_result, 'shell_results', [])
        if shells:
            response = getattr(shells[0], 'response', '')[:200]

        case = Case(
            case_id=uuid.uuid4().hex[:10],
            query=blackboard.query,
            features=features,
            preset_id=preset_used,
            reward=reward,
            response_summary=response,
            shell_count=loop_result.shell_count,
        )

        # Update reuse stats for retrieved cases
        cbr_hints = blackboard.flags.get("cbr_hints", [])
        if cbr_hints:
            for hint in cbr_hints:
                for existing in self._cases:
                    if existing.case_id == hint["case_id"]:
                        # Running average of reuse reward
                        n = existing.retrieval_count
                        existing.avg_reuse_reward = (
                            (existing.avg_reuse_reward * (n - 1) + reward) / n
                            if n > 0 else reward
                        )

        self._cases.append(case)

        # Store in graph
        if self._config.store_in_graph:
            self._store_case_node(graph, case)

        # Trim to max
        if len(self._cases) > self._config.max_cases:
            # Remove lowest-reward cases that are rarely retrieved
            self._cases.sort(key=lambda c: c.reward + 0.1 * c.retrieval_count)
            self._cases = self._cases[-self._config.max_cases:]

        logger.info(
            "CBR: retained case '%s' (reward=%.2f, total cases=%d)",
            case.case_id, reward, len(self._cases),
        )

    def _store_case_node(self, graph, case: Case) -> None:
        """Store case as graph node with SIMILAR_CASE edges to neighbors."""
        bus = getattr(graph, '_bus', None)
        if bus is None or not hasattr(bus, '_items'):
            return

        try:
            from hololoom.memory.lite_bus import StoredItem
        except ImportError:
            return

        case_node_id = f"case_{case.case_id}"
        item = StoredItem(
            id=case_node_id,
            content=f"[case] query: {case.query[:100]} | reward: {case.reward:.2f}",
            memory_type="case",
            importance=min(0.8, case.reward),
            timestamp=datetime.utcnow().isoformat(),
        )

        bus._items[case_node_id] = item
        case.node_id = case_node_id

        # Add SIMILAR_CASE edges to nearby cases
        if case_node_id not in bus._edges:
            bus._edges[case_node_id] = []
        for other in self._cases[-10:]:  # Recent cases
            if other.case_id != case.case_id and other.node_id:
                sim = self._similarity(case.features, other.features)
                if sim >= self._config.min_similarity:
                    bus._edges[case_node_id].append(
                        (other.node_id, "SIMILAR_CASE")
                    )

    # =========================================================================
    # Public API
    # =========================================================================

    def to_snapshot(self) -> Dict[str, Any]:
        """Serialize case library for persistence."""
        return {
            "cases": [c.to_snapshot() for c in self._cases],
        }

    def from_snapshot(self, data: Dict[str, Any]) -> None:
        """Restore case library from snapshot dict."""
        self._cases.clear()
        for case_dict in data.get("cases", []):
            self._cases.append(Case.from_snapshot(case_dict))
        logger.info("CBR: loaded %d cases from snapshot", len(self._cases))

    @property
    def cases(self) -> List[Case]:
        return list(self._cases)

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def report(self) -> Dict[str, Any]:
        """Diagnostic info about CBR state."""
        return {
            "total_cases": len(self._cases),
            "avg_reward": (
                sum(c.reward for c in self._cases) / max(1, len(self._cases))
            ),
            "total_retrievals": sum(c.retrieval_count for c in self._cases),
            "cases": [c.to_dict() for c in self._cases[-10:]],
        }
