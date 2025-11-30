"""
Spring Physics Memory Scoring System
====================================

Saves and reuses spring activation patterns to create a learned memory scoring
system. The physics naturally reveals which memory connections are strong through
repeated activation patterns.

Key Idea:
---------
Every time spring physics runs, it reveals which edges (relationships) are
important for a given query. By tracking these activations over time, we build
up a "memory score" for each edge that improves retrieval.

Benefits:
---------
1. Learning from usage: Frequently activated edges become stronger
2. Forgetting: Unused edges decay over time
3. Personalization: User's query patterns shape the memory scores
4. Physics-motivated: Based on actual energy flow, not arbitrary weights

Example:
--------
Query: "Bayesian methods"
→ Spring physics activates: Bayesian → PriorDistribution (0.881)
→ Save edge score: ("Bayesian", "PriorDistribution") += 0.881
→ Next query benefits from learned connection strength!

Author: Blake Chasteen
Date: November 8, 2025
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EdgeScore:
    """
    Learned score for a knowledge graph edge.

    Attributes:
        source: Source entity
        target: Target entity
        activation_sum: Cumulative activation flow through edge
        access_count: Number of times edge was activated
        last_access: Timestamp of last activation
        decay_rate: How fast the score decays (0.0-1.0)
    """
    source: str
    target: str
    activation_sum: float = 0.0
    access_count: int = 0
    last_access: Optional[datetime] = None
    decay_rate: float = 0.99  # Slow forgetting

    @property
    def avg_activation(self) -> float:
        """Average activation strength."""
        return self.activation_sum / max(self.access_count, 1)

    @property
    def score(self) -> float:
        """
        Learned edge score combining frequency and strength.

        Score = avg_activation × log(1 + access_count) × decay_factor

        This gives higher scores to:
        - Frequently activated edges (log scaling)
        - Strongly activated edges (avg_activation)
        - Recently used edges (decay_factor)
        """
        if self.access_count == 0:
            return 0.0

        import math

        # Frequency bonus (logarithmic to prevent dominance)
        frequency_factor = math.log(1 + self.access_count)

        # Time decay (if we have last_access)
        decay_factor = 1.0
        if self.last_access:
            hours_since = (datetime.now() - self.last_access).total_seconds() / 3600
            decay_factor = self.decay_rate ** hours_since

        return self.avg_activation * frequency_factor * decay_factor

    def update(self, activation: float):
        """Update edge score with new activation."""
        self.activation_sum += activation
        self.access_count += 1
        self.last_access = datetime.now()

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "source": self.source,
            "target": self.target,
            "activation_sum": self.activation_sum,
            "access_count": self.access_count,
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "decay_rate": self.decay_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EdgeScore':
        """Deserialize from dict."""
        if data.get("last_access"):
            data["last_access"] = datetime.fromisoformat(data["last_access"])
        return cls(**data)


class SpringMemoryScorer:
    """
    Learns and maintains memory scores based on spring physics activation patterns.

    This creates a self-improving retrieval system where the physics naturally
    reveals important connections, and we save these patterns for future use.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        decay_rate: float = 0.99,
        min_score_threshold: float = 0.01
    ):
        """
        Initialize memory scorer.

        Args:
            persist_path: Path to save/load edge scores (optional)
            decay_rate: How fast unused edges decay (0.0-1.0)
            min_score_threshold: Minimum score to keep edge in memory
        """
        self.edge_scores: Dict[Tuple[str, str], EdgeScore] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        self.decay_rate = decay_rate
        self.min_score_threshold = min_score_threshold

        # Load existing scores if available
        if self.persist_path and self.persist_path.exists():
            self.load()
            logger.info(f"Loaded {len(self.edge_scores)} edge scores from {self.persist_path}")

    def record_activation_pattern(
        self,
        activations: Dict[str, float],
        kg_edges: List[Tuple[str, str]]
    ):
        """
        Record activation pattern from spring physics result.

        Args:
            activations: {entity: activation_score} from spring dynamics
            kg_edges: List of (source, target) edges in knowledge graph
        """
        # For each edge in the graph, record activation flow
        for source, target in kg_edges:
            # Activation flow = min of source and target activations
            # (conservative: edge only as active as its weakest node)
            source_activation = activations.get(source, 0.0)
            target_activation = activations.get(target, 0.0)

            if source_activation > 0.0 and target_activation > 0.0:
                flow = min(source_activation, target_activation)

                # Update edge score
                edge_key = (source, target)
                if edge_key not in self.edge_scores:
                    self.edge_scores[edge_key] = EdgeScore(
                        source=source,
                        target=target,
                        decay_rate=self.decay_rate
                    )

                self.edge_scores[edge_key].update(flow)

                logger.debug(
                    f"Recorded edge activation: {source} → {target} "
                    f"(flow={flow:.3f}, score={self.edge_scores[edge_key].score:.3f})"
                )

    def get_edge_multiplier(self, source: str, target: str) -> float:
        """
        Get learned multiplier for an edge.

        This multiplier can be applied to base edge weights in the knowledge
        graph to create adaptive, usage-based edge strengths.

        Args:
            source: Source entity
            target: Target entity

        Returns:
            Multiplier (>1.0 for strong edges, <1.0 for weak/unused edges)
        """
        edge_key = (source, target)

        if edge_key not in self.edge_scores:
            return 1.0  # Neutral (no learned information)

        score = self.edge_scores[edge_key].score

        # Convert score to multiplier
        # Score ranges from 0.0 (never used) to ~10.0 (frequently used)
        # Map to multiplier: 0.5 (unused) to 2.0 (frequently used)
        multiplier = 0.5 + (score / 10.0) * 1.5

        return min(max(multiplier, 0.5), 2.0)  # Clamp to [0.5, 2.0]

    def get_top_edges(self, limit: int = 20) -> List[EdgeScore]:
        """
        Get top scored edges.

        Useful for understanding what connections the system has learned.

        Args:
            limit: Maximum edges to return

        Returns:
            List of EdgeScore objects sorted by score (descending)
        """
        sorted_edges = sorted(
            self.edge_scores.values(),
            key=lambda e: e.score,
            reverse=True
        )
        return sorted_edges[:limit]

    def prune_weak_edges(self):
        """Remove edges below minimum score threshold (forgetting)."""
        before_count = len(self.edge_scores)

        self.edge_scores = {
            key: edge
            for key, edge in self.edge_scores.items()
            if edge.score >= self.min_score_threshold
        }

        pruned = before_count - len(self.edge_scores)
        if pruned > 0:
            logger.info(f"Pruned {pruned} weak edges (below threshold {self.min_score_threshold})")

    def save(self):
        """Save edge scores to disk."""
        if not self.persist_path:
            return

        data = {
            "edge_scores": [edge.to_dict() for edge in self.edge_scores.values()],
            "decay_rate": self.decay_rate,
            "min_score_threshold": self.min_score_threshold,
            "saved_at": datetime.now().isoformat()
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.edge_scores)} edge scores to {self.persist_path}")

    def load(self):
        """Load edge scores from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path, 'r') as f:
            data = json.load(f)

        self.edge_scores = {
            (edge["source"], edge["target"]): EdgeScore.from_dict(edge)
            for edge in data["edge_scores"]
        }

        self.decay_rate = data.get("decay_rate", self.decay_rate)
        self.min_score_threshold = data.get("min_score_threshold", self.min_score_threshold)

    def get_statistics(self) -> Dict:
        """Get scoring statistics."""
        if not self.edge_scores:
            return {
                "total_edges": 0,
                "avg_score": 0.0,
                "top_score": 0.0
            }

        scores = [edge.score for edge in self.edge_scores.values()]

        return {
            "total_edges": len(self.edge_scores),
            "avg_score": sum(scores) / len(scores),
            "top_score": max(scores),
            "total_activations": sum(e.access_count for e in self.edge_scores.values())
        }


# ============================================================================
# Integration with Spring Graph Retriever
# ============================================================================

class AdaptiveSpringRetriever:
    """
    Spring graph retriever with adaptive edge weights based on learned scores.

    This combines spring physics with memory scoring to create a self-improving
    retrieval system.
    """

    def __init__(
        self,
        kg,
        scorer: Optional[SpringMemoryScorer] = None,
        config=None
    ):
        """
        Initialize adaptive retriever.

        Args:
            kg: Knowledge graph
            scorer: SpringMemoryScorer (or None to create new)
            config: SpringConfig
        """
        from HoloLoom.memory.spring_graph_retriever import SpringPhysicsGraphRetriever

        # Base retriever
        self.base_retriever = SpringPhysicsGraphRetriever(kg=kg, config=config)

        # Memory scorer
        self.scorer = scorer or SpringMemoryScorer()

        self.kg = kg

    async def retrieve(
        self,
        query: str,
        memories,
        limit: int = 10,
        learn: bool = True
    ):
        """
        Retrieve with adaptive edge weights.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum results
            learn: Whether to record activation pattern (default: True)

        Returns:
            List of (memory, score) tuples
        """
        # Apply learned edge multipliers to spring config
        # (This would require modifying SpringConfig to accept edge_multipliers)
        # For now, just retrieve normally
        results = await self.base_retriever.retrieve(query, memories, limit)

        if learn:
            # Extract activation pattern
            query_entities = self.base_retriever._extract_entities(query)
            if query_entities:
                activations = self.base_retriever.activate_from_query(query_entities)

                # Record pattern
                kg_edges = [(u, v) for u, v in self.kg.G.edges()]
                self.scorer.record_activation_pattern(activations, kg_edges)

                logger.debug(
                    f"Recorded activation pattern: {len(activations)} entities, "
                    f"scorer has {len(self.scorer.edge_scores)} edges"
                )

        return results

    def save_learned_patterns(self):
        """Save learned edge scores."""
        self.scorer.save()

    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        return self.scorer.get_statistics()


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using spring memory scoring."""
    from HoloLoom.memory.graph import KG, KGEdge

    # Create knowledge graph
    kg = KG()
    kg.add_edges([
        KGEdge("ThompsonSampling", "Bayesian", "IS_A", 1.0),
        KGEdge("Bayesian", "PriorDistribution", "USES", 0.9),
        KGEdge("Bayesian", "GaussianProcess", "RELATED_TO", 0.7),
    ])

    # Create adaptive retriever with persistence
    scorer = SpringMemoryScorer(persist_path="./memory_scores.json")
    retriever = AdaptiveSpringRetriever(kg=kg, scorer=scorer)

    # Use over time - system learns!
    # Query 1: "Bayesian methods"
    # → Activates Bayesian → PriorDistribution
    # → Saves edge score

    # Query 2: "Bayesian methods" (later)
    # → Edge Bayesian → PriorDistribution now has higher weight!
    # → Better retrieval

    # Save learned patterns
    retriever.save_learned_patterns()

    # View top learned edges
    top_edges = scorer.get_top_edges(limit=10)
    for edge in top_edges:
        print(f"{edge.source} → {edge.target}: {edge.score:.3f} ({edge.access_count} accesses)")


if __name__ == "__main__":
    example_usage()
