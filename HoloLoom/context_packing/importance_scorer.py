"""
Multi-Signal Importance Scoring
================================

Combines 6 importance signals to rank memory nodes for context packing.

Signals:
1. Recency - How recently was this accessed?
2. Relevance - How semantically similar to query?
3. Centrality - How central in knowledge graph?
4. Access Frequency - How often accessed historically?
5. Confidence - What was the historical confidence?
6. Heat - Hot pattern feedback score

Author: Claude Code
Date: 2025-11-22
"""

import math
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .protocol import ImportanceSignal, ActivationState, ImportanceMap
from .config import ImportanceScorerConfig

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Multi-signal importance scoring engine.

    Combines 6 different signals to compute comprehensive importance scores
    for memory nodes in context packing.
    """

    def __init__(
        self,
        config: Optional[ImportanceScorerConfig] = None,
        embedder: Optional[Any] = None
    ):
        """
        Initialize importance scorer.

        Args:
            config: Scorer configuration (uses defaults if None)
            embedder: Optional embedding model for relevance scoring
        """
        self.config = config or ImportanceScorerConfig()
        self.config.validate()

        self.embedder = embedder

        # Caches for expensive computations
        self._centrality_cache: Optional[Dict[str, float]] = None
        self._centrality_cache_time: float = 0.0
        self._centrality_cache_ttl: float = 300.0  # 5 minutes

        # Stats
        self._stats = {
            'total_scored': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def score_importance(
        self,
        node_id: str,
        query: str,
        graph: Any,
        activation_state: Optional[ActivationState] = None,
        node_content: Optional[str] = None
    ) -> Dict[ImportanceSignal, float]:
        """
        Compute multi-signal importance scores for a node.

        Args:
            node_id: Node to score
            query: Current query text
            graph: Knowledge graph
            activation_state: Optional current activation state
            node_content: Optional node text content (for relevance)

        Returns:
            Dict mapping ImportanceSignal -> score (0.0-1.0)
        """
        scores = {}

        # 1. Recency
        scores[ImportanceSignal.RECENCY] = self._score_recency(
            node_id, activation_state
        )

        # 2. Relevance (requires embedder)
        scores[ImportanceSignal.RELEVANCE] = self._score_relevance(
            node_id, query, node_content
        )

        # 3. Centrality
        scores[ImportanceSignal.CENTRALITY] = self._score_centrality(
            node_id, graph
        )

        # 4. Access Frequency
        scores[ImportanceSignal.ACCESS_FREQUENCY] = self._score_frequency(
            node_id, activation_state
        )

        # 5. Confidence
        scores[ImportanceSignal.CONFIDENCE] = self._score_confidence(
            node_id, activation_state
        )

        # 6. Heat
        scores[ImportanceSignal.HEAT] = self._score_heat(
            node_id, activation_state
        )

        self._stats['total_scored'] += 1

        return scores

    def aggregate_scores(
        self,
        scores: Dict[ImportanceSignal, float],
        weights: Optional[Dict[ImportanceSignal, float]] = None
    ) -> float:
        """
        Aggregate multi-signal scores into single importance value.

        Uses weighted average with configurable weights.

        Args:
            scores: Dict of signal -> score
            weights: Optional custom weights (uses config if None)

        Returns:
            Aggregated importance score (0.0-1.0)
        """
        weights = weights or self.config.weights

        total_score = 0.0
        total_weight = 0.0

        for signal, score in scores.items():
            if signal in weights:
                total_score += score * weights[signal]
                total_weight += weights[signal]

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def score_batch(
        self,
        node_ids: List[str],
        query: str,
        graph: Any,
        activation_states: Optional[Dict[str, ActivationState]] = None,
        node_contents: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Score importance for batch of nodes.

        More efficient than scoring one-by-one for large batches.

        Args:
            node_ids: List of nodes to score
            query: Current query
            graph: Knowledge graph
            activation_states: Optional dict of node_id -> state
            node_contents: Optional dict of node_id -> content

        Returns:
            Dict mapping node_id -> aggregated_importance
        """
        activation_states = activation_states or {}
        node_contents = node_contents or {}

        importance_scores = {}

        for node_id in node_ids:
            scores = self.score_importance(
                node_id=node_id,
                query=query,
                graph=graph,
                activation_state=activation_states.get(node_id),
                node_content=node_contents.get(node_id)
            )

            importance_scores[node_id] = self.aggregate_scores(scores)

        return importance_scores

    # Individual signal scoring methods

    def _score_recency(
        self,
        node_id: str,
        activation_state: Optional[ActivationState]
    ) -> float:
        """
        Score based on recency of access.

        Uses exponential decay: score = 2^(-t / half_life)
        """
        if activation_state is None or activation_state.last_accessed == 0:
            return 0.5  # Neutral score if no history

        now = time.time()
        time_since_access = now - activation_state.last_accessed

        # Exponential decay
        half_life = self.config.recency_half_life
        decay_factor = math.exp(-time_since_access * math.log(2) / half_life)

        return max(0.0, min(1.0, decay_factor))

    def _score_relevance(
        self,
        node_id: str,
        query: str,
        node_content: Optional[str]
    ) -> float:
        """
        Score based on semantic relevance to query.

        Requires embedder to be configured.
        """
        if self.embedder is None or node_content is None:
            return 0.5  # Neutral score if no embedder

        try:
            # Embed query and node content
            query_emb = self._embed_text(query)
            node_emb = self._embed_text(node_content)

            # Cosine similarity
            similarity = self._cosine_similarity(query_emb, node_emb)

            # Normalize to 0-1 range (cosine is -1 to 1)
            return (similarity + 1.0) / 2.0

        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            return 0.5

    def _score_centrality(
        self,
        node_id: str,
        graph: Any
    ) -> float:
        """
        Score based on graph centrality (PageRank-style).

        More central nodes are considered more important.
        """
        # Check cache first
        if self._centrality_cache is not None:
            cache_age = time.time() - self._centrality_cache_time
            if cache_age < self._centrality_cache_ttl:
                self._stats['cache_hits'] += 1
                return self._centrality_cache.get(node_id, 0.0)

        self._stats['cache_misses'] += 1

        # Compute centrality for entire graph
        centrality_scores = self._compute_centrality(graph)

        # Update cache
        self._centrality_cache = centrality_scores
        self._centrality_cache_time = time.time()

        return centrality_scores.get(node_id, 0.0)

    def _score_frequency(
        self,
        node_id: str,
        activation_state: Optional[ActivationState]
    ) -> float:
        """
        Score based on access frequency.

        Logarithmic scaling to prevent extremely popular nodes from dominating.
        """
        if activation_state is None:
            return 0.0

        access_count = activation_state.access_count

        if access_count == 0:
            return 0.0

        # Logarithmic scaling: log10(1 + count) / log10(1001)
        # Max score at ~1000 accesses
        max_accesses = 1000.0
        score = math.log10(1 + access_count) / math.log10(1 + max_accesses)

        return max(0.0, min(1.0, score))

    def _score_confidence(
        self,
        node_id: str,
        activation_state: Optional[ActivationState]
    ) -> float:
        """
        Score based on historical confidence.

        Uses average confidence from importance_scores if available.
        """
        if activation_state is None:
            return 0.5  # Neutral

        # Check if confidence signal exists in importance_scores
        if activation_state.importance_scores:
            conf = activation_state.importance_scores.get(ImportanceSignal.CONFIDENCE)
            if conf is not None:
                return max(0.0, min(1.0, conf))

        # Fallback: use activation level as proxy
        return max(0.0, min(1.0, activation_state.activation))

    def _score_heat(
        self,
        node_id: str,
        activation_state: Optional[ActivationState]
    ) -> float:
        """
        Score based on hot pattern heat.

        From recursive/hot_pattern_feedback.py system.
        """
        if activation_state is None:
            return 0.0

        heat = activation_state.heat

        # Normalize heat (typically 0-10 range)
        max_heat = 10.0
        normalized = heat / max_heat

        return max(0.0, min(1.0, normalized))

    # Helper methods

    def _compute_centrality(self, graph: Any) -> Dict[str, float]:
        """Compute centrality scores for all nodes in graph."""
        algorithm = self.config.centrality_algorithm

        # NetworkX graphs
        if NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
            if algorithm == "pagerank":
                scores = nx.pagerank(graph, alpha=0.85)
            elif algorithm == "betweenness":
                scores = nx.betweenness_centrality(graph)
            elif algorithm == "closeness":
                scores = nx.closeness_centrality(graph)
            else:
                scores = nx.pagerank(graph)  # Default

            # Normalize to 0-1
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {k: v / max_score for k, v in scores.items()}

            return scores

        # HoloLoom KG
        if hasattr(graph, 'compute_pagerank'):
            try:
                scores = graph.compute_pagerank()
                return scores
            except Exception as e:
                logger.warning(f"PageRank computation failed: {e}")

        # Fallback: degree centrality (simple)
        return self._compute_degree_centrality(graph)

    def _compute_degree_centrality(self, graph: Any) -> Dict[str, float]:
        """Fallback: simple degree-based centrality."""
        degree_scores = {}

        # Get all nodes
        if NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
            nodes = list(graph.nodes())
            max_degree = max(dict(graph.degree()).values()) if nodes else 1

            for node in nodes:
                degree = graph.degree(node)
                degree_scores[node] = degree / max_degree if max_degree > 0 else 0.0

        elif hasattr(graph, 'get_all_nodes'):
            nodes = graph.get_all_nodes()
            degrees = {node: len(graph.get_outgoing_edges(node)) +
                           len(graph.get_incoming_edges(node)) for node in nodes}
            max_degree = max(degrees.values()) if degrees else 1

            for node, degree in degrees.items():
                degree_scores[node] = degree / max_degree if max_degree > 0 else 0.0

        return degree_scores

    def _embed_text(self, text: str) -> Any:
        """Embed text using configured embedder."""
        if self.embedder is None:
            raise ValueError("Embedder not configured")

        # MatryoshkaEmbeddings (HoloLoom)
        if hasattr(self.embedder, 'embed'):
            return self.embedder.embed(text)

        # Sentence-transformers
        elif hasattr(self.embedder, 'encode'):
            return self.embedder.encode(text)

        # Fallback
        else:
            raise ValueError(f"Unknown embedder type: {type(self.embedder)}")

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity between vectors."""
        if NUMPY_AVAILABLE:
            # NumPy arrays
            if isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                return dot / (norm1 * norm2)

        # Python lists
        if isinstance(vec1, list) and isinstance(vec2, list):
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot / (norm1 * norm2)

        raise ValueError(f"Unsupported vector types: {type(vec1)}, {type(vec2)}")

    def invalidate_centrality_cache(self):
        """Invalidate centrality cache (call after graph changes)."""
        self._centrality_cache = None
        self._centrality_cache_time = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get scoring statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            'total_scored': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Convenience functions

def score_nodes(
    node_ids: List[str],
    query: str,
    graph: Any,
    config: Optional[ImportanceScorerConfig] = None
) -> ImportanceMap:
    """
    Quick importance scoring for list of nodes.

    Args:
        node_ids: Nodes to score
        query: Current query
        graph: Knowledge graph
        config: Optional config override

    Returns:
        Dict mapping node_id -> importance_score
    """
    scorer = ImportanceScorer(config)
    return scorer.score_batch(node_ids, query, graph)
