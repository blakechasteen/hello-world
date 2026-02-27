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

# Phase 5: Information Theory imports
try:
    from HoloLoom.warp.math.information import (
        entropy, mutual_information, DistributionPair, InformationMetrics
    )
    INFORMATION_THEORY_AVAILABLE = True
except ImportError:
    INFORMATION_THEORY_AVAILABLE = False
    entropy = None
    mutual_information = None
    DistributionPair = None
    InformationMetrics = None

from .protocol import ImportanceSignal, ActivationState, ImportanceMap
from .config import ImportanceScorerConfig
from collections import Counter
from typing import Tuple

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

        # Phase 5: MI caching for performance (<5ms overhead target)
        self._mi_cache: Dict[Tuple[str, str], float] = {}
        self._mi_cache_size: int = 1000  # Max cache entries
        self._entropy_cache: Dict[str, float] = {}  # Node entropy cache

        # Stats
        self._stats = {
            'total_scored': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'mi_cache_hits': 0,  # Phase 5
            'mi_cache_misses': 0  # Phase 5
        }

    def score_importance(
        self,
        node_id: str,
        query: str,
        graph: Any,
        activation_state: Optional[ActivationState] = None,
        node_content: Optional[str] = None,
        node_embedding: Optional[Any] = None,  # Phase 5
        query_embedding: Optional[Any] = None  # Phase 5
    ) -> Dict[ImportanceSignal, float]:
        """
        Compute multi-signal importance scores for a node.

        Args:
            node_id: Node to score
            query: Current query text
            graph: Knowledge graph
            activation_state: Optional current activation state
            node_content: Optional node text content (for relevance)
            node_embedding: Optional pre-computed node embedding (Phase 5)
            query_embedding: Optional pre-computed query embedding (Phase 5)

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

        # 7. Information Content (Phase 5 - Mutual Information)
        if self.config.enable_information_scoring:
            scores[ImportanceSignal.INFORMATION_CONTENT] = self._score_information_content(
                node_id, query, node_content, node_embedding, query_embedding
            )
        else:
            scores[ImportanceSignal.INFORMATION_CONTENT] = 0.5  # Neutral fallback

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

    # ==========================================================================
    # Phase 5: Information Theory Scoring (Tishby's Information Bottleneck)
    # ==========================================================================

    def _score_information_content(
        self,
        node_id: str,
        query: str,
        node_content: Optional[str],
        node_embedding: Optional[Any] = None,
        query_embedding: Optional[Any] = None
    ) -> float:
        """
        Score based on mutual information I(Node; Query).

        Implements Information Bottleneck principle: nodes that share
        more information with the query are more important for context.

        Uses two strategies:
        1. Embedding-based MI (if embeddings available) - more accurate
        2. Word overlap-based MI (fallback) - fast approximation

        Args:
            node_id: Node identifier
            query: Query text
            node_content: Node text content
            node_embedding: Pre-computed node embedding (optional)
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            Normalized MI score in [0, 1]
        """
        if not INFORMATION_THEORY_AVAILABLE or not NUMPY_AVAILABLE:
            return 0.5  # Neutral fallback

        if node_content is None or not node_content.strip():
            return 0.5

        # Check MI cache first
        query_hash = str(hash(query))
        cached_mi = self._get_cached_mi(node_id, query_hash)
        if cached_mi is not None:
            return cached_mi

        try:
            mi_score = 0.5  # Default neutral

            # Strategy 1: Embedding-based MI (if embeddings available)
            if node_embedding is not None and query_embedding is not None:
                mi_score = self._compute_embedding_mi(node_embedding, query_embedding)
            else:
                # Strategy 2: Word overlap-based MI (fallback)
                mi_score = self._compute_word_overlap_mi(node_content, query)

            # Cache the result
            self._set_cached_mi(node_id, query_hash, mi_score)

            return max(0.0, min(1.0, mi_score))

        except Exception as e:
            logger.warning(f"Information content scoring failed for {node_id}: {e}")
            return 0.5

    def _compute_embedding_mi(
        self,
        node_embedding: Any,
        query_embedding: Any
    ) -> float:
        """
        Compute MI from embeddings using distribution approximation.

        Converts embeddings to probability distributions via softmax,
        then computes MI from the joint distribution.
        """
        if not NUMPY_AVAILABLE or entropy is None:
            return 0.5

        try:
            # Convert to numpy arrays if needed
            node_emb = np.array(node_embedding) if not isinstance(node_embedding, np.ndarray) else node_embedding
            query_emb = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding

            # Softmax to get probability distributions
            def softmax(x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / (exp_x.sum() + 1e-10)

            p_node = softmax(node_emb)
            p_query = softmax(query_emb)

            # Create joint distribution (outer product normalized)
            # This approximates P(Node, Query) assuming independence structure
            joint = np.outer(p_node[:50], p_query[:50])  # Limit dimensions for speed
            joint = joint / (joint.sum() + 1e-10)

            # Compute marginals
            marginal_x = joint.sum(axis=1)
            marginal_y = joint.sum(axis=0)

            # Compute MI: I(X;Y) = H(X) + H(Y) - H(X,Y)
            h_x = entropy(marginal_x + 1e-10)
            h_y = entropy(marginal_y + 1e-10)
            h_xy = entropy(joint.flatten() + 1e-10)

            mi = h_x + h_y - h_xy
            mi = max(0.0, mi)  # MI is non-negative

            # Normalize by max possible MI
            max_mi = min(h_x, h_y) + 1e-10
            normalized_mi = mi / max_mi

            return max(0.0, min(1.0, normalized_mi))

        except Exception as e:
            logger.debug(f"Embedding MI computation failed: {e}")
            return 0.5

    def _compute_word_overlap_mi(
        self,
        node_content: str,
        query: str
    ) -> float:
        """
        Compute MI approximation from word overlap.

        Uses Jaccard-based mutual information proxy:
        - Word overlap captures shared information
        - Entropy of node content captures diversity
        """
        try:
            # Tokenize
            query_words = set(query.lower().split())
            node_words = set(node_content.lower().split())

            if not query_words or not node_words:
                return 0.0

            # Compute overlap metrics
            overlap = len(query_words & node_words)
            union = len(query_words | node_words)

            if union == 0:
                return 0.0

            # Jaccard similarity as MI proxy
            jaccard = overlap / union

            # Compute node entropy for diversity boost
            node_entropy = self._compute_content_entropy(node_content)

            # Score: Jaccard + entropy factor
            # Low entropy (focused) nodes with high overlap are most valuable
            # High entropy (diverse) nodes get slight boost for broader coverage
            entropy_factor = 1.0 / (1.0 + node_entropy * 0.5)

            score = 0.7 * jaccard + 0.3 * jaccard * entropy_factor

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"Word overlap MI computation failed: {e}")
            return 0.5

    def _compute_content_entropy(self, content: str) -> float:
        """
        Compute entropy of content from word distribution.

        Returns entropy normalized to [0, 1] range.
        Caches results for repeated calls.
        """
        # Check cache
        content_hash = str(hash(content[:100]))  # Hash first 100 chars
        if content_hash in self._entropy_cache:
            return self._entropy_cache[content_hash]

        if not NUMPY_AVAILABLE or entropy is None:
            return 0.5

        try:
            words = content.lower().split()
            if not words:
                return 0.0

            # Word frequency distribution
            word_counts = Counter(words)
            total = sum(word_counts.values())
            probs = np.array([c / total for c in word_counts.values()])

            # Entropy normalized by max possible (uniform distribution)
            h = entropy(probs)
            h_max = np.log2(len(word_counts)) if len(word_counts) > 1 else 1.0

            normalized = h / h_max if h_max > 0 else 0.0

            # Cache result
            self._entropy_cache[content_hash] = normalized

            return max(0.0, min(1.0, normalized))

        except Exception:
            return 0.5

    def _get_cached_mi(self, node_id: str, query_hash: str) -> Optional[float]:
        """Get cached MI score if available."""
        key = (node_id, query_hash)
        if key in self._mi_cache:
            self._stats['mi_cache_hits'] += 1
            return self._mi_cache[key]
        self._stats['mi_cache_misses'] += 1
        return None

    def _set_cached_mi(self, node_id: str, query_hash: str, mi_score: float):
        """Cache MI score with LRU eviction."""
        if len(self._mi_cache) >= self._mi_cache_size:
            # Simple eviction: clear oldest half
            keys = list(self._mi_cache.keys())[:self._mi_cache_size // 2]
            for k in keys:
                del self._mi_cache[k]

        self._mi_cache[(node_id, query_hash)] = mi_score

    # ==========================================================================
    # Phase 5: Entropy-Aware Aggregation
    # ==========================================================================

    def aggregate_scores_entropy_aware(
        self,
        scores: Dict[ImportanceSignal, float],
        node_content: Optional[str] = None,
        weights: Optional[Dict[ImportanceSignal, float]] = None
    ) -> float:
        """
        Entropy-aware score aggregation.

        Lower entropy nodes (more certain/focused) get higher weight.
        Higher entropy nodes (more diverse/uncertain) get lower weight.

        Formula:
            entropy_weight = 1 / (1 + H(node))
            adjusted_score = base_score * entropy_weight blend

        Args:
            scores: Dict of signal -> score
            node_content: Node text for entropy computation
            weights: Optional custom weights

        Returns:
            Entropy-adjusted importance score
        """
        # Get base weighted average
        base_score = self.aggregate_scores(scores, weights)

        if not self.config.use_entropy_aware_aggregation:
            return base_score

        if node_content is None:
            return base_score

        # Compute node entropy
        node_entropy = self._compute_content_entropy(node_content)

        # Apply entropy adjustment
        # Low entropy = certain/focused = boost importance
        # High entropy = uncertain/diverse = reduce importance
        entropy_weight = 1.0 / (1.0 + node_entropy)

        # Blend: 70% base score, 30% entropy-adjusted
        adjusted_score = 0.7 * base_score + 0.3 * (base_score * entropy_weight)

        return max(0.0, min(1.0, adjusted_score))

    def batch_compute_mi_scores(
        self,
        node_ids: List[str],
        query: str,
        node_contents: Dict[str, str],
        query_embedding: Optional[Any] = None,
        node_embeddings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Batch compute MI scores for multiple nodes efficiently.

        Uses caching to avoid recomputation for repeated queries.

        Args:
            node_ids: List of node IDs to score
            query: Query text
            node_contents: Dict mapping node_id -> content
            query_embedding: Pre-computed query embedding (optional)
            node_embeddings: Dict mapping node_id -> embedding (optional)

        Returns:
            Dict mapping node_id -> MI score
        """
        if not INFORMATION_THEORY_AVAILABLE:
            return {n: 0.5 for n in node_ids}

        mi_scores = {}
        query_hash = str(hash(query))

        for node_id in node_ids:
            # Check cache first
            cached = self._get_cached_mi(node_id, query_hash)
            if cached is not None:
                mi_scores[node_id] = cached
                continue

            # Compute MI
            content = node_contents.get(node_id, "")
            node_emb = node_embeddings.get(node_id) if node_embeddings else None

            mi = self._score_information_content(
                node_id, query, content, node_emb, query_embedding
            )
            mi_scores[node_id] = mi

        return mi_scores

    def invalidate_mi_cache(self):
        """Invalidate MI cache (call after significant content changes)."""
        self._mi_cache.clear()
        self._entropy_cache.clear()

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
            'cache_misses': 0,
            'mi_cache_hits': 0,  # Phase 5
            'mi_cache_misses': 0  # Phase 5
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
