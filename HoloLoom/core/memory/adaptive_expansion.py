"""
Adaptive Graph Expansion (Phase 1)
===================================

Importance-weighted, budget-aware graph traversal that replaces fixed-depth expansion.

Key Features:
- Priority queue-based BFS (importance × relevance scoring)
- Token budget tracking per node (Matryoshka-aware: 384D/256D/128D)
- Early stopping on relevance decay or budget exhaustion
- Multi-signal scoring: recency + relevance + centrality + heat
- Edge type weighting (IS_A > MENTIONS)

Author: Claude Code
Date: 2025-11-24
Phase: 1 of 4 (Streaming Graph Expansion Roadmap)
"""

import heapq
import time
import math
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol Definitions & Data Structures
# ============================================================================

class EdgeType(str, Enum):
    """Edge types with importance weights."""
    IS_A = "IS_A"                    # High importance (taxonomy)
    USES = "USES"                    # High importance (functional)
    LEADS_TO = "LEADS_TO"            # High importance (causal)
    PART_OF = "PART_OF"              # Medium importance (composition)
    IN_TIME = "IN_TIME"              # Medium importance (temporal)
    OCCURRED_AT = "OCCURRED_AT"      # Medium importance (event)
    MENTIONS = "MENTIONS"            # Low importance (reference)
    UNKNOWN = "UNKNOWN"              # Lowest importance


# Edge type importance weights (used in relevance scoring)
EDGE_IMPORTANCE_WEIGHTS = {
    EdgeType.IS_A: 1.0,
    EdgeType.USES: 1.0,
    EdgeType.LEADS_TO: 1.0,
    EdgeType.PART_OF: 0.7,
    EdgeType.IN_TIME: 0.7,
    EdgeType.OCCURRED_AT: 0.7,
    EdgeType.MENTIONS: 0.3,
    EdgeType.UNKNOWN: 0.1
}


@dataclass
class ExpansionNode:
    """
    Node in expansion frontier.

    Used in priority queue for adaptive BFS traversal.

    Attributes:
        node_id: Entity identifier
        priority: Negative priority for max-heap (higher = better)
        hop_distance: Hops from seed nodes
        parent_id: Node that led to discovery
        edge_type: Type of edge from parent
        relevance_score: Query relevance (0.0-1.0)
        importance_score: Multi-signal importance (0.0-1.0)
        token_estimate: Estimated token cost (Matryoshka-aware)
    """
    node_id: str
    priority: float  # Negative for max-heap
    hop_distance: int
    parent_id: Optional[str] = None
    edge_type: Optional[str] = None
    relevance_score: float = 0.0
    importance_score: float = 0.0
    token_estimate: int = 0

    def __lt__(self, other):
        """Comparison for heapq (lower priority value = popped first)."""
        return self.priority < other.priority


@dataclass
class ExpansionResult:
    """
    Result of adaptive graph expansion.

    Attributes:
        nodes: Expanded node IDs (ordered by discovery)
        edges: List of (src, dst, edge_type) tuples
        total_tokens: Total token count consumed
        avg_relevance: Average relevance score across nodes
        hops_taken: Number of BFS hops executed
        nodes_visited: Total nodes considered (including rejected)
        stopping_reason: Why expansion stopped
        execution_time_ms: Time taken for expansion
        metadata: Additional expansion statistics
    """
    nodes: List[str]
    edges: List[Tuple[str, str, str]]
    total_tokens: int
    avg_relevance: float
    hops_taken: int
    nodes_visited: int
    stopping_reason: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetStatus:
    """Token budget tracking status."""
    consumed: int
    remaining: int
    limit: int
    utilization: float  # 0.0-1.0


# ============================================================================
# Relevance Scoring
# ============================================================================

class RelevanceScorer:
    """
    Query-aware relevance scoring with edge type weighting.

    Extends existing importance scoring with:
    - Query text similarity (if embedder available)
    - Edge type importance weights
    - Distance decay factor
    - Multi-signal aggregation
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        distance_decay: float = 0.85,
        use_edge_weights: bool = True
    ):
        """
        Initialize relevance scorer.

        Args:
            embedder: Optional embedding model for semantic similarity
            distance_decay: Decay factor per hop (0.85 = 15% per hop)
            use_edge_weights: Whether to weight by edge type importance
        """
        self.embedder = embedder
        self.distance_decay = distance_decay
        self.use_edge_weights = use_edge_weights

        # Cache for query embedding
        self._query_embedding: Optional[Any] = None
        self._query_text: Optional[str] = None

    def set_query(self, query: str):
        """
        Set current query and cache embedding.

        Args:
            query: Query text
        """
        self._query_text = query
        if self.embedder:
            try:
                self._query_embedding = self.embedder.encode(query)
            except Exception as e:
                logger.warning(f"Failed to encode query: {e}")
                self._query_embedding = None

    def score_relevance(
        self,
        node_id: str,
        hop_distance: int,
        edge_type: Optional[str] = None,
        node_content: Optional[str] = None,
        importance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute query-aware relevance score.

        Combines:
        1. Distance decay (closer = more relevant)
        2. Edge type importance (IS_A > MENTIONS)
        3. Semantic similarity (if embedder available)
        4. Importance signals (if available)

        Args:
            node_id: Node to score
            hop_distance: Hops from seed nodes
            edge_type: Type of edge from parent
            node_content: Optional node text content
            importance_scores: Optional pre-computed importance scores

        Returns:
            Relevance score (0.0-1.0)
        """
        # Base relevance from distance decay
        distance_factor = self.distance_decay ** hop_distance

        # Edge type weighting
        edge_factor = 1.0
        if self.use_edge_weights and edge_type:
            try:
                edge_enum = EdgeType(edge_type)
                edge_factor = EDGE_IMPORTANCE_WEIGHTS.get(edge_enum, 0.5)
            except ValueError:
                edge_factor = EDGE_IMPORTANCE_WEIGHTS[EdgeType.UNKNOWN]

        # Semantic similarity (if embedder available and node content provided)
        semantic_factor = 0.5  # Neutral if no semantic info
        if self.embedder and self._query_embedding is not None and node_content:
            try:
                node_embedding = self.embedder.encode(node_content)
                if NUMPY_AVAILABLE:
                    # Cosine similarity
                    similarity = float(np.dot(self._query_embedding, node_embedding) /
                                     (np.linalg.norm(self._query_embedding) *
                                      np.linalg.norm(node_embedding) + 1e-8))
                    semantic_factor = (similarity + 1.0) / 2.0  # Normalize to 0-1
                else:
                    semantic_factor = 0.5
            except Exception as e:
                logger.debug(f"Semantic scoring failed for {node_id}: {e}")
                semantic_factor = 0.5

        # Importance boost (if available)
        importance_factor = 0.5  # Neutral if no importance scores
        if importance_scores:
            # Aggregate importance signals (weighted average)
            importance_factor = sum(importance_scores.values()) / max(len(importance_scores), 1)

        # Weighted combination
        # Distance and edge type are most important (70%)
        # Semantic and importance provide additional signal (30%)
        relevance = (
            0.4 * distance_factor +
            0.3 * edge_factor +
            0.2 * semantic_factor +
            0.1 * importance_factor
        )

        return max(0.0, min(1.0, relevance))


# ============================================================================
# Token Budget Tracking
# ============================================================================

class BudgetTracker:
    """
    Token budget tracking with Matryoshka-aware estimation.

    Tracks token consumption and provides utility for budget-aware decisions.
    """

    def __init__(
        self,
        token_budget: int,
        matryoshka_scales: List[int] = [384, 256, 128],
        avg_tokens_per_node: int = 100
    ):
        """
        Initialize budget tracker.

        Args:
            token_budget: Maximum token budget
            matryoshka_scales: Available embedding scales (descending)
            avg_tokens_per_node: Average tokens per node (fallback estimate)
        """
        self.token_budget = token_budget
        self.matryoshka_scales = sorted(matryoshka_scales, reverse=True)
        self.avg_tokens_per_node = avg_tokens_per_node

        self.consumed_tokens = 0
        self.node_token_estimates: Dict[str, int] = {}

    def estimate_tokens(
        self,
        node_id: str,
        importance_score: float,
        node_content: Optional[str] = None
    ) -> int:
        """
        Estimate token cost for a node (Matryoshka-aware).

        Higher importance nodes get higher-dimensional embeddings:
        - High (>0.75): 384D
        - Medium (0.5-0.75): 256D
        - Low (0.25-0.5): 128D
        - Very low (<0.25): Dropped

        Args:
            node_id: Node to estimate
            importance_score: Importance score (0.0-1.0)
            node_content: Optional node text content (for accurate count)

        Returns:
            Estimated token count
        """
        # Matryoshka scale selection based on importance
        if importance_score >= 0.75:
            scale_factor = 1.0  # 384D (full detail)
        elif importance_score >= 0.5:
            scale_factor = 0.67  # 256D (2/3 of full)
        elif importance_score >= 0.25:
            scale_factor = 0.33  # 128D (1/3 of full)
        else:
            return 0  # Drop very low importance nodes

        # Estimate tokens (use actual content length if available)
        if node_content:
            # Rough estimate: 1 token ≈ 4 characters
            base_tokens = len(node_content) // 4
        else:
            base_tokens = self.avg_tokens_per_node

        tokens = int(base_tokens * scale_factor)
        self.node_token_estimates[node_id] = tokens
        return tokens

    def consume(self, node_id: str, tokens: int) -> bool:
        """
        Consume tokens from budget.

        Args:
            node_id: Node being added
            tokens: Token cost

        Returns:
            True if consumption succeeded, False if budget exhausted
        """
        if self.consumed_tokens + tokens > self.token_budget:
            return False

        self.consumed_tokens += tokens
        self.node_token_estimates[node_id] = tokens
        return True

    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        return BudgetStatus(
            consumed=self.consumed_tokens,
            remaining=self.token_budget - self.consumed_tokens,
            limit=self.token_budget,
            utilization=self.consumed_tokens / max(self.token_budget, 1)
        )

    def can_afford(self, estimated_tokens: int, safety_margin: float = 0.9) -> bool:
        """
        Check if budget can afford additional tokens.

        Args:
            estimated_tokens: Tokens to check
            safety_margin: Stop at X% of budget (0.9 = 90%)

        Returns:
            True if can afford, False otherwise
        """
        threshold = self.token_budget * safety_margin
        return (self.consumed_tokens + estimated_tokens) <= threshold


# ============================================================================
# Adaptive Graph Expander
# ============================================================================

class AdaptiveExpander:
    """
    Priority queue-based BFS with importance-weighted, budget-aware traversal.

    Replaces fixed-depth expansion with adaptive algorithm that:
    1. Uses priority queue (importance × relevance)
    2. Tracks token budget (Matryoshka-aware)
    3. Stops early on relevance decay or budget exhaustion
    4. Provides complete provenance (nodes, edges, metadata)
    """

    def __init__(
        self,
        relevance_scorer: Optional[RelevanceScorer] = None,
        embedder: Optional[Any] = None,
        matryoshka_scales: List[int] = [384, 256, 128],
        avg_tokens_per_node: int = 100
    ):
        """
        Initialize adaptive expander.

        Args:
            relevance_scorer: Optional custom relevance scorer
            embedder: Optional embedding model
            matryoshka_scales: Embedding scales for token estimation
            avg_tokens_per_node: Average tokens per node (fallback)
        """
        self.relevance_scorer = relevance_scorer or RelevanceScorer(
            embedder=embedder,
            distance_decay=0.85,
            use_edge_weights=True
        )
        self.matryoshka_scales = matryoshka_scales
        self.avg_tokens_per_node = avg_tokens_per_node

        # Stats
        self._stats = {
            'total_expansions': 0,
            'avg_nodes_expanded': 0.0,
            'avg_execution_time_ms': 0.0
        }

    async def expand(
        self,
        query: str,
        seed_nodes: List[str],
        graph: Any,  # KG or NetworkX MultiDiGraph
        token_budget: int = 2000,
        min_relevance: float = 0.3,
        max_hops: int = 5,
        importance_scores: Optional[Dict[str, float]] = None,
        node_contents: Optional[Dict[str, str]] = None
    ) -> ExpansionResult:
        """
        Adaptive graph expansion with budget and relevance constraints.

        Algorithm:
        1. Initialize priority queue with seed nodes
        2. Pop highest priority node
        3. Check stopping conditions (budget, relevance, hops)
        4. Expand neighbors, score relevance, add to queue
        5. Repeat until queue empty or stopping condition met

        Args:
            query: Query text
            seed_nodes: Initial nodes to expand from
            graph: Knowledge graph (KG or NetworkX MultiDiGraph)
            token_budget: Maximum tokens allowed
            min_relevance: Minimum relevance threshold (0.3 = drop <30%)
            max_hops: Soft maximum hops (can stop earlier)
            importance_scores: Optional pre-computed importance scores
            node_contents: Optional node text content for semantic scoring

        Returns:
            ExpansionResult with expanded nodes, edges, and metadata
        """
        start_time = time.time()

        # Initialize
        self.relevance_scorer.set_query(query)
        budget_tracker = BudgetTracker(
            token_budget=token_budget,
            matryoshka_scales=self.matryoshka_scales,
            avg_tokens_per_node=self.avg_tokens_per_node
        )

        # Priority queue (max-heap via negative priorities)
        frontier: List[ExpansionNode] = []

        # Visited tracking
        visited: Set[str] = set()
        expanded_nodes: List[str] = []
        expanded_edges: List[Tuple[str, str, str]] = []

        # Initialize frontier with seed nodes
        for seed in seed_nodes:
            if seed not in graph.G if hasattr(graph, 'G') else graph:
                continue

            # Score seed node
            imp_score = importance_scores.get(seed, 0.5) if importance_scores else 0.5
            rel_score = 1.0  # Seeds are maximally relevant
            priority = -(imp_score * rel_score)  # Negative for max-heap

            # Estimate tokens
            node_content = node_contents.get(seed) if node_contents else None
            tokens = budget_tracker.estimate_tokens(seed, imp_score, node_content)

            # Add to frontier
            expansion_node = ExpansionNode(
                node_id=seed,
                priority=priority,
                hop_distance=0,
                parent_id=None,
                edge_type=None,
                relevance_score=rel_score,
                importance_score=imp_score,
                token_estimate=tokens
            )
            heapq.heappush(frontier, expansion_node)

        # Statistics
        nodes_visited = 0
        stopping_reason = "completed"

        # Adaptive BFS with priority queue
        while frontier:
            # Pop highest priority node
            current = heapq.heappop(frontier)
            nodes_visited += 1

            # Skip if already visited
            if current.node_id in visited:
                continue

            visited.add(current.node_id)

            # Check stopping conditions

            # 1. Relevance decay (node not relevant enough)
            if current.relevance_score < min_relevance:
                stopping_reason = "relevance_decay"
                break

            # 2. Budget exhaustion (can't afford this node)
            if not budget_tracker.can_afford(current.token_estimate):
                stopping_reason = "budget_exhausted"
                break

            # 3. Max hops reached (soft limit)
            if current.hop_distance >= max_hops:
                stopping_reason = "max_hops_reached"
                # Continue processing nodes at current hop distance
                if not any(n.hop_distance < max_hops for n in frontier):
                    break

            # Consume budget
            budget_tracker.consume(current.node_id, current.token_estimate)
            expanded_nodes.append(current.node_id)

            # Add edge (if not seed node)
            if current.parent_id:
                expanded_edges.append((
                    current.parent_id,
                    current.node_id,
                    current.edge_type or "UNKNOWN"
                ))

            # Expand neighbors
            try:
                # Get neighbors from graph
                if hasattr(graph, 'G'):
                    # HoloLoom KG
                    neighbors = list(graph.G.successors(current.node_id))
                elif NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
                    neighbors = list(graph.successors(current.node_id))
                else:
                    neighbors = []

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    # Get edge type
                    edge_type = self._get_edge_type(graph, current.node_id, neighbor)

                    # Score neighbor
                    imp_score = importance_scores.get(neighbor, 0.5) if importance_scores else 0.5
                    node_content = node_contents.get(neighbor) if node_contents else None

                    rel_score = self.relevance_scorer.score_relevance(
                        node_id=neighbor,
                        hop_distance=current.hop_distance + 1,
                        edge_type=edge_type,
                        node_content=node_content,
                        importance_scores={} if not importance_scores else {k: v for k, v in importance_scores.items() if k == neighbor}
                    )

                    # Skip low-relevance neighbors
                    if rel_score < min_relevance:
                        continue

                    # Estimate tokens
                    tokens = budget_tracker.estimate_tokens(neighbor, imp_score, node_content)

                    # Skip if can't afford
                    if not budget_tracker.can_afford(tokens):
                        continue

                    # Compute priority (importance × relevance)
                    priority = -(imp_score * rel_score)

                    # Add to frontier
                    expansion_node = ExpansionNode(
                        node_id=neighbor,
                        priority=priority,
                        hop_distance=current.hop_distance + 1,
                        parent_id=current.node_id,
                        edge_type=edge_type,
                        relevance_score=rel_score,
                        importance_score=imp_score,
                        token_estimate=tokens
                    )
                    heapq.heappush(frontier, expansion_node)

            except Exception as e:
                logger.warning(f"Failed to expand neighbors for {current.node_id}: {e}")
                continue

        # Compute statistics
        execution_time_ms = (time.time() - start_time) * 1000
        avg_relevance = (
            sum(n.relevance_score for n in [current] + list(frontier)) /
            max(len(expanded_nodes), 1)
        )
        budget_status = budget_tracker.get_status()

        # Update stats
        self._stats['total_expansions'] += 1
        self._stats['avg_nodes_expanded'] = (
            (self._stats['avg_nodes_expanded'] * (self._stats['total_expansions'] - 1) +
             len(expanded_nodes)) / self._stats['total_expansions']
        )
        self._stats['avg_execution_time_ms'] = (
            (self._stats['avg_execution_time_ms'] * (self._stats['total_expansions'] - 1) +
             execution_time_ms) / self._stats['total_expansions']
        )

        # Build result
        result = ExpansionResult(
            nodes=expanded_nodes,
            edges=expanded_edges,
            total_tokens=budget_status.consumed,
            avg_relevance=avg_relevance,
            hops_taken=max((n.hop_distance for n in frontier), default=0) if frontier else 0,
            nodes_visited=nodes_visited,
            stopping_reason=stopping_reason,
            execution_time_ms=execution_time_ms,
            metadata={
                'seed_nodes': seed_nodes,
                'budget_utilization': budget_status.utilization,
                'budget_remaining': budget_status.remaining,
                'frontier_size': len(frontier),
                'min_relevance_threshold': min_relevance,
                'max_hops': max_hops
            }
        )

        return result

    def _get_edge_type(self, graph: Any, src: str, dst: str) -> Optional[str]:
        """Get edge type between nodes."""
        try:
            if hasattr(graph, 'G'):
                # HoloLoom KG
                edges = list(graph.G.edges(src, dst, data=True))
                if edges:
                    return edges[0][2].get('type', 'UNKNOWN')
            elif NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
                edges = list(graph.edges(src, dst, data=True))
                if edges:
                    return edges[0][2].get('type', 'UNKNOWN')
        except Exception:
            pass
        return 'UNKNOWN'

    def get_stats(self) -> Dict[str, Any]:
        """Get expander statistics."""
        return self._stats.copy()


# ============================================================================
# Convenience Functions
# ============================================================================

async def expand_context_adaptive(
    query: str,
    seed_nodes: List[str],
    graph: Any,
    token_budget: int = 2000,
    min_relevance: float = 0.3,
    max_hops: int = 5,
    embedder: Optional[Any] = None,
    importance_scores: Optional[Dict[str, float]] = None,
    node_contents: Optional[Dict[str, str]] = None
) -> ExpansionResult:
    """
    Convenience function for adaptive graph expansion.

    Args:
        query: Query text
        seed_nodes: Initial nodes to expand from
        graph: Knowledge graph (KG or NetworkX MultiDiGraph)
        token_budget: Maximum tokens allowed (default: 2000)
        min_relevance: Minimum relevance threshold (default: 0.3)
        max_hops: Soft maximum hops (default: 5)
        embedder: Optional embedding model for semantic scoring
        importance_scores: Optional pre-computed importance scores
        node_contents: Optional node text content

    Returns:
        ExpansionResult with expanded nodes, edges, and metadata

    Example:
        >>> result = await expand_context_adaptive(
        ...     query="What is Thompson Sampling?",
        ...     seed_nodes=["thompson_sampling"],
        ...     graph=kg,
        ...     token_budget=2000
        ... )
        >>> print(f"Expanded {len(result.nodes)} nodes, {result.total_tokens} tokens")
    """
    expander = AdaptiveExpander(
        embedder=embedder,
        matryoshka_scales=[384, 256, 128],
        avg_tokens_per_node=100
    )

    return await expander.expand(
        query=query,
        seed_nodes=seed_nodes,
        graph=graph,
        token_budget=token_budget,
        min_relevance=min_relevance,
        max_hops=max_hops,
        importance_scores=importance_scores,
        node_contents=node_contents
    )
