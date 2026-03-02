"""
Navigator — Personalized PageRank traversal for context construction.

The navigator replaces the v1 4-tier cascade router. Instead of falling through
levels (exact → structured → graph → semantic), the navigator:

  1. Extracts seed entities from the query
  2. Resolves seeds to graph node IDs (alias table + fuzzy)
  3. Runs Personalized PageRank from seed nodes through the full graph
  4. Returns top-k ranked nodes as context candidates

PPR is the right primitive because:
  - It naturally diffuses relevance through relationships
  - Reset probability controls local vs global scope
  - Power iteration converges in ~10 iterations (cheap)
  - Edge weights modulate influence (importance × recency × access)

Supports two graph backends:
  - LiteMemoryBus._edges adjacency list (in-memory, no deps)
  - KG (NetworkX MultiDiGraph, full spectral methods available)

Inspired by HippoRAG v2's PPR-based retrieval.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Graph Protocol — Navigator works against any backend implementing this
# =============================================================================

class GraphBackend(Protocol):
    """Minimal interface for PPR traversal."""

    def nodes(self) -> List[str]:
        """Return all node IDs."""
        ...

    def neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        """Return [(neighbor_id, rel_type, weight)] for a node."""
        ...

    def node_data(self, node_id: str) -> Dict[str, Any]:
        """Return metadata dict for a node."""
        ...

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        ...


# =============================================================================
# Backend Adapters
# =============================================================================

class LiteBusGraph:
    """Adapter: LiteMemoryBus → GraphBackend."""

    def __init__(self, bus: Any):
        self._bus = bus

    def nodes(self) -> List[str]:
        return list(self._bus._items.keys())

    def neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        edges = self._bus._edges.get(node_id, [])
        result = []
        for target_id, rel_type in edges:
            if target_id in self._bus._items:
                weight = self._bus._items[target_id].importance
                result.append((target_id, rel_type, weight))
        return result

    def node_data(self, node_id: str) -> Dict[str, Any]:
        if node_id in self._bus._items:
            return self._bus._items[node_id].to_dict()
        return {}

    def has_node(self, node_id: str) -> bool:
        return node_id in self._bus._items


class NetworkXGraph:
    """Adapter: KG (NetworkX) → GraphBackend."""

    def __init__(self, kg: Any):
        self._kg = kg

    def nodes(self) -> List[str]:
        return list(self._kg.G.nodes())

    def neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        if node_id not in self._kg.G:
            return []
        result = []
        # Outgoing edges
        for _, dst, data in self._kg.G.out_edges(node_id, data=True):
            result.append((dst, data.get("type", "RELATED"), data.get("weight", 1.0)))
        # Incoming edges (bidirectional traversal)
        for src, _, data in self._kg.G.in_edges(node_id, data=True):
            result.append((src, data.get("type", "RELATED"), data.get("weight", 1.0)))
        return result

    def node_data(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self._kg.G:
            return {}
        return dict(self._kg.G.nodes[node_id])

    def has_node(self, node_id: str) -> bool:
        return node_id in self._kg.G


# =============================================================================
# Seed Extraction
# =============================================================================

@dataclass
class SeedNode:
    """A starting point for PPR traversal."""
    node_id: str
    weight: float  # How much reset probability this seed gets
    source: str    # How we found it: "alias", "fuzzy", "keyword", "entity_extract"


async def extract_seeds(
    query: str,
    graph: GraphBackend,
    alias_table: Optional[Dict[str, str]] = None,
    max_seeds: int = 10,
) -> List[SeedNode]:
    """Extract seed nodes from a query string.

    Strategy (in priority order):
      1. Alias resolution: exact match in alias table → weight 1.0
      2. Fuzzy alias: substring match in alias table → weight 0.6
      3. Keyword match: query words appear in node content → weight 0.4
      4. Entity extraction: capitalized words as entity candidates → weight 0.3

    Args:
        query: User's question text
        graph: Graph backend to search against
        alias_table: Optional {alias_lower: node_id} mapping
        max_seeds: Maximum number of seed nodes to return

    Returns:
        List of SeedNode objects, sorted by weight descending
    """
    seeds: Dict[str, SeedNode] = {}  # node_id → SeedNode (dedup by node_id)
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # 1. Exact alias match
    if alias_table:
        for word in query_words:
            if word in alias_table:
                nid = alias_table[word]
                if graph.has_node(nid) and nid not in seeds:
                    seeds[nid] = SeedNode(nid, 1.0, "alias")

        # Also try multi-word alias matches (2-grams, 3-grams)
        words_list = query_lower.split()
        for n in (2, 3):
            for i in range(len(words_list) - n + 1):
                phrase = " ".join(words_list[i:i + n])
                if phrase in alias_table:
                    nid = alias_table[phrase]
                    if graph.has_node(nid) and nid not in seeds:
                        seeds[nid] = SeedNode(nid, 1.0, "alias")

    # 2. Fuzzy alias match (substring)
    if alias_table and len(seeds) < max_seeds:
        for alias, nid in alias_table.items():
            if nid in seeds:
                continue
            if alias in query_lower or query_lower in alias:
                if graph.has_node(nid):
                    seeds[nid] = SeedNode(nid, 0.6, "fuzzy")
            if len(seeds) >= max_seeds:
                break

    # 3. Keyword match against node content
    if len(seeds) < max_seeds:
        for nid in graph.nodes():
            if nid in seeds:
                continue
            data = graph.node_data(nid)
            content = data.get("content", data.get("text", "")).lower()
            if not content:
                continue
            hits = sum(1 for w in query_words if w in content and len(w) > 2)
            if hits >= 2:  # Require at least 2 keyword matches
                weight = min(0.4, 0.1 * hits)
                seeds[nid] = SeedNode(nid, weight, "keyword")
            if len(seeds) >= max_seeds:
                break

    # 4. Entity extraction (capitalized words → match against node names)
    if len(seeds) < max_seeds:
        from ..graph import extract_entities_simple
        entities = extract_entities_simple(query)
        for entity in entities:
            entity_lower = entity.lower()
            for nid in graph.nodes():
                if nid in seeds:
                    continue
                data = graph.node_data(nid)
                content = data.get("content", data.get("text", "")).lower()
                if entity_lower in content:
                    seeds[nid] = SeedNode(nid, 0.3, "entity_extract")
                    break  # One match per entity
            if len(seeds) >= max_seeds:
                break

    result = sorted(seeds.values(), key=lambda s: s.weight, reverse=True)
    return result[:max_seeds]


# =============================================================================
# Personalized PageRank
# =============================================================================

@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank."""
    alpha: float = 0.15          # Reset probability (teleport back to seeds)
    max_iterations: int = 20     # Power iteration limit
    tolerance: float = 1e-6      # Convergence threshold
    max_results: int = 50        # Top-k nodes to return
    edge_weight_fn: str = "importance"  # How to weight edges: "importance", "uniform", "access"
    min_score: float = 1e-4      # Minimum PPR score to include in results


@dataclass
class NavigatorResult:
    """Output of the PPR navigator."""
    ranked_nodes: List[Tuple[str, float]]  # [(node_id, ppr_score)]
    seed_nodes: List[SeedNode]
    iterations: int
    converged: bool
    node_data: Dict[str, Dict[str, Any]]   # Cached metadata for ranked nodes

    @property
    def node_ids(self) -> List[str]:
        return [nid for nid, _ in self.ranked_nodes]

    @property
    def top_content(self) -> List[str]:
        """Get content strings for ranked nodes."""
        return [
            self.node_data.get(nid, {}).get("content", "")
            for nid in self.node_ids
        ]


def personalized_pagerank(
    graph: GraphBackend,
    seeds: List[SeedNode],
    config: PPRConfig = PPRConfig(),
) -> NavigatorResult:
    """Run Personalized PageRank from seed nodes.

    PPR computes a stationary distribution over the graph where a random walker:
      - With probability alpha, teleports back to a seed node
      - With probability (1-alpha), follows an edge to a neighbor

    The result is a ranking where nodes "close" to seeds (in graph distance,
    weighted by edge importance) get high scores.

    This is pure Python power iteration — no scipy/numpy required.

    Args:
        graph: Graph backend to traverse
        seeds: Starting nodes with their reset weights
        config: PPR parameters

    Returns:
        NavigatorResult with ranked nodes and metadata
    """
    all_nodes = graph.nodes()
    if not all_nodes or not seeds:
        return NavigatorResult([], seeds, 0, True, {})

    n = len(all_nodes)
    node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}

    # Build personalization vector (teleport distribution)
    personalization = [0.0] * n
    total_seed_weight = sum(s.weight for s in seeds)
    if total_seed_weight > 0:
        for seed in seeds:
            if seed.node_id in node_to_idx:
                idx = node_to_idx[seed.node_id]
                personalization[idx] = seed.weight / total_seed_weight

    # If no valid seeds mapped, use uniform
    if sum(personalization) == 0:
        personalization = [1.0 / n] * n

    # Build adjacency: for each node, its outgoing edges with weights
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    out_weight: List[float] = [0.0] * n

    for i, nid in enumerate(all_nodes):
        neighbors = graph.neighbors(nid)
        for target_id, rel_type, weight in neighbors:
            if target_id in node_to_idx:
                j = node_to_idx[target_id]
                # Weight selection
                if config.edge_weight_fn == "uniform":
                    w = 1.0
                else:
                    w = max(0.01, weight)  # Floor to avoid zero-weight edges
                adjacency[i].append((j, w))
                out_weight[i] += w

    # Power iteration
    alpha = config.alpha
    scores = list(personalization)  # Initialize with personalization
    converged = False

    for iteration in range(config.max_iterations):
        new_scores = [0.0] * n

        # Transition step: distribute score through edges
        for i in range(n):
            if out_weight[i] > 0:
                for j, w in adjacency[i]:
                    new_scores[j] += (1.0 - alpha) * scores[i] * (w / out_weight[i])

        # Teleport step: add reset to personalization
        for i in range(n):
            new_scores[i] += alpha * personalization[i]

        # Check convergence (L1 norm of difference)
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores

        if diff < config.tolerance:
            converged = True
            break

    # Collect results
    scored_nodes: List[Tuple[str, float]] = []
    node_data_cache: Dict[str, Dict[str, Any]] = {}

    for i, nid in enumerate(all_nodes):
        if scores[i] >= config.min_score:
            scored_nodes.append((nid, scores[i]))
            node_data_cache[nid] = graph.node_data(nid)

    # Sort by score descending
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    scored_nodes = scored_nodes[:config.max_results]

    # Keep only cached data for returned nodes
    final_cache = {nid: node_data_cache[nid] for nid, _ in scored_nodes if nid in node_data_cache}

    return NavigatorResult(
        ranked_nodes=scored_nodes,
        seed_nodes=seeds,
        iterations=iteration + 1 if not converged else iteration + 1,
        converged=converged,
        node_data=final_cache,
    )


# =============================================================================
# High-Level Navigator
# =============================================================================

class Navigator:
    """PPR-based context navigator.

    Usage:
        nav = Navigator(graph=LiteBusGraph(bus), alias_table=bus._aliases)
        result = await nav.navigate("Compare Thompson Sampling vs epsilon-greedy")
        # result.ranked_nodes → [(node_id, score), ...]
        # result.top_content → ["Thompson Sampling uses...", ...]
    """

    def __init__(
        self,
        graph: GraphBackend,
        alias_table: Optional[Dict[str, str]] = None,
        config: Optional[PPRConfig] = None,
    ):
        self.graph = graph
        self.alias_table = alias_table or {}
        self.config = config or PPRConfig()

    async def navigate(
        self,
        query: str,
        max_seeds: int = 10,
        override_config: Optional[PPRConfig] = None,
        activation_adapter=None,
    ) -> NavigatorResult:
        """Full navigation pipeline: extract seeds → [activation boost] → PPR → ranked context.

        Args:
            query: User's question
            max_seeds: Maximum seed nodes to extract
            override_config: Optional per-query config override
            activation_adapter: Optional ActivationAdapter for ACT-R seed boosting

        Returns:
            NavigatorResult with ranked context nodes
        """
        config = override_config or self.config

        # Step 1: Extract seeds
        seeds = await extract_seeds(
            query=query,
            graph=self.graph,
            alias_table=self.alias_table,
            max_seeds=max_seeds,
        )

        if not seeds:
            logger.info("Navigator: no seeds found for query '%s'", query[:80])
            return NavigatorResult([], [], 0, True, {})

        # Step 1.5: ACT-R activation boost (optional)
        # Recently-accessed nodes get higher seed weight, pulling PPR toward them
        if activation_adapter is not None:
            seeds = activation_adapter.boost_seeds(seeds)

        logger.info(
            "Navigator: %d seeds from '%s' [%s]",
            len(seeds),
            query[:50],
            ", ".join(f"{s.node_id[:12]}({s.source})" for s in seeds[:5]),
        )

        # Step 2: PPR traversal
        result = personalized_pagerank(self.graph, seeds, config)

        # Step 2.5: Update activation for retrieved nodes (recency boost)
        if activation_adapter is not None:
            retrieved_ids = [nid for nid, _ in result.ranked_nodes[:20]]
            activation_adapter.on_retrieval(retrieved_ids)

        logger.info(
            "Navigator: PPR converged=%s iter=%d → %d nodes (top: %.4f)",
            result.converged,
            result.iterations,
            len(result.ranked_nodes),
            result.ranked_nodes[0][1] if result.ranked_nodes else 0.0,
        )

        return result

    @classmethod
    def from_lite_bus(cls, bus: Any, config: Optional[PPRConfig] = None) -> "Navigator":
        """Create a Navigator from a LiteMemoryBus instance."""
        return cls(
            graph=LiteBusGraph(bus),
            alias_table=bus._aliases,
            config=config,
        )

    @classmethod
    def from_kg(cls, kg: Any, alias_table: Optional[Dict[str, str]] = None, config: Optional[PPRConfig] = None) -> "Navigator":
        """Create a Navigator from a KG (NetworkX) instance."""
        return cls(
            graph=NetworkXGraph(kg),
            alias_table=alias_table or {},
            config=config,
        )
