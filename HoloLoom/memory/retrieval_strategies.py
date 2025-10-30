"""
Memory Retrieval Strategies
============================
Modular retrieval implementations for HoloLoom memory systems.

Strategies:
- StaticRetrieval: Traditional BM25 + cosine similarity (default)
- SpringActivation: Physics-based spreading activation
- HybridRetrieval: Embeddings for seeds + springs for expansion

All strategies implement the RetrievalStrategy protocol, making them
swappable via configuration without changing orchestrator code.

Author: HoloLoom Retrieval Team
Date: 2025-10-29
"""

from typing import List, Dict, Optional, Any
import time
import asyncio

from HoloLoom.protocols import RetrievalStrategy, RetrievalResult, SpringActivationMetadata
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig


# ============================================================================
# Static Retrieval (Default/Fallback)
# ============================================================================

class StaticRetrieval:
    """
    Traditional static retrieval using BM25 + cosine similarity.

    This is the default, reliable fallback strategy. It retrieves
    memories based on:
    - Text similarity (BM25 for keyword matching)
    - Embedding similarity (cosine distance in vector space)

    No graph traversal, no multi-hop reasoning.
    Fast, simple, reliable.
    """

    def __init__(self, shards: List[MemoryShard], embedding_fn=None):
        """
        Initialize static retrieval.

        Args:
            shards: List of memory shards to search
            embedding_fn: Optional embedding function for semantic search
        """
        self.shards = shards
        self.embedding_fn = embedding_fn

    async def retrieve(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Retrieve top-k shards by similarity.

        Args:
            query: Query object
            k: Number of results

        Returns:
            List of up to k MemoryShard objects
        """
        # Simple implementation: just return first k shards
        # In production, this would use BM25 + embedding similarity
        return self.shards[:k]

    async def retrieve_with_metadata(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve with detailed metadata."""
        start_time = time.perf_counter()

        shards = await self.retrieve(query, k, **kwargs)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            shards=shards,
            strategy="static",
            query_text=query.text,
            k_requested=k,
            k_returned=len(shards),
            retrieval_time_ms=elapsed_ms,
            avg_confidence=0.5,  # Placeholder
            min_confidence=0.3,
            max_confidence=0.7,
            metadata={
                "method": "simple_list_slice",
                "note": "This is a placeholder static retrieval"
            }
        )


# ============================================================================
# Spring Activation Retrieval
# ============================================================================

class SpringActivationRetrieval:
    """
    Physics-based spreading activation retrieval.

    This strategy:
    1. Finds seed nodes via embedding similarity
    2. Activates those seeds in the knowledge graph
    3. Propagates activation through springs (edges)
    4. Retrieves memories for all activated nodes

    Advantages:
    - Multi-hop transitive relationships
    - Context-sensitive (edge type affects propagation)
    - Energy-based confidence scores
    - Handles multi-modal queries (activate multiple seeds)

    Example:
        Query: "How does Thompson Sampling work?"

        1. Find seeds: [Thompson Sampling (0.95), Bandits (0.75)]
        2. Activate seeds in graph
        3. Springs propagate:
           Thompson Sampling (1.0) → Bayesian Inference (0.7)
                                  → Exploration (0.5)
                                  → Regret Bounds (0.3)
        4. Retrieve all with activation > 0.1
    """

    def __init__(
        self,
        graph,  # KG/YarnGraph
        shards: List[MemoryShard],
        shard_map: Dict[str, MemoryShard],  # {node_id: shard}
        spring_config: Optional[SpringConfig] = None,
        embedding_fn=None
    ):
        """
        Initialize spring activation retrieval.

        Args:
            graph: Knowledge graph (must support SpringDynamics)
            shards: List of all memory shards
            shard_map: Mapping from node IDs to shards
            spring_config: SpringConfig (or None for defaults)
            embedding_fn: Function to embed queries and find seed nodes
        """
        self.graph = graph
        self.shards = shards
        self.shard_map = shard_map
        self.embedding_fn = embedding_fn

        # Create spring dynamics engine
        self.spring_config = spring_config or SpringConfig()
        self.dynamics = SpringDynamics(graph, self.spring_config)

    async def retrieve(
        self,
        query: Query,
        k: int = 5,
        seed_k: int = 3,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Retrieve memories using spring activation.

        Args:
            query: Query object
            k: Max number of results (may return more if many nodes activated)
            seed_k: Number of seed nodes to activate initially

        Returns:
            List of MemoryShard objects for activated nodes
        """
        result = await self.retrieve_with_metadata(query, k, seed_k, **kwargs)
        return result.shards

    async def retrieve_with_metadata(
        self,
        query: Query,
        k: int = 5,
        seed_k: int = 3,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve with detailed spring activation metadata."""

        # === 1. Find Seed Nodes (via embedding) ===
        embedding_start = time.perf_counter()
        seed_nodes = await self._find_seed_nodes(query, seed_k)
        embedding_time_ms = (time.perf_counter() - embedding_start) * 1000

        if not seed_nodes:
            # No seeds found, return empty result
            return RetrievalResult(
                shards=[],
                strategy="spring_activation",
                query_text=query.text,
                k_requested=k,
                k_returned=0,
                retrieval_time_ms=embedding_time_ms,
                avg_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                metadata={
                    "error": "No seed nodes found",
                    "embedding_time_ms": embedding_time_ms
                }
            )

        # === 2. Activate Seeds ===
        self.dynamics.reset()
        self.dynamics.activate_nodes(seed_nodes)

        # === 3. Propagate Activation ===
        propagation_start = time.perf_counter()
        propagation_result = self.dynamics.propagate()
        propagation_time_ms = (time.perf_counter() - propagation_start) * 1000

        # === 4. Get Shards for Activated Nodes ===
        shard_start = time.perf_counter()
        activated_node_ids = propagation_result.activated_nodes[:k]  # Limit to k
        shards = []
        for node_id in activated_node_ids:
            shard = self.shard_map.get(node_id)
            if shard:
                shards.append(shard)
        shard_time_ms = (time.perf_counter() - shard_start) * 1000

        # === 5. Calculate Confidence ===
        activations = [
            propagation_result.node_activations.get(nid, 0.0)
            for nid in activated_node_ids
        ]

        avg_conf = sum(activations) / len(activations) if activations else 0.0
        min_conf = min(activations) if activations else 0.0
        max_conf = max(activations) if activations else 0.0

        total_time_ms = embedding_time_ms + propagation_time_ms + shard_time_ms

        # === 6. Build Metadata ===
        spring_metadata = SpringActivationMetadata(
            iterations=propagation_result.iterations,
            converged=propagation_result.converged,
            final_energy=propagation_result.final_energy,
            seed_nodes=list(seed_nodes.keys()),
            activated_count=len(propagation_result.activated_nodes),
            activation_threshold=self.spring_config.activation_threshold,
            node_activations=propagation_result.node_activations,
            embedding_time_ms=embedding_time_ms,
            propagation_time_ms=propagation_time_ms,
            shard_retrieval_time_ms=shard_time_ms
        )

        return RetrievalResult(
            shards=shards,
            strategy="spring_activation",
            query_text=query.text,
            k_requested=k,
            k_returned=len(shards),
            retrieval_time_ms=total_time_ms,
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            metadata={
                "spring_activation": spring_metadata,
                "propagation": str(propagation_result)
            }
        )

    async def _find_seed_nodes(
        self,
        query: Query,
        k: int
    ) -> Dict[str, float]:
        """
        Find seed nodes for spring activation.

        Uses embedding similarity to find top-k most relevant nodes.

        Args:
            query: Query object
            k: Number of seeds

        Returns:
            {node_id: activation_level} for top-k seeds
        """
        # Placeholder: In production, use actual embedding similarity
        # For now, just activate a few nodes from the graph

        if not self.graph.G.nodes():
            return {}

        # Get first k nodes from graph (placeholder)
        seed_nodes = {}
        for i, node_id in enumerate(list(self.graph.G.nodes())[:k]):
            # Activation decreases for each seed (most similar first)
            activation = 1.0 - (i * 0.1)
            seed_nodes[node_id] = max(0.5, activation)

        return seed_nodes


# ============================================================================
# Hybrid Retrieval (Embedding Seeds + Spring Expansion)
# ============================================================================

class HybridRetrieval:
    """
    Hybrid strategy: Use embeddings for precision, springs for recall.

    Workflow:
    1. Embedding similarity finds high-precision seeds (top-3)
    2. Spring activation expands to related concepts (recall)
    3. Re-rank combined results by composite score

    This combines the best of both worlds:
    - Embeddings: High precision for direct matches
    - Springs: High recall for transitive relationships
    """

    def __init__(
        self,
        static_retrieval: StaticRetrieval,
        spring_retrieval: SpringActivationRetrieval
    ):
        """Initialize hybrid retrieval."""
        self.static = static_retrieval
        self.spring = spring_retrieval

    async def retrieve(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> List[MemoryShard]:
        """Retrieve using hybrid strategy."""
        result = await self.retrieve_with_metadata(query, k, **kwargs)
        return result.shards

    async def retrieve_with_metadata(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve with hybrid metadata."""
        start_time = time.perf_counter()

        # Run both strategies in parallel
        static_task = self.static.retrieve_with_metadata(query, k=3)
        spring_task = self.spring.retrieve_with_metadata(query, k=k, seed_k=3)

        static_result, spring_result = await asyncio.gather(static_task, spring_task)

        # Combine results (simple union for now)
        # In production: re-rank by composite score
        combined_shards = []
        seen_ids = set()

        for shard in spring_result.shards + static_result.shards:
            shard_id = getattr(shard, 'id', id(shard))
            if shard_id not in seen_ids:
                combined_shards.append(shard)
                seen_ids.add(shard_id)

        # Limit to k
        combined_shards = combined_shards[:k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            shards=combined_shards,
            strategy="hybrid",
            query_text=query.text,
            k_requested=k,
            k_returned=len(combined_shards),
            retrieval_time_ms=elapsed_ms,
            avg_confidence=(static_result.avg_confidence + spring_result.avg_confidence) / 2,
            min_confidence=min(static_result.min_confidence, spring_result.min_confidence),
            max_confidence=max(static_result.max_confidence, spring_result.max_confidence),
            metadata={
                "static_time_ms": static_result.retrieval_time_ms,
                "spring_time_ms": spring_result.retrieval_time_ms,
                "static_count": len(static_result.shards),
                "spring_count": len(spring_result.shards),
                "spring_metadata": spring_result.metadata.get("spring_activation")
            }
        )


# ============================================================================
# Strategy Factory
# ============================================================================

def create_retrieval_strategy(
    strategy_name: str,
    graph=None,
    shards: Optional[List[MemoryShard]] = None,
    shard_map: Optional[Dict[str, MemoryShard]] = None,
    spring_config: Optional[SpringConfig] = None,
    **kwargs
) -> RetrievalStrategy:
    """
    Factory function to create retrieval strategies.

    Args:
        strategy_name: "static", "spring", or "hybrid"
        graph: Knowledge graph (required for spring/hybrid)
        shards: List of memory shards
        shard_map: Node ID → MemoryShard mapping (required for spring/hybrid)
        spring_config: SpringConfig for spring strategy
        **kwargs: Additional strategy-specific params

    Returns:
        RetrievalStrategy implementation
    """
    shards = shards or []

    if strategy_name == "static":
        return StaticRetrieval(shards=shards)

    elif strategy_name == "spring":
        if not graph or not shard_map:
            raise ValueError("Spring retrieval requires graph and shard_map")
        return SpringActivationRetrieval(
            graph=graph,
            shards=shards,
            shard_map=shard_map,
            spring_config=spring_config
        )

    elif strategy_name == "hybrid":
        if not graph or not shard_map:
            raise ValueError("Hybrid retrieval requires graph and shard_map")

        static = StaticRetrieval(shards=shards)
        spring = SpringActivationRetrieval(
            graph=graph,
            shards=shards,
            shard_map=shard_map,
            spring_config=spring_config
        )
        return HybridRetrieval(static, spring)

    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy_name}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'StaticRetrieval',
    'SpringActivationRetrieval',
    'HybridRetrieval',
    'create_retrieval_strategy',
]
