"""
Memory Backend Factory
======================

Factory pattern for creating unified memory backends based on configuration.

Supports pure and hybrid strategies:
- Pure: Single backend (NetworkX, Neo4j, Qdrant, Mem0)
- Hybrid: Multiple backends working together (Neo4j+Qdrant, Triple, Hyperspace)

Design Philosophy:
- Protocol-based: All backends implement MemoryStore protocol
- Dependency injection: Orchestrator doesn't know concrete implementations
- Graceful degradation: Missing backends don't break the system
- Configuration-driven: Strategy selected via Config.memory_backend

Architecture:
This is the "shuttle" that weaves together different memory threads:
- NetworkX thread: Fast in-memory graph
- Neo4j thread: Persistent graph storage
- Qdrant thread: Semantic vector space
- Mem0 thread: Intelligent memory extraction
- Hyperspace thread: Gated recursive exploration

The factory creates the right combination based on your needs.
"""

from typing import Optional, Dict, Any
import warnings

from HoloLoom.config import Config, MemoryBackend

# Import protocol
try:
    from .protocol import MemoryStore, Memory, MemoryQuery, RetrievalResult
except ImportError:
    warnings.warn("Memory protocols not available")
    MemoryStore = None


# ============================================================================
# Backend Imports (with graceful degradation)
# ============================================================================

# NetworkX backend
try:
    from HoloLoom.memory.graph import KG as NetworkXKG
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX backend not available")

# Neo4j backend
try:
    from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn("Neo4j backend not available. Install with: pip install neo4j")

# Qdrant backend
try:
    from HoloLoom.memory.stores.qdrant import QdrantMemoryStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    warnings.warn("Qdrant backend not available. Install with: pip install qdrant-client")

# Mem0 backend
try:
    from HoloLoom.memory.mem0_adapter import Mem0Adapter
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    warnings.warn("Mem0 backend not available. Install with: pip install mem0ai")


# ============================================================================
# Hybrid Memory Store
# ============================================================================

class HybridMemoryStore:
    """
    Hybrid memory store that combines multiple backends.

    Routes operations to appropriate backend(s) based on strategy:
    - store(): All backends
    - recall(): Fusion of all backend results
    - health_check(): Aggregate health status

    This is the "woven fabric" - multiple threads working together.
    """

    def __init__(
        self,
        neo4j_store: Optional[Any] = None,
        qdrant_store: Optional[Any] = None,
        mem0_store: Optional[Any] = None,
        networkx_store: Optional[Any] = None,
        strategy: str = "balanced"
    ):
        """
        Initialize hybrid memory store.

        Args:
            neo4j_store: Neo4j backend (optional)
            qdrant_store: Qdrant backend (optional)
            mem0_store: Mem0 backend (optional)
            networkx_store: NetworkX backend (optional)
            strategy: Fusion strategy (balanced, semantic_heavy, graph_heavy)
        """
        self.neo4j = neo4j_store
        self.qdrant = qdrant_store
        self.mem0 = mem0_store
        self.networkx = networkx_store
        self.strategy = strategy

        # Count active backends
        self.backends = []
        if self.neo4j:
            self.backends.append(('neo4j', self.neo4j))
        if self.qdrant:
            self.backends.append(('qdrant', self.qdrant))
        if self.mem0:
            self.backends.append(('mem0', self.mem0))
        if self.networkx:
            self.backends.append(('networkx', self.networkx))

        if not self.backends:
            raise ValueError("HybridMemoryStore requires at least one backend")

    async def store(self, memory: 'Memory', user_id: str = "default") -> str:
        """
        Store memory in all available backends.

        Returns:
            Memory ID
        """
        memory_id = memory.id

        # Store in all backends
        for name, backend in self.backends:
            try:
                await backend.store(memory, user_id)
            except Exception as e:
                warnings.warn(f"Failed to store in {name}: {e}")

        return memory_id

    async def store_many(self, memories: list, user_id: str = "default") -> list:
        """
        Batch store memories in all backends.

        Returns:
            List of memory IDs
        """
        ids = []
        for memory in memories:
            memory_id = await self.store(memory, user_id)
            ids.append(memory_id)
        return ids

    async def recall(
        self,
        query: 'MemoryQuery',
        strategy: Optional[str] = None,
        limit: int = 10
    ) -> 'RetrievalResult':
        """
        Recall memories using hybrid fusion strategy.

        Queries all backends and fuses results based on strategy.

        Args:
            query: Memory query
            strategy: Override default strategy (semantic, graph, balanced)
            limit: Max results to return

        Returns:
            Fused retrieval results
        """
        strategy = strategy or self.strategy
        results_by_backend = {}

        # Query each backend
        for name, backend in self.backends:
            try:
                result = await backend.recall(query, limit=limit * 2)  # Over-fetch for fusion
                results_by_backend[name] = result
            except Exception as e:
                warnings.warn(f"Failed to recall from {name}: {e}")

        # Fuse results based on strategy
        if strategy == "semantic_heavy":
            # Prioritize vector similarity (Qdrant > Mem0 > others)
            weights = {'qdrant': 0.7, 'mem0': 0.2, 'neo4j': 0.05, 'networkx': 0.05}
        elif strategy == "graph_heavy":
            # Prioritize graph relationships (Neo4j > NetworkX > others)
            weights = {'neo4j': 0.7, 'networkx': 0.2, 'qdrant': 0.05, 'mem0': 0.05}
        else:  # balanced
            # Equal weighting
            weights = {name: 1.0 / len(self.backends) for name, _ in self.backends}

        # Fuse results
        fused_memories = self._fuse_results(results_by_backend, weights, limit)

        return RetrievalResult(
            memories=fused_memories,
            scores=[1.0] * len(fused_memories),  # TODO: Compute actual scores
            strategy_used=f"hybrid_{strategy}",
            metadata={'backends_used': list(results_by_backend.keys())}
        )

    def _fuse_results(
        self,
        results_by_backend: Dict[str, 'RetrievalResult'],
        weights: Dict[str, float],
        limit: int
    ) -> list:
        """
        Fuse results from multiple backends using weighted scoring.

        Args:
            results_by_backend: Results from each backend
            weights: Weight for each backend
            limit: Max results to return

        Returns:
            Fused and ranked memories
        """
        # Collect all unique memories with scores
        memory_scores = {}

        for backend_name, result in results_by_backend.items():
            weight = weights.get(backend_name, 0.0)

            for memory, score in zip(result.memories, result.scores):
                mem_id = memory.id

                if mem_id not in memory_scores:
                    memory_scores[mem_id] = {
                        'memory': memory,
                        'score': 0.0
                    }

                # Add weighted score
                memory_scores[mem_id]['score'] += score * weight

        # Sort by fused score
        ranked = sorted(
            memory_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # Return top memories
        return [item['memory'] for item in ranked[:limit]]

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all backends.

        Returns:
            Aggregate health status
        """
        health = {
            'status': 'healthy',
            'backends': {},
            'total_backends': len(self.backends),
            'healthy_backends': 0
        }

        for name, backend in self.backends:
            try:
                backend_health = await backend.health_check()
                health['backends'][name] = backend_health
                if backend_health['status'] == 'healthy':
                    health['healthy_backends'] += 1
            except Exception as e:
                health['backends'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }

        # Overall status
        if health['healthy_backends'] == 0:
            health['status'] = 'unhealthy'
        elif health['healthy_backends'] < health['total_backends']:
            health['status'] = 'degraded'

        return health


# ============================================================================
# Factory Function
# ============================================================================

async def create_memory_backend(config: Config, user_id: str = "default") -> MemoryStore:
    """
    Factory function to create memory backend based on configuration.

    Args:
        config: HoloLoom configuration
        user_id: Default user ID for memory operations

    Returns:
        Configured memory backend (single or hybrid)

    Raises:
        ValueError: If required backend is not available

    Examples:
        # Pure NetworkX (development)
        config = Config.fast()
        config.memory_backend = MemoryBackend.NETWORKX
        memory = await create_memory_backend(config)

        # Hybrid Neo4j + Qdrant (production)
        config = Config.fused()
        config.memory_backend = MemoryBackend.NEO4J_QDRANT
        memory = await create_memory_backend(config)

        # Hyperspace (research)
        config = Config.fused()
        config.memory_backend = MemoryBackend.HYPERSPACE
        memory = await create_memory_backend(config)
    """
    backend = config.memory_backend

    # ========================================================================
    # Pure Strategies (Single Backend)
    # ========================================================================

    if backend == MemoryBackend.NETWORKX:
        if not NETWORKX_AVAILABLE:
            raise ValueError("NetworkX backend not available")
        return NetworkXKG()

    elif backend == MemoryBackend.NEO4J:
        if not NEO4J_AVAILABLE:
            raise ValueError("Neo4j backend not available. Install with: pip install neo4j")

        neo4j_config = Neo4jConfig(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database
        )
        return Neo4jKG(neo4j_config)

    elif backend == MemoryBackend.QDRANT:
        if not QDRANT_AVAILABLE:
            raise ValueError("Qdrant backend not available. Install with: pip install qdrant-client")

        return QdrantMemoryStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection=config.qdrant_collection,
            use_https=config.qdrant_use_https
        )

    elif backend == MemoryBackend.MEM0:
        if not MEM0_AVAILABLE:
            raise ValueError("Mem0 backend not available. Install with: pip install mem0ai")

        return Mem0Adapter(
            api_key=config.mem0_api_key,
            org_id=config.mem0_org_id,
            project_id=config.mem0_project_id,
            user_id=user_id
        )

    # ========================================================================
    # Hybrid Strategies (Multiple Backends)
    # ========================================================================

    elif backend == MemoryBackend.NEO4J_QDRANT:
        # Most common production setup
        neo4j_store = None
        qdrant_store = None

        if NEO4J_AVAILABLE:
            neo4j_config = Neo4jConfig(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            )
            neo4j_store = Neo4jKG(neo4j_config)

        if QDRANT_AVAILABLE:
            qdrant_store = QdrantMemoryStore(
                host=config.qdrant_host,
                port=config.qdrant_port,
                collection=config.qdrant_collection,
                use_https=config.qdrant_use_https
            )

        if not neo4j_store and not qdrant_store:
            raise ValueError("Neither Neo4j nor Qdrant backend available")

        return HybridMemoryStore(
            neo4j_store=neo4j_store,
            qdrant_store=qdrant_store,
            strategy="balanced"
        )

    elif backend == MemoryBackend.TRIPLE:
        # Full hybrid: Neo4j + Qdrant + Mem0
        neo4j_store = None
        qdrant_store = None
        mem0_store = None

        if NEO4J_AVAILABLE:
            neo4j_config = Neo4jConfig(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            )
            neo4j_store = Neo4jKG(neo4j_config)

        if QDRANT_AVAILABLE:
            qdrant_store = QdrantMemoryStore(
                host=config.qdrant_host,
                port=config.qdrant_port,
                collection=config.qdrant_collection
            )

        if MEM0_AVAILABLE:
            mem0_store = Mem0Adapter(
                api_key=config.mem0_api_key,
                user_id=user_id
            )

        available_backends = sum([
            neo4j_store is not None,
            qdrant_store is not None,
            mem0_store is not None
        ])

        if available_backends == 0:
            raise ValueError("No backends available for TRIPLE strategy")

        return HybridMemoryStore(
            neo4j_store=neo4j_store,
            qdrant_store=qdrant_store,
            mem0_store=mem0_store,
            strategy="balanced"
        )

    elif backend == MemoryBackend.HYPERSPACE:
        # Specialized: Gated multipass with recursive importance
        # TODO: Implement HyperspaceMemoryStore
        raise NotImplementedError(
            "HYPERSPACE backend not yet implemented. "
            "Use NEO4J_QDRANT for now."
        )

    else:
        raise ValueError(f"Unknown memory backend: {backend}")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_unified_memory(
    user_id: str = "default",
    enable_neo4j: bool = True,
    enable_qdrant: bool = True,
    enable_mem0: bool = False
) -> MemoryStore:
    """
    Convenience function to create unified memory with boolean flags.

    This is the backward-compatible interface used by MCP server.

    Args:
        user_id: User identifier
        enable_neo4j: Enable Neo4j backend
        enable_qdrant: Enable Qdrant backend
        enable_mem0: Enable Mem0 backend

    Returns:
        Configured memory backend
    """
    # Determine backend based on flags
    if enable_neo4j and enable_qdrant and enable_mem0:
        backend = MemoryBackend.TRIPLE
    elif enable_neo4j and enable_qdrant:
        backend = MemoryBackend.NEO4J_QDRANT
    elif enable_neo4j and enable_mem0:
        backend = MemoryBackend.NEO4J_MEM0
    elif enable_qdrant and enable_mem0:
        backend = MemoryBackend.QDRANT_MEM0
    elif enable_neo4j:
        backend = MemoryBackend.NEO4J
    elif enable_qdrant:
        backend = MemoryBackend.QDRANT
    elif enable_mem0:
        backend = MemoryBackend.MEM0
    else:
        backend = MemoryBackend.NETWORKX

    # Create config
    config = Config.fused()
    config.memory_backend = backend

    return await create_memory_backend(config, user_id)
