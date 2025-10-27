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
except (ImportError, NameError) as e:
    MEM0_AVAILABLE = False
    warnings.warn(f"Mem0 backend not available: {e}")


# ============================================================================
# Hybrid Memory Store
# ============================================================================

class HybridMemoryStore:
    """
    Simplified Hybrid Memory Store (Neo4j + Qdrant).

    **SIMPLIFIED DESIGN (Task 1.3):**
    - Primary: Neo4j (graph) + Qdrant (vector)
    - Fallback: NetworkX (in-memory) if backends unavailable
    - Strategy: Always balanced (no complex routing)

    This is the "woven fabric" - two threads working together.
    """

    def __init__(
        self,
        neo4j_store: Optional[Any] = None,
        qdrant_store: Optional[Any] = None,
        networkx_store: Optional[Any] = None
    ):
        """
        Initialize hybrid memory store.

        Args:
            neo4j_store: Neo4j backend (optional, falls back to NetworkX)
            qdrant_store: Qdrant backend (optional, falls back to NetworkX)
            networkx_store: NetworkX fallback (always available)

        Note:
            If neither Neo4j nor Qdrant is available, uses NetworkX as fallback.
        """
        self.neo4j = neo4j_store
        self.qdrant = qdrant_store
        self.networkx = networkx_store

        # Build active backends list
        self.backends = []
        if self.neo4j:
            self.backends.append(('neo4j', self.neo4j))
        if self.qdrant:
            self.backends.append(('qdrant', self.qdrant))

        # Always use balanced strategy (simplified)
        self.strategy = "balanced"

        # Fallback mode: If no backends available, use NetworkX
        if not self.backends and not self.networkx:
            raise ValueError("HybridMemoryStore requires at least one backend")

        self.fallback_mode = len(self.backends) == 0

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
        limit: int = 10
    ) -> 'RetrievalResult':
        """
        Recall memories using simplified balanced fusion.

        **SIMPLIFIED (Task 1.3):**
        - Always uses balanced strategy (50/50 Neo4j + Qdrant)
        - Falls back to NetworkX if backends unavailable
        - No complex strategy selection

        Args:
            query: Memory query
            limit: Max results to return

        Returns:
            Fused retrieval results
        """
        # Fallback mode: Use NetworkX directly
        if self.fallback_mode:
            result = await self.networkx.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="networkx_fallback",
                metadata={'backends_used': ['networkx']}
            )

        # Query each backend
        results_by_backend = {}

        for name, backend in self.backends:
            try:
                result = await backend.recall(query, limit=limit * 2)  # Over-fetch for fusion
                results_by_backend[name] = result
            except Exception as e:
                warnings.warn(f"Failed to recall from {name}: {e}")

        # If all backends failed, fall back to NetworkX
        if not results_by_backend and self.networkx:
            warnings.warn("All backends failed, falling back to NetworkX")
            result = await self.networkx.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="networkx_fallback_emergency",
                metadata={'backends_used': ['networkx'], 'reason': 'backend_failures'}
            )

        # Balanced fusion: Equal weighting (simplified)
        weights = {name: 1.0 / len(self.backends) for name, _ in self.backends}

        # Fuse results
        fused_memories = self._fuse_results(results_by_backend, weights, limit)

        return RetrievalResult(
            memories=fused_memories,
            scores=[1.0] * len(fused_memories),  # TODO: Compute actual scores
            strategy_used="hybrid_balanced",
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
# Helper Functions
# ============================================================================

async def _create_hybrid_with_fallback(config: Config) -> MemoryStore:
    """
    Create hybrid backend with auto-fallback (Task 1.3).

    **Strategy:**
    1. Try Neo4j + Qdrant (production)
    2. Fallback to NetworkX if neither available

    Args:
        config: HoloLoom configuration

    Returns:
        HybridMemoryStore with appropriate backends
    """
    neo4j_store = None
    qdrant_store = None
    networkx_fallback = None

    # Try Neo4j
    if NEO4J_AVAILABLE:
        try:
            from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
            neo4j_config = Neo4jConfig(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            )
            neo4j_store = Neo4jKG(neo4j_config)
            print(f"[OK] Neo4j backend initialized ({config.neo4j_uri})")
        except Exception as e:
            warnings.warn(f"Neo4j initialization failed: {e}")
    else:
        warnings.warn("Neo4j not available (install: pip install neo4j)")

    # Try Qdrant
    if QDRANT_AVAILABLE:
        try:
            from HoloLoom.memory.stores.qdrant import QdrantMemoryStore
            qdrant_store = QdrantMemoryStore(
                host=config.qdrant_host,
                port=config.qdrant_port,
                collection=config.qdrant_collection,
                use_https=config.qdrant_use_https
            )
            print(f"[OK] Qdrant backend initialized ({config.qdrant_host}:{config.qdrant_port})")
        except Exception as e:
            warnings.warn(f"Qdrant initialization failed: {e}")
    else:
        warnings.warn("Qdrant not available (install: pip install qdrant-client)")

    # Fallback if both failed
    if not neo4j_store and not qdrant_store:
        warnings.warn(
            "⚠️  Neither Neo4j nor Qdrant available - using NetworkX fallback\n"
            "   For production, install backends:\n"
            "   pip install neo4j qdrant-client",
            RuntimeWarning
        )
        if NETWORKX_AVAILABLE:
            networkx_fallback = NetworkXKG()
            print("[OK] NetworkX fallback initialized (in-memory)")
        else:
            raise ValueError("No backends available (NetworkX, Neo4j, Qdrant all failed)")

    return HybridMemoryStore(
        neo4j_store=neo4j_store,
        qdrant_store=qdrant_store,
        networkx_store=networkx_fallback
    )


# ============================================================================
# Factory Function
# ============================================================================

async def create_memory_backend(config: Config, user_id: str = "default") -> MemoryStore:
    """
    Simplified Memory Backend Factory (Task 1.3).

    **SIMPLIFIED (3 backends only):**
    - INMEMORY: Fast in-memory (NetworkX) - Development/Testing
    - HYBRID: Neo4j + Qdrant with auto-fallback - Production (DEFAULT)
    - HYPERSPACE: Advanced research mode - Optional

    Args:
        config: HoloLoom configuration
        user_id: Default user ID for memory operations

    Returns:
        Configured memory backend with auto-fallback

    Examples:
        # Development (fast)
        config = Config.fast()  # Auto-uses INMEMORY
        memory = await create_memory_backend(config)

        # Production (recommended, DEFAULT)
        config = Config.fused()  # Auto-uses HYBRID
        memory = await create_memory_backend(config)

        # Research
        config.memory_backend = MemoryBackend.HYPERSPACE
        memory = await create_memory_backend(config)
    """
    backend = config.memory_backend

    # ========================================================================
    # SIMPLIFIED: 3 Core Backends (Task 1.3)
    # ========================================================================

    if backend == MemoryBackend.INMEMORY:
        # In-memory NetworkX (development/testing)
        if not NETWORKX_AVAILABLE:
            raise ValueError("NetworkX not available for INMEMORY backend")
        return NetworkXKG()

    elif backend == MemoryBackend.HYBRID:
        # **DEFAULT PRODUCTION: Neo4j + Qdrant with auto-fallback**
        return await _create_hybrid_with_fallback(config)

    elif backend == MemoryBackend.HYPERSPACE:
        # Research mode with gated multipass
        try:
            from HoloLoom.memory.hyperspace_backend import create_hyperspace_backend
            return create_hyperspace_backend(config)
        except ImportError:
            warnings.warn("HYPERSPACE not available, falling back to HYBRID")
            return await _create_hybrid_with_fallback(config)

    # ========================================================================
    # Legacy Backends (Auto-migrate with warning)
    # ========================================================================

    elif backend == MemoryBackend.NETWORKX:
        warnings.warn("NETWORKX is deprecated, use INMEMORY instead", DeprecationWarning)
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
    # Legacy Hybrid Backends (Auto-migrate to HYBRID)
    # ========================================================================

    elif backend == MemoryBackend.NEO4J_QDRANT:
        warnings.warn("NEO4J_QDRANT is deprecated, use HYBRID instead", DeprecationWarning)
        return await _create_hybrid_with_fallback(config)

    elif backend == MemoryBackend.TRIPLE:
        warnings.warn("TRIPLE is deprecated, use HYBRID instead (Mem0 removed for simplicity)", DeprecationWarning)
        return await _create_hybrid_with_fallback(config)

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
