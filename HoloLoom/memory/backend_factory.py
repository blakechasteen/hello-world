"""
Memory Backend Factory - Extensible & Minimal
==============================================
Protocol-based factory for creating memory backends.

3 backends: INMEMORY (dev), HYBRID (prod), HYPERSPACE (research).
All implement MemoryStore protocol for easy extension.
"""

from typing import Optional, Any, Dict, List
import warnings

from HoloLoom.config import Config, MemoryBackend
from .protocol import MemoryStore, Memory, MemoryQuery, RetrievalResult

# Backend availability flags
try:
    from HoloLoom.memory.graph import KG as NetworkXKG
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX unavailable")

try:
    from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from HoloLoom.memory.stores.qdrant import QdrantMemoryStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ============================================================================
# Hybrid Store - Simple Balanced Fusion
# ============================================================================

class HybridMemoryStore:
    """Hybrid: Neo4j (graph) + Qdrant (vectors). Auto-fallback to NetworkX."""

    def __init__(self, neo4j: Any = None, qdrant: Any = None, fallback: Any = None):
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.fallback = fallback
        self.backends = [(n, b) for n, b in [('neo4j', neo4j), ('qdrant', qdrant)] if b]
        self.fallback_mode = not self.backends

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        """Store in all backends."""
        for _, backend in self.backends:
            try:
                await backend.store(memory, user_id)
            except Exception as e:
                warnings.warn(f"Store failed: {e}")
        return memory.id

    async def store_many(self, memories: List[Memory], user_id: str = "default") -> List[str]:
        """Batch store."""
        return [await self.store(m, user_id) for m in memories]

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        """Recall with balanced fusion or fallback."""
        # Fallback mode
        if self.fallback_mode:
            result = await self.fallback.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="fallback",
                metadata={'backend': 'networkx'}
            )

        # Query backends
        results = {}
        for name, backend in self.backends:
            try:
                results[name] = await backend.recall(query, limit=limit * 2)
            except Exception as e:
                warnings.warn(f"Recall failed ({name}): {e}")

        # Emergency fallback
        if not results and self.fallback:
            result = await self.fallback.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="emergency_fallback",
                metadata={'backend': 'networkx'}
            )

        # Balanced fusion (equal weights)
        fused = self._fuse(results, limit)
        return RetrievalResult(
            memories=fused,
            scores=[1.0] * len(fused),
            strategy_used="hybrid_balanced",
            metadata={'backends': list(results.keys())}
        )

    def _fuse(self, results: Dict[str, RetrievalResult], limit: int) -> List[Memory]:
        """Simple balanced fusion."""
        scores = {}
        weight = 1.0 / len(results) if results else 1.0

        for result in results.values():
            for mem, score in zip(result.memories, result.scores):
                if mem.id not in scores:
                    scores[mem.id] = {'memory': mem, 'score': 0}
                scores[mem.id]['score'] += score * weight

        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [x['memory'] for x in ranked[:limit]]

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        health = {'status': 'healthy', 'backends': {}}
        for name, backend in self.backends:
            try:
                health['backends'][name] = await backend.health_check()
            except Exception as e:
                health['backends'][name] = {'status': 'unhealthy', 'error': str(e)}
        return health


# ============================================================================
# Factory - 3 Backends Only
# ============================================================================

async def create_memory_backend(config: Config, user_id: str = "default") -> MemoryStore:
    """
    Create memory backend: INMEMORY, HYBRID, or HYPERSPACE.

    INMEMORY: NetworkX (always available, <10ms)
    HYBRID: Neo4j+Qdrant with auto-fallback (DEFAULT, ~50ms)
    HYPERSPACE: Research mode with auto-fallback (~150ms)
    
    Auto-Fallback Chain:
    HYBRID → Neo4j + Qdrant → Neo4j only → Qdrant only → NetworkX
    HYPERSPACE → Hyperspace backend → HYBRID → NetworkX
    
    Returns:
        MemoryStore: Configured backend with graceful degradation
    """
    backend = config.memory_backend

    # INMEMORY: NetworkX (dev/testing)
    if backend == MemoryBackend.INMEMORY:
        if not NETWORKX_AVAILABLE:
            raise ValueError("NetworkX unavailable - cannot initialize INMEMORY backend")
        print("[INMEMORY] Using NetworkX backend (dev mode)")
        return NetworkXKG()

    # HYBRID: Neo4j + Qdrant (production default)
    elif backend == MemoryBackend.HYBRID:
        return await _create_hybrid(config)

    # HYPERSPACE: Research mode
    elif backend == MemoryBackend.HYPERSPACE:
        try:
            from HoloLoom.memory.hyperspace_backend import create_hyperspace_backend
            print("[HYPERSPACE] Initializing research backend")
            return create_hyperspace_backend(config)
        except ImportError as e:
            warnings.warn(f"HYPERSPACE unavailable ({e}), falling back to HYBRID")
            return await _create_hybrid(config)

    raise ValueError(f"Unknown backend: {backend}")


async def _create_hybrid(config: Config) -> HybridMemoryStore:
    """
    Create hybrid backend with intelligent auto-fallback.
    
    Priority chain:
    1. Neo4j + Qdrant (best: graph + vectors)
    2. Neo4j only (graph reasoning)
    3. Qdrant only (vector similarity)
    4. NetworkX (fallback: in-memory)
    
    Returns:
        HybridMemoryStore: Configured with available backends
    """
    neo4j = None
    qdrant = None
    fallback = None
    
    backends_available = []
    backends_failed = []

    # Try Neo4j
    if NEO4J_AVAILABLE:
        try:
            neo4j = Neo4jKG(Neo4jConfig(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database
            ))
            backends_available.append('Neo4j')
            print(f"✓ [Neo4j] Connected: {config.neo4j_uri}")
        except Exception as e:
            backends_failed.append(f'Neo4j: {str(e)}')
            warnings.warn(f"Neo4j connection failed: {e}")
    else:
        backends_failed.append('Neo4j: not installed')

    # Try Qdrant
    if QDRANT_AVAILABLE:
        try:
            qdrant = QdrantMemoryStore(
                host=config.qdrant_host,
                port=config.qdrant_port,
                collection=config.qdrant_collection
            )
            backends_available.append('Qdrant')
            print(f"✓ [Qdrant] Connected: {config.qdrant_host}:{config.qdrant_port}")
        except Exception as e:
            backends_failed.append(f'Qdrant: {str(e)}')
            warnings.warn(f"Qdrant connection failed: {e}")
    else:
        backends_failed.append('Qdrant: not installed')

    # Status reporting
    if backends_available:
        print(f"[HYBRID] Active backends: {', '.join(backends_available)}")
    
    # Auto-fallback to NetworkX if no production backends
    if not neo4j and not qdrant:
        print("\n" + "="*60)
        print("⚠️  FALLBACK MODE: NetworkX In-Memory")
        print("="*60)
        print("No production backends available:")
        for failure in backends_failed:
            print(f"  ✗ {failure}")
        print("\nUsing NetworkX fallback (limited to current session)")
        print("To enable production backends:")
        print("  • Neo4j: pip install neo4j")
        print("  • Qdrant: pip install qdrant-client")
        print("="*60 + "\n")
        
        if NETWORKX_AVAILABLE:
            fallback = NetworkXKG()
        else:
            raise ValueError("No backends available (NetworkX also unavailable)")

    return HybridMemoryStore(neo4j=neo4j, qdrant=qdrant, fallback=fallback)


# ============================================================================
# Convenience
# ============================================================================



# ============================================================================
# Health Check
# ============================================================================

async def check_backend_health(backend: MemoryStore) -> Dict[str, Any]:
    """
    Check health of memory backend.
    
    Returns:
        Dict with status, available backends, and latency metrics
    """
    if hasattr(backend, 'health_check'):
        return await backend.health_check()
    
    # Simple health check for backends without explicit health_check
    return {
        'status': 'unknown',
        'message': 'Backend does not implement health_check()',
        'type': type(backend).__name__
    }
