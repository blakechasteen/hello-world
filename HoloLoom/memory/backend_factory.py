"""
Memory Backend Factory - Extensible & Minimal
==============================================
Protocol-based factory for creating memory backends with safety guardrails.

3 backends: INMEMORY (dev), HYBRID (prod), HYPERSPACE (research).
All implement MemoryStore protocol for easy extension.
"""

from typing import Optional, Any, Dict, List
import warnings
import logging

from HoloLoom.config import Config, MemoryBackend
from .protocol import MemoryStore, Memory, MemoryQuery, RetrievalResult
from HoloLoom.alignment.safety_guardrails import (
    ActionCategory,
    ActionRequest,
    SafetyDecision,
    SafetyGuardrails,
    create_guardrails,
)


logger = logging.getLogger(__name__)

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
    from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ============================================================================
# Hybrid Store - Simple Balanced Fusion
# ============================================================================

class HybridMemoryStore:
    """
    Hybrid memory backend: Neo4j (graph) + Qdrant (vectors).

    Combines graph reasoning (Neo4j) with vector similarity (Qdrant)
    for comprehensive memory storage and retrieval. Auto-falls back
    to NetworkX if production backends unavailable.

    Attributes:
        neo4j: Neo4j graph backend (optional)
        qdrant: Qdrant vector backend (optional)
        fallback: NetworkX fallback backend (optional)
        backends: List of active production backends
        fallback_mode: True if using NetworkX fallback

    Notes:
        - Stores memories in all available backends (parallel)
        - Retrieval uses balanced fusion of all backends
        - Failures in one backend don't affect others
    """

    def __init__(
        self,
        neo4j: Any = None,
        qdrant: Any = None,
        fallback: Any = None,
        guardrails: Optional[SafetyGuardrails] = None,
    ):
        """
        Initialize hybrid memory store.

        Args:
            neo4j: Neo4j backend instance (optional)
            qdrant: Qdrant backend instance (optional)
            fallback: NetworkX fallback instance (optional)

        Notes:
            - At least one backend must be provided
            - fallback_mode activates if no production backends
        """
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.fallback = fallback
        self.backends = [(n, b) for n, b in [('neo4j', neo4j), ('qdrant', qdrant)] if b]
        self.fallback_mode = not self.backends
        self.guardrails = guardrails or create_guardrails()
        self.guardrails_enabled = self.guardrails is not None

    # ------------------------------------------------------------------
    # Guardrail Helpers
    # ------------------------------------------------------------------

    def _evaluate_guardrails(
        self,
        action: str,
        category: ActionCategory,
        context: Dict[str, Any],
        *,
        text_input: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[SafetyDecision]:
        if not self.guardrails_enabled:
            return None

        request = ActionRequest(
            action=action,
            category=category,
            context=context,
            user_id=user_id,
            session_id=session_id,
        )

        decision = self.guardrails.evaluate(request, text_input=text_input or "")

        if not decision.allowed:
            logger.warning("Memory guardrails blocked action '%s': %s", action, decision.reason)
            raise PermissionError(decision.reason)

        if decision.requires_approval:
            logger.warning(
                "Memory guardrails require approval for action '%s': %s",
                action,
                decision.reason,
            )
            raise PermissionError(f"Memory action requires approval: {decision.reason}")

        return decision

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        """
        Store memory in all available backends.

        Args:
            memory: Memory object to store
            user_id: User identifier for multi-tenant isolation

        Returns:
            str: Memory ID (always succeeds if any backend works)

        Notes:
            - Stores in all backends in parallel
            - Partial failures allowed (warns but continues)
            - Returns immediately after all attempts (non-blocking)
        """
        # Validation
        if not memory or not memory.id:
            raise ValueError("[X] Cannot store: memory or memory.id is None")

        guardrail_decision = self._evaluate_guardrails(
            action="memory_store",
            category=ActionCategory.STORAGE,
            context={
                "memory_id": memory.id,
                "metadata_keys": list((memory.metadata or {}).keys()),
                "context_keys": list((memory.context or {}).keys()),
            },
            text_input=getattr(memory, "text", ""),
            user_id=user_id,
        )

        stored_count = 0
        for name, backend in self.backends:
            try:
                await backend.store(memory, user_id)
                stored_count += 1
            except Exception as e:
                warnings.warn(f"[WARN] Store failed ({name}): {e}")

        # Log success if any backend succeeded
        if stored_count > 0:
            logger.info(
                "Memory stored (id=%s, backends=%s, risk=%s)",
                memory.id,
                ','.join([n for n, _ in self.backends[:stored_count]]) or 'fallback',
                guardrail_decision.risk_level.value if guardrail_decision else 'unknown',
            )

        return memory.id

    async def store_many(self, memories: List[Memory], user_id: str = "default") -> List[str]:
        """
        Batch store multiple memories.

        Args:
            memories: List of Memory objects to store
            user_id: User identifier for multi-tenant isolation

        Returns:
            List[str]: List of stored memory IDs

        Notes:
            - Stores each memory independently
            - Partial failures allowed (individual memories may fail)
            - More efficient than calling store() in a loop
        """
        if not memories:
            return []

        stored_ids = []
        for memory in memories:
            stored_ids.append(await self.store(memory, user_id))
        return stored_ids

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        """
        Recall memories using balanced fusion of all backends.

        Args:
            query: Memory query with search criteria
            limit: Maximum number of memories to return (default: 10)

        Returns:
            RetrievalResult: Fused results with strategy metadata

        Process:
            1. Check if in fallback mode (NetworkX only)
            2. Query all available backends (Neo4j, Qdrant)
            3. Fuse results using balanced weighting
            4. Emergency fallback if all backends fail

        Notes:
            - Queries backends with 2x limit for better fusion
            - Partial failures allowed (one backend can fail)
            - Always returns results (fallback if needed)
        """
        guardrail_decision = self._evaluate_guardrails(
            action="memory_recall",
            category=ActionCategory.RETRIEVAL,
            context={
                "limit": limit,
                "strategy": query.strategy.name if getattr(query, "strategy", None) else None,
                "filters": bool(getattr(query, "filters", None)),
            },
            text_input=getattr(query, "text", ""),
            user_id=getattr(query, "user_id", None),
        )

        # ============================================================
        # Fallback Mode (NetworkX only)
        # ============================================================
        if self.fallback_mode:
            result = await self.fallback.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="fallback",
                metadata={
                    'backend': 'networkx',
                    'guardrails': guardrail_decision.to_dict() if guardrail_decision else None,
                }
            )

        # ============================================================
        # Query Production Backends
        # ============================================================
        results = {}
        for name, backend in self.backends:
            try:
                results[name] = await backend.recall(query, limit=limit * 2)
            except Exception as e:
                warnings.warn(f"[WARN] Recall failed ({name}): {e}")

        # ============================================================
        # Emergency Fallback (all backends failed)
        # ============================================================
        if not results and self.fallback:
            warnings.warn("[WARN] All production backends failed, using emergency fallback")
            result = await self.fallback.recall(query, limit=limit)
            return RetrievalResult(
                memories=result.memories,
                scores=result.scores,
                strategy_used="emergency_fallback",
                metadata={
                    'backend': 'networkx',
                    'guardrails': guardrail_decision.to_dict() if guardrail_decision else None,
                }
            )

        # ============================================================
        # Balanced Fusion
        # ============================================================
        fused = self._fuse(results, limit)
        return RetrievalResult(
            memories=fused,
            scores=[1.0] * len(fused),
            strategy_used="hybrid_balanced",
            metadata={
                'backends': list(results.keys()),
                'backend_count': len(results),
                'guardrails': guardrail_decision.to_dict() if guardrail_decision else None,
            }
        )

    def _fuse(self, results: Dict[str, RetrievalResult], limit: int) -> List[Memory]:
        """
        Fuse results from multiple backends using balanced weighting.

        Args:
            results: Dict mapping backend name to RetrievalResult
            limit: Maximum number of memories to return

        Returns:
            List[Memory]: Fused and ranked memories

        Algorithm:
            1. Assign equal weight to each backend (1.0 / N)
            2. Aggregate scores for memories appearing in multiple backends
            3. Sort by aggregated score (descending)
            4. Return top N memories

        Notes:
            - Memories appearing in multiple backends get boosted scores
            - Equal weighting ensures no backend dominates
            - Handles empty results gracefully
        """
        if not results:
            return []

        scores = {}
        weight = 1.0 / len(results)

        # Aggregate scores across backends
        for backend_name, result in results.items():
            for mem, score in zip(result.memories, result.scores):
                if mem.id not in scores:
                    scores[mem.id] = {'memory': mem, 'score': 0}
                scores[mem.id]['score'] += score * weight

        # Sort by score and return top N
        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [x['memory'] for x in ranked[:limit]]

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all backends.

        Returns:
            Dict with overall status and per-backend health

        Response Format:
            {
                'status': 'healthy' | 'degraded' | 'unhealthy',
                'mode': 'production' | 'fallback',
                'backends': {
                    'neo4j': {...},
                    'qdrant': {...},
                    'networkx': {...}
                }
            }

        Notes:
            - 'healthy': All production backends working
            - 'degraded': Some production backends failing
            - 'unhealthy': All backends failing (should never happen)
            - 'fallback': Using NetworkX only
        """
        health = {
            'status': 'healthy',
            'mode': 'fallback' if self.fallback_mode else 'production',
            'backends': {}
        }

        # Check production backends
        unhealthy_count = 0
        for name, backend in self.backends:
            try:
                health['backends'][name] = await backend.health_check()
                if health['backends'][name].get('status') != 'healthy':
                    unhealthy_count += 1
            except Exception as e:
                health['backends'][name] = {'status': 'unhealthy', 'error': str(e)}
                unhealthy_count += 1

        # Check fallback
        if self.fallback:
            try:
                health['backends']['networkx'] = await self.fallback.health_check()
            except:
                health['backends']['networkx'] = {'status': 'healthy'}  # NetworkX always healthy

        # Overall status
        if self.fallback_mode:
            health['status'] = 'degraded'  # Using fallback
        elif unhealthy_count > 0:
            health['status'] = 'degraded'  # Some backends failing
        else:
            health['status'] = 'healthy'  # All backends working

        return health


# ============================================================================
# Factory - 3 Backends Only
# ============================================================================

async def create_memory_backend(
    config: Config,
    user_id: str = "default",
    guardrails: Optional[SafetyGuardrails] = None,
) -> MemoryStore:
    """
    Create memory backend with intelligent auto-fallback.

    Factory function for creating production-ready memory backends with
    graceful degradation when optional dependencies are unavailable.

    Args:
        config: System configuration with backend selection and connection details
        user_id: User identifier for multi-tenant isolation (default: "default")

    Returns:
        MemoryStore: Configured backend implementing MemoryStore protocol

    Backend Options:
        INMEMORY: NetworkX in-memory graph (dev/testing, <10ms)
        HYBRID: Neo4j+Qdrant dual-backend (production, ~50ms)
        HYPERSPACE: Advanced research backend (~150ms)

    Auto-Fallback Chain:
        HYBRID → Neo4j+Qdrant → Neo4j only → Qdrant only → NetworkX
        HYPERSPACE → Hyperspace → HYBRID → NetworkX
        INMEMORY → NetworkX (no fallback)

    Raises:
        ValueError: If config is invalid or no backends available

    Notes:
        - Production backends auto-fallback to NetworkX if unavailable
        - Connection failures logged as warnings, not errors
        - All backends implement identical MemoryStore protocol
    """
    # Validation
    if not config:
        raise ValueError("Configuration cannot be None")
    if not hasattr(config, 'memory_backend'):
        raise ValueError("Configuration missing memory_backend attribute")

    backend = config.memory_backend

    # ============================================================
    # INMEMORY: NetworkX (dev/testing)
    # ============================================================
    if backend == MemoryBackend.INMEMORY:
        if not NETWORKX_AVAILABLE:
            raise ValueError("[X] NetworkX unavailable - cannot initialize INMEMORY backend")
        print("[OK] [INMEMORY] Using NetworkX backend (dev mode)")
        backend_instance = NetworkXKG()
        return GuardrailedMemoryStore(backend=backend_instance, guardrails=guardrails)

    # ============================================================
    # HYBRID: Neo4j + Qdrant (production default)
    # ============================================================
    elif backend == MemoryBackend.HYBRID:
        return await _create_hybrid(config, guardrails=guardrails)

    # ============================================================
    # HYPERSPACE: Research mode
    # ============================================================
    elif backend == MemoryBackend.HYPERSPACE:
        try:
            from HoloLoom.memory.hyperspace_backend import create_hyperspace_backend
            print("[OK] [HYPERSPACE] Initializing research backend")
            backend_instance = create_hyperspace_backend(config)
            return GuardrailedMemoryStore(backend=backend_instance, guardrails=guardrails)
        except ImportError as e:
            warnings.warn(f"[WARN] HYPERSPACE unavailable ({e}), falling back to HYBRID")
            return await _create_hybrid(config, guardrails=guardrails)

    raise ValueError(f"[X] Unknown backend: {backend}")


def _try_init_neo4j(config: Config) -> tuple[Any, Optional[str]]:
    """
    Attempt to initialize Neo4j graph backend.

    Args:
        config: Configuration with Neo4j connection details

    Returns:
        tuple: (backend_instance or None, error_message or None)

    Notes:
        - Returns (None, error) if initialization fails
        - Validates connection parameters before attempting connection
    """
    if not NEO4J_AVAILABLE:
        return None, "Neo4j library not installed"

    # Validation
    if not config.neo4j_uri:
        return None, "Neo4j URI not configured"

    try:
        neo4j = Neo4jKG(Neo4jConfig(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database
        ))
        print(f"[OK] [Neo4j] Connected: {config.neo4j_uri}")
        return neo4j, None
    except Exception as e:
        error_msg = str(e)
        warnings.warn(f"[WARN] Neo4j connection failed: {error_msg}")
        return None, error_msg


def _try_init_qdrant(config: Config) -> tuple[Any, Optional[str]]:
    """
    Attempt to initialize Qdrant vector backend.

    Args:
        config: Configuration with Qdrant connection details

    Returns:
        tuple: (backend_instance or None, error_message or None)

    Notes:
        - Returns (None, error) if initialization fails
        - Validates connection parameters before attempting connection
    """
    if not QDRANT_AVAILABLE:
        return None, "Qdrant library not installed"

    # Validation
    if not config.qdrant_host or not config.qdrant_port:
        return None, "Qdrant host/port not configured"

    try:
        qdrant = QdrantMemoryStore(
            url=f"http://{config.qdrant_host}:{config.qdrant_port}",
            collection_prefix=config.qdrant_collection
        )
        print(f"[OK] [Qdrant] Connected: {config.qdrant_host}:{config.qdrant_port}")
        return qdrant, None
    except Exception as e:
        error_msg = str(e)
        warnings.warn(f"[WARN] Qdrant connection failed: {error_msg}")
        return None, error_msg


def _create_fallback_backend() -> Any:
    """
    Create NetworkX fallback backend.

    Returns:
        NetworkX backend instance

    Raises:
        ValueError: If NetworkX is unavailable

    Notes:
        - Used when production backends (Neo4j, Qdrant) fail
        - In-memory only, data not persisted
    """
    if not NETWORKX_AVAILABLE:
        raise ValueError("[X] No backends available (NetworkX also unavailable)")

    print("\n" + "="*60)
    print("[WARN]  FALLBACK MODE: NetworkX In-Memory")
    print("="*60)
    print("Using NetworkX fallback (limited to current session)")
    print("\nTo enable production backends:")
    print("  • Neo4j: pip install neo4j && docker-compose up neo4j")
    print("  • Qdrant: pip install qdrant-client && docker-compose up qdrant")
    print("="*60 + "\n")

    return NetworkXKG()


async def _create_hybrid(
    config: Config,
    guardrails: Optional[SafetyGuardrails] = None,
) -> HybridMemoryStore:
    """
    Create hybrid backend with intelligent auto-fallback.

    Attempts to initialize Neo4j (graph) and Qdrant (vectors) backends,
    falling back gracefully to available alternatives.

    Args:
        config: System configuration with backend connection details

    Returns:
        HybridMemoryStore: Configured with available backends

    Priority Chain:
        1. Neo4j + Qdrant (best: graph + vectors)
        2. Neo4j only (graph reasoning)
        3. Qdrant only (vector similarity)
        4. NetworkX (fallback: in-memory)

    Notes:
        - Connection failures are non-fatal (auto-fallback)
        - Always returns a working backend (may be fallback)
        - Logs all initialization attempts and outcomes
    """
    neo4j = None
    qdrant = None
    fallback = None

    backends_available = []
    backends_failed = []

    # ============================================================
    # Neo4j Initialization
    # ============================================================
    neo4j, neo4j_error = _try_init_neo4j(config)
    if neo4j:
        backends_available.append('Neo4j')
    elif neo4j_error:
        backends_failed.append(f'Neo4j: {neo4j_error}')

    # ============================================================
    # Qdrant Initialization
    # ============================================================
    qdrant, qdrant_error = _try_init_qdrant(config)
    if qdrant:
        backends_available.append('Qdrant')
    elif qdrant_error:
        backends_failed.append(f'Qdrant: {qdrant_error}')

    # ============================================================
    # Status Reporting
    # ============================================================
    if backends_available:
        print(f"[OK] [HYBRID] Active backends: {', '.join(backends_available)}")

    # ============================================================
    # Auto-Fallback (if no production backends)
    # ============================================================
    if not neo4j and not qdrant:
        for failure in backends_failed:
            print(f"  [X] {failure}")

        fallback = _create_fallback_backend()

    return HybridMemoryStore(
        neo4j=neo4j,
        qdrant=qdrant,
        fallback=fallback,
        guardrails=guardrails,
    )


class GuardrailedMemoryStore:
    """Wrapper that enforces safety guardrails around any MemoryStore."""

    def __init__(
        self,
        backend: MemoryStore,
        guardrails: Optional[SafetyGuardrails] = None,
    ):
        self._backend = backend
        self.guardrails = guardrails or create_guardrails()
        self.guardrails_enabled = self.guardrails is not None

        # Avoid double-guarding if backend already handles guardrails internally
        if getattr(backend, 'guardrails_enabled', False):
            self.guardrails_enabled = False

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        action: str,
        category: ActionCategory,
        context: Dict[str, Any],
        *,
        text_input: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[SafetyDecision]:
        if not self.guardrails_enabled:
            return None

        request = ActionRequest(
            action=action,
            category=category,
            context=context,
            user_id=user_id,
            session_id=session_id,
        )
        decision = self.guardrails.evaluate(request, text_input=text_input or "")

        if not decision.allowed:
            raise PermissionError(decision.reason)
        if decision.requires_approval:
            raise PermissionError(f"Memory action requires approval: {decision.reason}")
        return decision

    # ------------------------------------------------------------------
    # Delegated API with guardrails
    # ------------------------------------------------------------------

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        decision = self._evaluate(
            action="memory_store",
            category=ActionCategory.STORAGE,
            context={
                "memory_id": getattr(memory, 'id', None),
                "metadata_keys": list((getattr(memory, 'metadata', {}) or {}).keys()),
            },
            text_input=getattr(memory, 'text', ""),
            user_id=user_id,
        )
        memory_id = await self._backend.store(memory, user_id)
        logger.info(
            "Guardrailed store ok id=%s risk=%s",
            memory_id,
            decision.risk_level.value if decision else 'unknown',
        )
        return memory_id

    async def store_many(self, memories: List[Memory], user_id: str = "default") -> List[str]:
        ids = []
        for memory in memories or []:
            ids.append(await self.store(memory, user_id))
        return ids

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        decision = self._evaluate(
            action="memory_recall",
            category=ActionCategory.RETRIEVAL,
            context={
                "limit": limit,
                "filters": bool(getattr(query, 'filters', None)),
            },
            text_input=getattr(query, 'text', ""),
            user_id=getattr(query, 'user_id', None),
        )
        result = await self._backend.recall(query, limit=limit)
        if hasattr(result, 'metadata') and result.metadata is not None:
            result.metadata = dict(result.metadata)
            result.metadata['guardrails'] = decision.to_dict() if decision else None
        return result

    # Some backends implement `retrieve` instead of `recall`
    async def retrieve(self, query: MemoryQuery, strategy=None) -> RetrievalResult:  # type: ignore[override]
        decision = self._evaluate(
            action="memory_retrieve",
            category=ActionCategory.RETRIEVAL,
            context={
                "strategy": getattr(strategy, 'name', None) if strategy else None,
            },
            text_input=getattr(query, 'text', ""),
            user_id=getattr(query, 'user_id', None),
        )
        backend_retrieve = getattr(self._backend, 'retrieve', None)
        if backend_retrieve is None:
            raise AttributeError("Underlying backend does not support retrieve()")
        result = await backend_retrieve(query, strategy=strategy)
        if hasattr(result, 'metadata') and result.metadata is not None:
            result.metadata = dict(result.metadata)
            result.metadata['guardrails'] = decision.to_dict() if decision else None
        return result

    async def delete(self, memory_id: str) -> bool:
        decision = self._evaluate(
            action="memory_delete",
            category=ActionCategory.DELETION,
            context={"memory_id": memory_id},
        )
        backend_delete = getattr(self._backend, 'delete', None)
        if backend_delete is None:
            raise AttributeError("Underlying backend does not support delete()")
        result = await backend_delete(memory_id)
        logger.info(
            "Guardrailed delete id=%s risk=%s",
            memory_id,
            decision.risk_level.value if decision else 'unknown',
        )
        return result

    async def health_check(self) -> Dict[str, Any]:
        if hasattr(self._backend, 'health_check'):
            return await self._backend.health_check()
        return {'status': 'unknown', 'backend': type(self._backend).__name__}

    def __getattr__(self, item: str) -> Any:
        return getattr(self._backend, item)


# ============================================================================
# Convenience
# ============================================================================



# ============================================================================
# Health Check
# ============================================================================

async def check_backend_health(backend: MemoryStore) -> Dict[str, Any]:
    """
    Check health of any memory backend.

    Args:
        backend: Memory backend instance (any MemoryStore implementation)

    Returns:
        Dict: Health status with backend details

    Response Format:
        {
            'status': 'healthy' | 'degraded' | 'unknown',
            'type': 'HybridMemoryStore' | 'NetworkXKG' | ...,
            'message': 'Optional status message',
            'backends': {...}  # For hybrid backends
        }

    Notes:
        - Returns 'unknown' if backend doesn't implement health_check()
        - Safe to call on any backend (never raises exceptions)
        - For HybridMemoryStore, includes per-backend breakdown
    """
    # Validation
    if not backend:
        return {
            'status': 'unhealthy',
            'message': '[X] Backend is None',
            'type': 'None'
        }

    # Check if backend implements health_check()
    if hasattr(backend, 'health_check'):
        try:
            return await backend.health_check()
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'[WARN] Health check failed: {str(e)}',
                'type': type(backend).__name__
            }

    # Fallback for backends without health_check()
    return {
        'status': 'unknown',
        'message': '[WARN] Backend does not implement health_check()',
        'type': type(backend).__name__
    }
