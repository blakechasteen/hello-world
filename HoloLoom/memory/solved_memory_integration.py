"""
Solved Memory Integration
=========================

Integrates the 5 Pillars of Solved Memory into HoloLoom's WeavingOrchestrator.

**5 Pillars:**
1. Bounded Growth - Hard limits on KG with eviction (Phase 1)
2. Unified Forgetting - Centralized ForgetManager (Phase 2)
3. Outcome→Retrieval Loop - Boost helpful shards (Phase 3)
4. Delta Storage - Efficient reconstruction (Phase 4)
5. Anticipatory Retrieval - Predict & prefetch (Phase 5)

**Usage:**
    from HoloLoom.memory.solved_memory_integration import (
        SolvedMemoryIntegration,
        create_solved_memory_integration,
    )

    # Create integration
    integration = create_solved_memory_integration(orchestrator)

    # Process query with all 5 pillars active
    spacetime = await integration.weave_with_solved_memory(query)

**Created:** December 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Callable, Protocol, Set, runtime_checkable
from datetime import datetime
import asyncio
import logging
import time

# Phase 1: Bounded Growth (already in graph.py)
from HoloLoom.memory.graph import KG, EvictionStrategy, LifecycleScope

# Phase 2: Unified Forgetting
from HoloLoom.memory.forget_manager import (
    ForgetManager,
    ForgetConfig,
    ForgetPolicy,
    create_forget_manager,
)

# Phase 3: Outcome→Retrieval Loop
from HoloLoom.memory.shard_contribution import (
    ShardContributionTracker,
    ShardContributionConfig,
    RetrievalBooster,
    create_contribution_tracker,
    create_retrieval_booster,
)
from HoloLoom.memory.contribution_integration import (
    ContributionIntegration,
    create_contribution_integration,
)

# Phase 4: Delta Storage
from HoloLoom.memory.delta_storage import (
    DeltaStore,
    DeltaReconstructor,
    DeltaStoreConfig,
    DeltaOperation,
    DeltaTarget,
    create_delta_store,
    create_reconstructor,
)

# Phase 5: Anticipatory Retrieval
from HoloLoom.memory.anticipatory_retrieval import (
    AnticipatoryRetrieval,
    AnticipatoryConfig,
    SessionContext,
    QueryType,
    create_anticipatory_retrieval,
    create_session_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pillar Protocol (for extensibility)
# =============================================================================

@runtime_checkable
class Pillar(Protocol):
    """
    Protocol for memory pillars (extensibility point).

    Each pillar implements this protocol to enable:
    - Uniform lifecycle management (initialize/shutdown)
    - Consistent hook interface (pre/post retrieval)
    - Statistics collection
    - Easy addition of new pillars

    Example of adding a new pillar:

        class Phase6_ContextCompression:
            '''Pillar 6: Context Compression for token optimization.'''

            def initialize(self, config: Dict[str, Any]) -> None:
                self.compressor = create_compressor(config)

            def shutdown(self) -> None:
                self.compressor.flush()

            def pre_retrieval(self, query: str) -> Dict[str, Any]:
                return {'compression_hint': self.compressor.predict_ratio(query)}

            def post_retrieval(self, results: List[Any]) -> List[Any]:
                return self.compressor.compress_results(results)

            def get_stats(self) -> Dict[str, Any]:
                return {'compression_ratio': self.compressor.avg_ratio}
    """

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the pillar with configuration."""
        ...

    def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        ...

    def pre_retrieval(self, query: str) -> Dict[str, Any]:
        """Hook called before retrieval. Returns metadata to merge into result."""
        ...

    def post_retrieval(self, results: List[Any]) -> List[Any]:
        """Hook called after retrieval. Can modify/filter results."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return pillar-specific statistics."""
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SolvedMemoryConfig:
    """Configuration for the 5 Pillars of Solved Memory."""

    # Phase 1: Bounded Growth
    enable_bounded_growth: bool = True
    kg_max_nodes: int = 10000
    kg_max_edges: int = 50000
    kg_eviction_strategy: EvictionStrategy = EvictionStrategy.LRU

    # Phase 2: Unified Forgetting
    enable_forgetting: bool = True
    forget_interval_seconds: float = 3600.0  # 1 hour
    forget_policy: ForgetPolicy = ForgetPolicy.IMPORTANCE

    # Phase 3: Outcome→Retrieval Loop
    enable_contribution_tracking: bool = True
    contribution_boost_enabled: bool = True
    contribution_decay_rate: float = 0.95

    # Phase 4: Delta Storage
    enable_delta_storage: bool = True
    delta_checkpoint_interval: int = 100  # Checkpoint every 100 deltas
    delta_max_checkpoints: int = 10

    # Phase 5: Anticipatory Retrieval
    enable_anticipatory: bool = True
    anticipatory_prefetch_enabled: bool = True
    anticipatory_max_predictions: int = 3
    anticipatory_prefetch_k: int = 10

    @classmethod
    def default(cls) -> 'SolvedMemoryConfig':
        """Default configuration with all pillars enabled."""
        return cls()

    @classmethod
    def minimal(cls) -> 'SolvedMemoryConfig':
        """Minimal configuration for testing."""
        return cls(
            enable_forgetting=False,  # Don't run background forgetting in tests
            forget_interval_seconds=float('inf'),
            delta_checkpoint_interval=10,
            anticipatory_prefetch_enabled=False,
        )

    @classmethod
    def production(cls) -> 'SolvedMemoryConfig':
        """Production configuration with tuned parameters."""
        return cls(
            kg_max_nodes=50000,
            kg_max_edges=200000,
            forget_interval_seconds=1800.0,  # 30 minutes
            delta_checkpoint_interval=500,
            delta_max_checkpoints=20,
            anticipatory_max_predictions=5,
            anticipatory_prefetch_k=15,
        )


@dataclass
class SolvedMemoryStats:
    """Statistics for the 5 Pillars of Solved Memory."""

    # Overall
    total_queries: int = 0
    total_weave_time_ms: float = 0.0

    # Phase 1: Bounded Growth
    kg_nodes: int = 0
    kg_edges: int = 0
    kg_evictions: int = 0

    # Phase 2: Unified Forgetting
    forget_runs: int = 0
    total_forgotten: int = 0

    # Phase 3: Outcome→Retrieval Loop
    outcomes_recorded: int = 0
    avg_boost_factor: float = 1.0

    # Phase 4: Delta Storage
    deltas_recorded: int = 0
    checkpoints_created: int = 0

    # Phase 5: Anticipatory Retrieval
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_queries': self.total_queries,
            'avg_weave_time_ms': self.total_weave_time_ms / max(1, self.total_queries),
            'phase1_bounded_growth': {
                'kg_nodes': self.kg_nodes,
                'kg_edges': self.kg_edges,
                'evictions': self.kg_evictions,
            },
            'phase2_unified_forgetting': {
                'forget_runs': self.forget_runs,
                'total_forgotten': self.total_forgotten,
            },
            'phase3_outcome_retrieval': {
                'outcomes_recorded': self.outcomes_recorded,
                'avg_boost_factor': self.avg_boost_factor,
            },
            'phase4_delta_storage': {
                'deltas_recorded': self.deltas_recorded,
                'checkpoints_created': self.checkpoints_created,
            },
            'phase5_anticipatory': {
                'prefetch_hits': self.prefetch_hits,
                'prefetch_misses': self.prefetch_misses,
                'hit_rate': self.prefetch_hit_rate,
            },
        }


class SolvedMemoryIntegration:
    """
    Integrates the 5 Pillars of Solved Memory into WeavingOrchestrator.

    This class wraps memory operations to apply all 5 pillars:
    - Pre-retrieval: Anticipatory prefetching (Phase 5)
    - Retrieval: Contribution boosting (Phase 3)
    - Post-retrieval: Outcome recording (Phase 3)
    - Background: Forgetting (Phase 2), Delta storage (Phase 4)
    - Always: Bounded growth enforcement (Phase 1)
    """

    def __init__(
        self,
        config: Optional[SolvedMemoryConfig] = None,
        kg: Optional[KG] = None,
        memory_backend: Optional[Any] = None,
    ):
        """
        Initialize the Solved Memory Integration.

        Args:
            config: Configuration for all 5 pillars
            kg: Knowledge Graph instance (Phase 1 bounded growth)
            memory_backend: Memory backend for retrieval operations
        """
        self.config = config or SolvedMemoryConfig.default()
        self.kg = kg
        self.memory_backend = memory_backend
        self.stats = SolvedMemoryStats()

        # Phase 2: Unified Forgetting
        self.forget_manager: Optional[ForgetManager] = None
        self._forget_task: Optional[asyncio.Task] = None

        # Phase 3: Outcome→Retrieval Loop
        self.contribution_tracker: Optional[ShardContributionTracker] = None
        self.retrieval_booster: Optional[RetrievalBooster] = None
        self.contribution_integration: Optional[ContributionIntegration] = None

        # Phase 4: Delta Storage
        self.delta_store: Optional[DeltaStore] = None
        self.delta_reconstructor: Optional[DeltaReconstructor] = None

        # Phase 5: Anticipatory Retrieval
        self.anticipatory: Optional[AnticipatoryRetrieval] = None
        self.session_context: Optional[SessionContext] = None

        # Query tracking for outcome recording
        self._current_query_id: Optional[str] = None
        self._current_retrieved_shards: List[str] = []

        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # =========================================================================
    # Phase Initialization Methods (called from initialize())
    # =========================================================================

    def _init_phase1_bounded_growth(self) -> None:
        """Initialize Phase 1: Bounded Growth via KG limits."""
        if not self.config.enable_bounded_growth or not self.kg:
            return

        self.kg.configure_limits(
            max_nodes=self.config.kg_max_nodes,
            max_edges=self.config.kg_max_edges,
            eviction_strategy=self.config.kg_eviction_strategy,
            enforce=True,
        )
        self.logger.info(f"  [Phase 1] Bounded Growth: {self.kg.limits}")

    def _init_phase2_unified_forgetting(self) -> None:
        """Initialize Phase 2: Unified Forgetting manager."""
        if not self.config.enable_forgetting:
            return

        forget_config = ForgetConfig(
            kg_eviction_policy=self.config.forget_policy,
            enable_kg_eviction=True,
            enable_hot_decay=True,
            enable_lifecycle_ttl=True,
        )
        self.forget_manager = create_forget_manager(forget_config)

        if self.kg:
            self.forget_manager.register_kg(self.kg)

        self.logger.info(
            f"  [Phase 2] Unified Forgetting: policy={self.config.forget_policy.value}, "
            f"interval={self.config.forget_interval_seconds}s"
        )

    def _init_phase3_outcome_retrieval(self) -> None:
        """Initialize Phase 3: Outcome→Retrieval contribution tracking."""
        if not self.config.enable_contribution_tracking:
            return

        self.contribution_tracker = create_contribution_tracker(
            max_records=10000,
            enable_decay=True,
            decay_rate=self.config.contribution_decay_rate,
        )
        self.retrieval_booster = create_retrieval_booster(self.contribution_tracker)
        self.contribution_integration = create_contribution_integration(None)

        self.logger.info(
            f"  [Phase 3] Outcome→Retrieval: boost_enabled={self.config.contribution_boost_enabled}, "
            f"decay_rate={self.config.contribution_decay_rate}"
        )

    def _init_phase4_delta_storage(self) -> None:
        """Initialize Phase 4: Delta Storage for efficient snapshots."""
        if not self.config.enable_delta_storage:
            return

        self.delta_store = create_delta_store(
            checkpoint_interval=self.config.delta_checkpoint_interval,
        )
        self.delta_reconstructor = create_reconstructor(self.delta_store)

        self.logger.info(
            f"  [Phase 4] Delta Storage: checkpoint_interval={self.config.delta_checkpoint_interval}, "
            f"max_checkpoints={self.config.delta_max_checkpoints}"
        )

    def _init_phase5_anticipatory_retrieval(self) -> None:
        """Initialize Phase 5: Anticipatory Retrieval for query prediction."""
        if not self.config.enable_anticipatory:
            return

        import uuid

        anticipatory_config = AnticipatoryConfig(
            prefetch_enabled=self.config.anticipatory_prefetch_enabled,
            max_predictions=self.config.anticipatory_max_predictions,
            prefetch_k=self.config.anticipatory_prefetch_k,
        )
        self.anticipatory = create_anticipatory_retrieval(anticipatory_config)
        self.session_context = create_session_context(session_id=str(uuid.uuid4()))

        self.logger.info(
            f"  [Phase 5] Anticipatory Retrieval: prefetch={self.config.anticipatory_prefetch_enabled}, "
            f"max_predictions={self.config.anticipatory_max_predictions}"
        )

    # =========================================================================
    # Main Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize all 5 pillars."""
        if self._initialized:
            return

        self.logger.info("Initializing 5 Pillars of Solved Memory...")

        self._init_phase1_bounded_growth()
        self._init_phase2_unified_forgetting()
        self._init_phase3_outcome_retrieval()
        self._init_phase4_delta_storage()
        self._init_phase5_anticipatory_retrieval()

        self._initialized = True
        self.logger.info("5 Pillars of Solved Memory initialized successfully!")

    async def start_background_tasks(self) -> None:
        """Start background tasks (forgetting loop)."""
        if self.config.enable_forgetting and self.forget_manager:
            self._forget_task = asyncio.create_task(self._forgetting_loop())
            self.logger.info("Started background forgetting loop")

    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        if self._forget_task:
            self._forget_task.cancel()
            try:
                await self._forget_task
            except asyncio.CancelledError:
                pass
            self._forget_task = None
            self.logger.info("Stopped background forgetting loop")

    async def _forgetting_loop(self) -> None:
        """Background loop for unified forgetting."""
        while True:
            try:
                await asyncio.sleep(self.config.forget_interval_seconds)

                if self.forget_manager:
                    # forget_now() returns Dict[str, ForgetEvent]
                    events = await self.forget_manager.forget_now()
                    total_forgotten = sum(e.items_affected for e in events.values())
                    self.stats.forget_runs += 1
                    self.stats.total_forgotten += total_forgotten

                    if total_forgotten > 0:
                        self.logger.info(f"Forgetting cycle: removed {total_forgotten} items")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in forgetting loop: {e}")

    def pre_retrieval(self, query_text: str) -> Dict[str, Any]:
        """
        Pre-retrieval hook: Check if query was predicted (Phase 5).

        Note: "Prefetch" in this system means predicted queries have their
        memory IDs stored for priority boosting during retrieval. The actual
        shard content is NOT cached - we use the IDs to boost relevance scores
        when those shards appear in normal retrieval results.

        Returns:
            Dict with:
                - prefetch_hit: bool - True if query was predicted
                - prefetched_ids: Set[str] - IDs to boost in retrieval
        """
        result = {
            'prefetch_hit': False,
            'prefetched_ids': set(),
        }

        if not self.anticipatory or not self.session_context:
            return result

        # Check if this query was predicted by a previous cycle
        was_predicted = self.anticipatory.record_prediction_hit(
            self.session_context,
            query_text,
        )

        if was_predicted:
            result['prefetch_hit'] = True
            result['prefetched_ids'] = self.session_context.prefetched_memory_ids or set()
            self.stats.prefetch_hits += 1
        else:
            self.stats.prefetch_misses += 1

        # Update hit rate
        total = self.stats.prefetch_hits + self.stats.prefetch_misses
        self.stats.prefetch_hit_rate = self.stats.prefetch_hits / max(1, total)

        return result

    def boost_retrieval_results(
        self,
        results: List[Any],
        query_id: Optional[str] = None,
        prefetched_ids: Optional[Set[str]] = None,
    ) -> List[Any]:
        """
        Boost retrieval results based on contribution history (Phase 3) and
        prefetch predictions (Phase 5).

        Args:
            results: List of (shard, score) tuples
            query_id: Optional query ID for tracking
            prefetched_ids: Optional set of IDs predicted by Phase 5 to boost

        Returns:
            Boosted and re-ranked results
        """
        # Phase 5: Apply prefetch boost (1.2x for predicted shards)
        if prefetched_ids:
            prefetch_boost = 1.2
            results = [
                (shard, score * prefetch_boost if getattr(shard, 'id', str(i)) in prefetched_ids else score)
                for i, (shard, score) in enumerate(results)
            ]

        if not self.retrieval_booster or not self.config.enable_contribution_tracking:
            return results

        # Track query for later outcome recording
        self._current_query_id = query_id or f"query_{int(time.time() * 1000)}"
        self._current_retrieved_shards = [
            getattr(shard, 'id', str(i)) for i, (shard, _) in enumerate(results)
        ]

        # Record retrieval in contribution tracker (required before record_outcome)
        if self.contribution_tracker and self._current_retrieved_shards:
            retrieval_scores = {
                getattr(shard, 'id', str(i)): score
                for i, (shard, score) in enumerate(results)
            }
            self.contribution_tracker.record_retrieval(
                query_text=query_id or "",
                shard_ids=self._current_retrieved_shards,
                retrieval_scores=retrieval_scores,
                query_id=self._current_query_id,
            )

        # Boost results
        boosted = self.retrieval_booster.boost_results(results)

        # Calculate average boost factor
        if boosted:
            boost_factors = [
                self.contribution_tracker.get_boost_factor(
                    getattr(shard, 'id', str(i))
                )
                for i, (shard, _) in enumerate(boosted)
            ]
            self.stats.avg_boost_factor = sum(boost_factors) / len(boost_factors)

        return boosted

    async def record_outcome(
        self,
        confidence: float,
        feedback: Optional[float] = None,
        query_id: Optional[str] = None
    ) -> None:
        """
        Record query outcome for contribution tracking (Phase 3).

        Args:
            confidence: Query confidence [0, 1]
            feedback: Optional user feedback [0, 1]
            query_id: Query ID (uses current if not provided)
        """
        if not self.contribution_tracker:
            return

        query_id = query_id or self._current_query_id
        if not query_id:
            return

        # Record outcome on the tracker (matches record_retrieval's query_id)
        self.contribution_tracker.record_outcome(
            query_id=query_id,
            confidence=confidence,
            feedback=feedback,
        )

        self.stats.outcomes_recorded += 1

    # =========================================================================
    # Delta Operation Dispatch (Phase 4)
    # =========================================================================

    def _dispatch_delta_add(
        self, target_id: str, target_type: DeltaTarget,
        old_state: Optional[Dict], new_state: Optional[Dict], source: str
    ) -> None:
        """Dispatch handler for ADD operations."""
        self.delta_store.record_add(
            target_id=target_id,
            target_type=target_type,
            state=new_state or {},
            source=source,
        )

    def _dispatch_delta_update(
        self, target_id: str, target_type: DeltaTarget,
        old_state: Optional[Dict], new_state: Optional[Dict], source: str
    ) -> None:
        """Dispatch handler for UPDATE operations."""
        self.delta_store.record_update(
            target_id=target_id,
            target_type=target_type,
            before=old_state or {},
            after=new_state or {},
            source=source,
        )

    def _dispatch_delta_delete(
        self, target_id: str, target_type: DeltaTarget,
        old_state: Optional[Dict], new_state: Optional[Dict], source: str
    ) -> None:
        """Dispatch handler for DELETE operations."""
        self.delta_store.record_delete(
            target_id=target_id,
            target_type=target_type,
            before=old_state or {},
            source=source,
        )

    @property
    def _delta_dispatch_table(self) -> Dict[DeltaOperation, Callable]:
        """Dispatch table mapping operations to handlers."""
        return {
            DeltaOperation.ADD: self._dispatch_delta_add,
            DeltaOperation.UPDATE: self._dispatch_delta_update,
            DeltaOperation.DELETE: self._dispatch_delta_delete,
        }

    def record_delta(
        self,
        operation: DeltaOperation,
        target_type: DeltaTarget,
        target_id: str,
        old_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        source: str = "integration",
    ) -> None:
        """
        Record a delta for state changes (Phase 4).

        Uses dispatch pattern to delegate to the correct handler:
        - ADD → record_add(target_id, target_type, state)
        - UPDATE → record_update(target_id, target_type, before, after)
        - DELETE → record_delete(target_id, target_type, before)

        Args:
            operation: ADD, UPDATE, or DELETE
            target_type: NODE, EDGE, EMBEDDING, etc.
            target_id: ID of the target
            old_state: Previous state (for UPDATE/DELETE)
            new_state: New state (for ADD/UPDATE)
            source: What caused this change
        """
        if not self.delta_store:
            return

        handler = self._delta_dispatch_table.get(operation)
        if handler:
            handler(target_id, target_type, old_state, new_state, source)
            self.stats.deltas_recorded += 1
        else:
            self.logger.warning(f"Unknown delta operation: {operation}")

    async def process_query(
        self,
        query_text: str,
        retrieval_fn: Callable,
    ) -> List[Any]:
        """
        Process a query through all 5 pillars.

        Args:
            query_text: The query text
            retrieval_fn: Async function to perform retrieval

        Returns:
            Retrieval results (boosted if Phase 3 enabled)
        """
        start_time = time.time()

        # Phase 5: Check for prefetch hit (get IDs for boosting)
        prefetch_result = self.pre_retrieval(query_text)
        prefetched_ids = prefetch_result.get('prefetched_ids', set())

        if prefetch_result['prefetch_hit']:
            self.logger.debug(
                f"Prefetch hit! {len(prefetched_ids)} memory IDs will be boosted"
            )

        # Always perform retrieval (prefetch provides boost, not cached content)
        results = await retrieval_fn(query_text)

        # Phase 3 + 5: Boost results based on contribution history AND prefetch predictions
        if isinstance(results, list) and results:
            if isinstance(results[0], tuple):
                # Already (shard, score) format
                results = self.boost_retrieval_results(results, prefetched_ids=prefetched_ids)
            else:
                # Convert to (shard, score) format
                results = [(r, 1.0) for r in results]
                results = self.boost_retrieval_results(results, prefetched_ids=prefetched_ids)

        # Phase 5: Predict follow-up queries and prefetch
        if self.anticipatory and self.session_context:
            # process_query is synchronous — returns predicted follow-ups
            predictions = self.anticipatory.process_query(
                self.session_context,
                query_text,
            )
            # Prefetch results for predicted queries (async)
            if predictions:
                await self.anticipatory.prefetch(
                    self.session_context,
                    predictions,
                )

        # Update stats
        self.stats.total_queries += 1
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.total_weave_time_ms += elapsed_ms

        # Update KG stats
        if self.kg:
            self.stats.kg_nodes = len(self.kg.G.nodes())
            self.stats.kg_edges = len(self.kg.G.edges())

        return results

    def get_stats(self) -> SolvedMemoryStats:
        """Get current statistics."""
        return self.stats

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return self.stats.to_dict()

    async def close(self) -> None:
        """Cleanup resources."""
        await self.stop_background_tasks()
        self.logger.info("Solved Memory Integration closed")

    async def __aenter__(self) -> 'SolvedMemoryIntegration':
        """Async context manager entry."""
        await self.initialize()
        await self.start_background_tasks()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def create_solved_memory_integration(
    config: Optional[SolvedMemoryConfig] = None,
    kg: Optional[KG] = None,
    memory_backend: Optional[Any] = None,
) -> SolvedMemoryIntegration:
    """
    Factory function to create a Solved Memory Integration.

    Args:
        config: Configuration (uses default if not provided)
        kg: Knowledge Graph instance
        memory_backend: Memory backend for retrieval

    Returns:
        Configured SolvedMemoryIntegration instance
    """
    return SolvedMemoryIntegration(
        config=config or SolvedMemoryConfig.default(),
        kg=kg,
        memory_backend=memory_backend,
    )


# Convenience function to attach to orchestrator
async def attach_solved_memory(
    orchestrator: Any,
    config: Optional[SolvedMemoryConfig] = None,
) -> SolvedMemoryIntegration:
    """
    Attach Solved Memory Integration to a WeavingOrchestrator.

    Args:
        orchestrator: WeavingOrchestrator instance
        config: Configuration for 5 pillars

    Returns:
        Initialized SolvedMemoryIntegration

    Usage:
        integration = await attach_solved_memory(orchestrator)
        # Now orchestrator has all 5 pillars active
    """
    # Get KG and memory backend from orchestrator
    kg = getattr(orchestrator, 'kg', None) or getattr(orchestrator, 'yarn_graph', None)
    memory_backend = getattr(orchestrator, 'memory', None)

    integration = create_solved_memory_integration(
        config=config,
        kg=kg,
        memory_backend=memory_backend,
    )

    await integration.initialize()
    await integration.start_background_tasks()

    # Attach to orchestrator
    orchestrator._solved_memory = integration

    return integration
