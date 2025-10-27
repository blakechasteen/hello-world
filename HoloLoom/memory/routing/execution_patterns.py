"""
Execution Patterns - How to Process Queries Through Backends
============================================================

The router doesn't just select WHICH backend, but HOW to execute:

1. FEED-FORWARD: Single pass, one backend
   Query → Route → Execute → Result

2. RECURSIVE: Multi-pass refinement
   Query → Route → Execute → Reflect → Re-route → Execute → Result

3. STRANGE LOOP: Self-referential processing
   Query → Route → Execute → Router reflects on own routing → Adjust → Result

4. CHAINING: Sequential pipeline through multiple backends
   Query → Backend1 → Backend2 → Backend3 → Fused Result

5. PARALLEL: Execute on multiple backends, fuse results
   Query → [Backend1, Backend2, Backend3] → Fusion → Result
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

from .protocol import RoutingDecision, RoutingOutcome, BackendType
from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult, Strategy


class ExecutionPattern(Enum):
    """Different ways to execute a query."""
    FEED_FORWARD = "feed_forward"      # Single pass
    RECURSIVE = "recursive"            # Multi-pass refinement
    STRANGE_LOOP = "strange_loop"      # Self-referential
    CHAIN = "chain"                    # Sequential pipeline
    PARALLEL = "parallel"              # Parallel + fusion


@dataclass
class ExecutionPlan:
    """
    Complete plan for executing a query.

    Includes:
    - Pattern: How to execute (feed-forward, recursive, etc.)
    - Backends: Which backends to use
    - Steps: Ordered execution steps
    - Metadata: Additional context
    """
    pattern: ExecutionPattern
    primary_backend: BackendType
    secondary_backends: List[BackendType]
    max_iterations: int = 1
    confidence_threshold: float = 0.7
    fusion_strategy: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExecutionEngine(Protocol):
    """
    Protocol for execution engines.

    Executes queries according to different patterns.
    """

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]  # Backend implementations
    ) -> RetrievalResult:
        """
        Execute query according to plan.

        Args:
            query: Query to execute
            plan: Execution plan
            backends: Available backend implementations

        Returns:
            Fused results from execution
        """
        ...


# ============================================================================
# Feed-Forward Engine (Baseline)
# ============================================================================

class FeedForwardEngine:
    """
    Simple feed-forward execution.

    Query → Route → Execute → Result

    Fastest, simplest, no refinement.
    """

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute query on primary backend."""

        primary = backends.get(plan.primary_backend)
        if not primary:
            raise ValueError(f"Backend {plan.primary_backend} not available")

        # Single pass execution
        result = await primary.retrieve(query, Strategy.FUSED)

        # Add execution metadata
        result.metadata['execution_pattern'] = 'feed_forward'
        result.metadata['backend'] = plan.primary_backend.value

        return result


# ============================================================================
# Recursive Refinement Engine
# ============================================================================

class RecursiveEngine:
    """
    Recursive refinement execution.

    Query → Execute → Reflect → Re-route → Execute → Result

    Iteratively improves results based on intermediate outcomes.
    """

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute with recursive refinement."""

        current_backend = plan.primary_backend
        all_memories = []
        iteration = 0

        while iteration < plan.max_iterations:
            # Execute on current backend
            backend = backends.get(current_backend)
            if not backend:
                break

            result = await backend.retrieve(query, Strategy.FUSED)
            all_memories.extend(result.memories)

            # Check if we're satisfied
            avg_score = sum(result.scores) / len(result.scores) if result.scores else 0

            if avg_score >= plan.confidence_threshold:
                # Good enough, stop
                break

            # Refine: Try different backend
            if iteration < len(plan.secondary_backends):
                current_backend = plan.secondary_backends[iteration]
            else:
                break

            iteration += 1

        # Deduplicate and return
        unique_memories = self._deduplicate(all_memories)

        return RetrievalResult(
            memories=unique_memories[:query.limit],
            scores=[1.0] * len(unique_memories[:query.limit]),
            strategy_used='recursive_refinement',
            metadata={
                'execution_pattern': 'recursive',
                'iterations': iteration + 1,
                'backends_used': [plan.primary_backend.value] +
                               [b.value for b in plan.secondary_backends[:iteration]]
            }
        )

    def _deduplicate(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories."""
        seen = set()
        unique = []

        for mem in memories:
            if mem.text not in seen:
                seen.add(mem.text)
                unique.append(mem)

        return unique


# ============================================================================
# Strange Loop Engine (Self-Referential)
# ============================================================================

class StrangeLoopEngine:
    """
    Strange loop execution (Hofstadter-inspired).

    Query → Execute → Router reflects on routing → Adjust → Execute → Result

    The router examines its own routing decision and adjusts.
    """

    def __init__(self, router_strategy):
        """
        Initialize with reference to routing strategy.

        Args:
            router_strategy: The routing strategy (for self-reference)
        """
        self.router = router_strategy

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute with self-referential routing."""

        # First pass
        primary = backends.get(plan.primary_backend)
        if not primary:
            raise ValueError(f"Backend {plan.primary_backend} not available")

        result_1 = await primary.retrieve(query, Strategy.FUSED)

        # Meta-query: "Should I have routed this query differently?"
        meta_query = f"Was {plan.primary_backend.value} the right choice for: {query.text}"

        # Router reflects on its own decision (STRANGE LOOP!)
        available = list(backends.keys())
        meta_decision = self.router.select_backend(
            meta_query,
            available,
            context={'first_pass_score': sum(result_1.scores) / len(result_1.scores) if result_1.scores else 0}
        )

        # If meta-routing suggests different backend, try it
        if meta_decision.backend_type != plan.primary_backend:
            secondary = backends.get(meta_decision.backend_type)
            result_2 = await secondary.retrieve(query, Strategy.FUSED)

            # Fuse results from both attempts
            fused = self._fuse_results(result_1, result_2)
            fused.metadata['strange_loop'] = True
            fused.metadata['meta_routing_triggered'] = True
            fused.metadata['backends'] = [
                plan.primary_backend.value,
                meta_decision.backend_type.value
            ]
            return fused

        # Meta-routing confirmed original choice
        result_1.metadata['strange_loop'] = True
        result_1.metadata['meta_routing_confirmed'] = True
        return result_1

    def _fuse_results(self, r1: RetrievalResult, r2: RetrievalResult) -> RetrievalResult:
        """Fuse two retrieval results."""
        # Simple fusion: combine and deduplicate
        all_mems = r1.memories + r2.memories
        seen = set()
        unique = []

        for mem in all_mems:
            if mem.text not in seen:
                seen.add(mem.text)
                unique.append(mem)

        return RetrievalResult(
            memories=unique,
            scores=[1.0] * len(unique),
            strategy_used='strange_loop_fusion',
            metadata={'source_strategies': [r1.strategy_used, r2.strategy_used]}
        )


# ============================================================================
# Chaining Engine (Pipeline)
# ============================================================================

class ChainingEngine:
    """
    Chaining execution - sequential pipeline.

    Query → Backend1 → refine → Backend2 → refine → Backend3 → Result

    Each backend refines the results from the previous.
    """

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute as chain/pipeline."""

        # Build chain: primary + secondaries
        chain = [plan.primary_backend] + plan.secondary_backends

        current_memories = []
        execution_trace = []

        for backend_type in chain:
            backend = backends.get(backend_type)
            if not backend:
                continue

            # Execute on this backend
            result = await backend.retrieve(query, Strategy.FUSED)

            # Accumulate memories
            current_memories.extend(result.memories)
            execution_trace.append({
                'backend': backend_type.value,
                'memories_added': len(result.memories),
                'avg_score': sum(result.scores) / len(result.scores) if result.scores else 0
            })

            # Deduplicate
            current_memories = self._deduplicate(current_memories)

        return RetrievalResult(
            memories=current_memories[:query.limit],
            scores=[1.0] * len(current_memories[:query.limit]),
            strategy_used='chaining',
            metadata={
                'execution_pattern': 'chain',
                'chain_length': len(chain),
                'execution_trace': execution_trace
            }
        )

    def _deduplicate(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicates."""
        seen = set()
        unique = []

        for mem in memories:
            if mem.text not in seen:
                seen.add(mem.text)
                unique.append(mem)

        return unique


# ============================================================================
# Parallel Engine (Concurrent Execution + Fusion)
# ============================================================================

class ParallelEngine:
    """
    Parallel execution with fusion.

    Query → [Backend1 || Backend2 || Backend3] → Fusion → Result

    Execute on multiple backends concurrently, fuse results.
    """

    async def execute(
        self,
        query: MemoryQuery,
        plan: ExecutionPlan,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute in parallel and fuse."""
        import asyncio

        # Backends to execute on
        backend_types = [plan.primary_backend] + plan.secondary_backends

        # Execute concurrently
        tasks = []
        for backend_type in backend_types:
            backend = backends.get(backend_type)
            if backend:
                tasks.append(backend.retrieve(query, Strategy.FUSED))

        # Wait for all
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if isinstance(r, RetrievalResult)]

        if not valid_results:
            return RetrievalResult(
                memories=[],
                scores=[],
                strategy_used='parallel_fusion',
                metadata={'error': 'all backends failed'}
            )

        # Fuse results (weighted by backend)
        fused_memories = self._fuse_parallel_results(valid_results, backend_types)

        return RetrievalResult(
            memories=fused_memories[:query.limit],
            scores=[1.0] * len(fused_memories[:query.limit]),
            strategy_used='parallel_fusion',
            metadata={
                'execution_pattern': 'parallel',
                'backends_used': [bt.value for bt in backend_types],
                'results_count': len(valid_results)
            }
        )

    def _fuse_parallel_results(
        self,
        results: List[RetrievalResult],
        backend_types: List[BackendType]
    ) -> List[Memory]:
        """Fuse results from parallel execution."""

        # Weighted fusion based on backend type
        weights = {
            BackendType.NEO4J: 0.3,
            BackendType.QDRANT: 0.4,
            BackendType.MEM0: 0.2,
            BackendType.INMEMORY: 0.1
        }

        # Score each memory by weighted backend
        memory_scores = {}

        for result, backend_type in zip(results, backend_types):
            weight = weights.get(backend_type, 0.25)

            for mem, score in zip(result.memories, result.scores):
                key = mem.text
                if key not in memory_scores:
                    memory_scores[key] = {'memory': mem, 'score': 0}

                memory_scores[key]['score'] += score * weight

        # Sort by fused score
        sorted_items = sorted(
            memory_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [item['memory'] for item in sorted_items]


# ============================================================================
# Execution Pattern Selector
# ============================================================================

def select_execution_pattern(
    query: MemoryQuery,
    routing_confidence: float,
    available_backends: List[BackendType]
) -> ExecutionPattern:
    """
    Intelligently select execution pattern.

    Logic:
    - High confidence + single backend → Feed-forward
    - Medium confidence → Recursive refinement
    - Low confidence → Parallel fusion
    - Complex query → Chaining
    - Meta-query → Strange loop
    """

    # Check for meta-queries (strange loop trigger)
    if any(word in query.text.lower() for word in ['why', 'should', 'best', 'optimal']):
        return ExecutionPattern.STRANGE_LOOP

    # High confidence: simple feed-forward
    if routing_confidence > 0.8:
        return ExecutionPattern.FEED_FORWARD

    # Multiple backends available: try parallel
    if len(available_backends) >= 3:
        return ExecutionPattern.PARALLEL

    # Medium confidence: recursive refinement
    if routing_confidence > 0.5:
        return ExecutionPattern.RECURSIVE

    # Low confidence: chain through multiple
    return ExecutionPattern.CHAIN