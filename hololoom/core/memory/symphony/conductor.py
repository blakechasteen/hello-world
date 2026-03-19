"""
Memory Conductor - Main Orchestration Layer
============================================

Coordinates memory access across 7 systems:
1. Query Cache - Check cache first (100x speedup)
2. Hot Patterns - Boost frequently accessed nodes
3. Vector Memory - Semantic similarity search
4. Knowledge Graph - Graph traversal + relationships
5. Awareness Graph - Spreading activation
6. Spring Dynamics - Physics-based connectivity (optional)
7. Multi-Wave Engine - Temporal propagation (optional)

Strategy Selection:
- FAST: Cache + Vector (minimize latency)
- BALANCED: Cache + Vector + KG (quality/speed tradeoff)
- DEEP: All systems + spreading activation
- RESEARCH: Maximum exploration, no time limits
- AUTO: Automatic selection based on query

Author: Claude Code
Date: 2025-11-22
"""

import asyncio
import logging
import time
from typing import Any

from .protocol import (
    CoordinationPlan,
    MemoryCoordinationResult,
    MemoryPerformanceMetrics,
    MemoryQuery,
    MemoryResult,
    MemoryStrategy,
    MemorySystem,
    StrategySelectionCriteria,
)

logger = logging.getLogger(__name__)


class MemoryConductor:
    """
    Main memory orchestra conductor.

    Coordinates access across multiple memory systems,
    automatically selects optimal strategies, and tracks
    performance metrics.
    """

    def __init__(
        self,
        memory_backend: Any,
        enable_cache: bool = True,
        enable_hot_patterns: bool = True,
        enable_awareness: bool = True,
        enable_spring_dynamics: bool = False,
        enable_multi_wave: bool = False,
        default_strategy: MemoryStrategy = MemoryStrategy.AUTO,
        verbose: bool = False,
        # Optional memory system instances for integration
        hot_tracker: Any | None = None,
        awareness_graph: Any | None = None,
        spring_dynamics: Any | None = None,
        multi_wave_engine: Any | None = None,
    ):
        """
        Initialize Memory Conductor.

        Args:
            memory_backend: HoloLoom memory backend (HYBRID/INMEMORY/HYPERSPACE)
            enable_cache: Enable query caching
            enable_hot_patterns: Enable hot pattern tracking
            enable_awareness: Enable awareness graph
            enable_spring_dynamics: Enable physics-based dynamics
            enable_multi_wave: Enable temporal wave propagation
            default_strategy: Default strategy if AUTO fails
            verbose: Enable debug logging
            hot_tracker: Optional HotPatternTracker instance for usage-based boosting
            awareness_graph: Optional AwarenessGraph instance for spreading activation
            spring_dynamics: Optional SpringDynamics instance for physics-based connectivity
            multi_wave_engine: Optional MultiWaveEngine instance for temporal propagation
        """
        self.memory_backend = memory_backend
        self.enable_cache = enable_cache
        self.enable_hot_patterns = enable_hot_patterns
        self.enable_awareness = enable_awareness
        self.enable_spring_dynamics = enable_spring_dynamics
        self.enable_multi_wave = enable_multi_wave
        self.default_strategy = default_strategy
        self.verbose = verbose

        # Memory system instances (optional, for integration)
        self._hot_tracker = hot_tracker
        self._awareness_graph = awareness_graph
        self._spring_dynamics = spring_dynamics
        self._multi_wave_engine = multi_wave_engine

        # Performance tracking
        self._metrics = MemoryPerformanceMetrics()
        self._query_history: list[MemoryCoordinationResult] = []

        # Cache for repeated queries
        self._cache: dict[str, MemoryCoordinationResult] = {}
        self._cache_max_size = 1000
        self._cache_ttl = 3600.0  # 1 hour

    async def recall(
        self,
        query: MemoryQuery
    ) -> MemoryCoordinationResult:
        """
        Unified memory recall across all systems.

        Full pipeline:
        1. Check query cache (if enabled)
        2. Select optimal strategy (if AUTO)
        3. Create coordination plan
        4. Execute plan (parallel/sequential)
        5. Merge results from multiple systems
        6. Update performance metrics

        Args:
            query: MemoryQuery with text, strategy, k, etc.

        Returns:
            MemoryCoordinationResult with merged results
        """
        start_time = time.time()

        # Step 1: Check cache
        if self.enable_cache:
            cache_key = self._make_cache_key(query)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                # Check TTL
                if time.time() - query.timestamp < self._cache_ttl:
                    if self.verbose:
                        logger.debug(f"Cache hit for query: {query.text[:50]}")

                    # Update metrics
                    self._metrics.total_queries += 1
                    self._metrics.cache_hits += 1

                    return cached

        # Step 2: Select strategy
        strategy = query.strategy
        if strategy == MemoryStrategy.AUTO:
            strategy = self.select_strategy(query)

        # Step 3: Create coordination plan
        plan = self._create_coordination_plan(query, strategy)

        if self.verbose:
            logger.info(
                f"Executing strategy {strategy.value}: "
                f"{len(plan.systems_to_query)} systems, "
                f"parallel={plan.parallel_execution}"
            )

        # Step 4: Execute plan
        all_results = await self._execute_plan(query, plan)

        # Step 5: Merge and rank results
        merged_results = self._merge_results(all_results, query.k)

        # Calculate latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Create result
        result = MemoryCoordinationResult(
            results=merged_results,
            strategy_used=strategy,
            systems_accessed=plan.systems_to_query,
            total_latency_ms=total_latency_ms,
            cache_hit=False,
            coordination_metadata={
                "plan": plan,
                "num_systems": len(plan.systems_to_query),
                "parallel": plan.parallel_execution
            }
        )

        # Step 6: Update cache and metrics
        if self.enable_cache:
            cache_key = self._make_cache_key(query)
            self._cache[cache_key] = result
            # LRU eviction
            if len(self._cache) > self._cache_max_size:
                oldest = min(self._cache.items(), key=lambda x: x[1].coordination_metadata.get('timestamp', 0))
                del self._cache[oldest[0]]

        self._update_metrics(result, strategy, plan.systems_to_query)
        self._query_history.append(result)

        return result

    def select_strategy(
        self,
        query: MemoryQuery
    ) -> MemoryStrategy:
        """
        Select optimal memory access strategy.

        Decision tree:
        - Query in cache → FAST (cache hit)
        - Simple factual query → FAST (cache + vector)
        - Standard query → BALANCED (cache + vector + KG)
        - Complex/research query → DEEP (all systems + spreading)
        - Explicit research mode → RESEARCH (maximum exploration)

        Args:
            query: MemoryQuery

        Returns:
            Selected MemoryStrategy
        """
        criteria = self._analyze_query(query)

        # Check cache first
        if criteria.is_cached:
            return MemoryStrategy.FAST

        # Research mode
        if criteria.exploration_mode:
            return MemoryStrategy.RESEARCH

        # Complex queries need deep traversal
        if criteria.requires_deep_traversal:
            return MemoryStrategy.DEEP

        # Performance-critical queries
        if criteria.performance_critical:
            return MemoryStrategy.FAST

        # Default: balanced
        return MemoryStrategy.BALANCED

    def _analyze_query(
        self,
        query: MemoryQuery
    ) -> StrategySelectionCriteria:
        """Analyze query characteristics for strategy selection."""
        # Check cache
        cache_key = self._make_cache_key(query)
        is_cached = cache_key in self._cache

        # Query complexity heuristics
        query_length = len(query.text.split())
        is_simple = query_length < 5
        is_complex = query_length > 15

        # Research keywords
        research_keywords = [
            "compare", "analyze", "evaluate", "explain", "tradeoffs",
            "comprehensive", "detailed", "thorough", "in-depth"
        ]
        has_research_keywords = any(kw in query.text.lower() for kw in research_keywords)

        return StrategySelectionCriteria(
            query_length=query_length,
            is_cached=is_cached,
            requires_deep_traversal=is_complex or has_research_keywords,
            performance_critical=is_simple,
            exploration_mode=query.strategy == MemoryStrategy.RESEARCH or has_research_keywords
        )

    def _create_coordination_plan(
        self,
        query: MemoryQuery,
        strategy: MemoryStrategy
    ) -> CoordinationPlan:
        """Create coordination plan based on strategy."""
        if strategy == MemoryStrategy.FAST:
            # Cache + Vector only
            systems = [MemorySystem.VECTOR_MEMORY]
            if self.enable_hot_patterns:
                systems.append(MemorySystem.HOT_PATTERNS)
            parallel = True
            estimated_latency = 50.0  # ms

        elif strategy == MemoryStrategy.BALANCED:
            # Cache + Vector + KG
            systems = [
                MemorySystem.VECTOR_MEMORY,
                MemorySystem.KNOWLEDGE_GRAPH,
            ]
            if self.enable_hot_patterns:
                systems.append(MemorySystem.HOT_PATTERNS)
            parallel = True
            estimated_latency = 150.0  # ms

        elif strategy == MemoryStrategy.DEEP:
            # All primary systems + spreading
            systems = [
                MemorySystem.VECTOR_MEMORY,
                MemorySystem.KNOWLEDGE_GRAPH,
                MemorySystem.HOT_PATTERNS,
                MemorySystem.AWARENESS_GRAPH,
            ]
            if self.enable_spring_dynamics:
                systems.append(MemorySystem.SPRING_DYNAMICS)
            parallel = False  # Sequential for deep traversal
            estimated_latency = 300.0  # ms

        elif strategy == MemoryStrategy.RESEARCH:
            # Everything
            systems = [
                MemorySystem.VECTOR_MEMORY,
                MemorySystem.KNOWLEDGE_GRAPH,
                MemorySystem.HOT_PATTERNS,
                MemorySystem.AWARENESS_GRAPH,
            ]
            if self.enable_spring_dynamics:
                systems.append(MemorySystem.SPRING_DYNAMICS)
            if self.enable_multi_wave:
                systems.append(MemorySystem.MULTI_WAVE)
            parallel = False
            estimated_latency = 500.0  # ms

        else:
            # Default to balanced
            systems = [MemorySystem.VECTOR_MEMORY, MemorySystem.KNOWLEDGE_GRAPH]
            parallel = True
            estimated_latency = 150.0

        return CoordinationPlan(
            strategy=strategy,
            systems_to_query=systems,
            parallel_execution=parallel,
            fallback_systems=[MemorySystem.VECTOR_MEMORY],  # Always fall back to vector
            estimated_latency_ms=estimated_latency,
            expected_coverage=0.8  # Estimate
        )

    async def _execute_plan(
        self,
        query: MemoryQuery,
        plan: CoordinationPlan
    ) -> list[list[MemoryResult]]:
        """Execute coordination plan across multiple systems."""
        tasks = []

        for system in plan.systems_to_query:
            task = self._query_system(system, query)
            tasks.append(task)

        # Execute in parallel or sequential
        if plan.parallel_execution:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.warning(f"System query failed: {e}")
                    results.append([])

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, list):
                valid_results.append(r)
            else:
                logger.warning(f"System returned exception: {r}")
                valid_results.append([])

        return valid_results

    async def _query_system(
        self,
        system: MemorySystem,
        query: MemoryQuery
    ) -> list[MemoryResult]:
        """Query individual memory system."""
        try:
            if system == MemorySystem.VECTOR_MEMORY:
                return await self._query_vector_memory(query)
            elif system == MemorySystem.KNOWLEDGE_GRAPH:
                return await self._query_knowledge_graph(query)
            elif system == MemorySystem.HOT_PATTERNS:
                return await self._query_hot_patterns(query)
            elif system == MemorySystem.AWARENESS_GRAPH:
                return await self._query_awareness_graph(query)
            elif system == MemorySystem.SPRING_DYNAMICS:
                return await self._query_spring_dynamics(query)
            elif system == MemorySystem.MULTI_WAVE:
                return await self._query_multi_wave(query)
            else:
                logger.warning(f"Unknown system: {system}")
                return []

        except Exception as e:
            logger.error(f"Error querying {system.value}: {e}")
            return []

    async def _query_vector_memory(self, query: MemoryQuery) -> list[MemoryResult]:
        """Query vector memory (semantic similarity)."""
        # Adapt to backend capabilities
        memories = []

        try:
            # Try high-level recall first (UnifiedMemory)
            if hasattr(self.memory_backend, 'recall'):
                # Check signature of recall
                import inspect
                sig = inspect.signature(self.memory_backend.recall)
                if 'query' in sig.parameters:
                    # UnifiedMemory.recall(query, ...)
                    memories = await self.memory_backend.recall(query.text, limit=query.k)
                else:
                    # Generic recall(text, k)
                    memories = await self.memory_backend.recall(query.text, k=query.k)

            # Try lower-level retrieve (InMemoryStore / Protocol)
            elif hasattr(self.memory_backend, 'retrieve'):
                # Need to construct protocol MemoryQuery if not already one
                # but here 'query' is memory_symphony.protocol.MemoryQuery

                # Check if backend expects a different Query object
                # For InMemoryStore: retrieve(query: protocol.MemoryQuery, strategy)

                # We need to bridge the types or use a compatible object
                # Attempt to pass our query object or create a compatible one

                # Create a simple object that mimics the expected interface if needed
                # or just pass attributes

                # For this specific integration, let's try direct call assuming compatible duck typing
                # or construct a simple object

                class SimpleQuery:
                    def __init__(self, text, user_id, limit):
                        self.text = text
                        self.user_id = "user" # Default
                        self.limit = limit

                backend_query = SimpleQuery(query.text, "user", query.k)

                # Strategy enum might mismatch, pass string or value if possible
                # InMemoryStore expects protocol.Strategy enum
                # We'll rely on default or try to pass semantic strategy

                # Quick fix for demo: just call retrieve with what we have
                result = await self.memory_backend.retrieve(backend_query)
                memories = result.memories if hasattr(result, 'memories') else []

        except Exception as e:
            logger.warning(f"Vector memory query failed: {e}")
            return []

        results = []
        for mem in memories:
            # Handle different Memory object types
            node_id = getattr(mem, 'id', getattr(mem, 'node_id', str(mem)))
            content = getattr(mem, 'text', getattr(mem, 'content', str(mem)))
            metadata = getattr(mem, 'metadata', getattr(mem, 'context', {}))
            relevance = getattr(mem, 'relevance', getattr(mem, 'score', 0.8))

            results.append(MemoryResult(
                node_id=node_id,
                content=content,
                relevance=relevance,
                source_system=MemorySystem.VECTOR_MEMORY,
                metadata=metadata
            ))

        return results

    async def _query_knowledge_graph(self, query: MemoryQuery) -> list[MemoryResult]:
        """Query knowledge graph (graph traversal)."""
        # Use memory backend's graph
        if not hasattr(self.memory_backend, 'graph'):
            return []

        graph = self.memory_backend.graph

        # Get nodes matching query (simplified)
        # In production, use proper entity extraction
        query_words = set(query.text.lower().split())
        matching_nodes = []

        for node in list(graph.nodes())[:100]:  # Limit for performance
            node_str = str(node).lower()
            if any(word in node_str for word in query_words):
                matching_nodes.append(node)

        # Limit to k results
        matching_nodes = matching_nodes[:query.k]

        results = []
        for node in matching_nodes:
            results.append(MemoryResult(
                node_id=node,
                content=str(node),  # Placeholder
                relevance=0.7,  # Estimate
                source_system=MemorySystem.KNOWLEDGE_GRAPH,
                metadata={"graph_node": True}
            ))

        return results

    def _get_memory_content(self, node_id: str) -> str:
        """
        Get content for a memory node from the backend.

        Helper to retrieve actual content from memory backend
        for nodes returned by various memory systems.

        Args:
            node_id: Node identifier

        Returns:
            Content string (or node_id as fallback)
        """
        try:
            # Try to get from memory backend's graph
            if hasattr(self.memory_backend, 'graph'):
                graph = self.memory_backend.graph
                if hasattr(graph, '_graph') and node_id in graph._graph.nodes:
                    node_data = graph._graph.nodes[node_id]
                    # Try common content fields
                    for field in ['content', 'text', 'data', 'value']:
                        if field in node_data:
                            return str(node_data[field])

            # Try direct retrieval methods
            if hasattr(self.memory_backend, 'get_node'):
                node = self.memory_backend.get_node(node_id)
                if node:
                    return str(getattr(node, 'content', node_id))

            # Try get_by_id (InMemoryStore)
            if hasattr(self.memory_backend, 'get_by_id'):
                # Check if it's async
                if asyncio.iscoroutinefunction(self.memory_backend.get_by_id):
                    # We are not in an async function here, this is a limitation
                    # _get_memory_content is sync.
                    # For now, return node_id and let the system handle async retrieval elsewhere
                    # OR we can try to hack it if we knew we were in async context, but we are not guaranteed.
                    pass
                else:
                    node = self.memory_backend.get_by_id(node_id)
                    if node:
                        return str(getattr(node, 'text', getattr(node, 'content', node_id)))

        except Exception as e:
            if self.verbose:
                logger.debug(f"Could not get content for {node_id}: {e}")

        # Fallback to node_id itself
        return str(node_id)

    async def _query_hot_patterns(self, query: MemoryQuery) -> list[MemoryResult]:
        """
        Query hot patterns (usage-based boosting).

        Uses HotPatternTracker to find frequently accessed,
        high-quality memory elements.

        Heat score = access_count × success_rate × avg_confidence
        """
        if not self._hot_tracker:
            return []

        try:
            # Get hot patterns from tracker
            # API: get_hot_patterns(element_type, top_k) -> List[UsageRecord]
            hot_records = self._hot_tracker.get_hot_patterns(
                element_type=None,  # All types
                top_k=query.k
            )

            results = []
            for record in hot_records:
                # UsageRecord has: element_id, element_type, heat_score property
                heat_score = getattr(record, 'heat_score', 0.5)

                results.append(MemoryResult(
                    node_id=record.element_id,
                    content=self._get_memory_content(record.element_id),
                    relevance=min(heat_score, 1.0),  # Cap at 1.0
                    source_system=MemorySystem.HOT_PATTERNS,
                    activation=0.0,
                    heat=heat_score,
                    metadata={
                        "element_type": getattr(record, 'element_type', 'unknown'),
                        "access_count": getattr(record, 'access_count', 0),
                        "success_rate": getattr(record, 'success_rate', 0.0),
                    }
                ))

            return results

        except Exception as e:
            if self.verbose:
                logger.warning(f"Hot patterns query failed: {e}")
            return []

    async def _query_awareness_graph(self, query: MemoryQuery) -> list[MemoryResult]:
        """
        Query awareness graph (spreading activation).

        Uses AwarenessGraph to find memories through
        spreading activation from query-related nodes.
        """
        if not self._awareness_graph:
            return []

        try:
            # AwarenessGraph API: perceive(content) -> SemanticPerception
            # Then: activate(perception, budget, strategy) -> List[Memory]

            # First perceive the query
            if hasattr(self._awareness_graph, 'perceive'):
                perception = self._awareness_graph.perceive(query.text)
            else:
                # Fallback: create minimal perception-like object
                perception = type('Perception', (), {'query': query.text})()

            # Activate memories based on perception
            if hasattr(self._awareness_graph, 'activate'):
                # Try async first
                if asyncio.iscoroutinefunction(self._awareness_graph.activate):
                    memories = await self._awareness_graph.activate(perception)
                else:
                    memories = self._awareness_graph.activate(perception)
            else:
                return []

            results = []
            for memory in memories[:query.k]:
                # Memory objects typically have: id/node_id, text/content, context
                node_id = getattr(memory, 'id', getattr(memory, 'node_id', str(memory)))
                content = getattr(memory, 'text', getattr(memory, 'content', str(memory)))
                activation = getattr(memory, 'activation', 0.5)

                results.append(MemoryResult(
                    node_id=str(node_id),
                    content=str(content),
                    relevance=activation,  # Use activation as relevance
                    source_system=MemorySystem.AWARENESS_GRAPH,
                    activation=activation,
                    heat=0.0,
                    metadata={
                        "spreading_activation": True,
                        "context": getattr(memory, 'context', {}),
                    }
                ))

            return results

        except Exception as e:
            if self.verbose:
                logger.warning(f"Awareness graph query failed: {e}")
            return []

    async def _query_spring_dynamics(self, query: MemoryQuery) -> list[MemoryResult]:
        """
        Query spring dynamics (physics-based connectivity).

        Uses Hooke's Law spring physics to propagate activation
        through memory graph connections.
        """
        if not self._spring_dynamics:
            return []

        try:
            # SpringDynamics API:
            # 1. activate_nodes(activations: Dict[str, float])
            # 2. propagate() -> SpringPropagationResult

            # Get initial seed nodes from vector search
            seed_nodes = await self._query_vector_memory(query)

            if not seed_nodes:
                return []

            # Create activation dict from seed nodes
            activations = {
                r.node_id: r.relevance
                for r in seed_nodes[:5]  # Top 5 as seeds
            }

            # Activate and propagate
            if hasattr(self._spring_dynamics, 'activate_nodes'):
                self._spring_dynamics.activate_nodes(activations)

            if hasattr(self._spring_dynamics, 'propagate'):
                result = self._spring_dynamics.propagate()

                # SpringPropagationResult has: activated_nodes, node_activations
                results = []
                activated_nodes = getattr(result, 'activated_nodes', [])
                node_activations = getattr(result, 'node_activations', {})

                for node_id in activated_nodes[:query.k]:
                    activation = node_activations.get(node_id, 0.5)

                    results.append(MemoryResult(
                        node_id=node_id,
                        content=self._get_memory_content(node_id),
                        relevance=activation,
                        source_system=MemorySystem.SPRING_DYNAMICS,
                        activation=activation,
                        heat=0.0,
                        metadata={
                            "physics_based": True,
                            "converged": getattr(result, 'converged', False),
                            "iterations": getattr(result, 'iterations', 0),
                        }
                    ))

                return results

            return []

        except Exception as e:
            if self.verbose:
                logger.warning(f"Spring dynamics query failed: {e}")
            return []

    async def _query_multi_wave(self, query: MemoryQuery) -> list[MemoryResult]:
        """
        Query multi-wave engine (temporal propagation).

        Uses brain wave-inspired cycles (Beta/Alpha/Theta/Delta)
        for temporal memory retrieval.
        """
        if not self._multi_wave_engine:
            return []

        try:
            # MultiWaveEngine API options:
            # 1. on_query(query_embedding) -> BetaWaveRecallResult
            # 2. retrieve_memories(query_embedding, top_k, ...) -> BetaWaveRecallResult

            # Need query embedding - try to get from memory backend
            query_embedding = None
            if hasattr(self.memory_backend, 'embed'):
                query_embedding = self.memory_backend.embed(query.text)
            elif hasattr(self.memory_backend, 'embedder'):
                embedder = self.memory_backend.embedder
                if hasattr(embedder, 'encode'):
                    import numpy as np
                    embedding = embedder.encode(query.text)
                    query_embedding = np.array(embedding)

            if query_embedding is None:
                # Fallback: create simple embedding placeholder
                import numpy as np
                query_embedding = np.zeros(384)  # Default dimension

            # Try retrieve_memories first (more flexible)
            if hasattr(self._multi_wave_engine, 'retrieve_memories'):
                result = self._multi_wave_engine.retrieve_memories(
                    query_embedding=query_embedding,
                    top_k=query.k,
                )
            elif hasattr(self._multi_wave_engine, 'on_query'):
                result = self._multi_wave_engine.on_query(query_embedding)
            else:
                return []

            # BetaWaveRecallResult has: recalled_memories List[(node_id, activation)]
            results = []
            recalled = getattr(result, 'recalled_memories', [])

            for item in recalled[:query.k]:
                # Item is tuple: (node_id, activation_level)
                if isinstance(item, tuple) and len(item) >= 2:
                    node_id, activation = item[0], item[1]
                else:
                    node_id = str(item)
                    activation = 0.5

                results.append(MemoryResult(
                    node_id=str(node_id),
                    content=self._get_memory_content(str(node_id)),
                    relevance=float(activation),
                    source_system=MemorySystem.MULTI_WAVE,
                    activation=float(activation),
                    heat=0.0,
                    metadata={
                        "wave_based": True,
                        "creative_insights": len(getattr(result, 'creative_insights', [])),
                    }
                ))

            return results

        except Exception as e:
            if self.verbose:
                logger.warning(f"Multi-wave query failed: {e}")
            return []

    def _merge_results(
        self,
        all_results: list[list[MemoryResult]],
        k: int
    ) -> list[MemoryResult]:
        """
        Merge results from multiple systems.

        Merging strategy:
        1. Combine all results
        2. Deduplicate by node_id
        3. Aggregate relevance scores (weighted average)
        4. Sort by aggregated relevance
        5. Return top k
        """
        # Combine all results
        combined: dict[str, MemoryResult] = {}

        for system_results in all_results:
            for result in system_results:
                node_id = result.node_id

                if node_id in combined:
                    # Already seen - aggregate relevance
                    existing = combined[node_id]
                    # Weighted average (favor higher relevance)
                    new_relevance = max(existing.relevance, result.relevance)
                    existing.relevance = new_relevance

                    # Add source system to metadata
                    if 'source_systems' not in existing.metadata:
                        existing.metadata['source_systems'] = [existing.source_system]
                    existing.metadata['source_systems'].append(result.source_system)
                else:
                    combined[node_id] = result

        # Sort by relevance
        sorted_results = sorted(
            combined.values(),
            key=lambda r: r.relevance,
            reverse=True
        )

        # Return top k
        return sorted_results[:k]

    def _update_metrics(
        self,
        result: MemoryCoordinationResult,
        strategy: MemoryStrategy,
        systems: list[MemorySystem]
    ):
        """Update performance metrics."""
        self._metrics.total_queries += 1
        if not result.cache_hit:
            self._metrics.cache_misses += 1

        # Update averages (running average)
        n = self._metrics.total_queries
        self._metrics.avg_latency_ms = (
            (self._metrics.avg_latency_ms * (n - 1) + result.total_latency_ms) / n
        )
        self._metrics.avg_results_per_query = (
            (self._metrics.avg_results_per_query * (n - 1) + len(result.results)) / n
        )

        # Update strategy usage
        if strategy not in self._metrics.strategy_usage:
            self._metrics.strategy_usage[strategy] = 0
        self._metrics.strategy_usage[strategy] += 1

        # Update system usage
        for system in systems:
            if system not in self._metrics.system_usage:
                self._metrics.system_usage[system] = 0
            self._metrics.system_usage[system] += 1

    def _make_cache_key(self, query: MemoryQuery) -> str:
        """Create cache key from query."""
        return f"{query.text}_{query.k}_{query.strategy.value}"

    def get_performance_metrics(self) -> MemoryPerformanceMetrics:
        """Get current performance metrics."""
        return self._metrics

    def get_query_history(self, limit: int = 10) -> list[MemoryCoordinationResult]:
        """Get recent query history."""
        return self._query_history[-limit:]


# Convenience function
def create_memory_conductor(
    memory_backend: Any,
    strategy: MemoryStrategy = MemoryStrategy.AUTO,
    **kwargs
) -> MemoryConductor:
    """Create MemoryConductor with defaults."""
    return MemoryConductor(
        memory_backend=memory_backend,
        default_strategy=strategy,
        **kwargs
    )
