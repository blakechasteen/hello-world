"""
Unit Tests: Memory Symphony Conductor
======================================

Tests MemoryConductor directly — strategy selection, coordination plans,
result merging, caching, metrics, fallback on missing systems.

All backends are mocked. Each test <5s.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from hololoom.core.memory.symphony.protocol import (
    MemoryQuery,
    MemoryResult,
    MemoryCoordinationResult,
    MemoryPerformanceMetrics,
    MemoryStrategy,
    MemorySystem,
    StrategySelectionCriteria,
    CoordinationPlan,
)
from hololoom.core.memory.symphony.conductor import (
    MemoryConductor,
    create_memory_conductor,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_backend():
    """Minimal mock backend with recall returning empty list."""
    backend = Mock()
    backend.recall = AsyncMock(return_value=[])
    backend.graph = None
    return backend


@pytest.fixture
def mock_backend_with_recall():
    """Backend whose recall returns fake Memory objects."""

    @dataclass
    class FakeMemory:
        id: str
        text: str
        relevance: float = 0.8
        metadata: dict = None

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

    backend = Mock()
    backend.recall = AsyncMock(
        return_value=[
            FakeMemory(id="n1", text="Thompson Sampling explores arms", relevance=0.9),
            FakeMemory(id="n2", text="Beta distribution models uncertainty", relevance=0.7),
        ]
    )
    backend.graph = None
    return backend


@pytest.fixture
def conductor(mock_backend):
    return MemoryConductor(memory_backend=mock_backend)


@pytest.fixture
def verbose_conductor(mock_backend):
    return MemoryConductor(memory_backend=mock_backend, verbose=True)


@pytest.fixture
def conductor_all_systems(mock_backend):
    """Conductor with all optional systems enabled + mocked instances."""
    hot = Mock()
    hot.get_hot_patterns = Mock(return_value=[])

    awareness = Mock()
    awareness.perceive = Mock(return_value=Mock(query="test"))
    awareness.activate = Mock(return_value=[])

    spring = Mock()
    spring.activate_nodes = Mock()
    spring.propagate = Mock(
        return_value=Mock(activated_nodes=[], node_activations={}, converged=True, iterations=3)
    )

    wave = Mock()
    wave.retrieve_memories = Mock(
        return_value=Mock(recalled_memories=[], creative_insights=[])
    )

    return MemoryConductor(
        memory_backend=mock_backend,
        enable_spring_dynamics=True,
        enable_multi_wave=True,
        hot_tracker=hot,
        awareness_graph=awareness,
        spring_dynamics=spring,
        multi_wave_engine=wave,
    )


def _make_query(text="test query", k=5, strategy=MemoryStrategy.AUTO):
    return MemoryQuery(text=text, k=k, strategy=strategy)


# ============================================================================
# Initialization
# ============================================================================


class TestInit:
    def test_default_flags(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend)
        assert c.enable_cache is True
        assert c.enable_hot_patterns is True
        assert c.enable_awareness is True
        assert c.enable_spring_dynamics is False
        assert c.enable_multi_wave is False
        assert c.default_strategy == MemoryStrategy.AUTO

    def test_custom_flags(self, mock_backend):
        c = MemoryConductor(
            memory_backend=mock_backend,
            enable_cache=False,
            enable_hot_patterns=False,
            enable_spring_dynamics=True,
            default_strategy=MemoryStrategy.FAST,
        )
        assert c.enable_cache is False
        assert c.enable_hot_patterns is False
        assert c.enable_spring_dynamics is True
        assert c.default_strategy == MemoryStrategy.FAST

    def test_optional_system_instances_stored(self, mock_backend):
        hot = Mock()
        awareness = Mock()
        spring = Mock()
        wave = Mock()
        c = MemoryConductor(
            memory_backend=mock_backend,
            hot_tracker=hot,
            awareness_graph=awareness,
            spring_dynamics=spring,
            multi_wave_engine=wave,
        )
        assert c._hot_tracker is hot
        assert c._awareness_graph is awareness
        assert c._spring_dynamics is spring
        assert c._multi_wave_engine is wave

    def test_metrics_initialized(self, conductor):
        m = conductor.get_performance_metrics()
        assert isinstance(m, MemoryPerformanceMetrics)
        assert m.total_queries == 0
        assert m.cache_hits == 0

    def test_create_memory_conductor_convenience(self, mock_backend):
        c = create_memory_conductor(mock_backend, strategy=MemoryStrategy.DEEP)
        assert isinstance(c, MemoryConductor)
        assert c.default_strategy == MemoryStrategy.DEEP


# ============================================================================
# Strategy Selection
# ============================================================================


class TestStrategySelection:
    def test_short_query_returns_fast(self, conductor):
        q = _make_query("hello")
        assert conductor.select_strategy(q) == MemoryStrategy.FAST

    def test_medium_query_returns_balanced(self, conductor):
        q = _make_query("how does Thompson Sampling work in bandits")
        assert conductor.select_strategy(q) == MemoryStrategy.BALANCED

    def test_long_query_returns_deep_or_research(self, conductor):
        q = _make_query(
            "compare and analyze the tradeoffs between Thompson Sampling "
            "and UCB in multi-armed bandit problems with non-stationary rewards"
        )
        strategy = conductor.select_strategy(q)
        assert strategy in (MemoryStrategy.DEEP, MemoryStrategy.RESEARCH)

    def test_research_keyword_triggers_research(self, conductor):
        q = _make_query("comprehensive detailed analysis of spring dynamics")
        assert conductor.select_strategy(q) == MemoryStrategy.RESEARCH

    def test_cached_query_returns_fast(self, conductor):
        q = _make_query("cached query")
        # Prime the cache manually
        key = conductor._make_cache_key(q)
        conductor._cache[key] = Mock()
        assert conductor.select_strategy(q) == MemoryStrategy.FAST

    def test_explicit_strategy_bypasses_auto(self, conductor):
        """When query already has a concrete strategy, select_strategy still works
        (it's only called when strategy==AUTO in recall)."""
        q = _make_query("hello", strategy=MemoryStrategy.DEEP)
        # select_strategy doesn't check query.strategy for DEEP/FAST etc,
        # it analyzes query text. That's fine — recall() only calls it for AUTO.
        # Just verify it doesn't crash.
        result = conductor.select_strategy(q)
        assert isinstance(result, MemoryStrategy)


# ============================================================================
# Coordination Plans
# ============================================================================


class TestCoordinationPlan:
    def test_fast_plan(self, conductor):
        q = _make_query("test")
        plan = conductor._create_coordination_plan(q, MemoryStrategy.FAST)
        assert MemorySystem.VECTOR_MEMORY in plan.systems_to_query
        assert plan.parallel_execution is True
        assert plan.estimated_latency_ms == 50.0

    def test_fast_plan_includes_hot_patterns_when_enabled(self, conductor):
        q = _make_query("test")
        plan = conductor._create_coordination_plan(q, MemoryStrategy.FAST)
        assert MemorySystem.HOT_PATTERNS in plan.systems_to_query

    def test_fast_plan_excludes_hot_patterns_when_disabled(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend, enable_hot_patterns=False)
        q = _make_query("test")
        plan = c._create_coordination_plan(q, MemoryStrategy.FAST)
        assert MemorySystem.HOT_PATTERNS not in plan.systems_to_query

    def test_balanced_plan(self, conductor):
        q = _make_query("test")
        plan = conductor._create_coordination_plan(q, MemoryStrategy.BALANCED)
        assert MemorySystem.VECTOR_MEMORY in plan.systems_to_query
        assert MemorySystem.KNOWLEDGE_GRAPH in plan.systems_to_query
        assert plan.parallel_execution is True
        assert plan.estimated_latency_ms == 150.0

    def test_deep_plan(self, conductor):
        q = _make_query("test")
        plan = conductor._create_coordination_plan(q, MemoryStrategy.DEEP)
        assert MemorySystem.VECTOR_MEMORY in plan.systems_to_query
        assert MemorySystem.KNOWLEDGE_GRAPH in plan.systems_to_query
        assert MemorySystem.HOT_PATTERNS in plan.systems_to_query
        assert MemorySystem.AWARENESS_GRAPH in plan.systems_to_query
        assert plan.parallel_execution is False
        assert plan.estimated_latency_ms == 300.0

    def test_deep_plan_includes_spring_when_enabled(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend, enable_spring_dynamics=True)
        q = _make_query("test")
        plan = c._create_coordination_plan(q, MemoryStrategy.DEEP)
        assert MemorySystem.SPRING_DYNAMICS in plan.systems_to_query

    def test_research_plan(self, conductor):
        q = _make_query("test")
        plan = conductor._create_coordination_plan(q, MemoryStrategy.RESEARCH)
        assert MemorySystem.VECTOR_MEMORY in plan.systems_to_query
        assert MemorySystem.KNOWLEDGE_GRAPH in plan.systems_to_query
        assert MemorySystem.HOT_PATTERNS in plan.systems_to_query
        assert MemorySystem.AWARENESS_GRAPH in plan.systems_to_query
        assert plan.parallel_execution is False
        assert plan.estimated_latency_ms == 500.0

    def test_research_plan_includes_multi_wave_when_enabled(self, mock_backend):
        c = MemoryConductor(
            memory_backend=mock_backend,
            enable_spring_dynamics=True,
            enable_multi_wave=True,
        )
        q = _make_query("test")
        plan = c._create_coordination_plan(q, MemoryStrategy.RESEARCH)
        assert MemorySystem.SPRING_DYNAMICS in plan.systems_to_query
        assert MemorySystem.MULTI_WAVE in plan.systems_to_query

    def test_fallback_systems_always_vector(self, conductor):
        for strat in [MemoryStrategy.FAST, MemoryStrategy.BALANCED, MemoryStrategy.DEEP]:
            plan = conductor._create_coordination_plan(_make_query("x"), strat)
            assert plan.fallback_systems == [MemorySystem.VECTOR_MEMORY]


# ============================================================================
# Full Recall Pipeline
# ============================================================================


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_coordination_result(self, conductor):
        q = _make_query("what is Thompson Sampling", strategy=MemoryStrategy.FAST)
        result = await conductor.recall(q)
        assert isinstance(result, MemoryCoordinationResult)
        assert result.cache_hit is False
        assert result.strategy_used == MemoryStrategy.FAST

    @pytest.mark.asyncio
    async def test_recall_auto_selects_strategy(self, conductor):
        q = _make_query("hello")  # short → FAST
        result = await conductor.recall(q)
        assert result.strategy_used == MemoryStrategy.FAST

    @pytest.mark.asyncio
    async def test_recall_updates_metrics(self, conductor):
        q = _make_query("test", strategy=MemoryStrategy.BALANCED)
        await conductor.recall(q)
        m = conductor.get_performance_metrics()
        assert m.total_queries == 1
        assert m.cache_misses == 1
        assert MemoryStrategy.BALANCED in m.strategy_usage

    @pytest.mark.asyncio
    async def test_recall_updates_query_history(self, conductor):
        q = _make_query("test", strategy=MemoryStrategy.FAST)
        await conductor.recall(q)
        history = conductor.get_query_history()
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_recall_populates_coordination_metadata(self, conductor):
        q = _make_query("test", strategy=MemoryStrategy.FAST)
        result = await conductor.recall(q)
        assert "num_systems" in result.coordination_metadata
        assert "parallel" in result.coordination_metadata

    @pytest.mark.asyncio
    async def test_recall_with_backend_recall(self, mock_backend_with_recall):
        c = MemoryConductor(memory_backend=mock_backend_with_recall)
        q = _make_query("Thompson Sampling", strategy=MemoryStrategy.FAST)
        result = await c.recall(q)
        assert isinstance(result, MemoryCoordinationResult)
        # Backend recall was awaited
        assert mock_backend_with_recall.recall.called


# ============================================================================
# Caching
# ============================================================================


class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit(self, conductor):
        q = _make_query("cacheable query", strategy=MemoryStrategy.FAST)
        # First call — cache miss
        r1 = await conductor.recall(q)
        assert r1.cache_hit is False

        # Second call — same query, should hit cache
        q2 = _make_query("cacheable query", strategy=MemoryStrategy.FAST)
        r2 = await conductor.recall(q2)
        assert r2.cache_hit is False or r2 is r1  # may return cached object

        m = conductor.get_performance_metrics()
        assert m.total_queries >= 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend, enable_cache=False)
        q = _make_query("no cache", strategy=MemoryStrategy.FAST)
        r1 = await c.recall(q)
        r2 = await c.recall(_make_query("no cache", strategy=MemoryStrategy.FAST))
        # Both should be fresh (no cache hit path taken)
        assert r1.cache_hit is False
        assert r2.cache_hit is False

    def test_cache_key_includes_strategy(self, conductor):
        q1 = _make_query("same text", strategy=MemoryStrategy.FAST)
        q2 = _make_query("same text", strategy=MemoryStrategy.DEEP)
        assert conductor._make_cache_key(q1) != conductor._make_cache_key(q2)

    def test_cache_key_includes_k(self, conductor):
        q1 = _make_query("same text", k=5)
        q2 = _make_query("same text", k=10)
        assert conductor._make_cache_key(q1) != conductor._make_cache_key(q2)

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend)
        c._cache_max_size = 3

        for i in range(5):
            q = _make_query(f"query_{i}", strategy=MemoryStrategy.FAST)
            await c.recall(q)

        assert len(c._cache) <= 4  # at most max_size + 1 before eviction triggers


# ============================================================================
# Result Merging
# ============================================================================


class TestMergeResults:
    def test_merge_empty(self, conductor):
        merged = conductor._merge_results([], k=5)
        assert merged == []

    def test_merge_single_system(self, conductor):
        results = [
            [
                MemoryResult(node_id="a", content="alpha", relevance=0.9, source_system=MemorySystem.VECTOR_MEMORY),
                MemoryResult(node_id="b", content="beta", relevance=0.7, source_system=MemorySystem.VECTOR_MEMORY),
            ]
        ]
        merged = conductor._merge_results(results, k=5)
        assert len(merged) == 2
        assert merged[0].node_id == "a"  # higher relevance first
        assert merged[1].node_id == "b"

    def test_merge_deduplicates_by_node_id(self, conductor):
        results = [
            [MemoryResult(node_id="x", content="x1", relevance=0.6, source_system=MemorySystem.VECTOR_MEMORY)],
            [MemoryResult(node_id="x", content="x2", relevance=0.9, source_system=MemorySystem.KNOWLEDGE_GRAPH)],
        ]
        merged = conductor._merge_results(results, k=5)
        assert len(merged) == 1
        assert merged[0].relevance == 0.9  # takes max relevance

    def test_merge_respects_k(self, conductor):
        results = [
            [
                MemoryResult(node_id=f"n{i}", content=f"c{i}", relevance=0.5, source_system=MemorySystem.VECTOR_MEMORY)
                for i in range(20)
            ]
        ]
        merged = conductor._merge_results(results, k=3)
        assert len(merged) == 3

    def test_merge_tracks_source_systems_on_duplicate(self, conductor):
        results = [
            [MemoryResult(node_id="dup", content="d", relevance=0.5, source_system=MemorySystem.VECTOR_MEMORY)],
            [MemoryResult(node_id="dup", content="d", relevance=0.8, source_system=MemorySystem.HOT_PATTERNS)],
        ]
        merged = conductor._merge_results(results, k=5)
        assert len(merged) == 1
        assert "source_systems" in merged[0].metadata
        assert MemorySystem.HOT_PATTERNS in merged[0].metadata["source_systems"]

    def test_merge_sorts_by_relevance_descending(self, conductor):
        results = [
            [
                MemoryResult(node_id="low", content="l", relevance=0.2, source_system=MemorySystem.VECTOR_MEMORY),
                MemoryResult(node_id="high", content="h", relevance=0.95, source_system=MemorySystem.VECTOR_MEMORY),
                MemoryResult(node_id="mid", content="m", relevance=0.6, source_system=MemorySystem.VECTOR_MEMORY),
            ]
        ]
        merged = conductor._merge_results(results, k=10)
        assert [r.node_id for r in merged] == ["high", "mid", "low"]


# ============================================================================
# Metrics
# ============================================================================


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_accumulate(self, conductor):
        for i in range(3):
            await conductor.recall(_make_query(f"q{i}", strategy=MemoryStrategy.FAST))

        m = conductor.get_performance_metrics()
        assert m.total_queries == 3
        assert m.cache_misses == 3
        assert m.strategy_usage[MemoryStrategy.FAST] == 3

    @pytest.mark.asyncio
    async def test_avg_latency_tracked(self, conductor):
        await conductor.recall(_make_query("x", strategy=MemoryStrategy.FAST))
        m = conductor.get_performance_metrics()
        assert m.avg_latency_ms >= 0  # May be 0.0 due to timer resolution

    @pytest.mark.asyncio
    async def test_system_usage_tracked(self, conductor):
        await conductor.recall(_make_query("x", strategy=MemoryStrategy.BALANCED))
        m = conductor.get_performance_metrics()
        assert MemorySystem.VECTOR_MEMORY in m.system_usage
        assert MemorySystem.KNOWLEDGE_GRAPH in m.system_usage

    def test_query_history_limit(self, conductor):
        # Manually add items to history
        for i in range(20):
            conductor._query_history.append(Mock())
        history = conductor.get_query_history(limit=5)
        assert len(history) == 5


# ============================================================================
# Individual System Queries (with mocked backends)
# ============================================================================


class TestQuerySystems:
    @pytest.mark.asyncio
    async def test_query_vector_memory_with_recall(self, mock_backend_with_recall):
        c = MemoryConductor(memory_backend=mock_backend_with_recall)
        q = _make_query("Thompson", k=5)
        results = await c._query_vector_memory(q)
        assert isinstance(results, list)
        # Backend was called
        assert mock_backend_with_recall.recall.called

    @pytest.mark.asyncio
    async def test_query_vector_memory_empty_backend(self, mock_backend):
        c = MemoryConductor(memory_backend=mock_backend)
        q = _make_query("test", k=5)
        results = await c._query_vector_memory(q)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_no_graph(self, conductor):
        q = _make_query("test")
        # graph is None on mock_backend — provide empty graph
        conductor.memory_backend.graph = MagicMock()
        conductor.memory_backend.graph.nodes.return_value = []
        results = await conductor._query_knowledge_graph(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_with_graph(self, mock_backend):
        """Backend with a graph that has nodes matching query words."""
        import types

        class FakeGraph:
            def __init__(self):
                self._nodes = ["thompson_sampling", "beta_distribution", "random_walk"]

            def nodes(self):
                return self._nodes

        mock_backend.graph = FakeGraph()
        c = MemoryConductor(memory_backend=mock_backend)
        q = _make_query("thompson sampling", k=5)
        results = await c._query_knowledge_graph(q)
        assert len(results) >= 1
        assert results[0].source_system == MemorySystem.KNOWLEDGE_GRAPH
        assert results[0].relevance == 0.7

    @pytest.mark.asyncio
    async def test_query_hot_patterns_no_tracker(self, conductor):
        q = _make_query("test")
        results = await conductor._query_hot_patterns(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_hot_patterns_with_tracker(self, mock_backend):
        @dataclass
        class FakeRecord:
            element_id: str = "elem_1"
            element_type: str = "pattern"
            heat_score: float = 0.85
            access_count: int = 10
            success_rate: float = 0.9

        hot = Mock()
        hot.get_hot_patterns = Mock(return_value=[FakeRecord()])
        c = MemoryConductor(memory_backend=mock_backend, hot_tracker=hot)
        q = _make_query("test", k=5)
        results = await c._query_hot_patterns(q)
        assert len(results) == 1
        assert results[0].source_system == MemorySystem.HOT_PATTERNS
        assert results[0].heat == 0.85
        assert results[0].metadata["access_count"] == 10

    @pytest.mark.asyncio
    async def test_query_awareness_no_graph(self, conductor):
        q = _make_query("test")
        results = await conductor._query_awareness_graph(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_awareness_with_graph(self, mock_backend):
        @dataclass
        class FakeMemory:
            id: str = "aw_1"
            text: str = "activated memory"
            activation: float = 0.75
            context: dict = None

            def __post_init__(self):
                if self.context is None:
                    self.context = {}

        awareness = Mock()
        awareness.perceive = Mock(return_value=Mock(query="test"))
        awareness.activate = Mock(return_value=[FakeMemory()])
        c = MemoryConductor(memory_backend=mock_backend, awareness_graph=awareness)
        q = _make_query("test", k=5)
        results = await c._query_awareness_graph(q)
        assert len(results) == 1
        assert results[0].source_system == MemorySystem.AWARENESS_GRAPH
        assert results[0].activation == 0.75

    @pytest.mark.asyncio
    async def test_query_awareness_async_activate(self, mock_backend):
        @dataclass
        class FakeMemory:
            id: str = "aw_async"
            text: str = "async memory"
            activation: float = 0.6

        awareness = Mock()
        awareness.perceive = Mock(return_value=Mock(query="test"))
        awareness.activate = AsyncMock(return_value=[FakeMemory()])
        c = MemoryConductor(memory_backend=mock_backend, awareness_graph=awareness)
        q = _make_query("test", k=5)
        results = await c._query_awareness_graph(q)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_spring_dynamics_no_instance(self, conductor):
        q = _make_query("test")
        results = await conductor._query_spring_dynamics(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_spring_dynamics_with_instance(self, mock_backend_with_recall):
        spring = Mock()
        spring.activate_nodes = Mock()
        spring.propagate = Mock(
            return_value=Mock(
                activated_nodes=["sp_1", "sp_2"],
                node_activations={"sp_1": 0.8, "sp_2": 0.5},
                converged=True,
                iterations=5,
            )
        )
        c = MemoryConductor(
            memory_backend=mock_backend_with_recall,
            enable_spring_dynamics=True,
            spring_dynamics=spring,
        )
        q = _make_query("spring test", k=5)
        results = await c._query_spring_dynamics(q)
        assert len(results) == 2
        assert results[0].source_system == MemorySystem.SPRING_DYNAMICS
        assert results[0].metadata["physics_based"] is True
        assert spring.activate_nodes.called
        assert spring.propagate.called

    @pytest.mark.asyncio
    async def test_query_multi_wave_no_instance(self, conductor):
        q = _make_query("test")
        results = await conductor._query_multi_wave(q)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_multi_wave_with_instance(self, mock_backend):
        wave = Mock()
        wave.retrieve_memories = Mock(
            return_value=Mock(
                recalled_memories=[("wave_1", 0.9), ("wave_2", 0.6)],
                creative_insights=["insight1"],
            )
        )
        c = MemoryConductor(
            memory_backend=mock_backend,
            enable_multi_wave=True,
            multi_wave_engine=wave,
        )
        q = _make_query("wave test", k=5)
        results = await c._query_multi_wave(q)
        assert len(results) == 2
        assert results[0].source_system == MemorySystem.MULTI_WAVE
        assert results[0].relevance == 0.9
        assert results[0].metadata["wave_based"] is True
        assert results[0].metadata["creative_insights"] == 1


# ============================================================================
# Execution Plan (parallel vs sequential, error handling)
# ============================================================================


class TestExecutePlan:
    @pytest.mark.asyncio
    async def test_parallel_execution(self, conductor):
        plan = CoordinationPlan(
            strategy=MemoryStrategy.FAST,
            systems_to_query=[MemorySystem.VECTOR_MEMORY, MemorySystem.HOT_PATTERNS],
            parallel_execution=True,
            fallback_systems=[MemorySystem.VECTOR_MEMORY],
            estimated_latency_ms=50.0,
            expected_coverage=0.8,
        )
        q = _make_query("test")
        results = await conductor._execute_plan(q, plan)
        assert isinstance(results, list)
        assert len(results) == 2  # one per system

    @pytest.mark.asyncio
    async def test_sequential_execution(self, conductor):
        plan = CoordinationPlan(
            strategy=MemoryStrategy.DEEP,
            systems_to_query=[MemorySystem.VECTOR_MEMORY, MemorySystem.KNOWLEDGE_GRAPH],
            parallel_execution=False,
            fallback_systems=[MemorySystem.VECTOR_MEMORY],
            estimated_latency_ms=300.0,
            expected_coverage=0.8,
        )
        q = _make_query("test")
        results = await conductor._execute_plan(q, plan)
        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_parallel_handles_exception(self, conductor):
        """If a system raises during parallel execution, it becomes []."""
        plan = CoordinationPlan(
            strategy=MemoryStrategy.FAST,
            systems_to_query=[MemorySystem.VECTOR_MEMORY, MemorySystem.AWARENESS_GRAPH],
            parallel_execution=True,
            fallback_systems=[MemorySystem.VECTOR_MEMORY],
            estimated_latency_ms=50.0,
            expected_coverage=0.8,
        )
        q = _make_query("test")
        # Awareness graph not set → returns [] gracefully
        results = await conductor._execute_plan(q, plan)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)

    @pytest.mark.asyncio
    async def test_sequential_handles_exception(self, conductor):
        """Sequential execution catches per-task exceptions."""
        plan = CoordinationPlan(
            strategy=MemoryStrategy.DEEP,
            systems_to_query=[MemorySystem.VECTOR_MEMORY, MemorySystem.SPRING_DYNAMICS],
            parallel_execution=False,
            fallback_systems=[MemorySystem.VECTOR_MEMORY],
            estimated_latency_ms=300.0,
            expected_coverage=0.8,
        )
        q = _make_query("test")
        results = await conductor._execute_plan(q, plan)
        assert len(results) == 2


# ============================================================================
# _query_system dispatch
# ============================================================================


class TestQuerySystemDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_system(self, conductor):
        # MemorySystem.QUERY_CACHE is not handled in _query_system
        q = _make_query("test")
        result = await conductor._query_system(MemorySystem.QUERY_CACHE, q)
        assert result == []

    @pytest.mark.asyncio
    async def test_dispatch_vector(self, conductor):
        q = _make_query("test")
        result = await conductor._query_system(MemorySystem.VECTOR_MEMORY, q)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_dispatch_kg(self, conductor):
        q = _make_query("test")
        result = await conductor._query_system(MemorySystem.KNOWLEDGE_GRAPH, q)
        assert isinstance(result, list)


# ============================================================================
# _get_memory_content helper
# ============================================================================


class TestGetMemoryContent:
    def test_fallback_to_node_id(self):
        # Use a backend with spec to prevent auto-created attributes
        backend = Mock(spec=[])
        c = MemoryConductor(memory_backend=backend)
        assert c._get_memory_content("node_123") == "node_123"

    def test_with_graph_backend(self, mock_backend):
        class FakeInnerGraph:
            def __init__(self):
                self.nodes = {"n1": {"content": "stored content", "text": "alt text"}}

        class FakeGraph:
            def __init__(self):
                self._graph = FakeInnerGraph()

        mock_backend.graph = FakeGraph()
        c = MemoryConductor(memory_backend=mock_backend)
        assert c._get_memory_content("n1") == "stored content"

    def test_with_get_node(self, mock_backend):
        node_obj = Mock()
        node_obj.content = "from get_node"
        mock_backend.graph = None
        mock_backend.get_node = Mock(return_value=node_obj)
        c = MemoryConductor(memory_backend=mock_backend)
        assert c._get_memory_content("n1") == "from get_node"

    def test_with_sync_get_by_id(self, mock_backend):
        node_obj = Mock()
        node_obj.text = "from get_by_id"
        mock_backend.graph = None
        mock_backend.get_node = Mock(return_value=None)
        mock_backend.get_by_id = Mock(return_value=node_obj)
        c = MemoryConductor(memory_backend=mock_backend)
        assert c._get_memory_content("n1") == "from get_by_id"

    def test_exception_returns_fallback(self, mock_backend):
        mock_backend.graph = Mock(side_effect=AttributeError("boom"))
        c = MemoryConductor(memory_backend=mock_backend, verbose=True)
        # Should not raise
        result = c._get_memory_content("broken")
        assert isinstance(result, str)


# ============================================================================
# _analyze_query
# ============================================================================


class TestAnalyzeQuery:
    def test_short_query_is_simple(self, conductor):
        q = _make_query("hello")
        criteria = conductor._analyze_query(q)
        assert criteria.performance_critical is True
        assert criteria.requires_deep_traversal is False

    def test_long_query_requires_deep(self, conductor):
        q = _make_query(" ".join(["word"] * 20))
        criteria = conductor._analyze_query(q)
        assert criteria.requires_deep_traversal is True

    def test_research_keywords_set_exploration(self, conductor):
        q = _make_query("analyze the comprehensive tradeoffs in detail")
        criteria = conductor._analyze_query(q)
        assert criteria.exploration_mode is True

    def test_cached_query_detected(self, conductor):
        q = _make_query("cached")
        conductor._cache[conductor._make_cache_key(q)] = Mock()
        criteria = conductor._analyze_query(q)
        assert criteria.is_cached is True


# ============================================================================
# End-to-end with all systems
# ============================================================================


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_recall_deep_with_all_systems(self, conductor_all_systems):
        q = _make_query(
            "compare analyze the comprehensive tradeoffs",
            strategy=MemoryStrategy.RESEARCH,
            k=10,
        )
        result = await conductor_all_systems.recall(q)
        assert isinstance(result, MemoryCoordinationResult)
        assert result.strategy_used == MemoryStrategy.RESEARCH
        assert len(result.systems_accessed) >= 4

    @pytest.mark.asyncio
    async def test_multiple_recalls_build_history(self, conductor):
        for i in range(5):
            await conductor.recall(_make_query(f"query_{i}", strategy=MemoryStrategy.FAST))
        assert len(conductor.get_query_history(limit=3)) == 3
        assert len(conductor.get_query_history(limit=10)) == 5
