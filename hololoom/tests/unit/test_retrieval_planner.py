"""
Tests for Phase 1: Token-Aware Items + Retrieval Planner.

Covers:
- StoredItem.token_estimate computation and serialization
- RetrievalPlanner query classification
- Cascade short-circuiting via planner integration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from hololoom.core.memory.bus import MemoryQuery, MemoryItem, MemoryResult
from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.core.memory.retrieval_planner import (
    RetrievalPlanner,
    RetrievalPlan,
    RetrievalComplexity,
)


# =============================================================================
# StoredItem.token_estimate
# =============================================================================

class TestTokenEstimate:
    """Verify token_estimate is computed and persisted correctly."""

    def test_default_zero(self):
        item = StoredItem(id="x", content="hello", memory_type="episodic")
        assert item.token_estimate == 0

    @pytest.mark.asyncio
    async def test_computed_on_store(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="Thompson Sampling uses Beta distributions",
            memory_type="factual",
        ))
        stored = bus._items[item_id]
        assert stored.token_estimate > 0
        # ~10 tokens for that content (42 chars // 4 = 10)
        assert stored.token_estimate >= 10

    @pytest.mark.asyncio
    async def test_includes_properties(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="short",
            memory_type="entity",
            properties={"description": "A very long description " * 10},
        ))
        stored = bus._items[item_id]
        # Token estimate should be larger due to properties
        assert stored.token_estimate > 5  # "short" alone = ~1

    def test_to_dict_includes_token_estimate(self):
        item = StoredItem(
            id="x", content="hello", memory_type="episodic",
            token_estimate=42,
        )
        d = item.to_dict()
        assert d["token_estimate"] == 42

    def test_snapshot_round_trip(self):
        original = StoredItem(
            id="x", content="hello world", memory_type="factual",
            token_estimate=15,
        )
        snap = original.to_snapshot()
        restored = StoredItem.from_snapshot(snap)
        assert restored.token_estimate == 15

    def test_v1_snapshot_backfill(self):
        """V1 snapshots without token_estimate get backfilled."""
        v1_snap = {
            "id": "x",
            "content": "hello world testing",
            "memory_type": "factual",
            # no token_estimate key
        }
        restored = StoredItem.from_snapshot(v1_snap)
        assert restored.token_estimate > 0
        expected = max(1, len("hello world testing") // 4)
        assert restored.token_estimate == expected

    @pytest.mark.asyncio
    async def test_build_result_uses_cached(self):
        """_build_result should prefer cached token_estimate over heuristic."""
        bus = LiteMemoryBus()
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="test content for tokens",
            memory_type="episodic",
        ))
        stored = bus._items[item_id]
        cached = stored.token_estimate

        # Query to trigger _build_result
        result = await bus.query(MemoryQuery(
            intent="test content",
        ))
        assert result.items
        assert result.token_estimate == cached


# =============================================================================
# Retrieval Planner Classification
# =============================================================================

class TestRetrievalPlanner:
    """Verify planner classifies queries correctly."""

    def setup_method(self):
        self.planner = RetrievalPlanner()

    def test_entity_ids_is_point(self):
        q = MemoryQuery(intent="", entity_ids=["abc123"])
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.POINT
        assert plan.max_cascade_level == 2
        assert plan.expand_graph is False

    def test_structured_only_is_point(self):
        q = MemoryQuery(intent="", entity_type="ingredient")
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.POINT
        assert plan.max_cascade_level == 2

    def test_time_filter_only_is_point(self):
        q = MemoryQuery(intent="", time_after="2026-01-01")
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.POINT

    def test_intent_plus_type_is_neighborhood(self):
        q = MemoryQuery(intent="garlic recipes", entity_type="recipe")
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.NEIGHBORHOOD
        assert plan.max_cascade_level == 3
        assert plan.expand_graph is True

    def test_intent_plus_memory_type_is_neighborhood(self):
        q = MemoryQuery(intent="what happened", memory_type="episodic")
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.NEIGHBORHOOD

    def test_bare_intent_is_exploratory(self):
        q = MemoryQuery(intent="tell me about Thompson Sampling")
        plan = self.planner.plan(q)
        assert plan.complexity == RetrievalComplexity.EXPLORATORY
        assert plan.max_cascade_level == 4
        assert plan.expand_graph is True

    def test_max_results_preserved(self):
        q = MemoryQuery(intent="test", max_results=20)
        plan = self.planner.plan(q)
        assert plan.max_results == 20

    def test_empty_query_is_exploratory(self):
        q = MemoryQuery(intent="")
        plan = self.planner.plan(q)
        # No entity_ids, no structured, no intent → falls through to EXPLORATORY
        assert plan.complexity == RetrievalComplexity.EXPLORATORY


# =============================================================================
# Cascade Short-Circuiting
# =============================================================================

class TestCascadeShortCircuit:
    """Verify planner prevents unnecessary cascade levels."""

    @pytest.mark.asyncio
    async def test_point_plan_skips_semantic(self):
        """With POINT plan, semantic_fn should never be called."""
        semantic_mock = AsyncMock(return_value=[])
        planner = RetrievalPlanner()
        bus = LiteMemoryBus(
            semantic_fn=semantic_mock,
            retrieval_planner=planner,
        )
        await bus.initialize()

        # Store an item and query by entity_id
        item_id = await bus.store(MemoryItem(
            content="test item",
            memory_type="entity",
        ))
        result = await bus.query(MemoryQuery(
            intent="", entity_ids=[item_id],
        ))
        assert result.items  # Found via L1
        semantic_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_neighborhood_plan_skips_semantic(self):
        """With NEIGHBORHOOD plan (max_level=3), semantic_fn not called."""
        semantic_mock = AsyncMock(return_value=[])
        planner = RetrievalPlanner()
        bus = LiteMemoryBus(
            semantic_fn=semantic_mock,
            retrieval_planner=planner,
        )
        await bus.initialize()

        await bus.store(MemoryItem(
            content="garlic is delicious",
            memory_type="factual",
            properties={"type": "ingredient"},
        ))
        # This query has intent + entity_type → NEIGHBORHOOD (L1-3)
        result = await bus.query(MemoryQuery(
            intent="garlic",
            entity_type="ingredient",
        ))
        # Should find via L2 (structured) or L3 (graph), never hit L4
        semantic_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_exploratory_plan_allows_semantic(self):
        """With EXPLORATORY plan, semantic_fn IS called if earlier levels miss."""
        semantic_mock = AsyncMock(return_value=[{
            "id": "fake", "content": "found by semantic", "token_estimate": 10,
        }])
        planner = RetrievalPlanner()
        bus = LiteMemoryBus(
            semantic_fn=semantic_mock,
            retrieval_planner=planner,
        )
        await bus.initialize()

        # Query with no matching items in L1-3, should reach L4
        result = await bus.query(MemoryQuery(
            intent="quantum entanglement",
        ))
        semantic_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_planner_full_cascade(self):
        """Without planner, all levels are attempted (backward compat)."""
        semantic_mock = AsyncMock(return_value=[])
        bus = LiteMemoryBus(semantic_fn=semantic_mock)
        await bus.initialize()

        result = await bus.query(MemoryQuery(intent="anything"))
        # L4 should be called even though nothing found in L3
        semantic_mock.assert_called_once()


# =============================================================================
# Snapshot Version Compatibility
# =============================================================================

class TestSnapshotVersioning:
    """Verify v1 ↔ v2 snapshot compatibility."""

    @pytest.mark.asyncio
    async def test_v2_snapshot_has_version_2(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="test", memory_type="episodic"))
        snap = bus.to_snapshot_dict()
        assert snap["_version"] == 2

    @pytest.mark.asyncio
    async def test_v1_snapshot_loads_with_backfill(self):
        """V1 snapshots (no token_estimate) should load and backfill."""
        bus = LiteMemoryBus()
        await bus.initialize()

        v1_snapshot = {
            "_version": 1,
            "items": {
                "item1": {
                    "id": "item1",
                    "content": "hello world",
                    "memory_type": "episodic",
                }
            },
            "edges": {},
            "aliases": {},
        }
        bus.from_snapshot_dict(v1_snapshot)
        assert "item1" in bus._items
        assert bus._items["item1"].token_estimate > 0
