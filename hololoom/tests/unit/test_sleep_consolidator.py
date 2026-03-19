"""
Tests for Phase 5: Sleep-Time Consolidator.

Covers:
- run_cycle() consolidation and milestone promotion
- Version log records consolidation milestones
- Background task start/stop
- Statistics tracking
- Works without version_log (consolidator=None path)
"""

import asyncio
import pytest

from hololoom.core.memory.bus import MemoryItem
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.sleep_consolidator import SleepConsolidator
from hololoom.core.memory.version_log import VersionLog


# =============================================================================
# run_cycle()
# =============================================================================

class TestRunCycle:

    @pytest.mark.asyncio
    async def test_consolidates_overflowing_type(self):
        """Types with more than max items get consolidated."""
        bus = LiteMemoryBus()
        await bus.initialize()
        # Store 15 episodic items (max_items=10 → should consolidate)
        for i in range(15):
            await bus.store(MemoryItem(
                content=f"episode {i}", memory_type="episodic", importance=0.3,
            ))

        sc = SleepConsolidator(bus, consolidate_max_items=10)
        stats = await sc.run_cycle()

        assert "episodic" in stats["consolidated"]
        assert stats["consolidated"]["episodic"]["before"] == 15
        # After consolidation: should be <= max_items + 1 (summary)
        assert stats["consolidated"]["episodic"]["after"] <= 11

    @pytest.mark.asyncio
    async def test_no_consolidation_under_threshold(self):
        """Types under max_items are not consolidated."""
        bus = LiteMemoryBus()
        await bus.initialize()
        for i in range(5):
            await bus.store(MemoryItem(
                content=f"fact {i}", memory_type="factual",
            ))

        sc = SleepConsolidator(bus, consolidate_max_items=10)
        stats = await sc.run_cycle()
        assert stats["consolidated"] == {}

    @pytest.mark.asyncio
    async def test_milestone_promotion(self):
        """Items with high access + high importance → promoted to permanent."""
        bus = LiteMemoryBus(default_ttl=3600)  # 1hr TTL
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="important frequently accessed item",
            memory_type="entity",
            importance=0.9,
        ))
        # Simulate high access count
        bus._items[item_id].access_count = 10

        sc = SleepConsolidator(
            bus, milestone_access_threshold=5,
            milestone_importance_threshold=0.7,
        )
        stats = await sc.run_cycle()

        assert item_id in stats["promoted"]
        assert bus._items[item_id].ttl_seconds is None  # Now permanent

    @pytest.mark.asyncio
    async def test_no_promotion_already_permanent(self):
        """Items already permanent (ttl=None) are not re-promoted."""
        bus = LiteMemoryBus()  # default_ttl=None → permanent
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="already permanent", memory_type="entity", importance=0.9,
        ))
        bus._items[item_id].access_count = 10

        sc = SleepConsolidator(bus)
        stats = await sc.run_cycle()
        assert item_id not in stats["promoted"]

    @pytest.mark.asyncio
    async def test_no_promotion_low_access(self):
        """Items with low access count are not promoted."""
        bus = LiteMemoryBus(default_ttl=3600)
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="low access", memory_type="entity", importance=0.9,
        ))
        bus._items[item_id].access_count = 2  # Below threshold of 5

        sc = SleepConsolidator(bus)
        stats = await sc.run_cycle()
        assert item_id not in stats["promoted"]


# =============================================================================
# Version Log Integration
# =============================================================================

class TestVersionLogIntegration:

    @pytest.mark.asyncio
    async def test_milestone_recorded(self):
        """Consolidation cycles record milestones in version log."""
        bus = LiteMemoryBus()
        await bus.initialize()
        for i in range(15):
            await bus.store(MemoryItem(
                content=f"episode {i}", memory_type="episodic", importance=0.3,
            ))

        vlog = VersionLog()
        sc = SleepConsolidator(bus, version_log=vlog, consolidate_max_items=10)
        await sc.run_cycle()

        milestones = vlog.milestones()
        assert len(milestones) >= 1
        assert "Sleep consolidation" in milestones[0].metadata["description"]

    @pytest.mark.asyncio
    async def test_no_milestone_when_nothing_happens(self):
        """No milestone recorded when nothing was consolidated or promoted."""
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="one item", memory_type="factual"))

        vlog = VersionLog()
        sc = SleepConsolidator(bus, version_log=vlog)
        await sc.run_cycle()

        assert vlog.milestones() == []


# =============================================================================
# Statistics
# =============================================================================

class TestStatistics:

    @pytest.mark.asyncio
    async def test_cumulative_stats(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        for i in range(15):
            await bus.store(MemoryItem(
                content=f"ep {i}", memory_type="episodic", importance=0.3,
            ))

        sc = SleepConsolidator(bus, consolidate_max_items=10)
        await sc.run_cycle()

        stats = sc.statistics()
        assert stats["total_cycles"] == 1
        assert stats["total_consolidated"] > 0
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_initial_stats(self):
        bus = LiteMemoryBus()
        sc = SleepConsolidator(bus)
        stats = sc.statistics()
        assert stats == {
            "total_cycles": 0,
            "total_promoted": 0,
            "total_consolidated": 0,
            "running": False,
        }


# =============================================================================
# Background Task
# =============================================================================

class TestBackgroundTask:

    @pytest.mark.asyncio
    async def test_start_stop(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        sc = SleepConsolidator(bus)

        await sc.start_background(interval_seconds=1)
        assert sc._running is True
        assert sc._task is not None

        await sc.stop_background()
        assert sc._running is False

    @pytest.mark.asyncio
    async def test_double_start_ignored(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        sc = SleepConsolidator(bus)

        await sc.start_background(interval_seconds=1)
        task1 = sc._task
        await sc.start_background(interval_seconds=1)  # Should be ignored
        assert sc._task is task1

        await sc.stop_background()
