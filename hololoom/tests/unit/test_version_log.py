"""
Tests for Phase 2: Version Log.

Covers:
- MemoryDelta recording
- Replay (full and since a point)
- Rollback to reconstruct snapshot state
- Diff between two points
- Milestone recording and filtering
- Save/load persistence
- Integration with LiteMemoryBus
- Property test: replay == snapshot
"""

import pytest
import asyncio
from pathlib import Path

from hololoom.core.memory.bus import MemoryQuery, MemoryItem
from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.core.memory.version_log import VersionLog, MemoryDelta


# =============================================================================
# VersionLog Unit Tests
# =============================================================================

class TestVersionLogBasic:
    """Core record/replay/diff operations."""

    def setup_method(self):
        self.log = VersionLog()

    def test_empty_log(self):
        assert self.log.count == 0
        assert self.log.replay() == []
        assert self.log.last_delta() is None

    def test_record_and_count(self):
        delta = MemoryDelta(
            delta_id="d1", timestamp="2026-03-05T10:00:00",
            operation="store", item_id="item1",
            before=None, after={"content": "hello"},
        )
        self.log.record(delta)
        assert self.log.count == 1
        assert self.log.last_delta() == delta

    def test_replay_all(self):
        for i in range(5):
            self.log.record(MemoryDelta(
                delta_id=f"d{i}", timestamp=f"2026-03-05T10:0{i}:00",
                operation="store", item_id=f"item{i}",
                before=None, after={"id": f"item{i}", "content": f"c{i}"},
            ))
        deltas = self.log.replay()
        assert len(deltas) == 5
        assert deltas[0].delta_id == "d0"
        assert deltas[4].delta_id == "d4"

    def test_replay_since(self):
        for i in range(5):
            self.log.record(MemoryDelta(
                delta_id=f"d{i}", timestamp=f"2026-03-05T10:0{i}:00",
                operation="store", item_id=f"item{i}",
                before=None, after={"id": f"item{i}"},
            ))
        # Since d2 (inclusive) → d2, d3, d4
        since = self.log.replay(since="d2")
        assert len(since) == 3
        assert since[0].delta_id == "d2"

    def test_replay_since_unknown_returns_empty(self):
        self.log.record(MemoryDelta(
            delta_id="d0", timestamp="t", operation="store",
            item_id="i", before=None, after={},
        ))
        assert self.log.replay(since="unknown") == []

    def test_diff(self):
        for i in range(5):
            self.log.record(MemoryDelta(
                delta_id=f"d{i}", timestamp="t", operation="store",
                item_id=f"item{i}", before=None, after={},
            ))
        # Diff from d1 to d3 (exclusive of d1, inclusive of d3) → d2, d3
        diffs = self.log.diff("d1", "d3")
        assert len(diffs) == 2
        assert diffs[0].delta_id == "d2"
        assert diffs[1].delta_id == "d3"

    def test_diff_same_point_is_empty(self):
        self.log.record(MemoryDelta(
            delta_id="d0", timestamp="t", operation="store",
            item_id="i", before=None, after={},
        ))
        assert self.log.diff("d0", "d0") == []

    def test_diff_unknown_returns_empty(self):
        assert self.log.diff("unknown1", "unknown2") == []


# =============================================================================
# Rollback
# =============================================================================

class TestRollback:
    """Verify rollback reconstructs correct snapshot state."""

    def test_rollback_single_store(self):
        log = VersionLog()
        log.record(MemoryDelta(
            delta_id="d1", timestamp="t", operation="store", item_id="item1",
            before=None,
            after={"id": "item1", "content": "hello", "memory_type": "episodic",
                   "token_estimate": 2},
        ))
        snap = log.rollback("d1")
        assert "item1" in snap["items"]
        assert snap["items"]["item1"]["content"] == "hello"

    def test_rollback_store_then_forget(self):
        log = VersionLog()
        log.record(MemoryDelta(
            delta_id="d1", timestamp="t", operation="store", item_id="item1",
            before=None, after={"id": "item1", "content": "hello"},
        ))
        log.record(MemoryDelta(
            delta_id="d2", timestamp="t", operation="forget", item_id="item1",
            before={"id": "item1", "content": "hello"}, after=None,
        ))
        # Rollback to d1 → item exists
        snap1 = log.rollback("d1")
        assert "item1" in snap1["items"]
        # Rollback to d2 → item gone
        snap2 = log.rollback("d2")
        assert "item1" not in snap2["items"]

    def test_rollback_merge(self):
        log = VersionLog()
        log.record(MemoryDelta(
            delta_id="d1", timestamp="t", operation="store", item_id="item1",
            before=None, after={"id": "item1", "content": "original"},
        ))
        log.record(MemoryDelta(
            delta_id="d2", timestamp="t", operation="merge", item_id="item1",
            before={"id": "item1", "content": "original"},
            after={"id": "item1", "content": "merged content"},
        ))
        snap = log.rollback("d2")
        assert snap["items"]["item1"]["content"] == "merged content"

    def test_rollback_unknown_raises(self):
        log = VersionLog()
        with pytest.raises(ValueError, match="Unknown delta_id"):
            log.rollback("nonexistent")


# =============================================================================
# Milestones
# =============================================================================

class TestMilestones:
    """GCC COMMIT equivalent."""

    def test_record_milestone(self):
        log = VersionLog()
        m = log.record_milestone("First stable version")
        assert m.operation == "milestone"
        assert m.item_id == ""
        assert "description" in m.metadata
        assert log.count == 1

    def test_milestones_filter(self):
        log = VersionLog()
        log.record(MemoryDelta(
            delta_id="d1", timestamp="t", operation="store",
            item_id="i1", before=None, after={},
        ))
        log.record_milestone("Checkpoint 1")
        log.record(MemoryDelta(
            delta_id="d3", timestamp="t", operation="store",
            item_id="i2", before=None, after={},
        ))
        log.record_milestone("Checkpoint 2")

        milestones = log.milestones()
        assert len(milestones) == 2
        assert milestones[0].metadata["description"] == "Checkpoint 1"
        assert milestones[1].metadata["description"] == "Checkpoint 2"


# =============================================================================
# Persistence
# =============================================================================

class TestPersistence:
    """Save/load round-trip."""

    def test_save_load_round_trip(self, tmp_path):
        log = VersionLog()
        for i in range(10):
            log.record(MemoryDelta(
                delta_id=f"d{i}", timestamp=f"t{i}", operation="store",
                item_id=f"item{i}", before=None,
                after={"id": f"item{i}", "content": f"c{i}"},
            ))
        log.record_milestone("Test milestone")

        path = tmp_path / "test.deltas.json"
        log.save(path)
        assert path.exists()

        log2 = VersionLog()
        loaded = log2.load(path)
        assert loaded is True
        assert log2.count == 11
        assert log2.milestones()[0].metadata["description"] == "Test milestone"

    def test_load_nonexistent_returns_false(self, tmp_path):
        log = VersionLog()
        assert log.load(tmp_path / "nonexistent.json") is False

    def test_max_deltas_eviction(self):
        log = VersionLog(max_deltas=5)
        for i in range(8):
            log.record(MemoryDelta(
                delta_id=f"d{i}", timestamp="t", operation="store",
                item_id=f"i{i}", before=None, after={},
            ))
        assert log.count == 5
        # Oldest 3 evicted
        assert log.replay()[0].delta_id == "d3"


# =============================================================================
# Integration with LiteMemoryBus
# =============================================================================

class TestVersionLogBusIntegration:
    """Verify version log records all bus mutations."""

    @pytest.mark.asyncio
    async def test_store_records_delta(self):
        log = VersionLog()
        bus = LiteMemoryBus(version_log=log)
        await bus.initialize()

        item_id = await bus.store(MemoryItem(
            content="hello world", memory_type="episodic",
        ))
        assert log.count == 1
        delta = log.last_delta()
        assert delta.operation == "store"
        assert delta.item_id == item_id
        assert delta.before is None
        assert delta.after["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_forget_records_delta(self):
        log = VersionLog()
        bus = LiteMemoryBus(version_log=log)
        await bus.initialize()

        item_id = await bus.store(MemoryItem(
            content="temp item", memory_type="episodic",
        ))
        bus._forget(item_id)

        assert log.count == 2  # store + forget
        delta = log.last_delta()
        assert delta.operation == "forget"
        assert delta.item_id == item_id
        assert delta.before["content"] == "temp item"
        assert delta.after is None

    @pytest.mark.asyncio
    async def test_property_replay_equals_snapshot(self):
        """Property test: replay all deltas from empty == to_snapshot_dict()."""
        log = VersionLog()
        bus = LiteMemoryBus(version_log=log)
        await bus.initialize()

        # Store several items
        ids = []
        for i in range(10):
            iid = await bus.store(MemoryItem(
                content=f"item {i}",
                memory_type="episodic" if i % 2 == 0 else "factual",
                importance=0.5 + i * 0.05,
            ))
            ids.append(iid)

        # Forget a few
        bus._forget(ids[2])
        bus._forget(ids[7])

        # Rollback to last delta should match current bus state
        last = log.last_delta()
        reconstructed = log.rollback(last.delta_id)

        actual = bus.to_snapshot_dict()

        # Compare item sets (ignore edges/aliases — rollback doesn't track those yet)
        assert set(reconstructed["items"].keys()) == set(actual["items"].keys())
        for iid in reconstructed["items"]:
            assert reconstructed["items"][iid]["content"] == actual["items"][iid]["content"]

    @pytest.mark.asyncio
    async def test_save_load_with_version_log(self, tmp_path):
        """Snapshot save/load also persists version log."""
        log = VersionLog()
        bus = LiteMemoryBus(version_log=log)
        await bus.initialize()

        await bus.store(MemoryItem(content="hello", memory_type="episodic"))
        await bus.store(MemoryItem(content="world", memory_type="factual"))

        path = str(tmp_path / "bus.json")
        bus.save_snapshot(path)

        # Verify deltas file was created
        deltas_path = tmp_path / "bus.deltas.json"
        assert deltas_path.exists()

        # Load into new bus
        log2 = VersionLog()
        bus2 = LiteMemoryBus(version_log=log2)
        await bus2.initialize()
        loaded = bus2.load_snapshot(path)
        assert loaded
        assert log2.count == 2

    @pytest.mark.asyncio
    async def test_no_version_log_backward_compat(self):
        """Bus without version_log still works fine."""
        bus = LiteMemoryBus()
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="no log", memory_type="episodic",
        ))
        assert item_id in bus._items
        bus._forget(item_id)
        assert item_id not in bus._items
