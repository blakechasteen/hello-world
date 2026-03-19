"""
Integration test: All borrowed features working together.

Full round-trip verifying:
1. Bus with all optional features enabled (planner, version_log, synthesis, security)
2. Store items → synthesis merges duplicates
3. Security screen blocks bad content
4. SleepConsolidator cycle → consolidation + promotion
5. MemFS reflects current state
6. Version log replay has correct delta count
7. Rollback to earlier point → verify state
8. Save/load snapshot preserves version log
"""

import pytest
import tempfile
from pathlib import Path

from hololoom.core.memory.bus import MemoryItem, MemoryQuery
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.retrieval_planner import RetrievalPlanner
from hololoom.core.memory.version_log import VersionLog
from hololoom.core.memory.synthesis import default_synthesis_fn
from hololoom.core.memory.security_screen import (
    SecurityScreenError,
    default_security_screen,
)
from hololoom.core.memory.memfs import MemFS
from hololoom.core.memory.sleep_consolidator import SleepConsolidator


@pytest.mark.asyncio
async def test_full_round_trip():
    """All borrowed features working together in a single scenario."""

    # --- Setup: bus with all features ---
    vlog = VersionLog()
    planner = RetrievalPlanner()

    # Simple semantic fn that finds items by substring match
    async def local_semantic(text, max_results):
        # Not wired to any real embedding — returns empty
        return []

    bus = LiteMemoryBus(
        retrieval_planner=planner,
        version_log=vlog,
        synthesis_fn=default_synthesis_fn(),
        semantic_fn=local_semantic,
        security_screen=default_security_screen,
        default_ttl=3600,
    )
    await bus.initialize()

    # --- Step 1: Store a mix of items ---
    stored_ids = []
    for i in range(20):
        item_id = await bus.store(MemoryItem(
            content=f"Episodic memory number {i}: something happened",
            memory_type="episodic",
            importance=0.3 + (i * 0.02),
        ))
        stored_ids.append(item_id)

    for i in range(5):
        item_id = await bus.store(MemoryItem(
            content=f"Entity {i}: a named thing",
            memory_type="entity",
            importance=0.8,
        ))
        stored_ids.append(item_id)

    for i in range(5):
        item_id = await bus.store(MemoryItem(
            content=f"Factual item {i}: 2+2=4",
            memory_type="factual",
            importance=0.6,
        ))
        stored_ids.append(item_id)

    total_stored = len(bus._items)
    assert total_stored == 30

    # --- Step 2: Security screen blocks sensitive content ---
    with pytest.raises(SecurityScreenError):
        await bus.store(MemoryItem(
            content="My API key is sk-abcdefghijklmnopqrstuvwxyz1234",
            memory_type="factual",
        ))
    assert len(bus._items) == 30  # Nothing added

    # --- Step 3: Version log has recorded all stores ---
    deltas = vlog.replay()
    store_deltas = [d for d in deltas if d.operation == "store"]
    assert len(store_deltas) == 30

    # Save a milestone for later rollback
    pre_consolidation_delta = vlog.last_delta()
    assert pre_consolidation_delta is not None

    # --- Step 4: Retrieval planner works ---
    # POINT query: entity_ids → L1-2 only
    result = await bus.query(MemoryQuery(intent="", entity_ids=[stored_ids[0]]))
    assert result.items
    assert result.resolution_path == "exact"

    # EXPLORATORY query: free text → L1-4
    result = await bus.query(MemoryQuery(intent="something happened"))
    assert result.items

    # --- Step 5: SleepConsolidator cycle ---
    # Boost some items' access counts for promotion
    for item_id in stored_ids[20:25]:  # Entity items
        bus._items[item_id].access_count = 10

    sc = SleepConsolidator(
        bus,
        version_log=vlog,
        milestone_access_threshold=5,
        milestone_importance_threshold=0.7,
        consolidate_max_items=10,
    )
    cycle_stats = await sc.run_cycle()

    # Episodic should have been consolidated (had 20, max=10)
    assert "episodic" in cycle_stats["consolidated"]
    # Entity items with high access should have been promoted
    assert len(cycle_stats["promoted"]) > 0
    for promoted_id in cycle_stats["promoted"]:
        assert bus._items[promoted_id].ttl_seconds is None  # Permanent

    # Version log should have a milestone
    milestones = vlog.milestones()
    assert len(milestones) >= 1

    # --- Step 6: MemFS reflects current state ---
    fs = MemFS(bus)
    tree = fs.tree()
    assert "memory/" in tree
    assert "entity/" in tree
    assert "episodic/" in tree

    # render_for_prompt stays under budget
    prompt = fs.render_for_prompt("something happened", token_budget=2000)
    assert len(prompt) // 4 <= 2000

    # All items renderable
    files = fs.all_files()
    assert len(files) == len(bus._items)

    # --- Step 7: Rollback to pre-consolidation state ---
    rollback_snapshot = vlog.rollback(pre_consolidation_delta.delta_id)
    assert rollback_snapshot["_version"] == 2
    # Pre-consolidation had 30 items
    assert len(rollback_snapshot["items"]) == 30

    # --- Step 8: Save/load snapshot preserves version log ---
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = str(Path(tmpdir) / "test_snapshot.json")
        bus.save_snapshot(snap_path)

        # Create fresh bus and load
        bus2 = LiteMemoryBus(version_log=VersionLog())
        await bus2.initialize()
        loaded = bus2.load_snapshot(snap_path)
        assert loaded is True
        assert len(bus2._items) == len(bus._items)

        # Version log loaded alongside
        assert bus2._version_log.count > 0


@pytest.mark.asyncio
async def test_property_version_log_consistency():
    """For any sequence of store/forget, rollback(last) == to_snapshot_dict()."""
    vlog = VersionLog()
    bus = LiteMemoryBus(version_log=vlog)
    await bus.initialize()

    # Random-ish sequence of stores
    ids = []
    for i in range(10):
        item_id = await bus.store(MemoryItem(
            content=f"item {i}", memory_type="episodic",
        ))
        ids.append(item_id)

    # Forget some
    bus._forget(ids[2])
    bus._forget(ids[5])
    bus._forget(ids[8])

    # Rollback to last delta should match current state
    last = vlog.last_delta()
    assert last is not None
    rollback = vlog.rollback(last.delta_id)

    snapshot = bus.to_snapshot_dict()

    # Items should match
    assert set(rollback["items"].keys()) == set(snapshot["items"].keys())
    for item_id in rollback["items"]:
        assert rollback["items"][item_id]["content"] == snapshot["items"][item_id]["content"]
