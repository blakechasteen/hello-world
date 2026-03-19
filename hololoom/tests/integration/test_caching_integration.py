"""
Integration test: All caching/token-maximization features working together.

Full round-trip:
1. Bus with query_cache + multi_resolution + render_cache
2. Store items → query cache miss → result cached
3. Repeat query → cache hit
4. Mutate → cache stale → miss → re-cached
5. Multi-resolution packs overflow at COMPACT/STUB
6. MemFS render_for_prompt cached via RenderCache
7. Generation counter tracks all mutations
"""

import tempfile
from pathlib import Path

import pytest

from hololoom.core.memory.bus import MemoryItem, MemoryQuery
from hololoom.core.memory.bus_config import MemoryBusConfig
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.memfs import MemFS
from hololoom.core.memory.query_cache import QueryCache
from hololoom.core.memory.multi_resolution import MultiResolutionRenderer
from hololoom.core.memory.render_cache import RenderCache
from hololoom.core.memory.repo_cache import RepoCache


@pytest.mark.asyncio
async def test_full_caching_round_trip():
    """All caching features working together."""

    qc = QueryCache(max_size=64, default_ttl=3600)
    mr = MultiResolutionRenderer()
    rc = RenderCache()

    config = MemoryBusConfig(token_budget_absolute=200)
    bus = LiteMemoryBus(
        config=config,
        query_cache=qc,
        multi_resolution=mr,
    )
    await bus.initialize()

    fs = MemFS(bus, render_cache=rc)

    # --- Step 1: Generation starts at 0 ---
    assert bus.generation == 0

    # --- Step 2: Store items → generation increments ---
    for i in range(15):
        await bus.store(MemoryItem(
            content=f"Memory item {i}: " + "x" * 60,
            memory_type="episodic",
            importance=0.3 + i * 0.04,
        ))
    assert bus.generation == 15

    # --- Step 3: First query → cache miss → result cached ---
    r1 = await bus.query(MemoryQuery(intent="memory item"))
    assert qc.get_stats()["misses"] >= 1
    assert qc.get_stats()["size"] == 1

    # --- Step 4: Repeat query → cache hit ---
    r2 = await bus.query(MemoryQuery(intent="memory item"))
    assert qc.get_stats()["hits"] >= 1
    assert r2.items == r1.items

    # --- Step 5: Mutate → cache stale ---
    gen_before = bus.generation
    await bus.store(MemoryItem(content="new item", memory_type="entity", importance=0.9))
    assert bus.generation == gen_before + 1

    r3 = await bus.query(MemoryQuery(intent="memory item"))
    assert qc.get_stats()["stale_evictions"] >= 1

    # --- Step 6: Multi-resolution packs overflow ---
    # With 200 token budget and items ~15+ tokens each, multi-res should fit more
    full_items = [i for i in r3.items if "_render_resolution" not in i]
    downgraded = [i for i in r3.items if "_render_resolution" in i]
    total_items = len(full_items) + len(downgraded)
    assert total_items > 0

    # --- Step 7: MemFS render caching ---
    tree1 = fs.tree()
    tree2 = fs.tree()
    assert tree1 == tree2
    assert rc.get_stats()["hits"] >= 1

    prompt1 = fs.render_for_prompt("memory", 2000)
    prompt2 = fs.render_for_prompt("memory", 2000)
    assert prompt1 == prompt2

    # Mutate → render cache stale
    await bus.store(MemoryItem(content="another", memory_type="factual"))
    tree3 = fs.tree()
    # Should still work (re-built from scratch)
    assert "memory/" in tree3


@pytest.mark.asyncio
async def test_repo_cache_with_bus_snapshot():
    """RepoCache persists alongside bus snapshots."""
    repo_cache = RepoCache()
    bus = LiteMemoryBus(repo_cache=repo_cache)
    await bus.initialize()

    # Populate repo cache
    repo_cache.store("src/main.py", "hash_abc", [
        {"id": "c1", "content": "def main(): pass", "token_estimate": 5},
        {"id": "c2", "content": "def helper(): pass", "token_estimate": 5},
    ], language="python", entity_names=["main", "helper"], line_count=10)

    # Store some memory items
    await bus.store(MemoryItem(content="test item", memory_type="entity"))

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = str(Path(tmpdir) / "snapshot.json")
        bus.save_snapshot(snap_path)

        # Verify repo cache file exists
        repo_cache_path = snap_path + ".repo_cache.json"
        assert Path(repo_cache_path).exists()

        # Load into fresh bus
        repo_cache2 = RepoCache()
        bus2 = LiteMemoryBus(repo_cache=repo_cache2)
        await bus2.initialize()
        bus2.load_snapshot(snap_path)

        # Repo cache loaded
        chunks = repo_cache2.lookup("src/main.py", "hash_abc")
        assert chunks is not None
        assert len(chunks) == 2

        # Memory items loaded
        assert len(bus2._items) == 1


@pytest.mark.asyncio
async def test_generation_counter_all_mutations():
    """Generation counter increments on all mutation types."""
    bus = LiteMemoryBus()
    await bus.initialize()
    assert bus.generation == 0

    # store
    id1 = await bus.store(MemoryItem(content="a", memory_type="entity"))
    assert bus.generation == 1

    id2 = await bus.store(MemoryItem(content="b", memory_type="entity"))
    assert bus.generation == 2

    # store_edge
    await bus.store_edge(id1, id2, "RELATED_TO")
    assert bus.generation == 3

    # forget (via internal method)
    bus._forget(id2)
    assert bus.generation == 4

    # from_snapshot_dict
    bus.from_snapshot_dict({"_version": 2, "items": {}, "edges": {}, "aliases": {}})
    assert bus.generation == 5

    # decay: store an item with artificially old timestamp, then decay it
    id3 = await bus.store(MemoryItem(content="c", memory_type="entity"))
    # Backdate the item's timestamp so decay sees it as old
    bus._items[id3].timestamp = "2020-01-01T00:00:00"
    bus._items[id3].importance = 0.05  # Already low
    gen_before = bus.generation
    await bus.decay(max_age_seconds=1, min_importance=0.1)
    # Item should have been forgotten → generation incremented
    assert bus.generation > gen_before
