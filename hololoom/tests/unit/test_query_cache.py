"""Unit tests for QueryCache — LRU+TTL query result caching."""

import time
import pytest

from hololoom.core.memory.bus import MemoryQuery, MemoryResult
from hololoom.core.memory.query_cache import QueryCache


def _make_query(intent="test query", entity_ids=None, max_results=5):
    return MemoryQuery(intent=intent, entity_ids=entity_ids or [], max_results=max_results)


def _make_result(items=None, path="graph"):
    return MemoryResult(
        items=items or [{"id": "abc", "content": "test"}],
        query=_make_query(),
        resolution_path=path,
        token_estimate=10,
    )


class TestQueryCache:
    def test_miss_on_empty(self):
        cache = QueryCache()
        assert cache.get(_make_query(), current_generation=0) is None

    def test_hit_after_put(self):
        cache = QueryCache()
        q = _make_query()
        r = _make_result()
        cache.put(q, r, generation=1)
        assert cache.get(q, current_generation=1) is r

    def test_stale_after_generation_change(self):
        cache = QueryCache()
        q = _make_query()
        cache.put(q, _make_result(), generation=1)
        # Generation changed — cache is stale
        assert cache.get(q, current_generation=2) is None

    def test_ttl_expiry(self):
        cache = QueryCache(default_ttl=0.01)
        q = _make_query()
        cache.put(q, _make_result(), generation=1)
        time.sleep(0.02)
        assert cache.get(q, current_generation=1) is None

    def test_custom_ttl(self):
        cache = QueryCache(default_ttl=3600)
        q = _make_query()
        cache.put(q, _make_result(), generation=1, ttl=0.01)
        time.sleep(0.02)
        assert cache.get(q, current_generation=1) is None

    def test_lru_eviction(self):
        cache = QueryCache(max_size=2)
        q1, q2, q3 = _make_query("a"), _make_query("b"), _make_query("c")
        cache.put(q1, _make_result(), generation=1)
        cache.put(q2, _make_result(), generation=1)
        cache.put(q3, _make_result(), generation=1)
        # q1 should have been evicted
        assert cache.get(q1, current_generation=1) is None
        assert cache.get(q2, current_generation=1) is not None
        assert cache.get(q3, current_generation=1) is not None

    def test_different_queries_different_keys(self):
        cache = QueryCache()
        q1 = _make_query("intent a")
        q2 = _make_query("intent b")
        r1 = _make_result(path="exact")
        r2 = _make_result(path="graph")
        cache.put(q1, r1, generation=1)
        cache.put(q2, r2, generation=1)
        assert cache.get(q1, current_generation=1).resolution_path == "exact"
        assert cache.get(q2, current_generation=1).resolution_path == "graph"

    def test_stats(self):
        cache = QueryCache()
        q = _make_query()
        cache.put(q, _make_result(), generation=1)
        cache.get(q, current_generation=1)  # hit
        cache.get(_make_query("miss"), current_generation=1)  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stale_eviction_stats(self):
        cache = QueryCache()
        q = _make_query()
        cache.put(q, _make_result(), generation=1)
        cache.get(q, current_generation=2)  # stale
        stats = cache.get_stats()
        assert stats["stale_evictions"] == 1

    def test_clear(self):
        cache = QueryCache()
        cache.put(_make_query(), _make_result(), generation=1)
        cache.clear()
        assert cache.get_stats()["size"] == 0

    def test_normalize_query_deterministic(self):
        q1 = _make_query("test", entity_ids=["a", "b"])
        q2 = _make_query("test", entity_ids=["b", "a"])
        # Sorted entity_ids → same key
        assert QueryCache._normalize_query(q1) == QueryCache._normalize_query(q2)

    def test_hit_count_tracks(self):
        cache = QueryCache()
        q = _make_query()
        cache.put(q, _make_result(), generation=1)
        cache.get(q, current_generation=1)
        cache.get(q, current_generation=1)
        cache.get(q, current_generation=1)
        stats = cache.get_stats()
        assert stats["hits"] == 3


@pytest.mark.asyncio
async def test_bus_integration():
    """QueryCache works when wired into LiteMemoryBus."""
    from hololoom.core.memory.bus import MemoryItem
    from hololoom.core.memory.lite_bus import LiteMemoryBus

    qc = QueryCache()
    bus = LiteMemoryBus(query_cache=qc)
    await bus.initialize()

    await bus.store(MemoryItem(content="garlic is aromatic", memory_type="entity"))
    assert qc.get_stats()["size"] == 0  # Nothing cached yet

    # First query: cache miss
    r1 = await bus.query(MemoryQuery(intent="garlic"))
    assert qc.get_stats()["misses"] == 1
    assert qc.get_stats()["size"] == 1  # Now cached

    # Second query: cache hit (same generation)
    r2 = await bus.query(MemoryQuery(intent="garlic"))
    assert qc.get_stats()["hits"] == 1
    assert r2.items == r1.items

    # Mutate → cache stale
    await bus.store(MemoryItem(content="onion is pungent", memory_type="entity"))
    r3 = await bus.query(MemoryQuery(intent="garlic"))
    assert qc.get_stats()["stale_evictions"] >= 1
