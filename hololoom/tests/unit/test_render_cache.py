"""Unit tests for RenderCache — generation-tracked MemFS output cache."""

import pytest

from hololoom.core.memory.render_cache import RenderCache


class TestPromptCache:
    def test_miss_on_empty(self):
        cache = RenderCache()
        assert cache.get_prompt("query", 2000, generation=0) is None

    def test_hit_after_put(self):
        cache = RenderCache()
        cache.put_prompt("query", 2000, generation=1, rendered="rendered output")
        result = cache.get_prompt("query", 2000, generation=1)
        assert result == "rendered output"

    def test_stale_after_generation_change(self):
        cache = RenderCache()
        cache.put_prompt("query", 2000, generation=1, rendered="old")
        assert cache.get_prompt("query", 2000, generation=2) is None

    def test_different_queries(self):
        cache = RenderCache()
        cache.put_prompt("a", 2000, generation=1, rendered="result_a")
        cache.put_prompt("b", 2000, generation=1, rendered="result_b")
        assert cache.get_prompt("a", 2000, generation=1) == "result_a"
        assert cache.get_prompt("b", 2000, generation=1) == "result_b"

    def test_different_budgets(self):
        cache = RenderCache()
        cache.put_prompt("q", 1000, generation=1, rendered="small")
        cache.put_prompt("q", 4000, generation=1, rendered="big")
        assert cache.get_prompt("q", 1000, generation=1) == "small"
        assert cache.get_prompt("q", 4000, generation=1) == "big"

    def test_lru_eviction(self):
        cache = RenderCache(max_entries=2)
        cache.put_prompt("a", 100, generation=1, rendered="a")
        cache.put_prompt("b", 100, generation=1, rendered="b")
        cache.put_prompt("c", 100, generation=1, rendered="c")
        assert cache.get_prompt("a", 100, generation=1) is None
        assert cache.get_prompt("b", 100, generation=1) == "b"

    def test_hit_count_tracks(self):
        cache = RenderCache()
        cache.put_prompt("q", 100, generation=1, rendered="r")
        cache.get_prompt("q", 100, generation=1)
        cache.get_prompt("q", 100, generation=1)
        assert cache._prompt_cache[cache._prompt_key("q", 100)].hit_count == 2


class TestTreeCache:
    def test_miss_on_empty(self):
        cache = RenderCache()
        assert cache.get_tree(generation=0) is None

    def test_hit_after_put(self):
        cache = RenderCache()
        cache.put_tree(generation=1, tree_str="memory/\n  entity/ (5 items)")
        result = cache.get_tree(generation=1)
        assert "entity/" in result

    def test_stale_after_generation_change(self):
        cache = RenderCache()
        cache.put_tree(generation=1, tree_str="old tree")
        assert cache.get_tree(generation=2) is None

    def test_tree_independent_of_prompt(self):
        cache = RenderCache()
        cache.put_tree(generation=1, tree_str="tree")
        cache.put_prompt("q", 100, generation=1, rendered="prompt")
        assert cache.get_tree(generation=1) == "tree"
        assert cache.get_prompt("q", 100, generation=1) == "prompt"


class TestStats:
    def test_stats_report(self):
        cache = RenderCache()
        cache.put_prompt("q", 100, generation=1, rendered="r")
        cache.get_prompt("q", 100, generation=1)  # hit
        cache.get_prompt("miss", 100, generation=1)  # miss
        cache.get_tree(generation=1)  # miss (no tree cached)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["prompt_cache_size"] == 1
        assert stats["tree_cached"] is False

    def test_clear(self):
        cache = RenderCache()
        cache.put_prompt("q", 100, generation=1, rendered="r")
        cache.put_tree(generation=1, tree_str="t")
        cache.clear()
        assert cache.get_prompt("q", 100, generation=1) is None
        assert cache.get_tree(generation=1) is None


@pytest.mark.asyncio
async def test_memfs_integration():
    """RenderCache works when wired into MemFS."""
    from hololoom.core.memory.bus import MemoryItem
    from hololoom.core.memory.lite_bus import LiteMemoryBus
    from hololoom.core.memory.memfs import MemFS

    bus = LiteMemoryBus()
    await bus.initialize()
    rc = RenderCache()
    fs = MemFS(bus, render_cache=rc)

    await bus.store(MemoryItem(content="garlic is aromatic", memory_type="entity"))

    # First call: cache miss
    tree1 = fs.tree()
    assert rc.get_stats()["misses"] >= 1
    cached_tree = rc.get_stats()["tree_cached"]
    assert cached_tree is True

    # Second call: cache hit
    tree2 = fs.tree()
    assert tree1 == tree2
    assert rc.get_stats()["hits"] >= 1

    # Mutate bus → generation changes → cache stale
    await bus.store(MemoryItem(content="onion is pungent", memory_type="entity"))
    tree3 = fs.tree()
    assert "entity/" in tree3

    # render_for_prompt also cached
    p1 = fs.render_for_prompt("garlic", 2000)
    p2 = fs.render_for_prompt("garlic", 2000)
    assert p1 == p2
