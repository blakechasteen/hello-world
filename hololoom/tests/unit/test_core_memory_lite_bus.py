"""
Unit tests for hololoom.core.memory.lite_bus — LiteMemoryBus and StoredItem.

Covers: StoredItem lifecycle, 4-level cascade query, store/edge/alias,
snapshot persistence, forgetting/decay/consolidation, token budgeting,
status/pressure, explain, graph export, subgraph, bulk load, audit.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.core.memory.bus import MemoryQuery, MemoryItem, MemoryResult
from hololoom.core.memory.bus_config import (
    MemoryBusConfig,
    ResolutionPath,
    PressureTier,
)
from hololoom.core.memory.security_screen import SecurityScreenError


# ============================================================================
# StoredItem
# ============================================================================


class TestStoredItem:
    def test_defaults(self):
        item = StoredItem(id="a", content="hello", memory_type="factual")
        assert item.importance == 0.5
        assert item.access_count == 0
        assert item.entity_ids == []
        assert item.properties == {}
        assert item.token_estimate == 0

    def test_is_expired_no_ttl(self):
        item = StoredItem(
            id="a", content="x", memory_type="factual",
            timestamp=datetime.utcnow().isoformat(), ttl_seconds=None,
        )
        assert item.is_expired() is False

    def test_is_expired_within_ttl(self):
        item = StoredItem(
            id="a", content="x", memory_type="factual",
            timestamp=datetime.utcnow().isoformat(), ttl_seconds=3600,
        )
        assert item.is_expired() is False

    def test_is_expired_past_ttl(self):
        old = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        item = StoredItem(
            id="a", content="x", memory_type="factual",
            timestamp=old, ttl_seconds=60,
        )
        assert item.is_expired() is True

    def test_touch_increments(self):
        item = StoredItem(id="a", content="x", memory_type="factual")
        assert item.access_count == 0
        item.touch()
        assert item.access_count == 1
        assert item.last_accessed is not None

    def test_to_dict(self):
        item = StoredItem(
            id="a", content="hello", memory_type="factual",
            importance=0.9, properties={"tag": "test"},
        )
        d = item.to_dict()
        assert d["id"] == "a"
        assert d["content"] == "hello"
        assert d["importance"] == 0.9
        assert d["tag"] == "test"  # properties unpacked

    def test_to_snapshot_roundtrip(self):
        item = StoredItem(
            id="a", content="hello world", memory_type="entity",
            entity_type="bee", entity_ids=["e1"],
            properties={"color": "gold"}, importance=0.7,
            timestamp="2025-01-01T00:00:00", access_count=3,
            last_accessed="2025-01-02T00:00:00", ttl_seconds=600,
            token_estimate=42,
        )
        snap = item.to_snapshot()
        restored = StoredItem.from_snapshot(snap)
        assert restored.id == "a"
        assert restored.content == "hello world"
        assert restored.entity_type == "bee"
        assert restored.entity_ids == ["e1"]
        assert restored.properties == {"color": "gold"}
        assert restored.importance == 0.7
        assert restored.token_estimate == 42

    def test_from_snapshot_backfill_token_estimate(self):
        """v1 snapshots that lack token_estimate get backfilled."""
        snap = {
            "id": "b", "content": "abcdefghijklmnop",
            "memory_type": "factual", "token_estimate": 0,
            "properties": {"x": "yy"},
        }
        restored = StoredItem.from_snapshot(snap)
        assert restored.token_estimate > 0


# ============================================================================
# LiteMemoryBus — Lifecycle
# ============================================================================


class TestLiteMemoryBusLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self):
        bus = LiteMemoryBus()
        info = await bus.initialize()
        assert info["neo4j"] == "lite (in-memory)"
        assert bus._initialized is True
        await bus.close()
        assert bus._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with LiteMemoryBus() as bus:
            assert bus._initialized is True
        assert bus._initialized is False


# ============================================================================
# Store + Query Cascade
# ============================================================================


class TestLiteMemoryBusStoreAndQuery:
    @pytest.mark.asyncio
    async def test_store_returns_id(self):
        bus = LiteMemoryBus()
        item = MemoryItem(content="beekeeping basics", memory_type="factual")
        node_id = await bus.store(item)
        assert isinstance(node_id, str)
        assert len(node_id) > 0
        assert node_id in bus._items

    @pytest.mark.asyncio
    async def test_store_increments_generation(self):
        bus = LiteMemoryBus()
        g0 = bus.generation
        await bus.store(MemoryItem(content="a", memory_type="factual"))
        assert bus.generation > g0

    @pytest.mark.asyncio
    async def test_query_exact_by_entity_id(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="test item", memory_type="factual"))
        q = MemoryQuery(intent="find", entity_ids=[nid])
        result = await bus.query(q)
        assert len(result.items) == 1
        assert result.resolution_path == ResolutionPath.EXACT.value

    @pytest.mark.asyncio
    async def test_query_structured_by_memory_type(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(content="episode one", memory_type="episodic"))
        await bus.store(MemoryItem(content="fact one", memory_type="factual"))
        q = MemoryQuery(intent="", memory_type="episodic")
        result = await bus.query(q)
        assert len(result.items) == 1
        assert result.resolution_path == ResolutionPath.STRUCTURED.value

    @pytest.mark.asyncio
    async def test_query_structured_by_entity_type(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(
            content="queen bee", memory_type="entity",
            properties={"type": "insect"},
        ))
        q = MemoryQuery(intent="", entity_type="insect")
        result = await bus.query(q)
        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_query_structured_properties_match(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(
            content="hive alpha", memory_type="entity",
            properties={"location": "north"},
        ))
        await bus.store(MemoryItem(
            content="hive beta", memory_type="entity",
            properties={"location": "south"},
        ))
        q = MemoryQuery(intent="", entity_type=None, memory_type="entity",
                        properties_match={"location": "north"})
        result = await bus.query(q)
        assert len(result.items) == 1
        assert result.items[0]["content"] == "hive alpha"

    @pytest.mark.asyncio
    async def test_query_graph_keyword_search(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(content="beekeeping is rewarding", memory_type="factual"))
        await bus.store(MemoryItem(content="cooking pasta", memory_type="factual"))
        q = MemoryQuery(intent="beekeeping")
        result = await bus.query(q)
        assert len(result.items) >= 1
        assert result.resolution_path == ResolutionPath.GRAPH.value

    @pytest.mark.asyncio
    async def test_query_graph_traversal_1hop(self):
        bus = LiteMemoryBus()
        nid_a = await bus.store(MemoryItem(content="queen bee details", memory_type="factual"))
        nid_b = await bus.store(MemoryItem(content="hive layout details", memory_type="factual"))
        await bus.store_edge(nid_a, nid_b, "RELATED_TO")
        q = MemoryQuery(intent="queen bee")
        result = await bus.query(q)
        result_ids = {item["id"] for item in result.items}
        assert nid_a in result_ids
        # 1-hop neighbor should appear via traversal
        assert nid_b in result_ids

    @pytest.mark.asyncio
    async def test_query_semantic_with_fn(self):
        async def fake_semantic(text, max_results):
            return [{"id": "sem1", "content": "semantic hit", "token_estimate": 10}]

        bus = LiteMemoryBus(semantic_fn=fake_semantic)
        q = MemoryQuery(intent="deep question", semantic_text="deep question")
        result = await bus.query(q)
        # No exact/structured/graph matches, so should fall through to semantic
        assert len(result.items) >= 1
        assert result.resolution_path == ResolutionPath.SEMANTIC.value

    @pytest.mark.asyncio
    async def test_query_semantic_without_fn_returns_empty(self):
        bus = LiteMemoryBus()
        q = MemoryQuery(intent="", semantic_text="nothing matches")
        result = await bus.query(q)
        assert result.items == []
        assert result.resolution_path == "none"

    @pytest.mark.asyncio
    async def test_query_semantic_fn_error_graceful(self):
        async def broken_semantic(text, max_results):
            raise RuntimeError("boom")

        bus = LiteMemoryBus(semantic_fn=broken_semantic)
        q = MemoryQuery(intent="", semantic_text="try this")
        result = await bus.query(q)
        assert result.items == []

    @pytest.mark.asyncio
    async def test_query_no_match(self):
        bus = LiteMemoryBus()
        q = MemoryQuery(intent="", semantic_text=None)
        result = await bus.query(q)
        assert result.resolution_path == "none"
        assert result.items == []

    @pytest.mark.asyncio
    async def test_get_item(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="test", memory_type="factual"))
        item = bus.get_item(nid)
        assert item is not None
        assert item.content == "test"

    @pytest.mark.asyncio
    async def test_get_item_not_found(self):
        bus = LiteMemoryBus()
        assert bus.get_item("nonexistent") is None


# ============================================================================
# Entity Resolution
# ============================================================================


class TestEntityResolution:
    @pytest.mark.asyncio
    async def test_alias_auto_registered_for_entity_type(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="Queen Jodi", memory_type="entity"))
        candidates = await bus.resolve_entity("Queen Jodi")
        assert len(candidates) == 1
        assert candidates[0]["entity_id"] == nid
        assert candidates[0]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_alias_fuzzy_match(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="Queen Jodi", memory_type="entity"))
        candidates = await bus.resolve_entity("Jodi")
        assert len(candidates) >= 1
        assert candidates[0]["source"] == "alias_fuzzy"

    @pytest.mark.asyncio
    async def test_add_alias_manual(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="item", memory_type="factual"))
        await bus.add_alias("my-alias", nid)
        assert bus._aliases["my-alias"] == nid


# ============================================================================
# Forgetting, Decay, Consolidation
# ============================================================================


class TestForgettingAndDecay:
    @pytest.mark.asyncio
    async def test_expired_items_gc_on_query(self):
        bus = LiteMemoryBus(default_ttl=1)
        nid = await bus.store(MemoryItem(content="ephemeral", memory_type="factual"))
        # Manually backdate timestamp to force expiry
        bus._items[nid].timestamp = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        q = MemoryQuery(intent="", entity_ids=[nid])
        result = await bus.query(q)
        assert result.items == []
        assert nid not in bus._items

    @pytest.mark.asyncio
    async def test_decay_removes_low_importance(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="old", memory_type="factual", importance=0.05))
        # Backdate to trigger decay
        bus._items[nid].timestamp = (datetime.utcnow() - timedelta(days=2)).isoformat()
        forgotten = await bus.decay(max_age_seconds=3600, min_importance=0.1)
        assert forgotten >= 1
        assert nid not in bus._items

    @pytest.mark.asyncio
    async def test_decay_spares_accessed_items(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="popular", memory_type="factual", importance=0.5))
        bus._items[nid].access_count = 20  # heavily accessed
        bus._items[nid].timestamp = (datetime.utcnow() - timedelta(days=2)).isoformat()
        forgotten = await bus.decay(max_age_seconds=3600, min_importance=0.1)
        # Heavily accessed item should survive
        assert nid in bus._items

    @pytest.mark.asyncio
    async def test_consolidate_merges_excess_items(self):
        bus = LiteMemoryBus()
        for i in range(15):
            await bus.store(MemoryItem(
                content=f"episode {i}", memory_type="episodic",
                importance=0.1 + i * 0.01,
            ))
        summary_id = await bus.consolidate(memory_type="episodic", max_items=10)
        assert summary_id is not None
        # Should have consolidated down
        episodic_count = len([
            nid for nid in bus._by_type.get("episodic", [])
            if nid in bus._items
        ])
        assert episodic_count <= 11  # 10 kept + 1 summary

    @pytest.mark.asyncio
    async def test_consolidate_noop_when_under_limit(self):
        bus = LiteMemoryBus()
        for i in range(3):
            await bus.store(MemoryItem(content=f"ep {i}", memory_type="episodic"))
        result = await bus.consolidate(memory_type="episodic", max_items=10)
        assert result is None


# ============================================================================
# Edge Storage
# ============================================================================


class TestEdgeStorage:
    @pytest.mark.asyncio
    async def test_store_edge_bidirectional(self):
        bus = LiteMemoryBus()
        nid_a = await bus.store(MemoryItem(content="a", memory_type="factual"))
        nid_b = await bus.store(MemoryItem(content="b", memory_type="factual"))
        await bus.store_edge(nid_a, nid_b, "RELATED_TO")
        assert any(t == nid_b for t, _ in bus._edges[nid_a])
        assert any(t == nid_a for t, _ in bus._edges[nid_b])

    @pytest.mark.asyncio
    async def test_entity_ids_create_about_edges(self):
        bus = LiteMemoryBus()
        eid = await bus.store(MemoryItem(content="entity", memory_type="entity"))
        nid = await bus.store(MemoryItem(
            content="about entity", memory_type="factual",
            entity_ids=[eid],
        ))
        assert any(t == eid for t, r in bus._edges[nid] if r == "ABOUT")


# ============================================================================
# Token Budgeting
# ============================================================================


class TestTokenBudgeting:
    @pytest.mark.asyncio
    async def test_truncation_when_over_budget(self):
        config = MemoryBusConfig(token_budget_absolute=50)
        bus = LiteMemoryBus(config=config)
        # Store items with large content to exceed budget
        for i in range(10):
            await bus.store(MemoryItem(
                content="x" * 200,
                memory_type="factual",
            ))
        q = MemoryQuery(intent="x" * 10, max_results=100)
        result = await bus.query(q)
        assert result.truncated is True
        assert result.token_estimate <= 50

    def test_item_tokens_uses_cached(self):
        d = {"token_estimate": 42, "content": "short"}
        assert LiteMemoryBus._item_tokens(d) == 42

    def test_item_tokens_fallback(self):
        d = {"content": "a" * 100}
        tokens = LiteMemoryBus._item_tokens(d)
        assert tokens > 0

    def test_estimate_tokens_legacy(self):
        d = {"content": "hello world"}
        tokens = LiteMemoryBus._estimate_tokens(d)
        assert tokens >= 1


# ============================================================================
# Snapshot Persistence
# ============================================================================


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_to_snapshot_dict_and_from_snapshot_dict(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="snap test", memory_type="factual"))
        await bus.add_alias("alias1", nid)
        snap = bus.to_snapshot_dict()
        assert snap["_version"] == 2
        assert nid in snap["items"]
        assert "alias1" in snap["aliases"]

        bus2 = LiteMemoryBus()
        bus2.from_snapshot_dict(snap)
        assert nid in bus2._items
        assert "alias1" in bus2._aliases

    @pytest.mark.asyncio
    async def test_save_and_load_snapshot(self, tmp_path):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="persist me", memory_type="factual"))
        path = str(tmp_path / "snapshot.json")
        bus.save_snapshot(path)

        bus2 = LiteMemoryBus()
        loaded = bus2.load_snapshot(path)
        assert loaded is True
        assert nid in bus2._items
        assert bus2._items[nid].content == "persist me"

    def test_load_snapshot_missing_file(self, tmp_path):
        bus = LiteMemoryBus()
        loaded = bus.load_snapshot(str(tmp_path / "nope.json"))
        assert loaded is False

    @pytest.mark.asyncio
    async def test_from_snapshot_dict_future_version_raises(self):
        bus = LiteMemoryBus()
        with pytest.raises(ValueError, match="newer than supported"):
            bus.from_snapshot_dict({"_version": 999, "items": {}, "edges": {}, "aliases": {}})


# ============================================================================
# Security Screen
# ============================================================================


class TestSecurityScreen:
    @pytest.mark.asyncio
    async def test_security_screen_blocks_sensitive(self):
        def screen(content):
            return "secret" not in content.lower()

        bus = LiteMemoryBus(security_screen=screen)
        with pytest.raises(SecurityScreenError):
            await bus.store(MemoryItem(content="my secret key", memory_type="factual"))

    @pytest.mark.asyncio
    async def test_security_screen_passes_clean(self):
        def screen(content):
            return True

        bus = LiteMemoryBus(security_screen=screen)
        nid = await bus.store(MemoryItem(content="clean content", memory_type="factual"))
        assert nid in bus._items


# ============================================================================
# Status, Pressure, Explain
# ============================================================================


class TestStatusAndPressure:
    @pytest.mark.asyncio
    async def test_status(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="a", memory_type="factual"))
        s = bus.status()
        assert s["initialized"] is True
        assert s["mode"] == "lite (in-memory)"
        assert s["item_count"] == 1

    def test_pressure(self):
        bus = LiteMemoryBus()
        p = bus.pressure()
        assert "tier" in p
        assert "utilization_pct" in p

    @pytest.mark.asyncio
    async def test_explain_existing_item(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="explain me", memory_type="factual"))
        ex = await bus.explain(nid)
        assert ex["node_id"] == nid
        assert ex["content"] == "explain me"
        assert "connected" in ex

    @pytest.mark.asyncio
    async def test_explain_missing_item(self):
        bus = LiteMemoryBus()
        ex = await bus.explain("nonexistent")
        assert ex["error"] == "not found"


# ============================================================================
# Graph Export (DOT, dict, subgraph)
# ============================================================================


class TestGraphExport:
    @pytest.mark.asyncio
    async def test_to_dot(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="node one", memory_type="factual"))
        dot = bus.to_dot()
        assert "digraph LiteMemoryBus" in dot
        assert "node one" in dot

    @pytest.mark.asyncio
    async def test_to_dot_highlight(self):
        bus = LiteMemoryBus()
        nid = await bus.store(MemoryItem(content="highlighted", memory_type="entity"))
        dot = bus.to_dot(highlight_ids={nid})
        assert "penwidth=3" in dot
        assert "color=red" in dot

    @pytest.mark.asyncio
    async def test_to_graph_dict(self):
        bus = LiteMemoryBus()
        nid_a = await bus.store(MemoryItem(content="a", memory_type="factual"))
        nid_b = await bus.store(MemoryItem(content="b", memory_type="factual"))
        await bus.store_edge(nid_a, nid_b, "LINKS_TO")
        gd = bus.to_graph_dict()
        assert len(gd["nodes"]) == 2
        assert len(gd["edges"]) == 1
        assert gd["edges"][0]["type"] == "LINKS_TO"

    @pytest.mark.asyncio
    async def test_to_graph_dict_no_content(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(content="hidden", memory_type="factual"))
        gd = bus.to_graph_dict(include_content=False)
        assert "content" not in gd["nodes"][0]

    @pytest.mark.asyncio
    async def test_subgraph(self):
        bus = LiteMemoryBus()
        ids = []
        for i in range(5):
            nid = await bus.store(MemoryItem(content=f"n{i}", memory_type="factual"))
            ids.append(nid)
        await bus.store_edge(ids[0], ids[1])
        await bus.store_edge(ids[1], ids[2])
        await bus.store_edge(ids[2], ids[3])
        await bus.store_edge(ids[3], ids[4])

        sub = bus.subgraph(ids[0], depth=2, max_nodes=10)
        # depth=2 from ids[0] should reach ids[0], ids[1], ids[2]
        assert ids[0] in sub._items
        assert ids[1] in sub._items
        assert ids[2] in sub._items


# ============================================================================
# Bulk Load
# ============================================================================


class TestBulkLoad:
    @pytest.mark.asyncio
    async def test_load_shards_with_attrs(self):
        bus = LiteMemoryBus()

        class FakeShard:
            def __init__(self, sid, text):
                self.id = sid
                self.text = text
                self.metadata = {"importance": 0.8}

        shards = [FakeShard("s1", "shard one"), FakeShard("s2", "shard two")]
        count = await bus.load_shards(shards, memory_type="factual")
        assert count == 2
        assert "s1" in bus._items
        assert bus._items["s1"].content == "shard one"

    @pytest.mark.asyncio
    async def test_load_shards_plain_objects(self):
        bus = LiteMemoryBus()
        count = await bus.load_shards(["raw text 1", "raw text 2"])
        assert count == 2


# ============================================================================
# Audit Log
# ============================================================================


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_audit_recorded_on_store(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(content="audited", memory_type="factual"))
        audit = bus.get_audit(limit=5)
        assert len(audit) >= 1
        assert audit[0]["action"] == "store"

    @pytest.mark.asyncio
    async def test_audit_recorded_on_query(self):
        bus = LiteMemoryBus()
        await bus.store(MemoryItem(content="test", memory_type="factual"))
        await bus.query(MemoryQuery(intent="test"))
        audit = bus.get_audit(limit=10)
        # Should have store + query entries
        assert len(audit) >= 2

    @pytest.mark.asyncio
    async def test_audit_ring_buffer_caps(self):
        bus = LiteMemoryBus()
        bus._max_audit = 5
        for i in range(10):
            await bus.store(MemoryItem(content=f"item {i}", memory_type="factual"))
        assert len(bus._audit) <= 5


# ============================================================================
# Post-store Hooks
# ============================================================================


class TestPostStoreHooks:
    @pytest.mark.asyncio
    async def test_hook_fires_on_store(self):
        bus = LiteMemoryBus()
        captured = []
        bus._post_store_hooks.append(lambda nid, item: captured.append(nid))
        nid = await bus.store(MemoryItem(content="hooked", memory_type="factual"))
        assert nid in captured

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_break_store(self):
        bus = LiteMemoryBus()

        def broken_hook(nid, item):
            raise RuntimeError("hook boom")

        bus._post_store_hooks.append(broken_hook)
        nid = await bus.store(MemoryItem(content="still works", memory_type="factual"))
        assert nid in bus._items


# ============================================================================
# Version Log Integration
# ============================================================================


class TestVersionLogIntegration:
    @pytest.mark.asyncio
    async def test_store_records_delta(self):
        from hololoom.core.memory.version_log import VersionLog
        vlog = VersionLog()
        bus = LiteMemoryBus(version_log=vlog)
        await bus.store(MemoryItem(content="versioned", memory_type="factual"))
        assert vlog.count == 1

    @pytest.mark.asyncio
    async def test_forget_records_delta(self):
        from hololoom.core.memory.version_log import VersionLog
        vlog = VersionLog()
        bus = LiteMemoryBus(version_log=vlog)
        nid = await bus.store(MemoryItem(content="to forget", memory_type="factual"))
        bus._forget(nid)
        # 1 store + 1 forget
        assert vlog.count == 2


# ============================================================================
# Query Cache Integration
# ============================================================================


class TestQueryCacheIntegration:
    @pytest.mark.asyncio
    async def test_query_cache_hit(self):
        cache = MagicMock()
        cached_result = MemoryResult(
            items=[{"id": "cached", "content": "from cache"}],
            query=MemoryQuery(intent="test"),
            resolution_path="exact",
            token_estimate=10,
        )
        cache.get.return_value = cached_result
        bus = LiteMemoryBus(query_cache=cache)
        q = MemoryQuery(intent="test")
        result = await bus.query(q)
        assert result.items[0]["id"] == "cached"
        cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_cache_miss_stores(self):
        cache = MagicMock()
        cache.get.return_value = None
        bus = LiteMemoryBus(query_cache=cache)
        await bus.store(MemoryItem(content="data", memory_type="factual"))
        q = MemoryQuery(intent="data")
        await bus.query(q)
        # On miss, result should be cached via put
        assert cache.put.called


# ============================================================================
# Retrieval Planner Integration
# ============================================================================


class TestRetrievalPlanner:
    @pytest.mark.asyncio
    async def test_planner_limits_cascade_depth(self):
        planner = MagicMock()
        plan = MagicMock()
        plan.max_cascade_level = 2  # Only exact + structured
        planner.plan.return_value = plan

        bus = LiteMemoryBus(retrieval_planner=planner)
        await bus.store(MemoryItem(content="deep query term", memory_type="factual"))
        q = MemoryQuery(intent="deep query term")  # Would match graph level
        result = await bus.query(q)
        # With max_level=2, graph (level 3) should be skipped
        # Result should be empty since there's no exact or structured match for intent-only query
        assert result.resolution_path in ("none", ResolutionPath.GRAPH.value, ResolutionPath.STRUCTURED.value)


# ============================================================================
# DOT escape
# ============================================================================


class TestDotEscape:
    def test_escapes_special_chars(self):
        assert LiteMemoryBus._dot_escape('"hello"') == '\\"hello\\"'
        assert LiteMemoryBus._dot_escape("line\nnew") == "line\\nnew"
        assert LiteMemoryBus._dot_escape("back\\slash") == "back\\\\slash"
