"""Unit tests for MultiResolutionRenderer — FULL/COMPACT/STUB rendering."""

import pytest

from hololoom.core.memory.multi_resolution import (
    RenderResolution,
    ResolutionPolicy,
    MultiResolutionRenderer,
)


class FakeItem:
    """Minimal StoredItem stand-in for testing."""
    def __init__(self, id, content, importance=0.5, memory_type="entity", entity_type=None):
        self.id = id
        self.content = content
        self.importance = importance
        self.memory_type = memory_type
        self.entity_type = entity_type
        self.token_estimate = max(1, len(content) // 4)


class TestResolutionAssignment:
    def test_high_importance_gets_full(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "important thing", importance=0.9)]
        assignments = renderer.assign_resolution(items)
        assert assignments["a"] == RenderResolution.FULL

    def test_medium_importance_gets_compact(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "moderate thing", importance=0.5)]
        assignments = renderer.assign_resolution(items)
        assert assignments["a"] == RenderResolution.COMPACT

    def test_low_importance_gets_stub(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "minor thing", importance=0.1)]
        assignments = renderer.assign_resolution(items)
        assert assignments["a"] == RenderResolution.STUB

    def test_keyword_match_boosts_resolution(self):
        renderer = MultiResolutionRenderer()
        # Low importance but good keyword match → COMPACT (not STUB)
        items = [FakeItem("a", "garlic is aromatic", importance=0.1)]
        assignments = renderer.assign_resolution(items, query_keywords=["garlic", "aromatic"])
        # effective = 0.6*0.1 + 0.4*1.0 = 0.46 → COMPACT
        assert assignments["a"] == RenderResolution.COMPACT

    def test_no_keywords_uses_importance_only(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "test", importance=0.5)]
        a1 = renderer.assign_resolution(items)
        a2 = renderer.assign_resolution(items, query_keywords=[])
        assert a1 == a2

    def test_custom_policy(self):
        policy = ResolutionPolicy(full_threshold=0.9, compact_threshold=0.5)
        renderer = MultiResolutionRenderer(policy)
        items = [FakeItem("a", "test", importance=0.6)]
        assignments = renderer.assign_resolution(items)
        # No keywords → effective = importance = 0.6 → COMPACT (0.5 <= 0.6 < 0.9)
        assert assignments["a"] == RenderResolution.COMPACT

    def test_multiple_items(self):
        renderer = MultiResolutionRenderer()
        items = [
            FakeItem("a", "high", importance=0.95),
            FakeItem("b", "mid", importance=0.5),
            FakeItem("c", "low", importance=0.05),
        ]
        assignments = renderer.assign_resolution(items)
        assert assignments["a"] == RenderResolution.FULL
        assert assignments["b"] == RenderResolution.COMPACT
        assert assignments["c"] == RenderResolution.STUB


class TestRendering:
    def test_compact_renders_first_sentence(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("a", "Garlic is aromatic. It adds flavor to many dishes.", memory_type="entity", entity_type="ingredient")
        result = renderer.render_compact(item)
        assert "Garlic is aromatic." in result
        assert "[entity/ingredient]" in result
        assert "imp=" in result
        assert "It adds flavor" not in result

    def test_compact_truncates_long_content(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("a", "x" * 300, memory_type="entity")
        result = renderer.render_compact(item)
        assert len(result) < 200
        assert "..." in result

    def test_stub_renders_path_only(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("abcdefgh", "some content", importance=0.3, memory_type="episodic")
        result = renderer.render_stub(item)
        assert "episodic/general/abcdefgh" in result
        assert "imp=0.3" in result
        assert "some content" not in result

    def test_token_cost_full(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("a", "x" * 200)
        assert renderer.token_cost(RenderResolution.FULL, item) == item.token_estimate

    def test_token_cost_compact(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("a", "x" * 200)
        assert renderer.token_cost(RenderResolution.COMPACT, item) == 30

    def test_token_cost_stub(self):
        renderer = MultiResolutionRenderer()
        item = FakeItem("a", "x" * 200)
        assert renderer.token_cost(RenderResolution.STUB, item) == 8


class TestTokenSavings:
    def test_savings_from_compact(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "x" * 200)]  # token_estimate = 50
        assignments = {"a": RenderResolution.COMPACT}
        saved = renderer.estimate_tokens_saved(items, assignments)
        assert saved == 50 - 30  # full - compact

    def test_savings_from_stub(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "x" * 200)]
        assignments = {"a": RenderResolution.STUB}
        saved = renderer.estimate_tokens_saved(items, assignments)
        assert saved == 50 - 8

    def test_no_savings_from_full(self):
        renderer = MultiResolutionRenderer()
        items = [FakeItem("a", "test")]
        assignments = {"a": RenderResolution.FULL}
        assert renderer.estimate_tokens_saved(items, assignments) == 0


@pytest.mark.asyncio
async def test_bus_integration_multi_resolution():
    """Multi-resolution packs overflow items at COMPACT/STUB."""
    from hololoom.core.memory.bus import MemoryItem, MemoryQuery
    from hololoom.core.memory.bus_config import MemoryBusConfig
    from hololoom.core.memory.lite_bus import LiteMemoryBus

    # Budget=150, items ~40 tokens each → 3 fit at FULL (120), overflow at COMPACT/STUB
    config = MemoryBusConfig(token_budget_absolute=150)
    renderer = MultiResolutionRenderer()
    bus = LiteMemoryBus(config=config, multi_resolution=renderer)
    await bus.initialize()

    # Each item: ~160 chars → token_estimate ~40
    for i in range(10):
        await bus.store(MemoryItem(
            content=f"Item {i}: " + "x" * 152,
            memory_type="episodic",
            importance=0.5,
        ))

    result = await bus.query(MemoryQuery(intent="item", max_results=10))
    # 3 full (120 tokens) + at least 1 compact (30 tokens) = 150
    full_items = [i for i in result.items if i.get("_render_resolution") is None]
    compact_items = [i for i in result.items if i.get("_render_resolution") == "compact"]
    stub_items = [i for i in result.items if i.get("_render_resolution") == "stub"]
    assert len(result.items) > len(full_items), (
        f"Expected overflow items, got {len(full_items)} full, "
        f"{len(compact_items)} compact, {len(stub_items)} stub"
    )
    assert len(compact_items) + len(stub_items) > 0
