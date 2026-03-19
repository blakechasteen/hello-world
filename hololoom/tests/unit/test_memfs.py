"""
Tests for Phase 4: MemFS Virtual Filesystem View.

Covers:
- tree() output structure
- render() individual item with YAML frontmatter
- render_pinned() token budget enforcement
- render_for_prompt() progressive disclosure
- Empty bus edge cases
"""

import pytest

from hololoom.core.memory.bus import MemoryItem
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.memfs import MemFS, MemoryFile


# =============================================================================
# MemoryFile
# =============================================================================

class TestMemoryFile:

    def test_render_has_frontmatter(self):
        mf = MemoryFile(
            path="entity/ingredient/2026-03_abc.md",
            frontmatter={"importance": 0.8, "memory_type": "entity"},
            body="Garlic is a key ingredient",
            token_estimate=10,
        )
        rendered = mf.render()
        assert rendered.startswith("---\n")
        assert "importance: 0.8" in rendered
        assert "memory_type: entity" in rendered
        assert rendered.endswith("Garlic is a key ingredient")

    def test_render_tokens_includes_overhead(self):
        mf = MemoryFile(
            path="test.md",
            frontmatter={"a": 1, "b": 2, "c": 3},
            body="hello",
            token_estimate=5,
        )
        rt = mf.render_tokens()
        # 3 fields * 3 overhead + 4 for --- lines + 5 base
        assert rt == 5 + 3 * 3 + 4


# =============================================================================
# MemFS.tree()
# =============================================================================

class TestMemFSTree:

    @pytest.mark.asyncio
    async def test_empty_bus_tree(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        fs = MemFS(bus)
        tree = fs.tree()
        assert tree == "memory/"

    @pytest.mark.asyncio
    async def test_groups_by_memory_type(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="garlic", memory_type="entity",
                                   properties={"type": "ingredient"}))
        await bus.store(MemoryItem(content="dinner was good", memory_type="episodic"))
        await bus.store(MemoryItem(content="2+2=4", memory_type="factual"))

        fs = MemFS(bus)
        tree = fs.tree()
        assert "entity/" in tree
        assert "episodic/" in tree
        assert "factual/" in tree
        assert "ingredient/" in tree

    @pytest.mark.asyncio
    async def test_shows_item_counts(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        for i in range(5):
            await bus.store(MemoryItem(content=f"fact {i}", memory_type="factual"))

        fs = MemFS(bus)
        tree = fs.tree()
        assert "5 items" in tree

    @pytest.mark.asyncio
    async def test_shows_token_totals(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="x" * 100, memory_type="factual"))
        fs = MemFS(bus)
        tree = fs.tree()
        assert "tokens" in tree


# =============================================================================
# MemFS.render()
# =============================================================================

class TestMemFSRender:

    @pytest.mark.asyncio
    async def test_render_item(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        item_id = await bus.store(MemoryItem(
            content="Garlic is aromatic",
            memory_type="entity",
            properties={"type": "ingredient"},
        ))
        fs = MemFS(bus)
        mf = fs.render(item_id)
        assert mf is not None
        assert "entity" in mf.path
        assert "ingredient" in mf.path
        assert mf.body == "Garlic is aromatic"
        assert mf.frontmatter["memory_type"] == "entity"

    @pytest.mark.asyncio
    async def test_render_nonexistent(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        fs = MemFS(bus)
        assert fs.render("nonexistent") is None


# =============================================================================
# MemFS.render_pinned()
# =============================================================================

class TestMemFSRenderPinned:

    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        # Store 50 high-importance entity items with long content
        for i in range(50):
            await bus.store(MemoryItem(
                content=f"Important fact number {i}: " + "x" * 200,
                memory_type="entity",
                importance=0.9,
            ))

        fs = MemFS(bus)
        pinned = fs.render_pinned(max_tokens=500)
        total_tokens = sum(mf.render_tokens() for mf in pinned)
        assert total_tokens <= 500
        assert len(pinned) < 50  # Can't fit all

    @pytest.mark.asyncio
    async def test_filters_low_importance(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="low", memory_type="entity", importance=0.3))
        await bus.store(MemoryItem(content="high", memory_type="entity", importance=0.9))

        fs = MemFS(bus)
        pinned = fs.render_pinned(max_tokens=5000)
        assert len(pinned) == 1
        assert pinned[0].body == "high"

    @pytest.mark.asyncio
    async def test_empty_bus(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        fs = MemFS(bus)
        assert fs.render_pinned() == []


# =============================================================================
# MemFS.render_for_prompt()
# =============================================================================

class TestRenderForPrompt:

    @pytest.mark.asyncio
    async def test_stays_under_budget(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        for i in range(20):
            await bus.store(MemoryItem(
                content=f"Memory item {i} with some content: " + "x" * 100,
                memory_type="factual",
                importance=0.8 if i < 5 else 0.3,
            ))

        fs = MemFS(bus)
        output = fs.render_for_prompt("memory item", token_budget=2000)
        # Rough check: output tokens should be under budget
        assert len(output) // 4 <= 2000

    @pytest.mark.asyncio
    async def test_includes_tree(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="test", memory_type="factual"))

        fs = MemFS(bus)
        output = fs.render_for_prompt("test", token_budget=5000)
        assert "memory/" in output

    @pytest.mark.asyncio
    async def test_empty_bus_returns_tree(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        fs = MemFS(bus)
        output = fs.render_for_prompt("anything", token_budget=1000)
        assert output == "memory/"

    @pytest.mark.asyncio
    async def test_pinned_before_relevant(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(
            content="high importance entity", memory_type="entity", importance=0.95,
        ))
        await bus.store(MemoryItem(
            content="keyword match content", memory_type="episodic", importance=0.3,
        ))

        fs = MemFS(bus)
        output = fs.render_for_prompt("keyword match", token_budget=5000)
        # Pinned section should appear before relevant section
        pinned_pos = output.find("Pinned")
        relevant_pos = output.find("Relevant")
        if pinned_pos >= 0 and relevant_pos >= 0:
            assert pinned_pos < relevant_pos


# =============================================================================
# MemFS.all_files()
# =============================================================================

class TestAllFiles:

    @pytest.mark.asyncio
    async def test_returns_all(self):
        bus = LiteMemoryBus()
        await bus.initialize()
        await bus.store(MemoryItem(content="a", memory_type="factual"))
        await bus.store(MemoryItem(content="b", memory_type="episodic"))

        fs = MemFS(bus)
        files = fs.all_files()
        assert len(files) == 2
        assert all(isinstance(f, MemoryFile) for f in files)
