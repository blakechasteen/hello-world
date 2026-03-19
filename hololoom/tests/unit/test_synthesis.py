"""
Tests for Phase 3: Write-Time Synthesis.

Covers:
- SynthesisResult dataclass
- default_synthesis_fn heuristic behavior
- Integration with LiteMemoryBus.store() (merge, link, supersede, fallback)
- Version log records merge operations
"""

import pytest
from unittest.mock import AsyncMock

from hololoom.core.memory.bus import MemoryItem
from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.core.memory.synthesis import SynthesisResult, default_synthesis_fn
from hololoom.core.memory.version_log import VersionLog


# =============================================================================
# SynthesisResult
# =============================================================================

class TestSynthesisResult:

    def test_store_new_default(self):
        r = SynthesisResult(action="store_new")
        assert r.target_id is None
        assert r.merged_content is None
        assert r.edges == []

    def test_merge_into(self):
        r = SynthesisResult(
            action="merge_into", target_id="abc", merged_content="combined"
        )
        assert r.action == "merge_into"
        assert r.target_id == "abc"

    def test_link_to_with_edges(self):
        r = SynthesisResult(
            action="link_to", target_id="xyz",
            edges=[("__NEW__", "xyz", "SIMILAR_TO")],
        )
        assert len(r.edges) == 1
        assert r.edges[0][2] == "SIMILAR_TO"


# =============================================================================
# default_synthesis_fn
# =============================================================================

class TestDefaultSynthesisFn:

    @pytest.mark.asyncio
    async def test_exact_substring_merge(self):
        fn = default_synthesis_fn()
        candidate = StoredItem(
            id="existing", content="garlic is delicious and aromatic",
            memory_type="factual",
        )
        # New content is substring of existing
        result = await fn("garlic is delicious", [candidate])
        assert result.action == "merge_into"
        assert result.target_id == "existing"
        # Keeps the longer content
        assert result.merged_content == "garlic is delicious and aromatic"

    @pytest.mark.asyncio
    async def test_existing_is_substring(self):
        fn = default_synthesis_fn()
        candidate = StoredItem(
            id="short", content="garlic", memory_type="factual",
        )
        result = await fn("garlic is a wonderful ingredient", [candidate])
        assert result.action == "merge_into"
        assert result.target_id == "short"
        assert result.merged_content == "garlic is a wonderful ingredient"

    @pytest.mark.asyncio
    async def test_no_match_links(self):
        fn = default_synthesis_fn()
        candidate = StoredItem(
            id="other", content="completely different topic",
            memory_type="factual",
        )
        result = await fn("quantum mechanics rules", [candidate])
        assert result.action == "link_to"
        assert result.target_id == "other"
        assert len(result.edges) == 1

    @pytest.mark.asyncio
    async def test_no_candidates_stores_new(self):
        fn = default_synthesis_fn()
        result = await fn("brand new memory", [])
        assert result.action == "store_new"

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self):
        fn = default_synthesis_fn()
        candidate = StoredItem(
            id="cap", content="Garlic Is Delicious",
            memory_type="factual",
        )
        result = await fn("garlic is delicious", [candidate])
        assert result.action == "merge_into"


# =============================================================================
# Bus Integration: Synthesis wired into store()
# =============================================================================

class TestSynthesisBusIntegration:

    @pytest.mark.asyncio
    async def test_merge_deduplicates(self):
        """Storing duplicate content → merge, single item in bus."""
        # semantic_fn returns the existing item as a similar match
        existing_id = "exist123"

        async def fake_semantic(text, max_results):
            return [{"id": existing_id, "content": "garlic is great"}]

        fn = default_synthesis_fn()
        bus = LiteMemoryBus(semantic_fn=fake_semantic, synthesis_fn=fn)
        await bus.initialize()

        # Pre-seed an existing item
        bus._items[existing_id] = StoredItem(
            id=existing_id, content="garlic is great",
            memory_type="factual", timestamp="2026-03-05T00:00:00",
        )
        bus._by_type["factual"].append(existing_id)

        # Store substring content → should merge
        returned_id = await bus.store(MemoryItem(
            content="garlic is great and aromatic",
            memory_type="factual",
        ))
        assert returned_id == existing_id
        # Content updated to longer version
        assert bus._items[existing_id].content == "garlic is great and aromatic"
        # Only one item in bus
        assert len(bus._items) == 1

    @pytest.mark.asyncio
    async def test_link_creates_edge(self):
        """Storing related content → link_to, edge created."""
        existing_id = "exist456"

        async def fake_semantic(text, max_results):
            return [{"id": existing_id, "content": "unrelated but found"}]

        fn = default_synthesis_fn()
        bus = LiteMemoryBus(semantic_fn=fake_semantic, synthesis_fn=fn)
        await bus.initialize()

        bus._items[existing_id] = StoredItem(
            id=existing_id, content="unrelated but found",
            memory_type="factual", timestamp="2026-03-05T00:00:00",
        )
        bus._by_type["factual"].append(existing_id)

        new_id = await bus.store(MemoryItem(
            content="completely different topic",
            memory_type="factual",
        ))
        assert new_id != existing_id
        assert len(bus._items) == 2
        # Check SIMILAR_TO edge exists
        edges = bus._edges.get(new_id, [])
        similar_edges = [(t, r) for t, r in edges if r == "SIMILAR_TO"]
        assert len(similar_edges) == 1
        assert similar_edges[0][0] == existing_id

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_through(self):
        """If synthesis_fn raises, store normally."""
        async def fake_semantic(text, max_results):
            return [{"id": "x", "content": "match"}]

        async def broken_synth(content, candidates):
            raise RuntimeError("synthesis broke")

        bus = LiteMemoryBus(semantic_fn=fake_semantic, synthesis_fn=broken_synth)
        await bus.initialize()

        bus._items["x"] = StoredItem(
            id="x", content="match", memory_type="factual",
            timestamp="2026-03-05T00:00:00",
        )

        # Should not raise — falls through to normal store
        new_id = await bus.store(MemoryItem(
            content="new content", memory_type="factual",
        ))
        assert new_id != "x"
        assert len(bus._items) == 2

    @pytest.mark.asyncio
    async def test_no_synthesis_fn_stores_normally(self):
        """Without synthesis_fn, store works as before."""
        bus = LiteMemoryBus()
        await bus.initialize()
        id1 = await bus.store(MemoryItem(content="test", memory_type="factual"))
        id2 = await bus.store(MemoryItem(content="test", memory_type="factual"))
        assert id1 != id2
        assert len(bus._items) == 2

    @pytest.mark.asyncio
    async def test_merge_records_version_log(self):
        """Merge operation records delta in version log."""
        existing_id = "vlog_merge"

        async def fake_semantic(text, max_results):
            return [{"id": existing_id, "content": "original content"}]

        fn = default_synthesis_fn()
        vlog = VersionLog()
        bus = LiteMemoryBus(
            semantic_fn=fake_semantic, synthesis_fn=fn, version_log=vlog,
        )
        await bus.initialize()

        bus._items[existing_id] = StoredItem(
            id=existing_id, content="original content",
            memory_type="factual", timestamp="2026-03-05T00:00:00",
        )

        await bus.store(MemoryItem(
            content="original content with more detail",
            memory_type="factual",
        ))
        # Version log should have a merge delta
        deltas = vlog.replay()
        merge_deltas = [d for d in deltas if d.operation == "merge"]
        assert len(merge_deltas) == 1
        assert merge_deltas[0].item_id == existing_id

    @pytest.mark.asyncio
    async def test_supersedes_forgets_old(self):
        """Supersede action forgets old and stores new."""
        existing_id = "old_item"

        async def fake_semantic(text, max_results):
            return [{"id": existing_id, "content": "outdated info"}]

        async def supersede_fn(content, candidates):
            return SynthesisResult(action="supersedes", target_id=candidates[0].id)

        bus = LiteMemoryBus(semantic_fn=fake_semantic, synthesis_fn=supersede_fn)
        await bus.initialize()

        bus._items[existing_id] = StoredItem(
            id=existing_id, content="outdated info",
            memory_type="factual", timestamp="2026-03-05T00:00:00",
        )
        bus._by_type["factual"].append(existing_id)

        new_id = await bus.store(MemoryItem(
            content="updated info", memory_type="factual",
        ))
        assert existing_id not in bus._items
        assert new_id in bus._items
        assert bus._items[new_id].content == "updated info"
