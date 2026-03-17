"""
Tests for LiveSemanticIndex — Level 4 semantic search for LiteMemoryBus.

Validates:
- Index/search with sentence-transformers (if available)
- Keyword fallback when embeddings unavailable
- Bus integration via post-store hooks
- create_semantic_bus factory
- Batch indexing
- Item removal
- Diagnostics
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from hololoom.core.memory.live_semantic_index import LiveSemanticIndex, _HAVE_EMBEDDINGS
from hololoom.core.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.core.memory.bus import MemoryItem, MemoryQuery


# ============================================================================
# Helpers
# ============================================================================

def run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================================
# Keyword Fallback (always works, no GPU/model needed)
# ============================================================================

class TestKeywordFallback:
    """Test the Jaccard keyword fallback — no sentence-transformers needed."""

    def _make_index_without_embeddings(self):
        """Create an index that forces keyword fallback regardless of deps."""
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {}
        index._meta_map = {}
        return index

    def test_keyword_search_basic(self):
        index = self._make_index_without_embeddings()
        index.index_item("a", "the quick brown fox", "factual", 0.8)
        index.index_item("b", "lazy dog sleeps", "factual", 0.5)
        index.index_item("c", "quick fox jumps", "factual", 0.6)

        results = index.search_sync("quick fox", max_results=3)
        assert len(results) >= 2
        # "a" and "c" both contain "quick" and "fox"
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "c" in ids

    def test_keyword_search_empty_query(self):
        index = self._make_index_without_embeddings()
        index.index_item("a", "some content", "factual", 0.5)
        results = index.search_sync("", max_results=5)
        assert results == []

    def test_keyword_search_no_match(self):
        index = self._make_index_without_embeddings()
        index.index_item("a", "apples and oranges", "factual", 0.5)
        results = index.search_sync("quantum physics", max_results=5)
        assert results == []

    def test_keyword_result_format(self):
        index = self._make_index_without_embeddings()
        index.index_item("x", "test content here", "episodic", 0.9, {"source": "test"})
        results = index.search_sync("test content", max_results=1)
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "x"
        assert r["content"] == "test content here"
        assert r["memory_type"] == "episodic"
        assert r["importance"] == 0.9
        assert "_search_score" in r
        assert r["_resolution"] == "keyword_fallback"


# ============================================================================
# Index Management
# ============================================================================

class TestIndexManagement:
    def _make_index_without_embeddings(self):
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {}
        index._meta_map = {}
        return index

    def test_index_item_stores_content(self):
        index = self._make_index_without_embeddings()
        result = index.index_item("a", "hello world", "factual", 0.5)
        assert result is False  # No embeddings available
        assert index.count == 0  # No embedding-indexed items
        assert len(index._content_map) == 1  # But content is stored

    def test_remove_item(self):
        index = self._make_index_without_embeddings()
        index.index_item("a", "hello", "factual", 0.5)
        assert "a" in index._content_map
        removed = index.remove_item("a")
        assert removed is False  # Not in _ids (no embedding)
        assert "a" not in index._content_map

    def test_remove_nonexistent(self):
        index = self._make_index_without_embeddings()
        removed = index.remove_item("nope")
        assert removed is False

    def test_health_reports_status(self):
        index = self._make_index_without_embeddings()
        index.index_item("a", "test", "factual", 0.5)
        h = index.health()
        assert h["available"] is False
        assert h["fallback_only"] is True
        assert h["content_items"] == 1
        assert h["indexed_items"] == 0


# ============================================================================
# Bus Integration
# ============================================================================

class TestBusIntegration:
    def test_hook_store_indexes_item(self):
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {}
        index._meta_map = {}

        item = StoredItem(
            id="test1",
            content="beehive inspection results",
            memory_type="factual",
            importance=0.7,
            properties={"source": "apiary"},
        )
        index.hook_store("test1", item)

        assert "test1" in index._content_map
        assert index._content_map["test1"] == "beehive inspection results"

    def test_hook_delete_removes_item(self):
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {"x": "stuff"}
        index._meta_map = {"x": {"memory_type": "factual"}}

        index.hook_delete("x")
        assert "x" not in index._content_map

    def test_post_store_hook_fires_on_bus_store(self):
        """Verify LiteMemoryBus calls post-store hooks after store()."""
        hook_calls = []

        def spy_hook(item_id, stored_item):
            hook_calls.append((item_id, stored_item.content))

        bus = LiteMemoryBus()
        bus._post_store_hooks.append(spy_hook)

        async def _test():
            await bus.initialize()
            node_id = await bus.store(MemoryItem(
                content="varroa mite count: 3 per 100 bees",
                memory_type="factual",
                importance=0.8,
            ))
            return node_id

        node_id = run(_test())
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == node_id
        assert "varroa" in hook_calls[0][1]

    def test_as_semantic_fn_returns_callable(self):
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {}
        index._meta_map = {}

        fn = index.as_semantic_fn()
        assert callable(fn)

    def test_semantic_fn_works_in_bus_query(self):
        """End-to-end: store items, query with semantic_text, get results."""
        index = LiveSemanticIndex.__new__(LiveSemanticIndex)
        index.model_name = "none"
        index.embedding_dim = 384
        index.score_threshold = 0.0
        index.available = False
        index._model = None
        index._ids = []
        index._embeddings = None
        index._content_map = {}
        index._meta_map = {}

        bus = LiteMemoryBus(semantic_fn=index.as_semantic_fn())
        bus._post_store_hooks.append(index.hook_store)

        async def _test():
            await bus.initialize()
            await bus.store(MemoryItem(
                content="queen bee laying pattern is strong",
                memory_type="factual",
                importance=0.7,
            ))
            await bus.store(MemoryItem(
                content="honey super is 80 percent full",
                memory_type="factual",
                importance=0.6,
            ))
            await bus.store(MemoryItem(
                content="varroa mite treatment applied yesterday",
                memory_type="episodic",
                importance=0.8,
            ))

            # Query with semantic_text — should hit Level 4 keyword fallback
            result = await bus.query(MemoryQuery(
                intent="bee health",
                semantic_text="queen bee health",
                max_results=3,
            ))
            return result

        result = run(_test())
        assert result is not None
        # Should find items via keyword match on "bee" / "queen"
        assert len(result.items) > 0


# ============================================================================
# Embedding Tests (skipped if sentence-transformers not installed)
# ============================================================================

@pytest.mark.skipif(not _HAVE_EMBEDDINGS, reason="sentence-transformers not installed")
class TestWithEmbeddings:
    def test_index_creates_embedding(self):
        index = LiveSemanticIndex(embedding_dim=384)
        ok = index.index_item("a", "the quick brown fox jumps over the lazy dog")
        assert ok is True
        assert index.count == 1
        assert index._embeddings is not None
        assert index._embeddings.shape == (1, 384)

    def test_semantic_search_relevance(self):
        index = LiveSemanticIndex(embedding_dim=384)
        index.index_item("bee", "queen bee laying pattern healthy brood")
        index.index_item("mite", "varroa mite infestation treatment oxalic acid")
        index.index_item("honey", "honey harvest yield 40 pounds per hive")
        index.index_item("weather", "thunderstorm forecast heavy rain tonight")

        results = index.search_sync("bee colony health and disease", max_results=3)
        ids = [r["id"] for r in results]
        # "bee" and "mite" should rank higher than "weather"
        assert "weather" not in ids[:2]
        assert results[0]["_resolution"] == "semantic"
        assert results[0]["_model"] == "all-MiniLM-L6-v2"

    def test_batch_indexing(self):
        index = LiveSemanticIndex(embedding_dim=384)
        items = [
            ("a", "alpha content", "factual", 0.5, None),
            ("b", "beta content", "factual", 0.6, None),
            ("c", "gamma content", "factual", 0.7, None),
        ]
        count = index.index_batch(items)
        assert count == 3
        assert index.count == 3

    def test_remove_with_embedding(self):
        index = LiveSemanticIndex(embedding_dim=384)
        index.index_item("a", "alpha content")
        index.index_item("b", "beta content")
        assert index.count == 2

        removed = index.remove_item("a")
        assert removed is True
        assert index.count == 1
        assert index._embeddings.shape == (1, 384)

    def test_matryoshka_truncation(self):
        index = LiveSemanticIndex(embedding_dim=96)
        index.index_item("a", "test content for dimension check")
        assert index._embeddings.shape == (1, 96)

    def test_update_existing_item(self):
        index = LiveSemanticIndex(embedding_dim=384)
        index.index_item("a", "original content")
        index.index_item("a", "updated content completely different")
        assert index.count == 1  # Same ID, updated in place

    @pytest.mark.asyncio
    async def test_async_search(self):
        index = LiveSemanticIndex(embedding_dim=384)
        index.index_item("a", "beehive inspection spring 2026")
        results = await index.search("beehive inspection", max_results=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_end_to_end_bus_with_embeddings(self):
        """Full integration: create_semantic_bus → store → query."""
        from hololoom.core.memory.live_semantic_index import create_semantic_bus

        bus, index = create_semantic_bus(embedding_dim=384)

        async def _test():
            await bus.initialize()
            await bus.store(MemoryItem(
                content="queen bee laying pattern is strong and healthy",
                memory_type="factual",
                importance=0.7,
            ))
            await bus.store(MemoryItem(
                content="varroa mite count is dangerously high at 8 per 100",
                memory_type="factual",
                importance=0.9,
            ))
            await bus.store(MemoryItem(
                content="installed new solar panels on barn roof",
                memory_type="episodic",
                importance=0.4,
            ))

            assert index.count == 3

            result = await bus.query(MemoryQuery(
                intent="bee health",
                semantic_text="disease and pest problems in the hive",
                max_results=3,
            ))
            return result

        result = run(_test())
        assert result is not None
        # Semantic search should surface the varroa/bee items, not solar panels
        contents = [item.get("content", "") for item in result.items]
        has_bee_content = any("varroa" in c or "queen" in c for c in contents)
        assert has_bee_content, f"Expected bee content in results: {contents}"

    def test_health_with_embeddings(self):
        index = LiveSemanticIndex(embedding_dim=384)
        index.index_item("a", "test")
        h = index.health()
        assert h["available"] is True
        assert h["fallback_only"] is False
        assert h["indexed_items"] == 1
        assert h["model"] == "all-MiniLM-L6-v2"
        assert h["embedding_dim"] == 384
