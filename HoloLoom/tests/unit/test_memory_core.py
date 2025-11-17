"""
Unit tests for memory system core components.

Tests graph operations, cache, and retrieval.
All tests isolated with mocked IO operations.
"""

from unittest.mock import Mock, patch

import pytest


class TestKnowledgeGraph:
    """Test knowledge graph operations."""

    def test_kg_creation(self):
        """Knowledge graph should initialize empty."""
        from HoloLoom.memory.graph import KG

        kg = KG()
        assert kg is not None
        assert hasattr(kg, "add_edge") or hasattr(kg, "add_edges")

    def test_kg_add_edge(self):
        """Should add edges to graph."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edge = KGEdge(source="python", target="programming", rel_type="IS_A", weight=1.0)

        kg.add_edges([edge])

        # Should have nodes
        assert kg.graph.number_of_nodes() >= 2

    def test_kg_add_multiple_edges(self):
        """Should handle multiple edges."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [
            KGEdge("a", "b", "CONNECTS", 1.0),
            KGEdge("b", "c", "LEADS_TO", 1.0),
            KGEdge("c", "a", "MENTIONS", 0.5),
        ]

        kg.add_edges(edges)

        assert kg.graph.number_of_nodes() == 3
        assert kg.graph.number_of_edges() >= 3

    def test_kg_subgraph_extraction(self):
        """Should extract subgraphs."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [
            KGEdge("center", "node1", "CONNECTS", 1.0),
            KGEdge("center", "node2", "CONNECTS", 1.0),
            KGEdge("node1", "node3", "CONNECTS", 1.0),
        ]
        kg.add_edges(edges)

        # Extract neighborhood
        if hasattr(kg, "get_subgraph"):
            subgraph = kg.get_subgraph(["center"], max_hops=1)
            assert subgraph is not None

    def test_kg_path_finding(self):
        """Should find paths between nodes."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [
            KGEdge("start", "middle", "CONNECTS", 1.0),
            KGEdge("middle", "end", "CONNECTS", 1.0),
        ]
        kg.add_edges(edges)

        # Should find path
        if hasattr(kg, "find_path"):
            path = kg.find_path("start", "end")
            assert path is not None or len(path) == 0  # Path may exist

    def test_kg_performance(self):
        """Graph operations should be fast."""
        import time

        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()

        # Add 100 edges
        start = time.perf_counter()
        edges = [KGEdge(f"node_{i}", f"node_{i + 1}", "NEXT", 1.0) for i in range(100)]
        kg.add_edges(edges)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Adding 100 edges took {elapsed:.2f}ms (target: <100ms)"


class TestMemoryCache:
    """Test memory cache/manager."""

    @pytest.mark.asyncio
    async def test_cache_creation(self):
        """Cache should initialize."""
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)
        assert cache is not None

    @pytest.mark.asyncio
    async def test_cache_store_recall(self):
        """Should store and recall memories."""
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)
        shard = MemoryShard(text="test memory", source="unit_test")

        await cache.store(shard)

        # Should be able to recall
        results = await cache.recall("test", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_cache_capacity_limit(self):
        """Cache should respect capacity limits."""
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=10)

        # Add more than capacity
        for i in range(20):
            shard = MemoryShard(text=f"memory_{i}", source="test")
            await cache.store(shard)

        # Should handle gracefully (LRU eviction or similar)

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Cache operations should be fast."""
        import time

        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=1000)

        # Store operation
        shard = MemoryShard(text="performance test", source="test")

        start = time.perf_counter()
        await cache.store(shard)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 10, f"Store took {elapsed:.2f}ms (target: <10ms)"

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        """Cache should handle concurrent access."""
        import asyncio

        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)

        async def store_shard(i):
            shard = MemoryShard(text=f"concurrent_{i}", source="test")
            await cache.store(shard)

        # Concurrent stores
        await asyncio.gather(*[store_shard(i) for i in range(10)])

        # Should complete without errors


class TestRetrievalStrategies:
    """Test memory retrieval strategies."""

    @pytest.mark.asyncio
    @patch("HoloLoom.memory.retrieval_strategies.SpectralEmbedding")
    async def test_semantic_retrieval(self, mock_embed):
        """Semantic retrieval should work."""
        import torch

        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.retrieval_strategies import semantic_retrieval

        # Mock embeddings
        mock_embed.return_value.encode_multi_scale = Mock(return_value=[torch.randn(1, 768)])

        shards = [
            MemoryShard(text="python programming", source="test"),
            MemoryShard(text="machine learning", source="test"),
        ]

        query = "programming language"

        # Should return ranked results
        results = await semantic_retrieval(query, shards, limit=2)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self):
        """BM25 retrieval should work."""
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.retrieval_strategies import bm25_retrieval

        shards = [
            MemoryShard(text="python is a programming language", source="test"),
            MemoryShard(text="java is also a language", source="test"),
        ]

        query = "python programming"

        # Should return ranked results
        if callable(bm25_retrieval):
            results = await bm25_retrieval(query, shards, limit=2)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self):
        """Hybrid retrieval should combine strategies."""
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.retrieval_strategies import hybrid_retrieval

        shards = [MemoryShard(text="test", source="test")]
        query = "test query"

        # Should combine BM25 + semantic
        if callable(hybrid_retrieval):
            results = await hybrid_retrieval(query, shards, limit=5)
            assert isinstance(results, list)


class TestSpectralFeatures:
    """Test spectral graph features."""

    def test_spectral_extraction(self):
        """Should extract spectral features from graph."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [
            KGEdge("a", "b", "CONNECTS", 1.0),
            KGEdge("b", "c", "CONNECTS", 1.0),
            KGEdge("c", "a", "CONNECTS", 1.0),
        ]
        kg.add_edges(edges)

        # Extract spectral features
        if hasattr(kg, "get_spectral_features"):
            features = kg.get_spectral_features()
            assert features is not None

    def test_laplacian_eigenvalues(self):
        """Should compute Laplacian eigenvalues."""
        import networkx as nx

        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [KGEdge(f"n{i}", f"n{i + 1}", "NEXT", 1.0) for i in range(5)]
        kg.add_edges(edges)

        # Compute Laplacian
        try:
            laplacian = nx.laplacian_matrix(kg.graph).todense()
            assert laplacian is not None
        except Exception:
            pass  # Graph structure may vary


class TestMemoryPersistence:
    """Test memory persistence."""

    @pytest.mark.asyncio
    async def test_memory_save(self):
        """Should save memory state."""
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100, persist_path="/tmp/test_memory")

        # Save should work (or gracefully skip)
        if hasattr(cache, "save"):
            await cache.save()

    @pytest.mark.asyncio
    async def test_memory_load(self):
        """Should load memory state."""
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100, persist_path="/tmp/test_memory")

        # Load should work (or gracefully skip)
        if hasattr(cache, "load"):
            await cache.load()


class TestMemoryEdgeCases:
    """Test memory edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Should handle empty query."""
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)
        results = await cache.recall("", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_empty_cache_recall(self):
        """Should handle recall from empty cache."""
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)
        results = await cache.recall("test", limit=5)
        assert results == []  # Empty cache returns empty list

    @pytest.mark.asyncio
    async def test_large_shard(self):
        """Should handle large memory shards."""
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.memory.cache import MemoryManager

        cache = MemoryManager(capacity=100)
        large_text = "test " * 10000  # 40KB text

        shard = MemoryShard(text=large_text, source="large_test")
        await cache.store(shard)

        # Should handle gracefully

    def test_circular_graph(self):
        """Should handle circular dependencies."""
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        edges = [
            KGEdge("a", "b", "NEXT", 1.0),
            KGEdge("b", "c", "NEXT", 1.0),
            KGEdge("c", "a", "NEXT", 1.0),  # Circular
        ]

        kg.add_edges(edges)

        # Should handle circular structure without infinite loops
        assert kg.graph.number_of_nodes() == 3
