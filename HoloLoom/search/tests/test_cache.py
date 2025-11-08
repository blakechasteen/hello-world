"""
Tests for Search Cache
=======================
Comprehensive tests for SearchCache with TTL and LRU eviction.
"""

import pytest
import time
import asyncio
from HoloLoom.search.cache import SearchCache, CacheEntry
from HoloLoom.search.protocol import WebSearchResult


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating cache entries."""
        results = [
            WebSearchResult(
                url="https://example.com",
                title="Test",
                snippet="Snippet",
                domain="example.com",
                final_score=0.9
            )
        ]

        entry = CacheEntry(
            query="test query",
            results=results,
            timestamp=time.time(),
            ttl_seconds=3600
        )

        assert entry.query == "test query"
        assert len(entry.results) == 1
        assert entry.hit_count == 0

    def test_is_expired_fresh(self):
        """Test that fresh entries are not expired."""
        entry = CacheEntry(
            query="test",
            results=[],
            timestamp=time.time(),
            ttl_seconds=3600  # 1 hour
        )

        assert not entry.is_expired()

    def test_is_expired_old(self):
        """Test that old entries are expired."""
        entry = CacheEntry(
            query="test",
            results=[],
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600  # 1 hour TTL
        )

        assert entry.is_expired()

    def test_age_seconds(self):
        """Test age calculation."""
        timestamp = time.time() - 100
        entry = CacheEntry(
            query="test",
            results=[],
            timestamp=timestamp,
            ttl_seconds=3600
        )

        age = entry.age_seconds()
        assert 99 < age < 101  # Allow for small timing variations


class TestSearchCache:
    """Test SearchCache class."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return SearchCache(max_size=5, default_ttl=3600)

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            WebSearchResult(
                url=f"https://example.com/{i}",
                title=f"Result {i}",
                snippet=f"Snippet {i}",
                domain="example.com",
                final_score=0.9 - i * 0.1
            )
            for i in range(3)
        ]

    def test_put_and_get(self, cache, sample_results):
        """Test basic put and get operations."""
        cache.put("test query", sample_results)

        retrieved = cache.get("test query")
        assert retrieved is not None
        assert len(retrieved) == 3
        assert retrieved[0].url == "https://example.com/0"

    def test_get_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent query")
        assert result is None

    def test_query_normalization(self, cache, sample_results):
        """Test query normalization (case-insensitive, whitespace)."""
        cache.put("Test Query", sample_results)

        # Different case
        assert cache.get("test query") is not None

        # Extra whitespace
        assert cache.get("test  query") is not None

        # Different whitespace
        assert cache.get("test\tquery") is not None

    def test_hit_count_tracking(self, cache, sample_results):
        """Test hit count is tracked."""
        cache.put("test", sample_results)

        # Access multiple times
        cache.get("test")
        cache.get("test")
        cache.get("test")

        # Check entry stats
        entry_stats = cache.get_entry_stats()
        assert len(entry_stats) == 1
        assert entry_stats[0]["hit_count"] == 3

    def test_ttl_expiration(self, sample_results):
        """Test TTL expiration."""
        # Create cache with short TTL
        cache = SearchCache(max_size=5, default_ttl=1)  # 1 second TTL

        cache.put("test", sample_results)

        # Should be available immediately
        assert cache.get("test") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test") is None

    def test_lru_eviction(self, cache, sample_results):
        """Test LRU eviction when cache is full."""
        # Fill cache to max (5 items)
        for i in range(5):
            cache.put(f"query{i}", sample_results)

        # All should be present
        for i in range(5):
            assert cache.get(f"query{i}") is not None

        # Add 6th item (should evict oldest)
        cache.put("query5", sample_results)

        # First item should be evicted
        assert cache.get("query0") is None

        # Others should still be present
        for i in range(1, 6):
            assert cache.get(f"query{i}") is not None

    def test_lru_access_order(self, cache, sample_results):
        """Test that accessing items affects LRU order."""
        # Add 5 items
        for i in range(5):
            cache.put(f"query{i}", sample_results)

        # Access query0 (moves it to end)
        cache.get("query0")

        # Add new item (should evict query1, not query0)
        cache.put("query5", sample_results)

        assert cache.get("query0") is not None  # Should still be present
        assert cache.get("query1") is None  # Should be evicted

    def test_invalidate(self, cache, sample_results):
        """Test manual invalidation."""
        cache.put("test", sample_results)
        assert cache.get("test") is not None

        # Invalidate
        result = cache.invalidate("test")
        assert result is True

        # Should be gone
        assert cache.get("test") is None

        # Invalidating again returns False
        assert cache.invalidate("test") is False

    def test_clear(self, cache, sample_results):
        """Test clearing entire cache."""
        # Add multiple items
        for i in range(3):
            cache.put(f"query{i}", sample_results)

        # Clear
        cache.clear()

        # Stats should be reset immediately after clear
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # All should be gone (these will increment misses)
        for i in range(3):
            assert cache.get(f"query{i}") is None

    def test_cleanup_expired(self, sample_results):
        """Test cleanup of expired entries."""
        cache = SearchCache(max_size=10, default_ttl=1)

        # Add items
        cache.put("fresh", sample_results, ttl=3600)  # Long TTL
        cache.put("stale", sample_results, ttl=1)  # Short TTL

        # Wait for stale to expire
        time.sleep(1.1)

        # Cleanup
        removed = cache.cleanup_expired()

        assert removed == 1
        assert cache.get("fresh") is not None
        assert cache.get("stale") is None

    def test_get_stats(self, cache, sample_results):
        """Test statistics retrieval."""
        # Perform operations
        cache.put("query1", sample_results)
        cache.put("query2", sample_results)

        cache.get("query1")  # Hit
        cache.get("query1")  # Hit
        cache.get("query3")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0

    def test_get_entry_stats(self, cache, sample_results):
        """Test per-entry statistics."""
        cache.put("popular", sample_results)
        cache.put("unpopular", sample_results)

        # Access popular multiple times
        for _ in range(5):
            cache.get("popular")

        cache.get("unpopular")

        entry_stats = cache.get_entry_stats()

        # Should be sorted by hit count
        assert len(entry_stats) == 2
        assert entry_stats[0]["query"] == "popular"
        assert entry_stats[0]["hit_count"] == 5
        assert entry_stats[1]["query"] == "unpopular"
        assert entry_stats[1]["hit_count"] == 1


class TestCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_cache(self):
        """Test operations on empty cache."""
        cache = SearchCache()

        assert cache.get("anything") is None
        assert cache.invalidate("anything") is False

        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0

    def test_zero_size_cache(self):
        """Test cache with max_size=0."""
        cache = SearchCache(max_size=0)
        results = [WebSearchResult(
            url="https://example.com",
            title="Test",
            snippet="Test",
            domain="example.com",
            final_score=0.9
        )]

        # Should handle gracefully (always evict immediately)
        cache.put("test", results)
        # Can't test get() since it would be evicted immediately

    def test_very_long_query(self):
        """Test with very long query string."""
        cache = SearchCache()
        long_query = "a" * 10000

        results = [WebSearchResult(
            url="https://example.com",
            title="Test",
            snippet="Test",
            domain="example.com",
            final_score=0.9
        )]

        cache.put(long_query, results)
        assert cache.get(long_query) is not None

    def test_special_characters_in_query(self):
        """Test queries with special characters."""
        cache = SearchCache()
        special_queries = [
            "query with spaces",
            "query\twith\ttabs",
            "query\nwith\nnewlines",
            "query?with&special=chars",
            "query with emojis =%",
        ]

        results = [WebSearchResult(
            url="https://example.com",
            title="Test",
            snippet="Test",
            domain="example.com",
            final_score=0.9
        )]

        for query in special_queries:
            cache.put(query, results)
            assert cache.get(query) is not None

    def test_concurrent_access(self):
        """Test thread-safety (basic check)."""
        cache = SearchCache(max_size=100)
        results = [WebSearchResult(
            url="https://example.com",
            title="Test",
            snippet="Test",
            domain="example.com",
            final_score=0.9
        )]

        # Simulate concurrent puts
        import threading

        def put_many():
            for i in range(100):
                cache.put(f"query{i}", results)

        threads = [threading.Thread(target=put_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        stats = cache.get_stats()
        assert stats["size"] <= 100  # Respects max size


class TestCacheIntegration:
    """Integration tests with real usage patterns."""

    @pytest.mark.asyncio
    async def test_real_world_usage(self):
        """Test realistic usage pattern."""
        cache = SearchCache(max_size=100, default_ttl=3600)

        # Simulate search results
        def create_results(query):
            return [
                WebSearchResult(
                    url=f"https://example.com/{query}",
                    title=f"Result for {query}",
                    snippet=f"Content about {query}",
                    domain="example.com",
                    final_score=0.9
                )
            ]

        # Pattern: Popular queries hit cache often
        popular_queries = ["python", "javascript", "rust"]
        rare_queries = [f"rare_query_{i}" for i in range(50)]

        # Add popular queries
        for q in popular_queries:
            cache.put(q, create_results(q))

        # Simulate traffic
        for _ in range(100):
            # 80% popular, 20% rare
            if hash(_) % 5 != 0:  # 80%
                query = popular_queries[_ % len(popular_queries)]
                result = cache.get(query)
                assert result is not None  # Should hit cache
            else:  # 20%
                query = rare_queries[_ % len(rare_queries)]
                if cache.get(query) is None:
                    cache.put(query, create_results(query))

        stats = cache.get_stats()

        # Should have high hit rate due to popular queries
        assert stats["hit_rate"] > 0.5
        assert stats["size"] <= 100
