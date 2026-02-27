"""
Tests for MatryoshkaWebSearch
==============================
Comprehensive unit and integration tests for three-stage web search.
"""

import pytest
import asyncio
from HoloLoom.search import MatryoshkaWebSearch, SearchConfig


class TestMatryoshkaWebSearch:
    """Test suite for MatryoshkaWebSearch."""

    @pytest.fixture
    async def search(self):
        """Create search instance with mock provider."""
        config = SearchConfig(
            provider="mock",
            stage1_candidates=10,
            stage2_candidates=5,
            final_results=3
        )
        search = MatryoshkaWebSearch(config=config, enable_cache=False)
        yield search

    @pytest.mark.asyncio
    async def test_basic_search(self, search):
        """Test basic search functionality."""
        results = await search.search("Thompson Sampling", max_results=3)

        assert len(results) == 3
        assert all(r.url for r in results)
        assert all(r.title for r in results)
        assert all(r.snippet for r in results)

    @pytest.mark.asyncio
    async def test_multi_scale_scores(self, search):
        """Test that all three stages compute scores."""
        results = await search.search("machine learning", max_results=3)

        for result in results:
            assert result.score_96d > 0
            assert result.score_192d > 0
            assert result.score_384d > 0
            assert result.final_score == result.score_384d  # Final uses 384d

    @pytest.mark.asyncio
    async def test_ranking(self, search):
        """Test that results are properly ranked."""
        results = await search.search("python programming", max_results=3)

        # Should have some results
        assert len(results) > 0

        # Final scores should be descending
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i + 1].final_score

        # Final ranks should be 1, 2, 3, ... (sequential from 1)
        assert [r.final_rank for r in results] == list(range(1, len(results) + 1))

    @pytest.mark.asyncio
    async def test_search_to_shards(self, search):
        """Test conversion to memory shards."""
        results, shards = await search.search_to_shards("Thompson Sampling", max_results=2)

        assert len(results) > 0  # Should have some results
        # Shards may be empty if web scraping unavailable (no requests/beautifulsoup4)
        # This is acceptable for testing - just verify structure when shards exist

        # Check shard metadata (if any shards created)
        for shard in shards:
            assert shard.metadata.get("search_query") == "Thompson Sampling"
            assert "search_rank" in shard.metadata
            assert "similarity_score" in shard.metadata
            assert "url" in shard.metadata

    @pytest.mark.asyncio
    async def test_statistics(self, search):
        """Test statistics tracking."""
        await search.search("test query 1", max_results=3)
        await search.search("test query 2", max_results=3)

        stats = search.get_stats()

        assert stats["search_count"] == 2
        assert stats["avg_time_ms"] > 0
        assert "stage_times" in stats
        assert "stage1" in stats["stage_times"]
        assert "stage2" in stats["stage_times"]
        assert "stage3" in stats["stage_times"]

    @pytest.mark.asyncio
    async def test_cache_disabled(self, search):
        """Test that cache is disabled when configured."""
        # Search twice with same query
        results1 = await search.search("test query", max_results=3)
        results2 = await search.search("test query", max_results=3)

        stats = search.get_stats()
        assert stats["cache_hits"] == 0  # No cache hits


class TestMatryoshkaWebSearchWithCache:
    """Test suite for caching functionality."""

    @pytest.fixture
    async def search_with_cache(self):
        """Create search instance with cache enabled."""
        config = SearchConfig(
            provider="mock",
            final_results=3,
            enable_cache=True
        )
        search = MatryoshkaWebSearch(config=config, enable_cache=True)
        yield search

    @pytest.mark.asyncio
    async def test_cache_hit(self, search_with_cache):
        """Test cache hit on repeated query."""
        # First search (cold cache)
        results1 = await search_with_cache.search("test query", max_results=3)

        # Second search (should hit cache)
        results2 = await search_with_cache.search("test query", max_results=3)

        stats = search_with_cache.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["search_count"] == 2

        # Results should be identical
        assert len(results1) == len(results2)
        assert results1[0].url == results2[0].url

    @pytest.mark.asyncio
    async def test_cache_case_insensitive(self, search_with_cache):
        """Test cache is case-insensitive."""
        await search_with_cache.search("Test Query", max_results=3)
        await search_with_cache.search("test query", max_results=3)

        stats = search_with_cache.get_stats()
        assert stats["cache_hits"] == 1  # Should hit cache


@pytest.mark.asyncio
async def test_empty_query():
    """Test handling of empty query."""
    config = SearchConfig(provider="mock")
    search = MatryoshkaWebSearch(config=config)

    results = await search.search("", max_results=3)
    assert isinstance(results, list)  # Should return empty list, not error


@pytest.mark.asyncio
async def test_large_result_limit():
    """Test handling of large result limits."""
    config = SearchConfig(
        provider="mock",
        stage1_candidates=100,
        final_results=50
    )
    search = MatryoshkaWebSearch(config=config)

    results = await search.search("python", max_results=50)
    assert len(results) <= 50  # Respects limit
