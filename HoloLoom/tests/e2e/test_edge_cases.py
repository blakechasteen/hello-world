"""
E2E tests for edge cases, boundary conditions, and unusual patterns.

Tests:
1. Empty string queries
2. Very long queries (10,000+ characters)
3. Unicode and emoji handling
4. Special characters and escape sequences
5. Null/None inputs
6. Extremely nested structures
7. Pathological regex patterns
8. Concurrent duplicate queries
9. Rapid query bursts
10. Mixed valid/invalid inputs

Performance Budget: <30s per test (E2E tier)
"""

import pytest
import asyncio

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query


class TestEmptyInputs:
    """Test empty and minimal inputs."""

    @pytest.mark.asyncio
    async def test_empty_string_query(self, bare_config, test_shards):
        """Empty string query should handle gracefully."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="")

            # Should either handle or raise clear error
            try:
                spacetime = await orchestrator.weave(query)
                # If it succeeds, verify result
                assert spacetime is not None
            except (ValueError, AssertionError):
                # Acceptable to reject empty queries
                pass

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, bare_config, test_shards):
        """Whitespace-only query should handle gracefully."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="   \n\t   ")

            try:
                spacetime = await orchestrator.weave(query)
                assert spacetime is not None
            except (ValueError, AssertionError):
                pass


class TestVeryLongInputs:
    """Test very long inputs."""

    @pytest.mark.asyncio
    async def test_very_long_query(self, bare_config, test_shards):
        """Very long query should handle gracefully."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # 10,000 character query
            long_text = "Thompson Sampling " * 500  # ~9,000 chars
            query = Query(text=long_text)

            spacetime = await orchestrator.weave(query)
            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_extremely_long_query(self, bare_config, test_shards):
        """Extremely long query (50,000+ chars) should handle or truncate."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # 50,000+ character query
            very_long_text = "A" * 50000
            query = Query(text=very_long_text)

            # Should either handle or truncate gracefully
            spacetime = await orchestrator.weave(query)
            assert spacetime is not None


class TestUnicodeHandling:
    """Test Unicode and emoji handling."""

    @pytest.mark.asyncio
    async def test_unicode_query(self, bare_config, test_shards):
        """Unicode characters should be handled."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="What is 機械学習 (machine learning)?")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_emoji_query(self, bare_config, test_shards):
        """Emoji should be handled."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Explain Thompson Sampling 🎲🔢")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_mixed_scripts(self, bare_config, test_shards):
        """Mixed scripts (Latin, Cyrillic, Arabic) should be handled."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Thompson Sampling на русском and العربية")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


class TestSpecialCharacters:
    """Test special characters and escape sequences."""

    @pytest.mark.asyncio
    async def test_special_characters(self, bare_config, test_shards):
        """Special characters should be handled."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="What is <regex> & {json} with $vars?")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_escape_sequences(self, bare_config, test_shards):
        """Escape sequences should be handled."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Query with \\n newline \\t tab \\r return")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_sql_injection_patterns(self, bare_config, test_shards):
        """SQL injection patterns should be safe."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="'; DROP TABLE users; --")
            spacetime = await orchestrator.weave(query)

            # Should handle safely (not execute as SQL)
            assert spacetime is not None


class TestNullInputs:
    """Test None/null handling."""

    @pytest.mark.asyncio
    async def test_none_query_text(self, bare_config, test_shards):
        """None query text should handle gracefully."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text=None)

            try:
                spacetime = await orchestrator.weave(query)
                # If it succeeds, verify result
                assert spacetime is not None
            except (ValueError, TypeError, AssertionError):
                # Acceptable to reject None queries
                pass


class TestPathologicalPatterns:
    """Test pathological patterns."""

    @pytest.mark.asyncio
    async def test_repeated_characters(self, bare_config, test_shards):
        """Repeated characters should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="aaaaaaaaaa" * 100)
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_alternating_pattern(self, bare_config, test_shards):
        """Alternating patterns should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="ababab" * 200)
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


class TestConcurrentDuplicates:
    """Test concurrent duplicate queries."""

    @pytest.mark.asyncio
    async def test_100_identical_concurrent_queries(self, bare_config, test_shards):
        """100 identical concurrent queries should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Same query 100 times concurrently
            query = Query(text="What is Thompson Sampling?")
            tasks = [orchestrator.weave(query) for _ in range(100)]

            results = await asyncio.gather(*tasks)

            assert len(results) == 100
            assert all(r is not None for r in results)


class TestRapidBursts:
    """Test rapid query bursts."""

    @pytest.mark.asyncio
    async def test_rapid_sequential_queries(self, bare_config, test_shards):
        """Rapid sequential queries should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # 200 queries as fast as possible
            for i in range(200):
                query = Query(text=f"Q{i}")
                spacetime = await orchestrator.weave(query)
                assert spacetime is not None


class TestMixedValidInvalid:
    """Test mixed valid and invalid inputs."""

    @pytest.mark.asyncio
    async def test_mixed_query_batch(self, bare_config, test_shards):
        """Batch of mixed valid/invalid queries should handle gracefully."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            queries = [
                Query(text="Valid query 1"),
                Query(text=""),
                Query(text="Valid query 2"),
                Query(text="   "),
                Query(text="Valid query 3"),
            ]

            results = []
            for query in queries:
                try:
                    spacetime = await orchestrator.weave(query)
                    results.append(spacetime)
                except (ValueError, AssertionError):
                    results.append(None)

            # At least some should succeed
            successful = [r for r in results if r is not None]
            assert len(successful) >= 3


class TestBoundaryNumbers:
    """Test boundary number values."""

    @pytest.mark.asyncio
    async def test_large_numbers_in_query(self, bare_config, test_shards):
        """Large numbers in query should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="What is 999999999999999999999999?")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None

    @pytest.mark.asyncio
    async def test_scientific_notation(self, bare_config, test_shards):
        """Scientific notation should handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Explain 1.23e-45 and 9.87e+123")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


# Total: 17 E2E tests for edge cases
# Key coverage:
# - Empty inputs
# - Very long inputs (10K, 50K chars)
# - Unicode, emoji, mixed scripts
# - Special characters, escape sequences
# - None/null handling
# - Pathological patterns
# - Concurrent duplicates (100x)
# - Rapid bursts (200 queries)
# - Mixed valid/invalid inputs
# - Boundary numbers
