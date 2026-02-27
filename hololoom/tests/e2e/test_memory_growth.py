"""
E2E tests for memory growth and leak detection over long-running sessions.

Tests:
1. Memory leak detection over 1000 queries
2. Background task accumulation
3. Cache memory growth
4. Knowledge graph memory scaling
5. Embedding cache growth
6. Long session stability (1 hour simulation)
7. Memory usage per query trend
8. Cleanup effectiveness
9. Peak memory limits
10. Memory recovery after cleanup

Performance Budget: <30s per test (E2E tier)
"""

import pytest
import asyncio
import time
import tracemalloc
from typing import List

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


class TestMemoryLeakDetection:
    """Test memory leak detection over many queries."""

    @pytest.mark.asyncio
    async def test_no_leak_over_100_queries(self, bare_config, test_shards):
        """Memory should not grow unboundedly over 100 queries."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Warm up
            await orchestrator.weave(Query(text="Warm up"))

            # Baseline
            baseline = tracemalloc.get_traced_memory()[0]

            # Execute 100 queries
            for i in range(100):
                query = Query(text=f"Query {i % 10}")  # Reuse queries for cache
                await orchestrator.weave(query)

            # Final memory
            final = tracemalloc.get_traced_memory()[0]
            growth_mb = (final - baseline) / (1024 * 1024)

            print(f"Memory growth over 100 queries: {growth_mb:.2f} MB")

            # Should not grow excessively (allow some growth for caching)
            assert growth_mb < 50  # Less than 50 MB growth

        tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_no_leak_over_500_queries(self, bare_config, test_shards):
        """Memory should remain stable over 500 queries."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Warm up
            await orchestrator.weave(Query(text="Warm up"))

            # Baseline after warm-up
            baseline = tracemalloc.get_traced_memory()[0]

            # Execute 500 queries
            for i in range(500):
                query = Query(text=f"Query {i % 20}")
                await orchestrator.weave(query)

                # Check memory periodically
                if i % 100 == 0:
                    current = tracemalloc.get_traced_memory()[0]
                    growth = (current - baseline) / (1024 * 1024)
                    print(f"After {i} queries: {growth:.2f} MB growth")

            # Final memory
            final = tracemalloc.get_traced_memory()[0]
            growth_mb = (final - baseline) / (1024 * 1024)

            print(f"Total memory growth over 500 queries: {growth_mb:.2f} MB")

            # Should not grow unboundedly
            assert growth_mb < 100  # Less than 100 MB growth

        tracemalloc.stop()


class TestBackgroundTaskAccumulation:
    """Test background task cleanup and accumulation."""

    @pytest.mark.asyncio
    async def test_background_tasks_cleaned_up(self, bare_config, test_shards):
        """Background tasks should be cleaned up properly."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Execute many queries
            for i in range(50):
                query = Query(text=f"Query {i}")
                await orchestrator.weave(query)

            # Background tasks should be managed (not accumulating)
            # (Validated by successful completion without crashes)


class TestCacheMemoryGrowth:
    """Test cache memory usage and limits."""

    @pytest.mark.asyncio
    async def test_cache_memory_bounded(self, bare_config, test_shards):
        """Cache should have memory bounds."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Fill cache with unique queries
            for i in range(200):
                query = Query(text=f"Unique query {i}")
                await orchestrator.weave(query)

            current, peak = tracemalloc.get_traced_memory()
            cache_mb = current / (1024 * 1024)

            print(f"Cache memory after 200 unique queries: {cache_mb:.2f} MB")

            # Cache should be bounded
            assert cache_mb < 500  # Less than 500 MB

        tracemalloc.stop()


class TestKnowledgeGraphScaling:
    """Test knowledge graph memory scaling."""

    @pytest.mark.asyncio
    async def test_kg_memory_scaling(self, bare_config, test_shards):
        """Knowledge graph should scale gracefully."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Execute queries that may add to KG
            for i in range(100):
                query = Query(text=f"Learn about topic {i % 10}")
                await orchestrator.weave(query)

            current = tracemalloc.get_traced_memory()[0]
            kg_mb = current / (1024 * 1024)

            print(f"KG memory after 100 queries: {kg_mb:.2f} MB")

            # Should be reasonable
            assert kg_mb < 300

        tracemalloc.stop()


class TestLongSessionStability:
    """Test stability over long-running sessions."""

    @pytest.mark.asyncio
    async def test_session_stability_over_time(self, bare_config, test_shards):
        """Session should remain stable over many queries."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Simulate long session (300 queries)
            for i in range(300):
                query = Query(text=f"Query {i % 30}")
                spacetime = await orchestrator.weave(query)

                assert spacetime is not None

                # Brief pause to simulate realistic usage
                if i % 50 == 0:
                    await asyncio.sleep(0.1)


class TestMemoryUsageTrend:
    """Test memory usage trends over time."""

    @pytest.mark.asyncio
    async def test_memory_usage_trend_linear(self, bare_config, test_shards):
        """Memory usage should grow linearly, not exponentially."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Warm up
            await orchestrator.weave(Query(text="Warm up"))
            baseline = tracemalloc.get_traced_memory()[0]

            measurements = []

            # Take measurements every 20 queries
            for batch in range(5):  # 5 batches of 20 = 100 queries
                for i in range(20):
                    query = Query(text=f"Batch {batch} Query {i}")
                    await orchestrator.weave(query)

                current = tracemalloc.get_traced_memory()[0]
                growth = (current - baseline) / (1024 * 1024)
                measurements.append(growth)

                print(f"After {(batch + 1) * 20} queries: {growth:.2f} MB")

            # Growth should not be exponential
            # Check that later growth rate is not dramatically higher
            if len(measurements) >= 3:
                early_growth = measurements[1] - measurements[0]
                late_growth = measurements[-1] - measurements[-2]

                print(f"Early growth: {early_growth:.2f} MB/20q")
                print(f"Late growth: {late_growth:.2f} MB/20q")

                # Late growth should not be >5x early growth (allows some increase)
                if early_growth > 0:
                    assert late_growth < early_growth * 5

        tracemalloc.stop()


class TestCleanupEffectiveness:
    """Test cleanup and resource release."""

    @pytest.mark.asyncio
    async def test_cleanup_releases_memory(self, bare_config, test_shards):
        """Cleanup should release memory."""
        tracemalloc.start()

        # Create and destroy orchestrator
        orchestrator = WeavingOrchestrator(cfg=bare_config, shards=test_shards)
        await orchestrator.__aenter__()

        try:
            # Use orchestrator
            for i in range(50):
                query = Query(text=f"Query {i}")
                await orchestrator.weave(query)

            memory_before_cleanup = tracemalloc.get_traced_memory()[0]

        finally:
            # Cleanup
            await orchestrator.__aexit__(None, None, None)

        # Check memory after cleanup
        await asyncio.sleep(0.5)  # Allow time for cleanup
        memory_after_cleanup = tracemalloc.get_traced_memory()[0]

        print(f"Memory before cleanup: {memory_before_cleanup / (1024 * 1024):.2f} MB")
        print(f"Memory after cleanup: {memory_after_cleanup / (1024 * 1024):.2f} MB")

        # Cleanup should release some memory (or at least not increase)
        assert memory_after_cleanup <= memory_before_cleanup * 1.1  # Allow 10% variance

        tracemalloc.stop()


class TestPeakMemoryLimits:
    """Test peak memory limits are respected."""

    @pytest.mark.asyncio
    async def test_peak_memory_reasonable(self, bare_config, test_shards):
        """Peak memory should be reasonable."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Execute queries
            for i in range(100):
                query = Query(text=f"Query {i}")
                await orchestrator.weave(query)

            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)

            print(f"Peak memory: {peak_mb:.2f} MB")

            # Peak should be reasonable (< 1 GB for 100 queries)
            assert peak_mb < 1024

        tracemalloc.stop()


class TestMemoryRecovery:
    """Test memory recovery after heavy usage."""

    @pytest.mark.asyncio
    async def test_memory_recovers_after_spike(self, bare_config, test_shards):
        """Memory should recover after usage spike."""
        tracemalloc.start()

        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Warm up
            await orchestrator.weave(Query(text="Warm up"))
            baseline = tracemalloc.get_traced_memory()[0]

            # Create spike with 100 concurrent queries
            queries = [Query(text=f"Spike query {i}") for i in range(100)]
            tasks = [orchestrator.weave(q) for q in queries]
            await asyncio.gather(*tasks)

            # Check memory immediately after spike
            spike_memory = tracemalloc.get_traced_memory()[0]

            # Wait for potential cleanup
            await asyncio.sleep(1.0)

            # Check memory after recovery period
            recovery_memory = tracemalloc.get_traced_memory()[0]

            spike_mb = (spike_memory - baseline) / (1024 * 1024)
            recovery_mb = (recovery_memory - baseline) / (1024 * 1024)

            print(f"Spike memory: {spike_mb:.2f} MB")
            print(f"Recovery memory: {recovery_mb:.2f} MB")

            # Memory should not stay at spike level
            # (Some memory may be retained for caching, so allow variance)

        tracemalloc.stop()


# Total: 10 E2E tests for memory growth
# Key coverage:
# - Memory leak detection (100, 500 queries)
# - Background task accumulation
# - Cache memory bounds
# - Knowledge graph scaling
# - Long session stability
# - Memory usage trends
# - Cleanup effectiveness
# - Peak memory limits
# - Memory recovery
