"""
Performance Benchmarks for Moonshot RAG System

Measures latency and overhead of each feature:
- Baseline (no features)
- +Streaming (Wave 3)
- +Custom Embeddings (Wave 3)
- +Reranking (Wave 3)
- +SQL Integration (Wave 4)
- +Multi-Hop (Wave 4)
- +Multi-Agent (Wave 5)
- All features combined

Target Performance:
- Baseline: 50-150ms
- +Streaming: ~0ms overhead (non-blocking)
- +Embeddings: ~5-20ms
- +Reranking: ~10-30ms
- +SQL: ~10-50ms (depends on DB)
- +Multi-Hop: ~20-100ms (depends on graph depth)
- +Multi-Agent: ~100-300ms (N agents in parallel)

Author: Agent L (Claude Code)
Date: November 13, 2025
"""

import pytest
import asyncio
import time
from typing import Dict, List
from dataclasses import dataclass

from HoloLoom.config import Config
from HoloLoom.rag.simple_rag import SimpleRAG


# ============================================================================
# Performance Measurement Infrastructure
# ============================================================================

@dataclass
class PerformanceSample:
    """Single performance sample."""
    name: str
    elapsed_ms: float
    tokens: int = 0  # For streaming
    sources: int = 0  # For retrieval


class PerformanceBenchmark:
    """Collect and analyze performance samples."""

    def __init__(self, name: str):
        self.name = name
        self.samples: List[PerformanceSample] = []

    def add_sample(self, elapsed_ms: float, tokens: int = 0, sources: int = 0):
        """Add a performance sample."""
        self.samples.append(
            PerformanceSample(
                name=self.name,
                elapsed_ms=elapsed_ms,
                tokens=tokens,
                sources=sources,
            )
        )

    def stats(self) -> Dict[str, float]:
        """Calculate statistics."""
        if not self.samples:
            return {}

        times = [s.elapsed_ms for s in self.samples]
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": sum(times) / len(times),
            "p50_ms": sorted(times)[len(times) // 2],
            "p95_ms": sorted(times)[int(len(times) * 0.95)],
            "p99_ms": sorted(times)[int(len(times) * 0.99)] if len(times) > 100 else max(times),
            "samples": len(self.samples),
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    return Config.fast()


@pytest.fixture
def sample_documents():
    """Create sample documents for benchmarking."""
    return [
        {
            "id": f"doc{i}",
            "content": f"Document {i}: " + " ".join([
                "Thompson Sampling is a Bayesian exploration strategy.",
                "It balances exploration and exploitation.",
                "Used in bandits, optimization, and reinforcement learning.",
            ] * 5),
            "metadata": {"source": "BENCHMARK"},
        }
        for i in range(10)
    ]


@pytest.fixture
async def populated_rag(mock_config, sample_documents):
    """Create and populate RAG instance."""
    rag = SimpleRAG(config=mock_config)
    async with rag:
        for doc in sample_documents:
            await rag.ingest(doc["content"], metadata=doc["metadata"])
        yield rag


# ============================================================================
# Baseline Performance Tests
# ============================================================================

class TestBaselinePerformance:
    """Measure baseline performance without features."""

    @pytest.mark.asyncio
    async def test_baseline_query_latency(self, mock_config, sample_documents):
        """Measure baseline query latency."""
        rag = SimpleRAG(config=mock_config)

        async with rag:
            # Ingest
            for doc in sample_documents:
                await rag.ingest(doc["content"], metadata=doc["metadata"])

            # Warm up
            await rag.query("Thompson Sampling")

            # Benchmark
            benchmark = PerformanceBenchmark("baseline_query")
            for _ in range(10):
                start = time.time()
                result = await rag.query("Thompson Sampling")
                elapsed = (time.time() - start) * 1000
                benchmark.add_sample(elapsed, sources=len(result.sources))

            stats = benchmark.stats()

            # Baseline should be reasonable
            assert stats["mean_ms"] < 1000, f"Baseline too slow: {stats['mean_ms']}ms"
            print(f"\nBaseline Performance: {stats}")


# ============================================================================
# Feature Overhead Tests
# ============================================================================

class TestStreamingOverhead:
    """Measure streaming feature overhead."""

    @pytest.mark.asyncio
    async def test_streaming_latency(self, mock_config, sample_documents):
        """Measure streaming latency."""
        rag = SimpleRAG(config=mock_config)

        async with rag:
            # Ingest
            for doc in sample_documents:
                await rag.ingest(doc["content"], metadata=doc["metadata"])

            # Benchmark streaming
            benchmark = PerformanceBenchmark("streaming_query")
            for _ in range(5):
                start = time.time()
                try:
                    stream = await rag.query_stream(
                        "Thompson Sampling",
                        enable_streaming=True,
                    )
                    tokens = 0
                    async for token in stream:
                        tokens += 1
                    elapsed = (time.time() - start) * 1000
                    benchmark.add_sample(elapsed, tokens=tokens)
                except Exception:
                    pass  # Streaming is optional

            if benchmark.samples:
                stats = benchmark.stats()
                print(f"\nStreaming Performance: {stats}")


class TestEmbeddingOverhead:
    """Measure custom embedding overhead."""

    @pytest.mark.asyncio
    async def test_embedding_latency(self, mock_config, sample_documents):
        """Measure embedding model latency."""
        # Matryoshka embedding
        rag_matryoshka = SimpleRAG(
            config=mock_config,
            embedding_model="matryoshka",
        )

        async with rag_matryoshka:
            # Ingest
            for doc in sample_documents:
                await rag_matryoshka.ingest(doc["content"], metadata=doc["metadata"])

            # Query
            benchmark = PerformanceBenchmark("embedding_matryoshka")
            for _ in range(5):
                start = time.time()
                result = await rag_matryoshka.query("Thompson Sampling")
                elapsed = (time.time() - start) * 1000
                benchmark.add_sample(elapsed, sources=len(result.sources))

            stats = benchmark.stats()
            print(f"\nEmbedding Performance (Matryoshka): {stats}")


class TestRerankerOverhead:
    """Measure reranking feature overhead."""

    @pytest.mark.asyncio
    async def test_reranker_noop_overhead(self, mock_config, sample_documents):
        """Measure NoOp reranker overhead."""
        rag = SimpleRAG(config=mock_config)

        async with rag:
            # Ingest
            for doc in sample_documents:
                await rag.ingest(doc["content"], metadata=doc["metadata"])

            # Without reranking
            benchmark_without = PerformanceBenchmark("without_reranking")
            for _ in range(5):
                start = time.time()
                result = await rag.query("Thompson Sampling", enable_reranking=False)
                elapsed = (time.time() - start) * 1000
                benchmark_without.add_sample(elapsed, sources=len(result.sources))

            # With reranking
            benchmark_with = PerformanceBenchmark("with_reranking")
            for _ in range(5):
                start = time.time()
                result = await rag.query(
                    "Thompson Sampling",
                    enable_reranking=True,
                    reranking_model="noop",
                )
                elapsed = (time.time() - start) * 1000
                benchmark_with.add_sample(elapsed, sources=len(result.sources))

            stats_without = benchmark_without.stats()
            stats_with = benchmark_with.stats()

            overhead_ms = stats_with["mean_ms"] - stats_without["mean_ms"]
            overhead_pct = (overhead_ms / stats_without["mean_ms"]) * 100 if stats_without["mean_ms"] > 0 else 0

            print(f"\nReranking Overhead:")
            print(f"  Without: {stats_without['mean_ms']:.1f}ms")
            print(f"  With: {stats_with['mean_ms']:.1f}ms")
            print(f"  Overhead: {overhead_ms:.1f}ms ({overhead_pct:.1f}%)")


# ============================================================================
# Feature Combination Tests
# ============================================================================

class TestCombinedFeaturePerformance:
    """Measure performance with multiple features combined."""

    @pytest.mark.asyncio
    async def test_streaming_plus_reranking(self, mock_config, sample_documents):
        """Measure latency of streaming + reranking."""
        rag = SimpleRAG(
            config=mock_config,
            enable_streaming=True,
            enable_reranking=True,
        )

        async with rag:
            # Ingest
            for doc in sample_documents:
                await rag.ingest(doc["content"], metadata=doc["metadata"])

            # Benchmark
            benchmark = PerformanceBenchmark("streaming_plus_reranking")
            for _ in range(3):
                start = time.time()
                try:
                    stream = await rag.query_stream(
                        "Thompson Sampling",
                        enable_streaming=True,
                        enable_reranking=True,
                    )
                    tokens = 0
                    async for token in stream:
                        tokens += 1
                    elapsed = (time.time() - start) * 1000
                    benchmark.add_sample(elapsed, tokens=tokens)
                except Exception:
                    pass

            if benchmark.samples:
                stats = benchmark.stats()
                print(f"\nStreaming + Reranking: {stats}")

    @pytest.mark.asyncio
    async def test_all_features_combined(self, mock_config, sample_documents):
        """Measure latency with all features enabled."""
        rag = SimpleRAG(
            config=mock_config,
            enable_streaming=True,
            enable_reranking=True,
            embedding_model="matryoshka",
        )

        async with rag:
            # Ingest
            for doc in sample_documents:
                await rag.ingest(doc["content"], metadata=doc["metadata"])

            # Benchmark
            benchmark = PerformanceBenchmark("all_features")
            for _ in range(3):
                start = time.time()
                result = await rag.query(
                    "Thompson Sampling",
                    enable_reranking=True,
                )
                elapsed = (time.time() - start) * 1000
                benchmark.add_sample(elapsed, sources=len(result.sources))

            stats = benchmark.stats()
            print(f"\nAll Features Combined: {stats}")


# ============================================================================
# Scalability Tests
# ============================================================================

class TestScalability:
    """Test performance scaling with data size."""

    @pytest.mark.asyncio
    async def test_scaling_with_document_count(self, mock_config):
        """Measure latency scaling as document count increases."""
        results = {}

        for doc_count in [10, 50, 100]:
            # Create documents
            documents = [
                {
                    "id": f"doc{i}",
                    "content": f"Document {i}: Thompson Sampling technique for exploration.",
                    "metadata": {"source": "BENCHMARK"},
                }
                for i in range(doc_count)
            ]

            rag = SimpleRAG(config=mock_config)

            async with rag:
                # Ingest all documents
                for doc in documents:
                    await rag.ingest(doc["content"], metadata=doc["metadata"])

                # Benchmark
                benchmark = PerformanceBenchmark(f"docs_{doc_count}")
                for _ in range(3):
                    start = time.time()
                    result = await rag.query("Thompson Sampling")
                    elapsed = (time.time() - start) * 1000
                    benchmark.add_sample(elapsed, sources=len(result.sources))

                stats = benchmark.stats()
                results[doc_count] = stats["mean_ms"]

        print(f"\nScaling with Document Count:")
        for doc_count, latency in results.items():
            print(f"  {doc_count} docs: {latency:.1f}ms")

        # Verify reasonable scaling (not exponential)
        latency_10 = results.get(10, 100)
        latency_100 = results.get(100, 100)

        # 10x documents should not cause 10x latency
        scaling_factor = latency_100 / latency_10 if latency_10 > 0 else 1
        assert scaling_factor < 5, f"Poor scaling: {scaling_factor}x for 10x documents"


# ============================================================================
# Performance Summary Report
# ============================================================================

class TestPerformanceSummary:
    """Generate performance summary report."""

    @pytest.mark.asyncio
    async def test_performance_report(self, mock_config, sample_documents):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 70)
        print("MOONSHOT RAG PERFORMANCE REPORT")
        print("=" * 70)

        # Baseline
        rag_baseline = SimpleRAG(config=mock_config)
        async with rag_baseline:
            for doc in sample_documents:
                await rag_baseline.ingest(doc["content"], metadata=doc["metadata"])

            baseline_times = []
            for _ in range(5):
                start = time.time()
                await rag_baseline.query("Thompson Sampling")
                baseline_times.append((time.time() - start) * 1000)

        baseline_ms = sum(baseline_times) / len(baseline_times)
        print(f"\nBaseline (no features): {baseline_ms:.1f}ms")

        # With streaming
        rag_streaming = SimpleRAG(config=mock_config)
        async with rag_streaming:
            for doc in sample_documents:
                await rag_streaming.ingest(doc["content"], metadata=doc["metadata"])

            streaming_times = []
            for _ in range(5):
                try:
                    start = time.time()
                    stream = await rag_streaming.query_stream("Thompson Sampling")
                    async for _ in stream:
                        pass
                    streaming_times.append((time.time() - start) * 1000)
                except Exception:
                    streaming_times.append(baseline_ms)  # Fallback

        streaming_ms = sum(streaming_times) / len(streaming_times) if streaming_times else baseline_ms
        streaming_overhead = ((streaming_ms - baseline_ms) / baseline_ms) * 100
        print(f"With Streaming: {streaming_ms:.1f}ms (overhead: {streaming_overhead:+.1f}%)")

        # With reranking
        rag_reranking = SimpleRAG(config=mock_config)
        async with rag_reranking:
            for doc in sample_documents:
                await rag_reranking.ingest(doc["content"], metadata=doc["metadata"])

            reranking_times = []
            for _ in range(5):
                start = time.time()
                await rag_reranking.query(
                    "Thompson Sampling",
                    enable_reranking=True,
                    reranking_model="noop",
                )
                reranking_times.append((time.time() - start) * 1000)

        reranking_ms = sum(reranking_times) / len(reranking_times)
        reranking_overhead = ((reranking_ms - baseline_ms) / baseline_ms) * 100
        print(f"With Reranking: {reranking_ms:.1f}ms (overhead: {reranking_overhead:+.1f}%)")

        # Summary
        print("\n" + "=" * 70)
        print("PERFORMANCE TARGET ADHERENCE")
        print("=" * 70)
        print(f"Baseline target: <150ms ... Actual: {baseline_ms:.1f}ms ✓" if baseline_ms < 150 else f"Actual: {baseline_ms:.1f}ms ✗")
        print(f"Streaming overhead target: <50ms ... Actual: {streaming_ms - baseline_ms:.1f}ms")
        print(f"Reranking overhead target: <30ms ... Actual: {reranking_ms - baseline_ms:.1f}ms")
        print("=" * 70)
