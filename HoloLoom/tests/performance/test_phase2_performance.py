"""
Phase 2 Performance Tests
==========================
Benchmark tests for Phase 2 components at scale.

Tests performance under realistic loads:
1. Storage throughput (1K-10K items)
2. Clustering scalability (1K-100K vectors)
3. MCP concurrent execution (100+ concurrent tools)
4. File processing batch performance
5. Hybrid search latency
"""

import asyncio
import pytest
import numpy as np
import time
from pathlib import Path
from typing import List, Dict

# Phase 2 imports
from HoloLoom.memory import UnifiedStore, Neo4jConfig, QdrantConfig
from HoloLoom.clustering import ClusterEngine, ClusterConfig, ClusterAlgorithm
from HoloLoom.mcp import MCPServer, MCPConfig
from HoloLoom.ingestion import FileProcessor, ProcessorConfig


class TestStoragePerformance:
    """Storage performance benchmarks."""

    @pytest.fixture
    async def unified_store(self):
        """Create unified store with mock backends."""
        neo4j_config = Neo4jConfig(uri="bolt://localhost:7687", database="perf_test")
        qdrant_config = QdrantConfig(host="localhost", collection_name="perf_test")

        store = UnifiedStore(neo4j_config, qdrant_config)

        try:
            await store.connect()
            yield store
            await store.close()
        except Exception as e:
            print(f"Using mock mode: {e}")
            yield store

    @pytest.mark.asyncio
    async def test_write_throughput_1k(self, unified_store):
        """Test: Write throughput for 1,000 items."""
        n_items = 1000
        start = time.time()

        for i in range(n_items):
            embedding = np.random.randn(384).astype(np.float32)
            await unified_store.add_knowledge(
                f"item_{i}",
                {"index": i, "category": f"cat_{i % 10}"},
                embedding
            )

        duration = time.time() - start
        throughput = n_items / duration

        print(f"\n📊 Write throughput (1K items): {throughput:.1f} items/sec")
        print(f"   Total time: {duration:.2f}s")

        assert throughput > 50, "Write throughput too slow"

    @pytest.mark.asyncio
    async def test_write_throughput_10k(self, unified_store):
        """Test: Write throughput for 10,000 items."""
        n_items = 10000
        start = time.time()

        # Batch writes for efficiency
        batch_size = 100
        for batch_start in range(0, n_items, batch_size):
            batch_tasks = []
            for i in range(batch_start, min(batch_start + batch_size, n_items)):
                embedding = np.random.randn(384).astype(np.float32)
                task = unified_store.add_knowledge(
                    f"item_{i}",
                    {"index": i, "batch": batch_start // batch_size},
                    embedding
                )
                batch_tasks.append(task)

            await asyncio.gather(*batch_tasks)

        duration = time.time() - start
        throughput = n_items / duration

        print(f"\n📊 Write throughput (10K items, batched): {throughput:.1f} items/sec")
        print(f"   Total time: {duration:.2f}s")

        assert throughput > 100, "Batched write throughput too slow"

    @pytest.mark.asyncio
    async def test_search_latency(self, unified_store):
        """Test: Search latency for hybrid retrieval."""
        # Populate with 1000 items
        n_items = 1000
        for i in range(n_items):
            embedding = np.random.randn(384).astype(np.float32)
            await unified_store.add_knowledge(
                f"search_item_{i}",
                {"index": i},
                embedding
            )

        # Benchmark search
        query_emb = np.random.randn(384).astype(np.float32)

        latencies = []
        n_searches = 50

        for _ in range(n_searches):
            start = time.time()
            results = await unified_store.hybrid_search(
                query_emb,
                top_k=10,
                strategy="vector_first"
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\n📊 Search latency (1K corpus, 50 queries):")
        print(f"   Average: {avg_latency:.1f}ms")
        print(f"   P95: {p95_latency:.1f}ms")

        assert avg_latency < 100, "Average search latency too high"
        assert p95_latency < 200, "P95 search latency too high"


class TestClusteringPerformance:
    """Clustering performance benchmarks."""

    @pytest.fixture
    def cluster_engine(self):
        """Create cluster engine."""
        config = ClusterConfig(
            algorithm=ClusterAlgorithm.KMEANS,
            n_clusters=10
        )
        return ClusterEngine(config)

    def test_clustering_1k_vectors(self, cluster_engine):
        """Test: Clustering 1,000 vectors."""
        n_vectors = 1000
        n_dims = 384

        embeddings = np.random.randn(n_vectors, n_dims).astype(np.float32)

        start = time.time()
        result = cluster_engine.cluster(embeddings)
        duration = time.time() - start

        print(f"\n📊 Clustering performance (1K vectors, 384D):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Clusters found: {result.n_clusters}")

        assert duration < 2.0, "Clustering 1K vectors too slow"
        assert result.n_clusters > 0, "No clusters found"

    def test_clustering_10k_vectors(self, cluster_engine):
        """Test: Clustering 10,000 vectors."""
        n_vectors = 10000
        n_dims = 384

        embeddings = np.random.randn(n_vectors, n_dims).astype(np.float32)

        start = time.time()
        result = cluster_engine.cluster(embeddings)
        duration = time.time() - start

        print(f"\n📊 Clustering performance (10K vectors, 384D):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Clusters found: {result.n_clusters}")

        assert duration < 10.0, "Clustering 10K vectors too slow"
        assert result.n_clusters > 0, "No clusters found"

    def test_clustering_high_dim(self, cluster_engine):
        """Test: Clustering high-dimensional vectors."""
        n_vectors = 1000
        n_dims = 1536  # GPT-3 embedding size

        embeddings = np.random.randn(n_vectors, n_dims).astype(np.float32)

        start = time.time()
        result = cluster_engine.cluster(embeddings)
        duration = time.time() - start

        print(f"\n📊 Clustering performance (1K vectors, 1536D):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Clusters found: {result.n_clusters}")

        assert duration < 5.0, "High-dim clustering too slow"


class TestMCPPerformance:
    """MCP server performance benchmarks."""

    @pytest.fixture
    def mcp_server(self):
        """Create MCP server with test tools."""
        server = MCPServer(MCPConfig(max_concurrent_executions=50))

        @server.tool("fast_echo", "Fast echo")
        async def fast_echo(text: str) -> str:
            return f"Echo: {text}"

        @server.tool("slow_compute", "Simulated slow computation")
        async def slow_compute(duration_ms: float = 100) -> str:
            await asyncio.sleep(duration_ms / 1000)
            return f"Computed for {duration_ms}ms"

        return server

    @pytest.mark.asyncio
    async def test_concurrent_execution_100(self, mcp_server):
        """Test: 100 concurrent tool executions."""
        n_executions = 100

        start = time.time()

        tasks = [
            mcp_server.execute("fast_echo", {"text": f"Message {i}"})
            for i in range(n_executions)
        ]

        results = await asyncio.gather(*tasks)

        duration = time.time() - start
        throughput = n_executions / duration

        print(f"\n📊 Concurrent execution (100 fast tools):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Throughput: {throughput:.1f} tools/sec")
        print(f"   Success rate: {sum(1 for r in results if r.success)}/{n_executions}")

        assert all(r.success for r in results), "Some executions failed"
        assert throughput > 500, "Concurrent execution throughput too low"

    @pytest.mark.asyncio
    async def test_concurrent_slow_tools(self, mcp_server):
        """Test: Concurrent slow tool executions."""
        n_executions = 50

        start = time.time()

        tasks = [
            mcp_server.execute("slow_compute", {"duration_ms": 50})
            for _ in range(n_executions)
        ]

        results = await asyncio.gather(*tasks)

        duration = time.time() - start

        print(f"\n📊 Concurrent execution (50 slow tools, 50ms each):")
        print(f"   Wall time: {duration:.2f}s")
        print(f"   Sequential would take: {50 * 0.05:.2f}s")
        print(f"   Speedup: {(50 * 0.05) / duration:.1f}x")

        assert all(r.success for r in results), "Some executions failed"
        assert duration < 1.0, "Not properly concurrent (should finish in ~50ms + overhead)"


class TestFileProcessingPerformance:
    """File processing performance benchmarks."""

    @pytest.fixture
    def file_processor(self):
        """Create file processor."""
        from HoloLoom.ingestion import ChunkConfig
        config = ProcessorConfig(
            chunk_config=ChunkConfig(chunk_size=500, overlap=50),
            parallel_processing=True
        )
        return FileProcessor(config)

    @pytest.mark.asyncio
    async def test_batch_processing_10_files(self, file_processor):
        """Test: Batch processing of 10 files."""
        # Create 10 test files
        test_files = []
        for i in range(10):
            file_path = f"/tmp/perf_test_{i}.txt"
            Path(file_path).write_text(f"""
            Document {i}

            Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
            It balances exploration and exploitation by sampling from posterior distributions.
            The algorithm was introduced by William R. Thompson in 1933.
            Despite its age, it remains effective for online decision-making.

            This is paragraph {i} with additional content to test chunking performance.
            """ * 10)  # 10 paragraphs each
            test_files.append(file_path)

        start = time.time()

        results = await file_processor.process_batch(test_files)

        duration = time.time() - start
        throughput = len(test_files) / duration

        total_chunks = sum(len(r.chunks) for r in results if r.success)

        print(f"\n📊 Batch file processing (10 files, parallel):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Throughput: {throughput:.1f} files/sec")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Success rate: {sum(1 for r in results if r.success)}/{len(results)}")

        # Cleanup
        for f in test_files:
            Path(f).unlink()

        assert all(r.success for r in results), "Some files failed to process"
        assert throughput > 5, "Batch processing throughput too low"

    @pytest.mark.asyncio
    async def test_large_file_processing(self, file_processor):
        """Test: Processing a large file (1MB+)."""
        # Create large test file
        large_file = "/tmp/large_test.txt"
        content = "Thompson Sampling for multi-armed bandits. " * 20000  # ~1MB
        Path(large_file).write_text(content)

        start = time.time()
        result = await file_processor.process_file(large_file)
        duration = time.time() - start

        file_size_mb = Path(large_file).stat().st_size / (1024 * 1024)

        print(f"\n📊 Large file processing ({file_size_mb:.1f}MB):")
        print(f"   Time: {duration:.2f}s")
        print(f"   Chunks: {len(result.chunks)}")
        print(f"   Throughput: {file_size_mb / duration:.1f} MB/sec")

        # Cleanup
        Path(large_file).unlink()

        assert result.success, "Large file processing failed"
        assert duration < 5.0, "Large file processing too slow"


class TestEndToEndPerformance:
    """End-to-end pipeline performance."""

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(self):
        """Test: Full pipeline latency (ingest → store → cluster → search)."""
        print("\n📊 Full pipeline latency test...")

        # Setup components
        from HoloLoom.ingestion import ChunkConfig

        neo4j_config = Neo4jConfig(uri="bolt://localhost:7687", database="e2e_perf")
        qdrant_config = QdrantConfig(host="localhost", collection_name="e2e_perf")
        store = UnifiedStore(neo4j_config, qdrant_config)

        try:
            await store.connect()
        except:
            print("   Using mock mode")

        cluster_config = ClusterConfig(algorithm=ClusterAlgorithm.KMEANS, n_clusters=3)
        cluster_engine = ClusterEngine(cluster_config)

        processor_config = ProcessorConfig(
            chunk_config=ChunkConfig(chunk_size=500, overlap=50)
        )
        processor = FileProcessor(processor_config)

        # Create test document
        test_file = "/tmp/e2e_perf_test.txt"
        Path(test_file).write_text("""
        Multi-armed bandits are a classical problem in decision theory.
        Thompson Sampling provides a Bayesian solution to this problem.
        """ * 100)  # Medium-sized document

        # Measure pipeline stages
        timings = {}

        # 1. Ingestion
        start = time.time()
        result = await processor.process_file(test_file)
        timings['ingestion'] = (time.time() - start) * 1000

        # 2. Embedding & Storage
        start = time.time()
        embeddings_list = []
        for chunk in result.chunks:
            embedding = np.random.randn(384).astype(np.float32)
            embeddings_list.append(embedding)
            await store.add_knowledge(
                f"chunk_{chunk.chunk_id}",
                {"text": chunk.text},
                embedding
            )
        timings['storage'] = (time.time() - start) * 1000

        # 3. Clustering
        start = time.time()
        if len(embeddings_list) >= 3:
            embeddings_array = np.array(embeddings_list)
            clusters = cluster_engine.cluster(embeddings_array)
        timings['clustering'] = (time.time() - start) * 1000

        # 4. Search
        start = time.time()
        query = np.random.randn(384).astype(np.float32)
        search_results = await store.hybrid_search(query, top_k=5)
        timings['search'] = (time.time() - start) * 1000

        total_latency = sum(timings.values())

        print(f"   Ingestion:  {timings['ingestion']:.1f}ms")
        print(f"   Storage:    {timings['storage']:.1f}ms")
        print(f"   Clustering: {timings['clustering']:.1f}ms")
        print(f"   Search:     {timings['search']:.1f}ms")
        print(f"   TOTAL:      {total_latency:.1f}ms")

        # Cleanup
        Path(test_file).unlink()
        await store.close()

        assert total_latency < 5000, "Full pipeline latency too high (>5s)"


# ============================================================================
# Performance Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def performance_summary():
    """Generate performance summary after all tests."""
    yield

    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    print("\nBenchmarks completed. See individual test outputs for details.")
    print("\nKey Metrics to Monitor:")
    print("  • Write throughput: >50 items/sec (1K), >100 items/sec (10K batched)")
    print("  • Search latency: <100ms avg, <200ms P95")
    print("  • Clustering: <2s (1K vectors), <10s (10K vectors)")
    print("  • Concurrent tools: >500 tools/sec")
    print("  • File processing: >5 files/sec (batch)")
    print("  • E2E pipeline: <5s total latency")
    print("="*80 + "\n")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Phase 2 Performance Tests")
    print("="*80)

    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s"  # Show print statements
    ])
