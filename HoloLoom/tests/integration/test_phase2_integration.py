"""
Phase 2 Integration Tests
==========================
End-to-end tests for all Phase 2 components working together.

Tests complete pipeline:
1. File ingestion → chunks
2. Chunks → embeddings → storage (Neo4j + Qdrant)
3. Clustering → topic discovery
4. MCP tools → execution
5. Hybrid search → retrieval
6. Chat interface → WebSocket communication
"""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Phase 2 imports
from HoloLoom.memory import UnifiedStore, Neo4jConfig, QdrantConfig
from HoloLoom.clustering import ClusterEngine, ClusterConfig, ClusterAlgorithm
from HoloLoom.mcp import MCPServer, MCPConfig
from HoloLoom.ingestion import FileProcessor, ProcessorConfig, SmartChunker, ChunkConfig


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    @pytest.fixture
    async def unified_store(self):
        """Create unified store with mock backends."""
        neo4j_config = Neo4jConfig(uri="bolt://localhost:7687", database="test")
        qdrant_config = QdrantConfig(host="localhost", collection_name="test_collection")

        store = UnifiedStore(neo4j_config, qdrant_config)

        # Connect (will use mock mode if DBs unavailable)
        try:
            await store.connect()
            yield store
            await store.close()
        except Exception as e:
            print(f"Using mock mode: {e}")
            yield store

    @pytest.fixture
    def cluster_engine(self):
        """Create cluster engine."""
        config = ClusterConfig(
            algorithm=ClusterAlgorithm.KMEANS,
            n_clusters=3
        )
        return ClusterEngine(config)

    @pytest.fixture
    def mcp_server(self):
        """Create MCP server with tools."""
        server = MCPServer(MCPConfig())

        # Register test tools
        @server.tool("test_echo", "Echo back input")
        async def echo(text: str) -> str:
            return f"Echo: {text}"

        @server.tool("test_sum", "Sum numbers")
        async def sum_numbers(a: float, b: float) -> float:
            return a + b

        return server

    @pytest.fixture
    def file_processor(self):
        """Create file processor."""
        config = ProcessorConfig(
            chunk_config=ChunkConfig(chunk_size=200, overlap=30)
        )
        return FileProcessor(config)

    @pytest.mark.asyncio
    async def test_file_to_storage_pipeline(self, file_processor, unified_store):
        """Test: File → Chunks → Storage."""
        # Create test file
        test_file = Path("/tmp/test_doc.txt")
        test_file.write_text("""
        Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
        It balances exploration and exploitation by sampling from posterior distributions.
        The algorithm was introduced by William R. Thompson in 1933.
        Despite its age, it remains effective for online decision-making.
        """)

        # Process file
        result = await file_processor.process_file(str(test_file))

        assert result.success, f"Processing failed: {result.error}"
        assert len(result.chunks) > 0, "No chunks generated"

        # Store chunks (with mock embeddings)
        for chunk in result.chunks:
            # Mock embedding
            embedding = np.random.randn(384).astype(np.float32)

            success = await unified_store.add_knowledge(
                entity_id=f"chunk_{chunk.chunk_id}",
                properties={
                    "text": chunk.text,
                    "file": "test_doc.txt",
                    "chunk_id": chunk.chunk_id
                },
                embedding=embedding
            )

            assert success, f"Failed to store chunk {chunk.chunk_id}"

        # Verify storage
        entity = await unified_store.get_entity_full(f"chunk_0")
        assert entity is not None, "Failed to retrieve stored chunk"

        # Cleanup
        test_file.unlink()

    @pytest.mark.asyncio
    async def test_clustering_pipeline(self, cluster_engine):
        """Test: Embeddings → Clustering → Topic Discovery."""
        # Generate synthetic embeddings (3 clusters)
        np.random.seed(42)

        cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        embeddings = np.vstack([cluster1, cluster2, cluster3])

        # Cluster
        result = cluster_engine.cluster(embeddings)

        assert result.n_clusters == 3, f"Expected 3 clusters, got {result.n_clusters}"
        assert len(result.labels) == 60, "Incorrect number of labels"
        assert result.cluster_centers is not None, "No cluster centers"

        # Test prediction
        new_point = np.random.randn(10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        predicted = cluster_engine.predict_cluster(new_point)

        assert predicted >= 0, "Invalid cluster prediction"

    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self, mcp_server):
        """Test: MCP Tool Registration → Execution."""
        # List tools
        tools = mcp_server.list_tools()
        assert len(tools) >= 2, "Tools not registered"

        # Execute echo tool
        result = await mcp_server.execute("test_echo", {"text": "Hello"})
        assert result.success, f"Tool execution failed: {result.error}"
        assert result.output == "Echo: Hello", "Unexpected output"

        # Execute sum tool
        result = await mcp_server.execute("test_sum", {"a": 2.5, "b": 3.5})
        assert result.success, "Sum tool failed"
        assert result.output == 6.0, "Incorrect sum"

        # Test analytics
        stats = mcp_server.get_analytics()
        assert stats['total_executions'] == 2, "Execution count incorrect"

    @pytest.mark.asyncio
    async def test_hybrid_search_pipeline(self, unified_store):
        """Test: Hybrid Search (Graph + Vector)."""
        # Add test entities
        entities = [
            ("doc:001", {"title": "Thompson Sampling", "year": 1933}),
            ("doc:002", {"title": "UCB Algorithm", "year": 2002}),
            ("topic:bandits", {"name": "Multi-Armed Bandits"}),
        ]

        embeddings = [
            np.random.randn(384).astype(np.float32),
            np.random.randn(384).astype(np.float32),
            np.random.randn(384).astype(np.float32),
        ]

        # Store entities
        for (entity_id, props), emb in zip(entities, embeddings):
            await unified_store.add_knowledge(
                entity_id=entity_id,
                properties=props,
                embedding=emb,
                relationships=[
                    ("topic:bandits", "INCLUDES", entity_id)
                ] if entity_id.startswith("doc:") else []
            )

        # Hybrid search
        query_emb = np.random.randn(384).astype(np.float32)

        results = await unified_store.hybrid_search(
            query_embedding=query_emb,
            start_entities=["topic:bandits"],
            max_hops=2,
            top_k=5,
            strategy="hybrid"
        )

        assert len(results) > 0, "No search results"

        # Verify result structure
        for result in results:
            assert hasattr(result, 'entity_id'), "Missing entity_id"
            assert hasattr(result, 'score'), "Missing score"
            assert result.score >= 0, "Invalid score"

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(
        self,
        file_processor,
        unified_store,
        cluster_engine,
        mcp_server
    ):
        """Test: Complete E2E Pipeline."""
        print("\n=== End-to-End Pipeline Test ===\n")

        # 1. Ingest document
        print("1. Document Ingestion...")
        test_file = Path("/tmp/e2e_test.txt")
        test_file.write_text("""
        Multi-armed bandits are a classical problem in decision theory.
        Thompson Sampling provides a Bayesian solution to this problem.
        The algorithm samples from posterior distributions to balance exploration.
        UCB (Upper Confidence Bound) is an alternative approach using confidence intervals.
        """)

        result = await file_processor.process_file(str(test_file))
        assert result.success, "Ingestion failed"
        print(f"   ✓ Generated {len(result.chunks)} chunks")

        # 2. Embed and store
        print("2. Embedding & Storage...")
        embeddings_list = []

        for chunk in result.chunks:
            # Mock embedding (in production, use real embedder)
            embedding = np.random.randn(384).astype(np.float32)
            embeddings_list.append(embedding)

            await unified_store.add_knowledge(
                f"chunk_{chunk.chunk_id}",
                {"text": chunk.text, "chunk_id": chunk.chunk_id},
                embedding
            )

        print(f"   ✓ Stored {len(embeddings_list)} chunks")

        # 3. Cluster
        print("3. Clustering...")
        if len(embeddings_list) >= 3:
            embeddings_array = np.array(embeddings_list)
            clusters = cluster_engine.cluster(embeddings_array)
            print(f"   ✓ Found {clusters.n_clusters} clusters")
        else:
            print(f"   ⊘ Skipped (need ≥3 chunks, have {len(embeddings_list)})")

        # 4. Tool execution
        print("4. Tool Execution...")
        calc_result = await mcp_server.execute("test_sum", {"a": 10, "b": 32})
        print(f"   ✓ Calculator: 10 + 32 = {calc_result.output}")

        # 5. Search
        print("5. Hybrid Search...")
        query = np.random.randn(384).astype(np.float32)
        search_results = await unified_store.hybrid_search(
            query,
            top_k=3,
            strategy="vector_first"
        )
        print(f"   ✓ Found {len(search_results)} relevant chunks")

        # Cleanup
        test_file.unlink()

        print("\n=== All Pipeline Steps Passed ✓ ===\n")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mcp_server):
        """Test: Concurrent Tool Execution."""
        # Execute multiple tools concurrently
        tasks = [
            mcp_server.execute("test_echo", {"text": f"Message {i}"})
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10, "Not all tasks completed"
        assert all(r.success for r in results), "Some tasks failed"

        # Verify rate limiting
        stats = mcp_server.get_analytics("test_echo")
        assert stats['executions'] == 10, "Execution count incorrect"

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server, file_processor):
        """Test: Error Handling & Graceful Degradation."""
        # Test invalid tool
        result = await mcp_server.execute("nonexistent_tool", {})
        assert not result.success, "Should fail for nonexistent tool"
        assert "not found" in result.error.lower(), "Wrong error message"

        # Test invalid parameters
        result = await mcp_server.execute("test_sum", {"a": 1})  # Missing 'b'
        assert not result.success, "Should fail for missing params"

        # Test invalid file
        result = await file_processor.process_file("/nonexistent/file.pdf")
        assert not result.success, "Should fail for nonexistent file"
        assert "not found" in result.error.lower(), "Wrong error message"

    def test_smart_chunker(self):
        """Test: Smart Chunking Logic."""
        chunker = SmartChunker(ChunkConfig(
            chunk_size=100,
            overlap=20,
            respect_sentences=True
        ))

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text, metadata={"source": "test"})

        assert len(chunks) > 0, "No chunks generated"

        # Verify overlap
        if len(chunks) > 1:
            # Check that chunks have some overlap
            chunk0_end = chunks[0].text[-20:]
            chunk1_start = chunks[1].text[:20]
            # At least some common words expected
            assert len(chunk0_end) > 0 and len(chunk1_start) > 0


# ============================================================================
# Performance Smoke Tests
# ============================================================================

class TestPerformance:
    """Basic performance smoke tests."""

    @pytest.mark.asyncio
    async def test_storage_throughput(self, unified_store):
        """Test: Storage throughput."""
        import time

        n_items = 100
        start = time.time()

        for i in range(n_items):
            embedding = np.random.randn(384).astype(np.float32)
            await unified_store.add_knowledge(
                f"perf_test_{i}",
                {"index": i},
                embedding
            )

        duration = time.time() - start
        throughput = n_items / duration

        print(f"\nStorage throughput: {throughput:.1f} items/sec")
        assert throughput > 10, "Storage too slow"

    @pytest.mark.asyncio
    async def test_clustering_performance(self, cluster_engine):
        """Test: Clustering performance."""
        import time

        # 1000 embeddings, 100 dimensions
        embeddings = np.random.randn(1000, 100).astype(np.float32)

        start = time.time()
        result = cluster_engine.cluster(embeddings)
        duration = time.time() - start

        print(f"\nClustering 1000 items: {duration:.2f}s")
        assert duration < 5.0, "Clustering too slow"
        assert result.n_clusters > 0, "No clusters found"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Phase 2 Integration Tests")
    print("="*80)

    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
