# HoloLoom Performance Testing at Scale

Comprehensive guide for benchmarking and stress-testing Phase 2 components under realistic loads.

## Overview

HoloLoom's Phase 2 introduces several components that require performance validation:
- **Neo4j + Qdrant** hybrid storage
- **Clustering** algorithms on large embedding sets
- **MCP server** concurrent tool execution
- **File processing** batch operations
- **End-to-end pipeline** latency

This guide covers testing at scale with **real databases** and **production-like workloads**.

## Quick Start

### 1. Run Performance Tests (Mock Mode)

```bash
# No external dependencies required
pytest HoloLoom/tests/performance/ -v -s
```

### 2. Run with Real Databases (Docker)

```bash
# Start Neo4j and Qdrant
docker-compose -f HoloLoom/tests/docker-compose.yml up -d

# Run performance tests
pytest HoloLoom/tests/performance/ -v -s

# Cleanup
docker-compose -f HoloLoom/tests/docker-compose.yml down
```

### 3. Generate Performance Report

```bash
./HoloLoom/tests/run_performance_report.sh
```

## Performance Targets

### Baseline Performance (Minimum Acceptable)

| Component          | Metric                    | Target        | Critical |
|--------------------|---------------------------|---------------|----------|
| Storage (Write)    | 1K items                  | >50/sec       | >25/sec  |
| Storage (Write)    | 10K items (batched)       | >100/sec      | >50/sec  |
| Storage (Search)   | Average latency           | <100ms        | <200ms   |
| Storage (Search)   | P95 latency               | <200ms        | <500ms   |
| Clustering         | 1K vectors (384D)         | <2s           | <5s      |
| Clustering         | 10K vectors (384D)        | <10s          | <30s     |
| Clustering         | High-dim (1536D)          | <5s           | <15s     |
| MCP                | Concurrent fast tools     | >500/sec      | >250/sec |
| MCP                | Concurrent slow tools     | <1s wall time | <3s      |
| File Processing    | Batch (10 files)          | >5 files/sec  | >2/sec   |
| File Processing    | Large file (1MB+)         | <5s           | <15s     |
| E2E Pipeline       | Full latency              | <5s           | <15s     |

### Production Performance (Recommended)

| Component          | Metric                    | Target        |
|--------------------|---------------------------|---------------|
| Storage (Write)    | 1K items                  | >200/sec      |
| Storage (Write)    | 10K items (batched)       | >500/sec      |
| Storage (Search)   | Average latency           | <50ms         |
| Storage (Search)   | P99 latency               | <300ms        |
| Clustering         | 1K vectors (384D)         | <1s           |
| Clustering         | 10K vectors (384D)        | <5s           |
| Clustering         | 100K vectors (384D)       | <60s          |
| MCP                | Concurrent fast tools     | >1000/sec     |
| File Processing    | Batch (100 files)         | >20 files/sec |
| E2E Pipeline       | Full latency              | <2s           |

## Docker Setup for Real Database Testing

### Docker Compose Configuration

Create `HoloLoom/tests/docker-compose.yml`:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13.0
    container_name: hololoom-neo4j-test
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      - NEO4J_AUTH=neo4j/testpassword
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_max__size=2G
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "testpassword", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: hololoom-qdrant-test
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT_ALLOW_RECOVERY_MODE=true
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  qdrant_data:
```

### Start Services

```bash
# Start databases
docker-compose -f HoloLoom/tests/docker-compose.yml up -d

# Wait for health checks
docker-compose -f HoloLoom/tests/docker-compose.yml ps

# View logs
docker-compose -f HoloLoom/tests/docker-compose.yml logs -f

# Stop services
docker-compose -f HoloLoom/tests/docker-compose.yml down

# Clean volumes
docker-compose -f HoloLoom/tests/docker-compose.yml down -v
```

## Large-Scale Performance Tests

### 1. Storage Scalability

Test storage performance with increasing dataset sizes.

```python
import pytest
import asyncio
import numpy as np
import time
from HoloLoom.memory import UnifiedStore, Neo4jConfig, QdrantConfig

@pytest.mark.asyncio
@pytest.mark.parametrize("n_items", [100, 1_000, 10_000, 100_000])
async def test_storage_scalability(n_items):
    """Test storage performance at different scales."""

    # Setup
    neo4j_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        database="scale_test",
        auth=("neo4j", "testpassword")
    )
    qdrant_config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name=f"scale_test_{n_items}"
    )

    store = UnifiedStore(neo4j_config, qdrant_config)
    await store.connect()

    # Write test
    start = time.time()
    batch_size = 100

    for batch_start in range(0, n_items, batch_size):
        tasks = []
        for i in range(batch_start, min(batch_start + batch_size, n_items)):
            embedding = np.random.randn(384).astype(np.float32)
            task = store.add_knowledge(
                f"item_{i}",
                {"index": i, "category": f"cat_{i % 100}"},
                embedding
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    write_duration = time.time() - start
    write_throughput = n_items / write_duration

    # Search test
    query_emb = np.random.randn(384).astype(np.float32)
    search_latencies = []

    for _ in range(100):
        start = time.time()
        results = await store.hybrid_search(query_emb, top_k=10)
        latency = (time.time() - start) * 1000
        search_latencies.append(latency)

    avg_latency = np.mean(search_latencies)
    p95_latency = np.percentile(search_latencies, 95)
    p99_latency = np.percentile(search_latencies, 99)

    # Report
    print(f"\n{'='*60}")
    print(f"Storage Scalability Test - {n_items:,} items")
    print(f"{'='*60}")
    print(f"Write throughput: {write_throughput:.1f} items/sec")
    print(f"Write duration:   {write_duration:.2f}s")
    print(f"Search latency:")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  P95:     {p95_latency:.1f}ms")
    print(f"  P99:     {p99_latency:.1f}ms")
    print(f"{'='*60}\n")

    # Cleanup
    await store.close()
```

### 2. Clustering Scalability

Test clustering performance with large embedding sets.

```python
import numpy as np
import time
from HoloLoom.clustering import ClusterEngine, ClusterConfig, ClusterAlgorithm

@pytest.mark.parametrize("n_vectors,n_dims", [
    (1_000, 384),
    (10_000, 384),
    (100_000, 384),
    (1_000, 1536),
    (10_000, 1536),
])
def test_clustering_scalability(n_vectors, n_dims):
    """Test clustering performance at different scales."""

    # Generate synthetic embeddings
    embeddings = np.random.randn(n_vectors, n_dims).astype(np.float32)

    # Test K-means
    kmeans_config = ClusterConfig(
        algorithm=ClusterAlgorithm.KMEANS,
        n_clusters=min(10, n_vectors // 100)
    )
    kmeans_engine = ClusterEngine(kmeans_config)

    start = time.time()
    kmeans_result = kmeans_engine.cluster(embeddings)
    kmeans_duration = time.time() - start

    # Test HDBSCAN
    hdbscan_config = ClusterConfig(algorithm=ClusterAlgorithm.HDBSCAN)
    hdbscan_engine = ClusterEngine(hdbscan_config)

    start = time.time()
    hdbscan_result = hdbscan_engine.cluster(embeddings)
    hdbscan_duration = time.time() - start

    # Report
    print(f"\n{'='*60}")
    print(f"Clustering Scalability - {n_vectors:,} vectors × {n_dims}D")
    print(f"{'='*60}")
    print(f"K-means:")
    print(f"  Duration:  {kmeans_duration:.2f}s")
    print(f"  Clusters:  {kmeans_result.n_clusters}")
    print(f"  Silhouette: {kmeans_result.quality_metrics.get('silhouette', 'N/A')}")
    print(f"\nHDBSCAN:")
    print(f"  Duration:  {hdbscan_duration:.2f}s")
    print(f"  Clusters:  {hdbscan_result.n_clusters}")
    print(f"  Noise pts: {np.sum(hdbscan_result.labels == -1)}")
    print(f"{'='*60}\n")
```

### 3. MCP Concurrent Load Test

Stress-test MCP server with high concurrency.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("n_concurrent", [10, 50, 100, 500, 1000])
async def test_mcp_concurrent_load(n_concurrent):
    """Test MCP server under concurrent load."""
    from HoloLoom.mcp import MCPServer, MCPConfig

    # Create server with high concurrency limit
    server = MCPServer(MCPConfig(max_concurrent_executions=n_concurrent))

    # Register test tool
    @server.tool("compute", "Simulated computation")
    async def compute(duration_ms: float = 10) -> str:
        await asyncio.sleep(duration_ms / 1000)
        return f"Computed for {duration_ms}ms"

    # Concurrent execution test
    start = time.time()

    tasks = [
        server.execute("compute", {"duration_ms": 10})
        for _ in range(n_concurrent)
    ]

    results = await asyncio.gather(*tasks)

    wall_time = time.time() - start
    throughput = n_concurrent / wall_time

    # Report
    success_count = sum(1 for r in results if r.success)

    print(f"\n{'='*60}")
    print(f"MCP Concurrent Load - {n_concurrent} concurrent executions")
    print(f"{'='*60}")
    print(f"Wall time:     {wall_time:.2f}s")
    print(f"Throughput:    {throughput:.1f} tools/sec")
    print(f"Success rate:  {success_count}/{n_concurrent}")
    print(f"Expected time: {n_concurrent * 0.01:.2f}s (if sequential)")
    print(f"Speedup:       {(n_concurrent * 0.01) / wall_time:.1f}x")
    print(f"{'='*60}\n")

    assert success_count == n_concurrent, "Some executions failed"
```

## Performance Monitoring

### Real-Time Metrics Collection

```python
import psutil
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float

class PerformanceMonitor:
    """Collect system metrics during performance tests."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: list[SystemMetrics] = []
        self._running = False
        self._task = None

    async def start(self):
        """Start collecting metrics."""
        self._running = True
        self._task = asyncio.create_task(self._collect())

    async def stop(self):
        """Stop collecting metrics."""
        self._running = False
        if self._task:
            await self._task

    async def _collect(self):
        """Background metric collection."""
        while self._running:
            process = psutil.Process()

            self.metrics.append(SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=process.cpu_percent(),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                memory_percent=process.memory_percent()
            ))

            await asyncio.sleep(self.interval)

    def get_summary(self):
        """Get metric summary."""
        if not self.metrics:
            return {}

        cpu_values = [m.cpu_percent for m in self.metrics]
        mem_values = [m.memory_mb for m in self.metrics]

        return {
            "cpu": {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "p95": np.percentile(cpu_values, 95)
            },
            "memory": {
                "avg_mb": np.mean(mem_values),
                "max_mb": np.max(mem_values),
                "peak_mb": max(mem_values)
            },
            "duration": (self.metrics[-1].timestamp - self.metrics[0].timestamp).total_seconds()
        }
```

### Usage in Tests

```python
@pytest.mark.asyncio
async def test_with_monitoring():
    monitor = PerformanceMonitor(interval=0.5)

    await monitor.start()

    # Run your performance test
    # ...

    await monitor.stop()

    summary = monitor.get_summary()
    print(f"\nSystem Resource Usage:")
    print(f"  CPU avg: {summary['cpu']['avg']:.1f}%")
    print(f"  CPU max: {summary['cpu']['max']:.1f}%")
    print(f"  Memory avg: {summary['memory']['avg_mb']:.1f}MB")
    print(f"  Memory peak: {summary['memory']['peak_mb']:.1f}MB")
```

## Continuous Performance Testing

### GitHub Actions Workflow

Create `.github/workflows/performance-tests.yml`:

```yaml
name: Performance Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  performance:
    runs-on: ubuntu-latest

    services:
      neo4j:
        image: neo4j:5.13.0
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7687:7687

      qdrant:
        image: qdrant/qdrant:v1.7.0
        ports:
          - 6333:6333

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r HoloLoom/requirements-phase2.txt
          pip install pytest pytest-asyncio psutil

      - name: Run performance tests
        run: |
          pytest HoloLoom/tests/performance/ -v -s --tb=short

      - name: Generate performance report
        run: |
          ./HoloLoom/tests/run_performance_report.sh > performance-report.txt

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: performance-report.txt
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j status
docker logs hololoom-neo4j-test

# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p testpassword

# Reset database
docker-compose down -v && docker-compose up -d
```

### Qdrant Connection Issues

```bash
# Check Qdrant status
curl http://localhost:6333/

# View logs
docker logs hololoom-qdrant-test

# Reset
docker-compose restart qdrant
```

### Performance Degradation

**Symptoms:** Tests getting slower over time.

**Causes:**
- Database not indexed properly
- Accumulating data from previous tests
- Resource contention
- Memory leaks

**Fixes:**
```python
# Cleanup after each test
@pytest.fixture(autouse=True)
async def cleanup(unified_store):
    yield
    # Clear database
    await unified_store.clear_all()
```

## Best Practices

1. **Isolate tests:** Each test should use isolated collections/databases
2. **Warm-up runs:** First run may be slower (cold start)
3. **Multiple iterations:** Run tests 3-5 times, report median
4. **Resource monitoring:** Track CPU/memory during tests
5. **Real data:** Use production-like data distributions
6. **Consistent environment:** Lock Docker image versions
7. **Document baselines:** Track performance over time

## Performance Optimization Tips

### Neo4j Optimization

```cypher
-- Create indexes
CREATE INDEX entity_id_index FOR (n:Entity) ON (n.entity_id);
CREATE INDEX properties_index FOR (n:Entity) ON (n.properties);

-- Check query plan
EXPLAIN MATCH (n:Entity {entity_id: 'xyz'}) RETURN n;

-- Tune memory
-- In docker-compose.yml:
# NEO4J_dbms_memory_heap_max__size=4G
# NEO4J_dbms_memory_pagecache_size=2G
```

### Qdrant Optimization

```python
# Use batching for bulk inserts
await qdrant.upsert_batch(vectors, batch_size=100)

# Configure HNSW index
collection_config = {
    "hnsw_config": {
        "m": 16,  # Number of connections
        "ef_construct": 100  # Construction time vs quality tradeoff
    }
}

# Use quantization for memory efficiency
await qdrant.update_collection(
    collection_name="embeddings",
    quantization_config={"scalar": {"type": "int8"}}
)
```

### Python Optimization

```python
# Use asyncio.gather for parallel operations
await asyncio.gather(*tasks)

# Profile with cProfile
import cProfile
cProfile.run('my_function()', 'profile.stats')

# Monitor memory with memray
pip install memray
memray run my_script.py
memray flamegraph profile.bin
```

## Summary

Performance testing ensures HoloLoom can handle production workloads. Key takeaways:

- **Test early, test often** - Catch regressions before production
- **Use real infrastructure** - Mock mode is for development only
- **Monitor resources** - CPU/memory usage indicates bottlenecks
- **Document baselines** - Track performance trends over time
- **Optimize iteratively** - Profile, fix bottlenecks, re-test

For questions or performance issues, open an issue with:
- Test output
- System specs
- Database versions
- Performance metrics
