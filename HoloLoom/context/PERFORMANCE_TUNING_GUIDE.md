# HoloLoom Performance Tuning Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-13
**Part 5: Production Hardening - Day 25**

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Benchmarking Strategy](#benchmarking-strategy)
3. [Configuration Tuning](#configuration-tuning)
4. [Cache Optimization](#cache-optimization)
5. [Memory Optimization](#memory-optimization)
6. [CPU Optimization](#cpu-optimization)
7. [Backend Optimization](#backend-optimization)
8. [Network Optimization](#network-optimization)
9. [Workload-Specific Tuning](#workload-specific-tuning)
10. [Performance Testing](#performance-testing)

---

## Performance Overview

### Current Performance Characteristics

**Baseline Performance** (FAST mode, warm cache):
- Query latency p50: ~150ms
- Query latency p95: ~300ms
- Query latency p99: ~500ms
- Throughput: ~100 QPS (single instance)
- Memory: ~500MB baseline, ~2GB under load
- CPU: ~40% under typical load

**Performance Targets**:

| Metric | Target (p50) | Target (p95) | Target (p99) |
|--------|-------------|-------------|-------------|
| Latency | <200ms | <500ms | <1000ms |
| Throughput | >50 QPS | >100 QPS | >200 QPS |
| Memory | <1GB | <2GB | <4GB |
| CPU | <50% | <75% | <90% |
| Cache Hit Rate | >50% | >70% | >80% |

### Performance Overhead Breakdown

**Per-Query Overhead** (FAST mode):

| Stage | Duration | % of Total |
|-------|----------|-----------|
| Rate limiting | 0.1ms | <1% |
| Retrieval | 80-120ms | 50-60% |
| Feature extraction | 20-40ms | 15-20% |
| Decision | 15-30ms | 10-15% |
| Tool execution | 10-50ms | 10-20% |
| Monitoring | 0.5ms | <1% |
| **Total** | **125-250ms** | **100%** |

**Production Hardening Overhead**: <1ms per query (<1% of total)

---

## Benchmarking Strategy

### Setting Up Benchmarks

#### 1. Load Testing Setup

```python
import asyncio
import time
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard

async def benchmark_query_latency(
    orchestrator: WeavingOrchestrator,
    queries: List[str],
    iterations: int = 100
):
    """Benchmark query latency."""
    latencies = []

    for query_text in queries:
        query = Query(text=query_text)

        for _ in range(iterations):
            start = time.perf_counter()
            await orchestrator.weave(query)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

    # Calculate statistics
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"Latency p50: {p50:.1f}ms")
    print(f"Latency p95: {p95:.1f}ms")
    print(f"Latency p99: {p99:.1f}ms")

    return {'p50': p50, 'p95': p95, 'p99': p99}

# Usage
config = Config.fast()
shards = create_test_shards()

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    queries = [
        "What is Thompson Sampling?",
        "How does it work?",
        "Show me an example"
    ]

    results = await benchmark_query_latency(orchestrator, queries)
```

#### 2. Throughput Benchmark

```python
async def benchmark_throughput(
    orchestrator: WeavingOrchestrator,
    queries: List[str],
    duration_seconds: int = 60
):
    """Benchmark queries per second."""
    start_time = time.time()
    query_count = 0

    while time.time() - start_time < duration_seconds:
        query_text = queries[query_count % len(queries)]
        await orchestrator.weave(Query(text=query_text))
        query_count += 1

    elapsed = time.time() - start_time
    qps = query_count / elapsed

    print(f"Queries: {query_count}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"QPS: {qps:.1f}")

    return qps

# Usage
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    qps = await benchmark_throughput(orchestrator, queries, duration_seconds=60)
```

#### 3. Memory Profiling

```python
from memory_profiler import memory_usage
import tracemalloc

async def benchmark_memory(
    orchestrator: WeavingOrchestrator,
    queries: List[str],
    iterations: int = 100
):
    """Benchmark memory usage."""
    tracemalloc.start()

    # Baseline
    baseline = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

    # Run queries
    for _ in range(iterations):
        for query_text in queries:
            await orchestrator.weave(Query(text=query_text))

    # Peak memory
    current, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / 1024 / 1024

    print(f"Baseline: {baseline:.1f} MB")
    print(f"Peak: {peak_mb:.1f} MB")
    print(f"Delta: {peak_mb - baseline:.1f} MB")

    tracemalloc.stop()

    return {'baseline': baseline, 'peak': peak_mb, 'delta': peak_mb - baseline}
```

#### 4. Cache Effectiveness

```python
async def benchmark_cache_effectiveness(
    orchestrator: WeavingOrchestrator,
    queries: List[str],
    iterations: int = 100
):
    """Benchmark cache hit rate."""
    hits = 0
    total = 0

    for _ in range(iterations):
        for query_text in queries:
            spacetime = await orchestrator.weave(Query(text=query_text))
            total += 1
            if spacetime.metadata.get('cache_hit', False):
                hits += 1

    hit_rate = hits / total
    print(f"Cache hits: {hits}/{total}")
    print(f"Hit rate: {hit_rate * 100:.1f}%")

    return hit_rate
```

### Benchmark Scenarios

**Scenario 1: Cold Start**
- Clear all caches
- Measure first query latency
- Target: <500ms

**Scenario 2: Warm Cache**
- Run 100 queries to populate cache
- Measure subsequent query latency
- Target: <150ms (3x improvement)

**Scenario 3: Diverse Queries**
- 1000 unique queries (low cache hit rate)
- Measure average latency
- Target: <300ms

**Scenario 4: Burst Traffic**
- 100 concurrent queries
- Measure throughput and latency
- Target: >50 QPS, p95 <1000ms

---

## Configuration Tuning

### Execution Mode Selection

Choose the right mode for your workload:

```python
from HoloLoom.config import Config

# BARE: Fastest, minimal features
# - Use for: Simple factual queries, latency-critical applications
# - Latency: ~50-100ms
# - Memory: ~300MB
config = Config.bare()

# FAST: Balanced (recommended for production)
# - Use for: General-purpose queries, typical web applications
# - Latency: ~150-300ms
# - Memory: ~500MB-2GB
config = Config.fast()

# FUSED: Full features, highest quality
# - Use for: Complex queries, research applications
# - Latency: ~300-600ms
# - Memory: ~1-4GB
config = Config.fused()
```

### Production Hardening Settings

**Conservative (Strict limits)**:
```python
config.enable_production_hardening = True

config.rate_limit.global_qps = 100.0  # Low QPS
config.rate_limit.session_qps = 10.0
config.rate_limit.max_concurrent = 20

config.circuit_breaker.failure_threshold = 3  # Fast failure
config.circuit_breaker.recovery_timeout = 60.0  # Short recovery

config.resource.max_memory_mb = 1024  # 1GB limit
config.resource.max_cache_size = 3000  # Small cache
```

**Balanced (Recommended)**:
```python
config.enable_production_hardening = True

config.rate_limit.global_qps = 1000.0  # Moderate QPS
config.rate_limit.session_qps = 50.0
config.rate_limit.max_concurrent = 100

config.circuit_breaker.failure_threshold = 5  # Balanced
config.circuit_breaker.recovery_timeout = 120.0  # Standard

config.resource.max_memory_mb = 2048  # 2GB limit
config.resource.max_cache_size = 10000  # Moderate cache
```

**Aggressive (High performance)**:
```python
config.enable_production_hardening = True

config.rate_limit.global_qps = 5000.0  # High QPS
config.rate_limit.session_qps = 200.0
config.rate_limit.max_concurrent = 500

config.circuit_breaker.failure_threshold = 10  # Tolerant
config.circuit_breaker.recovery_timeout = 300.0  # Long recovery

config.resource.max_memory_mb = 8192  # 8GB limit
config.resource.max_cache_size = 50000  # Large cache
```

### Retrieval Tuning

**Optimize retrieval for different use cases**:

```python
# Fast retrieval (low latency)
config.retrieval.max_memories = 20  # Fewer memories
config.retrieval.max_depth = 1  # Shallow graph traversal
config.retrieval.similarity_threshold = 0.7  # Strict matching
config.retrieval.timeout = 50  # ms

# Balanced retrieval (recommended)
config.retrieval.max_memories = 50
config.retrieval.max_depth = 2
config.retrieval.similarity_threshold = 0.5
config.retrieval.timeout = 100  # ms

# Deep retrieval (high quality)
config.retrieval.max_memories = 100
config.retrieval.max_depth = 3
config.retrieval.similarity_threshold = 0.3  # Permissive
config.retrieval.timeout = 200  # ms
```

### Embedding Configuration

**Choose embedding model based on tradeoffs**:

```python
# Fast embeddings (384d)
config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
config.matryoshka_scales = [96, 192, 384]
# Latency: ~20ms, Memory: ~200MB

# Balanced embeddings (768d) - recommended
config.embedding_model = "sentence-transformers/all-mpnet-base-v2"
config.matryoshka_scales = [96, 192, 384]
# Latency: ~40ms, Memory: ~400MB

# High-quality embeddings (1024d)
config.embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
config.matryoshka_scales = [128, 256, 512, 1024]
# Latency: ~80ms, Memory: ~800MB
```

---

## Cache Optimization

### Compositional Cache (Phase 5)

**Enable Phase 5 compositional caching** for 10-300x speedup:

```python
config.enable_linguistic_gate = True
config.use_compositional_cache = True

# Cache sizes
config.parse_cache_size = 10000  # Parse cache (X-bar structures)
config.merge_cache_size = 50000  # Merge cache (phrase compositions)
config.semantic_cache_size = 20000  # Semantic cache (244D projections)

# Linguistic mode
config.linguistic_mode = "both"  # Pre-filter + embedding features
config.linguistic_weight = 0.3
config.prefilter_similarity_threshold = 0.3
config.prefilter_keep_ratio = 0.7
```

**Expected Speedups**:
- Cold cache: ~150ms (baseline)
- Warm cache (parse): ~15ms (10x speedup)
- Warm cache (merge): ~5ms (30x speedup)
- Warm cache (semantic): ~0.5ms (300x speedup)

**Production Settings** (90-99% hit rate):
- Expected speedup: 10-17x
- Latency p50: ~10-15ms (from ~150ms)
- Latency p95: ~50ms (from ~300ms)

### Cache Eviction Strategies

```python
# LRU (Least Recently Used) - recommended
config.cache_eviction_policy = "LRU"
config.cache_eviction_threshold = 0.8  # Evict at 80% full

# LFU (Least Frequently Used) - for hot-pattern workloads
config.cache_eviction_policy = "LFU"

# TTL (Time To Live) - for time-sensitive data
config.cache_eviction_policy = "TTL"
config.cache_ttl_seconds = 3600  # 1 hour

# FIFO (First In First Out) - simple, predictable
config.cache_eviction_policy = "FIFO"
```

### Cache Warming

**Pre-populate cache with common queries**:

```python
async def warm_cache(orchestrator, common_queries):
    """Warm cache with common queries."""
    for query_text in common_queries:
        await orchestrator.weave(Query(text=query_text))

# Common queries for typical application
common_queries = [
    "What is Thompson Sampling?",
    "How does reinforcement learning work?",
    "Explain the exploration-exploitation tradeoff",
    "What is a Bayesian prior?",
    "How to implement UCB?"
]

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    await warm_cache(orchestrator, common_queries)

    # Now serve production traffic with warm cache
    spacetime = await orchestrator.weave(user_query)
```

### Cache Monitoring

```python
# Get cache statistics
stats = orchestrator.get_metrics()

cache_stats = stats.get('cache', {})
print(f"Hit rate: {cache_stats.get('hit_rate', 0) * 100:.1f}%")
print(f"Size: {cache_stats.get('size', 0)}/{cache_stats.get('capacity', 0)}")
print(f"Evictions: {cache_stats.get('evictions', 0)}")

# Optimize based on hit rate:
# - <30%: Increase cache size or reduce query diversity
# - 30-60%: Balanced, acceptable
# - >60%: Excellent, cache is effective
```

---

## Memory Optimization

### Memory Budget Allocation

**Total memory budget** (2GB example):
- Baseline system: ~300MB (15%)
- Embeddings: ~400MB (20%)
- Knowledge graph: ~500MB (25%)
- Cache: ~600MB (30%)
- Buffers/overhead: ~200MB (10%)

**Adjust allocation based on workload**:

```python
# High-cache workload (repetitive queries)
config.resource.max_cache_size = 20000  # 40% of memory
config.graph_max_nodes = 10000  # 10% of memory
config.embedding_cache_size = 5000  # 20% of memory

# High-graph workload (complex knowledge traversal)
config.resource.max_cache_size = 5000  # 15% of memory
config.graph_max_nodes = 50000  # 60% of memory
config.embedding_cache_size = 5000  # 15% of memory

# High-embedding workload (diverse queries)
config.resource.max_cache_size = 5000  # 15% of memory
config.graph_max_nodes = 10000  # 20% of memory
config.embedding_cache_size = 20000  # 50% of memory
```

### Memory Leak Detection

```python
import tracemalloc
import gc

tracemalloc.start()

# Take snapshot before
snapshot1 = tracemalloc.take_snapshot()

# Run workload
for _ in range(1000):
    await orchestrator.weave(query)

# Take snapshot after
snapshot2 = tracemalloc.take_snapshot()

# Compare
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("Top 10 memory increases:")
for stat in top_stats[:10]:
    print(stat)

# Force garbage collection
gc.collect()

tracemalloc.stop()
```

### Memory-Efficient Backend

**Use persistent backends** instead of INMEMORY:

```python
# INMEMORY: All data in process memory
config.memory_backend = MemoryBackend.INMEMORY
# Memory usage: ~1-4GB (entire graph in RAM)

# HYBRID: Graph in Neo4j, vectors in Qdrant
config.memory_backend = MemoryBackend.HYBRID
# Memory usage: ~300-800MB (only working set in RAM)

# HYPERSPACE: Advanced gated multipass
config.memory_backend = MemoryBackend.HYPERSPACE
# Memory usage: ~500-1GB (optimized for research workloads)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
async def test_memory_usage():
    """Profile memory usage of specific operation."""
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        for i in range(100):
            await orchestrator.weave(Query(text=f"Query {i}"))

asyncio.run(test_memory_usage())

# Output shows line-by-line memory usage:
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      3    500.0 MB      0.0 MB   async with WeavingOrchestrator(...):
#      4    520.5 MB     20.5 MB       await orchestrator.weave(...)
#      ...
```

---

## CPU Optimization

### CPU Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run workload
await orchestrator.weave(query)

profiler.disable()

# Print top CPU consumers
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)

# Example output:
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#       100    0.050    0.001    2.500    0.025 embedding.py:120(encode)
#        50    0.030    0.001    1.200    0.024 graph.py:200(traverse)
#       200    0.020    0.000    0.800    0.004 policy.py:150(predict)
```

### Parallelization

**Enable parallel processing** for independent operations:

```python
import asyncio

# Parallel retrieval from multiple sources
async def parallel_retrieval(query, sources):
    tasks = [source.retrieve(query) for source in sources]
    results = await asyncio.gather(*tasks)
    return results

# Parallel embedding computation
async def parallel_embeddings(texts):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, embedder.encode, text)
        for text in texts
    ]
    embeddings = await asyncio.gather(*tasks)
    return embeddings

# Configure number of workers
config.num_workers = 4  # CPU cores
```

### Reduce Computation

```python
# Smaller policy network
config.policy_hidden_dim = 256  # from 512
config.policy_num_layers = 2  # from 4

# Fewer attention heads
config.policy_num_heads = 4  # from 8

# Disable expensive features
config.enable_spectral_features = False
config.enable_motif_detection = False

# Reduce embedding dimensions
config.matryoshka_scales = [96, 192]  # from [96, 192, 384]
```

### CPU Affinity (Linux)

```python
import os

# Pin process to specific CPUs
os.sched_setaffinity(0, {0, 1, 2, 3})  # Use CPUs 0-3

# Or in Docker:
# docker run --cpuset-cpus="0-3" hololoom-api
```

---

## Backend Optimization

### Neo4j Optimization

**Index Creation**:
```cypher
-- Create indexes on frequently queried properties
CREATE INDEX entity_name FOR (n:Entity) ON (n.name);
CREATE INDEX entity_type FOR (n:Entity) ON (n.type);
CREATE INDEX edge_type FOR ()-[r:RELATIONSHIP]-() ON (r.type);
CREATE INDEX edge_weight FOR ()-[r:RELATIONSHIP]-() ON (r.weight);

-- Composite index for common query patterns
CREATE INDEX entity_name_type FOR (n:Entity) ON (n.name, n.type);

-- Full-text index for search
CALL db.index.fulltext.createNodeIndex(
  "entityNameFulltext",
  ["Entity"],
  ["name", "description"]
);
```

**Query Optimization**:
```cypher
-- Bad: Full graph scan
MATCH (n:Entity)-[r]->(m:Entity)
RETURN n, r, m;

-- Good: Indexed lookup + limited depth
MATCH (n:Entity {name: 'thompson_sampling'})-[r*1..2]->(m:Entity)
RETURN n, r, m
LIMIT 100;

-- Best: Indexed lookup + filtered edges
MATCH (n:Entity {name: 'thompson_sampling'})-[r:USES|IS_A*1..2]->(m:Entity)
WHERE r.weight > 0.5
RETURN n, r, m
LIMIT 50;
```

**Connection Pooling**:
```python
config.neo4j_max_pool_size = 50  # from 10
config.neo4j_max_connection_lifetime = 3600  # 1 hour
config.neo4j_connection_timeout = 30.0  # seconds
config.neo4j_max_transaction_retry_time = 30.0  # seconds
```

**Memory Configuration** (neo4j.conf):
```conf
# Heap size (50% of RAM for Neo4j)
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=2g

# Page cache (25% of RAM)
dbms.memory.pagecache.size=1g

# Transaction log
dbms.tx_log.rotation.retention_policy=1 days
```

### Qdrant Optimization

**Collection Configuration**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="qdrant", port=6333)

# Optimized collection
client.create_collection(
    collection_name="hololoom",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
        on_disk=False  # Keep in RAM for speed
    ),
    optimizers_config={
        "indexing_threshold": 20000,
        "memmap_threshold": 50000
    },
    hnsw_config={
        "m": 16,  # Number of connections
        "ef_construct": 100,  # Construction quality
        "full_scan_threshold": 10000
    }
)
```

**Query Optimization**:
```python
# Bad: Large limit, no filter
results = client.search(
    collection_name="hololoom",
    query_vector=embedding,
    limit=1000
)

# Good: Small limit, filtered
results = client.search(
    collection_name="hololoom",
    query_vector=embedding,
    limit=50,
    query_filter={
        "must": [
            {"key": "confidence", "range": {"gte": 0.5}}
        ]
    },
    score_threshold=0.7
)
```

**Batch Operations**:
```python
# Bad: Individual inserts
for memory in memories:
    client.upsert(collection_name="hololoom", points=[memory])

# Good: Batch insert
client.upsert(
    collection_name="hololoom",
    points=memories,
    batch_size=100
)
```

### Backend Connection Pooling

```python
# Configure connection pools
config.backend_pool_size = 50  # Concurrent connections
config.backend_pool_timeout = 30.0  # Acquire timeout
config.backend_pool_recycle = 3600  # Recycle after 1 hour
config.backend_pool_pre_ping = True  # Verify connection alive
```

---

## Network Optimization

### HTTP/2 and Connection Reuse

```python
# Use HTTP/2 for multiplexing
config.http_version = "2.0"
config.http_max_connections = 100
config.http_keep_alive = True
config.http_keep_alive_timeout = 60.0
```

### Compression

```python
# Enable response compression
config.enable_compression = True
config.compression_level = 6  # 1-9 (higher = more CPU, smaller size)
config.compression_min_size = 1024  # Only compress >1KB responses
```

### Request Batching

```python
async def batch_queries(queries: List[Query], batch_size: int = 10):
    """Process queries in batches for efficiency."""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]

        # Process batch concurrently
        tasks = [orchestrator.weave(q) for q in batch]
        batch_results = await asyncio.gather(*tasks)

        results.extend(batch_results)

    return results

# Usage
queries = [Query(text=f"Query {i}") for i in range(100)]
results = await batch_queries(queries, batch_size=20)
```

### CDN and Edge Caching

For static resources and common queries:

```python
# Add cache headers for edge caching
config.cache_control_header = "public, max-age=3600"  # 1 hour
config.etag_enabled = True
config.vary_header = "Accept-Encoding"

# CloudFlare-specific
config.cf_cache_everything = True
config.cf_browser_cache_ttl = 3600
```

---

## Workload-Specific Tuning

### High-Throughput Workload

**Characteristics**: Many queries per second, low complexity

**Optimizations**:
```python
config = Config.bare()  # Minimal processing

config.rate_limit.global_qps = 5000.0
config.rate_limit.max_concurrent = 500

config.retrieval.max_memories = 20
config.retrieval.max_depth = 1

config.enable_compositional_cache = True
config.cache_size = 50000  # Large cache

config.memory_backend = MemoryBackend.HYBRID  # Offload to backends
```

### Low-Latency Workload

**Characteristics**: Latency-critical, <100ms target

**Optimizations**:
```python
config = Config.bare()

config.retrieval.timeout = 20  # ms
config.retrieval.max_memories = 10

config.policy_mode = "LITE"  # Simplest policy

config.enable_compositional_cache = True
config.parse_cache_size = 20000

# Disable expensive features
config.enable_spectral_features = False
config.enable_refinement = False

# Aggressive timeouts
config.backend_timeout = 50  # ms
```

### High-Quality Workload

**Characteristics**: Research, complex queries, quality over speed

**Optimizations**:
```python
config = Config.fused()  # Full features

config.retrieval.max_memories = 100
config.retrieval.max_depth = 3

config.enable_refinement = True
config.refinement_threshold = 0.75
config.refinement_max_iterations = 5

config.matryoshka_scales = [96, 192, 384]
config.enable_spectral_features = True

# Generous timeouts
config.backend_timeout = 1000  # ms
config.rate_limit.global_qps = 100  # Lower throughput acceptable
```

### Memory-Constrained Workload

**Characteristics**: Limited memory (<1GB)

**Optimizations**:
```python
config = Config.fast()

config.resource.max_memory_mb = 1024  # 1GB
config.resource.max_cache_size = 3000

config.matryoshka_scales = [96, 192]  # Smaller embeddings

config.memory_backend = MemoryBackend.HYBRID  # Offload to disk

# Aggressive eviction
config.cache_eviction_threshold = 0.6  # Evict at 60%
config.cache_eviction_policy = "LRU"
```

---

## Performance Testing

### Load Testing

**Use Locust for load testing**:

```python
# locustfile.py
from locust import HttpUser, task, between

class HoloLoomUser(HttpUser):
    wait_time = between(1, 3)  # Seconds between requests

    @task
    def query_hololoom(self):
        self.client.post("/query", json={
            "text": "What is Thompson Sampling?",
            "context": {}
        })

# Run load test:
# locust -f locustfile.py --host=http://localhost:8080 --users=100 --spawn-rate=10
```

**Expected Results** (100 concurrent users):
- Throughput: >50 QPS
- Latency p50: <300ms
- Latency p95: <800ms
- Error rate: <2%

### Stress Testing

**Push system to limits**:

```python
# stress_test.py
import asyncio
import aiohttp

async def stress_test(url, num_requests=10000):
    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(num_requests):
            task = session.post(url, json={
                "text": f"Query {i}",
                "context": {}
            })
            tasks.append(task)

        # Fire all requests at once
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successes = sum(1 for r in responses if not isinstance(r, Exception))
        failures = sum(1 for r in responses if isinstance(r, Exception))

        print(f"Successes: {successes}/{num_requests}")
        print(f"Failures: {failures}/{num_requests}")
        print(f"Success rate: {successes / num_requests * 100:.1f}%")

asyncio.run(stress_test("http://localhost:8080/query", num_requests=10000))
```

### Endurance Testing

**Test for memory leaks and degradation over time**:

```python
async def endurance_test(duration_hours=24):
    """Run continuous queries for extended period."""
    start_time = time.time()
    query_count = 0
    errors = 0

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        while time.time() - start_time < duration_hours * 3600:
            try:
                await orchestrator.weave(Query(text=f"Query {query_count}"))
                query_count += 1

                # Sample metrics every 1000 queries
                if query_count % 1000 == 0:
                    metrics = orchestrator.get_metrics()
                    print(f"[{query_count}] Memory: {metrics['resources']['memory_mb']:.1f}MB")
                    print(f"[{query_count}] Latency p95: {metrics['performance']['latency_p95']:.1f}ms")

                await asyncio.sleep(0.1)  # 10 QPS sustained

            except Exception as e:
                errors += 1
                print(f"Error: {e}")

    print(f"Endurance test complete:")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Queries: {query_count}")
    print(f"  Errors: {errors}")
    print(f"  Error rate: {errors / query_count * 100:.2f}%")

asyncio.run(endurance_test(duration_hours=24))
```

### Benchmark Regression Testing

**Track performance over releases**:

```python
# benchmark_suite.py
import json
from datetime import datetime

async def run_benchmark_suite():
    """Run full benchmark suite and save results."""
    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'config': 'FAST',
            'benchmarks': {}
        }

        # Latency benchmark
        latencies = await benchmark_query_latency(orchestrator, queries, iterations=100)
        results['benchmarks']['latency'] = latencies

        # Throughput benchmark
        qps = await benchmark_throughput(orchestrator, queries, duration_seconds=60)
        results['benchmarks']['throughput'] = {'qps': qps}

        # Memory benchmark
        memory = await benchmark_memory(orchestrator, queries, iterations=100)
        results['benchmarks']['memory'] = memory

        # Cache benchmark
        cache_hit_rate = await benchmark_cache_effectiveness(orchestrator, queries, iterations=100)
        results['benchmarks']['cache'] = {'hit_rate': cache_hit_rate}

        # Save results
        with open(f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

# Run and compare across releases
results = asyncio.run(run_benchmark_suite())
```

---

## Performance Monitoring

### Key Metrics to Track

**Latency Metrics**:
- p50, p95, p99 query latency
- Per-stage latency breakdown
- Slow query threshold violations (>1s)

**Throughput Metrics**:
- Queries per second (QPS)
- Requests per minute (RPM)
- Peak QPS during bursts

**Resource Metrics**:
- Memory usage (MB)
- CPU utilization (%)
- Disk I/O (MB/s)
- Network I/O (MB/s)

**Cache Metrics**:
- Cache hit rate (%)
- Cache evictions per minute
- Cache size utilization (%)

**Error Metrics**:
- Error rate (%)
- Circuit breaker trips
- Rate limit rejections

### Prometheus Metrics

```python
# Export metrics for Prometheus
from prometheus_client import Counter, Histogram, Gauge

# Latency histogram
query_latency = Histogram(
    'hololoom_query_latency_seconds',
    'Query latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# QPS counter
query_count = Counter(
    'hololoom_query_total',
    'Total queries processed'
)

# Cache hit rate
cache_hit_rate = Gauge(
    'hololoom_cache_hit_rate',
    'Cache hit rate'
)

# Record metrics
with query_latency.time():
    await orchestrator.weave(query)

query_count.inc()
cache_hit_rate.set(metrics['cache']['hit_rate'])
```

### Grafana Dashboard

**Key Panels**:
1. Query Latency (p50, p95, p99) - Line chart
2. QPS - Line chart
3. Cache Hit Rate - Gauge
4. Memory Usage - Line chart
5. CPU Usage - Line chart
6. Error Rate - Line chart
7. Circuit Breaker States - Bar chart

See `OPERATIONS_RUNBOOK.md` for full Grafana dashboard JSON.

---

## Performance Tuning Checklist

### Pre-Production

- [ ] Run benchmark suite and establish baseline
- [ ] Profile memory usage and identify leaks
- [ ] Profile CPU usage and optimize hot paths
- [ ] Test cache effectiveness (target >50% hit rate)
- [ ] Load test with 2x expected peak traffic
- [ ] Stress test to find breaking point
- [ ] Endurance test for 24 hours

### Configuration Review

- [ ] Choose appropriate execution mode (BARE/FAST/FUSED)
- [ ] Set rate limits based on capacity testing
- [ ] Configure circuit breaker thresholds
- [ ] Set resource limits (memory, cache)
- [ ] Enable compositional cache (Phase 5)
- [ ] Configure backend connection pools
- [ ] Set appropriate timeouts

### Backend Optimization

- [ ] Create Neo4j indexes on common query patterns
- [ ] Optimize Neo4j memory configuration
- [ ] Configure Qdrant HNSW parameters
- [ ] Enable backend connection pooling
- [ ] Test backend failover and auto-fallback

### Monitoring Setup

- [ ] Configure Prometheus metrics export
- [ ] Create Grafana dashboards
- [ ] Set up alerting rules
- [ ] Enable health check endpoint
- [ ] Configure log aggregation

### Post-Deployment

- [ ] Monitor latency metrics daily
- [ ] Review cache hit rates weekly
- [ ] Check for memory leaks monthly
- [ ] Run benchmark regression tests per release
- [ ] Tune configuration based on production patterns

---

## Common Performance Issues

### Issue 1: High Latency (p95 >1s)

**Symptoms**:
- Slow query responses
- User complaints about performance

**Diagnosis**:
1. Check stage timings: `spacetime.trace.stage_durations`
2. Identify bottleneck: retrieval? decision? backend?

**Solutions**:
- **Retrieval slow**: Reduce `max_memories`, increase `similarity_threshold`
- **Decision slow**: Use FAST mode instead of FUSED
- **Backend slow**: Add Neo4j indexes, optimize queries
- **Cache ineffective**: Increase cache size, enable compositional cache

### Issue 2: Low Throughput (<50 QPS)

**Symptoms**:
- System can't handle traffic load
- Rate limits triggered frequently

**Diagnosis**:
1. Check CPU usage: `docker stats`
2. Check concurrent connections: `get_metrics()['rate_limiter']`

**Solutions**:
- **CPU bound**: Increase replicas, reduce computation (smaller models)
- **Memory bound**: Increase memory limits, use HYBRID backend
- **Backend bound**: Increase backend pool size, add read replicas
- **Rate limited**: Increase rate limits if capacity allows

### Issue 3: High Memory Usage (>2GB)

**Symptoms**:
- OOM kills
- Swapping to disk

**Diagnosis**:
1. Profile memory: `tracemalloc`
2. Check cache size: `get_metrics()['cache']`

**Solutions**:
- **Cache too large**: Reduce `max_cache_size`
- **Graph too large**: Use HYBRID backend instead of INMEMORY
- **Memory leak**: Profile with `memory_profiler`, fix leaks
- **Embeddings large**: Use smaller model (384d instead of 768d)

---

## Additional Resources

- **Operations Runbook**: `OPERATIONS_RUNBOOK.md` (deployment, monitoring)
- **Troubleshooting Guide**: `TROUBLESHOOTING_GUIDE.md` (debugging)
- **Production Integration**: `PRODUCTION_INTEGRATION_COMPLETE.md` (usage examples)
- **Phase 5 Documentation**: `PHASE_5_COMPLETE.md` (compositional cache)
- **Benchmark Scripts**: `experiments/run_experiments.py`

---

**End of Performance Tuning Guide**
