"""
End-to-End Benchmarks: Complete Memory System (Weeks 1-4)

Performance benchmarks covering:
- Storage latency (Week 2)
- Retrieval latency (Week 4)
- Consolidation throughput (Week 3)
- Memory scaling (all weeks)
- Production scenarios

Target Performance:
- Store: <5ms per memory
- Retrieve (hybrid): <100ms per query
- Consolidate: <500ms per batch (rule-based)
- Scale: 10k+ memories without degradation
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import List

from HoloLoom.memory.integrated_memory_system import (
    create_integrated_memory_system,
    create_production_memory_system
)
from HoloLoom.memory.lifecycle_manager import MemoryScope
from HoloLoom.memory.consolidation import ConsolidationStrategy
from HoloLoom.documentation.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_memories():
    """Create diverse sample memories for benchmarking."""
    topics = [
        "Thompson Sampling",
        "Epsilon-Greedy",
        "UCB",
        "Bayesian Optimization",
        "Multi-Armed Bandits",
        "Exploration vs Exploitation",
        "Reinforcement Learning",
        "Policy Gradient",
        "Q-Learning",
        "Deep RL"
    ]

    memories = []
    for i, topic in enumerate(topics * 10):  # 100 memories
        memories.append({
            "content": f"{topic} is a technique in machine learning. Detail {i}.",
            "importance": 0.5 + (i % 5) * 0.1,
            "entities": [topic.replace(" ", ""), "MachineLearning"]
        })

    return memories


# ============================================================================
# Benchmark 1: Storage Performance
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_storage_latency():
    """Benchmark: Storage latency (Week 2)."""
    system = create_integrated_memory_system()

    # Warm-up
    await system.store("Warm-up memory", importance=0.5)

    # Benchmark 100 stores
    latencies = []

    for i in range(100):
        start = time.perf_counter()

        result = await system.store(
            f"Memory {i}: Thompson Sampling detail",
            importance=0.8,
            entities=["ThompsonSampling"]
        )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        assert result.success is True

    # Statistics
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Storage Latency (100 memories)")
    print(f"{'='*60}")
    print(f"Average: {avg_latency:.2f}ms")
    print(f"P50: {p50:.2f}ms")
    print(f"P95: {p95:.2f}ms")
    print(f"P99: {p99:.2f}ms")
    print(f"{'='*60}")

    # Assert performance targets
    assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms target"
    assert p95 < 10.0, f"P95 latency {p95:.2f}ms exceeds 10ms target"


@pytest.mark.asyncio
async def test_benchmark_storage_throughput():
    """Benchmark: Storage throughput (memories/second)."""
    system = create_integrated_memory_system()

    num_memories = 1000

    start = time.perf_counter()

    # Store 1000 memories
    tasks = [
        system.store(f"Memory {i}", importance=0.5)
        for i in range(num_memories)
    ]

    results = await asyncio.gather(*tasks)

    end = time.perf_counter()
    duration = end - start
    throughput = num_memories / duration

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Storage Throughput")
    print(f"{'='*60}")
    print(f"Memories: {num_memories}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.0f} memories/second")
    print(f"{'='*60}")

    assert all(r.success for r in results)
    assert throughput > 100, f"Throughput {throughput:.0f}/s below 100/s target"


# ============================================================================
# Benchmark 2: Retrieval Performance
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_retrieval_latency(sample_memories):
    """Benchmark: Hybrid retrieval latency (Week 4)."""
    system = create_integrated_memory_system()

    # Store memories
    for mem in sample_memories:
        await system.store(
            mem["content"],
            importance=mem["importance"],
            entities=mem["entities"]
        )

    # Benchmark queries
    queries = [
        "Thompson Sampling exploration",
        "Bayesian approach",
        "Multi-armed bandits",
        "Epsilon-greedy strategy",
        "Reinforcement learning"
    ]

    latencies = []

    for query in queries * 20:  # 100 queries
        start = time.perf_counter()

        result = await system.retrieve(query, limit=10)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        assert len(result.memories) > 0

    # Statistics
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Hybrid Retrieval Latency (100 memories)")
    print(f"{'='*60}")
    print(f"Average: {avg_latency:.2f}ms")
    print(f"P50: {p50:.2f}ms")
    print(f"P95: {p95:.2f}ms")
    print(f"P99: {p99:.2f}ms")
    print(f"{'='*60}")

    # Assert performance targets (100ms average)
    assert avg_latency < 100.0, f"Average latency {avg_latency:.2f}ms exceeds 100ms target"
    assert p95 < 200.0, f"P95 latency {p95:.2f}ms exceeds 200ms target"


@pytest.mark.asyncio
async def test_benchmark_retrieval_methods_comparison():
    """Benchmark: Compare BM25 vs Semantic vs Graph vs Hybrid."""
    from HoloLoom.memory.hybrid_retrieval import (
        BM25Retriever,
        SemanticRetriever,
        GraphRetriever,
        HybridRetriever,
        create_hybrid_retriever
    )
    from HoloLoom.memory.graph import KG

    # Create sample memories
    memories = [
        MemoryShard(
            id=f"m{i}",
            text=f"Thompson Sampling is a Bayesian approach. Detail {i}.",
            entities=["ThompsonSampling", "Bayesian"],
            metadata={"timestamp": datetime.now().isoformat()}
        )
        for i in range(100)
    ]

    kg = KG()
    query = "Thompson Sampling Bayesian"

    # Test each method
    methods = {
        "BM25": BM25Retriever(),
        "Semantic": SemanticRetriever(enable_fallback=True),
        "Graph": GraphRetriever(kg=kg),
        "Hybrid": create_hybrid_retriever(kg=kg, enable_all=True)
    }

    results = {}

    for name, retriever in methods.items():
        latencies = []

        for _ in range(50):
            start = time.perf_counter()

            if name == "Hybrid":
                result = await retriever.retrieve(query, memories, limit=10)
                retrieved = len(result.memories)
            else:
                result = await retriever.retrieve(query, memories, limit=10)
                retrieved = len(result)

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        avg = sum(latencies) / len(latencies)
        results[name] = {"avg_ms": avg, "retrieved": retrieved}

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Retrieval Method Comparison (100 memories)")
    print(f"{'='*60}")
    for name, stats in results.items():
        print(f"{name:15} {stats['avg_ms']:8.2f}ms  (retrieved: {stats['retrieved']})")
    print(f"{'='*60}")


# ============================================================================
# Benchmark 3: Consolidation Performance
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_consolidation_throughput():
    """Benchmark: Consolidation throughput (Week 3)."""
    system = create_integrated_memory_system()

    # Store 50 episodic memories
    for i in range(50):
        await system.store(
            f"Episode {i}: Discussion about Thompson Sampling and exploration strategies",
            metadata={"session_only": True},
            importance=0.6
        )

    # Benchmark consolidation
    start = time.perf_counter()

    result = await system.consolidate(
        strategy=ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours=24
    )

    end = time.perf_counter()
    duration_ms = (end - start) * 1000

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Consolidation (50 episodes)")
    print(f"{'='*60}")
    print(f"Input episodes: {result.input_episodes}")
    print(f"Output facts: {result.output_facts}")
    print(f"Duration: {duration_ms:.2f}ms")
    print(f"Throughput: {result.input_episodes / (duration_ms / 1000):.1f} episodes/second")
    print(f"{'='*60}")

    # Assert performance targets (rule-based: <500ms for 50 episodes)
    assert duration_ms < 500.0, f"Consolidation {duration_ms:.2f}ms exceeds 500ms target"
    assert result.input_episodes == 50
    assert result.output_facts > 0


# ============================================================================
# Benchmark 4: Memory Scaling
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_memory_scaling():
    """Benchmark: Performance scaling with memory size."""
    system = create_integrated_memory_system()

    # Test at different memory sizes: 100, 500, 1000, 5000
    sizes = [100, 500, 1000, 5000]
    results = {}

    for size in sizes:
        # Store memories
        for i in range(size):
            await system.store(
                f"Memory {i}: Information about topic {i % 10}",
                importance=0.5 + (i % 5) * 0.1
            )

        # Measure retrieval latency
        latencies = []

        for _ in range(20):
            start = time.perf_counter()
            result = await system.retrieve("Information about topic", limit=10)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        results[size] = avg_latency

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Memory Scaling (retrieval latency)")
    print(f"{'='*60}")
    for size, latency in results.items():
        print(f"{size:6} memories: {latency:8.2f}ms")
    print(f"{'='*60}")

    # Assert no significant degradation (<3x from 100 to 5000)
    degradation = results[5000] / results[100]
    print(f"Degradation (5000/100): {degradation:.2f}x")

    assert degradation < 3.0, f"Degradation {degradation:.2f}x exceeds 3x target"


# ============================================================================
# Benchmark 5: Production Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_production_workload():
    """Benchmark: Realistic production workload."""
    system = create_production_memory_system(
        llm_provider=None,  # Rule-based for benchmark
        llm_model=None
    )

    # Start background consolidation
    await system.start_consolidation()

    # Simulate production workload (5 minutes compressed to 10 seconds)
    # - 100 stores
    # - 200 retrievals
    # - 2 consolidations

    start = time.perf_counter()

    # Concurrent stores
    store_tasks = [
        system.store(f"User action {i}: {i % 10}", importance=0.6)
        for i in range(100)
    ]

    # Concurrent retrievals
    retrieve_tasks = [
        system.retrieve(f"Query {i % 20}", limit=5)
        for i in range(200)
    ]

    # Execute all
    store_results = await asyncio.gather(*store_tasks)
    retrieve_results = await asyncio.gather(*retrieve_tasks)

    # Manual consolidations
    consolidation_results = []
    for _ in range(2):
        result = await system.consolidate()
        consolidation_results.append(result)

    end = time.perf_counter()
    duration = end - start

    # Stop consolidation
    await system.stop_consolidation()

    # Statistics
    total_ops = len(store_results) + len(retrieve_results) + len(consolidation_results)
    throughput = total_ops / duration

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Production Workload")
    print(f"{'='*60}")
    print(f"Operations:")
    print(f"  Stores: {len(store_results)}")
    print(f"  Retrievals: {len(retrieve_results)}")
    print(f"  Consolidations: {len(consolidation_results)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.1f} ops/second")
    print(f"{'='*60}")

    assert all(r.success for r in store_results)
    assert all(len(r.memories) >= 0 for r in retrieve_results)
    assert throughput > 50, f"Throughput {throughput:.1f}/s below 50 ops/s target"


@pytest.mark.asyncio
async def test_benchmark_complete_pipeline():
    """Benchmark: Complete pipeline (store → retrieve → consolidate → retrieve)."""
    system = create_integrated_memory_system()

    start = time.perf_counter()

    # Step 1: Store 20 episodic memories
    for i in range(20):
        await system.store(
            f"Episode {i}: Thompson Sampling balances exploration and exploitation",
            metadata={"session_only": True},
            importance=0.7
        )

    step1 = time.perf_counter()

    # Step 2: Retrieve before consolidation
    result_before = await system.retrieve("Thompson Sampling", limit=5)

    step2 = time.perf_counter()

    # Step 3: Consolidate episodes → facts
    consolidation = await system.consolidate(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    step3 = time.perf_counter()

    # Step 4: Retrieve after consolidation (should include facts)
    result_after = await system.retrieve("Thompson Sampling", limit=5)

    step4 = time.perf_counter()

    # Calculate timings
    timings = {
        "Store (20)": (step1 - start) * 1000,
        "Retrieve (before)": (step2 - step1) * 1000,
        "Consolidate": (step3 - step2) * 1000,
        "Retrieve (after)": (step4 - step3) * 1000,
        "Total": (step4 - start) * 1000
    }

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Complete Pipeline")
    print(f"{'='*60}")
    for step, duration in timings.items():
        print(f"{step:20} {duration:8.2f}ms")
    print(f"{'='*60}")
    print(f"Consolidation: {consolidation.input_episodes} episodes → {consolidation.output_facts} facts")
    print(f"{'='*60}")

    # Assertions
    assert len(result_before.memories) > 0
    assert consolidation.input_episodes == 20
    assert consolidation.output_facts > 0
    assert len(result_after.memories) > 0
    assert timings["Total"] < 1000.0, f"Total pipeline {timings['Total']:.2f}ms exceeds 1s"


# ============================================================================
# Benchmark Summary
# ============================================================================

@pytest.mark.asyncio
async def test_benchmark_summary():
    """Print benchmark summary and targets."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK TARGETS (Production)")
    print(f"{'='*60}")
    print(f"Storage:")
    print(f"  - Latency: <5ms average, <10ms P95")
    print(f"  - Throughput: >100 memories/second")
    print(f"")
    print(f"Retrieval:")
    print(f"  - Latency (hybrid): <100ms average, <200ms P95")
    print(f"  - BM25: ~20ms")
    print(f"  - Semantic: ~50ms (with fallback)")
    print(f"  - Graph: ~30ms")
    print(f"")
    print(f"Consolidation:")
    print(f"  - Rule-based: <500ms for 50 episodes")
    print(f"  - LLM-based: <3s for 50 episodes (depends on API)")
    print(f"")
    print(f"Scaling:")
    print(f"  - <3x degradation from 100 to 5000 memories")
    print(f"")
    print(f"Production:")
    print(f"  - >50 ops/second (mixed workload)")
    print(f"  - Complete pipeline <1s")
    print(f"{'='*60}")
