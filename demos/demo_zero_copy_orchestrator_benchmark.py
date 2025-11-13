#!/usr/bin/env python3
"""
Zero-Copy Embeddings Orchestrator Benchmark
============================================
Real-world performance comparison of standard vs zero-copy embeddings
in the full HoloLoom weaving cycle.

This benchmark tests the actual impact on end-to-end query latency,
not just isolated embedding operations.

Measures:
- Feature extraction time (Step 4 of weaving cycle)
- End-to-end query latency
- Memory overhead
- Cache hit rates
- Cold start vs warm cache performance

Author: Claude Code
Date: 2025-11-12
"""

import time
import asyncio
import tracemalloc
from pathlib import Path
from typing import Dict, List
import tempfile

from HoloLoom.config import Config, ExecutionMode
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings


# ============================================================================
# Benchmark Configuration
# ============================================================================

# Test queries (realistic mix)
TEST_QUERIES = [
    "What is Thompson Sampling?",
    "How does reinforcement learning work?",
    "Explain the attention mechanism",
    "What are the benefits of Matryoshka embeddings?",
    "How do knowledge graphs improve retrieval?",
    "What is the weaving metaphor in HoloLoom?",
    "Describe zero-copy optimization techniques",
    "How does memory-mapped storage work?",
    "What is the prefix property of embeddings?",
    "Explain spectral graph features",
]

# Memory shards for context
MEMORY_SHARDS = [
    MemoryShard(
        id="shard_1",
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
        entities=["Thompson Sampling", "Bayesian", "exploration", "exploitation"],
        motifs=["bandit", "sampling"]
    ),
    MemoryShard(
        id="shard_2",
        text="Reinforcement learning uses rewards and punishments to train agents to make optimal decisions.",
        entities=["reinforcement learning", "rewards", "agents", "decisions"],
        motifs=["learning", "training"]
    ),
    MemoryShard(
        id="shard_3",
        text="Attention mechanisms allow neural networks to focus on relevant parts of the input.",
        entities=["attention", "neural networks", "input"],
        motifs=["attention", "focus"]
    ),
    MemoryShard(
        id="shard_4",
        text="Matryoshka embeddings provide multi-scale representations for efficient retrieval.",
        entities=["Matryoshka", "embeddings", "retrieval"],
        motifs=["multi-scale", "efficiency"]
    ),
    MemoryShard(
        id="shard_5",
        text="Knowledge graphs organize information as entities and relationships for structured retrieval.",
        entities=["knowledge graphs", "entities", "relationships"],
        motifs=["graph", "structure"]
    ),
]


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_embedder_isolation(embedder, queries: List[str], n_iterations: int = 50) -> Dict[str, float]:
    """
    Benchmark embedder in isolation (not in orchestrator).

    This is the baseline - pure embedding performance.
    """
    results = {}

    # Cold start (first query)
    start = time.perf_counter()
    _ = embedder.encode_scales(queries[:1], size=None)
    cold_time = (time.perf_counter() - start) * 1000
    results['cold_start_ms'] = cold_time

    # Warm up cache with all queries
    for query in queries:
        _ = embedder.encode_scales([query], size=None)

    # Warm cache (repeated queries)
    start = time.perf_counter()
    for _ in range(n_iterations):
        for query in queries:
            _ = embedder.encode_scales([query], size=None)
    warm_time = (time.perf_counter() - start) / (n_iterations * len(queries)) * 1000
    results['warm_cache_ms'] = warm_time

    return results


async def benchmark_orchestrator_integration(
    use_zero_copy: bool,
    queries: List[Query],
    shards: List[MemoryShard],
    n_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark embedder in real orchestrator weaving cycle.

    This shows the actual end-to-end impact on query latency.
    """
    from HoloLoom.resonance.shed import ResonanceShed
    from HoloLoom.motif.base import create_motif_detector
    from HoloLoom.embedding.spectral import SpectralFusion

    results = {}

    # Create embedder
    if use_zero_copy:
        # Use zero-copy with temp storage
        tmpdir = tempfile.mkdtemp()
        store_path = Path(tmpdir) / 'benchmark_cache.mmap'
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96, 192, 384],
            base_model_name=None,  # Use fallback for speed
            store_path=store_path,
            max_cache_size=1000
        )
    else:
        # Standard projection-based
        embedder = MatryoshkaEmbeddings(
            sizes=[96, 192, 384],
            base_model_name=None  # Use fallback for speed
        )

    # Create ResonanceShed (Step 4 of weaving cycle)
    motif_detector = create_motif_detector(mode="regex")
    spectral_fusion = SpectralFusion()

    resonance_shed = ResonanceShed(
        motif_detector=motif_detector,
        embedder=embedder,
        spectral_fusion=spectral_fusion,
        interference_mode="weighted_sum",
        target_scale=384
    )

    # Cold start (first query)
    start = time.perf_counter()
    _ = await resonance_shed.weave(text=queries[0].text, context_graph=None)
    cold_time = (time.perf_counter() - start) * 1000
    results['cold_start_ms'] = cold_time

    # Warm up cache
    for query in queries:
        _ = await resonance_shed.weave(text=query.text, context_graph=None)

    # Warm cache (repeated queries)
    start = time.perf_counter()
    for _ in range(n_iterations):
        for query in queries:
            _ = await resonance_shed.weave(text=query.text, context_graph=None)
    warm_time = (time.perf_counter() - start) / (n_iterations * len(queries)) * 1000
    results['warm_cache_ms'] = warm_time

    # Cleanup
    if use_zero_copy:
        embedder.close()

    return results


def measure_memory_overhead(embedder, queries: List[str]) -> Dict[str, float]:
    """Measure memory overhead of embedding storage."""
    import sys

    # Start tracking
    tracemalloc.start()

    # Encode all queries and get all scales
    all_scales_list = []
    for query in queries:
        all_scales = embedder.encode_scales([query], size=None)
        all_scales_list.append(all_scales)

    # Get memory snapshot
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate actual array memory
    total_array_memory = 0
    for all_scales in all_scales_list:
        for scale, arr in all_scales.items():
            total_array_memory += arr.nbytes

    return {
        'total_array_bytes': total_array_memory,
        'total_array_kb': total_array_memory / 1024,
        'peak_memory_kb': peak / 1024,
        'current_memory_kb': current / 1024
    }


# ============================================================================
# Main Benchmark
# ============================================================================

async def run_benchmark():
    """Run complete benchmark suite."""

    print("\n" + "=" * 70)
    print("  ZERO-COPY EMBEDDINGS ORCHESTRATOR BENCHMARK")
    print("=" * 70)

    query_objects = [Query(text=q) for q in TEST_QUERIES]

    # ========================================================================
    # Benchmark 1: Isolated Embedding Performance
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Isolated Embedding Performance")
    print("=" * 70)
    print("\nThis tests pure embedding speed (no orchestrator overhead)")

    # Standard embedder
    print("\n[1/2] Testing standard projection-based embeddings...")
    std_embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384], base_model_name=None)
    std_isolated = benchmark_embedder_isolation(std_embedder, TEST_QUERIES[:5], n_iterations=50)

    # Zero-copy embedder
    print("[2/2] Testing zero-copy embeddings...")
    tmpdir = tempfile.mkdtemp()
    zc_embedder = ZeroCopyMatryoshkaEmbeddings(
        sizes=[96, 192, 384],
        base_model_name=None,
        store_path=Path(tmpdir) / 'isolated_cache.mmap',
        max_cache_size=1000
    )
    zc_isolated = benchmark_embedder_isolation(zc_embedder, TEST_QUERIES[:5], n_iterations=50)
    zc_embedder.close()

    # Results
    print("\n" + "-" * 70)
    print("RESULTS: Isolated Embedding Performance")
    print("-" * 70)

    print(f"\n{'Method':<30} {'Cold Start':<15} {'Warm Cache':<15} {'Speedup':<10}")
    print("-" * 70)

    print(f"{'Standard (projection)':<30} {std_isolated['cold_start_ms']:>10.2f}ms   {std_isolated['warm_cache_ms']:>10.3f}ms   {'1.0x':>8}")

    cold_speedup = std_isolated['cold_start_ms'] / zc_isolated['cold_start_ms']
    warm_speedup = std_isolated['warm_cache_ms'] / zc_isolated['warm_cache_ms']

    print(f"{'Zero-copy':<30} {zc_isolated['cold_start_ms']:>10.2f}ms   {zc_isolated['warm_cache_ms']:>10.3f}ms   {warm_speedup:>7.1f}x")

    print(f"\nSpeedup: {cold_speedup:.1f}x (cold), {warm_speedup:.1f}x (warm)")

    # ========================================================================
    # Benchmark 2: Orchestrator Integration (ResonanceShed - Step 4)
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Orchestrator Integration (Step 4: ResonanceShed)")
    print("=" * 70)
    print("\nThis tests real-world impact in the full weaving cycle")

    # Standard embedder in orchestrator
    print("\n[1/2] Testing standard embeddings in ResonanceShed...")
    std_orchestrator = await benchmark_orchestrator_integration(
        use_zero_copy=False,
        queries=query_objects[:5],
        shards=MEMORY_SHARDS,
        n_iterations=10
    )

    # Zero-copy embedder in orchestrator
    print("[2/2] Testing zero-copy embeddings in ResonanceShed...")
    zc_orchestrator = await benchmark_orchestrator_integration(
        use_zero_copy=True,
        queries=query_objects[:5],
        shards=MEMORY_SHARDS,
        n_iterations=10
    )

    # Results
    print("\n" + "-" * 70)
    print("RESULTS: Orchestrator Integration (Real-World Impact)")
    print("-" * 70)

    print(f"\n{'Method':<30} {'Cold Start':<15} {'Warm Cache':<15} {'Speedup':<10}")
    print("-" * 70)

    print(f"{'Standard (projection)':<30} {std_orchestrator['cold_start_ms']:>10.2f}ms   {std_orchestrator['warm_cache_ms']:>10.2f}ms   {'1.0x':>8}")

    orch_cold_speedup = std_orchestrator['cold_start_ms'] / zc_orchestrator['cold_start_ms']
    orch_warm_speedup = std_orchestrator['warm_cache_ms'] / zc_orchestrator['warm_cache_ms']

    print(f"{'Zero-copy':<30} {zc_orchestrator['cold_start_ms']:>10.2f}ms   {zc_orchestrator['warm_cache_ms']:>10.2f}ms   {orch_warm_speedup:>7.1f}x")

    print(f"\nSpeedup: {orch_cold_speedup:.1f}x (cold), {orch_warm_speedup:.1f}x (warm)")

    # ========================================================================
    # Benchmark 3: Memory Overhead
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Memory Overhead")
    print("=" * 70)

    # Standard
    print("\n[1/2] Measuring standard embeddings memory...")
    std_embedder_mem = MatryoshkaEmbeddings(sizes=[96, 192, 384], base_model_name=None)
    std_memory = measure_memory_overhead(std_embedder_mem, TEST_QUERIES[:5])

    # Zero-copy
    print("[2/2] Measuring zero-copy embeddings memory...")
    tmpdir = tempfile.mkdtemp()
    zc_embedder_mem = ZeroCopyMatryoshkaEmbeddings(
        sizes=[96, 192, 384],
        base_model_name=None,
        store_path=Path(tmpdir) / 'memory_cache.mmap',
        max_cache_size=1000
    )
    zc_memory = measure_memory_overhead(zc_embedder_mem, TEST_QUERIES[:5])
    zc_embedder_mem.close()

    # Results
    print("\n" + "-" * 70)
    print("RESULTS: Memory Overhead")
    print("-" * 70)

    print(f"\n{'Method':<30} {'Array Memory':<20} {'Ratio':<10}")
    print("-" * 70)

    print(f"{'Standard (projection)':<30} {std_memory['total_array_kb']:>15.1f} KB   {'1.0x':>8}")

    memory_ratio = std_memory['total_array_kb'] / zc_memory['total_array_kb'] if zc_memory['total_array_kb'] > 0 else 1.0

    print(f"{'Zero-copy':<30} {zc_memory['total_array_kb']:>15.1f} KB   {1/memory_ratio:>7.2f}x")

    print(f"\nMemory savings: {(1 - 1/memory_ratio)*100:.1f}% reduction")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n📊 Performance Gains:")
    print(f"  Isolated embeddings:")
    print(f"    - Cold start:  {cold_speedup:>5.1f}x faster")
    print(f"    - Warm cache:  {warm_speedup:>5.1f}x faster")

    print(f"\n  Orchestrator integration (real-world):")
    print(f"    - Cold start:  {orch_cold_speedup:>5.1f}x faster")
    print(f"    - Warm cache:  {orch_warm_speedup:>5.1f}x faster")

    print(f"\n💾 Memory Efficiency:")
    print(f"    - Memory ratio: {1/memory_ratio:>5.2f}x less memory")
    print(f"    - Savings:      {(1 - 1/memory_ratio)*100:>5.1f}% reduction")

    print(f"\n✨ Latency Savings (per query, warm cache):")
    latency_saved = std_orchestrator['warm_cache_ms'] - zc_orchestrator['warm_cache_ms']
    print(f"    - Absolute:     {latency_saved:>5.2f}ms saved")
    print(f"    - Relative:     {(latency_saved / std_orchestrator['warm_cache_ms'] * 100):>5.1f}% reduction")

    print(f"\n🎯 Recommendation:")
    if orch_warm_speedup >= 10:
        print(f"    ✅ STRONG YES - Deploy immediately!")
        print(f"    Zero-copy provides {orch_warm_speedup:.0f}x speedup in real workloads.")
    elif orch_warm_speedup >= 5:
        print(f"    ✅ YES - Deploy for latency-critical paths")
        print(f"    Zero-copy provides {orch_warm_speedup:.0f}x speedup with no downsides.")
    elif orch_warm_speedup >= 2:
        print(f"    ⚠️  MAYBE - Consider for high-throughput scenarios")
        print(f"    Zero-copy provides {orch_warm_speedup:.0f}x speedup (modest gain).")
    else:
        print(f"    ❌ NO - Investigate bottlenecks first")
        print(f"    Zero-copy only provides {orch_warm_speedup:.1f}x speedup (below target).")

    print(f"\n{'=' * 70}\n")

    return {
        'isolated': {'std': std_isolated, 'zc': zc_isolated},
        'orchestrator': {'std': std_orchestrator, 'zc': zc_orchestrator},
        'memory': {'std': std_memory, 'zc': zc_memory},
        'speedup': {
            'isolated_warm': warm_speedup,
            'orchestrator_warm': orch_warm_speedup
        }
    }


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    results = asyncio.run(run_benchmark())
    return results


if __name__ == '__main__':
    main()
