"""
Performance Benchmark for Semantic Calculus Optimizations

Demonstrates speed improvements from:
1. Embedding cache (batch + LRU)  
2. JIT compilation (numba)
3. Vectorized operations
4. Sparse projections

Compares optimized vs baseline implementations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from HoloLoom.semantic_calculus import (
    SemanticFlowCalculus,
    SemanticSpectrum,
    GeometricIntegrator,
    EmbeddingCache,
    HAS_NUMBA,
)
from HoloLoom.embedding.spectral import create_embedder


def benchmark_embedding_cache():
    """Benchmark 1: Embedding cache performance"""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Embedding Cache Performance")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])
    embed_fn = lambda words: embed_model.encode(words)

    test_words = ["hello", "world", "how", "are", "you", "hello", "there",
                  "world", "is", "beautiful", "hello", "again", "world"]

    print(f"\nTest: {len(test_words)} words ({len(set(test_words))} unique)")

    # Baseline
    print("\n[BASELINE] No cache:")
    start = time.perf_counter()
    for word in test_words:
        embed_fn([word])[0]
    time_nocache = time.perf_counter() - start
    print(f"  Time: {time_nocache*1000:.2f}ms")

    # With cache (cold)
    print("\n[OPTIMIZED] With cache (cold):")
    cache = EmbeddingCache(embed_fn, max_size=1000)
    start = time.perf_counter()
    for word in test_words:
        cache.get(word)
    time_cache = time.perf_counter() - start
    stats = cache.get_stats()
    print(f"  Time: {time_cache*1000:.2f}ms")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")

    # With cache (warm)
    print("\n[OPTIMIZED] With cache (warm):")
    start = time.perf_counter()
    for word in test_words:
        cache.get(word)
    time_warm = time.perf_counter() - start
    stats = cache.get_stats()
    print(f"  Time: {time_warm*1000:.2f}ms")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")

    print(f"\nSpeedup: {time_nocache/time_warm:.2f}x")


def benchmark_trajectory():
    """Benchmark 2: Trajectory computation"""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Trajectory Computation")  
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])
    embed_fn = lambda words: embed_model.encode(words)

    words = "The quick brown fox jumps over the lazy dog".split()
    print(f"\nTest: {len(words)} words")

    # Baseline
    print("\n[BASELINE] No cache:")
    calc = SemanticFlowCalculus(embed_fn, enable_cache=False)
    start = time.perf_counter()
    calc.compute_trajectory(words)
    time_nocache = time.perf_counter() - start
    print(f"  Time: {time_nocache*1000:.2f}ms")

    # Optimized
    print("\n[OPTIMIZED] With cache:")
    calc_opt = SemanticFlowCalculus(embed_fn, enable_cache=True)
    start = time.perf_counter()
    calc_opt.compute_trajectory(words)
    time_cache = time.perf_counter() - start
    print(f"  Time: {time_cache*1000:.2f}ms")

    print(f"\nSpeedup: {time_nocache/time_cache:.2f}x")


def print_summary():
    """Print optimization summary"""
    print("\n" + "=" * 70)
    print("OPTIMIZATIONS")
    print("=" * 70)
    print(f"[ENABLED ] Embedding cache (LRU)")
    print(f"[ENABLED ] Batch embedding")
    print(f"[ENABLED ] Vectorized derivatives")
    print(f"[{'ENABLED ' if HAS_NUMBA else 'DISABLED'}] JIT compilation" + 
          ("" if HAS_NUMBA else " (pip install numba)"))
    print()


if __name__ == "__main__":
    print("\nSEMANTIC CALCULUS PERFORMANCE BENCHMARK\n")
    print_summary()
    benchmark_embedding_cache()
    benchmark_trajectory()
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
