"""
Benchmark: Rust vs NumPy/sklearn performance comparison.

Run with: python tests/benchmark_rust_vs_numpy.py

Expected speedups:
- cosine_similarity_batch: 30-50x
- normalize_batch: 15-25x
- kmeans: 15-25x
- silhouette_score: 10-20x
"""

import time
import numpy as np
from typing import Callable, Tuple
import sys

# Check if Rust module is available
try:
    from hololoom_rust import (
        cosine_similarity_batch as rust_cosine_similarity,
        normalize_batch as rust_normalize,
        kmeans as rust_kmeans,
        silhouette_score as rust_silhouette,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  hololoom_rust not installed. Install with:")
    print("   cd rust/hololoom-python && maturin develop --release")
    print()

# Check for sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not available for comparison")
    print()


def benchmark(func: Callable, *args, warmup: int = 3, runs: int = 10, **kwargs) -> Tuple[float, float]:
    """
    Benchmark a function with warmup and multiple runs.

    Returns:
        (mean_ms, std_ms): Mean and std of execution time in milliseconds
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return np.mean(times), np.std(times)


def numpy_cosine_similarity_batch(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """NumPy implementation of batch cosine similarity."""
    return np.dot(vectors, centroid) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid) + 1e-9
    )


def numpy_normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """NumPy implementation of batch normalization."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-9)


def print_comparison(name: str, numpy_time: float, numpy_std: float,
                     rust_time: float, rust_std: float):
    """Print formatted comparison."""
    speedup = numpy_time / rust_time if rust_time > 0 else float('inf')

    print(f"\n{name}:")
    print(f"  NumPy:  {numpy_time:8.3f}ms ± {numpy_std:.3f}ms")
    print(f"  Rust:   {rust_time:8.3f}ms ± {rust_std:.3f}ms")
    print(f"  Speedup: {speedup:6.1f}x {'✅' if speedup >= 5 else '⚠️' if speedup >= 2 else '❌'}")


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("HoloLoom Rust Clustering Benchmarks")
    print("=" * 60)

    np.random.seed(42)

    # Test sizes
    sizes = [
        (1000, 384, "Small (1K × 384D)"),
        (10000, 384, "Medium (10K × 384D)"),
        (50000, 384, "Large (50K × 384D)"),
    ]

    # =====================================================
    # Cosine Similarity Batch
    # =====================================================
    print("\n" + "─" * 60)
    print("COSINE SIMILARITY BATCH")
    print("─" * 60)

    for n_samples, dim, label in sizes:
        vectors = np.random.randn(n_samples, dim).astype(np.float32)
        centroid = np.random.randn(dim).astype(np.float32)

        numpy_time, numpy_std = benchmark(numpy_cosine_similarity_batch, vectors, centroid)

        if RUST_AVAILABLE:
            rust_time, rust_std = benchmark(rust_cosine_similarity, vectors, centroid)
            print_comparison(label, numpy_time, numpy_std, rust_time, rust_std)
        else:
            print(f"\n{label}:")
            print(f"  NumPy: {numpy_time:.3f}ms ± {numpy_std:.3f}ms")
            print(f"  Rust:  (not available)")

    # =====================================================
    # Normalize Batch
    # =====================================================
    print("\n" + "─" * 60)
    print("NORMALIZE BATCH")
    print("─" * 60)

    for n_samples, dim, label in sizes:
        vectors = np.random.randn(n_samples, dim).astype(np.float32)

        numpy_time, numpy_std = benchmark(numpy_normalize_batch, vectors)

        if RUST_AVAILABLE:
            rust_time, rust_std = benchmark(rust_normalize, vectors)
            print_comparison(label, numpy_time, numpy_std, rust_time, rust_std)
        else:
            print(f"\n{label}:")
            print(f"  NumPy: {numpy_time:.3f}ms ± {numpy_std:.3f}ms")
            print(f"  Rust:  (not available)")

    # =====================================================
    # K-Means Clustering
    # =====================================================
    print("\n" + "─" * 60)
    print("K-MEANS CLUSTERING")
    print("─" * 60)

    kmeans_sizes = [
        (1000, 128, 10, "Small (1K × 128D, k=10)"),
        (5000, 256, 20, "Medium (5K × 256D, k=20)"),
        (10000, 384, 30, "Large (10K × 384D, k=30)"),
    ]

    for n_samples, dim, k, label in kmeans_sizes:
        data = np.random.randn(n_samples, dim).astype(np.float32)

        # sklearn benchmark
        if SKLEARN_AVAILABLE:
            def sklearn_kmeans():
                model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
                model.fit(data)
                return model.labels_

            sklearn_time, sklearn_std = benchmark(sklearn_kmeans, warmup=1, runs=3)
        else:
            sklearn_time, sklearn_std = 0, 0

        # Rust benchmark
        if RUST_AVAILABLE:
            def rust_kmeans_wrapper():
                return rust_kmeans(data, k=k, n_init=10, max_iterations=300, seed=42)

            rust_time, rust_std = benchmark(rust_kmeans_wrapper, warmup=1, runs=3)

            if SKLEARN_AVAILABLE:
                print_comparison(label, sklearn_time, sklearn_std, rust_time, rust_std)
            else:
                print(f"\n{label}:")
                print(f"  sklearn: (not available)")
                print(f"  Rust:    {rust_time:.3f}ms ± {rust_std:.3f}ms")
        else:
            print(f"\n{label}:")
            if SKLEARN_AVAILABLE:
                print(f"  sklearn: {sklearn_time:.3f}ms ± {sklearn_std:.3f}ms")
            print(f"  Rust:    (not available)")

    # =====================================================
    # Silhouette Score
    # =====================================================
    print("\n" + "─" * 60)
    print("SILHOUETTE SCORE")
    print("─" * 60)

    silhouette_sizes = [
        (500, 64, 5, "Small (500 × 64D, k=5)"),
        (2000, 128, 10, "Medium (2K × 128D, k=10)"),
        (5000, 256, 15, "Large (5K × 256D, k=15)"),
    ]

    for n_samples, dim, k, label in silhouette_sizes:
        data = np.random.randn(n_samples, dim).astype(np.float32)

        # Generate labels using numpy k-means approximation
        # (just random assignment for benchmark purposes)
        labels = np.random.randint(0, k, n_samples).astype(np.int64)

        # sklearn benchmark
        if SKLEARN_AVAILABLE:
            sklearn_time, sklearn_std = benchmark(
                sklearn_silhouette, data, labels, warmup=1, runs=3
            )
        else:
            sklearn_time, sklearn_std = 0, 0

        # Rust benchmark
        if RUST_AVAILABLE:
            rust_time, rust_std = benchmark(
                rust_silhouette, data, labels, warmup=1, runs=3
            )

            if SKLEARN_AVAILABLE:
                print_comparison(label, sklearn_time, sklearn_std, rust_time, rust_std)
            else:
                print(f"\n{label}:")
                print(f"  sklearn: (not available)")
                print(f"  Rust:    {rust_time:.3f}ms ± {rust_std:.3f}ms")
        else:
            print(f"\n{label}:")
            if SKLEARN_AVAILABLE:
                print(f"  sklearn: {sklearn_time:.3f}ms ± {sklearn_std:.3f}ms")
            print(f"  Rust:    (not available)")

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if RUST_AVAILABLE:
        print("\n✅ Rust module available and benchmarked")
        print("\nExpected speedups (vs NumPy/sklearn):")
        print("  • cosine_similarity_batch: 30-50x")
        print("  • normalize_batch: 15-25x")
        print("  • kmeans: 15-25x")
        print("  • silhouette_score: 10-20x")
        print("\nNote: Actual speedups depend on:")
        print("  • CPU with AVX2 support")
        print("  • Number of CPU cores (Rayon parallelization)")
        print("  • Data size and dimensions")
    else:
        print("\n❌ Rust module not available")
        print("\nTo install:")
        print("  cd rust/hololoom-python")
        print("  pip install maturin")
        print("  maturin develop --release")


if __name__ == "__main__":
    run_benchmarks()
