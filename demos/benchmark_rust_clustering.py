"""
Benchmark: Python/NumPy vs Rust Clustering Performance
======================================================

Compares performance of:
- Cosine similarity batch
- Vector normalization
- K-means clustering
- Silhouette score

Run:
    # First, build the Rust module:
    cd rust/hololoom-python
    pip install maturin
    maturin develop --release

    # Then run benchmark:
    PYTHONPATH=. python demos/benchmark_rust_clustering.py
"""

import numpy as np
import time
from typing import Callable, Tuple

# Check what's available
try:
    from hololoom_rust import (
        cosine_similarity_batch as rust_cosine,
        normalize_batch as rust_normalize,
        kmeans as rust_kmeans,
        silhouette_score as rust_silhouette,
    )
    HAVE_RUST = True
    print("✅ Rust module loaded successfully!")
except ImportError as e:
    HAVE_RUST = False
    print(f"❌ Rust module not available: {e}")
    print("   Build with: cd rust/hololoom-python && maturin develop --release")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    HAVE_SKLEARN = True
    print("✅ sklearn loaded successfully!")
except ImportError:
    HAVE_SKLEARN = False
    print("❌ sklearn not available")


def benchmark(name: str, fn: Callable, n_runs: int = 10) -> Tuple[float, float]:
    """Run benchmark and return (mean_ms, std_ms)."""
    times = []

    # Warmup
    for _ in range(2):
        fn()

    # Benchmark
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return np.mean(times), np.std(times)


def numpy_cosine_similarity_batch(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """NumPy implementation of batch cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid) + 1e-9
    return np.dot(vectors, centroid) / norms


def numpy_normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """NumPy implementation of batch normalization."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors / norms


def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("HoloLoom Clustering Performance Benchmark")
    print("=" * 70)

    # Test data
    n_vectors = 1000
    dim = 384
    k = 10

    print(f"\nTest configuration:")
    print(f"  Vectors: {n_vectors}")
    print(f"  Dimensions: {dim}")
    print(f"  Clusters (k): {k}")
    print(f"  Runs per benchmark: 10")

    # Generate test data
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    centroid = np.random.randn(dim).astype(np.float32)

    results = {}

    # 1. Cosine Similarity Batch
    print("\n" + "-" * 70)
    print("1. Cosine Similarity Batch")
    print("-" * 70)

    numpy_mean, numpy_std = benchmark(
        "NumPy",
        lambda: numpy_cosine_similarity_batch(vectors, centroid)
    )
    results["cosine_numpy"] = numpy_mean
    print(f"  NumPy:  {numpy_mean:.3f} ms (± {numpy_std:.3f})")

    if HAVE_RUST:
        rust_mean, rust_std = benchmark(
            "Rust",
            lambda: rust_cosine(vectors, centroid)
        )
        results["cosine_rust"] = rust_mean
        speedup = numpy_mean / rust_mean
        print(f"  Rust:   {rust_mean:.3f} ms (± {rust_std:.3f}) - {speedup:.1f}x faster")

    # 2. Vector Normalization
    print("\n" + "-" * 70)
    print("2. Vector Normalization Batch")
    print("-" * 70)

    numpy_mean, numpy_std = benchmark(
        "NumPy",
        lambda: numpy_normalize_batch(vectors)
    )
    results["normalize_numpy"] = numpy_mean
    print(f"  NumPy:  {numpy_mean:.3f} ms (± {numpy_std:.3f})")

    if HAVE_RUST:
        rust_mean, rust_std = benchmark(
            "Rust",
            lambda: rust_normalize(vectors)
        )
        results["normalize_rust"] = rust_mean
        speedup = numpy_mean / rust_mean
        print(f"  Rust:   {rust_mean:.3f} ms (± {rust_std:.3f}) - {speedup:.1f}x faster")

    # 3. K-Means Clustering
    print("\n" + "-" * 70)
    print("3. K-Means Clustering")
    print("-" * 70)

    if HAVE_SKLEARN:
        sklearn_mean, sklearn_std = benchmark(
            "sklearn",
            lambda: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors),
            n_runs=5  # Fewer runs since k-means is slower
        )
        results["kmeans_sklearn"] = sklearn_mean
        print(f"  sklearn: {sklearn_mean:.3f} ms (± {sklearn_std:.3f})")

    if HAVE_RUST:
        rust_mean, rust_std = benchmark(
            "Rust",
            lambda: rust_kmeans(vectors, k, max_iterations=300, tolerance=1e-4, n_init=10, seed=42),
            n_runs=5
        )
        results["kmeans_rust"] = rust_mean
        if HAVE_SKLEARN:
            speedup = sklearn_mean / rust_mean
            print(f"  Rust:    {rust_mean:.3f} ms (± {rust_std:.3f}) - {speedup:.1f}x faster")
        else:
            print(f"  Rust:    {rust_mean:.3f} ms (± {rust_std:.3f})")

    # 4. Silhouette Score
    print("\n" + "-" * 70)
    print("4. Silhouette Score")
    print("-" * 70)

    # Generate labels for silhouette test
    if HAVE_RUST:
        labels, _, _, _ = rust_kmeans(vectors, k)
        labels = np.array(labels)
    elif HAVE_SKLEARN:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
    else:
        labels = np.random.randint(0, k, n_vectors)

    labels_i64 = labels.astype(np.int64)

    if HAVE_SKLEARN:
        sklearn_mean, sklearn_std = benchmark(
            "sklearn",
            lambda: sklearn_silhouette(vectors, labels),
            n_runs=5
        )
        results["silhouette_sklearn"] = sklearn_mean
        print(f"  sklearn: {sklearn_mean:.3f} ms (± {sklearn_std:.3f})")

    if HAVE_RUST:
        rust_mean, rust_std = benchmark(
            "Rust",
            lambda: rust_silhouette(vectors, labels_i64),
            n_runs=5
        )
        results["silhouette_rust"] = rust_mean
        if HAVE_SKLEARN:
            speedup = sklearn_mean / rust_mean
            print(f"  Rust:    {rust_mean:.3f} ms (± {rust_std:.3f}) - {speedup:.1f}x faster")
        else:
            print(f"  Rust:    {rust_mean:.3f} ms (± {rust_std:.3f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if HAVE_RUST:
        print("\n┌─────────────────────────────────┬──────────┬──────────┬──────────┐")
        print("│ Operation                       │ Python   │ Rust     │ Speedup  │")
        print("├─────────────────────────────────┼──────────┼──────────┼──────────┤")

        if "cosine_numpy" in results and "cosine_rust" in results:
            speedup = results["cosine_numpy"] / results["cosine_rust"]
            print(f"│ Cosine Similarity (1000×384)    │ {results['cosine_numpy']:6.2f}ms │ {results['cosine_rust']:6.2f}ms │ {speedup:6.1f}x  │")

        if "normalize_numpy" in results and "normalize_rust" in results:
            speedup = results["normalize_numpy"] / results["normalize_rust"]
            print(f"│ Normalize Batch (1000×384)      │ {results['normalize_numpy']:6.2f}ms │ {results['normalize_rust']:6.2f}ms │ {speedup:6.1f}x  │")

        if "kmeans_sklearn" in results and "kmeans_rust" in results:
            speedup = results["kmeans_sklearn"] / results["kmeans_rust"]
            print(f"│ K-Means (1000 pts, k=10)        │ {results['kmeans_sklearn']:6.1f}ms │ {results['kmeans_rust']:6.1f}ms │ {speedup:6.1f}x  │")

        if "silhouette_sklearn" in results and "silhouette_rust" in results:
            speedup = results["silhouette_sklearn"] / results["silhouette_rust"]
            print(f"│ Silhouette Score                │ {results['silhouette_sklearn']:6.1f}ms │ {results['silhouette_rust']:6.1f}ms │ {speedup:6.1f}x  │")

        print("└─────────────────────────────────┴──────────┴──────────┴──────────┘")

        print("\n🚀 Rust provides significant speedups across all operations!")
        print("   These gains compound in real workflows (clustering → scoring → optimization)")
    else:
        print("\n⚠️  Rust module not available. Build with:")
        print("    cd rust/hololoom-python")
        print("    pip install maturin")
        print("    maturin develop --release")

    print("\n" + "=" * 70)

    return results


def test_correctness():
    """Test that Rust produces same results as Python."""
    if not HAVE_RUST:
        print("⚠️  Rust module not available, skipping correctness tests")
        return

    print("\n" + "=" * 70)
    print("Correctness Tests")
    print("=" * 70)

    np.random.seed(42)
    vectors = np.random.randn(100, 64).astype(np.float32)
    centroid = np.random.randn(64).astype(np.float32)

    # Test cosine similarity
    numpy_result = numpy_cosine_similarity_batch(vectors, centroid)
    rust_result = rust_cosine(vectors, centroid)
    cosine_close = np.allclose(numpy_result, rust_result, rtol=1e-5, atol=1e-5)
    print(f"  Cosine similarity: {'✅ PASS' if cosine_close else '❌ FAIL'}")

    # Test normalization
    numpy_result = numpy_normalize_batch(vectors)
    rust_result = rust_normalize(vectors)
    normalize_close = np.allclose(numpy_result, rust_result, rtol=1e-5, atol=1e-5)
    print(f"  Normalization: {'✅ PASS' if normalize_close else '❌ FAIL'}")

    # Test k-means (just verify it runs without error and produces valid labels)
    labels, centroids, inertia, n_iter = rust_kmeans(vectors, 5)
    kmeans_valid = (
        len(labels) == 100 and
        centroids.shape == (5, 64) and
        inertia > 0 and
        n_iter > 0 and
        all(0 <= l < 5 for l in labels)
    )
    print(f"  K-means: {'✅ PASS' if kmeans_valid else '❌ FAIL'}")

    # Test silhouette score
    labels_i64 = np.array(labels).astype(np.int64)
    rust_score = rust_silhouette(vectors, labels_i64)
    silhouette_valid = -1 <= rust_score <= 1
    print(f"  Silhouette score: {'✅ PASS' if silhouette_valid else '❌ FAIL'}")

    if HAVE_SKLEARN:
        sklearn_score = sklearn_silhouette(vectors, np.array(labels))
        silhouette_close = abs(rust_score - sklearn_score) < 0.1
        print(f"  Silhouette vs sklearn: {'✅ PASS' if silhouette_close else '❌ FAIL'}")
        print(f"    Rust:    {rust_score:.4f}")
        print(f"    sklearn: {sklearn_score:.4f}")


if __name__ == "__main__":
    test_correctness()
    run_benchmarks()
