"""
Test suite for hololoom_rust Python bindings.

Run with: pytest tests/test_bindings.py -v
"""

import pytest
import numpy as np

# Try to import the Rust module
try:
    from hololoom_rust import (
        cosine_similarity_batch,
        normalize_batch,
        kmeans,
        silhouette_score,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytest.skip("hololoom_rust not installed", allow_module_level=True)


class TestCosineSimilarityBatch:
    """Tests for cosine_similarity_batch function."""

    def test_basic_similarity(self):
        """Test basic cosine similarity computation."""
        # Create simple test vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],  # Unit vector along x
            [0.0, 1.0, 0.0],  # Unit vector along y
            [1.0, 1.0, 0.0],  # 45 degrees in xy plane
        ], dtype=np.float32)

        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarities = cosine_similarity_batch(vectors, centroid)

        assert similarities.shape == (3,)
        assert abs(similarities[0] - 1.0) < 1e-5  # Same direction
        assert abs(similarities[1] - 0.0) < 1e-5  # Orthogonal
        assert abs(similarities[2] - 0.7071) < 1e-3  # 45 degrees

    def test_random_vectors(self):
        """Test with random vectors."""
        np.random.seed(42)
        vectors = np.random.randn(1000, 384).astype(np.float32)
        centroid = np.random.randn(384).astype(np.float32)

        similarities = cosine_similarity_batch(vectors, centroid)

        assert similarities.shape == (1000,)
        assert similarities.dtype == np.float32
        # All similarities should be in [-1, 1]
        assert np.all(similarities >= -1.0 - 1e-5)
        assert np.all(similarities <= 1.0 + 1e-5)

    def test_matches_numpy(self):
        """Verify results match NumPy implementation."""
        np.random.seed(42)
        vectors = np.random.randn(100, 64).astype(np.float32)
        centroid = np.random.randn(64).astype(np.float32)

        # Rust implementation
        rust_result = cosine_similarity_batch(vectors, centroid)

        # NumPy reference
        numpy_result = np.dot(vectors, centroid) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid) + 1e-9
        )

        np.testing.assert_allclose(rust_result, numpy_result, rtol=1e-5)

    def test_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises error."""
        vectors = np.random.randn(10, 64).astype(np.float32)
        centroid = np.random.randn(128).astype(np.float32)  # Wrong dimension

        with pytest.raises(ValueError):
            cosine_similarity_batch(vectors, centroid)


class TestNormalizeBatch:
    """Tests for normalize_batch function."""

    def test_basic_normalization(self):
        """Test basic vector normalization."""
        vectors = np.array([
            [3.0, 4.0],  # Norm = 5
            [0.0, 5.0],  # Norm = 5
            [1.0, 0.0],  # Norm = 1
        ], dtype=np.float32)

        normalized = normalize_batch(vectors)

        assert normalized.shape == vectors.shape

        # Check norms are ~1.0
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_random_vectors(self):
        """Test normalization of random vectors."""
        np.random.seed(42)
        vectors = np.random.randn(1000, 384).astype(np.float32)

        normalized = normalize_batch(vectors)

        assert normalized.shape == vectors.shape

        # All norms should be ~1.0
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_zero_vector_handling(self):
        """Test that zero vectors don't cause NaN."""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)

        normalized = normalize_batch(vectors)

        # Should not produce NaN
        assert not np.any(np.isnan(normalized))


class TestKMeans:
    """Tests for kmeans function."""

    def test_basic_clustering(self):
        """Test basic k-means clustering."""
        np.random.seed(42)

        # Create 3 well-separated clusters
        cluster1 = np.random.randn(50, 10).astype(np.float32) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(50, 10).astype(np.float32) + [0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster3 = np.random.randn(50, 10).astype(np.float32) + [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]

        data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

        labels, centroids, inertia, n_iter = kmeans(data, k=3)

        assert labels.shape == (150,)
        assert centroids.shape == (3, 10)
        assert inertia > 0
        assert n_iter > 0

        # Check all labels are valid
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

        # Check we have all 3 clusters
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 3

    def test_single_cluster(self):
        """Test k=1 clustering."""
        np.random.seed(42)
        data = np.random.randn(100, 20).astype(np.float32)

        labels, centroids, inertia, n_iter = kmeans(data, k=1)

        assert np.all(labels == 0)
        assert centroids.shape == (1, 20)

    def test_optional_parameters(self):
        """Test optional parameters."""
        np.random.seed(42)
        data = np.random.randn(100, 10).astype(np.float32)

        labels, centroids, inertia, n_iter = kmeans(
            data,
            k=3,
            max_iterations=50,
            tolerance=1e-3,
            n_init=5,
            seed=123
        )

        assert labels.shape == (100,)
        assert centroids.shape == (3, 10)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        np.random.seed(42)
        data = np.random.randn(100, 20).astype(np.float32)

        labels1, _, _, _ = kmeans(data, k=3, seed=42)
        labels2, _, _, _ = kmeans(data, k=3, seed=42)

        np.testing.assert_array_equal(labels1, labels2)

    def test_invalid_k_raises(self):
        """Test that invalid k raises error."""
        data = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError):
            kmeans(data, k=0)

        with pytest.raises(ValueError):
            kmeans(data, k=100)  # More clusters than samples


class TestSilhouetteScore:
    """Tests for silhouette_score function."""

    def test_perfect_clustering(self):
        """Test silhouette score for well-separated clusters."""
        np.random.seed(42)

        # Create very well-separated clusters
        cluster1 = np.random.randn(50, 10).astype(np.float32) * 0.1 + [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(50, 10).astype(np.float32) * 0.1 + [0, 10, 0, 0, 0, 0, 0, 0, 0, 0]

        data = np.vstack([cluster1, cluster2]).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)

        score = silhouette_score(data, labels)

        # Well-separated clusters should have high score
        assert score > 0.8

    def test_random_labels(self):
        """Test silhouette score with random labels."""
        np.random.seed(42)
        data = np.random.randn(100, 20).astype(np.float32)
        labels = np.random.randint(0, 5, size=100).astype(np.int64)

        score = silhouette_score(data, labels)

        # Score should be in valid range
        assert -1.0 <= score <= 1.0

    def test_integration_with_kmeans(self):
        """Test silhouette score with kmeans output."""
        np.random.seed(42)
        data = np.random.randn(200, 30).astype(np.float32)

        labels, _, _, _ = kmeans(data, k=5)
        score = silhouette_score(data, labels)

        assert -1.0 <= score <= 1.0

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        data = np.random.randn(100, 20).astype(np.float32)
        labels = np.random.randint(0, 3, size=50).astype(np.int64)  # Wrong size

        with pytest.raises(ValueError):
            silhouette_score(data, labels)


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_clustering_pipeline(self):
        """Test complete clustering pipeline."""
        np.random.seed(42)

        # Generate data
        data = np.random.randn(500, 128).astype(np.float32)

        # Normalize
        normalized = normalize_batch(data)

        # Cluster
        labels, centroids, inertia, n_iter = kmeans(normalized, k=10)

        # Evaluate
        score = silhouette_score(normalized, labels)

        # Compute similarities to centroids
        for i in range(10):
            centroid = centroids[i]
            sims = cosine_similarity_batch(normalized, centroid)
            assert sims.shape == (500,)

        print(f"\nClustering results:")
        print(f"  Clusters: 10")
        print(f"  Samples: 500")
        print(f"  Dimensions: 128")
        print(f"  Iterations: {n_iter}")
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette: {score:.4f}")

    def test_large_scale(self):
        """Test with larger data."""
        np.random.seed(42)

        # 10K samples, 384D (typical embedding size)
        data = np.random.randn(10000, 384).astype(np.float32)

        # This should complete in reasonable time
        labels, centroids, inertia, n_iter = kmeans(data, k=20, n_init=3)

        assert labels.shape == (10000,)
        assert centroids.shape == (20, 384)

        score = silhouette_score(data, labels)
        assert -1.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
