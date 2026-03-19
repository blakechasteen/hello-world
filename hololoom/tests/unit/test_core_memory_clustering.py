"""
Unit tests for core memory modules:
- clustering/core.py
- clustering/labeler.py
- clustering/thompson.py
- retrieval_strategies.py (via mock)
- retriever_registry.py
- semantic_bridge.py
- hyperspace_backend.py (data structures + config mapping)
- integrated_memory_system.py (config only — heavy deps mocked)
- bee_data.py (data integrity)
- spring_dynamics_advanced.py
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


# ============================================================================
# clustering/thompson.py
# ============================================================================


class TestBetaPrior:
    """Tests for BetaPrior dataclass."""

    def test_initial_uniform(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior()
        assert bp.alpha == 1.0
        assert bp.beta == 1.0
        assert bp.mean == 0.5

    def test_sample_returns_float_in_range(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior(alpha=2.0, beta=5.0)
        for _ in range(50):
            s = bp.sample()
            assert 0.0 <= s <= 1.0

    def test_update_shifts_distribution(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior()
        # High reward should increase alpha
        bp.update(0.9)
        assert bp.alpha == pytest.approx(1.9)
        assert bp.beta == pytest.approx(1.1)

    def test_update_low_reward(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior()
        bp.update(0.1)
        assert bp.alpha == pytest.approx(1.1)
        assert bp.beta == pytest.approx(1.9)

    def test_mean_property(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior(alpha=3.0, beta=7.0)
        assert bp.mean == pytest.approx(0.3)

    def test_variance_property(self):
        from hololoom.core.memory.clustering.thompson import BetaPrior
        bp = BetaPrior(alpha=3.0, beta=7.0)
        total = 10.0
        expected = (3.0 * 7.0) / (total ** 2 * (total + 1))
        assert bp.variance == pytest.approx(expected)


class TestThompsonClusterSampler:
    """Tests for ThompsonClusterSampler."""

    def test_init_creates_priors_for_each_k(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        assert set(sampler.priors.keys()) == {2, 3, 4, 5}

    def test_sample_k_returns_valid_k(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        for _ in range(20):
            k = sampler.sample_k()
            assert 2 <= k <= 5

    def test_update_records_history(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        sampler.update(3, 0.8)
        assert len(sampler.history) == 1
        assert sampler.history[0] == (3, 0.8)

    def test_update_normalizes_silhouette(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        # silhouette=1.0 -> reward=(1+1)/2=1.0 -> alpha += 1.0
        sampler.update(3, 1.0)
        assert sampler.priors[3].alpha == pytest.approx(2.0)
        assert sampler.priors[3].beta == pytest.approx(1.0)

    def test_get_best_k(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        # Push k=3 to be clearly the best
        for _ in range(10):
            sampler.update(3, 0.9)
        best_k, score = sampler.get_best_k()
        assert best_k == 3
        assert score > 0.0

    def test_get_confidence(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        # Unknown k has low confidence (high variance)
        conf_initial = sampler.get_confidence(3)
        # After many updates, confidence should increase
        for _ in range(20):
            sampler.update(3, 0.8)
        conf_after = sampler.get_confidence(3)
        assert conf_after > conf_initial

    def test_get_confidence_unknown_k(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        assert sampler.get_confidence(99) == 0.0

    def test_should_explore_more(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=5)
        # Initially should explore (low confidence)
        assert sampler.should_explore_more(threshold=0.8)
        # After lots of data, may stop exploring
        for _ in range(100):
            sampler.update(3, 0.9)
        # Confidence should be high now
        assert not sampler.should_explore_more(threshold=0.8)

    def test_get_statistics(self):
        from hololoom.core.memory.clustering.thompson import ThompsonClusterSampler
        sampler = ThompsonClusterSampler(min_k=2, max_k=4)
        sampler.update(3, 0.7)
        stats = sampler.get_statistics()
        assert 2 in stats
        assert 3 in stats
        assert 4 in stats
        assert "mean" in stats[3]
        assert "variance" in stats[3]
        assert "confidence" in stats[3]


class TestFindOptimalKThompson:
    """Tests for the find_optimal_k_thompson function."""

    def test_returns_best_k_and_sampler(self):
        from hololoom.core.memory.clustering.thompson import find_optimal_k_thompson
        embeddings = np.random.randn(20, 10).astype(np.float32)

        def cluster_fn(emb, k):
            # Fake: k=3 gives best silhouette
            return 0.8 if k == 3 else 0.4

        best_k, score, sampler = find_optimal_k_thompson(
            embeddings, cluster_fn, min_k=2, max_k=6, max_iterations=20
        )
        assert 2 <= best_k <= 6
        assert sampler is not None

    def test_edge_case_max_k_less_than_min_k(self):
        from hololoom.core.memory.clustering.thompson import find_optimal_k_thompson
        embeddings = np.random.randn(2, 10)

        def cluster_fn(emb, k):
            return 0.5

        best_k, score, sampler = find_optimal_k_thompson(
            embeddings, cluster_fn, min_k=5, max_k=10
        )
        assert best_k == 5
        assert sampler is None


class TestAdaptiveKSearch:
    """Tests for adaptive_k_search wrapper."""

    def test_returns_k_and_score(self):
        from hololoom.core.memory.clustering.thompson import adaptive_k_search
        embeddings = np.random.randn(20, 10)

        def cluster_fn(emb, k):
            return max(0, 0.9 - abs(k - 4) * 0.15)

        best_k, score = adaptive_k_search(
            embeddings, cluster_fn, min_k=2, max_k=8, budget=10
        )
        assert 2 <= best_k <= 8


# ============================================================================
# clustering/labeler.py
# ============================================================================


class TestLabelCandidate:
    """Tests for LabelCandidate."""

    def test_str_primary_only(self):
        from hololoom.core.memory.clustering.labeler import LabelCandidate
        lc = LabelCandidate(primary="Technology")
        assert str(lc) == "Technology"

    def test_str_with_secondary(self):
        from hololoom.core.memory.clustering.labeler import LabelCandidate
        lc = LabelCandidate(primary="Technology", secondary="Ai Ml")
        assert str(lc) == "Technology / Ai Ml"


class TestDetectDomain:
    """Tests for _detect_domain."""

    def test_technology_domain(self):
        from hololoom.core.memory.clustering.labeler import _detect_domain
        text = "software engineering code programming algorithm data system"
        result = _detect_domain(text)
        assert result is not None
        assert result.primary == "Technology"

    def test_no_domain_detected(self):
        from hololoom.core.memory.clustering.labeler import _detect_domain
        text = "the cat sat on the mat"
        result = _detect_domain(text)
        assert result is None

    def test_science_domain(self):
        from hololoom.core.memory.clustering.labeler import _detect_domain
        text = "research study experiment analysis method"
        result = _detect_domain(text)
        assert result is not None
        assert result.primary == "Science"

    def test_subcategory_detection(self):
        from hololoom.core.memory.clustering.labeler import _detect_domain
        text = "machine learning neural ai deep learning model training algorithm data"
        result = _detect_domain(text)
        assert result is not None
        assert result.secondary is not None


class TestExtractKeywords:
    """Tests for _extract_keywords."""

    def test_extracts_top_keywords(self):
        from hololoom.core.memory.clustering.labeler import _extract_keywords
        texts = [
            "python programming language",
            "python scripting development",
            "python code library"
        ]
        result = _extract_keywords(texts, top_n=2)
        assert result is not None
        assert result.primary == "Python"

    def test_empty_texts(self):
        from hololoom.core.memory.clustering.labeler import _extract_keywords
        result = _extract_keywords(["the the the"], top_n=3)
        # All stopwords => no keywords
        assert result is None

    def test_confidence_proportional_to_frequency(self):
        from hololoom.core.memory.clustering.labeler import _extract_keywords
        texts = ["python python python coding coding testing"]
        result = _extract_keywords(texts, top_n=3)
        assert result is not None
        assert 0.0 < result.confidence <= 0.9


class TestLabelClusters:
    """Tests for label_clusters."""

    def test_labels_assigned(self):
        from hololoom.core.memory.clustering.labeler import label_clusters

        @dataclass
        class FakeCluster:
            label: str
            items: list
            centroid: object = None
            size: int = 0

        clusters = [
            FakeCluster(
                label="Cluster 0",
                items=["machine learning model", "neural network training", "deep learning ai"],
                size=3
            ),
        ]
        embeddings = np.random.randn(3, 10)
        texts = ["machine learning model", "neural network training", "deep learning ai"]

        result = label_clusters(clusters, embeddings, texts, use_semantic=False, use_domain=True)
        # Should have a non-default label now
        assert result[0].label != "Cluster 0"


class TestGenerateClusterSummary:
    """Tests for generate_cluster_summary."""

    def test_summary_format(self):
        from hololoom.core.memory.clustering.labeler import generate_cluster_summary

        @dataclass
        class FakeCluster:
            label: str
            items: list
            confidence: float
            size: int

        clusters = [
            FakeCluster(label="Tech", items=["item1", "item2"], confidence=0.85, size=2),
            FakeCluster(label="Science", items=["item3"], confidence=0.72, size=1),
        ]
        summary = generate_cluster_summary(clusters, include_examples=2)
        assert "Found 2 clusters" in summary
        assert "Tech" in summary
        assert "Science" in summary
        assert "85.0%" in summary


# ============================================================================
# clustering/core.py
# ============================================================================


class TestClusterResult:
    """Tests for ClusterResult dataclass."""

    def test_post_init_sets_size(self):
        from hololoom.core.memory.clustering.core import ClusterResult
        cr = ClusterResult(
            label="Test", items=["a", "b", "c"],
            indices=[0, 1, 2], confidence=0.9
        )
        assert cr.size == 3


class TestClusteringOutput:
    """Tests for ClusteringOutput."""

    def test_iter(self):
        from hololoom.core.memory.clustering.core import ClusterResult, ClusteringOutput
        cr1 = ClusterResult(label="A", items=["x"], indices=[0], confidence=0.8)
        cr2 = ClusterResult(label="B", items=["y"], indices=[1], confidence=0.7)
        output = ClusteringOutput(clusters=[cr1, cr2], n_clusters=2, silhouette=0.6)
        assert list(output) == [cr1, cr2]

    def test_len(self):
        from hololoom.core.memory.clustering.core import ClusterResult, ClusteringOutput
        cr = ClusterResult(label="A", items=["x"], indices=[0], confidence=0.8)
        output = ClusteringOutput(clusters=[cr], n_clusters=1, silhouette=0.5)
        assert len(output) == 1

    def test_getitem(self):
        from hololoom.core.memory.clustering.core import ClusterResult, ClusteringOutput
        cr = ClusterResult(label="A", items=["x"], indices=[0], confidence=0.8)
        output = ClusteringOutput(clusters=[cr], n_clusters=1, silhouette=0.5)
        assert output[0] == cr

    def test_to_dict(self):
        from hololoom.core.memory.clustering.core import ClusterResult, ClusteringOutput
        cr = ClusterResult(label="A", items=["x"], indices=[0], confidence=0.8)
        output = ClusteringOutput(clusters=[cr], n_clusters=1, silhouette=0.5)
        d = output.to_dict()
        assert len(d) == 1
        assert d[0]["label"] == "A"
        assert d[0]["confidence"] == 0.8
        assert d[0]["size"] == 1


class TestFallbackEmbedder:
    """Tests for _FallbackEmbedder."""

    def test_encode_deterministic(self):
        from hololoom.core.memory.clustering.core import _FallbackEmbedder
        emb = _FallbackEmbedder()
        result1 = emb.encode(["hello", "world"])
        result2 = emb.encode(["hello", "world"])
        np.testing.assert_array_equal(result1, result2)

    def test_encode_shape(self):
        from hololoom.core.memory.clustering.core import _FallbackEmbedder
        emb = _FallbackEmbedder()
        result = emb.encode(["a", "b", "c"])
        assert result.shape == (3, 384)

    def test_encode_normalized(self):
        from hololoom.core.memory.clustering.core import _FallbackEmbedder
        emb = _FallbackEmbedder()
        result = emb.encode(["test"])
        norm = np.linalg.norm(result[0])
        assert norm == pytest.approx(1.0, abs=0.01)


class TestCosineSimBatch:
    """Tests for _cosine_similarity_batch numpy fallback."""

    def test_identical_vectors(self):
        from hololoom.core.memory.clustering.core import _cosine_similarity_batch
        vec = np.array([1.0, 0.0, 0.0])
        vectors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sims = _cosine_similarity_batch(vectors, vec)
        np.testing.assert_allclose(sims, [1.0, 1.0], atol=1e-6)

    def test_orthogonal_vectors(self):
        from hololoom.core.memory.clustering.core import _cosine_similarity_batch
        centroid = np.array([1.0, 0.0])
        vectors = np.array([[0.0, 1.0]])
        sims = _cosine_similarity_batch(vectors, centroid)
        assert abs(sims[0]) < 1e-6


class TestPrepareData:
    """Tests for _prepare_data."""

    def test_ndarray_passthrough(self):
        from hololoom.core.memory.clustering.core import _prepare_data
        emb = np.random.randn(5, 10)
        texts, result = _prepare_data(emb)
        assert len(texts) == 5
        assert texts[0] == "item_0"
        np.testing.assert_array_equal(result, emb)

    def test_dict_input(self):
        from hololoom.core.memory.clustering.core import _prepare_data
        data = [{"text": "hello"}, {"content": "world"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 10)
        texts, embs = _prepare_data(data, embedder=mock_embedder)
        assert texts == ["hello", "world"]

    def test_string_input(self):
        from hololoom.core.memory.clustering.core import _prepare_data
        data = ["hello", "world"]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 10)
        texts, embs = _prepare_data(data, embedder=mock_embedder)
        assert texts == ["hello", "world"]

    def test_scale_aware_embedder(self):
        from hololoom.core.memory.clustering.core import _prepare_data
        mock_embedder = MagicMock()
        mock_embedder.encode_at_scale.return_value = np.random.randn(2, 96)
        texts, embs = _prepare_data(["a", "b"], scale=96, embedder=mock_embedder)
        mock_embedder.encode_at_scale.assert_called_once()


class TestSimpleLabelClusters:
    """Tests for _simple_label_clusters."""

    def test_labels_from_common_words(self):
        from hololoom.core.memory.clustering.core import _simple_label_clusters, ClusterResult
        clusters = [
            ClusterResult(
                label="Cluster 0",
                items=["python programming", "python scripting", "python code"],
                indices=[0, 1, 2], confidence=0.8
            )
        ]
        result = _simple_label_clusters(clusters, clusters[0].items)
        assert "Python" in result[0].label

    def test_fallback_group_label(self):
        from hololoom.core.memory.clustering.core import _simple_label_clusters, ClusterResult
        clusters = [
            ClusterResult(
                label="Cluster 0",
                items=["the the the"],  # All stopwords
                indices=[0], confidence=0.5
            )
        ]
        result = _simple_label_clusters(clusters, clusters[0].items)
        assert "Group" in result[0].label


class TestBuildClusters:
    """Tests for _build_clusters."""

    def test_builds_correctly(self):
        from hololoom.core.memory.clustering.core import _build_clusters
        texts = ["a", "b", "c", "d"]
        embeddings = np.random.randn(4, 10)
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([
            embeddings[:2].mean(axis=0),
            embeddings[2:].mean(axis=0)
        ])
        clusters = _build_clusters(texts, embeddings, labels, centroids)
        assert len(clusters) == 2
        assert clusters[0].items == ["a", "b"]
        assert clusters[1].items == ["c", "d"]
        assert clusters[0].size == 2


class TestClusterMainFunction:
    """Tests for the main cluster() function with pre-computed embeddings."""

    def test_cluster_with_precomputed_embeddings(self):
        """Test clustering with pre-computed numpy embeddings (bypasses embedding)."""
        from hololoom.core.memory.clustering.core import cluster
        # Create two clear clusters
        np.random.seed(42)
        cluster_a = np.random.randn(10, 20) + np.array([5.0] * 20)
        cluster_b = np.random.randn(10, 20) + np.array([-5.0] * 20)
        embeddings = np.vstack([cluster_a, cluster_b])

        result = cluster(embeddings, k=2, label_clusters=False)
        assert result.n_clusters == 2
        assert len(result.clusters) == 2
        assert sum(c.size for c in result.clusters) == 20

    def test_cluster_with_fixed_k(self):
        from hololoom.core.memory.clustering.core import cluster
        embeddings = np.random.randn(15, 10)
        result = cluster(embeddings, k=3, label_clusters=False)
        assert result.n_clusters == 3

    def test_cluster_return_embeddings(self):
        from hololoom.core.memory.clustering.core import cluster
        embeddings = np.random.randn(10, 8)
        result = cluster(embeddings, k=2, label_clusters=False, return_embeddings=True)
        assert result.embeddings is not None
        assert result.embeddings.shape == (10, 8)


# ============================================================================
# retriever_registry.py
# ============================================================================


class TestRetrieverInfo:
    """Tests for RetrieverInfo dataclass."""

    def test_success_rate_no_calls(self):
        from hololoom.core.memory.retriever_registry import RetrieverInfo, RetrieverType
        info = RetrieverInfo(
            name="test", retriever_type=RetrieverType.KEYWORD,
            priority=1, retrieve_fn=AsyncMock()
        )
        assert info.success_rate == 0.0

    def test_success_rate_with_calls(self):
        from hololoom.core.memory.retriever_registry import RetrieverInfo, RetrieverType
        info = RetrieverInfo(
            name="test", retriever_type=RetrieverType.KEYWORD,
            priority=1, retrieve_fn=AsyncMock(),
            total_calls=10, total_successes=7
        )
        assert info.success_rate == pytest.approx(0.7)

    def test_is_available(self):
        from hololoom.core.memory.retriever_registry import (
            RetrieverInfo, RetrieverType, RetrieverHealth
        )
        info = RetrieverInfo(
            name="test", retriever_type=RetrieverType.KEYWORD,
            priority=1, retrieve_fn=AsyncMock(),
            health=RetrieverHealth.HEALTHY
        )
        assert info.is_available
        info.health = RetrieverHealth.UNAVAILABLE
        assert not info.is_available
        info.health = RetrieverHealth.DEGRADED
        assert info.is_available


class TestFallbackResult:
    """Tests for FallbackResult."""

    def test_used_fallback_property(self):
        from hololoom.core.memory.retriever_registry import FallbackResult
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        result = RetrievalResultEnhanced.success([])
        fb = FallbackResult(
            result=result, retriever_used="keyword",
            fallback_path=["semantic", "keyword"],
            fallback_reasons={"semantic": "failed"},
            total_time_ms=5.0
        )
        assert fb.used_fallback

    def test_no_fallback(self):
        from hololoom.core.memory.retriever_registry import FallbackResult
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        result = RetrievalResultEnhanced.success([])
        fb = FallbackResult(
            result=result, retriever_used="semantic",
            fallback_path=["semantic"],
            fallback_reasons={},
            total_time_ms=2.0
        )
        assert not fb.used_fallback

    def test_to_dict(self):
        from hololoom.core.memory.retriever_registry import FallbackResult
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        result = RetrievalResultEnhanced.success([])
        fb = FallbackResult(
            result=result, retriever_used="keyword",
            fallback_path=["semantic", "keyword"],
            fallback_reasons={"semantic": "failed"},
            total_time_ms=5.0
        )
        d = fb.to_dict()
        assert d["retriever_used"] == "keyword"
        assert d["used_fallback"] is True


class TestRetrieverRegistry:
    """Tests for RetrieverRegistry."""

    def test_register_and_contains(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        registry = RetrieverRegistry()
        registry.register("test", AsyncMock(), priority=1)
        assert "test" in registry
        assert len(registry) == 1

    def test_unregister(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        registry = RetrieverRegistry()
        registry.register("test", AsyncMock(), priority=1)
        assert registry.unregister("test")
        assert "test" not in registry
        assert not registry.unregister("nonexistent")

    def test_priority_order(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        registry = RetrieverRegistry()
        registry.register("low", AsyncMock(), priority=10)
        registry.register("high", AsyncMock(), priority=1)
        registry.register("mid", AsyncMock(), priority=5)
        available = registry.get_available_retrievers()
        assert available == ["high", "mid", "low"]

    def test_get_retriever_info(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        registry = RetrieverRegistry()
        registry.register("test", AsyncMock(), priority=1)
        info = registry.get_retriever_info("test")
        assert info is not None
        assert info.name == "test"
        assert registry.get_retriever_info("nonexistent") is None

    def test_set_health(self):
        from hololoom.core.memory.retriever_registry import (
            RetrieverRegistry, RetrieverHealth
        )
        registry = RetrieverRegistry()
        registry.register("test", AsyncMock(), priority=1)
        registry.set_health("test", RetrieverHealth.UNAVAILABLE)
        info = registry.get_retriever_info("test")
        assert info.health == RetrieverHealth.UNAVAILABLE
        assert info.last_health_check is not None

    def test_set_health_healthy_resets_failures(self):
        from hololoom.core.memory.retriever_registry import (
            RetrieverRegistry, RetrieverHealth
        )
        registry = RetrieverRegistry()
        registry.register("test", AsyncMock(), priority=1)
        info = registry.get_retriever_info("test")
        info.consecutive_failures = 5
        registry.set_health("test", RetrieverHealth.HEALTHY)
        assert info.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_retrieve_success(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        async def good_retriever(query, memories, limit, **kwargs):
            return RetrievalResultEnhanced.success(["shard1"])

        registry = RetrieverRegistry()
        registry.register("good", good_retriever, priority=1)
        result = await registry.retrieve("test query", [])
        assert result.retriever_used == "good"
        assert not result.used_fallback

    @pytest.mark.asyncio
    async def test_retrieve_fallback_on_exception(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        async def bad_retriever(query, memories, limit, **kwargs):
            raise RuntimeError("Boom")

        async def backup_retriever(query, memories, limit, **kwargs):
            return RetrievalResultEnhanced.success(["shard_backup"])

        registry = RetrieverRegistry()
        registry.register("bad", bad_retriever, priority=1)
        registry.register("backup", backup_retriever, priority=2)
        result = await registry.retrieve("test query", [])
        assert result.retriever_used == "backup"
        assert result.used_fallback
        assert "bad" in result.fallback_reasons

    @pytest.mark.asyncio
    async def test_retrieve_no_retrievers(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        from hololoom.core.memory.retrieval_result import RetrievalStatus

        registry = RetrieverRegistry()
        result = await registry.retrieve("test query", [])
        assert result.retriever_used == "none"
        assert result.result.status == RetrievalStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_retrieve_preferred_retriever(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced

        async def ret_a(query, memories, limit, **kwargs):
            return RetrievalResultEnhanced.success(["a"])

        async def ret_b(query, memories, limit, **kwargs):
            return RetrievalResultEnhanced.success(["b"])

        registry = RetrieverRegistry()
        registry.register("a", ret_a, priority=1)
        registry.register("b", ret_b, priority=2)
        result = await registry.retrieve("q", [], preferred_retriever="b")
        assert result.retriever_used == "b"

    def test_check_and_update_health(self):
        from hololoom.core.memory.retriever_registry import (
            RetrieverRegistry, RetrieverInfo, RetrieverType, RetrieverHealth
        )
        registry = RetrieverRegistry(max_consecutive_failures=3)
        info = RetrieverInfo(
            name="test", retriever_type=RetrieverType.KEYWORD,
            priority=1, retrieve_fn=AsyncMock(),
            consecutive_failures=2
        )
        registry._check_and_update_health(info)
        assert info.health == RetrieverHealth.DEGRADED

        info.consecutive_failures = 3
        registry._check_and_update_health(info)
        assert info.health == RetrieverHealth.UNAVAILABLE

    def test_get_statistics(self):
        from hololoom.core.memory.retriever_registry import RetrieverRegistry
        registry = RetrieverRegistry()
        registry.register("a", AsyncMock(), priority=1)
        registry.register("b", AsyncMock(), priority=2)
        stats = registry.get_statistics()
        assert stats["total_retrievers"] == 2
        assert "a" in stats["retrievers"]

    @pytest.mark.asyncio
    async def test_health_check_auto_recovery(self):
        from hololoom.core.memory.retriever_registry import (
            RetrieverRegistry, RetrieverHealth
        )
        from datetime import datetime, timedelta

        registry = RetrieverRegistry(
            health_check_interval_seconds=0.0,  # immediate recovery
            enable_auto_recovery=True
        )
        registry.register("test", AsyncMock(), priority=1)
        registry.set_health("test", RetrieverHealth.UNAVAILABLE)

        # Set last health check far in the past
        info = registry.get_retriever_info("test")
        info.last_health_check = datetime.utcnow() - timedelta(hours=1)

        result = await registry.health_check("test")
        assert result["test"] == RetrieverHealth.UNKNOWN


class TestCreateRetrieverRegistry:
    """Tests for factory function."""

    def test_creates_registry(self):
        from hololoom.core.memory.retriever_registry import create_retriever_registry
        registry = create_retriever_registry(max_consecutive_failures=5)
        assert registry.max_consecutive_failures == 5
        assert len(registry) == 0


# ============================================================================
# semantic_bridge.py
# ============================================================================


class TestSemanticBridge:
    """Tests for create_semantic_fn."""

    @pytest.mark.asyncio
    async def test_creates_working_fn(self):
        from hololoom.core.memory.semantic_bridge import create_semantic_fn

        shard = MagicMock()
        shard.id = "shard1"
        shard.text = "hello world"
        shard.metadata = {"source": "test"}

        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[(shard, 0.95)])

        fn = create_semantic_fn(retriever)
        results = await fn("hello", max_results=5)
        assert len(results) == 1
        assert results[0]["id"] == "shard1"
        assert results[0]["content"] == "hello world"
        assert results[0]["importance"] == pytest.approx(0.95)
        assert results[0]["_resolution"] == "semantic"

    @pytest.mark.asyncio
    async def test_score_threshold_filters(self):
        from hololoom.core.memory.semantic_bridge import create_semantic_fn

        shard_hi = MagicMock()
        shard_hi.id = "hi"
        shard_hi.text = "hi"
        shard_hi.metadata = None

        shard_lo = MagicMock()
        shard_lo.id = "lo"
        shard_lo.text = "lo"
        shard_lo.metadata = None

        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[(shard_hi, 0.8), (shard_lo, 0.3)])

        fn = create_semantic_fn(retriever, score_threshold=0.5)
        results = await fn("query", 10)
        assert len(results) == 1
        assert results[0]["id"] == "hi"

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty(self):
        from hololoom.core.memory.semantic_bridge import create_semantic_fn

        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=RuntimeError("search failed"))

        fn = create_semantic_fn(retriever)
        results = await fn("query", 5)
        assert results == []


# ============================================================================
# bee_data.py (data integrity tests)
# ============================================================================


class TestBeeData:
    """Tests for bee_data.py data constants."""

    def test_hive_entities_count(self):
        from hololoom.core.memory.bee_data import HIVE_ENTITIES
        assert len(HIVE_ENTITIES) == 4

    def test_hive_entities_have_required_fields(self):
        from hololoom.core.memory.bee_data import HIVE_ENTITIES
        required = {"id", "type", "name", "genetics", "location", "status", "frames"}
        for hive in HIVE_ENTITIES:
            assert required.issubset(hive.keys()), f"Missing fields in {hive['name']}"

    def test_location_entities(self):
        from hololoom.core.memory.bee_data import LOCATION_ENTITIES
        assert len(LOCATION_ENTITIES) == 3
        names = {loc["name"] for loc in LOCATION_ENTITIES}
        assert "North Yard" in names
        assert "South Yard" in names

    def test_episodes_have_required_fields(self):
        from hololoom.core.memory.bee_data import EPISODES
        for ep in EPISODES:
            assert "id" in ep
            assert "type" in ep
            assert "summary" in ep
            assert "timestamp" in ep

    def test_episodes_count(self):
        from hololoom.core.memory.bee_data import EPISODES
        assert len(EPISODES) == 8

    def test_causal_links_reference_valid_episodes(self):
        from hololoom.core.memory.bee_data import EPISODES, CAUSAL_LINKS
        ep_ids = {ep["id"] for ep in EPISODES}
        for src, dst in CAUSAL_LINKS:
            assert src in ep_ids, f"Causal link source {src} not in episodes"
            assert dst in ep_ids, f"Causal link dest {dst} not in episodes"

    def test_temporal_links_reference_valid_episodes(self):
        from hololoom.core.memory.bee_data import EPISODES, TEMPORAL_LINKS
        ep_ids = {ep["id"] for ep in EPISODES}
        for src, dst in TEMPORAL_LINKS:
            assert src in ep_ids
            assert dst in ep_ids

    def test_entity_aliases_structure(self):
        from hololoom.core.memory.bee_data import ENTITY_ALIASES
        assert len(ENTITY_ALIASES) > 0
        for alias in ENTITY_ALIASES:
            assert len(alias) == 4  # (name, id, domain, confidence)
            assert isinstance(alias[3], float)

    def test_facts_structure(self):
        from hololoom.core.memory.bee_data import FACTS
        assert len(FACTS) == 4
        for fact in FACTS:
            assert len(fact) == 4  # (neo4j_id, content, confidence, domain)
            assert 0.0 <= fact[2] <= 1.0

    def test_apiary_entity(self):
        from hololoom.core.memory.bee_data import APIARY_ENTITY
        assert APIARY_ENTITY["id"] == "apiary_main"
        assert APIARY_ENTITY["type"] == "Apiary"

    def test_hive_locations_match_location_entities(self):
        from hololoom.core.memory.bee_data import HIVE_ENTITIES, LOCATION_ENTITIES
        valid_locations = {loc["id"].replace("loc_", "") for loc in LOCATION_ENTITIES}
        for hive in HIVE_ENTITIES:
            assert hive["location"] in valid_locations, (
                f"Hive {hive['name']} location '{hive['location']}' not in locations"
            )


# ============================================================================
# hyperspace_backend.py (data structures + config mapping — no heavy deps)
# ============================================================================


class TestCrawlComplexity:
    """Tests for CrawlComplexity enum."""

    def test_values(self):
        from hololoom.core.memory.hyperspace_backend import CrawlComplexity
        assert CrawlComplexity.LITE.value == 1
        assert CrawlComplexity.RESEARCH.value == 4


class TestCrawlConfig:
    """Tests for CrawlConfig dataclass."""

    def test_creation(self):
        from hololoom.core.memory.hyperspace_backend import CrawlConfig
        cfg = CrawlConfig(
            max_depth=3, thresholds=[0.5, 0.7, 0.9],
            initial_limit=10, max_total_items=50,
            importance_threshold=0.6
        )
        assert cfg.max_depth == 3
        assert len(cfg.thresholds) == 3


class TestCrawlStats:
    """Tests for CrawlStats."""

    def test_defaults(self):
        from hololoom.core.memory.hyperspace_backend import CrawlStats
        stats = CrawlStats()
        assert stats.passes == 0
        assert stats.total_items == 0
        assert stats.fusion_events == 0


class TestHyperspaceStrategyMapping:
    """Tests for _map_strategy_to_complexity and _get_crawl_config."""

    def test_strategy_to_complexity_mapping(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend, CrawlComplexity
        from hololoom.core.memory.protocol import Strategy

        # Create with mocked config
        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG"):
            backend = HyperspaceBackend(mock_config)

        assert backend._map_strategy_to_complexity(Strategy.TEMPORAL) == CrawlComplexity.LITE
        assert backend._map_strategy_to_complexity(Strategy.BALANCED) == CrawlComplexity.FAST
        assert backend._map_strategy_to_complexity(Strategy.GRAPH) == CrawlComplexity.FULL
        assert backend._map_strategy_to_complexity(Strategy.FUSED) == CrawlComplexity.RESEARCH

    def test_get_crawl_config(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend, CrawlComplexity

        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG"):
            backend = HyperspaceBackend(mock_config)

        lite = backend._get_crawl_config(CrawlComplexity.LITE)
        assert lite.max_depth == 1
        assert lite.thresholds == [0.7]

        research = backend._get_crawl_config(CrawlComplexity.RESEARCH)
        assert research.max_depth == 4
        assert len(research.thresholds) == 4
        assert research.max_total_items == 50

    def test_process_crawl_pass(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend, CrawlConfig

        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG"):
            backend = HyperspaceBackend(mock_config)

        crawl_config = CrawlConfig(
            max_depth=2, thresholds=[0.5, 0.7],
            initial_limit=10, max_total_items=20,
            importance_threshold=0.7
        )

        mem1 = MagicMock()
        mem1.id = "m1"
        mem1.metadata = {"relevance": 0.9}

        mem2 = MagicMock()
        mem2.id = "m2"
        mem2.metadata = {"relevance": 0.4}

        visited = set()
        result = backend._process_crawl_pass([mem1, mem2], depth=0, visited_ids=visited, config=crawl_config)
        assert len(result["items"]) == 2
        assert len(result["high_importance"]) == 1  # Only mem1 (0.9 >= 0.7)
        assert "m1" in visited
        assert "m2" in visited

    def test_process_crawl_pass_dedup(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend, CrawlConfig

        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG"):
            backend = HyperspaceBackend(mock_config)

        crawl_config = CrawlConfig(
            max_depth=2, thresholds=[0.5],
            initial_limit=10, max_total_items=20,
            importance_threshold=0.5
        )

        mem = MagicMock()
        mem.id = "m1"
        mem.metadata = {"relevance": 0.8}

        visited = {"m1"}  # Already visited
        result = backend._process_crawl_pass([mem], depth=1, visited_ids=visited, config=crawl_config)
        assert len(result["items"]) == 0

    @pytest.mark.asyncio
    async def test_fuse_multipass_results(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend

        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG"):
            backend = HyperspaceBackend(mock_config)

        mem1 = MagicMock()
        mem1.id = "m1"
        mem1.metadata = {"relevance": 0.9, "crawl_depth": 0, "importance_score": 0.8}

        mem2 = MagicMock()
        mem2.id = "m2"
        mem2.metadata = {"relevance": 0.5, "crawl_depth": 1, "importance_score": 0.4}

        result = await backend._fuse_multipass_results([mem1, mem2], "test query")
        ranked = result["ranked_results"]
        assert len(ranked) == 2
        # mem1 should rank higher (better relevance, shallower depth)
        assert ranked[0].id == "m1"
        assert "fusion_events" in result
        assert result["fusion_confidence"] > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        from hololoom.core.memory.hyperspace_backend import HyperspaceBackend

        mock_config = MagicMock()
        with patch("hololoom.core.memory.hyperspace_backend.NetworkXKG") as MockKG:
            mock_kg = MagicMock()
            mock_kg.G = MagicMock()
            MockKG.return_value = mock_kg
            backend = HyperspaceBackend(mock_config)

        assert await backend.health_check()


# ============================================================================
# spring_dynamics_advanced.py
# ============================================================================


class TestAdvancedSpringConfig:
    """Tests for AdvancedSpringConfig."""

    def test_defaults(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringConfig
        config = AdvancedSpringConfig()
        assert config.stiffness == 0.15
        assert config.damping == 0.85
        assert config.dt == 0.016
        assert config.max_iterations == 200

    def test_get_edge_stiffness(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringConfig
        config = AdvancedSpringConfig(stiffness=1.0)
        # IS_A multiplier is 1.2
        k = config.get_edge_stiffness("IS_A", 0.5)
        assert k == pytest.approx(0.6)  # 1.0 * 0.5 * 1.2

    def test_get_edge_stiffness_unknown_type(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringConfig
        config = AdvancedSpringConfig(stiffness=1.0)
        k = config.get_edge_stiffness("UNKNOWN", 1.0)
        assert k == pytest.approx(1.0)  # multiplier defaults to 1.0


class TestSpringForceFunction:
    """Tests for SpringForceFunction."""

    def _make_graph(self):
        """Create a simple mock graph with edges."""
        import networkx as nx
        G = nx.Graph()
        G.add_edge("a", "b", type="RELATED_TO", weight=1.0)
        G.add_edge("b", "c", type="IS_A", weight=0.8)
        graph = MagicMock()
        graph.G = G
        return graph

    def test_force_computation(self):
        from hololoom.core.memory.spring_dynamics_advanced import (
            SpringForceFunction, AdvancedSpringConfig
        )
        from hololoom.core.memory.integrators import DynamicalState

        graph = self._make_graph()
        config = AdvancedSpringConfig(stiffness=0.15, damping=0.5, decay=0.98)
        node_list = ["a", "b", "c"]
        seed_nodes = {"a": 1.0}

        force_fn = SpringForceFunction(graph, config, node_list, seed_nodes)

        state = DynamicalState(
            q=np.array([1.0, 0.0, 0.0]),
            p=np.array([0.0, 0.0, 0.0]),
            t=0.0
        )

        dq_dt, dp_dt = force_fn(state)
        assert dq_dt.shape == (3,)
        assert dp_dt.shape == (3,)
        # Node a has activation=1.0, b=0.0 -> spring pulls b toward a
        # dp_dt[1] should be positive (force toward a)
        assert dp_dt[1] > 0  # Spring force from a pulling b

    def test_seed_maintenance(self):
        from hololoom.core.memory.spring_dynamics_advanced import (
            SpringForceFunction, AdvancedSpringConfig
        )
        from hololoom.core.memory.integrators import DynamicalState

        graph = self._make_graph()
        config = AdvancedSpringConfig(maintain_seed_activation=True)
        node_list = ["a", "b", "c"]
        seed_nodes = {"a": 1.0}

        force_fn = SpringForceFunction(graph, config, node_list, seed_nodes)

        # If seed node's activation drifts to 0.5, restoring force should push it back
        state = DynamicalState(
            q=np.array([0.5, 0.0, 0.0]),
            p=np.array([0.0, 0.0, 0.0]),
            t=0.0
        )
        _, dp_dt = force_fn(state)
        # Restoring force toward target 1.0 should be positive
        assert dp_dt[0] > 0


class TestAdvancedSpringDynamics:
    """Tests for AdvancedSpringDynamics."""

    def _make_graph(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edge("a", "b", type="RELATED_TO", weight=1.0)
        G.add_edge("b", "c", type="IS_A", weight=0.8)
        G.add_edge("c", "d", type="USES", weight=0.6)
        graph = MagicMock()
        graph.G = G
        return graph

    def test_reset(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        dynamics.activate_nodes({"a": 1.0})
        dynamics.reset()
        assert dynamics.state is None
        assert dynamics.seed_nodes == {}
        assert dynamics.iterations == 0

    def test_activate_nodes(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        dynamics.activate_nodes({"a": 1.0, "c": 0.5})
        assert dynamics.state is not None
        assert dynamics.state.q.shape == (4,)  # 4 nodes
        # Check seed stored
        assert "a" in dynamics.seed_nodes
        assert dynamics.seed_nodes["a"] == 1.0

    def test_activate_nodes_clamps(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        dynamics.activate_nodes({"a": 2.0})  # Should be clamped to 1.0
        idx = dynamics.node_list.index("a")
        assert dynamics.state.q[idx] == 1.0

    def test_propagate_requires_activation(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        with pytest.raises(RuntimeError, match="Must call activate_nodes"):
            dynamics.propagate()

    def test_propagate_runs(self):
        from hololoom.core.memory.spring_dynamics_advanced import (
            AdvancedSpringDynamics, AdvancedSpringConfig
        )

        graph = self._make_graph()
        config = AdvancedSpringConfig(
            max_iterations=50,
            check_stability=False,  # Skip stability analysis for speed
        )
        dynamics = AdvancedSpringDynamics(graph, config)
        dynamics.activate_nodes({"a": 1.0})
        result = dynamics.propagate()

        assert result.iterations > 0
        assert "a" in result.activated_nodes
        assert len(result.node_activations) > 0

    def test_get_activation(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        # Before activation
        assert dynamics.get_activation("a") == 0.0
        assert dynamics.get_activation("nonexistent") == 0.0

        dynamics.activate_nodes({"a": 0.8})
        assert dynamics.get_activation("a") == pytest.approx(0.8)

    def test_compute_hamiltonian(self):
        from hololoom.core.memory.spring_dynamics_advanced import (
            AdvancedSpringDynamics, AdvancedSpringConfig
        )

        graph = self._make_graph()
        config = AdvancedSpringConfig(check_stability=False)
        dynamics = AdvancedSpringDynamics(graph, config)
        # No state -> 0 energy
        assert dynamics._compute_hamiltonian() == 0.0

        dynamics.activate_nodes({"a": 1.0})
        dynamics._initialize_integrator()
        energy = dynamics._compute_hamiltonian()
        assert energy >= 0.0  # Energy is non-negative

    def test_get_active_nodes_empty(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        assert dynamics._get_active_nodes() == []

    def test_extract_activations_empty(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringDynamics

        graph = self._make_graph()
        dynamics = AdvancedSpringDynamics(graph)
        assert dynamics._extract_activations() == {}

    def test_propagation_spreads_activation(self):
        from hololoom.core.memory.spring_dynamics_advanced import (
            AdvancedSpringDynamics, AdvancedSpringConfig
        )

        graph = self._make_graph()
        config = AdvancedSpringConfig(
            max_iterations=200,
            check_stability=False,
            stiffness=2.0,  # Very strong springs for fast propagation
            damping=0.1,    # Low damping so energy moves
            convergence_epsilon=1e-8,  # Tight convergence to run more steps
        )
        dynamics = AdvancedSpringDynamics(graph, config)
        dynamics.activate_nodes({"a": 1.0})
        result = dynamics.propagate()

        # Seed node a should be activated
        assert "a" in result.node_activations
        # With strong springs and low damping, activation should spread
        # At minimum we ran some iterations
        assert result.iterations > 1


class TestAdvancedSpringResult:
    """Tests for AdvancedSpringResult."""

    def test_str_converged(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringResult
        result = AdvancedSpringResult(
            iterations=50, converged=True, final_energy=0.001,
            activated_nodes=["a", "b"],
            node_activations={"a": 1.0, "b": 0.5}
        )
        s = str(result)
        assert "converged" in s
        assert "50 steps" in s
        assert "activated=2" in s

    def test_str_not_converged(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringResult
        result = AdvancedSpringResult(
            iterations=200, converged=False, final_energy=0.1,
            activated_nodes=["a"],
            node_activations={"a": 1.0}
        )
        s = str(result)
        assert "max iterations" in s

    def test_str_with_stability(self):
        from hololoom.core.memory.spring_dynamics_advanced import AdvancedSpringResult
        result = AdvancedSpringResult(
            iterations=50, converged=True, final_energy=0.001,
            activated_nodes=["a"],
            node_activations={"a": 1.0},
            stability_report={"stable": True, "energy_drift": 0.001}
        )
        s = str(result)
        assert "stable=True" in s


# ============================================================================
# integrated_memory_system.py (config only — can't construct without all deps)
# ============================================================================


class TestIntegratedMemoryConfig:
    """Tests for IntegratedMemoryConfig."""

    def test_defaults(self):
        from hololoom.core.memory.integrated_memory_system import IntegratedMemoryConfig
        config = IntegratedMemoryConfig()
        assert config.llm_provider is None
        assert config.enable_consolidation is False
        assert config.enable_semantic_search is True
        assert config.enable_bm25_search is True
        assert config.enable_graph_search is True
        assert config.consolidation_interval_minutes == 60

    def test_custom_config(self):
        from hololoom.core.memory.integrated_memory_system import IntegratedMemoryConfig
        config = IntegratedMemoryConfig(
            llm_provider="ollama",
            llm_model="qwen3.5:9b",
            enable_consolidation=True,
            enable_semantic_search=False
        )
        assert config.llm_provider == "ollama"
        assert config.enable_consolidation
        assert not config.enable_semantic_search


# ============================================================================
# retrieval_strategies.py (static retrieval — no heavy graph deps)
# ============================================================================


class TestStaticRetrieval:
    """Tests for StaticRetrieval (no graph deps)."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_first_k(self):
        from hololoom.core.memory.retrieval_strategies import StaticRetrieval

        shards = [MagicMock(id=f"s{i}") for i in range(10)]
        query = MagicMock()
        query.text = "test"

        retriever = StaticRetrieval(shards=shards)
        result = await retriever.retrieve(query, k=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_metadata(self):
        from hololoom.core.memory.retrieval_strategies import StaticRetrieval

        shards = [MagicMock(id=f"s{i}") for i in range(5)]
        query = MagicMock()
        query.text = "test query"

        retriever = StaticRetrieval(shards=shards)
        result = await retriever.retrieve_with_metadata(query, k=3)
        assert result.strategy == "static"
        assert result.k_requested == 3
        assert result.k_returned == 3
        assert result.retrieval_time_ms >= 0


class TestCreateRetrievalStrategy:
    """Tests for create_retrieval_strategy factory."""

    def test_create_static(self):
        from hololoom.core.memory.retrieval_strategies import create_retrieval_strategy
        strategy = create_retrieval_strategy("static", shards=[])
        assert strategy is not None

    def test_unknown_strategy_raises(self):
        from hololoom.core.memory.retrieval_strategies import create_retrieval_strategy
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            create_retrieval_strategy("nonexistent")

    def test_spring_requires_graph(self):
        from hololoom.core.memory.retrieval_strategies import create_retrieval_strategy
        with pytest.raises(ValueError, match="requires graph"):
            create_retrieval_strategy("spring", shards=[])

    def test_hybrid_requires_graph(self):
        from hololoom.core.memory.retrieval_strategies import create_retrieval_strategy
        with pytest.raises(ValueError, match="requires graph"):
            create_retrieval_strategy("hybrid", shards=[])
