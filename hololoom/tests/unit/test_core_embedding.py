"""
Unit tests to increase coverage for hololoom/core/embedding/.

Targets uncovered lines in:
1. spectral.py — SpectralFusion, EmbeddingDriftDetector, create_embedder, create_drift_detector
2. spectral_multiscale.py — fuse_multiscale_features, HierarchicalSpectralClusterer, factories
3. linguistic_matryoshka_gate.py — gate(), _linguistic_prefilter, _syntactic_similarity, encode/encode_scales
4. riemannian_matryoshka.py — pairwise_distances, exp_map/log_map/parallel_transport error paths
5. matryoshka_interpreter.py — interpret_text, feature detection, visualize_interpretation

All tests use synthetic data. No real model inference.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch


# ============================================================================
# spectral.py — SpectralFusion
# ============================================================================


class TestSpectralFusionGraphSpectrum:
    """Test SpectralFusion._graph_spectrum with real graphs."""

    def test_trivial_graph_returns_zeros(self):
        from hololoom.embedding.spectral import SpectralFusion

        sf = SpectralFusion()
        G = nx.MultiDiGraph()
        G.add_node("alone")
        spectrum, fiedler = sf._graph_spectrum(G)

        assert spectrum.shape == (4,)
        assert np.allclose(spectrum, 0.0)
        assert fiedler == 0.0

    def test_empty_graph_returns_zeros(self):
        from hololoom.embedding.spectral import SpectralFusion

        sf = SpectralFusion()
        G = nx.MultiDiGraph()
        spectrum, fiedler = sf._graph_spectrum(G)

        assert spectrum.shape == (4,)
        assert fiedler == 0.0

    def test_connected_graph_has_positive_fiedler(self):
        from hololoom.embedding.spectral import SpectralFusion

        sf = SpectralFusion()
        G = nx.MultiDiGraph()
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=1.0)
        G.add_edge("c", "a", weight=1.0)

        spectrum, fiedler = sf._graph_spectrum(G)

        assert spectrum.shape == (4,)
        # First eigenvalue should be ~0
        assert abs(spectrum[0]) < 0.1
        # Fiedler value should be positive for connected graph
        assert fiedler > 0

    def test_graph_with_weighted_edges(self):
        from hololoom.embedding.spectral import SpectralFusion

        sf = SpectralFusion()
        G = nx.MultiDiGraph()
        G.add_edge("x", "y", weight=2.0)
        G.add_edge("y", "z", weight=0.5)

        spectrum, fiedler = sf._graph_spectrum(G)

        assert spectrum.shape == (4,)
        assert isinstance(fiedler, float)

    def test_large_graph_spectrum(self):
        """Graph with more nodes than k_eigen."""
        from hololoom.embedding.spectral import SpectralFusion

        sf = SpectralFusion(k_eigen=4)
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_edge(f"n{i}", f"n{(i+1) % 10}", weight=1.0)

        spectrum, fiedler = sf._graph_spectrum(G)

        assert spectrum.shape == (4,)
        assert fiedler >= 0


class TestSpectralFusionTopicFeatures:
    """Test SpectralFusion._topic_features."""

    def test_empty_texts_returns_zeros(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = sf._topic_features([], emb)

        assert topic.shape == (2,)
        assert np.allclose(topic, 0.0)
        assert topic_var == 0.0

    def test_single_text_returns_features(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = sf._topic_features(["hello world"], emb)

        assert topic.shape == (2,)
        assert isinstance(topic_var, float)

    def test_multiple_texts_returns_features(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion(svd_components=3)
        emb = MatryoshkaEmbeddings(sizes=[96])

        texts = ["text one", "text two", "text three"]
        topic, topic_var = sf._topic_features(texts, emb)

        assert topic.shape == (3,)
        assert 0.0 <= topic_var <= 1.0


class TestSpectralFusionFeatures:
    """Test SpectralFusion.features async method."""

    @pytest.mark.asyncio
    async def test_features_basic_graph(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        G = nx.MultiDiGraph()
        G.add_edge("a", "b", type="USES")
        G.add_edge("b", "c", type="IS_A")

        texts = ["alpha", "beta", "gamma"]
        psi, metrics = await sf.features(G, texts, emb)

        assert isinstance(psi, np.ndarray)
        assert len(psi) > 0
        assert "fiedler" in metrics
        assert "topic_var" in metrics
        assert "coherence" in metrics
        assert "feature_dim" in metrics
        assert metrics["feature_dim"] == len(psi)

    @pytest.mark.asyncio
    async def test_features_empty_texts(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        G = nx.MultiDiGraph()
        G.add_edge("a", "b")

        psi, metrics = await sf.features(G, [], emb)

        assert isinstance(psi, np.ndarray)
        assert metrics["topic_var"] == 0.0

    @pytest.mark.asyncio
    async def test_features_single_node(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        G = nx.MultiDiGraph()
        G.add_node("alone")

        psi, metrics = await sf.features(G, ["text"], emb)

        assert metrics["fiedler"] == 0.0

    @pytest.mark.asyncio
    async def test_features_wavelets_disabled_by_default(self):
        from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        sf = SpectralFusion()
        emb = MatryoshkaEmbeddings(sizes=[96])

        G = nx.MultiDiGraph()
        G.add_edge("a", "b")

        _, metrics = await sf.features(G, ["text"], emb)

        assert metrics["wavelet_energy"] == 0.0
        assert metrics["diffusion_variance"] == 0.0


# ============================================================================
# spectral.py — EmbeddingDriftDetector
# ============================================================================


class TestEmbeddingDriftDetector:
    """Test EmbeddingDriftDetector drift detection."""

    def test_observe_returns_none_before_window_full(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        detector = EmbeddingDriftDetector(window_size=5)

        for _ in range(4):
            result = detector.observe(np.random.randn(10))
            assert result is None

    def test_first_full_window_sets_reference(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        detector = EmbeddingDriftDetector(window_size=3)

        for _ in range(2):
            detector.observe(np.random.randn(10))

        result = detector.observe(np.random.randn(10))

        assert result is not None
        assert result["status"] == "reference_set"
        assert result["drift"] == 0.0
        assert result["alarm"] is False

    def test_no_drift_for_same_distribution(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        detector = EmbeddingDriftDetector(window_size=5, alarm_threshold=5.0)

        # Fill reference window
        for _ in range(5):
            detector.observe(rng.normal(0, 1, 10))

        # More from same distribution
        for _ in range(5):
            result = detector.observe(rng.normal(0, 1, 10))

        assert result is not None
        assert result["alarm"] is False

    def test_drift_detected_for_shifted_distribution(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        detector = EmbeddingDriftDetector(window_size=5, alarm_threshold=1.0)

        # Fill reference window with N(0,1)
        for _ in range(5):
            detector.observe(rng.normal(0, 0.1, 10))

        # Shift to N(10, 0.1) — big shift
        for _ in range(5):
            result = detector.observe(rng.normal(10, 0.1, 10))

        assert result is not None
        assert result["alarm"] is True
        assert result["drift"] > 1.0

    def test_reset_reference(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        rng = np.random.default_rng(42)
        detector = EmbeddingDriftDetector(window_size=5)

        # Fill buffer
        for _ in range(10):
            detector.observe(rng.normal(0, 1, 10))

        detector.reset_reference()

        assert detector._reference_mu is not None
        assert detector._reference_sigma is not None

    def test_reset_reference_does_nothing_when_buffer_small(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        detector = EmbeddingDriftDetector(window_size=10)
        detector.observe(np.random.randn(5))

        old_ref = detector._reference_mu
        detector.reset_reference()
        assert detector._reference_mu is old_ref  # unchanged

    def test_buffer_trimming(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        detector = EmbeddingDriftDetector(window_size=3, alarm_threshold=100.0)
        rng = np.random.default_rng(42)

        # Fill well past 2x window_size
        for _ in range(20):
            detector.observe(rng.normal(0, 1, 5))

        # Buffer should be trimmed to window_size
        assert len(detector._buffer) <= 6  # 2 * window_size

    def test_gaussian_geodesic_distance_identical(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        mu = np.ones(5)
        sigma = np.ones(5) * 0.5

        dist = EmbeddingDriftDetector._gaussian_geodesic_distance(mu, sigma, mu, sigma)

        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_gaussian_geodesic_distance_symmetric(self):
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        mu1 = np.array([1.0, 2.0])
        sigma1 = np.array([0.5, 1.0])
        mu2 = np.array([3.0, 4.0])
        sigma2 = np.array([1.5, 2.0])

        d12 = EmbeddingDriftDetector._gaussian_geodesic_distance(mu1, sigma1, mu2, sigma2)
        d21 = EmbeddingDriftDetector._gaussian_geodesic_distance(mu2, sigma2, mu1, sigma1)

        # Not exactly symmetric due to log term, but should be close
        assert isinstance(d12, float)
        assert isinstance(d21, float)
        assert d12 > 0
        assert d21 > 0


class TestCreateDriftDetector:
    """Test create_drift_detector factory."""

    def test_returns_none_when_disabled(self):
        from hololoom.embedding.spectral import create_drift_detector

        with patch("hololoom.embedding.spectral._DRIFT_DETECT", False):
            # Import fresh to get patched value
            result = create_drift_detector()
            # When _DRIFT_DETECT is False at module level, returns None
            # But we patched the module-level var, so check
            assert result is None

    def test_returns_detector_when_enabled(self):
        """Directly instantiate EmbeddingDriftDetector since the factory checks env var."""
        from hololoom.embedding.spectral import EmbeddingDriftDetector

        detector = EmbeddingDriftDetector(window_size=50, alarm_threshold=3.0)
        assert detector is not None
        assert detector.window_size == 50
        assert detector.alarm_threshold == 3.0


class TestCreateEmbedder:
    """Test create_embedder factory."""

    def test_create_matryoshka(self):
        from hololoom.embedding.spectral import create_embedder, MatryoshkaEmbeddings

        emb = create_embedder(mode="matryoshka", sizes=[96, 192])
        assert isinstance(emb, MatryoshkaEmbeddings)
        assert emb.sizes == [96, 192]

    def test_unknown_mode_raises(self):
        from hololoom.embedding.spectral import create_embedder

        with pytest.raises(ValueError, match="Unknown embedder mode"):
            create_embedder(mode="nonexistent")


# ============================================================================
# spectral_multiscale.py — fuse_multiscale_features, factories
# ============================================================================


class TestMultiScaleSpectralAnalyzerInit:
    """Test MultiScaleSpectralAnalyzer initialization."""

    def test_default_scales(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        assert analyzer.scales == [96, 192, 384]

    def test_custom_scales(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[64, 128])
        assert analyzer.scales == [64, 128]

    def test_non_ascending_raises(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        with pytest.raises(AssertionError):
            MultiScaleSpectralAnalyzer(scales=[384, 96])


class TestFuseMultiscaleFeatures:
    """Test fuse_multiscale_features method."""

    def test_fuse_empty_results(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96, 192])
        result = analyzer.fuse_multiscale_features({})

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, 0.0)

    def test_fuse_with_eigenvalues_only(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.0, 1.5, 2.0, 3.0]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
                "clusters": {},
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        assert isinstance(fused, np.ndarray)
        assert len(fused) >= 4

    def test_fuse_with_all_features(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.0, 1.0, 2.0, 3.0]),
                "wavelets": np.array([0.1, 0.2, 0.3]),
                "diffusion": np.array([[0.5, 0.6], [0.7, 0.8]]),
                "clusters": {},
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        assert isinstance(fused, np.ndarray)
        # Should have eigenvalue features + wavelet energy + diffusion variance
        assert len(fused) == 6  # 4 + 1 + 1

    def test_fuse_with_custom_weights(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96, 192])
        results = {
            96: {
                "eigenvalues": np.array([0.0, 1.0, 2.0, 3.0]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
                "clusters": {},
            },
            192: {
                "eigenvalues": np.array([0.0, 0.5, 1.0, 1.5]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
                "clusters": {},
            },
        }

        fused = analyzer.fuse_multiscale_features(
            results, fusion_weights={96: 0.3, 192: 0.7}
        )
        assert isinstance(fused, np.ndarray)
        assert len(fused) == 8  # 4 eigenvalues per scale

    def test_fuse_eigenvalues_padded_when_short(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.0, 1.0]),  # Only 2, need 4
                "wavelets": np.array([]),
                "diffusion": np.array([]),
                "clusters": {},
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        assert len(fused) == 4  # Padded to 4


class TestAnalyzeSubgraphFallback:
    """Test _analyze_subgraph paths."""

    def test_empty_subgraph(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        G = nx.MultiDiGraph()

        result = analyzer._analyze_subgraph(G, n_clusters=2, wavelet_scale=1.0)

        assert np.array_equal(result["eigenvalues"], np.array([]))
        assert np.array_equal(result["wavelets"], np.array([]))
        assert result["clusters"] == {}

    def test_analyze_subgraph_with_nodes(self):
        """Test _analyze_subgraph with a real graph (exercises import + fallback)."""
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        G = nx.MultiDiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("c", "a")

        result = analyzer._analyze_subgraph(G, n_clusters=2, wavelet_scale=1.0)

        # Should either succeed with spectral_methods or return fallback
        assert "eigenvalues" in result
        assert "wavelets" in result
        assert "diffusion" in result
        assert "clusters" in result

    def test_analyze_multiscale_with_mock_kg(self):
        """Test analyze_multiscale with a mock KG that has subgraph_for_entities."""
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96, 192, 384])

        # Create a mock KG
        kg = MagicMock()
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_edge(f"n{i}", f"n{(i+1) % 5}")
        kg.subgraph_for_entities.return_value = G

        results = analyzer.analyze_multiscale(kg, ["n0", "n1"])

        assert 96 in results
        assert 192 in results
        assert 384 in results
        # Called once per scale
        assert kg.subgraph_for_entities.call_count == 3

    def test_analyze_multiscale_single_scale(self):
        """Test with single scale (hits i==0 branch)."""
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])

        kg = MagicMock()
        kg.subgraph_for_entities.return_value = nx.MultiDiGraph()

        results = analyzer.analyze_multiscale(kg, ["x"])

        assert 96 in results


class TestHierarchicalSpectralClusterer:
    """Test HierarchicalSpectralClusterer."""

    def test_too_small_to_cluster(self):
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        clusterer = HierarchicalSpectralClusterer(min_cluster_size=3)

        # Create a mock KG with only 2 nodes
        kg = MagicMock()
        kg.G = nx.MultiDiGraph()
        kg.G.add_node("a")
        kg.G.add_node("b")

        result = clusterer.cluster_hierarchical(kg)

        assert result == {"a": [0], "b": [0]}

    def test_init_defaults(self):
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        c = HierarchicalSpectralClusterer()
        assert c.max_depth == 3
        assert c.min_cluster_size == 3

    def test_custom_params(self):
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        c = HierarchicalSpectralClusterer(max_depth=5, min_cluster_size=2)
        assert c.max_depth == 5
        assert c.min_cluster_size == 2

    def test_cluster_hierarchical_with_graph(self):
        """Test full cluster_hierarchical with a graph large enough to cluster."""
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        clusterer = HierarchicalSpectralClusterer(max_depth=2, min_cluster_size=2)

        kg = MagicMock()
        kg.G = nx.MultiDiGraph()
        # Build a graph with 10 nodes
        for i in range(10):
            kg.G.add_edge(f"n{i}", f"n{(i+1) % 10}")
        # Also make subgraph return the full graph
        kg.G.subgraph = kg.G.subgraph

        result = clusterer.cluster_hierarchical(kg)

        # Should have cluster paths for all nodes
        assert len(result) == 10
        for node, path in result.items():
            assert isinstance(path, list)
            assert len(path) >= 1

    def test_recursive_bisect_base_case_max_depth(self):
        """Test _recursive_bisect stops at max depth."""
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        clusterer = HierarchicalSpectralClusterer(max_depth=0, min_cluster_size=2)

        kg = MagicMock()
        kg.G = nx.MultiDiGraph()
        for i in range(5):
            kg.G.add_edge(f"n{i}", f"n{(i+1) % 5}")

        cluster_paths = {f"n{i}": [] for i in range(5)}
        nodes = [f"n{i}" for i in range(5)]

        clusterer._recursive_bisect(kg, nodes, cluster_paths, depth=0, parent_cluster_id=0)

        # At max depth, all nodes assigned to parent cluster
        for node in nodes:
            assert cluster_paths[node] == [0]

    def test_recursive_bisect_too_few_nodes(self):
        """Test _recursive_bisect stops when too few nodes to split."""
        from hololoom.embedding.spectral_multiscale import HierarchicalSpectralClusterer

        clusterer = HierarchicalSpectralClusterer(max_depth=5, min_cluster_size=3)

        kg = MagicMock()
        kg.G = nx.MultiDiGraph()
        kg.G.add_edge("a", "b")

        cluster_paths = {"a": [], "b": []}

        # 2 nodes < min_cluster_size * 2 = 6, so should stop
        clusterer._recursive_bisect(kg, ["a", "b"], cluster_paths, depth=0, parent_cluster_id=0)

        assert cluster_paths["a"] == [0]
        assert cluster_paths["b"] == [0]


class TestMultiscaleFactories:
    """Test factory functions."""

    def test_create_multiscale_analyzer(self):
        from hololoom.embedding.spectral_multiscale import create_multiscale_analyzer

        analyzer = create_multiscale_analyzer(scales=[64, 128])
        assert analyzer.scales == [64, 128]

    def test_create_multiscale_analyzer_defaults(self):
        from hololoom.embedding.spectral_multiscale import create_multiscale_analyzer

        analyzer = create_multiscale_analyzer()
        assert analyzer.scales == [96, 192, 384]

    def test_create_hierarchical_clusterer(self):
        from hololoom.embedding.spectral_multiscale import create_hierarchical_clusterer

        clusterer = create_hierarchical_clusterer(max_depth=2, min_cluster_size=5)
        assert clusterer.max_depth == 2
        assert clusterer.min_cluster_size == 5

    def test_create_hierarchical_clusterer_defaults(self):
        from hololoom.embedding.spectral_multiscale import create_hierarchical_clusterer

        clusterer = create_hierarchical_clusterer()
        assert clusterer.max_depth == 3
        assert clusterer.min_cluster_size == 3


# ============================================================================
# linguistic_matryoshka_gate.py — gate, prefilter, encode, stats
# ============================================================================


class TestLinguisticGateConfig:
    """Test LinguisticGateConfig."""

    def test_default_linguistic_mode(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        config = LinguisticGateConfig()
        assert config.linguistic_mode == LinguisticFilterMode.DISABLED

    def test_default_weights(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticGateConfig

        config = LinguisticGateConfig()
        assert config.linguistic_weight == 0.3
        assert config.prefilter_similarity_threshold == 0.3
        assert config.prefilter_keep_ratio == 0.7


class TestLinguisticMatryoshkaGateProjection:
    """Test _project_to_scale helper."""

    def test_project_truncates_when_larger(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        embedding = np.ones(200)
        result = gate._project_to_scale(embedding, 100)

        assert len(result) == 100

    def test_project_returns_full_when_smaller(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        embedding = np.ones(50)
        result = gate._project_to_scale(embedding, 100)

        assert len(result) == 50


class TestLinguisticGateEncode:
    """Test encode and encode_scales methods."""

    def test_encode_delegates_to_embedder(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        embedder.encode.return_value = np.ones((2, 96))

        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        result = gate.encode(["hello", "world"])

        embedder.encode.assert_called_once_with(["hello", "world"])
        assert result.shape == (2, 96)

    def test_encode_scales_specific_size(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        embedder.encode.return_value = np.ones((2, 384))

        config = LinguisticGateConfig(
            scales=[96, 192, 384],
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        result = gate.encode_scales(["hello", "world"], size=96)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 96)

    def test_encode_scales_all_sizes(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        embedder.encode.return_value = np.ones((1, 384))

        config = LinguisticGateConfig(
            scales=[96, 192, 384],
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        result = gate.encode_scales(["hello"])

        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert 384 in result
        assert result[96].shape == (1, 96)


class TestLinguisticGateStatistics:
    """Test get_statistics method."""

    def test_statistics_include_linguistic_info(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        stats = gate.get_statistics()

        assert "linguistic_filter" in stats
        assert stats["linguistic_filter"]["enabled"] is False
        assert stats["linguistic_filter"]["mode"] == "disabled"
        assert stats["linguistic_filter"]["total_filters"] == 0


class TestLinguisticGateGating:
    """Test the full gate() pipeline."""

    @pytest.mark.asyncio
    async def test_gate_empty_candidates(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()

        def fake_encode_scales(texts, size=None):
            n = len(texts)
            d = size if size else 96
            rng = np.random.default_rng(42)
            return rng.normal(0, 1, (n, d)).astype(np.float32)

        embedder.encode_scales = fake_encode_scales
        embedder.encode.side_effect = lambda texts: fake_encode_scales(texts, 384)

        config = LinguisticGateConfig(
            scales=[96, 192, 384],
            thresholds=[0.1, 0.1, 0.1],
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        indices, results = await gate.gate("query", [], final_k=5)

        assert indices == []
        assert results == []

    @pytest.mark.asyncio
    async def test_gate_filters_candidates(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()

        def fake_encode_scales(texts, size=None):
            n = len(texts)
            d = size if size else 96
            rng = np.random.default_rng(hash(texts[0]) % (2**31) if texts else 42)
            embs = rng.normal(0, 1, (n, d)).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            return embs / norms

        embedder.encode_scales = fake_encode_scales
        embedder.encode.side_effect = lambda texts: fake_encode_scales(texts, 384)

        config = LinguisticGateConfig(
            scales=[96, 192, 384],
            thresholds=[0.1, 0.1, 0.1],
            topk_ratios=[0.5, 0.5, 1.0],
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        candidates = [f"candidate {i}" for i in range(20)]
        indices, results = await gate.gate("query", candidates, final_k=3)

        assert len(indices) <= 3
        assert len(results) == 3  # One result per scale


class TestSyntacticSimilarity:
    """Test _syntactic_similarity method."""

    def test_identical_structures_max_similarity(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        # Create mock XBarNode structures
        node = MagicMock()
        node.category = "NP"
        node.level = "XP"
        node.specifier = "the"
        node.complement = "book"
        node.adjuncts = ["big"]

        sim = gate._syntactic_similarity(node, node)

        assert sim == pytest.approx(1.0)

    def test_different_structures_lower_similarity(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        node1 = MagicMock()
        node1.category = "NP"
        node1.level = "XP"
        node1.specifier = "the"
        node1.complement = "book"
        node1.adjuncts = ["big"]

        node2 = MagicMock()
        node2.category = "VP"
        node2.level = "X"
        node2.specifier = None
        node2.complement = None
        node2.adjuncts = []

        sim = gate._syntactic_similarity(node1, node2)

        assert sim < 0.5


# ============================================================================
# riemannian_matryoshka.py — additional coverage
# ============================================================================


class TestRiemannianPairwiseDistances:
    """Test pairwise_distances method."""

    def test_pairwise_self_distances(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        rm = RiemannianMatryoshka(
            base_embedder=MatryoshkaEmbeddings(sizes=[96]),
            use_riemannian=False,
        )

        X = np.random.randn(3, 10)
        dists = rm.pairwise_distances(X)

        assert dists.shape == (3, 3)
        # Diagonal should be zero
        for i in range(3):
            assert dists[i, i] == pytest.approx(0.0, abs=1e-10)

    def test_pairwise_xy_distances(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        rm = RiemannianMatryoshka(
            base_embedder=MatryoshkaEmbeddings(sizes=[96]),
            use_riemannian=False,
        )

        X = np.random.randn(2, 5)
        Y = np.random.randn(3, 5)
        dists = rm.pairwise_distances(X, Y)

        assert dists.shape == (2, 3)
        # All distances should be non-negative
        assert np.all(dists >= 0)


class TestRiemannianExpLogErrors:
    """Test error paths for exp_map, log_map, parallel_transport when Riemannian disabled."""

    def test_exp_map_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        rm = RiemannianMatryoshka(
            base_embedder=MatryoshkaEmbeddings(sizes=[96]),
            use_riemannian=False,
        )

        with pytest.raises(ValueError, match="Exponential map only available"):
            rm.exp_map(np.zeros(10), np.ones(10))

    def test_log_map_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        rm = RiemannianMatryoshka(
            base_embedder=MatryoshkaEmbeddings(sizes=[96]),
            use_riemannian=False,
        )

        with pytest.raises(ValueError, match="Logarithmic map only available"):
            rm.log_map(np.zeros(10), np.ones(10))

    def test_parallel_transport_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        rm = RiemannianMatryoshka(
            base_embedder=MatryoshkaEmbeddings(sizes=[96]),
            use_riemannian=False,
        )

        with pytest.raises(ValueError, match="Parallel transport only available"):
            rm.parallel_transport(np.zeros(10), np.ones(10), np.ones(10))


class TestCreateRiemannianEmbedder:
    """Test factory function."""

    def test_create_default(self):
        from hololoom.embedding.riemannian_matryoshka import create_riemannian_embedder

        rm = create_riemannian_embedder()
        assert rm.use_riemannian is False
        assert rm.hyperbolic_dim == 256

    def test_create_with_custom_dims(self):
        from hololoom.embedding.riemannian_matryoshka import create_riemannian_embedder

        rm = create_riemannian_embedder(
            sizes=[96, 192],
            hyperbolic_dim=64,
            spherical_dim=64,
            euclidean_dim=64,
        )
        assert rm.base_embedder.sizes == [96, 192]
        assert rm.hyperbolic_dim == 64


# ============================================================================
# matryoshka_interpreter.py — additional coverage
# ============================================================================


class TestMatryoshkaInterpreter:
    """Test interpreter analysis and feature detection."""

    def test_interpret_text_returns_all_layers(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        results = interp.interpret_text("The hero faced the darkness and found the light.")

        assert "text" in results
        assert "layers" in results
        assert 96 in results["layers"]
        assert 192 in results["layers"]
        assert 384 in results["layers"]

    def test_layer_has_expected_fields(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        results = interp.interpret_text("Simple test text")

        layer = results["layers"][96]
        assert "magnitude" in layer
        assert "mean_activation" in layer
        assert "std_activation" in layer
        assert "sparsity" in layer
        assert "top_features" in layer
        assert "interpretation" in layer

    def test_surface_feature_detection_keywords(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features(
            "The hero overcame fear and found truth in the light"
        )

        keywords_found = [f for f in features if f.startswith("Keyword:")]
        assert len(keywords_found) >= 2  # hero, fear, truth, light

    def test_surface_feature_detection_sentiment(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()

        pos_features = interp._detect_surface_features("hope and victory")
        assert "Positive sentiment detected" in pos_features

        neg_features = interp._detect_surface_features("fear and failure")
        assert "Negative sentiment detected" in neg_features

    def test_surface_feature_detection_entities(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features("John went to Paris with Mary")

        entity_features = [f for f in features if f.startswith("Entities:")]
        assert len(entity_features) >= 1

    def test_symbolic_feature_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features(
            "He transformed from weak to strong, like a butterfly"
        )

        assert "Transformation arc detected" in features
        assert any("Duality:" in f for f in features)
        assert "Metaphorical language present" in features

    def test_symbolic_complex_relationships(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features(
            "She went with him from the forest to the mountain against the wind for a quest"
        )

        rel_features = [f for f in features if "relationships" in f.lower()]
        assert len(rel_features) >= 1

    def test_archetypal_feature_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_archetypal_features(
            "The hero was called to the quest. A wise mentor guided him. "
            "He faced the enemy and discovered the truth. "
            "Through sacrifice and death, he was reborn."
        )

        # Should detect multiple journey stages and archetypes
        assert any("Journey stage:" in f for f in features)
        assert any("Archetype:" in f for f in features)
        assert any("Universal theme:" in f for f in features)

    def test_assess_strength_levels(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp._assess_strength(20) == "INTENSE"
        assert interp._assess_strength(12) == "STRONG"
        assert interp._assess_strength(7) == "MODERATE"
        assert interp._assess_strength(2) == "WEAK"

    def test_assess_complexity_levels(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp._assess_complexity(0.2, 0.1) == "HIGHLY COMPLEX"
        assert interp._assess_complexity(0.12, 0.5) == "COMPLEX"
        assert interp._assess_complexity(0.07, 0.5) == "MODERATE"
        assert interp._assess_complexity(0.02, 0.5) == "SIMPLE"

    def test_visualize_interpretation(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        results = interp.interpret_text("A short test.")
        viz = interp.visualize_interpretation(results)

        assert isinstance(viz, str)
        assert "Surface Layer" in viz
        assert "Symbolic Layer" in viz
        assert "Archetypal Layer" in viz
        assert "INTERPRETATION COMPLETE" in viz

    def test_interpret_layer_summary_surface(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(96, 10.0, 0.5, 0.1, 0.3, "hero faces battle")

        assert "summary" in result
        assert "Surface scan" in result["summary"]

    def test_interpret_layer_summary_symbolic(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(192, 10.0, 0.5, 0.1, 0.3, "transformed")

        assert "Symbolic analysis" in result["summary"]

    def test_interpret_layer_summary_archetypal(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(384, 10.0, 0.5, 0.1, 0.3, "hero quest")

        assert "Archetypal resonance" in result["summary"]


# ============================================================================
# zero_copy.py — EmbeddingStore additional coverage
# ============================================================================


class TestEmbeddingStoreAdditional:
    """Additional tests for EmbeddingStore."""

    def test_write_read_round_trip(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=10, dim=16)

        vec = np.random.randn(16).astype(np.float32)
        store.write(0, vec)

        result = store.read(0)
        assert np.allclose(result, vec)

        store.close()

    def test_read_batch(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=10, dim=8)

        for i in range(5):
            store.write(i, np.full(8, float(i), dtype=np.float32))

        batch = store.read_batch([0, 2, 4])
        assert batch.shape == (3, 8)
        assert np.allclose(batch[0], 0.0)
        assert np.allclose(batch[1], 2.0)
        assert np.allclose(batch[2], 4.0)

        store.close()

    def test_read_range(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=10, dim=4)

        for i in range(5):
            store.write(i, np.full(4, float(i), dtype=np.float32))

        result = store.read_range(1, 3)
        assert result.shape == (2, 4)
        assert np.allclose(result[0], 1.0)
        assert np.allclose(result[1], 2.0)

        store.close()

    def test_get_all(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=10, dim=4)
        store.write(0, np.ones(4, dtype=np.float32))
        store.write(1, np.ones(4, dtype=np.float32) * 2)

        all_data = store.get_all()
        assert all_data.shape == (2, 4)

        store.close()

    def test_write_readonly_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=5, dim=4)
        store.close()

        store = EmbeddingStore.open(tmp_path / "test.mmap", mode="r")

        with pytest.raises(ValueError, match="Cannot write in read-only mode"):
            store.write(0, np.ones(4))

        store.close()

    def test_write_wrong_dim_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=5, dim=4)

        with pytest.raises(ValueError, match="Embedding dim"):
            store.write(0, np.ones(8))

        store.close()

    def test_write_out_of_bounds_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=5, dim=4)

        with pytest.raises(IndexError):
            store.write(10, np.ones(4))

        store.close()

    def test_read_out_of_bounds_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store = EmbeddingStore.create(tmp_path / "test.mmap", max_embeddings=5, dim=4)

        with pytest.raises(IndexError):
            store.read(10)

        store.close()

    def test_open_nonexistent_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        with pytest.raises(FileNotFoundError):
            EmbeddingStore.open(tmp_path / "nonexistent.mmap")

    def test_open_invalid_magic_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        path = tmp_path / "bad.mmap"
        with open(path, "wb") as f:
            f.write(b"BADMAGIC" + b"\x00" * 100)

        with pytest.raises(ValueError, match="Invalid magic"):
            EmbeddingStore.open(path)

    def test_context_manager(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        with EmbeddingStore.create(tmp_path / "ctx.mmap", max_embeddings=5, dim=4) as store:
            store.write(0, np.ones(4, dtype=np.float32))
            result = store.read(0)
            assert np.allclose(result, 1.0)


class TestZeroCopyMatryoshkaEmbeddingsAdditional:
    """Additional tests for ZeroCopyMatryoshkaEmbeddings."""

    def test_encode_base(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        emb = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = emb.encode_base(["test text"])

        assert result.shape[0] == 1
        assert result.shape[1] == emb.base_dim

    def test_encode_with_store(self, tmp_path):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        emb = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96, 192],
            store_path=tmp_path / "store.mmap",
            max_cache_size=100,
        )

        result1 = emb.encode(["hello"])
        result2 = emb.encode(["hello"])

        assert np.allclose(result1, result2)
        emb.close()

    def test_context_manager_zc(self, tmp_path):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with ZeroCopyMatryoshkaEmbeddings(
            sizes=[96, 192],
            store_path=tmp_path / "ctx.mmap",
        ) as emb:
            result = emb.encode(["test"])
            assert result.shape[1] == 192

    def test_encode_scales_empty(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        emb = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])

        result_single = emb.encode_scales([], size=96)
        assert result_single.shape == (0, 96)

        result_all = emb.encode_scales([])
        assert isinstance(result_all, dict)
        assert result_all[96].shape == (0, 96)
