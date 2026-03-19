"""
Unit tests for embedding variant modules (everything except spectral.py).

Modules tested:
1. matryoshka_gate.py — GateConfig, MatryoshkaGate, GateResult, GateStrategy
2. matryoshka_interpreter.py — MatryoshkaInterpreter analysis and feature detection
3. riemannian_matryoshka.py — RiemannianMatryoshka distance and manifold ops
4. zero_copy.py — EmbeddingStore, ZeroCopyMatryoshkaEmbeddings
5. spectral_multiscale.py — MultiScaleSpectralAnalyzer, fuse_multiscale_features
6. linguistic_matryoshka_gate.py — LinguisticGateConfig, LinguisticFilterMode

All tests use synthetic data (small numpy arrays). No real model inference.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


# ============================================================================
# matryoshka_gate.py
# ============================================================================


class TestGateConfig:
    """Test GateConfig defaults and validation."""

    def test_default_scales(self):
        from hololoom.embedding.matryoshka_gate import GateConfig

        config = GateConfig()
        assert config.scales == [96, 192, 384]

    def test_default_thresholds(self):
        from hololoom.embedding.matryoshka_gate import GateConfig

        config = GateConfig()
        assert config.thresholds == [0.6, 0.75, 0.85]
        assert len(config.thresholds) == len(config.scales)

    def test_default_topk_ratios(self):
        from hololoom.embedding.matryoshka_gate import GateConfig

        config = GateConfig()
        assert config.topk_ratios == [0.3, 0.5, 1.0]

    def test_custom_scales(self):
        from hololoom.embedding.matryoshka_gate import GateConfig

        config = GateConfig(scales=[64, 128], thresholds=[0.5, 0.7], topk_ratios=[0.4, 1.0])
        assert config.scales == [64, 128]
        assert config.thresholds == [0.5, 0.7]

    def test_default_strategy_is_progressive(self):
        from hololoom.embedding.matryoshka_gate import GateConfig, GateStrategy

        config = GateConfig()
        assert config.strategy == GateStrategy.PROGRESSIVE

    def test_min_candidates_default(self):
        from hololoom.embedding.matryoshka_gate import GateConfig

        config = GateConfig()
        assert config.min_candidates == 5


class TestGateStrategy:
    """Test GateStrategy enum values."""

    def test_all_strategies_exist(self):
        from hololoom.embedding.matryoshka_gate import GateStrategy

        assert GateStrategy.FIXED_THRESHOLD.value == "fixed_threshold"
        assert GateStrategy.FIXED_TOPK.value == "fixed_topk"
        assert GateStrategy.ADAPTIVE.value == "adaptive"
        assert GateStrategy.PROGRESSIVE.value == "progressive"


class TestMatryoshkaGate:
    """Test MatryoshkaGate gating logic with mock embedder."""

    def _make_mock_embedder(self, dim=96):
        """Create a mock embedder returning deterministic embeddings."""
        embedder = MagicMock()

        def fake_encode_scales(texts, size=None):
            rng = np.random.default_rng(42)
            n = len(texts)
            d = size if size is not None else dim
            embs = rng.normal(0, 1, (n, d)).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            return embs / norms

        embedder.encode_scales = fake_encode_scales
        return embedder

    def test_gate_empty_candidates(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate

        embedder = self._make_mock_embedder()
        gate = MatryoshkaGate(embedder)

        indices, results = gate.gate("query", [], final_k=5)
        assert indices == []
        assert results == []

    def test_gate_returns_indices_and_results(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

        embedder = self._make_mock_embedder()
        config = GateConfig(scales=[96], thresholds=[0.0], topk_ratios=[1.0])
        gate = MatryoshkaGate(embedder, config)

        candidates = [f"candidate {i}" for i in range(10)]
        indices, results = gate.gate("query", candidates, final_k=5)

        assert isinstance(indices, list)
        assert len(indices) <= 5
        assert len(results) == 1  # one scale
        assert results[0].scale == 96

    def test_gate_result_shape(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

        embedder = self._make_mock_embedder()
        config = GateConfig(scales=[96], thresholds=[0.0], topk_ratios=[1.0])
        gate = MatryoshkaGate(embedder, config)

        candidates = [f"c{i}" for i in range(8)]
        _, results = gate.gate("q", candidates, final_k=3)

        assert results[0].candidates_in == 8
        assert results[0].candidates_out <= 8
        assert len(results[0].scores) == 8

    def test_gate_progressive_filtering(self):
        """Multi-scale gating should produce one result per scale."""
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

        embedder = self._make_mock_embedder(dim=384)
        config = GateConfig(
            scales=[96, 192, 384],
            thresholds=[0.0, 0.0, 0.0],
            topk_ratios=[0.5, 0.5, 1.0],
        )
        gate = MatryoshkaGate(embedder, config)

        candidates = [f"text {i}" for i in range(20)]
        indices, results = gate.gate("query", candidates, final_k=5)

        assert len(results) == 3
        assert results[0].scale == 96
        assert results[1].scale == 192
        assert results[2].scale == 384
        # Progressive: each stage inputs the survivors of the previous
        for i in range(1, len(results)):
            assert results[i].candidates_in <= results[i - 1].candidates_in

    def test_gate_statistics(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

        embedder = self._make_mock_embedder()
        config = GateConfig(scales=[96], thresholds=[0.0], topk_ratios=[1.0])
        gate = MatryoshkaGate(embedder, config)

        gate.gate("q", ["a", "b", "c"], final_k=2)
        stats = gate.get_statistics()

        assert stats["total_gates"] == 1
        assert "scale_stats" in stats
        assert 96 in stats["scale_stats"]

    def test_compute_similarity_returns_correct_shape(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate

        embedder = self._make_mock_embedder()
        gate = MatryoshkaGate(embedder)

        query = np.random.randn(96).astype(np.float32)
        candidates = np.random.randn(5, 96).astype(np.float32)
        scores = gate._compute_similarity(query, candidates)

        assert scores.shape == (5,)
        # Cosine similarity should be in [-1, 1]
        assert np.all(scores >= -1.01)
        assert np.all(scores <= 1.01)

    def test_apply_gate_respects_min_candidates(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

        config = GateConfig(scales=[96], thresholds=[0.99], topk_ratios=[0.1], min_candidates=3)
        embedder = self._make_mock_embedder()
        gate = MatryoshkaGate(embedder, config)

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = gate._apply_gate(scores, threshold=0.99, scale_idx=0)

        # Even with high threshold, min_candidates=3 should keep at least 3
        assert np.sum(mask) >= 3

    def test_apply_final_gate_returns_topk(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate

        embedder = self._make_mock_embedder()
        gate = MatryoshkaGate(embedder)

        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        mask = gate._apply_final_gate(scores, final_k=2)

        assert np.sum(mask) == 2
        # Top-2 should be indices 1 and 3
        assert mask[1] and mask[3]

    def test_adaptive_threshold(self):
        from hololoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig, GateStrategy

        config = GateConfig(strategy=GateStrategy.ADAPTIVE)
        embedder = self._make_mock_embedder()
        gate = MatryoshkaGate(embedder, config)

        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        threshold = gate._get_threshold(0, scores)

        # Adaptive: mean + 0.5 * std
        expected = np.mean(scores) + 0.5 * np.std(scores)
        assert abs(threshold - expected) < 1e-6


# ============================================================================
# matryoshka_interpreter.py
# ============================================================================


class TestMatryoshkaInterpreterInit:
    """Test MatryoshkaInterpreter initialization."""

    def test_scales(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp.scales == [96, 192, 384]

    def test_layer_info_keys(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert set(interp.layer_info.keys()) == {96, 192, 384}
        for scale in [96, 192, 384]:
            info = interp.layer_info[scale]
            assert "name" in info
            assert "captures" in info
            assert "depth" in info
            assert isinstance(info["captures"], list)


class TestInterpreterStrengthAssessment:
    """Test _assess_strength categorization."""

    def test_intensity_levels(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp._assess_strength(20.0) == "INTENSE"
        assert interp._assess_strength(12.0) == "STRONG"
        assert interp._assess_strength(7.0) == "MODERATE"
        assert interp._assess_strength(2.0) == "WEAK"

    def test_boundary_values(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp._assess_strength(15.0) == "STRONG"  # exactly boundary
        assert interp._assess_strength(10.0) == "MODERATE"
        assert interp._assess_strength(5.0) == "WEAK"  # boundary: > 5 is MODERATE


class TestInterpreterComplexityAssessment:
    """Test _assess_complexity categorization."""

    def test_complexity_levels(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        assert interp._assess_complexity(0.20, 0.1) == "HIGHLY COMPLEX"
        assert interp._assess_complexity(0.12, 0.5) == "COMPLEX"
        assert interp._assess_complexity(0.08, 0.5) == "MODERATE"
        assert interp._assess_complexity(0.02, 0.5) == "SIMPLE"


class TestSurfaceFeatureDetection:
    """Test _detect_surface_features."""

    def test_keyword_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features("The hero faced darkness and found light")

        keyword_features = [f for f in features if f.startswith("Keyword:")]
        assert len(keyword_features) >= 3  # hero, darkness, light

    def test_entity_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features("Odysseus met Athena near Troy")

        entity_features = [f for f in features if f.startswith("Entities:")]
        assert len(entity_features) >= 1

    def test_sentiment_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features("hope and victory")
        assert "Positive sentiment detected" in features

        features = interp._detect_surface_features("fear and failure")
        assert "Negative sentiment detected" in features

    def test_empty_text(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_surface_features("")
        assert isinstance(features, list)


class TestSymbolicFeatureDetection:
    """Test _detect_symbolic_features."""

    def test_transformation_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features("He became something greater")
        assert "Transformation arc detected" in features

    def test_duality_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features("the light and the dark")
        assert any("Duality: light/dark" in f for f in features)

    def test_metaphor_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features("It seemed like a dream")
        assert "Metaphorical language present" in features

    def test_no_features_in_plain_text(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_symbolic_features("the cat sat")
        # Plain text without transformation or duality markers
        assert not any("Transformation" in f for f in features)


class TestArchetypalFeatureDetection:
    """Test _detect_archetypal_features."""

    def test_journey_stage_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_archetypal_features("The hero was called to battle")

        stage_features = [f for f in features if f.startswith("Journey stage:")]
        assert len(stage_features) >= 1

    def test_archetype_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_archetypal_features("The wise mentor guided the hero")

        archetype_features = [f for f in features if f.startswith("Archetype:")]
        assert len(archetype_features) >= 2  # Mentor + Hero

    def test_universal_theme_detection(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        features = interp._detect_archetypal_features("death and rebirth of the soul")

        theme_features = [f for f in features if f.startswith("Universal theme:")]
        assert any("Death & Rebirth" in f for f in theme_features)


class TestInterpreterInterpretLayer:
    """Test _interpret_layer returns structured output."""

    def test_surface_layer_interpretation(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(96, 10.0, 0.1, 0.1, 0.5, "The hero fought")

        assert "strength" in result
        assert "complexity" in result
        assert "detected_features" in result
        assert "summary" in result
        assert isinstance(result["detected_features"], list)

    def test_symbolic_layer_interpretation(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(192, 10.0, 0.1, 0.12, 0.5, "he became transformed")

        assert "summary" in result
        assert "relational" in result["summary"].lower() or "Symbolic" in result["summary"]

    def test_archetypal_layer_interpretation(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()
        result = interp._interpret_layer(384, 10.0, 0.1, 0.1, 0.5, "quest for wisdom")

        assert "summary" in result
        assert "Archetypal" in result["summary"] or "mythic" in result["summary"]


class TestInterpreterInterpretText:
    """Test interpret_text end-to-end with mocked embeddings."""

    def test_interpret_text_structure(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()

        # Mock the embedder to avoid model loading
        rng = np.random.default_rng(42)

        def fake_encode(text):
            # Return list of embeddings per scale
            return [rng.normal(0, 0.1, s) for s in interp.scales]

        interp.embedder.encode = fake_encode

        result = interp.interpret_text("The hero transformed")

        assert "text" in result
        assert "layers" in result
        assert 96 in result["layers"]
        assert 192 in result["layers"]
        assert 384 in result["layers"]

        for scale in [96, 192, 384]:
            layer = result["layers"][scale]
            assert "magnitude" in layer
            assert "mean_activation" in layer
            assert "std_activation" in layer
            assert "sparsity" in layer
            assert "top_features" in layer
            assert "interpretation" in layer
            assert layer["dimensions"] == scale

    def test_visualize_interpretation_returns_string(self):
        from hololoom.embedding.matryoshka_interpreter import MatryoshkaInterpreter

        interp = MatryoshkaInterpreter()

        # Build a fake result structure
        fake_results = {
            "text": "test text for visualization",
            "layers": {},
        }
        for scale in [96, 192, 384]:
            fake_results["layers"][scale] = {
                "info": interp.layer_info[scale],
                "dimensions": scale,
                "magnitude": 10.0,
                "mean_activation": 0.05,
                "std_activation": 0.1,
                "sparsity": 0.3,
                "top_features": {"indices": [0, 1], "values": [0.5, 0.4]},
                "interpretation": {
                    "strength": "STRONG",
                    "complexity": "MODERATE",
                    "detected_features": ["Feature A"],
                    "summary": "Test summary.",
                },
            }

        output = interp.visualize_interpretation(fake_results)
        assert isinstance(output, str)
        assert "MATRYOSHKA" in output
        assert "STRONG" in output
        assert "Feature A" in output


# ============================================================================
# riemannian_matryoshka.py
# ============================================================================


class TestRiemannianMatryoshkaEuclidean:
    """Test RiemannianMatryoshka in Euclidean fallback mode."""

    def test_init_euclidean_mode(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        assert rm.use_riemannian is False
        assert rm.manifold is None

    def test_euclidean_distance(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        dist = rm.distance(x, y)

        expected = np.sqrt(2.0)
        assert abs(dist - expected) < 1e-6

    def test_distance_to_self_is_zero(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        x = np.array([0.5, 0.3, 0.8])
        assert rm.distance(x, x) < 1e-10

    def test_distance_symmetry(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10)
        y = rng.normal(0, 1, 10)

        assert abs(rm.distance(x, y) - rm.distance(y, x)) < 1e-10

    def test_pairwise_distances_shape(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        X = np.random.randn(4, 10)
        Y = np.random.randn(3, 10)
        dists = rm.pairwise_distances(X, Y)

        assert dists.shape == (4, 3)

    def test_pairwise_self_diagonal_zero(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        X = np.random.randn(5, 8)
        dists = rm.pairwise_distances(X)

        assert dists.shape == (5, 5)
        for i in range(5):
            assert dists[i, i] < 1e-10

    def test_pairwise_symmetry(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        X = np.random.randn(4, 8)
        dists = rm.pairwise_distances(X)

        assert np.allclose(dists, dists.T, atol=1e-10)


class TestRiemannianManifoldOps:
    """Test that manifold ops raise properly when disabled."""

    def test_exp_map_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        with pytest.raises(ValueError, match="Exponential map"):
            rm.exp_map(np.zeros(3), np.ones(3))

    def test_log_map_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        with pytest.raises(ValueError, match="Logarithmic map"):
            rm.log_map(np.zeros(3), np.ones(3))

    def test_parallel_transport_raises_without_riemannian(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        with pytest.raises(ValueError, match="Parallel transport"):
            rm.parallel_transport(np.zeros(3), np.ones(3), np.ones(3))


class TestRiemannianFactory:
    """Test create_riemannian_embedder factory."""

    def test_factory_creates_instance(self):
        from hololoom.embedding.riemannian_matryoshka import create_riemannian_embedder

        rm = create_riemannian_embedder(use_riemannian=False, sizes=[96])
        assert rm is not None
        assert rm.use_riemannian is False

    def test_factory_custom_dims(self):
        from hololoom.embedding.riemannian_matryoshka import create_riemannian_embedder

        rm = create_riemannian_embedder(
            use_riemannian=False,
            sizes=[96],
            hyperbolic_dim=64,
            spherical_dim=64,
            euclidean_dim=64,
        )
        assert rm.hyperbolic_dim == 64
        assert rm.spherical_dim == 64
        assert rm.euclidean_dim == 64


class TestRiemannianEncode:
    """Test that encode delegates to base embedder."""

    def test_encode_delegates(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        result = rm.encode(["test"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == 96

    def test_encode_scales_delegates(self):
        from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        base = MatryoshkaEmbeddings(sizes=[96, 192])
        rm = RiemannianMatryoshka(base_embedder=base, use_riemannian=False)

        result = rm.encode_scales(["test"], size=96)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 96)


# ============================================================================
# zero_copy.py — EmbeddingStore
# ============================================================================


class TestEmbeddingStoreCreateAndWrite:
    """Test EmbeddingStore creation and write/read cycle."""

    def test_create_store(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=10, dim=64)

        assert store.max_embeddings == 10
        assert store.dim == 64
        assert store_path.exists()
        store.close()

    def test_write_and_read(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=10, dim=8)

        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        store.write(0, vec)
        result = store.read(0)

        assert np.allclose(result, vec)
        store.close()

    def test_write_multiple(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=5, dim=4)

        for i in range(5):
            vec = np.full(4, float(i), dtype=np.float32)
            store.write(i, vec)

        for i in range(5):
            result = store.read(i)
            assert np.allclose(result, np.full(4, float(i)))

        store.close()

    def test_write_out_of_bounds_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=3, dim=4)

        with pytest.raises(IndexError):
            store.write(3, np.zeros(4, dtype=np.float32))

        store.close()

    def test_write_wrong_dim_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=3, dim=4)

        with pytest.raises(ValueError):
            store.write(0, np.zeros(8, dtype=np.float32))

        store.close()

    def test_read_out_of_bounds_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=3, dim=4)

        with pytest.raises(IndexError):
            store.read(3)

        store.close()

    def test_read_only_mode_write_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "test.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=3, dim=4)
        store.write(0, np.ones(4, dtype=np.float32))
        store.close()

        store_ro = EmbeddingStore.open(store_path, mode='r')
        with pytest.raises(ValueError):
            store_ro.write(0, np.zeros(4, dtype=np.float32))
        store_ro.close()


class TestEmbeddingStoreOpen:
    """Test EmbeddingStore.open for persistence."""

    def test_reopen_reads_persisted_data(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "persist.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=5, dim=4)
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        store.write(0, vec)
        store.close()

        store2 = EmbeddingStore.open(store_path, mode='r')
        result = store2.read(0)
        assert np.allclose(result, vec)
        store2.close()

    def test_open_nonexistent_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        with pytest.raises(FileNotFoundError):
            EmbeddingStore.open(tmp_path / "nope.mmap")

    def test_open_invalid_magic_raises(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        bad_path = tmp_path / "bad.mmap"
        bad_path.write_bytes(b"NOTMAGIC" + b"\x00" * 100)

        with pytest.raises(ValueError, match="Invalid magic"):
            EmbeddingStore.open(bad_path)


class TestEmbeddingStoreBatchOps:
    """Test batch and range reads."""

    def test_read_batch(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "batch.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=10, dim=4)

        for i in range(5):
            store.write(i, np.full(4, float(i), dtype=np.float32))

        batch = store.read_batch([0, 2, 4])
        assert batch.shape == (3, 4)
        assert np.allclose(batch[0], np.full(4, 0.0))
        assert np.allclose(batch[1], np.full(4, 2.0))
        assert np.allclose(batch[2], np.full(4, 4.0))
        store.close()

    def test_read_range(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "range.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=10, dim=4)

        for i in range(5):
            store.write(i, np.full(4, float(i), dtype=np.float32))

        result = store.read_range(1, 3)
        assert result.shape == (2, 4)
        assert np.allclose(result[0], np.full(4, 1.0))
        assert np.allclose(result[1], np.full(4, 2.0))
        store.close()

    def test_get_all(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "all.mmap"
        store = EmbeddingStore.create(store_path, max_embeddings=10, dim=4)

        for i in range(3):
            store.write(i, np.full(4, float(i), dtype=np.float32))

        result = store.get_all()
        assert result.shape == (3, 4)
        store.close()


class TestEmbeddingStoreContextManager:
    """Test context manager protocol."""

    def test_context_manager(self, tmp_path):
        from hololoom.embedding.zero_copy import EmbeddingStore

        store_path = tmp_path / "ctx.mmap"
        with EmbeddingStore.create(store_path, max_embeddings=5, dim=4) as store:
            store.write(0, np.ones(4, dtype=np.float32))
            result = store.read(0)
            assert np.allclose(result, np.ones(4))


# ============================================================================
# zero_copy.py — ZeroCopyMatryoshkaEmbeddings
# ============================================================================


class TestZeroCopyMatryoshkaEmbeddingsInit:
    """Test ZeroCopyMatryoshkaEmbeddings initialization."""

    def test_default_sizes(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings()
        assert zc.sizes == [96, 192, 384, 768]

    def test_sizes_ascending_validation(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            ZeroCopyMatryoshkaEmbeddings(sizes=[384, 96])

    def test_max_size_validation(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            ZeroCopyMatryoshkaEmbeddings(sizes=[1024])

    def test_lazy_loading(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        assert zc._model_loaded is False


class TestZeroCopyEncoding:
    """Test ZeroCopyMatryoshkaEmbeddings encoding with fallback."""

    def test_encode_scales_empty_input(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])

        # Specific size
        result = zc.encode_scales([], size=96)
        assert result.shape == (0, 96)

        # All sizes
        result = zc.encode_scales([])
        assert isinstance(result, dict)
        assert result[96].shape == (0, 96)
        assert result[192].shape == (0, 192)

    def test_encode_with_fallback(self):
        """Without sentence-transformers, should use deterministic hash fallback."""
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        # Force fallback
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 192

        result = zc.encode_scales(["test text"], size=96)
        assert result.shape == (1, 96)

    def test_encode_fallback_deterministic(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc1 = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        zc1._model = None
        zc1._model_loaded = True
        zc1.base_dim = 96

        zc2 = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        zc2._model = None
        zc2._model_loaded = True
        zc2.base_dim = 96

        r1 = zc1.encode_scales(["same text"], size=96)
        r2 = zc2.encode_scales(["same text"], size=96)

        assert np.allclose(r1, r2)

    def test_encode_fallback_normalized(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 96

        result = zc.encode_scales(["test"], size=96)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    def test_encode_all_scales(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 192

        result = zc.encode_scales(["text"])
        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert result[96].shape == (1, 96)
        assert result[192].shape == (1, 192)

    def test_encode_calls_encode_scales(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 192

        result = zc.encode(["test"])
        assert result.shape == (1, 192)  # max size

    def test_encode_base(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        zc = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 768

        result = zc.encode_base(["test"])
        assert result.shape == (1, 768)

    def test_encode_with_store(self, tmp_path):
        """Test that store path is used for caching."""
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        store_path = tmp_path / "zc_store.mmap"
        zc = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=str(store_path),
            max_cache_size=100,
        )
        zc._model = None
        zc._model_loaded = True
        zc.base_dim = 96

        # Force store creation
        from hololoom.embedding.zero_copy import EmbeddingStore
        zc._store = EmbeddingStore.create(store_path, max_embeddings=100, dim=96)

        result = zc.encode_scales(["cached text"], size=96)
        assert result.shape == (1, 96)

        # Second call should hit cache
        result2 = zc.encode_scales(["cached text"], size=96)
        assert np.allclose(result, result2)

        zc.close()


class TestZeroCopyContextManager:
    """Test context manager for cleanup."""

    def test_context_manager(self):
        from hololoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with ZeroCopyMatryoshkaEmbeddings(sizes=[96]) as zc:
            assert zc is not None


# ============================================================================
# spectral_multiscale.py
# ============================================================================


class TestMultiScaleSpectralAnalyzerInit:
    """Test MultiScaleSpectralAnalyzer initialization."""

    def test_default_scales(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        assert analyzer.scales == [96, 192, 384]

    def test_default_wavelet_scales(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        assert analyzer.wavelet_scales == [0.1, 1.0, 10.0]

    def test_custom_scales(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[64, 128, 256])
        assert analyzer.scales == [64, 128, 256]

    def test_ascending_validation(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        with pytest.raises(AssertionError):
            MultiScaleSpectralAnalyzer(scales=[384, 96])


class TestFuseMultiscaleFeatures:
    """Test fuse_multiscale_features with synthetic spectral data."""

    def test_fuse_empty_results(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        fused = analyzer.fuse_multiscale_features({})

        assert isinstance(fused, np.ndarray)
        assert np.allclose(fused, np.zeros(6))

    def test_fuse_with_eigenvalues(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.1, 0.5, 0.8, 1.2]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        assert isinstance(fused, np.ndarray)
        assert len(fused) >= 4

    def test_fuse_with_wavelets(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.1, 0.5]),
                "wavelets": np.array([0.1, 0.2, 0.3]),
                "diffusion": np.array([]),
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        assert len(fused) > 0

    def test_fuse_with_custom_weights(self):
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96, 192])
        results = {
            96: {
                "eigenvalues": np.array([1.0, 2.0, 3.0, 4.0]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
            },
            192: {
                "eigenvalues": np.array([0.5, 1.0, 1.5, 2.0]),
                "wavelets": np.array([]),
                "diffusion": np.array([]),
            },
        }

        fused_equal = analyzer.fuse_multiscale_features(results)
        fused_weighted = analyzer.fuse_multiscale_features(
            results, fusion_weights={96: 0.9, 192: 0.1}
        )

        # Different weights should give different results
        assert not np.allclose(fused_equal, fused_weighted)

    def test_fuse_eigenvalue_padding(self):
        """Eigenvalues shorter than 4 should be zero-padded."""
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer(scales=[96])
        results = {
            96: {
                "eigenvalues": np.array([0.5, 1.0]),  # only 2
                "wavelets": np.array([]),
                "diffusion": np.array([]),
            }
        }

        fused = analyzer.fuse_multiscale_features(results)
        # Should have padded eigenvalues (4 values total)
        assert len(fused) == 4


class TestAnalyzeSubgraph:
    """Test _analyze_subgraph with empty graph edge case."""

    def test_empty_graph(self):
        import networkx as nx
        from hololoom.embedding.spectral_multiscale import MultiScaleSpectralAnalyzer

        analyzer = MultiScaleSpectralAnalyzer()
        result = analyzer._analyze_subgraph(
            nx.MultiDiGraph(), n_clusters=2, wavelet_scale=1.0
        )

        assert len(result["eigenvalues"]) == 0
        assert len(result["wavelets"]) == 0
        assert len(result["diffusion"]) == 0
        assert result["clusters"] == {}


class TestSpectralMultiscaleFactory:
    """Test factory functions."""

    def test_create_multiscale_analyzer(self):
        from hololoom.embedding.spectral_multiscale import create_multiscale_analyzer

        analyzer = create_multiscale_analyzer(scales=[64, 128])
        assert analyzer.scales == [64, 128]

    def test_create_hierarchical_clusterer(self):
        from hololoom.embedding.spectral_multiscale import create_hierarchical_clusterer

        clusterer = create_hierarchical_clusterer(max_depth=5, min_cluster_size=2)
        assert clusterer.max_depth == 5
        assert clusterer.min_cluster_size == 2


# ============================================================================
# linguistic_matryoshka_gate.py
# ============================================================================


class TestLinguisticFilterMode:
    """Test LinguisticFilterMode enum."""

    def test_all_modes_exist(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticFilterMode

        assert LinguisticFilterMode.DISABLED.value == "disabled"
        assert LinguisticFilterMode.PREFILTER.value == "prefilter"
        assert LinguisticFilterMode.EMBEDDING.value == "embedding"
        assert LinguisticFilterMode.BOTH.value == "both"


class TestLinguisticGateConfig:
    """Test LinguisticGateConfig defaults."""

    def test_default_linguistic_mode(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        config = LinguisticGateConfig()
        assert config.linguistic_mode == LinguisticFilterMode.DISABLED

    def test_default_linguistic_weight(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticGateConfig

        config = LinguisticGateConfig()
        assert config.linguistic_weight == 0.3

    def test_default_prefilter_settings(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticGateConfig

        config = LinguisticGateConfig()
        assert config.prefilter_similarity_threshold == 0.3
        assert config.prefilter_keep_ratio == 0.7

    def test_inherits_gate_config(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticGateConfig

        config = LinguisticGateConfig()
        # Should have GateConfig fields
        assert config.scales == [96, 192, 384]
        assert config.thresholds == [0.6, 0.75, 0.85]

    def test_custom_config(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.BOTH,
            linguistic_weight=0.5,
            prefilter_similarity_threshold=0.5,
        )
        assert config.linguistic_mode == LinguisticFilterMode.BOTH
        assert config.linguistic_weight == 0.5
        assert config.prefilter_similarity_threshold == 0.5


class TestLinguisticGateProjectToScale:
    """Test _project_to_scale helper."""

    def test_truncation(self):
        from hololoom.embedding.linguistic_matryoshka_gate import LinguisticMatryoshkaGate, LinguisticGateConfig

        embedder = MagicMock()
        config = LinguisticGateConfig(
            linguistic_mode=__import__(
                "hololoom.embedding.linguistic_matryoshka_gate", fromlist=["LinguisticFilterMode"]
            ).LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        embedding = np.arange(384, dtype=np.float32)
        projected = gate._project_to_scale(embedding, 96)

        assert len(projected) == 96
        assert np.allclose(projected, np.arange(96, dtype=np.float32))

    def test_no_truncation_when_already_small(self):
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

        embedding = np.arange(96, dtype=np.float32)
        projected = gate._project_to_scale(embedding, 192)

        # Should return unchanged
        assert len(projected) == 96
        assert np.allclose(projected, embedding)


class TestLinguisticGateEncodeScales:
    """Test encode_scales on LinguisticMatryoshkaGate."""

    def test_encode_scales_specific_size(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        embedder.encode.return_value = np.random.randn(2, 384).astype(np.float32)

        config = LinguisticGateConfig(
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        result = gate.encode_scales(["text1", "text2"], size=96)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 96)

    def test_encode_scales_all_sizes(self):
        from hololoom.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode,
        )

        embedder = MagicMock()
        embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

        config = LinguisticGateConfig(
            scales=[96, 192, 384],
            linguistic_mode=LinguisticFilterMode.DISABLED,
            use_compositional_cache=False,
        )
        gate = LinguisticMatryoshkaGate(embedder, config)

        result = gate.encode_scales(["text"])
        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert 384 in result
        assert result[96].shape == (1, 96)
        assert result[384].shape == (1, 384)


class TestLinguisticGateStatistics:
    """Test get_statistics on LinguisticMatryoshkaGate."""

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


class TestSyntacticSimilarity:
    """Test _syntactic_similarity with mock XBarNode."""

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

        # Create mock XBarNode-like objects
        @dataclass
        class FakeXBarNode:
            category: str = "NP"
            level: str = "XP"
            specifier: object = None
            complement: object = None
            adjuncts: list = None

            def __post_init__(self):
                if self.adjuncts is None:
                    self.adjuncts = []

        node1 = FakeXBarNode(category="NP", level="XP", specifier="det", complement="n")
        node2 = FakeXBarNode(category="NP", level="XP", specifier="det", complement="n")

        sim = gate._syntactic_similarity(node1, node2)
        assert sim == pytest.approx(1.0)

    def test_different_categories_lower_similarity(self):
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

        @dataclass
        class FakeXBarNode:
            category: str = "NP"
            level: str = "XP"
            specifier: object = None
            complement: object = None
            adjuncts: list = None

            def __post_init__(self):
                if self.adjuncts is None:
                    self.adjuncts = []

        node1 = FakeXBarNode(category="NP", level="XP")
        node2 = FakeXBarNode(category="VP", level="XP")

        sim = gate._syntactic_similarity(node1, node2)
        # Category mismatch: lose 0.4
        assert sim < 1.0
        assert sim == pytest.approx(0.6)  # 0.2 + 0.1 + 0.15 + 0.15

    def test_similarity_range(self):
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

        @dataclass
        class FakeXBarNode:
            category: str = "NP"
            level: str = "XP"
            specifier: object = None
            complement: object = None
            adjuncts: list = None

            def __post_init__(self):
                if self.adjuncts is None:
                    self.adjuncts = []

        node1 = FakeXBarNode(category="NP", level="X", specifier=None, complement=None)
        node2 = FakeXBarNode(
            category="VP", level="XP", specifier="det", complement="n", adjuncts=["adj1", "adj2"]
        )

        sim = gate._syntactic_similarity(node1, node2)
        assert 0.0 <= sim <= 1.0


# Total test classes: 28
# Total test methods: 80+
# Coverage: matryoshka_gate, matryoshka_interpreter, riemannian_matryoshka,
#           zero_copy, spectral_multiscale, linguistic_matryoshka_gate
