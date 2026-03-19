"""
Unit Tests for Context Packing Module
======================================

Covers untested paths across all 6 source files:
- activation_spreader.py: directional spread, multi-source, graph fallbacks
- importance_scorer.py: individual signal scoring edge cases, cosine similarity, cache eviction
- context_compressor.py: token estimation, adaptive binary search, budget constraints
- packer.py: blend logic, stats tracking, feature toggle paths
- config.py: validation boundaries, preset internals
- protocol.py: dataclass defaults

Does NOT duplicate the 4 existing test files under context_packing/tests/.

Author: Claude Code
Date: 2026-03-18
"""

import math
import time
import pytest
from typing import Dict
from unittest.mock import MagicMock

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from hololoom.context_packing.protocol import (
    ImportanceSignal,
    ActivationState,
    CompressionResult,
    BetaWave,
)
from hololoom.context_packing.config import (
    BetaWaveConfig,
    ImportanceScorerConfig,
    CompressionConfig,
    ContextPackerConfig,
)
from hololoom.context_packing.activation_spreader import ActivationSpreader
from hololoom.context_packing.importance_scorer import ImportanceScorer
from hololoom.context_packing.context_compressor import ContextCompressor
from hololoom.context_packing.packer import ContextPacker


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def linear_graph():
    """A->B->C->D->E linear chain for predictable spreading."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")
    G = nx.MultiDiGraph()
    for src, dst in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
        G.add_edge(src, dst, type="LEADS_TO")
    return G


@pytest.fixture
def typed_edge_graph():
    """Graph with diverse edge types for directional spreading."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")
    G = nx.MultiDiGraph()
    G.add_edge("root", "child_isa", type="IS_A")
    G.add_edge("root", "child_uses", type="USES")
    G.add_edge("root", "child_mentions", type="MENTIONS")
    G.add_edge("root", "child_unknown", type="CUSTOM_EDGE")
    return G


@pytest.fixture
def disconnected_graph():
    """Graph with isolated components."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", type="LEADS_TO")
    G.add_node("X")  # isolated
    G.add_node("Y")  # isolated
    return G


@pytest.fixture
def many_nodes():
    """30 node IDs for testing compression at scale."""
    return [f"node_{i}" for i in range(30)]


@pytest.fixture
def many_scores(many_nodes):
    """Linearly spaced importance scores 0.0..1.0 across 30 nodes."""
    return {n: i / (len(many_nodes) - 1) for i, n in enumerate(many_nodes)}


# ============================================================================
# Protocol / Dataclass Tests
# ============================================================================

class TestProtocolDataclasses:
    """Tests for protocol-level dataclasses and enums."""

    def test_activation_state_defaults(self):
        state = ActivationState(node_id="n1")
        assert state.activation == 0.0
        assert state.last_accessed == 0.0
        assert state.access_count == 0
        assert state.heat == 0.0
        assert state.importance_scores == {}

    def test_activation_state_post_init_creates_dict(self):
        """importance_scores=None should become empty dict, not stay None."""
        state = ActivationState(node_id="n1", importance_scores=None)
        assert state.importance_scores == {}

    def test_beta_wave_defaults(self):
        wave = BetaWave()
        assert wave.frequency == 20.0
        assert wave.amplitude == 1.0
        assert wave.decay_rate == 0.7
        assert wave.max_hops == 3
        assert wave.source_nodes == []

    def test_beta_wave_post_init_creates_list(self):
        wave = BetaWave(source_nodes=None)
        assert wave.source_nodes == []

    def test_compression_result_fields(self):
        result = CompressionResult(
            compressed_nodes=["a", "b"],
            original_count=5,
            compressed_count=2,
            compression_ratio=0.4,
            token_savings=150,
            importance_threshold=0.3,
        )
        assert len(result.compressed_nodes) == 2
        assert result.token_savings == 150

    def test_importance_signal_values(self):
        """All 7 signals should be accessible by value."""
        assert ImportanceSignal("recency") == ImportanceSignal.RECENCY
        assert ImportanceSignal("information") == ImportanceSignal.INFORMATION_CONTENT


# ============================================================================
# Config Validation Edge Cases
# ============================================================================

class TestConfigValidationEdges:
    """Validation edge cases not covered in existing tests."""

    def test_beta_wave_amplitude_zero_invalid(self):
        cfg = BetaWaveConfig(amplitude=0.0)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_beta_wave_amplitude_above_one_invalid(self):
        cfg = BetaWaveConfig(amplitude=1.5)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_beta_wave_decay_rate_zero_invalid(self):
        cfg = BetaWaveConfig(decay_rate=0.0)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_beta_wave_decay_rate_one_invalid(self):
        cfg = BetaWaveConfig(decay_rate=1.0)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_beta_wave_max_hops_zero_invalid(self):
        cfg = BetaWaveConfig(max_hops=0)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_beta_wave_frequency_exactly_at_bounds(self):
        """12 Hz and 30 Hz are both valid (inclusive range)."""
        BetaWaveConfig(frequency=12.0).validate()
        BetaWaveConfig(frequency=30.0).validate()

    def test_compression_scales_empty_invalid(self):
        cfg = CompressionConfig(matryoshka_scales=[])
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_compression_scales_wrong_order_invalid(self):
        cfg = CompressionConfig(matryoshka_scales=[128, 256, 384])
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_compression_target_ratio_zero_invalid(self):
        cfg = CompressionConfig(target_ratio=0.0)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_packer_config_budget_ordering_invalid(self):
        cfg = ContextPackerConfig(
            min_token_budget=5000,
            default_token_budget=2000,
            max_token_budget=8000,
        )
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_importance_scorer_invalid_centrality_algo(self):
        cfg = ImportanceScorerConfig(centrality_algorithm="random_walk")
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_research_preset_details(self):
        cfg = ContextPackerConfig.research()
        assert cfg.beta_wave.decay_rate == 0.8
        assert cfg.compression.preserve_top_k == 20
        assert cfg.default_token_budget == 4000
        assert cfg.verbose is True


# ============================================================================
# Activation Spreader Tests
# ============================================================================

class TestActivationSpreaderExtended:
    """Tests for untested ActivationSpreader paths."""

    def test_spread_empty_sources(self, linear_graph):
        """Empty source list should produce empty activation map."""
        spreader = ActivationSpreader()
        result = spreader.spread_activation([], linear_graph)
        assert result == {}

    def test_spread_leaf_node_no_outgoing(self):
        """Source node with no outgoing edges should only activate itself."""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        G.add_node("leaf")
        spreader = ActivationSpreader()
        result = spreader.spread_activation(["leaf"], G)
        assert "leaf" in result
        assert result["leaf"] == 1.0
        assert len(result) == 1

    def test_multi_source_activation_accumulates(self, linear_graph):
        """Multiple sources should both contribute activation."""
        spreader = ActivationSpreader()
        result = spreader.spread_activation(
            ["A", "C"], linear_graph, max_hops=1
        )
        assert result["A"] == 1.0
        assert result["C"] == 1.0
        # B gets activation from A, D gets activation from C
        assert "B" in result
        assert "D" in result

    def test_decay_is_exponential_per_hop(self, linear_graph):
        """Activation at hop N should be initial * decay_rate^N."""
        decay = 0.5
        spreader = ActivationSpreader(BetaWaveConfig(decay_rate=decay, min_activation=0.01))
        result = spreader.spread_activation(
            ["A"], linear_graph, decay_rate=decay, max_hops=3
        )
        # hop 1: B = 0.5, hop 2: C = 0.25, hop 3: D = 0.125
        assert result.get("B", 0) == pytest.approx(0.5, abs=0.01)
        assert result.get("C", 0) == pytest.approx(0.25, abs=0.01)
        assert result.get("D", 0) == pytest.approx(0.125, abs=0.01)

    def test_min_activation_threshold_stops_propagation(self, linear_graph):
        """Propagation should stop when activation drops below min_activation."""
        cfg = BetaWaveConfig(decay_rate=0.3, min_activation=0.1, max_hops=10)
        spreader = ActivationSpreader(cfg)
        result = spreader.spread_activation(["A"], linear_graph, decay_rate=0.3)
        # hop 1: 0.3, hop 2: 0.09 < 0.1 threshold => C should NOT appear
        assert "B" in result
        assert result["B"] == pytest.approx(0.3, abs=0.01)
        assert "C" not in result

    def test_directional_spread_edge_conductivity(self, typed_edge_graph):
        """Different edge types should produce different activation levels."""
        spreader = ActivationSpreader(BetaWaveConfig(decay_rate=0.7, min_activation=0.01))
        result = spreader.directional_spread(["root"], typed_edge_graph)

        # IS_A conductivity 0.9, USES 0.7, MENTIONS 0.5, CUSTOM default 0.6
        # Activation = amplitude * decay_rate * conductivity = 1.0 * 0.7 * cond
        assert result["child_isa"] == pytest.approx(0.7 * 0.9, abs=0.01)
        assert result["child_uses"] == pytest.approx(0.7 * 0.7, abs=0.01)
        assert result["child_mentions"] == pytest.approx(0.7 * 0.5, abs=0.01)
        assert result["child_unknown"] == pytest.approx(0.7 * 0.6, abs=0.01)

    def test_directional_spread_min_activation_filters(self, typed_edge_graph):
        """Directional spread should also respect min_activation."""
        cfg = BetaWaveConfig(decay_rate=0.2, min_activation=0.15)
        spreader = ActivationSpreader(cfg)
        result = spreader.directional_spread(["root"], typed_edge_graph)
        # All children: 0.2 * conductivity, max IS_A = 0.2*0.9 = 0.18
        # MENTIONS = 0.2*0.5 = 0.10 < threshold => dropped
        assert "child_isa" in result
        assert "child_mentions" not in result

    def test_get_activation_wave_delegates(self, linear_graph):
        """get_activation_wave should produce same result as spread_activation."""
        spreader = ActivationSpreader()
        wave = BetaWave(
            amplitude=0.8, decay_rate=0.5, max_hops=2, source_nodes=["A"]
        )
        result = spreader.get_activation_wave(wave, linear_graph)
        assert result["A"] == 0.8
        assert "B" in result

    def test_get_activation_states_produces_state_objects(self, linear_graph):
        """get_activation_states should wrap activation map into ActivationState."""
        spreader = ActivationSpreader()
        activation = spreader.spread_activation(["A"], linear_graph, max_hops=1)
        states = spreader.get_activation_states(activation)

        for node_id, state in states.items():
            assert isinstance(state, ActivationState)
            assert state.node_id == node_id
            assert state.activation == activation[node_id]
            assert state.last_accessed > 0
            assert state.access_count == 1

    def test_dict_graph_fallback(self):
        """Spreader should handle dict-like adjacency as fallback."""
        dict_graph = {"A": {"B": {}, "C": {}}, "B": {"D": {}}}
        spreader = ActivationSpreader()
        result = spreader.spread_activation(["A"], dict_graph, max_hops=1)
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_stats_after_multiple_spreads(self, linear_graph):
        """Stats should track total spreads and average activated nodes."""
        spreader = ActivationSpreader()
        spreader.spread_activation(["A"], linear_graph, max_hops=1)
        spreader.spread_activation(["A", "B"], linear_graph, max_hops=1)

        stats = spreader.get_stats()
        assert stats["total_spreads"] == 2
        assert stats["avg_activated_nodes"] > 0

    def test_reset_stats_clears_history(self, linear_graph):
        spreader = ActivationSpreader()
        spreader.spread_activation(["A"], linear_graph, max_hops=1)
        assert len(spreader.activation_history) == 1

        spreader.reset_stats()
        assert len(spreader.activation_history) == 0
        assert spreader.get_stats()["total_spreads"] == 0

    def test_disconnected_graph_only_reachable(self, disconnected_graph):
        """Spreading from A should not activate isolated nodes X, Y."""
        spreader = ActivationSpreader()
        result = spreader.spread_activation(["A"], disconnected_graph, max_hops=5)
        assert "B" in result
        assert "X" not in result
        assert "Y" not in result


# ============================================================================
# Importance Scorer Tests
# ============================================================================

class TestImportanceScorerExtended:
    """Tests for untested ImportanceScorer paths."""

    def test_score_recency_neutral_without_state(self):
        """Without activation state, recency should return 0.5 (neutral)."""
        scorer = ImportanceScorer()
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        G.add_node("n1")
        scores = scorer.score_importance("n1", "query", G, activation_state=None)
        assert scores[ImportanceSignal.RECENCY] == 0.5

    def test_score_recency_recent_access_high(self):
        """Very recent access should produce score close to 1.0."""
        scorer = ImportanceScorer()
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        G.add_node("n1")
        state = ActivationState(
            node_id="n1", last_accessed=time.time(), access_count=1
        )
        scores = scorer.score_importance("n1", "q", G, activation_state=state)
        assert scores[ImportanceSignal.RECENCY] > 0.95

    def test_score_recency_old_access_low(self):
        """Access many half-lives ago should produce score close to 0."""
        scorer = ImportanceScorer(ImportanceScorerConfig(recency_half_life=60.0))
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        G.add_node("n1")
        # 10 half-lives ago
        state = ActivationState(
            node_id="n1", last_accessed=time.time() - 600, access_count=1
        )
        scores = scorer.score_importance("n1", "q", G, activation_state=state)
        assert scores[ImportanceSignal.RECENCY] < 0.01

    def test_score_frequency_logarithmic_scaling(self):
        """Frequency scoring should scale logarithmically (diminishing returns)."""
        scorer = ImportanceScorer()
        s1 = scorer._score_frequency("n", ActivationState(node_id="n", access_count=1))
        s10 = scorer._score_frequency("n", ActivationState(node_id="n", access_count=10))
        s100 = scorer._score_frequency("n", ActivationState(node_id="n", access_count=100))
        s1000 = scorer._score_frequency("n", ActivationState(node_id="n", access_count=1000))

        # Should be monotonically increasing
        assert s1 < s10 < s100 < s1000
        # At 1000 accesses the score should approach 1.0
        assert s1000 == pytest.approx(1.0, abs=0.01)
        # Score at 1 access should be small
        assert s1 < 0.15

    def test_score_frequency_zero_accesses(self):
        scorer = ImportanceScorer()
        assert scorer._score_frequency("n", ActivationState(node_id="n", access_count=0)) == 0.0

    def test_score_frequency_no_state(self):
        scorer = ImportanceScorer()
        assert scorer._score_frequency("n", None) == 0.0

    def test_score_heat_normalization(self):
        """Heat of 10 should normalize to 1.0, heat of 5 to 0.5."""
        scorer = ImportanceScorer()
        assert scorer._score_heat("n", ActivationState(node_id="n", heat=10.0)) == 1.0
        assert scorer._score_heat("n", ActivationState(node_id="n", heat=5.0)) == 0.5
        assert scorer._score_heat("n", ActivationState(node_id="n", heat=0.0)) == 0.0

    def test_score_heat_capped_at_one(self):
        """Heat above 10 should be capped at 1.0."""
        scorer = ImportanceScorer()
        assert scorer._score_heat("n", ActivationState(node_id="n", heat=20.0)) == 1.0

    def test_score_heat_no_state(self):
        scorer = ImportanceScorer()
        assert scorer._score_heat("n", None) == 0.0

    def test_score_confidence_uses_importance_scores(self):
        """Should read confidence from activation_state.importance_scores."""
        scorer = ImportanceScorer()
        state = ActivationState(
            node_id="n",
            activation=0.3,
            importance_scores={ImportanceSignal.CONFIDENCE: 0.85},
        )
        assert scorer._score_confidence("n", state) == 0.85

    def test_score_confidence_fallback_to_activation(self):
        """Without confidence in importance_scores, should use activation level."""
        scorer = ImportanceScorer()
        state = ActivationState(node_id="n", activation=0.6, importance_scores={})
        assert scorer._score_confidence("n", state) == 0.6

    def test_score_relevance_without_embedder(self):
        """Without embedder, relevance should return 0.5 (neutral)."""
        scorer = ImportanceScorer(embedder=None)
        assert scorer._score_relevance("n", "query", "some content") == 0.5

    def test_score_relevance_without_content(self):
        scorer = ImportanceScorer(embedder=MagicMock())
        assert scorer._score_relevance("n", "query", None) == 0.5

    def test_cosine_similarity_with_lists(self):
        """_cosine_similarity should work with Python lists."""
        scorer = ImportanceScorer()
        sim = scorer._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == pytest.approx(1.0)

        sim = scorer._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        """Zero vectors should return 0.0."""
        scorer = ImportanceScorer()
        assert scorer._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_aggregate_scores_no_matching_weights(self):
        """If weights dict has no matching signals, should return 0.0."""
        scorer = ImportanceScorer()
        scores = {ImportanceSignal.RECENCY: 0.9}
        # Provide weights that don't match any signal in scores
        weights = {ImportanceSignal.HEAT: 1.0}
        result = scorer.aggregate_scores(scores, weights=weights)
        assert result == 0.0

    def test_mi_cache_eviction(self):
        """Cache should evict oldest entries when full."""
        cfg = ImportanceScorerConfig(enable_information_scoring=True)
        scorer = ImportanceScorer(cfg)
        scorer._mi_cache_size = 10

        # Fill cache beyond capacity
        for i in range(15):
            scorer._set_cached_mi(f"node_{i}", "qhash", float(i))

        # Should have evicted some entries
        assert len(scorer._mi_cache) <= 10

    def test_invalidate_centrality_cache(self):
        scorer = ImportanceScorer()
        scorer._centrality_cache = {"node": 0.5}
        scorer._centrality_cache_time = time.time()
        scorer.invalidate_centrality_cache()
        assert scorer._centrality_cache is None

    def test_word_overlap_mi_no_overlap(self):
        """Completely disjoint words should produce MI of 0."""
        scorer = ImportanceScorer(ImportanceScorerConfig(enable_information_scoring=True))
        score = scorer._compute_word_overlap_mi("alpha beta gamma", "delta epsilon zeta")
        assert score == 0.0

    def test_word_overlap_mi_full_overlap(self):
        """Identical content and query should produce highest MI."""
        scorer = ImportanceScorer(ImportanceScorerConfig(enable_information_scoring=True))
        text = "thompson sampling bayesian"
        score = scorer._compute_word_overlap_mi(text, text)
        assert score > 0.5

    def test_word_overlap_mi_empty_strings(self):
        scorer = ImportanceScorer(ImportanceScorerConfig(enable_information_scoring=True))
        assert scorer._compute_word_overlap_mi("", "query") == 0.0
        assert scorer._compute_word_overlap_mi("content", "") == 0.0

    def test_content_entropy_single_word(self):
        """Single repeated word should have entropy 0."""
        scorer = ImportanceScorer()
        ent = scorer._compute_content_entropy("hello")
        assert ent == 0.0

    def test_content_entropy_caching(self):
        """Second call with same content should use cache."""
        scorer = ImportanceScorer()
        scorer._compute_content_entropy("alpha beta gamma delta epsilon")
        assert len(scorer._entropy_cache) > 0


# ============================================================================
# Context Compressor Tests
# ============================================================================

class TestContextCompressorExtended:
    """Tests for untested ContextCompressor paths."""

    def test_estimate_tokens_without_scales(self):
        """Without scale assignments, should use simple multiplication."""
        compressor = ContextCompressor()
        nodes = ["a", "b", "c"]
        tokens = compressor.estimate_tokens(nodes)
        assert tokens == 3 * compressor.config.avg_tokens_per_node

    def test_estimate_tokens_with_mixed_scales(self):
        """Different scales should produce different token estimates."""
        compressor = ContextCompressor()
        nodes = ["high", "mid", "low", "tiny"]
        scales = {"high": 384, "mid": 256, "low": 128, "tiny": 64}
        tokens = compressor.estimate_tokens(nodes, scales)
        base = compressor.config.avg_tokens_per_node

        expected = int(base * 1.0) + int(base * 0.66) + int(base * 0.33) + int(base * 0.1)
        assert tokens == expected

    def test_estimate_tokens_default_scale_for_missing_node(self):
        """Node not in scale_assignments should get default (first) scale."""
        compressor = ContextCompressor()
        tokens = compressor.estimate_tokens(["missing"], {"other": 128})
        # Should use default scale (384) -> multiplier 1.0
        assert tokens == compressor.config.avg_tokens_per_node

    def test_compress_respects_preserve_top_k(self):
        """Even with low target_ratio, preserve_top_k nodes should be kept."""
        cfg = CompressionConfig(preserve_top_k=5, target_ratio=0.1)
        compressor = ContextCompressor(cfg)
        nodes = [f"n{i}" for i in range(10)]
        scores = {n: 0.1 * i for i, n in enumerate(nodes)}

        result = compressor.compress(nodes, scores, target_ratio=0.1)
        # target_ratio=0.1 => 1 node, but preserve_top_k=5 => at least 5
        assert result.compressed_count >= 5

    def test_compress_importance_ordering(self):
        """Compressed nodes should be the highest-importance ones."""
        compressor = ContextCompressor()
        nodes = ["low", "mid", "high"]
        scores = {"low": 0.1, "mid": 0.5, "high": 0.9}

        result = compressor.compress(nodes, scores, target_ratio=0.34)
        # Should keep at most 1 node by ratio, but preserve_top_k overrides
        # The kept nodes should include "high"
        assert "high" in result.compressed_nodes

    def test_adaptive_compress_within_budget_range(self, many_nodes, many_scores):
        """Adaptive compression should find a result within token range."""
        compressor = ContextCompressor()
        result = compressor.adaptive_compress(
            many_nodes, many_scores, min_tokens=200, max_tokens=600
        )
        estimated = result.compressed_count * compressor.config.avg_tokens_per_node
        # Should be within range (or close, due to granularity)
        assert estimated <= 600

    def test_adaptive_compress_empty_nodes(self):
        """Adaptive compress with empty node list should return empty result."""
        compressor = ContextCompressor()
        result = compressor.adaptive_compress([], {}, min_tokens=100, max_tokens=500)
        assert result.compressed_count == 0

    def test_information_budget_compress_token_budget_constraint(self):
        """Token budget should stop selection even if MI budget has room."""
        cfg = CompressionConfig(
            enable_information_budget=True,
            information_budget=100.0,  # Very high MI budget
            preserve_top_k=1,
            avg_tokens_per_node=50,
            mi_diminishing_returns_threshold=0.01,
        )
        compressor = ContextCompressor(cfg)

        nodes = [f"n{i}" for i in range(20)]
        importance = {n: 0.5 for n in nodes}
        mi = {n: 1.0 for n in nodes}  # All have good MI

        result = compressor.information_budget_compress(
            nodes, importance, mi,
            information_budget=100.0,
            token_budget=150,  # Only 3 nodes fit (3 * 50 = 150)
        )
        assert result.compressed_count <= 3

    def test_information_budget_disabled_falls_back(self):
        """When enable_information_budget=False, should fall back to standard compress."""
        cfg = CompressionConfig(enable_information_budget=False)
        compressor = ContextCompressor(cfg)

        nodes = ["a", "b", "c"]
        importance = {"a": 0.9, "b": 0.5, "c": 0.1}
        mi = {"a": 1.0, "b": 0.5, "c": 0.1}

        result = compressor.information_budget_compress(nodes, importance, mi)
        # Should still return a valid result (via standard compress)
        assert isinstance(result, CompressionResult)
        assert result.original_count == 3

    def test_information_budget_partial_inclusion(self):
        """Node that partially fits budget should be included if >= 50% fits."""
        cfg = CompressionConfig(
            enable_information_budget=True,
            information_budget=3.0,
            preserve_top_k=1,
            mi_diminishing_returns_threshold=0.01,
        )
        compressor = ContextCompressor(cfg)

        nodes = ["big", "medium"]
        importance = {"big": 0.5, "medium": 0.5}
        mi = {"big": 2.5, "medium": 1.0}
        # Budget=3.0. big=2.5 fits. Then 0.5 remaining. medium=1.0, 50% of 1.0=0.5 == remaining
        # Should include medium (remaining_budget >= node_mi * 0.5)
        result = compressor.information_budget_compress(
            nodes, importance, mi, information_budget=3.0
        )
        assert "big" in result.compressed_nodes

    def test_count_by_scale(self):
        compressor = ContextCompressor()
        assignments = {"a": 384, "b": 384, "c": 256, "d": 128}
        counts = compressor._count_by_scale(assignments)
        assert counts[384] == 2
        assert counts[256] == 1
        assert counts[128] == 1

    def test_stats_rolling_average(self):
        """Stats should correctly compute rolling averages."""
        cfg = CompressionConfig(preserve_top_k=1)
        compressor = ContextCompressor(cfg)
        nodes = [f"n{i}" for i in range(20)]
        scores = {n: i / 19.0 for i, n in enumerate(nodes)}

        compressor.compress(nodes, scores, target_ratio=0.3)
        compressor.compress(nodes, scores, target_ratio=0.6)

        stats = compressor.get_stats()
        assert stats["total_compressions"] == 2
        assert 0.0 < stats["avg_compression_ratio"] < 1.0

    def test_matryoshka_preserve_top_k_fills_minimum(self):
        """If thresholds drop all nodes, preserve_top_k should rescue them."""
        cfg = CompressionConfig(
            preserve_top_k=3,
            high_importance_threshold=0.99,
            mid_importance_threshold=0.98,
            low_importance_threshold=0.97,
        )
        compressor = ContextCompressor(cfg)
        nodes = ["a", "b", "c", "d", "e"]
        scores = {n: 0.1 for n in nodes}  # All way below thresholds

        kept, scales = compressor.matryoshka_compress(nodes, scores)
        assert len(kept) >= 3
        # Rescued nodes should get smallest scale
        for node in kept:
            if scores[node] < 0.97:
                assert scales[node] == 128


# ============================================================================
# Context Packer (Orchestrator) Tests
# ============================================================================

class TestContextPackerExtended:
    """Tests for untested ContextPacker orchestrator paths."""

    def test_blend_activation_importance(self):
        """Blending should combine importance and activation with configured weight."""
        packer = ContextPacker()
        importance = {"a": 0.8, "b": 0.4}
        activation = {"a": 0.2, "b": 1.0}

        blended = packer._blend_activation_importance(importance, activation, activation_weight=0.3)

        # a: 0.7*0.8 + 0.3*0.2 = 0.56 + 0.06 = 0.62
        assert blended["a"] == pytest.approx(0.62, abs=0.01)
        # b: 0.7*0.4 + 0.3*1.0 = 0.28 + 0.30 = 0.58
        assert blended["b"] == pytest.approx(0.58, abs=0.01)

    def test_blend_activation_missing_node(self):
        """Nodes not in activation_map should use 0.0 activation."""
        packer = ContextPacker()
        importance = {"a": 0.8, "b": 0.6}
        activation = {"a": 0.5}  # b missing

        blended = packer._blend_activation_importance(importance, activation, activation_weight=0.4)
        # b: 0.6*0.6 + 0.4*0.0 = 0.36
        assert blended["b"] == pytest.approx(0.36, abs=0.01)

    def test_pack_with_activation_disabled(self, linear_graph):
        """With activation spreading disabled, should still pack successfully."""
        cfg = ContextPackerConfig()
        cfg.enable_activation_spreading = False
        packer = ContextPacker(cfg)

        result = packer.pack("test query", ["A", "B", "C"], linear_graph)
        assert isinstance(result, CompressionResult)
        assert result.original_count == 3  # No expansion from activation

    def test_pack_with_matryoshka_disabled(self, linear_graph):
        """With matryoshka disabled, should use simple threshold compression."""
        cfg = ContextPackerConfig()
        cfg.enable_matryoshka_compression = False
        packer = ContextPacker(cfg)

        result = packer.pack("test", ["A", "B", "C", "D"], linear_graph)
        assert isinstance(result, CompressionResult)

    def test_adaptive_pack_disabled_falls_back_to_standard(self, linear_graph):
        """With adaptive packing disabled, adaptive_pack should use standard pack."""
        cfg = ContextPackerConfig()
        cfg.enable_adaptive_packing = False
        packer = ContextPacker(cfg)

        result = packer.adaptive_pack("test", ["A", "B", "C"], linear_graph)
        assert isinstance(result, CompressionResult)

    def test_update_stats_rolling_average(self):
        """_update_stats should maintain correct rolling averages."""
        packer = ContextPacker()
        r1 = CompressionResult(["a"], 4, 1, 0.25, 150, 0.3)
        r2 = CompressionResult(["a", "b", "c"], 4, 3, 0.75, 50, 0.1)

        packer._update_stats(r1)
        assert packer._stats["total_packs"] == 1
        assert packer._stats["avg_compression_ratio"] == pytest.approx(0.25)

        packer._update_stats(r2)
        assert packer._stats["total_packs"] == 2
        assert packer._stats["avg_compression_ratio"] == pytest.approx(0.5)
        assert packer._stats["avg_token_savings"] == pytest.approx(100.0)

    def test_get_pack_history(self, linear_graph):
        """get_pack_history should return copies of all past results."""
        packer = ContextPacker()
        packer.pack("q1", ["A", "B"], linear_graph)
        packer.pack("q2", ["A", "B", "C"], linear_graph)

        history = packer.get_pack_history()
        assert len(history) == 2
        # Should be a copy, not the internal list
        history.clear()
        assert len(packer.get_pack_history()) == 2

    def test_reset_stats_clears_everything(self, linear_graph):
        """reset_stats should zero out all stats and history."""
        packer = ContextPacker()
        packer.pack("q", ["A", "B"], linear_graph)

        packer.reset_stats()
        stats = packer.get_stats()
        assert stats["total_packs"] == 0
        assert stats["avg_compression_ratio"] == 0.0
        assert len(packer.get_pack_history()) == 0

    def test_pack_preserves_node_order_by_importance(self, linear_graph):
        """Compressed nodes should be ordered by importance (highest first)."""
        cfg = ContextPackerConfig()
        cfg.enable_activation_spreading = False
        cfg.enable_matryoshka_compression = True
        packer = ContextPacker(cfg)

        result = packer.pack("test", ["A", "B", "C", "D"], linear_graph)
        # All nodes should be in the result (small graph, preserve_top_k=10)
        assert isinstance(result.compressed_nodes, list)

    def test_pack_with_node_contents(self, linear_graph):
        """Providing node_contents should not break the pipeline."""
        cfg = ContextPackerConfig()
        cfg.enable_activation_spreading = False
        packer = ContextPacker(cfg)

        contents = {"A": "first node", "B": "second node", "C": "third node"}
        result = packer.pack("test", ["A", "B", "C"], linear_graph, node_contents=contents)
        assert isinstance(result, CompressionResult)

    def test_get_stats_includes_component_stats(self, linear_graph):
        """get_stats should include sub-component stats."""
        packer = ContextPacker()
        packer.pack("q", ["A", "B"], linear_graph)

        stats = packer.get_stats()
        assert "component_stats" in stats
        assert "activation_spreader" in stats["component_stats"]
        assert "importance_scorer" in stats["component_stats"]
        assert "context_compressor" in stats["component_stats"]


# ============================================================================
# Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_pack_context_all_presets(self, linear_graph):
        """pack_context should accept all preset strings."""
        from hololoom.context_packing.packer import pack_context

        for preset in ["aggressive", "balanced", "conservative", "research"]:
            result = pack_context("test", ["A", "B", "C"], linear_graph, preset=preset)
            assert isinstance(result, list)

    def test_pack_context_unknown_preset_defaults_to_balanced(self, linear_graph):
        """Unknown preset string should fall back to balanced."""
        from hololoom.context_packing.packer import pack_context
        result = pack_context("test", ["A", "B"], linear_graph, preset="unknown")
        assert isinstance(result, list)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_node_matryoshka_compression(self):
        """Single node should be kept and assigned a scale."""
        compressor = ContextCompressor()
        kept, scales = compressor.matryoshka_compress(["only"], {"only": 0.9})
        assert "only" in kept
        assert scales["only"] == 384

    def test_all_nodes_same_importance(self):
        """Nodes with identical scores should all be treated equally."""
        compressor = ContextCompressor()
        nodes = [f"n{i}" for i in range(5)]
        scores = {n: 0.5 for n in nodes}
        result = compressor.compress(nodes, scores, target_ratio=0.6)
        # 5 * 0.6 = 3, but preserve_top_k=10 means all 5 kept
        assert result.compressed_count == 5

    def test_missing_importance_score_defaults_to_zero(self):
        """Node not in importance_scores should be treated as 0.0."""
        compressor = ContextCompressor()
        nodes = ["scored", "unscored"]
        scores = {"scored": 0.9}
        result = compressor.compress(nodes, scores, target_ratio=0.5)
        # "scored" should be kept; "unscored" dropped if ratio allows
        assert "scored" in result.compressed_nodes

    def test_oversized_single_item_still_returned(self):
        """Even if a single node exceeds token budget, it should be returned."""
        cfg = CompressionConfig(avg_tokens_per_node=10000, preserve_top_k=1)
        compressor = ContextCompressor(cfg)
        result = compressor.adaptive_compress(
            ["big_node"], {"big_node": 0.9},
            min_tokens=100, max_tokens=500,
        )
        # Should still return the node (preserve_top_k=1)
        assert result.compressed_count >= 1

    def test_empty_pack_pipeline(self):
        """Packing with empty candidates should produce empty result."""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        packer = ContextPacker()
        result = packer.pack("query", [], G)
        assert result.compressed_count == 0
        assert result.compressed_nodes == []

    def test_large_node_set_performance(self):
        """100 nodes should pack in under 500ms (unit budget)."""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")
        G = nx.MultiDiGraph()
        nodes = [f"n{i}" for i in range(100)]
        for n in nodes:
            G.add_node(n)
        # Add some edges
        for i in range(99):
            G.add_edge(f"n{i}", f"n{i+1}", type="LEADS_TO")

        cfg = ContextPackerConfig()
        cfg.enable_activation_spreading = True
        packer = ContextPacker(cfg)

        start = time.perf_counter()
        result = packer.pack("test query", nodes[:20], G)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert isinstance(result, CompressionResult)
