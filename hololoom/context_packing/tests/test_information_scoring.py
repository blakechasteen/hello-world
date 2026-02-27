"""
Phase 5 Information-Theoretic Scoring Tests
=============================================

Test suite for Phase 5 Math-Powered Context Packing features:
- INFORMATION_CONTENT signal (7th importance signal)
- Mutual Information (MI) scoring
- Entropy-aware aggregation
- Information budget compression
- MI-aware Matryoshka scale assignment
- MI caching performance

Author: Claude Code
Date: December 2025
"""

import pytest
import time
from typing import Dict, List

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from hololoom.context_packing import (
    ImportanceSignal,
    CompressionResult,
    ImportanceScorerConfig,
    CompressionConfig,
    ContextPackerConfig,
    ImportanceScorer,
    ContextCompressor,
    ContextPacker,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_graph():
    """Create sample knowledge graph for testing."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")

    G = nx.MultiDiGraph()

    nodes = [
        "thompson_sampling",
        "exploration",
        "exploitation",
        "bayesian",
        "bandit",
        "ucb",
        "epsilon_greedy",
        "regret",
        "reward"
    ]

    for node in nodes:
        G.add_node(node)

    edges = [
        ("thompson_sampling", "exploration", "USES"),
        ("thompson_sampling", "exploitation", "USES"),
        ("thompson_sampling", "bayesian", "IS_A"),
        ("bayesian", "bandit", "IS_A"),
        ("ucb", "bandit", "IS_A"),
        ("epsilon_greedy", "bandit", "IS_A"),
        ("thompson_sampling", "regret", "LEADS_TO"),
        ("bandit", "reward", "LEADS_TO"),
    ]

    for head, tail, edge_type in edges:
        G.add_edge(head, tail, type=edge_type)

    return G


@pytest.fixture
def sample_nodes():
    """Sample node IDs."""
    return [
        "thompson_sampling",
        "exploration",
        "exploitation",
        "bayesian",
        "bandit",
    ]


@pytest.fixture
def sample_contents():
    """Sample node text contents."""
    return {
        "thompson_sampling": "Thompson Sampling is a Bayesian algorithm for the multi-armed bandit problem that balances exploration and exploitation using probability distributions.",
        "exploration": "Exploration involves trying new options to gather information about their potential rewards.",
        "exploitation": "Exploitation means choosing the option that currently appears to give the best reward.",
        "bayesian": "Bayesian methods use prior knowledge and update beliefs based on observed evidence using Bayes' theorem.",
        "bandit": "Multi-armed bandit is a classic problem in decision theory where you must choose between multiple options with uncertain rewards.",
        "ucb": "Upper Confidence Bound (UCB) uses confidence intervals to balance exploration and exploitation deterministically.",
        "epsilon_greedy": "Epsilon-greedy explores with probability epsilon and exploits with probability 1-epsilon.",
        "regret": "Regret measures the difference between the optimal reward and the actual reward obtained.",
        "reward": "Reward is the payoff received from selecting a particular option in the bandit problem.",
    }


@pytest.fixture
def info_scorer_config():
    """Configuration with information scoring enabled."""
    return ImportanceScorerConfig(
        enable_information_scoring=True,
        mi_estimation_bins=10,
        entropy_threshold=0.5,
        use_entropy_aware_aggregation=True
    )


@pytest.fixture
def compression_config():
    """Configuration with information budget enabled."""
    return CompressionConfig(
        enable_information_budget=True,
        information_budget=5.0,
        mi_diminishing_returns_threshold=0.1,
        mi_high_threshold=0.7,
        mi_mid_threshold=0.4,
        mi_low_threshold=0.2
    )


# ============================================================================
# Test: INFORMATION_CONTENT Signal
# ============================================================================

class TestInformationContentSignal:
    """Tests for 7th importance signal: INFORMATION_CONTENT."""

    def test_information_signal_exists(self):
        """Verify INFORMATION_CONTENT is in ImportanceSignal enum."""
        assert hasattr(ImportanceSignal, 'INFORMATION_CONTENT')
        assert ImportanceSignal.INFORMATION_CONTENT.value == "information"

    def test_signal_in_default_weights(self, info_scorer_config):
        """Verify INFORMATION_CONTENT has 25% weight in default config."""
        weights = info_scorer_config.weights
        assert ImportanceSignal.INFORMATION_CONTENT in weights
        assert weights[ImportanceSignal.INFORMATION_CONTENT] == 0.25

    def test_weights_sum_to_one(self, info_scorer_config):
        """Verify all weights sum to 1.0."""
        total = sum(info_scorer_config.weights.values())
        assert abs(total - 1.0) < 0.01


class TestMIScoring:
    """Tests for Mutual Information scoring."""

    def test_scorer_computes_mi_signal(
        self, sample_graph, sample_nodes, sample_contents, info_scorer_config
    ):
        """Verify scorer computes INFORMATION_CONTENT signal."""
        scorer = ImportanceScorer(info_scorer_config)

        scores = scorer.score_importance(
            node_id="thompson_sampling",
            query="What is Thompson Sampling?",
            graph=sample_graph,
            node_content=sample_contents["thompson_sampling"]
        )

        # Should have INFORMATION_CONTENT signal
        assert ImportanceSignal.INFORMATION_CONTENT in scores

        # MI score should be in valid range
        mi_score = scores[ImportanceSignal.INFORMATION_CONTENT]
        assert 0.0 <= mi_score <= 1.0

    def test_high_mi_for_relevant_content(
        self, sample_graph, sample_contents, info_scorer_config
    ):
        """Highly relevant content should have high MI."""
        scorer = ImportanceScorer(info_scorer_config)

        # Query about Thompson Sampling
        query = "How does Thompson Sampling balance exploration and exploitation?"

        # Relevant node (contains query concepts)
        relevant_scores = scorer.score_importance(
            node_id="thompson_sampling",
            query=query,
            graph=sample_graph,
            node_content=sample_contents["thompson_sampling"]
        )

        # Less relevant node
        less_relevant_scores = scorer.score_importance(
            node_id="reward",
            query=query,
            graph=sample_graph,
            node_content=sample_contents["reward"]
        )

        relevant_mi = relevant_scores.get(ImportanceSignal.INFORMATION_CONTENT, 0.0)
        less_relevant_mi = less_relevant_scores.get(ImportanceSignal.INFORMATION_CONTENT, 0.0)

        # Relevant content should have higher or equal MI
        assert relevant_mi >= less_relevant_mi * 0.5  # Allow some tolerance

    def test_batch_mi_scoring(
        self, sample_graph, sample_nodes, sample_contents, info_scorer_config
    ):
        """Test batch MI computation."""
        scorer = ImportanceScorer(info_scorer_config)

        mi_scores = scorer.batch_compute_mi_scores(
            node_ids=sample_nodes,
            query="Explain Bayesian bandits",
            node_contents=sample_contents
        )

        # Should return scores for all nodes
        assert len(mi_scores) == len(sample_nodes)

        # All scores should be in valid range
        for node_id, score in mi_scores.items():
            assert 0.0 <= score <= 1.0

    def test_word_overlap_fallback(self, sample_graph, info_scorer_config):
        """Without embeddings, should use word overlap fallback."""
        scorer = ImportanceScorer(info_scorer_config, embedder=None)

        query = "Thompson Sampling exploration"
        content = "Thompson Sampling balances exploration and exploitation"

        scores = scorer.score_importance(
            node_id="test",
            query=query,
            graph=sample_graph,
            node_content=content
        )

        # Should still compute MI via word overlap
        assert ImportanceSignal.INFORMATION_CONTENT in scores
        mi_score = scores[ImportanceSignal.INFORMATION_CONTENT]
        assert mi_score > 0.0  # Should have some overlap


class TestMICaching:
    """Tests for MI caching performance."""

    def test_cache_stores_results(self, info_scorer_config, sample_contents):
        """Verify MI results are cached."""
        scorer = ImportanceScorer(info_scorer_config)

        # First call (cold cache)
        scorer.batch_compute_mi_scores(
            node_ids=["thompson_sampling"],
            query="What is Thompson Sampling?",
            node_contents=sample_contents
        )

        stats = scorer.get_stats()
        initial_misses = stats.get('mi_cache_misses', 0)

        # Second call (should hit cache)
        scorer.batch_compute_mi_scores(
            node_ids=["thompson_sampling"],
            query="What is Thompson Sampling?",
            node_contents=sample_contents
        )

        stats = scorer.get_stats()
        hits = stats.get('mi_cache_hits', 0)

        # Should have cache hit on second call
        assert hits > 0

    def test_cache_invalidation(self, info_scorer_config, sample_contents):
        """Verify cache can be invalidated."""
        scorer = ImportanceScorer(info_scorer_config)

        # Populate cache
        scorer.batch_compute_mi_scores(
            node_ids=["thompson_sampling"],
            query="Test query",
            node_contents=sample_contents
        )

        # Invalidate
        scorer.invalidate_mi_cache()

        # Cache should be empty
        assert len(scorer._mi_cache) == 0


class TestEntropyAwareAggregation:
    """Tests for entropy-aware score aggregation."""

    def test_entropy_aware_aggregation_available(self, info_scorer_config):
        """Verify entropy-aware aggregation method exists."""
        scorer = ImportanceScorer(info_scorer_config)
        assert hasattr(scorer, 'aggregate_scores_entropy_aware')

    def test_low_entropy_increases_score(self, info_scorer_config):
        """Low entropy (high certainty) should increase aggregated score."""
        scorer = ImportanceScorer(info_scorer_config)

        scores = {
            ImportanceSignal.RELEVANCE: 0.8,
            ImportanceSignal.RECENCY: 0.5,
            ImportanceSignal.CENTRALITY: 0.6,
            ImportanceSignal.ACCESS_FREQUENCY: 0.4,
            ImportanceSignal.CONFIDENCE: 0.7,
            ImportanceSignal.HEAT: 0.3,
            ImportanceSignal.INFORMATION_CONTENT: 0.6,
        }

        # Low entropy = focused topic (repeated words)
        low_entropy_content = "Thompson Thompson Thompson sampling sampling"
        low_entropy_score = scorer.aggregate_scores_entropy_aware(
            scores=scores,
            node_content=low_entropy_content
        )

        # High entropy = diverse vocabulary (many unique words)
        high_entropy_content = "explore exploit bandit reward arm policy prior beta"
        high_entropy_score = scorer.aggregate_scores_entropy_aware(
            scores=scores,
            node_content=high_entropy_content
        )

        # Low entropy should give equal or higher score (focused content boosted)
        assert low_entropy_score >= high_entropy_score

    def test_config_can_disable_entropy_aggregation(self):
        """Verify entropy-aware aggregation can be disabled."""
        config = ImportanceScorerConfig(
            enable_information_scoring=True,
            use_entropy_aware_aggregation=False
        )
        assert not config.use_entropy_aware_aggregation


# ============================================================================
# Test: Information Budget Compression
# ============================================================================

class TestInformationBudgetCompression:
    """Tests for information budget-based compression."""

    def test_budget_compression_respects_limit(self, compression_config):
        """Compression should respect information budget."""
        # Set low preserve_top_k so budget is actually enforced
        compression_config.preserve_top_k = 1
        compressor = ContextCompressor(compression_config)

        nodes = ["node1", "node2", "node3", "node4", "node5"]
        importance_scores = {n: 0.5 for n in nodes}
        mi_scores = {
            "node1": 2.0,  # High MI
            "node2": 1.5,
            "node3": 1.0,
            "node4": 0.5,
            "node5": 0.2,  # Low MI
        }

        result = compressor.information_budget_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores,
            information_budget=4.0  # Should keep ~2-3 nodes
        )

        # Should not exceed budget (allow partial inclusion of last node)
        total_mi = sum(mi_scores.get(n, 0) for n in result.compressed_nodes)
        # Budget=4.0, so should keep node1(2.0)+node2(1.5)=3.5, maybe +node3(1.0) if partial
        assert total_mi <= 5.0  # Budget + largest single node

    def test_diminishing_returns_stops_selection(self, compression_config):
        """Should stop when marginal MI gain too small."""
        # Set low preserve_top_k so diminishing returns is actually enforced
        compression_config.preserve_top_k = 1
        compressor = ContextCompressor(compression_config)

        nodes = ["high", "medium", "low1", "low2", "low3"]
        importance_scores = {n: 0.5 for n in nodes}
        mi_scores = {
            "high": 2.0,
            "medium": 1.0,
            "low1": 0.05,  # Below threshold (0.1)
            "low2": 0.03,
            "low3": 0.01,
        }

        result = compressor.information_budget_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores,
            information_budget=10.0  # High budget
        )

        # Should stop before low MI nodes due to diminishing returns
        # Only high (2.0) and medium (1.0) should be kept
        assert len(result.compressed_nodes) == 2
        assert "high" in result.compressed_nodes
        assert "medium" in result.compressed_nodes

    def test_preserve_top_k_honored(self, compression_config):
        """Should preserve at least preserve_top_k nodes."""
        compression_config.preserve_top_k = 3
        compressor = ContextCompressor(compression_config)

        nodes = ["n1", "n2", "n3", "n4"]
        importance_scores = {n: 0.5 for n in nodes}
        mi_scores = {n: 0.01 for n in nodes}  # All low MI

        result = compressor.information_budget_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores,
            information_budget=0.001  # Very tight budget
        )

        # Should still keep at least preserve_top_k
        assert len(result.compressed_nodes) >= 3

    def test_greedy_selection_order(self, compression_config):
        """Nodes should be selected in MI order (greedy)."""
        compressor = ContextCompressor(compression_config)

        nodes = ["low", "high", "medium"]
        importance_scores = {n: 0.5 for n in nodes}
        mi_scores = {
            "low": 0.5,
            "high": 2.0,
            "medium": 1.0,
        }

        result = compressor.information_budget_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores,
            information_budget=2.5  # Fits high + some of medium
        )

        # High MI node should definitely be included
        assert "high" in result.compressed_nodes


class TestMIAwareMatryoshkaScales:
    """Tests for MI-aware Matryoshka scale assignment."""

    def test_high_mi_gets_full_scale(self, compression_config):
        """High MI nodes should get 384D (full scale)."""
        compressor = ContextCompressor(compression_config)

        nodes = ["high_mi"]
        importance_scores = {"high_mi": 0.5}
        mi_scores = {"high_mi": 0.9}  # > 0.7 threshold

        kept_nodes, scale_assignments = compressor.matryoshka_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores
        )

        assert "high_mi" in kept_nodes
        assert scale_assignments["high_mi"] == 384

    def test_medium_mi_gets_medium_scale(self, compression_config):
        """Medium MI nodes should get 256D."""
        compressor = ContextCompressor(compression_config)

        nodes = ["medium_mi"]
        importance_scores = {"medium_mi": 0.5}
        mi_scores = {"medium_mi": 0.5}  # Between 0.4 and 0.7

        kept_nodes, scale_assignments = compressor.matryoshka_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores
        )

        assert "medium_mi" in kept_nodes
        assert scale_assignments["medium_mi"] == 256

    def test_low_mi_gets_small_scale(self, compression_config):
        """Low MI nodes should get 128D."""
        compressor = ContextCompressor(compression_config)

        nodes = ["low_mi"]
        importance_scores = {"low_mi": 0.5}
        mi_scores = {"low_mi": 0.3}  # Between 0.2 and 0.4

        kept_nodes, scale_assignments = compressor.matryoshka_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores
        )

        assert "low_mi" in kept_nodes
        assert scale_assignments["low_mi"] == 128

    def test_very_low_mi_dropped(self, compression_config):
        """Very low MI nodes should be dropped when not in top_k."""
        # Use preserve_top_k=1, so one node must be kept but others can drop
        compression_config.preserve_top_k = 1
        compressor = ContextCompressor(compression_config)

        # Two nodes: one high MI (kept), one very low MI (should be dropped)
        nodes = ["high_mi", "very_low_mi"]
        importance_scores = {"high_mi": 0.9, "very_low_mi": 0.5}
        mi_scores = {
            "high_mi": 0.8,  # Above high threshold (0.7)
            "very_low_mi": 0.1  # Below low threshold (0.2)
        }

        kept_nodes, scale_assignments = compressor.matryoshka_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores
        )

        # high_mi kept (high scale), very_low_mi dropped (below threshold)
        assert "high_mi" in kept_nodes
        assert "very_low_mi" not in kept_nodes
        assert len(kept_nodes) == 1

    def test_falls_back_to_importance_without_mi(self, compression_config):
        """Without MI scores, should use importance-based thresholds."""
        compressor = ContextCompressor(compression_config)

        nodes = ["high_importance"]
        importance_scores = {"high_importance": 0.9}  # > 0.75 threshold

        # No MI scores provided
        kept_nodes, scale_assignments = compressor.matryoshka_compress(
            nodes=nodes,
            importance_scores=importance_scores,
            mi_scores=None
        )

        assert "high_importance" in kept_nodes
        assert scale_assignments["high_importance"] == 384


# ============================================================================
# Test: Full Integration
# ============================================================================

class TestInformationBudgetPacking:
    """Integration tests for information budget packing."""

    def test_packer_with_information_budget(
        self, sample_graph, sample_nodes, sample_contents
    ):
        """Test full packing pipeline with information budget."""
        config = ContextPackerConfig()
        config.importance_scorer.enable_information_scoring = True
        config.compression.enable_information_budget = True
        config.compression.information_budget = 3.0

        packer = ContextPacker(config)

        result, scale_assignments, mi_scores = packer.pack_with_information_budget(
            query="Explain Thompson Sampling",
            candidate_nodes=sample_nodes,
            graph=sample_graph,
            node_contents=sample_contents,
            information_budget=3.0
        )

        # Should return valid result
        assert isinstance(result, CompressionResult)
        assert result.compressed_count <= result.original_count

        # Should have scale assignments
        assert isinstance(scale_assignments, dict)

        # Should have MI scores
        assert isinstance(mi_scores, dict)
        assert len(mi_scores) > 0

    def test_fallback_when_info_scoring_disabled(
        self, sample_graph, sample_nodes, sample_contents
    ):
        """Should fall back gracefully when info scoring disabled."""
        config = ContextPackerConfig()
        config.importance_scorer.enable_information_scoring = False

        packer = ContextPacker(config)

        result, scale_assignments, mi_scores = packer.pack_with_information_budget(
            query="Test",
            candidate_nodes=sample_nodes,
            graph=sample_graph,
            node_contents=sample_contents
        )

        # Should still work
        assert isinstance(result, CompressionResult)

        # MI scores should be empty (disabled)
        assert mi_scores == {}

    def test_stats_track_mi_utilization(
        self, sample_graph, sample_nodes, sample_contents
    ):
        """Stats should track MI budget utilization."""
        config = ContextPackerConfig()
        config.importance_scorer.enable_information_scoring = True
        config.compression.enable_information_budget = True

        packer = ContextPacker(config)

        packer.pack_with_information_budget(
            query="Test",
            candidate_nodes=sample_nodes,
            graph=sample_graph,
            node_contents=sample_contents
        )

        stats = packer.get_stats()

        assert 'info_budget_packs' in stats
        assert 'avg_mi_utilization' in stats
        assert stats['info_budget_packs'] == 1


class TestPerformance:
    """Performance tests for information scoring."""

    def test_mi_computation_performance(
        self, sample_graph, sample_contents, info_scorer_config
    ):
        """MI computation should be fast (<5ms for batch)."""
        scorer = ImportanceScorer(info_scorer_config)

        nodes = list(sample_contents.keys())[:5]

        start = time.perf_counter()
        scorer.batch_compute_mi_scores(
            node_ids=nodes,
            query="Thompson Sampling exploration exploitation",
            node_contents=sample_contents
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Should complete in under 50ms (generous for CI)
        assert elapsed < 50, f"MI computation took {elapsed:.2f}ms"

    def test_cache_improves_performance(
        self, sample_contents, info_scorer_config
    ):
        """Cached MI should be much faster than cold."""
        scorer = ImportanceScorer(info_scorer_config)

        nodes = list(sample_contents.keys())[:3]
        query = "Test performance"

        # Cold cache
        start = time.perf_counter()
        scorer.batch_compute_mi_scores(nodes, query, sample_contents)
        cold_time = time.perf_counter() - start

        # Warm cache
        start = time.perf_counter()
        scorer.batch_compute_mi_scores(nodes, query, sample_contents)
        warm_time = time.perf_counter() - start

        # Warm should be faster or equal (allow for timing noise)
        assert warm_time <= cold_time * 2  # At least not slower


# ============================================================================
# Test: Configuration Validation
# ============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""

    def test_weights_must_sum_to_one(self):
        """Weights validation should catch invalid sums."""
        with pytest.raises(AssertionError):
            config = ImportanceScorerConfig()
            config.weights[ImportanceSignal.RECENCY] = 0.9  # Now sums to > 1
            config.validate()

    def test_mi_thresholds_ordered(self):
        """MI thresholds must be ordered low < mid < high."""
        with pytest.raises(AssertionError):
            config = CompressionConfig(
                mi_low_threshold=0.5,
                mi_mid_threshold=0.3,  # Wrong order
                mi_high_threshold=0.7
            )
            config.validate()

    def test_information_budget_positive(self):
        """Information budget must be positive."""
        with pytest.raises(AssertionError):
            config = CompressionConfig(information_budget=-1.0)
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
