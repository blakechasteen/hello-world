"""
Context Packing System Tests
=============================

Comprehensive test suite for the context packing system.

Tests:
- Protocol definitions and data classes
- Configuration presets and validation
- Beta wave activation spreading
- Multi-signal importance scoring
- Matryoshka-aware compression
- Full context packing pipeline

Author: Claude Code
Date: 2025-11-22
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
    # Protocols and data classes
    ImportanceSignal,
    ActivationState,
    CompressionResult,
    BetaWave,
    # Configuration
    BetaWaveConfig,
    ImportanceScorerConfig,
    CompressionConfig,
    ContextPackerConfig,
    # Core components
    ActivationSpreader,
    ImportanceScorer,
    ContextCompressor,
    ContextPacker,
)


# Test Fixtures

@pytest.fixture
def sample_graph():
    """Create sample knowledge graph for testing."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")

    G = nx.MultiDiGraph()

    # Add nodes
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

    # Add edges with types
    edges = [
        ("thompson_sampling", "exploration", "USES"),
        ("thompson_sampling", "exploitation", "USES"),
        ("thompson_sampling", "bayesian", "IS_A"),
        ("bayesian", "bandit", "IS_A"),
        ("ucb", "bandit", "IS_A"),
        ("epsilon_greedy", "bandit", "IS_A"),
        ("thompson_sampling", "regret", "LEADS_TO"),
        ("bandit", "reward", "LEADS_TO"),
        ("exploration", "reward", "LEADS_TO"),
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
        "bandit"
    ]


@pytest.fixture
def sample_importance_scores():
    """Sample importance scores."""
    return {
        "thompson_sampling": 0.95,
        "exploration": 0.80,
        "exploitation": 0.75,
        "bayesian": 0.70,
        "bandit": 0.65,
        "ucb": 0.60,
        "epsilon_greedy": 0.55,
        "regret": 0.50,
        "reward": 0.45
    }


# Configuration Tests

def test_beta_wave_config_defaults():
    """Test BetaWaveConfig default values."""
    config = BetaWaveConfig()

    assert config.frequency == 20.0
    assert config.amplitude == 1.0
    assert config.decay_rate == 0.7
    assert config.max_hops == 3
    assert config.min_activation == 0.1


def test_beta_wave_config_validation():
    """Test BetaWaveConfig validation."""
    config = BetaWaveConfig()
    config.validate()  # Should pass

    # Test invalid frequency
    config.frequency = 5.0  # Below beta range
    with pytest.raises(AssertionError):
        config.validate()


def test_importance_scorer_config_weights():
    """Test ImportanceScorerConfig weight validation."""
    config = ImportanceScorerConfig()

    # Weights should sum to 1.0
    total = sum(config.weights.values())
    assert abs(total - 1.0) < 0.01

    config.validate()  # Should pass


def test_compression_config_presets():
    """Test CompressionConfig thresholds."""
    config = CompressionConfig()

    assert config.low_importance_threshold < config.mid_importance_threshold
    assert config.mid_importance_threshold < config.high_importance_threshold

    config.validate()


def test_context_packer_config_presets():
    """Test ContextPackerConfig preset methods."""
    # Aggressive preset
    aggressive = ContextPackerConfig.aggressive()
    assert aggressive.compression.target_ratio == 0.3  # Keep only 30%

    # Balanced preset
    balanced = ContextPackerConfig.balanced()
    assert balanced.compression.target_ratio == 0.5  # Keep 50%

    # Conservative preset
    conservative = ContextPackerConfig.conservative()
    assert conservative.compression.target_ratio == 0.7  # Keep 70%

    # Research preset
    research = ContextPackerConfig.research()
    assert research.compression.target_ratio == 0.9  # Keep 90%
    assert research.beta_wave.max_hops == 5


# Activation Spreading Tests

def test_activation_spreader_initialization():
    """Test ActivationSpreader initialization."""
    spreader = ActivationSpreader()

    assert spreader.config is not None
    assert spreader.activation_history == []


def test_activation_spreading_basic(sample_graph, sample_nodes):
    """Test basic activation spreading."""
    spreader = ActivationSpreader()

    activation = spreader.spread_activation(
        source_nodes=["thompson_sampling"],
        graph=sample_graph,
        initial_activation=1.0,
        max_hops=2
    )

    # Check source node has activation
    assert "thompson_sampling" in activation
    assert activation["thompson_sampling"] == 1.0

    # Check neighbors activated
    assert "exploration" in activation
    assert "exploitation" in activation
    assert "bayesian" in activation

    # Activation should decay
    assert activation["exploration"] < 1.0


def test_activation_spreading_decay(sample_graph):
    """Test activation decay with distance."""
    spreader = ActivationSpreader()

    activation = spreader.spread_activation(
        source_nodes=["thompson_sampling"],
        graph=sample_graph,
        decay_rate=0.5,  # 50% decay per hop
        max_hops=3
    )

    # Direct neighbors should have 0.5 activation
    neighbors = list(sample_graph.successors("thompson_sampling"))
    for neighbor in neighbors:
        if neighbor in activation:
            assert activation[neighbor] <= 0.5


def test_multi_frequency_spread(sample_graph):
    """Test multi-frequency activation spreading."""
    spreader = ActivationSpreader()

    results = spreader.multi_frequency_spread(
        source_nodes=["thompson_sampling"],
        graph=sample_graph,
        frequencies=[15.0, 20.0, 25.0]
    )

    assert len(results) == 3
    assert 15.0 in results
    assert 20.0 in results
    assert 25.0 in results


# Importance Scoring Tests

def test_importance_scorer_initialization():
    """Test ImportanceScorer initialization."""
    scorer = ImportanceScorer()

    assert scorer.config is not None
    assert scorer._centrality_cache is None


def test_score_importance_signals(sample_graph):
    """Test individual importance signal scoring."""
    scorer = ImportanceScorer()

    # Create activation state
    state = ActivationState(
        node_id="thompson_sampling",
        activation=0.9,
        last_accessed=time.time(),
        access_count=10,
        heat=5.0
    )

    scores = scorer.score_importance(
        node_id="thompson_sampling",
        query="What is Thompson Sampling?",
        graph=sample_graph,
        activation_state=state
    )

    # Check all signals present
    assert ImportanceSignal.RECENCY in scores
    assert ImportanceSignal.RELEVANCE in scores  # Will be 0.5 without embedder
    assert ImportanceSignal.CENTRALITY in scores
    assert ImportanceSignal.ACCESS_FREQUENCY in scores
    assert ImportanceSignal.CONFIDENCE in scores
    assert ImportanceSignal.HEAT in scores

    # All scores should be 0-1
    for signal, score in scores.items():
        assert 0.0 <= score <= 1.0


def test_aggregate_scores():
    """Test score aggregation."""
    scorer = ImportanceScorer()

    scores = {
        ImportanceSignal.RECENCY: 0.9,
        ImportanceSignal.RELEVANCE: 0.8,
        ImportanceSignal.CENTRALITY: 0.7,
        ImportanceSignal.ACCESS_FREQUENCY: 0.6,
        ImportanceSignal.CONFIDENCE: 0.5,
        ImportanceSignal.HEAT: 0.4
    }

    aggregated = scorer.aggregate_scores(scores)

    assert 0.0 <= aggregated <= 1.0
    # Should be weighted average
    assert 0.5 < aggregated < 0.9


def test_batch_scoring(sample_graph, sample_nodes):
    """Test batch importance scoring."""
    scorer = ImportanceScorer()

    importance_scores = scorer.score_batch(
        node_ids=sample_nodes,
        query="Explain Thompson Sampling",
        graph=sample_graph
    )

    assert len(importance_scores) == len(sample_nodes)

    for node in sample_nodes:
        assert node in importance_scores
        assert 0.0 <= importance_scores[node] <= 1.0


# Compression Tests

def test_context_compressor_initialization():
    """Test ContextCompressor initialization."""
    compressor = ContextCompressor()

    assert compressor.config is not None


def test_basic_compression(sample_nodes, sample_importance_scores):
    """Test basic threshold-based compression."""
    compressor = ContextCompressor()

    result = compressor.compress(
        nodes=sample_nodes,
        importance_scores=sample_importance_scores,
        target_ratio=0.6  # Keep 60%
    )

    assert result.compressed_count <= result.original_count
    assert result.compressed_count >= 0.5 * result.original_count  # At least 50%
    assert result.compression_ratio > 0
    assert result.token_savings >= 0


def test_matryoshka_compression(sample_nodes, sample_importance_scores):
    """Test Matryoshka-aware multi-scale compression."""
    compressor = ContextCompressor()

    kept_nodes, scale_assignments = compressor.matryoshka_compress(
        nodes=sample_nodes,
        importance_scores=sample_importance_scores
    )

    # Check that nodes are kept
    assert len(kept_nodes) > 0
    assert len(scale_assignments) == len(kept_nodes)

    # Check scale assignments are valid
    for node, scale in scale_assignments.items():
        assert scale in [384, 256, 128]

    # Higher importance nodes should get larger scales
    if "thompson_sampling" in scale_assignments and "bandit" in scale_assignments:
        assert scale_assignments["thompson_sampling"] >= scale_assignments["bandit"]


def test_adaptive_compression(sample_nodes, sample_importance_scores):
    """Test adaptive compression within token budget."""
    compressor = ContextCompressor()

    result = compressor.adaptive_compress(
        nodes=sample_nodes,
        importance_scores=sample_importance_scores,
        min_tokens=100,
        max_tokens=300
    )

    # Should stay within budget
    estimated_tokens = result.compressed_count * 50  # avg_tokens_per_node
    assert 100 <= estimated_tokens <= 300


# Full Packing Pipeline Tests

def test_context_packer_initialization():
    """Test ContextPacker initialization."""
    packer = ContextPacker()

    assert packer.config is not None
    assert packer.activation_spreader is not None
    assert packer.importance_scorer is not None
    assert packer.context_compressor is not None


def test_full_pack_pipeline(sample_graph, sample_nodes):
    """Test complete context packing pipeline."""
    packer = ContextPacker()

    result = packer.pack(
        query="Explain Thompson Sampling",
        candidate_nodes=sample_nodes,
        graph=sample_graph,
        target_tokens=2000
    )

    # Check result structure
    assert isinstance(result, CompressionResult)
    assert result.compressed_count <= result.original_count
    assert result.compression_ratio <= 1.0
    assert result.token_savings >= 0
    assert len(result.compressed_nodes) == result.compressed_count


def test_pack_with_different_presets(sample_graph, sample_nodes):
    """Test packing with different configuration presets."""
    # Aggressive preset
    aggressive_packer = ContextPacker(ContextPackerConfig.aggressive())
    aggressive_result = aggressive_packer.pack(
        "test query", sample_nodes, sample_graph, 2000
    )

    # Conservative preset
    conservative_packer = ContextPacker(ContextPackerConfig.conservative())
    conservative_result = conservative_packer.pack(
        "test query", sample_nodes, sample_graph, 2000
    )

    # Conservative should keep more nodes
    assert conservative_result.compressed_count >= aggressive_result.compressed_count


def test_adaptive_pack_pipeline(sample_graph, sample_nodes):
    """Test adaptive packing pipeline."""
    packer = ContextPacker()

    result = packer.adaptive_pack(
        query="Explain Thompson Sampling",
        candidate_nodes=sample_nodes,
        graph=sample_graph,
        min_tokens=500,
        max_tokens=2000
    )

    estimated_tokens = result.compressed_count * 50
    assert 500 <= estimated_tokens <= 2000


def test_pack_with_scales(sample_graph, sample_nodes):
    """Test packing with Matryoshka scale assignments."""
    packer = ContextPacker()

    result, scale_assignments = packer.pack_with_scales(
        query="Explain Thompson Sampling",
        candidate_nodes=sample_nodes,
        graph=sample_graph,
        target_tokens=2000
    )

    assert len(scale_assignments) == result.compressed_count

    for node in result.compressed_nodes:
        assert node in scale_assignments
        assert scale_assignments[node] in [384, 256, 128]


def test_packer_statistics_tracking(sample_graph, sample_nodes):
    """Test statistics tracking across multiple packs."""
    packer = ContextPacker()

    # Pack multiple times
    for i in range(3):
        packer.pack(
            query=f"query {i}",
            candidate_nodes=sample_nodes,
            graph=sample_graph
        )

    stats = packer.get_stats()

    assert stats['total_packs'] == 3
    assert stats['avg_compression_ratio'] > 0
    assert 'component_stats' in stats


def test_convenience_functions(sample_graph, sample_nodes):
    """Test convenience functions."""
    from hololoom.context_packing.packer import pack_context, adaptive_pack_context

    # pack_context
    packed = pack_context(
        query="test",
        candidate_nodes=sample_nodes,
        graph=sample_graph,
        target_tokens=2000,
        preset="balanced"
    )

    assert isinstance(packed, list)
    assert len(packed) <= len(sample_nodes)

    # adaptive_pack_context
    adaptive_packed = adaptive_pack_context(
        query="test",
        candidate_nodes=sample_nodes,
        graph=sample_graph,
        min_tokens=500,
        max_tokens=2000
    )

    assert isinstance(adaptive_packed, list)


# Edge Cases and Error Handling

def test_empty_node_list():
    """Test handling of empty node list."""
    compressor = ContextCompressor()

    result = compressor.compress(
        nodes=[],
        importance_scores={}
    )

    assert result.original_count == 0
    assert result.compressed_count == 0
    assert result.compression_ratio == 0.0


def test_all_low_importance_nodes():
    """Test compression with all low-importance nodes."""
    compressor = ContextCompressor()

    nodes = ["a", "b", "c", "d", "e"]
    scores = {n: 0.1 for n in nodes}  # All very low importance

    kept, scales = compressor.matryoshka_compress(nodes, scores)

    # Should still keep preserve_top_k nodes
    assert len(kept) >= compressor.config.preserve_top_k


def test_single_node():
    """Test packing with single node."""
    if not NETWORKX_AVAILABLE:
        pytest.skip("NetworkX not available")

    G = nx.MultiDiGraph()
    G.add_node("single")

    packer = ContextPacker()
    result = packer.pack(
        query="test",
        candidate_nodes=["single"],
        graph=G
    )

    assert result.compressed_count == 1
    assert "single" in result.compressed_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
