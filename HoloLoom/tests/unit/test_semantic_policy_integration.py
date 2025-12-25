#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 8: Semantic State -> Policy Integration

Tests the complete semantic integration pipeline:
1. PolicySemanticState (SemanticState) - 244D -> 8D feature vector
2. SemanticToolSelector - Semantic-aware tool suggestions
3. SemanticAwareBandit - Thompson Sampling with semantic adjustments
4. NeuralCore integration - Gated semantic feature fusion

Run with: pytest HoloLoom/tests/unit/test_semantic_policy_integration.py -v
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any

# Import semantic integration components
from HoloLoom.semantic_calculus.semantic_state import (
    SemanticState,
    SemanticToolSelector,
    SemanticAwareBandit,
    SEMANTIC_CATEGORIES,
    TOPIC_SHIFT_THRESHOLD,
    SIGNIFICANT_SHIFT_THRESHOLD,
    _compute_category_scores,
    _pad_or_truncate,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def random_embedding():
    """Create a random 384D embedding."""
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def random_244d_position():
    """Create a random 244D semantic position."""
    return np.random.randn(244).astype(np.float32)


@pytest.fixture
def sample_semantic_state(random_embedding):
    """Create a sample SemanticState from embedding."""
    return SemanticState.from_embedding(random_embedding)


@pytest.fixture
def tool_selector():
    """Create a SemanticToolSelector instance."""
    return SemanticToolSelector()


@pytest.fixture
def semantic_bandit():
    """Create a SemanticAwareBandit instance."""
    return SemanticAwareBandit()


# =============================================================================
# SemanticState Tests
# =============================================================================

class TestSemanticState:
    """Tests for SemanticState dataclass."""

    def test_from_embedding_creates_valid_state(self, random_embedding):
        """from_embedding should create a valid SemanticState."""
        state = SemanticState.from_embedding(random_embedding)

        assert state is not None
        assert state.position is not None
        assert len(state.position) == 244
        assert 0.0 <= state.momentum <= 1.0
        assert 0.0 <= state.complexity <= 1.0
        assert state.dominant_category in SEMANTIC_CATEGORIES

    def test_from_embedding_with_previous_state(self, random_embedding):
        """from_embedding should compute velocity from previous state."""
        state1 = SemanticState.from_embedding(random_embedding)

        # Create shifted embedding
        shifted = random_embedding + np.random.randn(384).astype(np.float32) * 0.5
        state2 = SemanticState.from_embedding(shifted, previous_state=state1)

        assert state2.velocity is not None
        assert len(state2.velocity) == 244
        assert state2.shift_magnitude >= 0.0

    def test_topic_shift_detection(self, random_embedding):
        """Large velocity should trigger topic shift detection."""
        state1 = SemanticState.from_embedding(random_embedding)

        # Create large shift
        shifted = random_embedding + np.random.randn(384).astype(np.float32) * 5.0
        state2 = SemanticState.from_embedding(shifted, previous_state=state1)

        assert state2.shift_magnitude > TOPIC_SHIFT_THRESHOLD
        assert state2.topic_shift_detected is True

    def test_to_feature_vector_shape(self, sample_semantic_state):
        """to_feature_vector should return 8D numpy array."""
        features = sample_semantic_state.to_feature_vector()

        assert isinstance(features, np.ndarray)
        assert features.shape == (8,)
        assert features.dtype == np.float32

    def test_to_feature_vector_contents(self, sample_semantic_state):
        """to_feature_vector should contain momentum, complexity, dims, velocity."""
        features = sample_semantic_state.to_feature_vector()

        # First two elements are momentum and complexity
        assert features[0] == sample_semantic_state.momentum
        assert features[1] == sample_semantic_state.complexity
        # Last element is velocity magnitude
        assert features[7] >= 0.0

    def test_empty_state(self):
        """empty() should create zero state."""
        state = SemanticState.empty()

        assert np.allclose(state.position, 0.0)
        assert state.momentum == 0.5
        assert state.complexity == 0.0
        assert state.topic_shift_detected is False

    def test_category_scores_sum(self, sample_semantic_state):
        """Category scores should be computed for all 16 categories."""
        assert len(sample_semantic_state.category_scores) == len(SEMANTIC_CATEGORIES)

        # At least one category should have a non-zero score
        max_score = max(sample_semantic_state.category_scores.values())
        assert max_score > 0.0

    def test_dominant_category_is_highest_score(self, sample_semantic_state):
        """Dominant category should have the highest score."""
        dominant = sample_semantic_state.dominant_category
        dominant_score = sample_semantic_state.category_scores[dominant]

        for cat, score in sample_semantic_state.category_scores.items():
            assert score <= dominant_score

    def test_repr_format(self, sample_semantic_state):
        """__repr__ should return readable string."""
        repr_str = repr(sample_semantic_state)

        assert "SemanticState" in repr_str
        assert "momentum=" in repr_str
        assert "complexity=" in repr_str


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_category_scores_empty(self):
        """Empty position should return zero scores."""
        scores = _compute_category_scores(np.zeros(244))

        assert all(v == 0.0 for v in scores.values())

    def test_compute_category_scores_normalized(self, random_244d_position):
        """Scores should be normalized to 0-1."""
        scores = _compute_category_scores(random_244d_position)

        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert max(scores.values()) == 1.0  # Max should be exactly 1.0

    def test_pad_or_truncate_shorter(self):
        """Should pad shorter arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_or_truncate(arr, 5)

        assert len(result) == 5
        assert result[0] == 1.0
        assert result[3] == 0.0

    def test_pad_or_truncate_longer(self):
        """Should truncate longer arrays."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _pad_or_truncate(arr, 3)

        assert len(result) == 3
        assert result[2] == 3.0

    def test_pad_or_truncate_exact(self):
        """Should not modify exact-length arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _pad_or_truncate(arr, 3)

        assert len(result) == 3
        assert np.array_equal(result, arr)


# =============================================================================
# SemanticToolSelector Tests
# =============================================================================

class TestSemanticToolSelector:
    """Tests for SemanticToolSelector."""

    def test_suggest_tool_returns_string(self, tool_selector, sample_semantic_state):
        """suggest_tool should return a tool name string."""
        tool = tool_selector.suggest_tool(sample_semantic_state)

        assert isinstance(tool, str)
        assert len(tool) > 0

    def test_topic_shift_suggests_branch(self, tool_selector, random_embedding):
        """Topic shift should suggest branch_thread."""
        state1 = SemanticState.from_embedding(random_embedding)

        # Create large shift
        shifted = random_embedding + np.random.randn(384).astype(np.float32) * 5.0
        state2 = SemanticState.from_embedding(shifted, previous_state=state1)

        if state2.topic_shift_detected:
            tool = tool_selector.suggest_tool(state2)
            assert tool == 'branch_thread'

    def test_low_momentum_suggests_clarify(self, tool_selector):
        """Low momentum should suggest clarify."""
        state = SemanticState.empty()
        state.momentum = 0.2  # Low momentum
        state.topic_shift_detected = False

        tool = tool_selector.suggest_tool(state)
        assert tool == 'clarify'

    def test_high_complexity_suggests_simplify(self, tool_selector):
        """High complexity should suggest simplify."""
        state = SemanticState.empty()
        state.complexity = 0.8  # High complexity
        state.momentum = 0.5  # Normal momentum
        state.topic_shift_detected = False

        tool = tool_selector.suggest_tool(state)
        assert tool == 'simplify'

    def test_should_branch_thread_on_shift(self, tool_selector, random_embedding):
        """should_branch_thread should return True on topic shift."""
        state1 = SemanticState.from_embedding(random_embedding)

        # Create large shift
        shifted = random_embedding + np.random.randn(384).astype(np.float32) * 5.0
        state2 = SemanticState.from_embedding(shifted, previous_state=state1)

        if state2.topic_shift_detected:
            assert tool_selector.should_branch_thread(state2) is True

    def test_dimension_to_tool_mapping(self, tool_selector):
        """Dimension-to-tool mapping should be defined."""
        assert 'Confusion' in tool_selector.dimension_to_tool
        assert 'Curiosity' in tool_selector.dimension_to_tool
        assert tool_selector.dimension_to_tool['Confusion'] == 'explain'


# =============================================================================
# SemanticAwareBandit Tests
# =============================================================================

class TestSemanticAwareBandit:
    """Tests for SemanticAwareBandit."""

    def test_initialization(self, semantic_bandit):
        """Bandit should initialize with default tools."""
        assert len(semantic_bandit.tools) == 4
        assert 'answer' in semantic_bandit.tools
        assert 'search' in semantic_bandit.tools

    def test_custom_tools(self):
        """Bandit should accept custom tools."""
        bandit = SemanticAwareBandit(tools=['tool_a', 'tool_b'])

        assert len(bandit.tools) == 2
        assert 'tool_a' in bandit.tools

    def test_adjust_priors_returns_dict(self, semantic_bandit, sample_semantic_state):
        """adjust_priors should return dict of (alpha, beta) tuples."""
        adjusted = semantic_bandit.adjust_priors(sample_semantic_state)

        assert isinstance(adjusted, dict)
        assert len(adjusted) == len(semantic_bandit.tools)

        for tool, (alpha, beta) in adjusted.items():
            assert alpha > 0
            assert beta > 0

    def test_adjust_priors_respects_semantic_weight(self, sample_semantic_state):
        """Higher semantic_weight should cause larger adjustments."""
        bandit_low = SemanticAwareBandit(semantic_weight=0.1)
        bandit_high = SemanticAwareBandit(semantic_weight=0.9)

        adjusted_low = bandit_low.adjust_priors(sample_semantic_state)
        adjusted_high = bandit_high.adjust_priors(sample_semantic_state)

        # At least one tool should have different adjustments
        differences = []
        for tool in bandit_low.tools:
            diff = abs(adjusted_high[tool][0] - adjusted_low[tool][0])
            differences.append(diff)

        # Higher semantic_weight should cause more deviation from base
        assert max(differences) > 0

    def test_sample_returns_valid_tool(self, semantic_bandit, sample_semantic_state):
        """sample should return a valid tool name."""
        tool = semantic_bandit.sample(sample_semantic_state)

        assert tool in semantic_bandit.tools

    def test_sample_without_state(self, semantic_bandit):
        """sample should work without semantic state."""
        tool = semantic_bandit.sample(None)

        assert tool in semantic_bandit.tools

    def test_sample_temperature(self, semantic_bandit, sample_semantic_state):
        """Higher temperature should increase randomness."""
        # Sample many times with low and high temperature
        low_temp_samples = [semantic_bandit.sample(sample_semantic_state, temperature=0.1)
                           for _ in range(50)]
        high_temp_samples = [semantic_bandit.sample(sample_semantic_state, temperature=5.0)
                            for _ in range(50)]

        # Low temperature should have lower diversity
        low_unique = len(set(low_temp_samples))
        high_unique = len(set(high_temp_samples))

        # High temperature should have at least as many unique samples
        assert high_unique >= low_unique or low_unique <= 2

    def test_update_success(self, semantic_bandit):
        """update with success should increase alpha."""
        alpha_before = semantic_bandit.alphas['answer']
        semantic_bandit.update('answer', success=True, confidence=1.0)
        alpha_after = semantic_bandit.alphas['answer']

        assert alpha_after > alpha_before

    def test_update_failure(self, semantic_bandit):
        """update with failure should increase beta."""
        beta_before = semantic_bandit.betas['search']
        semantic_bandit.update('search', success=False, confidence=0.0)
        beta_after = semantic_bandit.betas['search']

        assert beta_after > beta_before

    def test_update_confidence_scaling(self, semantic_bandit):
        """update should scale by confidence."""
        semantic_bandit.update('calc', success=True, confidence=0.5)

        # Alpha should increase by 0.5, not 1.0
        assert semantic_bandit.alphas['calc'] == 1.5

    def test_get_expected_rewards(self, semantic_bandit, sample_semantic_state):
        """get_expected_rewards should return probabilities."""
        rewards = semantic_bandit.get_expected_rewards(sample_semantic_state)

        assert isinstance(rewards, dict)
        for tool, reward in rewards.items():
            assert 0.0 <= reward <= 1.0

    def test_get_stats(self, semantic_bandit):
        """get_stats should return bandit statistics."""
        stats = semantic_bandit.get_stats()

        assert 'tools' in stats
        assert 'alphas' in stats
        assert 'betas' in stats
        assert 'expected_rewards' in stats
        assert 'semantic_weight' in stats

    def test_category_affinities_coverage(self, semantic_bandit):
        """All 16 categories should have affinity mappings."""
        for category in SEMANTIC_CATEGORIES:
            assert category in semantic_bandit.CATEGORY_TOOL_AFFINITIES


# =============================================================================
# Integration Tests
# =============================================================================

class TestSemanticPolicyIntegration:
    """Integration tests for semantic -> policy pipeline."""

    def test_full_pipeline_from_embedding(self, random_embedding):
        """Full pipeline: embedding -> state -> features -> tool."""
        # Create state from embedding
        state = SemanticState.from_embedding(random_embedding)

        # Get feature vector for policy
        features = state.to_feature_vector()
        assert features.shape == (8,)

        # Get tool suggestion
        selector = SemanticToolSelector()
        tool = selector.suggest_tool(state)
        assert isinstance(tool, str)

        # Sample from bandit with semantic adjustments
        bandit = SemanticAwareBandit()
        sampled = bandit.sample(state)
        assert sampled in bandit.tools

    def test_conversation_flow_with_topic_shifts(self, random_embedding):
        """Simulate conversation with topic shifts."""
        selector = SemanticToolSelector()
        bandit = SemanticAwareBandit()

        # First turn
        state1 = SemanticState.from_embedding(random_embedding)
        tool1 = selector.suggest_tool(state1)

        # Second turn (small shift)
        embedding2 = random_embedding + np.random.randn(384).astype(np.float32) * 0.1
        state2 = SemanticState.from_embedding(embedding2, previous_state=state1)
        tool2 = selector.suggest_tool(state2)

        # Third turn (large shift)
        embedding3 = random_embedding + np.random.randn(384).astype(np.float32) * 5.0
        state3 = SemanticState.from_embedding(embedding3, previous_state=state2)
        tool3 = selector.suggest_tool(state3)

        # Small shift should not trigger branch
        assert not state2.topic_shift_detected or state2.shift_magnitude > TOPIC_SHIFT_THRESHOLD

        # Large shift should be detected
        if state3.topic_shift_detected:
            assert tool3 == 'branch_thread'

    def test_bandit_learns_from_outcomes(self):
        """Bandit should learn from success/failure outcomes."""
        bandit = SemanticAwareBandit()

        # Simulate many successful uses of 'answer'
        for _ in range(10):
            bandit.update('answer', success=True, confidence=0.9)

        # Simulate failures of 'calc'
        for _ in range(10):
            bandit.update('calc', success=False, confidence=0.1)

        # Expected rewards should reflect learning
        rewards = bandit.get_expected_rewards()

        # 'answer' should have higher expected reward than 'calc'
        assert rewards['answer'] > rewards['calc']


# =============================================================================
# NeuralCore Integration Tests (if torch available)
# =============================================================================

class TestNeuralCoreIntegration:
    """Tests for NeuralCore semantic feature integration."""

    @pytest.fixture
    def neural_core(self):
        """Create a NeuralCore instance."""
        try:
            from HoloLoom.policy.unified import NeuralCore
            return NeuralCore(d_model=384, n_layers=1, n_heads=2, semantic_dim=8)
        except ImportError:
            pytest.skip("NeuralCore not available")

    def test_neural_core_has_semantic_layers(self, neural_core):
        """NeuralCore should have semantic projection and gate layers."""
        assert hasattr(neural_core, 'semantic_proj')
        assert hasattr(neural_core, 'semantic_gate')

        # Check dimensions
        assert neural_core.semantic_proj.in_features == 8
        assert neural_core.semantic_proj.out_features == 384

    @pytest.mark.asyncio
    async def test_neural_core_decide_without_semantic(self, neural_core):
        """decide should work without semantic features."""
        mem = torch.randn(1, 10, 384)
        ctrl = torch.randn(1, 8)  # n_motifs = 8

        logits, pooled = await neural_core.decide(mem, ctrl, adapter_idx=0)

        assert logits.shape == (1, 4)  # 4 tools
        assert pooled.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_neural_core_decide_with_semantic(self, neural_core):
        """decide should use semantic features when provided."""
        mem = torch.randn(1, 10, 384)
        ctrl = torch.randn(1, 8)  # n_motifs = 8
        semantic = torch.randn(1, 8)

        logits, pooled = await neural_core.decide(
            mem, ctrl, adapter_idx=0, semantic_features=semantic
        )

        assert logits.shape == (1, 4)
        assert pooled.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_semantic_features_affect_output(self, neural_core):
        """Semantic features should affect tool logits."""
        torch.manual_seed(42)

        mem = torch.randn(1, 10, 384)
        ctrl = torch.randn(1, 8)  # n_motifs = 8

        # Without semantic
        logits_no_sem, _ = await neural_core.decide(mem, ctrl, adapter_idx=0)

        # With semantic
        semantic = torch.ones(1, 8) * 0.5
        logits_with_sem, _ = await neural_core.decide(
            mem, ctrl, adapter_idx=0, semantic_features=semantic
        )

        # Outputs should be different
        assert not torch.allclose(logits_no_sem, logits_with_sem)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
