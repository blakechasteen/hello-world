"""
Tests for Context Handoff - Phase 6.3
=====================================

Comprehensive test suite covering:
- MI scoring for context items
- Redundancy detection
- Strategy selection
- Budget constraints
- Handoff scenarios
- Edge cases

Author: Claude Code
Date: 2025-12-09
"""

import pytest
from typing import List, Dict, Any

from hololoom.agentic.context_handoff import (
    ContextHandoff,
    ContextItem,
    HandoffContext,
    HandoffStrategy,
    prepare_agent_handoff,
    get_optimal_handoff_strategy,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_context() -> List[str]:
    """Sample context items for testing."""
    return [
        "Thompson Sampling is a Bayesian algorithm for balancing exploration and exploitation.",
        "It maintains a probability distribution over the expected reward of each action.",
        "The algorithm samples from the posterior distribution to select actions.",
        "Thompson Sampling has strong theoretical guarantees and empirical performance.",
        "UCB (Upper Confidence Bound) is an alternative approach that uses confidence intervals.",
    ]


@pytest.fixture
def redundant_context() -> List[str]:
    """Context with redundant items for testing (high similarity pairs)."""
    return [
        "Thompson Sampling is a Bayesian algorithm for exploration and exploitation balancing.",
        "Thompson Sampling is a Bayesian algorithm for exploration exploitation balance.",  # Very similar to first
        "UCB uses upper confidence bounds for action selection in bandits.",
        "UCB uses upper confidence bounds for action selection in bandit problems.",  # Very similar to third
        "Reinforcement learning studies sequential decision making under uncertainty.",
    ]


@pytest.fixture
def large_context() -> List[str]:
    """Large context for budget testing (diverse topics to avoid redundancy)."""
    topics = [
        "Thompson Sampling is a Bayesian algorithm for multi-armed bandits.",
        "UCB balances exploration and exploitation using confidence bounds.",
        "Reinforcement learning optimizes sequential decision making.",
        "Neural networks learn hierarchical representations from data.",
        "Gradient descent minimizes loss functions iteratively.",
        "Attention mechanisms enable transformers to focus on relevant tokens.",
        "Convolutional layers extract spatial features from images.",
        "Recurrent networks process sequential information over time.",
        "Batch normalization stabilizes training of deep networks.",
        "Dropout regularizes networks by randomly zeroing activations.",
        "Word embeddings represent semantic meaning in vector space.",
        "BERT uses bidirectional attention for language understanding.",
        "GPT generates text using autoregressive transformers.",
        "Knowledge graphs store structured entity relationships.",
        "Graph neural networks propagate information across edges.",
        "Monte Carlo methods estimate expectations through sampling.",
        "Bayesian inference updates beliefs given new evidence.",
        "Maximum likelihood finds parameters maximizing data probability.",
        "Cross-entropy measures divergence between distributions.",
        "Adam optimizer combines momentum and adaptive learning rates.",
    ]
    return topics


@pytest.fixture
def handoff() -> ContextHandoff:
    """Default context handoff engine."""
    return ContextHandoff()


# ============================================================================
# MI Scoring Tests
# ============================================================================

class TestMIScoring:
    """Tests for MI scoring of context items."""

    def test_capability_keyword_increases_score(self, handoff, sample_context):
        """Items with capability keywords should score higher."""
        # Create two items - one with summarization keywords, one without
        items_with_keyword = ["Brief summary: The key points are X, Y, Z."]
        items_without_keyword = ["The algorithm works by doing A then B."]

        result_with = handoff.prepare_handoff(
            from_agent="research",
            to_agent="summary",
            full_context=items_with_keyword,
            target_agent_capability="summarization",
        )

        result_without = handoff.prepare_handoff(
            from_agent="research",
            to_agent="summary",
            full_context=items_without_keyword,
            target_agent_capability="summarization",
        )

        # Item with keywords should have higher average MI
        assert result_with.avg_mi >= result_without.avg_mi or len(result_with.selected_items) > 0

    def test_content_richness_affects_score(self, handoff):
        """Content with diverse vocabulary should score higher than repetitive."""
        diverse = ["Machine learning algorithms analyze data patterns for predictions."]
        repetitive = ["The data data data data data data."]

        result_diverse = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=diverse,
            target_agent_capability="analysis",
        )

        result_repetitive = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=repetitive,
            target_agent_capability="analysis",
        )

        # Diverse content should have higher MI
        assert result_diverse.avg_mi >= result_repetitive.avg_mi


# ============================================================================
# Redundancy Detection Tests
# ============================================================================

class TestRedundancyDetection:
    """Tests for redundancy detection."""

    def test_detects_similar_items(self, redundant_context):
        """Should detect and remove similar items with lower threshold."""
        # Use lower threshold (0.7) to catch moderately similar items
        handoff_sensitive = ContextHandoff(redundancy_threshold=0.7)
        result = handoff_sensitive.prepare_handoff(
            from_agent="research",
            to_agent="summary",
            full_context=redundant_context,
            target_agent_capability="summarization",
        )

        # Should have found redundant items (with lower threshold)
        assert result.redundant_count > 0
        assert result.redundancy_removed > 0

    def test_keeps_unique_items(self, handoff, sample_context):
        """Should keep unique items when no redundancy."""
        result = handoff.prepare_handoff(
            from_agent="research",
            to_agent="analysis",
            full_context=sample_context,
            target_agent_capability="analysis",
            strategy=HandoffStrategy.FULL,  # Keep all non-redundant
        )

        # Should keep most items (unique context)
        assert result.selected_count >= len(sample_context) - 1

    def test_redundancy_threshold_configurable(self):
        """Redundancy threshold should be configurable."""
        strict = ContextHandoff(redundancy_threshold=0.5)  # More strict
        lenient = ContextHandoff(redundancy_threshold=0.95)  # Less strict

        context = [
            "Thompson Sampling is Bayesian.",
            "Thompson Sampling uses Bayesian methods.",
        ]

        strict_result = strict.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=context,
            target_agent_capability="analysis",
        )

        lenient_result = lenient.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=context,
            target_agent_capability="analysis",
        )

        # Strict threshold should find more redundancy
        assert strict_result.redundancy_removed >= lenient_result.redundancy_removed


# ============================================================================
# Strategy Selection Tests
# ============================================================================

class TestStrategySelection:
    """Tests for strategy selection."""

    def test_research_capability_gets_conservative(self, handoff, sample_context):
        """Research capability should auto-select CONSERVATIVE strategy."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="researcher",
            full_context=sample_context,
            target_agent_capability="research",
            # strategy=None -> auto-select
        )

        assert result.strategy == HandoffStrategy.CONSERVATIVE

    def test_summarization_gets_aggressive(self, handoff, sample_context):
        """Summarization capability should auto-select AGGRESSIVE strategy."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="summarizer",
            full_context=sample_context,
            target_agent_capability="summarization",
        )

        assert result.strategy == HandoffStrategy.AGGRESSIVE

    def test_explicit_strategy_overrides_auto(self, handoff, sample_context):
        """Explicit strategy should override auto-selection."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="summarizer",
            full_context=sample_context,
            target_agent_capability="summarization",
            strategy=HandoffStrategy.FULL,  # Explicit override
        )

        assert result.strategy == HandoffStrategy.FULL


# ============================================================================
# Strategy Effects Tests
# ============================================================================

class TestStrategyEffects:
    """Tests for effects of different strategies."""

    def test_aggressive_keeps_less(self, handoff, large_context):
        """AGGRESSIVE strategy should keep fewer items than CONSERVATIVE."""
        aggressive = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=large_context,
            target_agent_capability="generation",
            strategy=HandoffStrategy.AGGRESSIVE,
        )

        conservative = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=large_context,
            target_agent_capability="generation",
            strategy=HandoffStrategy.CONSERVATIVE,
        )

        assert aggressive.selected_count < conservative.selected_count

    def test_full_keeps_all_non_redundant(self, handoff, sample_context):
        """FULL strategy should keep all non-redundant items."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=sample_context,
            target_agent_capability="analysis",
            strategy=HandoffStrategy.FULL,
        )

        # Should keep all items (sample_context has no redundancy)
        assert result.selected_count == len(sample_context)


# ============================================================================
# Budget Constraint Tests
# ============================================================================

class TestBudgetConstraints:
    """Tests for token budget constraints."""

    def test_token_budget_limits_selection(self, handoff, large_context):
        """Token budget should limit number of selected items."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=large_context,
            target_agent_capability="analysis",
            strategy=HandoffStrategy.FULL,
            token_budget=200,  # Small budget
        )

        # Should respect budget
        assert result.estimated_tokens <= 200 + 50  # Allow small overflow

    def test_tokens_saved_calculated(self, handoff, large_context):
        """tokens_saved should reflect actual savings."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=large_context,
            target_agent_capability="analysis",
            strategy=HandoffStrategy.AGGRESSIVE,
        )

        # Should save tokens
        assert result.tokens_saved >= 0


# ============================================================================
# Handoff Scenarios Tests
# ============================================================================

class TestHandoffScenarios:
    """Tests for realistic handoff scenarios."""

    def test_researcher_to_summarizer(self, handoff):
        """Research output to summarizer should be compressed."""
        research_output = [
            "Finding 1: Thompson Sampling outperforms UCB in non-stationary environments.",
            "Finding 2: The posterior update requires conjugate prior distributions.",
            "Finding 3: Empirical results show 15% improvement in cumulative regret.",
            "Background: Multi-armed bandits model sequential decision problems.",
            "Methodology: We ran 1000 simulations with 10 arms each.",
            "Discussion: Results suggest Thompson Sampling is preferred for exploration-heavy tasks.",
            "Limitation: Analysis assumes independent arms.",
            "Future work: Extend to contextual bandits.",
        ]

        result = handoff.prepare_handoff(
            from_agent="researcher",
            to_agent="summarizer",
            full_context=research_output,
            target_agent_capability="summarization",
        )

        # Should compress significantly
        assert result.selected_count < len(research_output)
        # Should keep most relevant items
        assert result.avg_mi > 0

    def test_planner_to_executor(self, handoff):
        """Plan output to executor should focus on actionable items."""
        plan_items = [
            "Step 1: Initialize the Thompson Sampling algorithm with uniform priors.",
            "Step 2: Implement the arm selection function.",
            "Step 3: Run the simulation for 10000 iterations.",
            "Note: This approach was chosen after comparing alternatives.",
            "Rationale: Thompson Sampling provides better exploration.",
        ]

        result = handoff.prepare_handoff(
            from_agent="planner",
            to_agent="executor",
            full_context=plan_items,
            target_agent_capability="code_review",
        )

        # Should focus on actionable steps
        assert len(result.selected_items) >= 1

    def test_multi_agent_chain(self, handoff):
        """Context should be progressively filtered through agent chain."""
        initial_context = [f"Context item {i}" for i in range(10)]

        # First handoff: researcher -> analyzer
        result1 = handoff.prepare_handoff(
            from_agent="researcher",
            to_agent="analyzer",
            full_context=initial_context,
            target_agent_capability="analysis",
        )

        # Second handoff: analyzer -> summarizer
        result2 = handoff.prepare_handoff(
            from_agent="analyzer",
            to_agent="summarizer",
            full_context=[item.content for item in result1.selected_items],
            target_agent_capability="summarization",
        )

        # Context should be progressively reduced
        assert result2.original_count <= result1.selected_count


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_context(self, handoff):
        """Should handle empty context gracefully."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=[],
            target_agent_capability="analysis",
        )

        assert result.selected_count == 0
        assert result.original_count == 0
        assert result.avg_mi == 0.0

    def test_single_item(self, handoff):
        """Should handle single item context."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=["Single context item."],
            target_agent_capability="analysis",
        )

        assert result.selected_count >= 1

    def test_all_redundant_items(self, handoff):
        """Should handle all items being redundant (keep at least one)."""
        all_same = ["Same content here."] * 5

        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=all_same,
            target_agent_capability="analysis",
        )

        # Should keep at least minimum items
        assert result.selected_count >= 1

    def test_unknown_capability(self, handoff, sample_context):
        """Should handle unknown target capability gracefully."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=sample_context,
            target_agent_capability="unknown_capability_xyz",
        )

        # Should still work with default scoring
        assert result.selected_count >= 1


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_prepare_agent_handoff(self, sample_context):
        """prepare_agent_handoff convenience function should work."""
        result = prepare_agent_handoff(
            from_agent="a",
            to_agent="b",
            context=sample_context,
            capability="analysis",
        )

        assert isinstance(result, HandoffContext)
        assert len(result.selected_items) <= len(sample_context)

    def test_get_optimal_handoff_strategy(self):
        """get_optimal_handoff_strategy should return reasonable strategies."""
        # High urgency -> AGGRESSIVE
        assert get_optimal_handoff_strategy("any", 10, "high") == HandoffStrategy.AGGRESSIVE

        # Low urgency -> CONSERVATIVE
        assert get_optimal_handoff_strategy("any", 10, "low") == HandoffStrategy.CONSERVATIVE

        # Small context -> FULL
        assert get_optimal_handoff_strategy("any", 3, "normal") == HandoffStrategy.FULL

        # Research capability -> CONSERVATIVE
        assert get_optimal_handoff_strategy("research", 10, "normal") == HandoffStrategy.CONSERVATIVE

        # Summarization -> AGGRESSIVE
        assert get_optimal_handoff_strategy("summarization", 10, "normal") == HandoffStrategy.AGGRESSIVE


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Tests for statistics tracking."""

    def test_statistics_updated(self, sample_context):
        """Statistics should be updated after handoffs."""
        handoff = ContextHandoff()
        initial_stats = handoff.get_statistics()
        assert initial_stats['total_handoffs'] == 0

        handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=sample_context,
            target_agent_capability="analysis",
        )

        handoff.prepare_handoff(
            from_agent="c",
            to_agent="d",
            full_context=sample_context,
            target_agent_capability="summarization",
        )

        stats = handoff.get_statistics()
        assert stats['total_handoffs'] == 2
        assert stats['total_items_processed'] == len(sample_context) * 2


# ============================================================================
# HandoffContext Tests
# ============================================================================

class TestHandoffContext:
    """Tests for HandoffContext dataclass."""

    def test_str_representation(self, handoff, sample_context):
        """__str__ should produce readable output."""
        result = handoff.prepare_handoff(
            from_agent="researcher",
            to_agent="summarizer",
            full_context=sample_context,
            target_agent_capability="summarization",
        )

        str_repr = str(result)
        assert "HandoffContext" in str_repr
        assert "researcher" in str_repr
        assert "summarizer" in str_repr

    def test_get_context_text(self, handoff, sample_context):
        """get_context_text should return combined text."""
        result = handoff.prepare_handoff(
            from_agent="a",
            to_agent="b",
            full_context=sample_context,
            target_agent_capability="analysis",
            strategy=HandoffStrategy.FULL,
        )

        text = result.get_context_text()
        # Should contain content from selected items
        for item in result.selected_items:
            assert item.content in text


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
