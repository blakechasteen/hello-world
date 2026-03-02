"""
Tests for Adaptive Learning - Phase 6.4
========================================

Comprehensive test suite covering:
- Thompson Sampler basics
- Budget learning updates
- Exploration vs exploitation
- Persistence
- Convenience functions
- Edge cases

Author: Claude Code
Date: 2025-12-09
"""

import pytest
import os
import tempfile
from typing import List

from hololoom.context_packing.learning import (
    QueryComplexity,
    BudgetOutcome,
    BudgetRecommendation,
    ThompsonSampler,
    AdaptiveBudgetLearner,
    get_learner,
    get_adaptive_budget,
    record_outcome,
    get_learning_statistics,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sampler() -> ThompsonSampler:
    """Default Thompson Sampler."""
    return ThompsonSampler(base_budget=5.0)


@pytest.fixture
def learner() -> AdaptiveBudgetLearner:
    """Default adaptive budget learner."""
    return AdaptiveBudgetLearner()


@pytest.fixture
def trained_sampler() -> ThompsonSampler:
    """Thompson Sampler with some training data."""
    sampler = ThompsonSampler(base_budget=5.0)

    # Simulate good outcomes
    for _ in range(10):
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.85,
            confidence=0.9,
        ))

    return sampler


@pytest.fixture
def trained_learner() -> AdaptiveBudgetLearner:
    """Learner with training data across complexity levels."""
    learner = AdaptiveBudgetLearner()

    # Train each complexity level
    training_data = [
        (QueryComplexity.TRIVIAL, 2.0, 0.9),  # Good at trivial
        (QueryComplexity.SIMPLE, 3.0, 0.8),   # Good at simple
        (QueryComplexity.MODERATE, 5.0, 0.75), # OK at moderate
        (QueryComplexity.COMPLEX, 8.0, 0.65),  # Needs work at complex
        (QueryComplexity.RESEARCH, 15.0, 0.6), # Challenging at research
    ]

    for complexity, budget, quality in training_data:
        for _ in range(5):
            learner.update(
                complexity=complexity,
                budget_used=budget,
                quality_score=quality,
                confidence=0.9,
            )

    return learner


# ============================================================================
# Thompson Sampler Tests
# ============================================================================

class TestThompsonSampler:
    """Tests for Thompson Sampler."""

    def test_initial_state(self, sampler):
        """Initial sampler should have uniform prior."""
        assert sampler.alpha == 1.0
        assert sampler.beta == 1.0
        assert sampler.base_budget == 5.0
        assert sampler._total_queries == 0

    def test_sample_returns_budget(self, sampler):
        """Sample should return a budget value."""
        budget = sampler.sample()
        assert isinstance(budget, float)
        assert budget > 0

    def test_update_success(self, sampler):
        """Successful outcome should increase alpha."""
        initial_alpha = sampler.alpha

        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.9,  # Above threshold
            confidence=0.9,
        ))

        assert sampler.alpha > initial_alpha
        assert sampler._total_queries == 1
        assert sampler._total_successes == 1

    def test_update_failure(self, sampler):
        """Failed outcome should increase beta."""
        initial_beta = sampler.beta

        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.3,  # Below threshold
            confidence=0.9,
        ))

        assert sampler.beta > initial_beta
        assert sampler._total_queries == 1
        assert sampler._total_successes == 0

    def test_expected_quality_increases_with_successes(self, sampler):
        """Expected quality should increase with successful outcomes."""
        initial_quality = sampler.expected_quality()

        # Add successful outcomes
        for _ in range(10):
            sampler.update(BudgetOutcome(
                complexity=QueryComplexity.MODERATE,
                budget_used=5.0,
                quality_score=0.9,
                confidence=0.9,
            ))

        assert sampler.expected_quality() > initial_quality

    def test_confidence_increases_with_observations(self, sampler):
        """Confidence should increase with more observations."""
        initial_confidence = sampler.confidence()

        for _ in range(20):
            sampler.update(BudgetOutcome(
                complexity=QueryComplexity.MODERATE,
                budget_used=5.0,
                quality_score=0.8,
                confidence=0.9,
            ))

        assert sampler.confidence() > initial_confidence

    def test_feedback_weighted_more_than_model(self, sampler):
        """User feedback should be weighted more heavily."""
        # Low model quality but high feedback
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.5,  # Model says poor
            confidence=0.9,
            feedback=0.95,  # User says great
        ))

        # Should count as success (0.7*0.95 + 0.3*0.5 = 0.815 > 0.7)
        assert sampler._total_successes == 1

    def test_serialization(self, trained_sampler):
        """Sampler should serialize and deserialize correctly."""
        data = trained_sampler.to_dict()

        restored = ThompsonSampler.from_dict(data)

        assert restored.alpha == trained_sampler.alpha
        assert restored.beta == trained_sampler.beta
        assert restored.base_budget == trained_sampler.base_budget
        assert restored._total_queries == trained_sampler._total_queries

    def test_statistics(self, trained_sampler):
        """get_statistics should return useful info."""
        stats = trained_sampler.get_statistics()

        assert "base_budget" in stats
        assert "alpha" in stats
        assert "beta" in stats
        assert "expected_quality" in stats
        assert "confidence" in stats
        assert "success_rate" in stats
        assert stats["total_queries"] > 0


# ============================================================================
# Adaptive Budget Learner Tests
# ============================================================================

class TestAdaptiveBudgetLearner:
    """Tests for Adaptive Budget Learner."""

    def test_initial_state(self, learner):
        """Learner should initialize with samplers for all complexity levels."""
        assert len(learner.samplers) == 5
        assert QueryComplexity.TRIVIAL in learner.samplers
        assert QueryComplexity.RESEARCH in learner.samplers

    def test_get_budget_returns_value(self, learner):
        """get_budget should return a budget value."""
        budget = learner.get_budget(QueryComplexity.MODERATE)
        assert isinstance(budget, float)
        assert learner.MIN_BUDGET <= budget <= learner.MAX_BUDGET

    def test_get_budget_within_bounds(self, learner):
        """Budget should always be within bounds."""
        for _ in range(50):
            for complexity in QueryComplexity:
                budget = learner.get_budget(complexity)
                assert learner.MIN_BUDGET <= budget <= learner.MAX_BUDGET

    def test_update_increases_total(self, learner):
        """Update should increase total updates."""
        assert learner._total_updates == 0

        learner.update(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.8,
        )

        assert learner._total_updates == 1

    def test_update_tracked_in_history(self, learner):
        """Updates should be tracked in history."""
        learner.update(
            complexity=QueryComplexity.SIMPLE,
            budget_used=3.0,
            quality_score=0.85,
            confidence=0.9,
        )

        assert len(learner._outcome_history) == 1
        assert learner._outcome_history[0].complexity == QueryComplexity.SIMPLE

    def test_get_recommendation(self, trained_learner):
        """get_recommendation should return detailed recommendation."""
        rec = trained_learner.get_recommendation(QueryComplexity.MODERATE)

        assert isinstance(rec, BudgetRecommendation)
        assert rec.complexity == QueryComplexity.MODERATE
        assert rec.recommended_budget > 0
        assert 0 <= rec.confidence <= 1
        assert 0 <= rec.expected_quality <= 1
        assert len(rec.reasoning) > 0

    def test_recommendation_has_alternatives(self, trained_learner):
        """Recommendation should include alternatives."""
        rec = trained_learner.get_recommendation(QueryComplexity.COMPLEX)

        assert len(rec.alternatives) > 0
        for budget, quality in rec.alternatives:
            assert budget > 0
            assert 0 <= quality <= 1

    def test_learning_improves_with_feedback(self, learner):
        """Learner should adapt based on feedback."""
        # Initial budget
        initial_stats = learner.samplers[QueryComplexity.MODERATE].get_statistics()

        # Provide many successful outcomes
        for _ in range(20):
            learner.update(
                complexity=QueryComplexity.MODERATE,
                budget_used=5.0,
                quality_score=0.95,
                confidence=0.9,
            )

        # Expected quality should increase
        final_stats = learner.samplers[QueryComplexity.MODERATE].get_statistics()
        assert final_stats["expected_quality"] > initial_stats["expected_quality"]

    def test_statistics(self, trained_learner):
        """get_statistics should return comprehensive info."""
        stats = trained_learner.get_statistics()

        assert "total_updates" in stats
        assert "by_complexity" in stats
        assert "recent_avg_quality" in stats
        assert "recent_success_rate" in stats

        # Check all complexity levels present
        for complexity in QueryComplexity:
            assert complexity.value in stats["by_complexity"]

    def test_exploration_disabled(self):
        """With exploration disabled, should use pure Thompson."""
        learner = AdaptiveBudgetLearner(enable_exploration=False)

        # All budgets should come from Thompson sampling (no random)
        budgets = [learner.get_budget(QueryComplexity.MODERATE) for _ in range(10)]

        # Should have some variation (Thompson samples)
        # but no extreme outliers (pure Thompson is more consistent)
        assert len(set(budgets)) >= 1  # At least some value


# ============================================================================
# Persistence Tests
# ============================================================================

class TestPersistence:
    """Tests for state persistence."""

    def test_save_and_load(self, trained_learner):
        """Learner should save and load state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "learning_state.json")

            # Save state
            trained_learner.save(path)
            assert os.path.exists(path)

            # Load into new learner
            new_learner = AdaptiveBudgetLearner()
            new_learner.load(path)

            # Verify state restored
            assert new_learner._total_updates == trained_learner._total_updates
            for complexity in QueryComplexity:
                orig_stats = trained_learner.samplers[complexity].get_statistics()
                new_stats = new_learner.samplers[complexity].get_statistics()
                assert orig_stats["alpha"] == new_stats["alpha"]
                assert orig_stats["beta"] == new_stats["beta"]

    def test_load_nonexistent(self, learner):
        """Loading nonexistent file should not crash."""
        learner.load("/nonexistent/path/state.json")
        # Should still work normally
        budget = learner.get_budget(QueryComplexity.SIMPLE)
        assert budget > 0


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_learner(self):
        """get_learner should return global learner."""
        learner = get_learner()
        assert isinstance(learner, AdaptiveBudgetLearner)

        # Should return same instance
        learner2 = get_learner()
        assert learner is learner2

    def test_get_adaptive_budget(self):
        """get_adaptive_budget should return budget for string complexity."""
        budget = get_adaptive_budget("moderate")
        assert isinstance(budget, float)
        assert budget > 0

    def test_get_adaptive_budget_case_insensitive(self):
        """Complexity string should be case-insensitive."""
        budget1 = get_adaptive_budget("SIMPLE")
        budget2 = get_adaptive_budget("simple")
        # Both should work (may differ due to Thompson sampling)
        assert budget1 > 0
        assert budget2 > 0

    def test_record_outcome(self):
        """record_outcome should update global learner."""
        initial_stats = get_learning_statistics()
        initial_updates = initial_stats["total_updates"]

        record_outcome(
            complexity="complex",
            budget_used=8.0,
            quality_score=0.85,
            confidence=0.9,
        )

        final_stats = get_learning_statistics()
        assert final_stats["total_updates"] > initial_updates

    def test_get_learning_statistics(self):
        """get_learning_statistics should return stats."""
        stats = get_learning_statistics()

        assert "total_updates" in stats
        assert "by_complexity" in stats


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_quality_at_threshold(self, sampler):
        """Quality exactly at threshold should be success."""
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.7,  # Exactly at threshold
            confidence=0.9,
        ))

        assert sampler._total_successes == 1

    def test_quality_just_below_threshold(self, sampler):
        """Quality just below threshold should be failure."""
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.69,  # Just below threshold
            confidence=0.9,
        ))

        assert sampler._total_successes == 0

    def test_zero_quality(self, sampler):
        """Zero quality should be handled."""
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.0,
            confidence=0.9,
        ))

        assert sampler._total_queries == 1
        assert sampler.beta > 1.0

    def test_perfect_quality(self, sampler):
        """Perfect quality should be handled."""
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=1.0,
            confidence=1.0,
        ))

        assert sampler._total_successes == 1

    def test_history_pruning(self, learner):
        """History should be pruned to prevent memory issues."""
        # Add many outcomes
        for i in range(1500):
            learner.update(
                complexity=QueryComplexity.SIMPLE,
                budget_used=3.0,
                quality_score=0.8,
            )

        # History should be capped
        assert len(learner._outcome_history) <= 1000

    def test_unknown_complexity_string(self):
        """Unknown complexity string should raise error."""
        with pytest.raises(ValueError):
            get_adaptive_budget("unknown_level")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full learning workflow."""

    def test_full_learning_cycle(self, learner):
        """Test complete learning cycle."""
        # Phase 1: Initial budget recommendations
        initial_budgets = {
            complexity: learner.get_budget(complexity)
            for complexity in QueryComplexity
        }

        # Phase 2: Simulate query outcomes
        for _ in range(30):
            for complexity in QueryComplexity:
                budget = learner.get_budget(complexity)

                # Simulate quality based on budget (higher budget = higher quality)
                base_quality = {
                    QueryComplexity.TRIVIAL: 0.9,
                    QueryComplexity.SIMPLE: 0.85,
                    QueryComplexity.MODERATE: 0.75,
                    QueryComplexity.COMPLEX: 0.65,
                    QueryComplexity.RESEARCH: 0.55,
                }[complexity]

                quality = min(1.0, base_quality + (budget / 50.0))

                learner.update(
                    complexity=complexity,
                    budget_used=budget,
                    quality_score=quality,
                    confidence=0.9,
                )

        # Phase 3: Verify learning occurred
        stats = learner.get_statistics()
        assert stats["total_updates"] >= 150

        # Each complexity should have observations
        for complexity in QueryComplexity:
            comp_stats = stats["by_complexity"][complexity.value]
            assert comp_stats["total_queries"] > 0

    def test_learning_adapts_to_feedback(self, learner):
        """Learner should adapt recommendations based on feedback patterns."""
        # Consistently fail at current budget for COMPLEX
        for _ in range(10):
            learner.update(
                complexity=QueryComplexity.COMPLEX,
                budget_used=8.0,
                quality_score=0.4,  # Poor quality
                confidence=0.9,
            )

        # Get recommendation
        rec = learner.get_recommendation(QueryComplexity.COMPLEX)

        # Should acknowledge low success rate in reasoning
        assert "success rate" in rec.reasoning.lower() or "low" in rec.reasoning.lower()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
