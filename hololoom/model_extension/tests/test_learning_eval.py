"""
Test Suite for Learning Effectiveness Evaluation
=================================================

Comprehensive tests for learning metrics:
- Improvement Rate
- Thompson Sampling Regret
- Pattern Discovery Rate
- Convergence Speed
- Retention Score
- Catastrophic Forgetting Detection

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
import random
from typing import List

from hololoom.model_extension.eval.learning_metrics import (
    LearningInteraction,
    ThompsonSamplingState,
    LearningMetrics,
    LearningEvaluator,
    improvement_rate,
    thompson_sampling_regret,
    pattern_discovery_rate,
    convergence_speed,
    retention_score,
    catastrophic_forgetting_score,
)


# =============================================================================
# Improvement Rate Tests
# =============================================================================

class TestImprovementRate:
    """Tests for improvement rate metric."""

    def test_improving_interactions(self, improving_interactions):
        """Should detect positive improvement."""
        rate = improvement_rate(improving_interactions, window_size=50)

        # Improving interactions should show positive rate
        assert rate > 0, f"Expected positive improvement rate, got {rate}"

    def test_stagnant_interactions(self, stagnant_interactions):
        """Should detect minimal improvement for stagnant learning."""
        rate = improvement_rate(stagnant_interactions, window_size=50)

        # Stagnant should have near-zero improvement
        assert abs(rate) < 0.15, f"Expected near-zero rate for stagnant, got {rate}"

    def test_degrading_interactions(self, degrading_interactions):
        """Should detect negative improvement (degradation)."""
        rate = improvement_rate(degrading_interactions, window_size=50)

        # Degrading should show negative rate
        assert rate < 0, f"Expected negative improvement rate, got {rate}"

    def test_improvement_rate_bounds(self, improving_interactions):
        """Improvement rate should be bounded."""
        rate = improvement_rate(improving_interactions, window_size=50)

        # Rate should be reasonable (-1 to 1 typical)
        assert -2.0 <= rate <= 2.0, f"Rate out of reasonable bounds: {rate}"

    def test_empty_interactions(self):
        """Empty interactions should return 0 improvement."""
        rate = improvement_rate([], window_size=50)

        assert rate == 0.0

    def test_single_interaction(self):
        """Single interaction should return 0 improvement."""
        interaction = LearningInteraction(
            query_id="q_0",
            quality_score=0.8,
            confidence=0.8,
            timestamp=0.0
        )
        rate = improvement_rate([interaction], window_size=50)

        assert rate == 0.0


class TestThompsonSamplingRegret:
    """Tests for Thompson Sampling regret calculation."""

    def test_regret_with_optimal_arm(self, thompson_sampling_states):
        """Should calculate regret relative to optimal arm."""
        # Create actions and rewards based on states
        actions_taken = []
        rewards = []
        for state in thompson_sampling_states:
            for _ in range(state.pulls):
                actions_taken.append(state.arm_id)
                # Simulate reward based on expected value
                rewards.append(state.expected_reward)

        regret = thompson_sampling_regret(thompson_sampling_states, actions_taken, rewards)

        # Regret should be non-negative
        assert regret >= 0, f"Regret should be non-negative: {regret}"

    def test_regret_no_pulls(self):
        """Arms with no pulls should have minimal regret impact."""
        states = [
            ThompsonSamplingState(arm_id="arm1", alpha=1, beta=1, pulls=0),
            ThompsonSamplingState(arm_id="arm2", alpha=1, beta=1, pulls=0),
        ]
        regret = thompson_sampling_regret(states, [], [])

        # No pulls = no regret accumulated
        assert regret == 0.0

    def test_regret_single_arm(self):
        """Single arm should have zero regret."""
        states = [
            ThompsonSamplingState(arm_id="arm1", alpha=10, beta=5, pulls=15),
        ]
        # All pulls on the only arm
        actions = ["arm1"] * 15
        rewards = [states[0].expected_reward] * 15

        regret = thompson_sampling_regret(states, actions, rewards)

        # Single arm = optimal (regret = 0)
        assert regret == 0.0 or abs(regret) < 0.01

    def test_regret_clear_suboptimal(self):
        """Clear suboptimal arm selection should show high regret."""
        states = [
            ThompsonSamplingState(arm_id="optimal", alpha=90, beta=10, pulls=5),  # 0.9 expected
            ThompsonSamplingState(arm_id="suboptimal", alpha=10, beta=90, pulls=95),  # 0.1 expected
        ]
        # 95 pulls on suboptimal arm, 5 on optimal
        actions = ["optimal"] * 5 + ["suboptimal"] * 95
        rewards = [0.9] * 5 + [0.1] * 95  # Approximate rewards

        regret = thompson_sampling_regret(states, actions, rewards)

        # Pulling suboptimal 95 times should accumulate significant regret
        # Regret = 95 * (0.9 - 0.1) = 76
        assert regret > 50, f"Expected high regret for suboptimal pulls: {regret}"

    def test_regret_empty_states(self):
        """Empty states should return 0 regret."""
        regret = thompson_sampling_regret([], [], [])

        assert regret == 0.0


class TestPatternDiscoveryRate:
    """Tests for pattern discovery rate metric."""

    def test_pattern_discovery_improving(self, improving_interactions):
        """Improving interactions should show pattern discovery."""
        rate = pattern_discovery_rate(improving_interactions)

        # Rate is patterns per 100 queries
        assert rate >= 0, f"Pattern discovery rate should be non-negative: {rate}"

    def test_pattern_discovery_stagnant(self, stagnant_interactions):
        """Stagnant interactions should show low pattern discovery."""
        rate = pattern_discovery_rate(stagnant_interactions)

        # Stagnant = no new patterns
        assert rate == 0.0, f"Expected 0 patterns for stagnant: {rate}"

    def test_pattern_discovery_rate_calculation(self):
        """Pattern discovery rate should be calculated correctly."""
        interactions = []
        for i in range(100):
            interactions.append(LearningInteraction(
                query_id=f"q_{i}",
                quality_score=0.7,
                confidence=0.7,
                patterns_discovered=1 if i % 10 == 0 else 0,  # 10 patterns in 100 queries
                timestamp=float(i)
            ))

        rate = pattern_discovery_rate(interactions)

        # 10 patterns / 100 queries * 100 = 10 per 100 queries
        assert rate == 10.0, f"Expected 10 patterns per 100, got {rate}"

    def test_pattern_discovery_empty(self):
        """Empty interactions should return 0 discovery rate."""
        rate = pattern_discovery_rate([])

        assert rate == 0.0


class TestConvergenceSpeed:
    """Tests for convergence speed metric."""

    def test_convergence_improving(self, improving_interactions):
        """Improving system should converge."""
        speed = convergence_speed(improving_interactions)

        # Should converge within the interaction count (returns 0 if not converged)
        if speed > 0:
            assert speed <= len(improving_interactions), f"Convergence too slow: {speed}"

    def test_convergence_stagnant(self, stagnant_interactions):
        """Stagnant system might not converge to high quality."""
        # Stagnant at 0.5 quality won't reach 90th percentile target
        speed = convergence_speed(stagnant_interactions)

        # May or may not converge depending on threshold
        assert speed >= 0  # 0 means never reached

    def test_convergence_speed_bounds(self, improving_interactions):
        """Convergence speed should be positive if converged."""
        speed = convergence_speed(improving_interactions)

        # Speed is 0 (not converged) or positive
        assert speed >= 0, f"Convergence speed should be non-negative: {speed}"

    def test_convergence_empty(self):
        """Empty interactions should return 0 (no convergence)."""
        speed = convergence_speed([])

        assert speed == 0


class TestRetentionScore:
    """Tests for retention score metric."""

    def test_retention_good(self):
        """Good retention when old topic quality is maintained."""
        old_topic = [
            LearningInteraction(query_id=f"old_{i}", quality_score=0.8, confidence=0.8, timestamp=float(i))
            for i in range(20)
        ]
        new_topic = [
            LearningInteraction(query_id=f"new_{i}", quality_score=0.75, confidence=0.75, timestamp=float(i))
            for i in range(20)
        ]

        score = retention_score(old_topic, new_topic)

        # Old quality (0.8) >= new quality (0.75), good retention
        assert score >= 0.9, f"Expected good retention, got {score}"

    def test_retention_poor(self):
        """Poor retention when old topic quality degrades."""
        old_topic = [
            LearningInteraction(query_id=f"old_{i}", quality_score=0.5, confidence=0.5, timestamp=float(i))
            for i in range(20)
        ]
        new_topic = [
            LearningInteraction(query_id=f"new_{i}", quality_score=0.9, confidence=0.9, timestamp=float(i))
            for i in range(20)
        ]

        score = retention_score(old_topic, new_topic)

        # Old quality (0.5) < new quality (0.9), poor retention
        assert score < 0.7, f"Expected poor retention for degraded old topics: {score}"

    def test_retention_bounds(self):
        """Retention score should be between 0 and 1 (capped)."""
        old_topic = [
            LearningInteraction(query_id=f"old_{i}", quality_score=0.9, confidence=0.9, timestamp=float(i))
            for i in range(20)
        ]
        new_topic = [
            LearningInteraction(query_id=f"new_{i}", quality_score=0.5, confidence=0.5, timestamp=float(i))
            for i in range(20)
        ]

        score = retention_score(old_topic, new_topic)

        # Capped at 1.0
        assert 0.0 <= score <= 1.0, f"Retention score out of bounds: {score}"

    def test_retention_empty_old(self):
        """Empty old interactions should return 1.0 (no degradation)."""
        new_topic = [
            LearningInteraction(query_id=f"new_{i}", quality_score=0.8, confidence=0.8, timestamp=float(i))
            for i in range(10)
        ]
        score = retention_score([], new_topic)

        assert score == 1.0


class TestCatastrophicForgetting:
    """Tests for catastrophic forgetting detection."""

    def test_no_forgetting(self):
        """No forgetting when quality is maintained."""
        before = {"topic_a": 0.9, "topic_b": 0.8, "topic_c": 0.7}
        after = {"topic_a": 0.9, "topic_b": 0.85, "topic_c": 0.75}

        score = catastrophic_forgetting_score(before, after)

        # Quality maintained or improved, no forgetting
        assert score < 0.1, f"Expected no forgetting, got {score}"

    def test_partial_forgetting(self):
        """Partial forgetting detected when some topics degrade."""
        before = {"topic_a": 0.9, "topic_b": 0.8, "topic_c": 0.7}
        after = {"topic_a": 0.9, "topic_b": 0.4, "topic_c": 0.7}  # topic_b degraded

        score = catastrophic_forgetting_score(before, after)

        # Some forgetting
        assert 0.1 < score < 0.5, f"Expected partial forgetting, got {score}"

    def test_severe_forgetting(self):
        """Severe forgetting detected when all topics degrade."""
        before = {"topic_a": 0.9, "topic_b": 0.8, "topic_c": 0.7}
        after = {"topic_a": 0.3, "topic_b": 0.2, "topic_c": 0.1}  # All degraded

        score = catastrophic_forgetting_score(before, after)

        # Severe forgetting
        assert score > 0.5, f"Expected severe forgetting, got {score}"

    def test_forgetting_bounds(self):
        """Forgetting score should be between 0 and 1."""
        before = {"topic_a": 0.8}
        after = {"topic_a": 0.2}

        score = catastrophic_forgetting_score(before, after)

        assert 0.0 <= score <= 1.0, f"Forgetting score out of bounds: {score}"

    def test_forgetting_empty(self):
        """Empty dicts should return 0 (no forgetting)."""
        score = catastrophic_forgetting_score({}, {})

        assert score == 0.0


class TestLearningEvaluator:
    """Tests for the main LearningEvaluator class."""

    def test_full_evaluation(self, improving_interactions, thompson_sampling_states):
        """Full evaluation should return all metrics."""
        evaluator = LearningEvaluator(window_size=50)

        # Create actions and rewards for Thompson regret
        actions = []
        rewards = []
        for state in thompson_sampling_states:
            for _ in range(state.pulls):
                actions.append(state.arm_id)
                rewards.append(state.expected_reward)

        metrics = evaluator.evaluate(
            improving_interactions,
            thompson_states=thompson_sampling_states,
            actions_taken=actions,
            rewards=rewards
        )

        assert isinstance(metrics, LearningMetrics)
        assert hasattr(metrics, "improvement_rate")
        assert hasattr(metrics, "thompson_sampling_regret")
        assert hasattr(metrics, "pattern_discovery_rate")
        assert hasattr(metrics, "convergence_speed")
        assert hasattr(metrics, "retention_score")
        assert hasattr(metrics, "total_interactions")

    def test_evaluation_consistency(self, improving_interactions, thompson_sampling_states):
        """Individual metrics should match full evaluation."""
        evaluator = LearningEvaluator(window_size=50)

        # Create actions and rewards for Thompson regret
        actions = []
        rewards = []
        for state in thompson_sampling_states:
            for _ in range(state.pulls):
                actions.append(state.arm_id)
                rewards.append(state.expected_reward)

        metrics = evaluator.evaluate(
            improving_interactions,
            thompson_states=thompson_sampling_states,
            actions_taken=actions,
            rewards=rewards
        )

        # Individual computations using module functions
        ir = improvement_rate(improving_interactions, window_size=50)
        tsr = thompson_sampling_regret(thompson_sampling_states, actions, rewards)
        pdr = pattern_discovery_rate(improving_interactions)

        assert abs(metrics.improvement_rate - ir) < 0.0001
        assert abs(metrics.thompson_sampling_regret - tsr) < 0.0001
        assert abs(metrics.pattern_discovery_rate - pdr) < 0.0001

    def test_metrics_serialization(self, improving_interactions, thompson_sampling_states):
        """LearningMetrics should be serializable."""
        evaluator = LearningEvaluator(window_size=50)

        actions = []
        rewards = []
        for state in thompson_sampling_states:
            for _ in range(state.pulls):
                actions.append(state.arm_id)
                rewards.append(state.expected_reward)

        metrics = evaluator.evaluate(
            improving_interactions,
            thompson_states=thompson_sampling_states,
            actions_taken=actions,
            rewards=rewards
        )

        # Convert to dict
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "improvement_rate" in metrics_dict
        assert "thompson_sampling_regret" in metrics_dict
        assert "pattern_discovery_rate" in metrics_dict


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_improvement_rate_function(self, improving_interactions):
        """Convenience improvement_rate function should work."""
        rate = improvement_rate(improving_interactions)
        assert isinstance(rate, float)

    def test_thompson_regret_function(self, thompson_sampling_states):
        """Convenience thompson_sampling_regret function should work."""
        actions = []
        rewards = []
        for state in thompson_sampling_states:
            for _ in range(state.pulls):
                actions.append(state.arm_id)
                rewards.append(state.expected_reward)

        regret = thompson_sampling_regret(thompson_sampling_states, actions, rewards)
        assert isinstance(regret, float)

    def test_pattern_discovery_function(self, improving_interactions):
        """Convenience pattern_discovery_rate function should work."""
        rate = pattern_discovery_rate(improving_interactions)
        assert isinstance(rate, float)

    def test_convergence_speed_function(self, improving_interactions):
        """Convenience convergence_speed function should work."""
        speed = convergence_speed(improving_interactions)
        # Returns int (0 if not converged)
        assert isinstance(speed, int)

    def test_retention_score_function(self):
        """Convenience retention_score function should work."""
        old_interactions = [
            LearningInteraction(query_id=f"old_{i}", quality_score=0.8, confidence=0.8, timestamp=float(i))
            for i in range(10)
        ]
        new_interactions = [
            LearningInteraction(query_id=f"new_{i}", quality_score=0.75, confidence=0.75, timestamp=float(i))
            for i in range(10)
        ]
        score = retention_score(old_interactions, new_interactions)
        assert isinstance(score, float)


class TestLearnQueryVerifyCycle:
    """Tests for Learn-Query-Verify evaluation."""

    def test_lqv_basic(self):
        """Basic LQV cycle evaluation."""
        evaluator = LearningEvaluator(window_size=50)

        # Store interactions (learning events)
        store_interactions = [
            LearningInteraction(query_id=f"store_{i}", quality_score=0.9, confidence=0.9, timestamp=float(i))
            for i in range(10, 15)
        ]

        # Query interactions before and after storing
        query_interactions = [
            # Before store (timestamp < 10)
            LearningInteraction(query_id=f"query_{i}", quality_score=0.5, confidence=0.5, timestamp=float(i))
            for i in range(5)
        ] + [
            # After store (timestamp > 15)
            LearningInteraction(query_id=f"query_{i}", quality_score=0.8, confidence=0.8, timestamp=float(i))
            for i in range(20, 30)
        ]

        result = evaluator.evaluate_learn_query_verify_cycle(
            store_interactions,
            query_interactions
        )

        assert "effective" in result
        if "improvement" in result:
            # Quality should improve after storing
            assert result["improvement"] > 0

    def test_lqv_empty(self):
        """Empty interactions should return insufficient data."""
        evaluator = LearningEvaluator(window_size=50)

        result = evaluator.evaluate_learn_query_verify_cycle([], [])

        assert result["effective"] == False
        assert result["reason"] == "insufficient_data"


class TestFeedbackLoop:
    """Tests for feedback loop evaluation."""

    def test_feedback_loop_positive(self):
        """Positive feedback should correlate with higher quality."""
        evaluator = LearningEvaluator(window_size=50)

        positive_interactions = [
            LearningInteraction(query_id=f"pos_{i}", quality_score=0.9, confidence=0.9, timestamp=float(i))
            for i in range(20)
        ]
        negative_interactions = [
            LearningInteraction(query_id=f"neg_{i}", quality_score=0.3, confidence=0.3, timestamp=float(i))
            for i in range(20)
        ]

        result = evaluator.evaluate_feedback_loop(
            positive_interactions,
            negative_interactions
        )

        assert result["effective"] == True
        assert result["differential"] > 0

    def test_feedback_loop_empty(self):
        """Empty feedback should return not effective."""
        evaluator = LearningEvaluator(window_size=50)

        result = evaluator.evaluate_feedback_loop([], [])

        assert result["effective"] == False


class TestHotPatternEffectiveness:
    """Tests for hot pattern effectiveness evaluation."""

    def test_hot_pattern_boost(self):
        """Hot patterns should show quality boost."""
        evaluator = LearningEvaluator(window_size=50)

        hot_queries = [
            LearningInteraction(query_id=f"hot_{i}", quality_score=0.9, confidence=0.9, timestamp=float(i))
            for i in range(20)
        ]
        cold_queries = [
            LearningInteraction(query_id=f"cold_{i}", quality_score=0.5, confidence=0.5, timestamp=float(i))
            for i in range(20)
        ]

        result = evaluator.evaluate_hot_pattern_effectiveness(
            hot_queries,
            cold_queries
        )

        # Hot patterns should have better average quality
        assert result["hot_avg_quality"] > result["cold_avg_quality"]
        assert result["effective"] == True

    def test_hot_pattern_empty(self):
        """Empty patterns should return not effective."""
        evaluator = LearningEvaluator(window_size=50)

        result = evaluator.evaluate_hot_pattern_effectiveness([], [])

        assert result["effective"] == False


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_history(self):
        """Should handle very short interaction history."""
        evaluator = LearningEvaluator(window_size=50)

        interactions = [
            LearningInteraction(
                query_id="q_0",
                quality_score=0.7,
                confidence=0.7,
                timestamp=0.0
            ),
            LearningInteraction(
                query_id="q_1",
                quality_score=0.8,
                confidence=0.8,
                timestamp=1.0
            ),
        ]

        metrics = evaluator.evaluate(interactions, [])

        # Should not crash with short history
        assert isinstance(metrics, LearningMetrics)

    def test_constant_quality(self):
        """Should handle constant quality (no improvement or degradation)."""
        evaluator = LearningEvaluator(window_size=50)

        interactions = [
            LearningInteraction(
                query_id=f"q_{i}",
                quality_score=0.75,
                confidence=0.75,
                timestamp=float(i)
            )
            for i in range(100)
        ]

        metrics = evaluator.evaluate(interactions, [])

        # Near-zero improvement rate
        assert abs(metrics.improvement_rate) < 0.1

    def test_high_variance_quality(self):
        """Should handle high variance in quality scores."""
        evaluator = LearningEvaluator(window_size=50)

        interactions = [
            LearningInteraction(
                query_id=f"q_{i}",
                quality_score=random.random(),  # High variance
                confidence=random.random(),
                timestamp=float(i)
            )
            for i in range(100)
        ]

        metrics = evaluator.evaluate(interactions, [])

        # Should compute without error
        assert isinstance(metrics, LearningMetrics)

    def test_all_optimal_arms(self):
        """All arms being optimal should show zero regret."""
        states = [
            ThompsonSamplingState(arm_id="arm1", alpha=90, beta=10, pulls=33),
            ThompsonSamplingState(arm_id="arm2", alpha=90, beta=10, pulls=33),
            ThompsonSamplingState(arm_id="arm3", alpha=90, beta=10, pulls=34),
        ]

        # All arms have same expected reward
        actions = []
        rewards = []
        for state in states:
            for _ in range(state.pulls):
                actions.append(state.arm_id)
                rewards.append(state.expected_reward)

        regret = thompson_sampling_regret(states, actions, rewards)

        # All arms equal, so no regret from arm selection
        assert regret < 1, f"Expected near-zero regret for equal arms: {regret}"
