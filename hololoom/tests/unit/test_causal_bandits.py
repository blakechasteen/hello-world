"""
Tests for causal_bandits.py — Deconfounded Thompson Sampling.

Wire 6 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.causal_bandits import (
    CausalBandit,
    CausalBanditArm,
    CausalBanditState,
    DeconfoundedUpdate,
    create_causal_bandit,
)


# ============================================================================
# CausalBanditArm
# ============================================================================

class TestCausalBanditArm:
    def test_initial_uniform(self):
        arm = CausalBanditArm()
        assert arm.successes == 1.0
        assert arm.failures == 1.0
        assert arm.mean() == pytest.approx(0.5)

    def test_sample_in_range(self):
        arm = CausalBanditArm(successes=10, failures=2)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0

    def test_update_positive(self):
        arm = CausalBanditArm()
        arm.update(0.8)
        assert arm.successes == 1.8
        assert arm.failures == 1.0

    def test_update_negative(self):
        arm = CausalBanditArm()
        arm.update(-0.5)
        assert arm.successes == 1.0
        assert arm.failures == 1.5

    def test_update_zero(self):
        arm = CausalBanditArm()
        arm.update(0.0)
        # Zero reward: not positive, so failures don't change either
        assert arm.successes == 1.0
        assert arm.failures == 1.0

    def test_mean_shifts_with_updates(self):
        arm = CausalBanditArm()
        for _ in range(10):
            arm.update(1.0)
        assert arm.mean() > 0.8


# ============================================================================
# CausalBandit without DAG (standard Thompson Sampling)
# ============================================================================

class TestCausalBanditStandard:
    def test_select_returns_valid_arm(self):
        bandit = CausalBandit(n_arms=4)
        for _ in range(50):
            arm = bandit.select()
            assert 0 <= arm < 4

    def test_update_without_context(self):
        bandit = CausalBandit(n_arms=3)
        result = bandit.update(0, reward=1.0)
        assert result.adjustment_applied is False
        assert result.adjusted_reward == 1.0
        assert bandit.total_pulls == 1

    def test_get_priors_initial(self):
        bandit = CausalBandit(n_arms=3)
        priors = bandit.get_priors()
        assert len(priors) == 3
        np.testing.assert_allclose(priors, [0.5, 0.5, 0.5])

    def test_learning_shifts_priors(self):
        """Arm 0 always rewarded → its prior should increase."""
        bandit = CausalBandit(n_arms=3)
        for _ in range(20):
            bandit.update(0, reward=1.0)
        priors = bandit.get_priors()
        assert priors[0] > priors[1]
        assert priors[0] > priors[2]

    def test_convergence_to_best_arm(self):
        """After many pulls, bandit should mostly select the best arm."""
        bandit = CausalBandit(n_arms=3)
        # Train: arm 1 is best
        for _ in range(50):
            bandit.update(0, reward=0.3)
            bandit.update(1, reward=0.9)
            bandit.update(2, reward=0.1)

        # Selection should favor arm 1
        selections = [bandit.select() for _ in range(100)]
        arm1_freq = selections.count(1) / len(selections)
        assert arm1_freq > 0.5


# ============================================================================
# CausalBandit with DAG (causal deconfounding)
# ============================================================================

class TestCausalBanditWithDAG:
    @pytest.fixture
    def simple_dag(self):
        """Build a simple causal DAG for tool selection.

        query_type → tool_selected → outcome_quality
                  ↘                ↗
                   (confounder path)
        """
        try:
            from hololoom.causal.dag import CausalDAG, CausalNode, CausalEdge
            dag = CausalDAG()
            dag.add_node(CausalNode(name="query_type"))
            dag.add_node(CausalNode(name="tool_selected"))
            dag.add_node(CausalNode(name="outcome_quality"))
            dag.add_edge(CausalEdge(source="query_type", target="tool_selected"))
            dag.add_edge(CausalEdge(source="query_type", target="outcome_quality"))
            dag.add_edge(CausalEdge(source="tool_selected", target="outcome_quality"))
            return dag
        except ImportError:
            pytest.skip("Causal module not available")

    def test_deconfound_with_dag(self, simple_dag):
        bandit = CausalBandit(n_arms=3, dag=simple_dag)
        result = bandit.deconfound(
            arm=0, raw_reward=0.9,
            context={"query_type": 0.8}
        )
        assert result.adjustment_applied is True
        assert result.confounders_found  # Should find query_type as confounder
        assert "query_type" in result.confounders_found
        assert result.adjusted_reward != result.raw_reward  # Adjustment happened

    def test_deconfound_no_context(self, simple_dag):
        bandit = CausalBandit(n_arms=3, dag=simple_dag)
        result = bandit.deconfound(arm=0, raw_reward=0.9, context=None)
        assert result.adjustment_applied is False
        assert result.adjusted_reward == 0.9

    def test_deconfound_irrelevant_context(self, simple_dag):
        """Context variables not in DAG should not trigger adjustment."""
        bandit = CausalBandit(n_arms=3, dag=simple_dag)
        result = bandit.deconfound(
            arm=0, raw_reward=0.9,
            context={"irrelevant_var": 0.5}
        )
        assert result.adjustment_applied is False

    def test_update_with_deconfounding(self, simple_dag):
        bandit = CausalBandit(n_arms=3, dag=simple_dag)
        result = bandit.update(
            arm=0, reward=0.9,
            context={"query_type": 0.8}
        )
        assert result.adjustment_applied is True
        assert bandit.total_pulls == 1

    def test_extreme_confounder_dampens_more(self, simple_dag):
        """Extreme confounder values (near 0 or 1) should dampen reward more."""
        bandit = CausalBandit(n_arms=3, dag=simple_dag)

        result_neutral = bandit.deconfound(0, 1.0, {"query_type": 0.5})
        result_extreme = bandit.deconfound(0, 1.0, {"query_type": 1.0})

        # Extreme confounder should produce more damping
        if result_neutral.adjustment_applied and result_extreme.adjustment_applied:
            assert result_extreme.adjusted_reward <= result_neutral.adjusted_reward

    def test_confounder_stats_tracked(self, simple_dag):
        bandit = CausalBandit(n_arms=3, dag=simple_dag)
        bandit.update(0, 0.9, {"query_type": 0.5})
        bandit.update(1, 0.3, {"query_type": 0.8})
        assert bandit.confounder_stats.get("query_type", 0) == 2


# ============================================================================
# State persistence
# ============================================================================

class TestCausalBanditState:
    def test_state_roundtrip(self):
        bandit = CausalBandit(n_arms=3)
        bandit.update(0, 1.0)
        bandit.update(1, -0.5)

        state = bandit.state()
        assert len(state.arms) == 3
        assert state.total_pulls == 2

        # Restore into new bandit
        new_bandit = CausalBandit(n_arms=3)
        new_bandit.load_state(state)
        np.testing.assert_allclose(
            new_bandit.get_priors(), bandit.get_priors()
        )

    def test_state_includes_confounder_stats(self):
        try:
            from hololoom.causal.dag import CausalDAG, CausalNode, CausalEdge
            dag = CausalDAG()
            dag.add_node(CausalNode(name="x"))
            dag.add_node(CausalNode(name="tool_selected"))
            dag.add_node(CausalNode(name="outcome_quality"))
            dag.add_edge(CausalEdge(source="x", target="tool_selected"))
            dag.add_edge(CausalEdge(source="x", target="outcome_quality"))
            dag.add_edge(CausalEdge(source="tool_selected", target="outcome_quality"))
        except ImportError:
            pytest.skip("Causal module not available")

        bandit = CausalBandit(n_arms=2, dag=dag)
        bandit.update(0, 0.8, {"x": 0.5})
        state = bandit.state()
        assert "x" in state.confounder_stats


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create_causal_bandit(self):
        bandit = create_causal_bandit(n_arms=5)
        assert isinstance(bandit, CausalBandit)
        assert bandit.n_arms == 5

    def test_create_with_custom_names(self):
        bandit = create_causal_bandit(
            n_arms=3, treatment="action", outcome="score"
        )
        assert bandit.treatment == "action"
        assert bandit.outcome == "score"
