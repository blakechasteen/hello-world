"""
Tests for Thompson Sampling Bandit Mathematical Properties.

Validates:
- Division-by-zero edge cases (Phase 1.2 fix)
- Beta distribution properties
- Strategy selection correctness
- Update mechanics and convergence

Author: Claude Code
Date: December 2025 (Math Module Upgrade)
"""

import numpy as np
import pytest
from HoloLoom.policy.thompson_sampling import TSBandit
from HoloLoom.protocols.types import BanditStrategy


class TestDivisionByZeroGuards:
    """Test division-by-zero guards added in Phase 1.2."""

    def test_get_priors_with_zero_counts(self):
        """Priors should handle zero counts gracefully."""
        bandit = TSBandit(n_arms=3)
        # Force zero counts (shouldn't happen normally, but defensive)
        bandit.success = np.array([0.0, 0.0, 0.0])
        bandit.fail = np.array([0.0, 0.0, 0.0])

        priors = bandit.get_priors()

        # Should not raise, should return valid values
        assert priors.shape == (3,)
        assert np.all(np.isfinite(priors))
        assert np.all(priors >= 0)
        assert np.all(priors <= 1)

    def test_get_stats_with_zero_counts(self):
        """Stats should handle zero counts gracefully."""
        bandit = TSBandit(n_arms=2)
        bandit.success = np.array([0.0, 0.0])
        bandit.fail = np.array([0.0, 0.0])

        stats = bandit.get_stats()

        # Should not raise
        assert 0 in stats
        assert 1 in stats
        assert np.isfinite(stats[0]['mean'])
        assert np.isfinite(stats[1]['mean'])

    def test_get_priors_with_tiny_counts(self):
        """Priors should handle very small counts correctly."""
        bandit = TSBandit(n_arms=2)
        bandit.success = np.array([1e-15, 1e-15])
        bandit.fail = np.array([1e-15, 1e-15])

        priors = bandit.get_priors()

        assert np.all(np.isfinite(priors))


class TestBetaDistributionProperties:
    """Test Beta distribution mathematical properties."""

    def test_initial_uniform_priors(self):
        """Initial Beta(1,1) should give uniform priors."""
        bandit = TSBandit(n_arms=3)

        priors = bandit.get_priors()

        # Beta(1,1) has mean = 1/(1+1) = 0.5
        np.testing.assert_array_almost_equal(priors, [0.5, 0.5, 0.5])

    def test_beta_mean_formula(self):
        """Mean of Beta(alpha, beta) = alpha / (alpha + beta)."""
        bandit = TSBandit(n_arms=2)
        bandit.success = np.array([3.0, 7.0])
        bandit.fail = np.array([2.0, 3.0])

        priors = bandit.get_priors()

        # Arm 0: 3/(3+2) = 0.6
        # Arm 1: 7/(7+3) = 0.7
        np.testing.assert_array_almost_equal(priors, [0.6, 0.7])

    def test_beta_concentration(self):
        """More observations should concentrate the distribution."""
        bandit = TSBandit(n_arms=2)

        # Arm 0: few observations (high variance)
        bandit.success[0] = 2
        bandit.fail[0] = 2

        # Arm 1: many observations (low variance)
        bandit.success[1] = 200
        bandit.fail[1] = 200

        # Sample many times to check variance
        samples_0 = [np.random.beta(bandit.success[0], bandit.fail[0]) for _ in range(1000)]
        samples_1 = [np.random.beta(bandit.success[1], bandit.fail[1]) for _ in range(1000)]

        # Arm 1 should have much lower variance
        assert np.var(samples_0) > np.var(samples_1) * 5

    def test_thompson_sampling_explores(self):
        """Thompson Sampling should explore uncertain arms."""
        bandit = TSBandit(n_arms=2)

        # Arm 0: Definitely good (99% success rate, many trials)
        bandit.success[0] = 99
        bandit.fail[0] = 1

        # Arm 1: Unknown (could be better)
        bandit.success[1] = 2
        bandit.fail[1] = 1

        # Run many samples - arm 1 should sometimes be chosen
        choices = [bandit.choose() for _ in range(100)]

        # Arm 1 should be chosen sometimes due to uncertainty
        assert 1 in choices, "Thompson Sampling should explore uncertain arms"


class TestStrategySelection:
    """Test different bandit strategies."""

    def test_epsilon_greedy_explore_exploit(self):
        """Epsilon-greedy should balance exploration and exploitation."""
        bandit = TSBandit(n_arms=3, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.3)
        neural_probs = np.array([0.1, 0.8, 0.1])

        explore_count = 0
        exploit_count = 0

        for _ in range(1000):
            arm, info = bandit.select_with_strategy(neural_probs)
            if info['mode'] == 'explore':
                explore_count += 1
            else:
                exploit_count += 1

        # Should explore roughly 30% of the time
        explore_rate = explore_count / 1000
        assert 0.2 < explore_rate < 0.4, f"Expected ~30% exploration, got {explore_rate:.1%}"

    def test_bayesian_blend_combines_sources(self):
        """Bayesian blend should combine neural and bandit priors."""
        bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND)

        # Neural says arm 0 is best
        neural_probs = np.array([0.8, 0.1, 0.1])

        # Make bandit think arm 2 is best
        bandit.success = np.array([1.0, 1.0, 10.0])
        bandit.fail = np.array([10.0, 10.0, 1.0])

        arm, info = bandit.select_with_strategy(neural_probs)

        # Combined should reflect both sources
        combined = np.array(info['combined'])

        # Neither arm 0 nor arm 2 should completely dominate
        assert combined[0] > combined[1], "Neural influence on arm 0"
        assert combined[2] > combined[1], "Bandit influence on arm 2"

    def test_pure_thompson_ignores_neural(self):
        """Pure Thompson should ignore neural network entirely."""
        bandit = TSBandit(n_arms=2, strategy=BanditStrategy.PURE_THOMPSON)

        # Make arm 0 clearly better in bandit
        bandit.success = np.array([100.0, 1.0])
        bandit.fail = np.array([1.0, 100.0])

        # Neural says arm 1 is best (should be ignored)
        neural_probs = np.array([0.0, 1.0])

        # Most choices should be arm 0
        choices = [bandit.select_with_strategy(neural_probs)[0] for _ in range(100)]

        arm0_rate = choices.count(0) / 100
        assert arm0_rate > 0.9, f"Pure Thompson should favor arm 0, got {arm0_rate:.1%}"


class TestUpdateMechanics:
    """Test bandit update mechanics."""

    def test_positive_reward_increases_success(self):
        """Positive reward should increase success count."""
        bandit = TSBandit(n_arms=2)
        initial_success = bandit.success[0]

        bandit.update(0, reward=1.0)

        assert bandit.success[0] == initial_success + 1.0

    def test_negative_reward_increases_fail(self):
        """Negative reward should increase fail count."""
        bandit = TSBandit(n_arms=2)
        initial_fail = bandit.fail[1]

        bandit.update(1, reward=-0.5)

        assert bandit.fail[1] == initial_fail + 0.5

    def test_fractional_rewards(self):
        """Bandit should handle fractional rewards."""
        bandit = TSBandit(n_arms=1)

        bandit.update(0, reward=0.7)
        bandit.update(0, reward=0.3)

        # Initial success was 1.0, added 0.7 + 0.3 = 1.0
        assert bandit.success[0] == 2.0

    def test_convergence_with_updates(self):
        """Priors should converge to true success rate."""
        bandit = TSBandit(n_arms=1)
        true_rate = 0.7

        # Simulate many trials
        for _ in range(1000):
            if np.random.rand() < true_rate:
                bandit.update(0, reward=1.0)
            else:
                bandit.update(0, reward=-1.0)

        estimated_rate = bandit.get_priors()[0]

        # Should converge to true rate within 5%
        assert abs(estimated_rate - true_rate) < 0.05


class TestPullTracking:
    """Test arm pull tracking."""

    def test_pulls_increment(self):
        """Pulls should increment on selection."""
        bandit = TSBandit(n_arms=3)
        neural_probs = np.array([0.33, 0.34, 0.33])

        initial_pulls = bandit.pulls.copy()

        arm, _ = bandit.select_with_strategy(neural_probs)

        assert bandit.pulls[arm] == initial_pulls[arm] + 1

    def test_pulls_in_stats(self):
        """Stats should include pull counts."""
        bandit = TSBandit(n_arms=2)
        neural_probs = np.array([1.0, 0.0])

        for _ in range(5):
            bandit.select_with_strategy(neural_probs)

        stats = bandit.get_stats()

        total_pulls = stats[0]['pulls'] + stats[1]['pulls']
        assert total_pulls == 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_arm(self):
        """Single arm bandit should always choose arm 0."""
        bandit = TSBandit(n_arms=1)

        for _ in range(10):
            assert bandit.choose() == 0

    def test_many_arms(self):
        """Should handle many arms correctly."""
        bandit = TSBandit(n_arms=100)

        arm = bandit.choose()

        assert 0 <= arm < 100

    def test_zero_epsilon(self):
        """Zero epsilon should always exploit."""
        bandit = TSBandit(n_arms=2, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.0)
        neural_probs = np.array([0.3, 0.7])

        for _ in range(10):
            arm, info = bandit.select_with_strategy(neural_probs)
            assert info['mode'] == 'exploit'
            assert arm == 1  # Always choose highest neural prob

    def test_full_epsilon(self):
        """Full epsilon should always explore."""
        bandit = TSBandit(n_arms=2, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=1.0)
        neural_probs = np.array([0.3, 0.7])

        for _ in range(10):
            _, info = bandit.select_with_strategy(neural_probs)
            assert info['mode'] == 'explore'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
