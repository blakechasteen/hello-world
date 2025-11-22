"""
Thompson Sampling Policy Tests (Week 2 - Priority 1)
=====================================================
Tests Thompson Sampling bandit implementation with all 3 exploration strategies.

Thompson Sampling is the core exploration/exploitation mechanism in HoloLoom's
policy engine. It maintains Beta distributions for each tool and uses Bayesian
updating to balance trying new tools (exploration) vs. using known-good tools
(exploitation).

Test Coverage (25 tests):
- Initialization and defaults - 3 tests
- Arm selection (choose) - 4 tests
- Prior calculation (get_priors) - 3 tests
- Strategy: Epsilon-Greedy - 4 tests
- Strategy: Bayesian Blend - 4 tests
- Strategy: Pure Thompson - 3 tests
- Bayesian updates - 4 tests

Created: 2025-11-21 (Week 2 Test Creation)
"""

import pytest
import numpy as np
from unittest.mock import patch

from HoloLoom.policy.thompson_sampling import TSBandit
from HoloLoom.protocols.types import BanditStrategy


# ============================================================================
# Test 1-3: Initialization and Defaults
# ============================================================================

def test_initialization_defaults():
    """Test bandit initializes with correct defaults."""
    bandit = TSBandit(n_arms=4)

    # Default: Beta(1, 1) for all arms (uniform prior)
    assert len(bandit.success) == 4
    assert len(bandit.fail) == 4
    assert np.all(bandit.success == 1.0)
    assert np.all(bandit.fail == 1.0)

    # Default strategy: EPSILON_GREEDY
    assert bandit.strategy == BanditStrategy.EPSILON_GREEDY
    assert bandit.epsilon == 0.1

    # Zero pulls initially
    assert np.all(bandit.pulls == 0)


def test_initialization_custom_strategy():
    """Test bandit initializes with custom strategy."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND, epsilon=0.15)

    assert bandit.strategy == BanditStrategy.BAYESIAN_BLEND
    assert bandit.epsilon == 0.15
    assert bandit.n_arms == 3


def test_initialization_single_arm():
    """Test bandit with single arm (edge case)."""
    bandit = TSBandit(n_arms=1)

    assert len(bandit.success) == 1
    assert len(bandit.fail) == 1

    # Single arm always selected
    arm = bandit.choose()
    assert arm == 0


# ============================================================================
# Test 4-7: Arm Selection (choose)
# ============================================================================

def test_choose_returns_valid_arm():
    """Test choose() returns valid arm index."""
    bandit = TSBandit(n_arms=5)

    # Choose 100 times, all should be valid
    for _ in range(100):
        arm = bandit.choose()
        assert 0 <= arm < 5
        assert isinstance(arm, int)


def test_choose_samples_from_beta():
    """Test choose() samples from Beta distributions."""
    bandit = TSBandit(n_arms=3)

    # Make arm 0 clearly better: Beta(10, 1) vs Beta(1, 1)
    bandit.success[0] = 10.0
    bandit.fail[0] = 1.0

    # Arm 0 should be chosen most often (but not always, due to sampling)
    choices = [bandit.choose() for _ in range(1000)]

    # Arm 0 should be chosen >80% of the time
    assert choices.count(0) > 800


def test_choose_exploration_vs_exploitation():
    """Test choose() balances exploration and exploitation."""
    bandit = TSBandit(n_arms=4)

    # Make arm 0 clearly best
    bandit.success[0] = 50.0
    bandit.fail[0] = 1.0

    # Arms 1-3 are weak
    bandit.success[1:] = 1.0
    bandit.fail[1:] = 10.0

    # Run 1000 trials
    choices = [bandit.choose() for _ in range(1000)]

    # Arm 0 should dominate (>90%)
    assert choices.count(0) > 900

    # But other arms should occasionally be tried (exploration)
    assert any(c != 0 for c in choices)


def test_choose_uniform_prior():
    """Test choose() with uniform prior (all arms equal)."""
    bandit = TSBandit(n_arms=4)

    # All arms Beta(1, 1) - should be roughly uniform
    choices = [bandit.choose() for _ in range(4000)]

    # Each arm should be chosen ~1000 times (allow 20% variance)
    for arm in range(4):
        count = choices.count(arm)
        assert 800 < count < 1200


# ============================================================================
# Test 8-10: Prior Calculation (get_priors)
# ============================================================================

def test_get_priors_uniform():
    """Test get_priors() with uniform prior."""
    bandit = TSBandit(n_arms=3)

    priors = bandit.get_priors()

    # Beta(1, 1) has mean 0.5
    assert np.allclose(priors, [0.5, 0.5, 0.5])


def test_get_priors_after_updates():
    """Test get_priors() after Bayesian updates."""
    bandit = TSBandit(n_arms=3)

    # Update arm 0: 10 successes
    bandit.success[0] = 11.0  # 1 (prior) + 10
    bandit.fail[0] = 1.0

    # Update arm 1: 5 successes, 5 failures
    bandit.success[1] = 6.0
    bandit.fail[1] = 6.0

    # Arm 2 unchanged

    priors = bandit.get_priors()

    # Arm 0: 11/(11+1) = 0.917
    assert abs(priors[0] - 0.917) < 0.01

    # Arm 1: 6/(6+6) = 0.5
    assert abs(priors[1] - 0.5) < 0.01

    # Arm 2: 1/(1+1) = 0.5
    assert abs(priors[2] - 0.5) < 0.01


def test_get_priors_extreme_values():
    """Test get_priors() with extreme success/failure."""
    bandit = TSBandit(n_arms=2)

    # Arm 0: 100% success
    bandit.success[0] = 100.0
    bandit.fail[0] = 1.0

    # Arm 1: 100% failure
    bandit.success[1] = 1.0
    bandit.fail[1] = 100.0

    priors = bandit.get_priors()

    # Arm 0: ~0.99
    assert priors[0] > 0.98

    # Arm 1: ~0.01
    assert priors[1] < 0.02


# ============================================================================
# Test 11-14: Strategy - Epsilon-Greedy
# ============================================================================

def test_epsilon_greedy_exploit_mode():
    """Test epsilon-greedy exploits with high probability."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.1)

    # Neural network says arm 2 is best
    neural_probs = np.array([0.1, 0.2, 0.7])

    # Run 1000 trials
    results = [bandit.select_with_strategy(neural_probs) for _ in range(1000)]
    arms = [r[0] for r in results]

    # Arm 2 should be chosen ~90% (exploit) + ~3.3% (explore) = ~93%
    assert arms.count(2) > 850


def test_epsilon_greedy_explore_mode():
    """Test epsilon-greedy explores with epsilon probability."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.3)

    # Neural network says arm 0 is best
    neural_probs = np.array([0.8, 0.1, 0.1])

    # Run 1000 trials
    results = [bandit.select_with_strategy(neural_probs) for _ in range(1000)]
    arms = [r[0] for r in results]
    modes = [r[1]['mode'] for r in results]

    # ~30% should be explore mode
    explore_count = modes.count('explore')
    assert 250 < explore_count < 350

    # ~70% should be exploit mode
    exploit_count = modes.count('exploit')
    assert 650 < exploit_count < 750


def test_epsilon_greedy_debug_info():
    """Test epsilon-greedy returns correct debug info."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.1)
    neural_probs = np.array([0.5, 0.3, 0.2])

    arm, debug = bandit.select_with_strategy(neural_probs)

    assert debug['strategy'] == 'epsilon_greedy'
    assert 'mode' in debug  # 'explore' or 'exploit'
    assert 'selected_arm' in debug
    assert 'total_pulls' in debug

    if debug['mode'] == 'explore':
        assert debug['exploration_rate'] == 0.1


def test_epsilon_greedy_zero_epsilon():
    """Test epsilon-greedy with epsilon=0 (pure exploitation)."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.0)

    # Neural network says arm 1 is best
    neural_probs = np.array([0.2, 0.7, 0.1])

    # Run 100 trials - should ALWAYS choose arm 1
    results = [bandit.select_with_strategy(neural_probs) for _ in range(100)]
    arms = [r[0] for r in results]

    assert all(a == 1 for a in arms)


# ============================================================================
# Test 15-18: Strategy - Bayesian Blend
# ============================================================================

def test_bayesian_blend_combines_priors():
    """Test Bayesian blend combines neural + bandit priors."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND)

    # Neural says arm 0 is best (0.7)
    neural_probs = np.array([0.7, 0.2, 0.1])

    # Bandit says arm 1 is best
    bandit.success[1] = 10.0
    bandit.fail[1] = 1.0

    arm, debug = bandit.select_with_strategy(neural_probs)

    # Should blend: 0.7 * neural + 0.3 * bandit
    assert 'combined' in debug
    combined = np.array(debug['combined'])

    # Combined should be weighted average
    bandit_priors = bandit.get_priors()
    expected = 0.7 * neural_probs + 0.3 * bandit_priors

    assert np.allclose(combined, expected)


def test_bayesian_blend_selects_best_combined():
    """Test Bayesian blend selects arm with highest combined score."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND)

    # Neural slightly prefers arm 0
    neural_probs = np.array([0.4, 0.35, 0.25])

    # Bandit strongly prefers arm 1
    bandit.success[1] = 20.0
    bandit.fail[1] = 1.0

    # Run 10 trials - should consistently choose arm with highest blend
    results = [bandit.select_with_strategy(neural_probs) for _ in range(10)]
    arms = [r[0] for r in results]

    # Should choose arm 1 (bandit prior dominates)
    assert all(a == 1 for a in arms)


def test_bayesian_blend_debug_info():
    """Test Bayesian blend returns full debug info."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND)
    neural_probs = np.array([0.5, 0.3, 0.2])

    arm, debug = bandit.select_with_strategy(neural_probs)

    assert debug['strategy'] == 'bayesian_blend'
    assert debug['mode'] == 'blend'
    assert 'neural_probs' in debug
    assert 'bandit_priors' in debug
    assert 'combined' in debug
    assert len(debug['combined']) == 3


def test_bayesian_blend_uniform_prior():
    """Test Bayesian blend with uniform bandit prior."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.BAYESIAN_BLEND)

    # Neural says arm 2 is best
    neural_probs = np.array([0.2, 0.3, 0.5])

    # Bandit uniform (Beta(1,1) for all)

    arm, debug = bandit.select_with_strategy(neural_probs)

    # With uniform bandit prior (0.5, 0.5, 0.5):
    # combined = 0.7 * [0.2, 0.3, 0.5] + 0.3 * [0.5, 0.5, 0.5]
    #          = [0.14, 0.21, 0.35] + [0.15, 0.15, 0.15]
    #          = [0.29, 0.36, 0.50]

    # Should still choose arm 2
    assert arm == 2


# ============================================================================
# Test 19-21: Strategy - Pure Thompson
# ============================================================================

def test_pure_thompson_ignores_neural():
    """Test Pure Thompson ignores neural network entirely."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.PURE_THOMPSON)

    # Make arm 1 clearly best in bandit
    bandit.success[1] = 20.0
    bandit.fail[1] = 1.0

    # Neural says arm 0 is best (should be ignored)
    neural_probs = np.array([0.9, 0.05, 0.05])

    # Run 1000 trials
    results = [bandit.select_with_strategy(neural_probs) for _ in range(1000)]
    arms = [r[0] for r in results]

    # Arm 1 should dominate (>80%) since bandit prefers it
    assert arms.count(1) > 800


def test_pure_thompson_samples_from_beta():
    """Test Pure Thompson samples from Beta distributions."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.PURE_THOMPSON)

    # Irrelevant neural probs
    neural_probs = np.array([0.33, 0.33, 0.34])

    # Make arm 0 best
    bandit.success[0] = 15.0
    bandit.fail[0] = 1.0

    # Run trials
    results = [bandit.select_with_strategy(neural_probs) for _ in range(1000)]
    arms = [r[0] for r in results]

    # Arm 0 should be chosen most (>85%)
    assert arms.count(0) > 850


def test_pure_thompson_debug_info():
    """Test Pure Thompson returns correct debug info."""
    bandit = TSBandit(n_arms=3, strategy=BanditStrategy.PURE_THOMPSON)
    neural_probs = np.array([0.5, 0.3, 0.2])

    arm, debug = bandit.select_with_strategy(neural_probs)

    assert debug['strategy'] == 'pure_thompson'
    assert debug['mode'] == 'pure_thompson'
    assert 'sampled_values' in debug
    assert len(debug['sampled_values']) == 3


# ============================================================================
# Test 22-25: Bayesian Updates
# ============================================================================

def test_update_success():
    """Test update() with positive reward (success)."""
    bandit = TSBandit(n_arms=3)

    # Initial: Beta(1, 1)
    assert bandit.success[0] == 1.0
    assert bandit.fail[0] == 1.0

    # Success with reward 0.8
    bandit.update(arm=0, reward=0.8)

    # Success should increase
    assert bandit.success[0] == 1.8
    assert bandit.fail[0] == 1.0


def test_update_failure():
    """Test update() with negative reward (failure)."""
    bandit = TSBandit(n_arms=3)

    # Initial: Beta(1, 1)
    assert bandit.success[1] == 1.0
    assert bandit.fail[1] == 1.0

    # Failure with reward -0.3
    bandit.update(arm=1, reward=-0.3)

    # Fail should increase (abs value)
    assert bandit.success[1] == 1.0
    assert bandit.fail[1] == 1.3


def test_update_multiple_arms():
    """Test update() on multiple arms independently."""
    bandit = TSBandit(n_arms=3)

    # Update arm 0: success
    bandit.update(0, 1.0)

    # Update arm 1: failure
    bandit.update(1, -0.5)

    # Update arm 2: success
    bandit.update(2, 0.7)

    # Check each arm updated independently
    assert bandit.success[0] == 2.0
    assert bandit.fail[0] == 1.0

    assert bandit.success[1] == 1.0
    assert bandit.fail[1] == 1.5

    assert bandit.success[2] == 1.7
    assert bandit.fail[2] == 1.0


def test_get_stats():
    """Test get_stats() returns accurate statistics."""
    bandit = TSBandit(n_arms=3)

    # Update arm 0
    bandit.success[0] = 10.0
    bandit.fail[0] = 2.0
    bandit.pulls[0] = 15

    # Update arm 1
    bandit.success[1] = 5.0
    bandit.fail[1] = 5.0
    bandit.pulls[1] = 10

    stats = bandit.get_stats()

    # Arm 0: mean = 10/(10+2) = 0.833
    assert abs(stats[0]['mean'] - 0.833) < 0.01
    assert stats[0]['success'] == 10.0
    assert stats[0]['fail'] == 2.0
    assert stats[0]['pulls'] == 15.0

    # Arm 1: mean = 5/(5+5) = 0.5
    assert abs(stats[1]['mean'] - 0.5) < 0.01
    assert stats[1]['success'] == 5.0
    assert stats[1]['fail'] == 5.0
    assert stats[1]['pulls'] == 10.0

    # Arm 2: unchanged
    assert abs(stats[2]['mean'] - 0.5) < 0.01
