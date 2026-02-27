"""
Tests for unified Thompson Sampling models.

Tests all TS variants:
- Discrete Bernoulli (Beta-Bernoulli)
- Bayesian Linear
- GP-TS
- Unified factory
"""

import pytest
import numpy as np
from HoloLoom.bandits.models.discrete_bernoulli import DiscreteBernoulliTS, create_discrete_ts
from HoloLoom.bandits.models.bayes_linear import BayesianLinearTS, create_bayesian_linear_ts
from HoloLoom.bandits.models.gp_ts import GaussianProcessTS, create_gp_ts
from HoloLoom.bandits.samplers import (
    create_thompson_sampler,
    create_discrete_mab,
    create_contextual_bandit,
    create_continuous_optimizer,
)


# ============================================================================
# Discrete Bernoulli Tests
# ============================================================================


def test_discrete_bernoulli_init():
    """Test discrete TS initialization."""
    sampler = DiscreteBernoulliTS(n_arms=5, alpha_prior=1.0, beta_prior=1.0)

    assert sampler.n_arms == 5
    assert len(sampler.alpha) == 5
    assert len(sampler.beta) == 5
    assert np.all(sampler.alpha == 1.0)
    assert np.all(sampler.beta == 1.0)


def test_discrete_select():
    """Test arm selection."""
    sampler = DiscreteBernoulliTS(n_arms=3, seed=42)

    arm = sampler.select()
    assert 0 <= arm < 3


def test_discrete_update():
    """Test posterior update."""
    sampler = DiscreteBernoulliTS(n_arms=3, alpha_prior=1.0, beta_prior=1.0)

    # Observe success on arm 0
    sampler.update(arm=0, reward=1.0)

    # Posterior should shift
    assert sampler.alpha[0] == 2.0  # 1 + 1
    assert sampler.beta[0] == 1.0   # 1 + 0
    assert sampler.pulls[0] == 1


def test_discrete_learning():
    """Test that discrete TS learns best arm."""
    np.random.seed(42)
    sampler = DiscreteBernoulliTS(n_arms=3, seed=42)

    # True probabilities: arm 2 is best
    true_probs = [0.3, 0.5, 0.9]

    # Run bandit
    for _ in range(200):
        arm = sampler.select()
        reward = float(np.random.rand() < true_probs[arm])
        sampler.update(arm, reward)

    # Should identify arm 2 as best
    stats = sampler.get_statistics()
    assert stats["best_arm"] == 2


def test_discrete_factory():
    """Test discrete factory functions."""
    sampler = create_discrete_ts(n_arms=5, prior="uniform")
    assert sampler.alpha_prior == 1.0
    assert sampler.beta_prior == 1.0

    sampler = create_discrete_ts(n_arms=5, prior="optimistic")
    assert sampler.alpha_prior == 10.0


# ============================================================================
# Bayesian Linear Tests
# ============================================================================


def test_bayesian_linear_init():
    """Test Bayesian linear initialization."""
    sampler = BayesianLinearTS(context_dim=10, n_actions=5)

    assert sampler.context_dim == 10
    assert sampler.n_actions == 5
    assert sampler.mu.shape == (5, 10)
    assert sampler.Sigma.shape == (5, 10, 10)


def test_bayesian_linear_select():
    """Test action selection."""
    sampler = BayesianLinearTS(context_dim=10, n_actions=5, seed=42)

    context = np.random.randn(10)
    action = sampler.select(context)

    assert 0 <= action < 5


def test_bayesian_linear_update():
    """Test posterior update."""
    sampler = BayesianLinearTS(context_dim=10, n_actions=3)

    context = np.random.randn(10)
    action = 1
    reward = 0.8

    # Before update
    mu_before = sampler.mu[action].copy()

    # Update
    sampler.update(context, action, reward)

    # Posterior should change
    assert not np.allclose(sampler.mu[action], mu_before)
    assert sampler.pulls[action] == 1


def test_bayesian_linear_learning():
    """Test that linear TS learns linear rewards."""
    np.random.seed(42)
    sampler = BayesianLinearTS(context_dim=5, n_actions=3, seed=42)

    # True linear weights for each action
    true_weights = np.random.randn(3, 5)

    # Run bandit
    for _ in range(100):
        context = np.random.randn(5)
        action = sampler.select(context)

        # True reward (linear)
        true_reward = true_weights[action] @ context
        reward = true_reward + 0.1 * np.random.randn()

        sampler.update(context, action, reward)

    # Posterior should have learned something
    stats = sampler.get_statistics()
    assert stats["total_pulls"] == 100
    assert stats["pulls"].sum() == 100


def test_bayesian_linear_factory():
    """Test Bayesian linear factory."""
    sampler = create_bayesian_linear_ts(
        context_dim=20,
        n_actions=5,
        prior="default"
    )
    assert sampler.lambda_prior == 1.0


# ============================================================================
# GP-TS Tests
# ============================================================================


def test_gp_ts_init():
    """Test GP-TS initialization."""
    sampler = GaussianProcessTS(param_dim=5, kernel="rbf")

    assert sampler.param_dim == 5
    assert sampler.kernel_type == "rbf"
    assert len(sampler.X) == 0
    assert len(sampler.y) == 0


def test_gp_select():
    """Test GP parameter selection."""
    sampler = GaussianProcessTS(param_dim=5, seed=42)

    # First selection (no data, random)
    params = sampler.select(n_candidates=10)
    assert params.shape == (5,)
    assert np.all((params >= 0.0) & (params <= 1.0))


def test_gp_update():
    """Test GP update."""
    sampler = GaussianProcessTS(param_dim=3)

    params = np.array([0.5, 0.5, 0.5])
    reward = 0.8

    sampler.update(params, reward)

    assert len(sampler.X) == 1
    assert len(sampler.y) == 1
    assert sampler.y[0] == 0.8


def test_gp_optimization():
    """Test that GP-TS finds maximum."""
    np.random.seed(42)
    sampler = GaussianProcessTS(param_dim=2, kernel="rbf", seed=42)

    # Simple quadratic: f(x) = -||x - [0.7, 0.3]||²
    center = np.array([0.7, 0.3])

    def objective(x):
        return -np.linalg.norm(x - center) ** 2

    # Run optimization
    for _ in range(30):
        params = sampler.select(n_candidates=50)
        reward = objective(params)
        sampler.update(params, reward)

    # Should find near-optimal
    best_params = sampler.get_best_params()
    best_reward = objective(best_params)

    # Should be close to maximum (0)
    assert best_reward > -0.5  # Within reasonable distance


def test_gp_factory():
    """Test GP factory."""
    sampler = create_gp_ts(param_dim=10, kernel="matern")
    assert sampler.kernel_type == "matern"


# ============================================================================
# Unified Factory Tests
# ============================================================================


def test_unified_factory_discrete():
    """Test unified factory for discrete."""
    sampler = create_thompson_sampler("discrete", n_arms=5, prior="uniform")
    assert isinstance(sampler, DiscreteBernoulliTS)
    assert sampler.n_arms == 5


def test_unified_factory_linear():
    """Test unified factory for linear."""
    sampler = create_thompson_sampler(
        "linear",
        context_dim=20,
        n_actions=5,
        prior="default"
    )
    assert isinstance(sampler, BayesianLinearTS)
    assert sampler.context_dim == 20


def test_unified_factory_gp():
    """Test unified factory for GP."""
    sampler = create_thompson_sampler("gp", param_dim=5, kernel="rbf")
    assert isinstance(sampler, GaussianProcessTS)
    assert sampler.param_dim == 5


def test_unified_factory_neural():
    """Test unified factory for neural."""
    sampler = create_thompson_sampler(
        "neural",
        context_dim=32,
        action_dim=0,
        hidden_dims=[64],
        n_ensemble=3,
        replay_warmup=10,
    )
    # Should create NeuralThompsonPolicy
    from HoloLoom.bandits.neural_ts.policy import NeuralThompsonPolicy
    assert isinstance(sampler, NeuralThompsonPolicy)


def test_convenience_functions():
    """Test convenience factory functions."""
    # Discrete MAB
    sampler = create_discrete_mab(n_arms=5)
    assert isinstance(sampler, DiscreteBernoulliTS)

    # Contextual bandit (linear)
    sampler = create_contextual_bandit(
        context_dim=20,
        n_actions=5,
        model="linear"
    )
    assert isinstance(sampler, BayesianLinearTS)

    # Continuous optimizer
    sampler = create_continuous_optimizer(param_dim=10)
    assert isinstance(sampler, GaussianProcessTS)


def test_invalid_sampler_type():
    """Test error on invalid sampler type."""
    with pytest.raises(ValueError, match="Unknown sampler_type"):
        create_thompson_sampler("invalid_type")


# ============================================================================
# Integration Tests
# ============================================================================


def test_discrete_to_linear_comparison():
    """Compare discrete (context-blind) vs linear (contextual)."""
    np.random.seed(42)

    # Discrete sampler (ignores context)
    discrete = DiscreteBernoulliTS(n_arms=3, seed=42)

    # Linear sampler (uses context)
    linear = BayesianLinearTS(context_dim=5, n_actions=3, seed=42)

    # True rewards depend on context
    true_weights = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # Action 0: likes first feature
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Action 1: likes second feature
        [0.0, 0.0, 1.0, 0.0, 0.0],  # Action 2: likes third feature
    ])

    discrete_rewards = []
    linear_rewards = []

    for _ in range(100):
        # Sample context
        context = np.random.randn(5)

        # Discrete selection (no context)
        discrete_action = discrete.select()
        discrete_reward_raw = true_weights[discrete_action] @ context
        # Normalize to [0, 1] using sigmoid
        discrete_reward = 1.0 / (1.0 + np.exp(-discrete_reward_raw))
        discrete.update(discrete_action, discrete_reward)
        discrete_rewards.append(discrete_reward_raw)  # Track raw for comparison

        # Linear selection (uses context)
        linear_action = linear.select(context)
        linear_reward = true_weights[linear_action] @ context
        linear.update(context, linear_action, linear_reward)
        linear_rewards.append(linear_reward)

    # Linear should outperform discrete (context matters)
    assert np.mean(linear_rewards[-50:]) > np.mean(discrete_rewards[-50:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
