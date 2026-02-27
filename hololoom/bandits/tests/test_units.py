"""
Unit tests for bandit components.

Tests each module in isolation:
- Types (Context, Action, Observation)
- Models (MLP)
- Posterior (Bootstrap, MC-Dropout)
- Replay buffer
- Trainer
- Featurizer
- Evaluator
- Policy
"""

import pytest
import numpy as np
import torch
from hololoom.bandits.neural_ts.types import Context, Action, Observation
from hololoom.bandits.neural_ts.models import MLP, create_mlp_ensemble, estimate_model_size
from hololoom.bandits.neural_ts.posterior import (
    BootstrapPosterior,
    MCDropoutPosterior,
    create_bootstrap_posterior,
    create_mc_dropout_posterior,
)
from hololoom.bandits.neural_ts.replay import ReplayBuffer
from hololoom.bandits.neural_ts.trainer import BanditTrainer
from hololoom.bandits.neural_ts.featurizer import ContextActionFeaturizer, create_loom_featurizer
from hololoom.bandits.neural_ts.eval import BanditEvaluator, compute_cumulative_regret
from hololoom.bandits.config import BanditConfig, create_neural_ts_policy


# ============================================================================
# Types
# ============================================================================


def test_context_creation():
    """Test Context creation and validation."""
    x = np.random.randn(384)
    ctx = Context(id="ctx_123", x=x, meta={"task": "factual"})

    assert ctx.id == "ctx_123"
    assert ctx.x.shape == (384,)
    assert ctx.meta["task"] == "factual"


def test_context_validation():
    """Test Context dimension validation."""
    # 2D array should fail
    with pytest.raises(ValueError, match="must be 1D"):
        Context(id="ctx", x=np.random.randn(10, 10))

    # Non-ndarray should fail
    with pytest.raises(TypeError, match="must be np.ndarray"):
        Context(id="ctx", x=[1, 2, 3])


def test_action_creation():
    """Test Action creation."""
    # Without features
    action = Action(id="tool_a")
    assert action.id == "tool_a"
    assert action.a is None

    # With features
    a = np.array([1.0, 0.5, 0.0])
    action = Action(id="tool_b", a=a, meta={"cost": 0.01})
    assert action.a.shape == (3,)
    assert action.meta["cost"] == 0.01


def test_observation_creation():
    """Test Observation creation."""
    obs = Observation(
        context_id="ctx_1",
        action_id="act_1",
        reward=0.92,
        info={"success": True},
    )

    assert obs.context_id == "ctx_1"
    assert obs.action_id == "act_1"
    assert obs.reward == 0.92
    assert obs.info["success"] is True
    assert obs.ts > 0


# ============================================================================
# Models
# ============================================================================


def test_mlp_forward():
    """Test MLP forward pass."""
    mlp = MLP(in_dim=64, hidden=[32, 16], out_dim=1, dropout_p=0.0)

    x = torch.randn(10, 64)
    out = mlp(x)

    assert out.shape == (10,)  # Squeezed output


def test_mlp_parameter_count():
    """Test MLP parameter counting."""
    mlp = MLP(in_dim=64, hidden=[32, 16], out_dim=1)
    n_params = mlp.num_parameters()

    # 64*32 + 32 + 32*16 + 16 + 16*1 + 1 = 2048 + 32 + 512 + 16 + 16 + 1 = 2625
    assert n_params == 2625


def test_mlp_ensemble_creation():
    """Test ensemble factory."""
    ensemble = create_mlp_ensemble(
        n_models=5,
        in_dim=32,
        hidden=[16, 8],
        device="cpu",
    )

    assert len(ensemble) == 5
    assert all(isinstance(m, MLP) for m in ensemble)
    assert all(m.in_dim == 32 for m in ensemble)


def test_estimate_model_size():
    """Test model size estimation."""
    size = estimate_model_size(in_dim=384, hidden=[256, 128])

    assert size["params"] > 0
    assert size["memory_mb"] > 0
    assert size["memory_mb"] < 10  # Should be small


# ============================================================================
# Posterior
# ============================================================================


def test_bootstrap_posterior_sample():
    """Test Bootstrap posterior sampling."""
    models = create_mlp_ensemble(n_models=3, in_dim=32, hidden=[16])
    posterior = BootstrapPosterior(models)

    # Sample should return one model
    f_theta = posterior.sample_fn()
    assert isinstance(f_theta, MLP)
    assert f_theta.training is False  # Should be in eval mode


def test_bootstrap_ensemble_predict():
    """Test Bootstrap ensemble prediction."""
    models = create_mlp_ensemble(n_models=3, in_dim=32, hidden=[16])
    posterior = BootstrapPosterior(models)

    x = torch.randn(5, 32)

    # Mean prediction
    mean = posterior.ensemble_predict(x, mode="mean")
    assert mean.shape == (5,)

    # Samples
    samples = posterior.ensemble_predict(x, mode="samples")
    assert len(samples) == 3
    assert all(s.shape == (5,) for s in samples)


def test_bootstrap_uncertainty():
    """Test Bootstrap epistemic uncertainty."""
    models = create_mlp_ensemble(n_models=5, in_dim=32, hidden=[16])
    posterior = BootstrapPosterior(models)

    x = torch.randn(10, 32)
    uncertainty = posterior.epistemic_uncertainty(x)

    assert uncertainty.shape == (10,)
    assert (uncertainty >= 0).all()  # Std is non-negative


def test_mc_dropout_posterior():
    """Test MC-Dropout posterior."""
    model = MLP(in_dim=32, hidden=[16], dropout_p=0.1)
    posterior = MCDropoutPosterior(model, n_samples=10)

    # Sample should return model with dropout ON
    f_theta = posterior.sample_fn()
    assert f_theta.training is True  # Dropout active


def test_mc_dropout_predict():
    """Test MC-Dropout prediction."""
    model = MLP(in_dim=32, hidden=[16], dropout_p=0.1)
    posterior = MCDropoutPosterior(model, n_samples=20)

    x = torch.randn(5, 32)
    mean, std = posterior.mc_predict(x, n_samples=20)

    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert (std >= 0).all()


# ============================================================================
# Replay Buffer
# ============================================================================


def test_replay_buffer_add():
    """Test adding to replay buffer."""
    buffer = ReplayBuffer(capacity=100, warmup=10)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    action = Action(id="act_1")
    obs = Observation(context_id="ctx_1", action_id="act_1", reward=0.8)

    buffer.add_observation(obs, ctx, action)

    assert len(buffer) == 1
    assert not buffer.is_ready()  # Below warmup


def test_replay_buffer_warmup():
    """Test replay buffer warmup."""
    buffer = ReplayBuffer(capacity=100, warmup=5)

    for i in range(10):
        ctx = Context(id=f"ctx_{i}", x=np.random.randn(32))
        action = Action(id="act_1")
        obs = Observation(context_id=ctx.id, action_id="act_1", reward=0.5)
        buffer.add_observation(obs, ctx, action)

    assert buffer.is_ready()
    assert len(buffer) == 10


def test_replay_buffer_sample():
    """Test replay buffer sampling."""
    buffer = ReplayBuffer(capacity=100, warmup=5)

    # Add data
    for i in range(20):
        ctx = Context(id=f"ctx_{i}", x=np.random.randn(32))
        action = Action(id="act_1")
        obs = Observation(context_id=ctx.id, action_id="act_1", reward=0.5)
        buffer.add_observation(obs, ctx, action)

    # Sample
    features, rewards = buffer.sample(batch_size=10, bootstrap=False)

    assert features.shape == (10, 32)
    assert rewards.shape == (10,)


def test_replay_buffer_bootstrap_sample():
    """Test bootstrap sampling (with replacement)."""
    buffer = ReplayBuffer(capacity=100, warmup=5)

    for i in range(20):
        ctx = Context(id=f"ctx_{i}", x=np.random.randn(32))
        action = Action(id="act_1")
        obs = Observation(context_id=ctx.id, action_id="act_1", reward=0.5)
        buffer.add_observation(obs, ctx, action)

    # Bootstrap sample can be larger than buffer
    features, rewards = buffer.sample(batch_size=50, bootstrap=True)

    assert features.shape == (50, 32)
    assert rewards.shape == (50,)


# ============================================================================
# Trainer
# ============================================================================


def test_trainer_bootstrap():
    """Test Bootstrap trainer."""
    posterior = create_bootstrap_posterior(
        n_models=3, in_dim=32, hidden=[16], device="cpu"
    )
    trainer = BanditTrainer(posterior, lr=1e-3, batch_size=16)

    # Create replay buffer with data
    buffer = ReplayBuffer(capacity=100, warmup=20)
    for i in range(50):
        ctx = Context(id=f"ctx_{i}", x=np.random.randn(32))
        action = Action(id="act_1")
        obs = Observation(context_id=ctx.id, action_id="act_1", reward=np.random.rand())
        buffer.add_observation(obs, ctx, action)

    # Train
    stats = trainer.fit_steps(buffer, n_steps=10)

    assert stats["total_steps"] == 10
    assert stats["mean_loss"] >= 0


def test_trainer_mc_dropout():
    """Test MC-Dropout trainer."""
    posterior = create_mc_dropout_posterior(
        in_dim=32, hidden=[16], dropout_p=0.1, device="cpu"
    )
    trainer = BanditTrainer(posterior, lr=1e-3, batch_size=16)

    # Create replay buffer
    buffer = ReplayBuffer(capacity=100, warmup=20)
    for i in range(50):
        ctx = Context(id=f"ctx_{i}", x=np.random.randn(32))
        action = Action(id="act_1")
        obs = Observation(context_id=ctx.id, action_id="act_1", reward=np.random.rand())
        buffer.add_observation(obs, ctx, action)

    # Train
    stats = trainer.fit_steps(buffer, n_steps=10)

    assert stats["total_steps"] == 10
    assert stats["mean_loss"] >= 0


# ============================================================================
# Featurizer
# ============================================================================


def test_featurizer_context_only():
    """Test context-only featurizer."""
    feat = ContextActionFeaturizer(context_dim=32, action_dim=0)

    ctx = Context(id="ctx_1", x=np.ones(32))
    actions = [Action(id=f"act_{i}") for i in range(5)]

    features = feat(ctx, actions)

    # All actions get same features (context repeated)
    assert features.shape == (5, 32)
    assert np.allclose(features[0], features[1])


def test_featurizer_context_plus_action():
    """Test context + action featurizer."""
    feat = ContextActionFeaturizer(context_dim=32, action_dim=8)

    ctx = Context(id="ctx_1", x=np.ones(32))
    actions = [
        Action(id=f"act_{i}", a=np.random.randn(8))
        for i in range(5)
    ]

    features = feat(ctx, actions)

    assert features.shape == (5, 40)  # 32 + 8
    # Actions should have different features
    assert not np.allclose(features[0], features[1])


def test_loom_featurizer():
    """Test HoloLoom-specific featurizer factory."""
    feat = create_loom_featurizer(embedding_dim=384, action_dim=0, normalize=False)

    assert feat.context_dim == 384
    assert feat.action_dim == 0
    assert feat.feature_dim == 384


def test_loom_featurizer_normalization():
    """Test L2 normalization."""
    feat = create_loom_featurizer(embedding_dim=32, normalize=True)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    actions = [Action(id="act_1")]

    features = feat(ctx, actions)

    # Should be unit norm
    norm = np.linalg.norm(features[0])
    assert abs(norm - 1.0) < 1e-6


# ============================================================================
# Evaluator
# ============================================================================


def test_evaluator_ece():
    """Test ECE computation."""
    evaluator = BanditEvaluator(n_bins=5)

    # Perfect calibration
    for i in range(100):
        pred = 0.5
        actual = 0.5
        evaluator.record(pred, actual)

    ece = evaluator.compute_ece()
    assert ece < 0.01  # Nearly perfect


def test_evaluator_ece_miscalibrated():
    """Test ECE with miscalibration."""
    evaluator = BanditEvaluator(n_bins=5)

    # Overconfident predictions
    for i in range(100):
        pred = 0.9
        actual = 0.5
        evaluator.record(pred, actual)

    ece = evaluator.compute_ece()
    assert ece > 0.3  # Badly calibrated


def test_evaluator_regret():
    """Test regret computation."""
    evaluator = BanditEvaluator()

    # Suboptimal selections
    for i in range(100):
        evaluator.record(predicted=0.5, actual=0.5)

    oracle = np.full(100, 0.8)  # Oracle gets 0.8
    regret = evaluator.compute_regret_proxy(oracle_rewards=oracle)

    assert abs(regret - 0.3) < 0.01  # Regret = 0.8 - 0.5


def test_evaluator_metrics():
    """Test comprehensive metrics."""
    evaluator = BanditEvaluator()

    for i in range(50):
        evaluator.record(predicted=0.6, actual=0.65)

    metrics = evaluator.compute_metrics()

    assert "ece" in metrics
    assert "regret_proxy" in metrics
    assert "mean_reward" in metrics
    assert metrics["count"] == 50


# ============================================================================
# Policy
# ============================================================================


def test_policy_select():
    """Test policy selection."""
    config = BanditConfig(context_dim=32, hidden_dims=[16], replay_warmup=10)
    policy = create_neural_ts_policy(config)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    actions = [Action(id=f"act_{i}") for i in range(5)]

    selected = policy.select(ctx, actions)

    assert selected in actions


def test_policy_select_with_diagnostics():
    """Test policy selection with diagnostics."""
    config = BanditConfig(context_dim=32, hidden_dims=[16])
    policy = create_neural_ts_policy(config)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    actions = [Action(id=f"act_{i}") for i in range(3)]

    selected, diag = policy.select(ctx, actions, return_diagnostics=True)

    assert selected in actions
    assert "pred_mean" in diag
    assert "latency_ms" in diag
    assert len(diag["all_preds"]) == 3


def test_policy_update():
    """Test policy update."""
    config = BanditConfig(context_dim=32, hidden_dims=[16], replay_warmup=5)
    policy = create_neural_ts_policy(config)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    actions = [Action(id=f"act_{i}") for i in range(3)]

    selected = policy.select(ctx, actions)
    obs = Observation(context_id=ctx.id, action_id=selected.id, reward=0.8)

    # Should not crash
    policy.update(obs)

    stats = policy.get_statistics()
    assert stats["replay_stats"]["size"] == 1


def test_policy_recommend_k():
    """Test top-k recommendations."""
    config = BanditConfig(context_dim=32, hidden_dims=[16])
    policy = create_neural_ts_policy(config)

    ctx = Context(id="ctx_1", x=np.random.randn(32))
    actions = [Action(id=f"act_{i}") for i in range(10)]

    top3 = policy.recommend_k(ctx, actions, k=3)

    assert len(top3) == 3
    assert all(a in actions for a in top3)


# ============================================================================
# Config
# ============================================================================


def test_config_defaults():
    """Test BanditConfig defaults."""
    config = BanditConfig()

    assert config.backend == "bootstrap"
    assert config.context_dim == 384
    assert config.action_dim == 0
    assert config.n_ensemble == 7


def test_config_loom_default():
    """Test loom_default factory."""
    config = BanditConfig.loom_default()

    assert config.context_dim == 384
    assert config.backend == "bootstrap"


def test_config_fast():
    """Test fast config."""
    config = BanditConfig.fast()

    assert config.n_ensemble == 3  # Smaller ensemble
    assert len(config.hidden_dims) == 2


def test_config_mc_dropout():
    """Test MC-Dropout config."""
    config = BanditConfig.mc_dropout()

    assert config.backend == "mc_dropout"
    assert config.dropout_p > 0


def test_create_policy_bootstrap():
    """Test policy factory with Bootstrap."""
    config = BanditConfig(backend="bootstrap", context_dim=32, hidden_dims=[16])
    policy = create_neural_ts_policy(config)

    assert isinstance(policy.posterior, BootstrapPosterior)


def test_create_policy_mc_dropout():
    """Test policy factory with MC-Dropout."""
    config = BanditConfig(backend="mc_dropout", context_dim=32, hidden_dims=[16])
    policy = create_neural_ts_policy(config)

    assert isinstance(policy.posterior, MCDropoutPosterior)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
