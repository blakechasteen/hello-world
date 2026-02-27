"""
Configuration and factory functions for Neural Thompson Sampling bandits.

Provides simple YAML-compatible config and one-line policy creation.
"""

from dataclasses import dataclass, field
from typing import Literal
from HoloLoom.bandits.neural_ts.policy import NeuralThompsonPolicy
from HoloLoom.bandits.neural_ts.posterior import (
    create_bootstrap_posterior,
    create_mc_dropout_posterior,
)
from HoloLoom.bandits.neural_ts.featurizer import create_loom_featurizer
from HoloLoom.bandits.neural_ts.replay import ReplayBuffer
from HoloLoom.bandits.neural_ts.trainer import BanditTrainer


@dataclass
class BanditConfig:
    """
    Configuration for Neural Thompson Sampling bandit.

    All parameters have sensible defaults for HoloLoom integration.
    Can be loaded from YAML or created programmatically.

    Attributes:
        backend: Uncertainty backend ("bootstrap" or "mc_dropout")
        context_dim: Context feature dimension (match HoloLoom embeddings)
        action_dim: Action feature dimension (0 for context-only)
        hidden_dims: Hidden layer sizes
        activation: Activation function
        n_ensemble: Bootstrap ensemble size (ignored for mc_dropout)
        dropout_p: Dropout probability (ignored for bootstrap)
        ts_samples: Thompson samples per decision (typically 1)
        lr: Learning rate
        batch_size: Training batch size
        train_every: Observations between training runs
        train_steps: Training steps per run
        replay_capacity: Replay buffer size
        replay_warmup: Minimum observations before training
        priority: Priority sampling (not yet implemented)
        device: Compute device
        normalize_features: L2-normalize features

    Example (YAML):
        ```yaml
        neural_ts:
          backend: bootstrap
          context_dim: 384
          action_dim: 0
          hidden_dims: [256, 128]
          n_ensemble: 7
          lr: 0.001
          batch_size: 256
          train_every: 200
          train_steps: 100
          replay_capacity: 200000
          replay_warmup: 5000
        ```

    Example (Python):
        >>> config = BanditConfig(
        ...     context_dim=384,  # HoloLoom's default embedding dim
        ...     action_dim=0,     # Context-only mode
        ...     backend="bootstrap",
        ... )
        >>> policy = create_neural_ts_policy(config)
    """

    # Model architecture
    backend: Literal["bootstrap", "mc_dropout"] = "bootstrap"
    context_dim: int = 384  # Match HoloLoom's Matryoshka embeddings
    action_dim: int = 0  # 0 = context-only mode
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: Literal["relu", "gelu"] = "gelu"

    # Backend-specific
    n_ensemble: int = 7  # Bootstrap only
    dropout_p: float = 0.1  # MC-Dropout only
    ts_samples: int = 1  # Thompson samples per decision (1 is standard)

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    train_every: int = 200  # Observations between training runs
    train_steps: int = 100  # Gradient steps per training run

    # Replay buffer
    replay_capacity: int = 200_000
    replay_warmup: int = 5_000  # Minimum before training
    priority: bool = False  # Priority sampling (future)

    # System
    device: str = "cpu"
    normalize_features: bool = False

    def __post_init__(self):
        """Validate config."""
        if self.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}")
        if self.action_dim < 0:
            raise ValueError(f"action_dim must be non-negative, got {self.action_dim}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if self.n_ensemble <= 0:
            raise ValueError(f"n_ensemble must be positive, got {self.n_ensemble}")
        if not (0.0 < self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in (0, 1), got {self.dropout_p}")
        if self.replay_capacity <= self.replay_warmup:
            raise ValueError(
                f"replay_capacity ({self.replay_capacity}) must be > "
                f"replay_warmup ({self.replay_warmup})"
            )

    @property
    def feature_dim(self) -> int:
        """Total feature dimension (context + action)."""
        return self.context_dim + self.action_dim

    @classmethod
    def loom_default(cls) -> "BanditConfig":
        """
        Default config for HoloLoom integration.

        Uses Bootstrap ensemble with context-only mode (no action features).
        Matches HoloLoom's 384-dim embeddings.

        Returns:
            BanditConfig with sensible defaults for HoloLoom

        Example:
            >>> config = BanditConfig.loom_default()
            >>> policy = create_neural_ts_policy(config)
        """
        return cls(
            backend="bootstrap",
            context_dim=384,
            action_dim=0,
            hidden_dims=[256, 128],
            n_ensemble=7,
            lr=1e-3,
            batch_size=256,
            train_every=200,
            train_steps=100,
            replay_capacity=200_000,
            replay_warmup=5_000,
            device="cpu",
        )

    @classmethod
    def fast(cls) -> "BanditConfig":
        """
        Fast config (smaller models, less training).

        Good for development/testing or low-latency requirements.

        Returns:
            BanditConfig optimized for speed

        Example:
            >>> config = BanditConfig.fast()
        """
        return cls(
            backend="bootstrap",
            context_dim=384,
            action_dim=0,
            hidden_dims=[128, 64],  # Smaller network
            n_ensemble=3,  # Fewer heads
            lr=1e-3,
            batch_size=128,  # Smaller batches
            train_every=500,  # Train less often
            train_steps=50,  # Fewer steps
            replay_capacity=50_000,
            replay_warmup=2_000,
            device="cpu",
        )

    @classmethod
    def mc_dropout(cls) -> "BanditConfig":
        """
        MC-Dropout config (single model, parameter-efficient).

        Uses dropout-based uncertainty instead of ensemble.
        More memory-efficient but may be less calibrated.

        Returns:
            BanditConfig for MC-Dropout backend

        Example:
            >>> config = BanditConfig.mc_dropout()
        """
        return cls(
            backend="mc_dropout",
            context_dim=384,
            action_dim=0,
            hidden_dims=[256, 128],
            dropout_p=0.1,
            lr=1e-3,
            batch_size=256,
            train_every=200,
            train_steps=100,
            replay_capacity=200_000,
            replay_warmup=5_000,
            device="cpu",
        )


def create_neural_ts_policy(config: BanditConfig) -> NeuralThompsonPolicy:
    """
    Factory for Neural Thompson Sampling policy.

    One-line policy creation from config. Handles all component initialization.

    Args:
        config: BanditConfig specifying backend, architecture, training params

    Returns:
        Fully initialized NeuralThompsonPolicy ready for use

    Raises:
        ValueError: If config is invalid

    Example:
        >>> config = BanditConfig.loom_default()
        >>> policy = create_neural_ts_policy(config)
        >>>
        >>> # Use policy
        >>> ctx = Context(id="trace_123", x=embeddings)
        >>> actions = [Action(id="tool_a"), Action(id="tool_b")]
        >>> chosen = policy.select(ctx, actions)

    Example (custom config):
        >>> config = BanditConfig(
        ...     backend="bootstrap",
        ...     context_dim=512,
        ...     action_dim=16,
        ...     hidden_dims=[512, 256, 128],
        ...     n_ensemble=10,
        ... )
        >>> policy = create_neural_ts_policy(config)
    """
    # Create featurizer
    featurizer = create_loom_featurizer(
        embedding_dim=config.context_dim,
        action_dim=config.action_dim,
        normalize=config.normalize_features,
    )

    # Create posterior (backend-specific)
    if config.backend == "bootstrap":
        posterior = create_bootstrap_posterior(
            n_models=config.n_ensemble,
            in_dim=config.feature_dim,
            hidden=config.hidden_dims,
            activation=config.activation,
            device=config.device,
        )
    elif config.backend == "mc_dropout":
        posterior = create_mc_dropout_posterior(
            in_dim=config.feature_dim,
            hidden=config.hidden_dims,
            dropout_p=config.dropout_p,
            activation=config.activation,
            n_samples=10,
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Create replay buffer
    replay = ReplayBuffer(
        capacity=config.replay_capacity,
        warmup=config.replay_warmup,
        priority=config.priority,
    )

    # Create trainer
    trainer = BanditTrainer(
        posterior=posterior,
        lr=config.lr,
        batch_size=config.batch_size,
        device=config.device,
    )

    # Create policy
    policy = NeuralThompsonPolicy(
        posterior=posterior,
        featurizer=featurizer,
        replay=replay,
        trainer=trainer,
        train_every=config.train_every,
        train_steps=config.train_steps,
        device=config.device,
    )

    return policy
