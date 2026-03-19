"""
Neural Thompson Sampling policy implementation.

Main entry point for bandit-based action selection in HoloLoom.
Handles Thompson Sampling, online learning, and diagnostics.
"""

import time
from typing import Any

import torch

from hololoom.bandits.neural_ts.featurizer import ContextActionFeaturizer
from hololoom.bandits.neural_ts.posterior import BootstrapPosterior, MCDropoutPosterior
from hololoom.bandits.neural_ts.replay import ReplayBuffer
from hololoom.bandits.neural_ts.trainer import BanditTrainer
from hololoom.bandits.neural_ts.types import Action, Context, Observation


class NeuralThompsonPolicy:
    """
    Neural Thompson Sampling policy for contextual bandits.

    Main interface for HoloLoom's bandit-based exploration. Combines:
    - Thompson Sampling for exploration/exploitation
    - Neural reward models with uncertainty quantification
    - Online learning from observations
    - Diagnostic logging

    Attributes:
        posterior: Uncertainty-aware reward model (Bootstrap or MC-Dropout)
        featurizer: Converts (context, actions) → feature matrix
        replay: Replay buffer for training data
        trainer: Online trainer for posterior updates
        train_every: Update frequency (observations between training)
        train_steps: Training steps per update
        device: Compute device

    Example:
        >>> from hololoom.bandits import create_neural_ts_policy
        >>> from hololoom.bandits.config import BanditConfig
        >>>
        >>> config = BanditConfig()  # Defaults to Bootstrap
        >>> policy = create_neural_ts_policy(config)
        >>>
        >>> # Selection
        >>> ctx = Context(id="trace_123", x=embeddings)
        >>> actions = [Action(id="tool_a"), Action(id="tool_b")]
        >>> chosen = policy.select(ctx, actions)
        >>>
        >>> # Learning
        >>> obs = Observation(ctx.id, chosen.id, reward=0.92)
        >>> policy.update(obs)
    """

    def __init__(
        self,
        posterior: BootstrapPosterior | MCDropoutPosterior,
        featurizer: ContextActionFeaturizer,
        replay: ReplayBuffer,
        trainer: BanditTrainer,
        train_every: int = 200,
        train_steps: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize Neural Thompson Sampling policy.

        Args:
            posterior: Posterior for Thompson Sampling
            featurizer: Feature engineering
            replay: Replay buffer
            trainer: Online trainer
            train_every: Observations between training runs
            train_steps: Training steps per run
            device: Compute device

        Raises:
            ValueError: If train_every or train_steps <= 0
        """
        if train_every <= 0:
            raise ValueError(f"train_every must be positive, got {train_every}")
        if train_steps <= 0:
            raise ValueError(f"train_steps must be positive, got {train_steps}")

        self.posterior = posterior
        self.featurizer = featurizer
        self.replay = replay
        self.trainer = trainer
        self.train_every = train_every
        self.train_steps = train_steps
        self.device = device

        # State
        self._obs_since_train = 0
        self._total_selections = 0
        self._warmup_mode = True  # ε-greedy until replay ready

        # Cache for async add (context_id → (Context, Action))
        self._selection_cache: dict[str, tuple[Context, Action]] = {}
        self._max_cache_size = 10000

    @torch.inference_mode()
    def select(
        self,
        ctx: Context,
        actions: list[Action],
        return_diagnostics: bool = False,
    ) -> Action | tuple[Action, dict[str, Any]]:
        """
        Select action using Thompson Sampling.

        Args:
            ctx: Current context (features + metadata)
            actions: Candidate actions
            return_diagnostics: If True, return (action, diagnostics_dict)

        Returns:
            Selected action, or (action, diagnostics) if return_diagnostics=True

        Raises:
            ValueError: If actions list is empty

        Example:
            >>> action, diag = policy.select(ctx, actions, return_diagnostics=True)
            >>> print(diag["sampled_head"])  # Bootstrap: which head sampled
            >>> print(diag["pred_mean"])  # Predicted reward
        """
        if not actions:
            raise ValueError("actions list cannot be empty")

        start_time = time.perf_counter()

        # Featurize all actions
        features = self.featurizer(ctx, actions)  # [len(actions), feature_dim]
        x = torch.from_numpy(features).float().to(self.device)

        # Thompson Sampling: sample function from posterior
        f_theta = self.posterior.sample_fn()
        predictions = f_theta(x).cpu().numpy()  # [len(actions)]

        # Greedy w.r.t. sampled function
        best_idx = int(predictions.argmax())
        selected_action = actions[best_idx]

        # Cache for update() [context_id → (ctx, action)]
        self._selection_cache[ctx.id] = (ctx, selected_action)
        if len(self._selection_cache) > self._max_cache_size:
            # Evict oldest (FIFO-ish)
            oldest_key = next(iter(self._selection_cache))
            del self._selection_cache[oldest_key]

        self._total_selections += 1

        # Diagnostics
        latency_ms = (time.perf_counter() - start_time) * 1000

        diagnostics = {
            "selected_action_id": selected_action.id,
            "pred_mean": float(predictions[best_idx]),
            "pred_std": None,  # Computed if requested
            "all_preds": predictions.tolist(),
            "latency_ms": latency_ms,
            "warmup_mode": self._warmup_mode,
            "total_selections": self._total_selections,
        }

        # Add uncertainty if available
        if isinstance(self.posterior, BootstrapPosterior):
            # Can compute ensemble std (expensive, only if requested)
            if return_diagnostics:
                with torch.no_grad():
                    uncertainty = self.posterior.epistemic_uncertainty(x).cpu().numpy()
                    diagnostics["pred_std"] = float(uncertainty[best_idx])
        elif isinstance(self.posterior, MCDropoutPosterior):
            # MC-Dropout uncertainty (multiple forward passes)
            if return_diagnostics:
                mean, std = self.posterior.mc_predict(x, n_samples=10)
                diagnostics["pred_std"] = float(std[best_idx].cpu().numpy())

        if return_diagnostics:
            return selected_action, diagnostics
        else:
            return selected_action

    def update(self, obs: Observation) -> None:
        """
        Update policy from observation (online learning).

        Adds observation to replay buffer and triggers training when ready.

        Args:
            obs: Observation with context_id, action_id, reward

        Raises:
            KeyError: If context_id not in cache (must call select() first)

        Example:
            >>> obs = Observation(
            ...     context_id="trace_123",
            ...     action_id="tool_a",
            ...     reward=0.92
            ... )
            >>> policy.update(obs)
        """
        # Retrieve cached context and action
        if obs.context_id not in self._selection_cache:
            raise KeyError(
                f"Context {obs.context_id} not found in cache. "
                f"Must call select() before update()."
            )

        ctx, action = self._selection_cache[obs.context_id]

        # Verify action matches
        if action.id != obs.action_id:
            import warnings
            warnings.warn(
                f"Action mismatch: cache has {action.id}, obs has {obs.action_id}. "
                f"Using cached action."
            )

        # Add to replay
        self.replay.add_observation(obs, ctx, action)
        self._obs_since_train += 1

        # Check if replay is ready (exit warmup)
        if self._warmup_mode and self.replay.is_ready():
            self._warmup_mode = False

        # Train if enough observations accumulated
        if (
            not self._warmup_mode
            and self._obs_since_train >= self.train_every
        ):
            self.trainer.fit_steps(self.replay, self.train_steps)
            self._obs_since_train = 0

    def recommend_k(
        self,
        ctx: Context,
        actions: list[Action],
        k: int,
    ) -> list[Action]:
        """
        Recommend top-k actions (for ensembling/exploration).

        Default implementation: sample k times with Thompson Sampling.
        Returns actions in order of selection frequency.

        Args:
            ctx: Current context
            actions: Candidate actions
            k: Number of actions to recommend

        Returns:
            Top-k actions ranked by Thompson Sampling frequency

        Raises:
            ValueError: If k > len(actions)

        Example:
            >>> top3 = policy.recommend_k(ctx, actions, k=3)
            >>> print([a.id for a in top3])
        """
        if k > len(actions):
            raise ValueError(f"k ({k}) > len(actions) ({len(actions)})")

        # Sample k times, count selections
        selection_counts = {action.id: 0 for action in actions}

        for _ in range(k * 10):  # Oversample for better ranking
            selected = self.select(ctx, actions, return_diagnostics=False)
            selection_counts[selected.id] += 1

        # Sort by count
        sorted_actions = sorted(
            actions,
            key=lambda a: selection_counts[a.id],
            reverse=True,
        )

        return sorted_actions[:k]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get policy statistics.

        Returns:
            Dict with:
                - total_selections: Total select() calls
                - warmup_mode: Whether in warmup (ε-greedy)
                - replay_stats: Replay buffer stats
                - trainer_stats: Trainer stats
                - obs_since_train: Observations since last training

        Example:
            >>> stats = policy.get_statistics()
            >>> print(f"Selections: {stats['total_selections']}")
            >>> print(f"Replay size: {stats['replay_stats']['size']}")
        """
        return {
            "total_selections": self._total_selections,
            "warmup_mode": self._warmup_mode,
            "obs_since_train": self._obs_since_train,
            "replay_stats": self.replay.get_statistics(),
            "trainer_stats": self.trainer.get_statistics(),
            "cache_size": len(self._selection_cache),
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save policy checkpoint.

        Saves trainer state (models + optimizers).
        Replay buffer not saved (too large, regenerate from logs).

        Args:
            path: Path to save checkpoint

        Example:
            >>> policy.save_checkpoint("checkpoints/bandit_epoch_100.pt")
        """
        self.trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load policy checkpoint.

        Args:
            path: Path to checkpoint file

        Example:
            >>> policy.load_checkpoint("checkpoints/bandit_epoch_100.pt")
        """
        self.trainer.load_checkpoint(path)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"NeuralThompsonPolicy(selections={stats['total_selections']}, "
            f"warmup={stats['warmup_mode']}, "
            f"replay_size={stats['replay_stats']['size']})"
        )
