"""
Replay buffer for contextual bandit learning.

Stores (context, action, reward) observations for offline training.
Supports bootstrap sampling for ensemble training and priority sampling (future).
"""

import random
import numpy as np
import numpy.typing as npt
from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from HoloLoom.bandits.neural_ts.types import Observation, Context, Action


@dataclass
class ReplayTransition:
    """
    Single transition for training.

    Combines context features, action features (if any), and reward.
    Stored in compact numpy format for efficient batching.

    Attributes:
        x: Context features, shape [context_dim]
        a: Action features, shape [action_dim] or None
        reward: Scalar reward
        context_id: For tracing/debugging
        action_id: For tracing/debugging
        ts: Timestamp
    """
    x: npt.NDArray[np.floating]
    a: npt.NDArray[np.floating] | None
    reward: float
    context_id: str = ""
    action_id: str = ""
    ts: float = 0.0

    @property
    def features(self) -> npt.NDArray[np.floating]:
        """Concatenated [x, a] or just x if action features not used."""
        if self.a is None:
            return self.x
        return np.concatenate([self.x, self.a])


class ReplayBuffer:
    """
    Circular buffer for bandit observations.

    Stores recent transitions for online learning. Supports:
    - Bootstrap sampling (sample with replacement, different for each ensemble head)
    - Warmup period (don't train until sufficient data)
    - Efficient batching

    Attributes:
        capacity: Maximum buffer size
        buffer: Deque of transitions
        warmup: Minimum transitions before training
        priority: Whether to use priority sampling (future)

    Example:
        >>> buffer = ReplayBuffer(capacity=10000, warmup=1000)
        >>> obs = Observation(context_id="ctx1", action_id="act1", reward=0.9)
        >>> # ... assume ctx and action are available ...
        >>> buffer.add_observation(obs, ctx, action)
        >>>
        >>> if buffer.is_ready():
        ...     batch = buffer.sample(batch_size=256, bootstrap=True)
        ...     # Train on batch
    """

    def __init__(
        self,
        capacity: int = 200_000,
        warmup: int = 5_000,
        priority: bool = False,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            warmup: Minimum transitions before training
            priority: Enable priority sampling (not yet implemented)

        Raises:
            ValueError: If capacity <= warmup
        """
        if capacity <= warmup:
            raise ValueError(f"capacity ({capacity}) must be > warmup ({warmup})")

        self.capacity = capacity
        self.warmup = warmup
        self.priority = priority

        self.buffer: deque[ReplayTransition] = deque(maxlen=capacity)

        # Statistics
        self._total_added = 0
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0

    def add_observation(
        self,
        obs: Observation,
        ctx: Context,
        action: Action,
    ) -> None:
        """
        Add observation to buffer.

        Args:
            obs: Observation with reward
            ctx: Context that was used
            action: Action that was selected

        Note:
            Automatically handles circular buffer behavior (oldest evicted).
        """
        transition = ReplayTransition(
            x=ctx.x,
            a=action.a,
            reward=obs.reward,
            context_id=obs.context_id,
            action_id=obs.action_id,
            ts=obs.ts,
        )

        self.buffer.append(transition)
        self._total_added += 1
        self._reward_sum += obs.reward
        self._reward_sq_sum += obs.reward ** 2

    def add(self, obs: Observation) -> None:
        """
        Legacy add method (requires context/action lookup).

        For now, raises NotImplementedError. Use add_observation() instead.

        TODO: Implement context/action cache for async adds.
        """
        raise NotImplementedError(
            "Use add_observation(obs, ctx, action) instead. "
            "Context/action cache not yet implemented."
        )

    def sample(
        self,
        batch_size: int,
        bootstrap: bool = False,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Sample batch for training.

        Args:
            batch_size: Number of transitions to sample
            bootstrap: Sample with replacement (for Bootstrap ensemble)

        Returns:
            Tuple of (features, rewards):
                - features: shape [batch_size, feature_dim]
                - rewards: shape [batch_size]

        Raises:
            ValueError: If batch_size > len(buffer) and bootstrap=False

        Example:
            >>> features, rewards = buffer.sample(batch_size=256, bootstrap=True)
            >>> # Convert to torch and train
        """
        if not bootstrap and batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size ({batch_size}) > buffer size ({len(self.buffer)}) "
                f"with bootstrap=False"
            )

        # Sample indices
        if bootstrap:
            indices = random.choices(range(len(self.buffer)), k=batch_size)
        else:
            indices = random.sample(range(len(self.buffer)), k=batch_size)

        # Extract features and rewards
        features_list = []
        rewards_list = []

        for idx in indices:
            trans = self.buffer[idx]
            features_list.append(trans.features)
            rewards_list.append(trans.reward)

        features = np.stack(features_list)  # [batch_size, feature_dim]
        rewards = np.array(rewards_list)  # [batch_size]

        return features, rewards

    def is_ready(self) -> bool:
        """Check if buffer has enough data for training."""
        return len(self.buffer) >= self.warmup

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self.buffer)

    def get_statistics(self) -> dict[str, float]:
        """
        Get buffer statistics.

        Returns:
            Dict with:
                - size: Current buffer size
                - total_added: Total observations ever added
                - mean_reward: Mean reward over buffer
                - std_reward: Std dev of rewards
                - utilization: size / capacity
        """
        n = len(self.buffer)

        if n == 0:
            return {
                "size": 0,
                "total_added": self._total_added,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "utilization": 0.0,
            }

        # Recompute from buffer (handles wraparound)
        rewards = [t.reward for t in self.buffer]
        mean = np.mean(rewards)
        std = np.std(rewards)

        return {
            "size": n,
            "total_added": self._total_added,
            "mean_reward": float(mean),
            "std_reward": float(std),
            "utilization": n / self.capacity,
        }

    def clear(self) -> None:
        """Clear buffer (for testing/reset)."""
        self.buffer.clear()
        # Keep total_added and reward stats for diagnostics

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ReplayBuffer(size={stats['size']}/{self.capacity}, "
            f"warmup={self.warmup}, mean_reward={stats['mean_reward']:.3f})"
        )
