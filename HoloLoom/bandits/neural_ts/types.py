"""
Core data types for Neural Thompson Sampling bandits.

These types define the contracts between the bandit policy and the rest of HoloLoom.
All types use simple dataclasses for easy serialization and testing.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
import numpy.typing as npt


@dataclass
class Context:
    """
    Contextual information for bandit decision.

    The context represents the current state in which an action must be chosen.
    In HoloLoom, this typically includes:
    - Semantic embeddings from the query
    - Motif features
    - Runtime metadata (latency budget, user segment, task type)

    Attributes:
        id: Unique identifier for tracing (e.g., from Spacetime.trace_id)
        x: Feature vector (typically from Loom embeddings, shape: [context_dim])
        meta: Optional metadata (user segment, task type, etc.)

    Example:
        >>> ctx = Context(
        ...     id="trace_123",
        ...     x=np.random.randn(512),
        ...     meta={"task_type": "factual", "user_segment": "power"}
        ... )
    """
    id: str
    x: npt.NDArray[np.floating]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate context dimensions."""
        if not isinstance(self.x, np.ndarray):
            raise TypeError(f"Context.x must be np.ndarray, got {type(self.x)}")
        if self.x.ndim != 1:
            raise ValueError(f"Context.x must be 1D, got shape {self.x.shape}")


@dataclass
class Action:
    """
    Candidate action for bandit selection.

    Actions represent choices the bandit can make (e.g., which tool to use,
    which prompt template, which route to take).

    Attributes:
        id: Unique identifier (e.g., tool name, prompt id, route key)
        a: Optional action features (e.g., prompt characteristics, tool metadata)
           If None, bandit uses context-only model
        meta: Optional metadata (cost, latency estimates, etc.)

    Example:
        >>> action = Action(
        ...     id="answer_tool",
        ...     a=np.array([1.0, 0.5, 0.0]),  # [is_expensive, avg_latency, uses_llm]
        ...     meta={"cost_estimate": 0.01, "latency_p50": 120.0}
        ... )
    """
    id: str
    a: npt.NDArray[np.floating] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate action features."""
        if self.a is not None:
            if not isinstance(self.a, np.ndarray):
                raise TypeError(f"Action.a must be np.ndarray or None, got {type(self.a)}")
            if self.a.ndim != 1:
                raise ValueError(f"Action.a must be 1D, got shape {self.a.shape}")


@dataclass
class Observation:
    """
    Outcome observation after action execution.

    The bandit learns from observations to improve future selections.
    Rewards should be normalized to [0,1] or z-scored for stable learning.

    Attributes:
        context_id: Links to Context.id for tracing
        action_id: Links to Action.id for credit assignment
        reward: Normalized reward signal (0=failure, 1=perfect)
                Can combine multiple signals (e.g., success - λ·cost)
        info: Optional diagnostic info (latency, cost, user feedback, etc.)
        ts: Unix timestamp for temporal analysis

    Example:
        >>> obs = Observation(
        ...     context_id="trace_123",
        ...     action_id="answer_tool",
        ...     reward=0.85,  # success=1.0, cost_penalty=-0.15
        ...     info={"success": True, "latency_ms": 120, "cost": 0.01},
        ...     ts=time.time()
        ... )
    """
    context_id: str
    action_id: str
    reward: float
    info: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate reward range."""
        if not isinstance(self.reward, (int, float)):
            raise TypeError(f"Observation.reward must be numeric, got {type(self.reward)}")
        # Warn if reward is outside typical range (but don't enforce, z-scores are valid)
        if self.reward < -3.0 or self.reward > 3.0:
            import warnings
            warnings.warn(
                f"Observation reward {self.reward} is outside [-3, 3]. "
                f"Ensure rewards are normalized for stable learning."
            )


class BanditPolicy(Protocol):
    """
    Protocol for bandit action selection policies.

    This defines the interface that all bandit policies must implement.
    HoloLoom's orchestrator depends only on this protocol, not concrete classes.

    Methods:
        select: Choose one action given context and candidates
        update: Learn from observation
        recommend_k: (Optional) Return top-k actions instead of one
    """

    def select(self, ctx: Context, actions: list[Action]) -> Action:
        """
        Select one action using Thompson Sampling.

        Args:
            ctx: Current context (features + metadata)
            actions: Candidate actions to choose from

        Returns:
            The selected action (one of the input actions)

        Raises:
            ValueError: If actions list is empty

        Example:
            >>> policy = create_neural_ts_policy(config)
            >>> ctx = Context(id="trace_123", x=embeddings)
            >>> actions = [Action(id="tool_a"), Action(id="tool_b")]
            >>> chosen = policy.select(ctx, actions)
            >>> print(chosen.id)  # e.g., "tool_a"
        """
        ...

    def update(self, obs: Observation) -> None:
        """
        Update policy from observation (online learning).

        Args:
            obs: Observation containing context_id, action_id, reward

        Note:
            Updates are batched internally. Actual training happens every
            `train_every` observations (configurable).

        Example:
            >>> obs = Observation(
            ...     context_id="trace_123",
            ...     action_id="tool_a",
            ...     reward=0.92
            ... )
            >>> policy.update(obs)
        """
        ...

    def recommend_k(self, ctx: Context, actions: list[Action], k: int) -> list[Action]:
        """
        Recommend top-k actions (optional, for ensembling/exploration).

        Args:
            ctx: Current context
            actions: Candidate actions
            k: Number of actions to recommend

        Returns:
            Top-k actions ranked by expected reward

        Note:
            Default implementation: sample k times with Thompson Sampling.
            Subclasses can override for more efficient implementations.
        """
        ...


class Posterior(Protocol):
    """
    Protocol for uncertainty-aware reward model posteriors.

    Posteriors provide function samples for Thompson Sampling.
    Different backends (Bootstrap, MC-Dropout, BNN) implement this protocol.
    """

    def sample_fn(self) -> Any:
        """
        Sample a reward function from the posterior.

        Returns:
            A callable model (e.g., nn.Module) representing one posterior sample.
            For Thompson Sampling: act greedily using this sampled function.

        Example (Bootstrap):
            >>> posterior = BootstrapPosterior(models=[mlp1, mlp2, mlp3])
            >>> f_theta = posterior.sample_fn()  # Randomly picks one MLP
            >>> predictions = f_theta(context_action_features)

        Example (MC-Dropout):
            >>> posterior = MCDropoutPosterior(model=mlp)
            >>> f_theta = posterior.sample_fn()  # Returns model with dropout ON
            >>> predictions = f_theta(context_action_features)  # Stochastic forward
        """
        ...
