"""
HoloLoom Bandits - Neural Thompson Sampling for exploration/exploitation.

This package provides uncertainty-aware contextual bandits for action selection
in HoloLoom's decision pipeline. Uses Thompson Sampling over neural reward models
with pluggable uncertainty backends (Bootstrap Ensemble, MC-Dropout, BNN).

Key Components:
- BanditPolicy: Main interface for action selection and learning
- Context/Action/Observation: Core data types
- Bootstrap/MCDropout posteriors: Uncertainty quantification
- Replay buffer + online trainer: Continual learning
- Evaluation: ECE, regret proxy, reward tracking

Example:
    >>> from hololoom.bandits import create_neural_ts_policy
    >>> policy = create_neural_ts_policy(config)
    >>> action = policy.select(context, candidate_actions)
    >>> # ... execute action, compute reward ...
    >>> policy.update(Observation(ctx.id, action.id, reward))
"""

from hololoom.bandits.config import BanditConfig, create_neural_ts_policy
from hololoom.bandits.neural_ts.policy import NeuralThompsonPolicy
from hololoom.bandits.neural_ts.types import (
    Action,
    BanditPolicy,
    Context,
    Observation,
)

__all__ = [
    "Context",
    "Action",
    "Observation",
    "BanditPolicy",
    "NeuralThompsonPolicy",
    "BanditConfig",
    "create_neural_ts_policy",
]
