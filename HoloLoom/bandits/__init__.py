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
    >>> from HoloLoom.bandits import create_neural_ts_policy
    >>> policy = create_neural_ts_policy(config)
    >>> action = policy.select(context, candidate_actions)
    >>> # ... execute action, compute reward ...
    >>> policy.update(Observation(ctx.id, action.id, reward))
"""

from HoloLoom.bandits.neural_ts.types import (
    Context,
    Action,
    Observation,
    BanditPolicy,
)
from HoloLoom.bandits.neural_ts.policy import NeuralThompsonPolicy
from HoloLoom.bandits.config import BanditConfig, create_neural_ts_policy

# Re-exports from merged ts_core module
from HoloLoom.bandits.ts_base import ThompsonSampler, ThompsonSamplerConfig
from HoloLoom.bandits.samplers import create_thompson_sampler

__all__ = [
    "Context",
    "Action",
    "Observation",
    "BanditPolicy",
    "NeuralThompsonPolicy",
    "BanditConfig",
    "create_neural_ts_policy",
    # From merged ts_core
    "ThompsonSampler",
    "ThompsonSamplerConfig",
    "create_thompson_sampler",
]
