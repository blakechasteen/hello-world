"""
HoloLoom Thompson Sampling Core - Unified TS Underlayer

Provides a complete Thompson Sampling ecosystem:
- Discrete Bernoulli (Beta-Bernoulli for multi-armed bandits)
- Bayesian Linear (closed-form posterior for linear rewards)
- Neural Bandits (deep contextual bandits with Bootstrap/MC-Dropout)
- GP-TS (Gaussian Process Thompson Sampling)
- Deep Kernel GP-TS (neural embedding + GP kernel)

All models implement the unified ThompsonSampler protocol for seamless swapping.

Example:
    >>> from hololoom.ts_core import create_thompson_sampler
    >>>
    >>> # Discrete MAB
    >>> sampler = create_thompson_sampler("discrete", n_arms=5)
    >>> arm = sampler.select()
    >>> sampler.update(arm, reward=1.0)
    >>>
    >>> # Bayesian Linear
    >>> sampler = create_thompson_sampler("linear", context_dim=10, n_actions=3)
    >>> action = sampler.select(context)
    >>> sampler.update(context, action, reward)
    >>>
    >>> # Neural Bandit
    >>> sampler = create_thompson_sampler("neural", context_dim=384, n_actions=5)
    >>> action = sampler.select(context)
    >>> sampler.update(context, action, reward)
    >>>
    >>> # GP-TS
    >>> sampler = create_thompson_sampler("gp", param_dim=5)
    >>> params = sampler.select()  # Continuous parameter vector
    >>> sampler.update(params, reward)
"""

from hololoom.ts_core.base import ThompsonSampler, ThompsonSamplerConfig
from hololoom.ts_core.samplers import create_thompson_sampler

__all__ = [
    "ThompsonSampler",
    "ThompsonSamplerConfig",
    "create_thompson_sampler",
]
