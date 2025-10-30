"""
HoloLoom Neural Components

Option C: Deep Enhancement - Advanced neural architectures.

Provides neural components for the cognitive architecture:
- Twin networks for counterfactual reasoning
- Meta-learning for fast adaptation
- Learned value functions for decision making

Public API:
    TwinNetwork: Counterfactual reasoning via parallel models
    MetaLearner: Fast adaptation to new tasks
    ValueNetwork: Learned value functions
"""

from .twin_networks import (
    TwinNetwork,
    CounterfactualQuery,
    CounterfactualResult,
    CounterfactualReasoner,
    InterventionType,
)

__all__ = [
    # Twin networks
    'TwinNetwork',
    'CounterfactualQuery',
    'CounterfactualResult',
    'CounterfactualReasoner',
    'InterventionType',
]
