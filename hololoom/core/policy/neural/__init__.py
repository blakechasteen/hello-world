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

from .meta_learning import (
    MetaAlgorithm,
    MetaLearner,
    MetaLearningConfig,
    Task,
    generate_linear_task,
    generate_sinusoid_task,
)
from .twin_networks import (
    CounterfactualQuery,
    CounterfactualReasoner,
    CounterfactualResult,
    InterventionType,
    TwinNetwork,
)
from .value_functions import (
    ActorCriticPyTorch,
    Experience,
    QNetworkPyTorch,
    ValueEstimate,
    ValueFunctionLearner,
    ValueNetworkPyTorch,
)

__all__ = [
    # Twin networks
    'TwinNetwork',
    'CounterfactualQuery',
    'CounterfactualResult',
    'CounterfactualReasoner',
    'InterventionType',

    # Meta-learning
    'MetaLearner',
    'MetaLearningConfig',
    'MetaAlgorithm',
    'Task',
    'generate_sinusoid_task',
    'generate_linear_task',

    # Value functions
    'ValueNetworkPyTorch',
    'QNetworkPyTorch',
    'ActorCriticPyTorch',
    'ValueFunctionLearner',
    'Experience',
    'ValueEstimate',
]
