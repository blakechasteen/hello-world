# policy/__init__.py
from .unified import (
    PolicyEngine,
    NeuralCore,
    UnifiedPolicy,
    create_policy
)

# Import Thompson Sampling from dedicated module (backward compatibility)
from .thompson_sampling import BanditStrategy, TSBandit

__all__ = [
    'PolicyEngine',
    'NeuralCore',
    'UnifiedPolicy',
    'TSBandit',
    'BanditStrategy',
    'create_policy'
]