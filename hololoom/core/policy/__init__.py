# policy/__init__.py
from .unified import (
    NeuralCore,
    UnifiedPolicy,
    create_policy
)
# PolicyEngine protocol is defined in HoloLoom.protocols
from hololoom.protocols import PolicyEngine

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