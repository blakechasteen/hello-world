"""
HoloLoom Prompting Module

Comprehensive prompting system with two major components:

1. **Promptly Strategy Framework** (legacy)
   - Extensible prompting strategy system implementing the Strategy Pattern
   - Advanced prompting techniques (verification, adversarial, scaffolding, etc.)

2. **Contract-First Prompting** (November 2025)
   - Clarity of intent through structured questioning
   - Iterative gap identification and contract formation
   - Integration with HoloLoom memory and agentic systems

Created: 2025-11-18
"""

# Legacy Promptly exports
from .strategy import (
    PromptingStrategy,
    StrategyContext,
    StrategyResult
)
from .registry import (
    StrategyRegistry,
    get_strategy,
    suggest_strategies,
    list_strategies
)
from .composite import CompositeStrategy
from .auto_detect import AutoDetector

# Contract-First Prompting exports (November 2025)
from .contract_first import (
    ContractFirstPrompting,
    Contract,
    GapAnalysis,
    Blueprint,
    RiskAnalysis,
)
from .types import (
    DiggingStrategy,
    ConfidenceLevel,
    UserResponse,
    DeliverableType,
)
from .strategies import (
    BreadthFirstStrategy,
    DepthFirstStrategy,
    EssentialOnlyStrategy,
    AdaptiveStrategy,
)

__all__ = [
    # Legacy Promptly exports
    'PromptingStrategy',
    'StrategyContext',
    'StrategyResult',
    'StrategyRegistry',
    'get_strategy',
    'suggest_strategies',
    'list_strategies',
    'CompositeStrategy',
    'AutoDetector',
    # Contract-First Prompting exports
    'ContractFirstPrompting',
    'Contract',
    'GapAnalysis',
    'Blueprint',
    'RiskAnalysis',
    'DiggingStrategy',
    'ConfidenceLevel',
    'UserResponse',
    'DeliverableType',
    'BreadthFirstStrategy',
    'DepthFirstStrategy',
    'EssentialOnlyStrategy',
    'AdaptiveStrategy',
]
