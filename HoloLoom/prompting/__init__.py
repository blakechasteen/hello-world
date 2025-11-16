"""
Promptly Strategy Framework

Extensible prompting strategy system implementing the Strategy Pattern.
Each advanced prompting technique (verification, adversarial, scaffolding, etc.)
is implemented as a strategy that can be:
- Composed with other strategies
- Auto-detected based on query context
- Configured via YAML
- Discovered automatically from the strategies directory
"""

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

__all__ = [
    'PromptingStrategy',
    'StrategyContext',
    'StrategyResult',
    'StrategyRegistry',
    'get_strategy',
    'suggest_strategies',
    'list_strategies',
    'CompositeStrategy',
    'AutoDetector'
]
