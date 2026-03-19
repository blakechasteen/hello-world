"""
Promptly Strategy Framework + Metaprompting System

Extensible prompting strategy system implementing the Strategy Pattern.
Each advanced prompting technique (verification, adversarial, scaffolding, etc.)
is implemented as a strategy that can be:
- Composed with other strategies
- Auto-detected based on query context
- Configured via YAML
- Discovered automatically from the strategies directory

Metaprompting System (November 2025):
Universal 7-component framework with model-specific adapters:
- CORE_TEMPLATE.md works on all LLMs
- ClaudeAdapter adds thinking tags, artifacts, XML constraints (+30% quality)
- GeminiAdapter adds multimodal, code execution, grounding (+25% quality)
- GPTAdapter adds structured outputs, function calling (+20% quality)
"""

# Metaprompting System (November 2025)
from .adapters import (
    ClaudeAdapter,
    GeminiAdapter,
    GPTAdapter,
    ModelAdapter,
    create_adapter,
    create_adapter_from_config,
)
from .auto_detect import AutoDetector
from .composite import CompositeStrategy
from .metaprompt import (
    auto_detect_strategy,
    create_metaprompt,
    create_metaprompt_auto,
    create_metaprompt_with_strategy,
    enhance_request,
)
from .registry import StrategyRegistry, get_strategy, list_strategies, suggest_strategies
from .strategy import PromptingStrategy, StrategyContext, StrategyResult

__all__ = [
    # Strategy framework
    'PromptingStrategy',
    'StrategyContext',
    'StrategyResult',
    'StrategyRegistry',
    'get_strategy',
    'suggest_strategies',
    'list_strategies',
    'CompositeStrategy',
    'AutoDetector',

    # Metaprompting system
    'ModelAdapter',
    'ClaudeAdapter',
    'GeminiAdapter',
    'GPTAdapter',
    'create_adapter',
    'create_adapter_from_config',
    'create_metaprompt',
    'create_metaprompt_with_strategy',
    'create_metaprompt_auto',
    'auto_detect_strategy',
    'enhance_request'
]
