"""
Promptly Integrations

- hololoom_bridge.py: HoloLoom memory integration with Thompson Sampling
- llm_backends.py: LLM provider abstraction (Ollama, Claude, OpenAI)

HoloLoom Bridge Features:
- Store prompt execution results in HoloLoom knowledge graph
- Retrieve similar prompts via semantic search
- Thompson Sampling for prompt selection optimization
- Learn which prompts work best for which task types
"""

from .hololoom_bridge import (
    # Data classes
    PromptExecution,
    PromptStats,

    # Main classes
    HoloLoomBridge,
    ThompsonSamplerPrompts,

    # Convenience functions
    create_bridge,
    get_bridge,
    store_prompt_execution,
    recommend_prompt_for_task,
)

__all__ = [
    # Data classes
    'PromptExecution',
    'PromptStats',

    # Main classes
    'HoloLoomBridge',
    'ThompsonSamplerPrompts',

    # Convenience functions
    'create_bridge',
    'get_bridge',
    'store_prompt_execution',
    'recommend_prompt_for_task',
]
