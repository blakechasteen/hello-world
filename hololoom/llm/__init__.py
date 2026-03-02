"""
HoloLoom LLM Integration
=========================

Multi-model LLM support with unified interface, automatic fallback,
cost tracking, and A/B testing capabilities.

Supported Providers:
- Ollama (local models)
- Anthropic (Claude)
- OpenAI (GPT-4)
- Google (Gemini)

Quick Start:
    from hololoom.llm import UnifiedLLMClient, LLMConfig

    # Simple usage with defaults
    client = UnifiedLLMClient.create_default()
    response = await client.complete("What is Thompson Sampling?")

    # With custom configuration
    primary = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
    fallback = LLMConfig(provider="ollama", model="llama3.2:3b")
    client = UnifiedLLMClient(primary=primary, fallbacks=[fallback])

    # Compare models
    results = await client.compare_models(
        "Explain recursion",
        models=["ollama/llama3.2:3b", "anthropic/claude-3-5-sonnet"]
    )

Created: 2025-01-20
"""

from hololoom.llm.unified_client import (
    UnifiedLLMClient,
    LLMConfig,
    LLMResponse,
    LLMProvider,
)

from hololoom.llm.cost_tracker import CostTracker, ModelPricing

__all__ = [
    "UnifiedLLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    "CostTracker",
    "ModelPricing",
]
