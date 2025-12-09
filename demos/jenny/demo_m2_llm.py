"""
Jenny Demo: M2 - Async LLM Client Infrastructure
================================================

Demonstrates async LLM client with:
- httpx connection pooling
- Exponential backoff with jitter
- Provider abstraction

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M2)
"""

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from HoloLoom.visualization.jenny_llm_client import (
        LLMClientConfig,
        LLMResponse,
        LLMError,
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@register_demo(
    phase_id="m2",
    name="Async LLM Client",
    order=2,
    tags=["core", "llm"],
    description="httpx connection pooling, exponential backoff"
)
@demo_phase("M2: Async LLM Client Infrastructure", emoji="🔌")
async def demo_m2_async_llm_client() -> dict:
    """
    M2: Async LLM Clients - Connection pooling, exponential backoff.

    Returns:
        Demo results with config info
    """
    if not LLM_AVAILABLE:
        return {"status": "skipped", "reason": "LLM client not available"}

    results = {
        "configs_shown": [],
        "features": [],
    }

    # Show configuration presets
    print("  Configuration Presets:")
    print("  " + "-" * 50)

    # Fast config for development
    fast_config = LLMClientConfig.fast()
    print(f"\n  Fast (Development):")
    print(f"    Timeout: {fast_config.timeout_seconds}s")
    print(f"    Retries: {fast_config.max_retries}")
    print(f"    Max Connections: {fast_config.max_connections}")
    results["configs_shown"].append("fast")

    # Reliable config for production
    reliable_config = LLMClientConfig.reliable()
    print(f"\n  Reliable (Production):")
    print(f"    Timeout: {reliable_config.timeout_seconds}s")
    print(f"    Retries: {reliable_config.max_retries}")
    print(f"    Max Connections: {reliable_config.max_connections}")
    results["configs_shown"].append("reliable")
    print()

    # Key features
    print("  Key Features:")
    print("  " + "-" * 50)
    features = [
        "httpx AsyncClient with connection pooling",
        "Exponential backoff with jitter on retries",
        "Provider abstraction (Ollama, OpenAI, Anthropic)",
        "Graceful degradation when LLM unavailable",
        "Structured LLMResponse with latency tracking",
    ]
    for feature in features:
        print(f"    - {feature}")
        results["features"].append(feature)
    print()

    # Mock response structure
    print("  Response Structure:")
    print("  " + "-" * 50)
    mock_response = LLMResponse(
        content="Thompson Sampling balances exploration and exploitation.",
        model="llama3.2:3b",
        provider="ollama",
        latency_ms=245.5,
        tokens_used=42,
        finish_reason="stop",
        retries=0,
    )
    print(f"  LLMResponse:")
    print(f"    content: \"{mock_response.content[:50]}...\"")
    print(f"    model: {mock_response.model}")
    print(f"    latency_ms: {mock_response.latency_ms}")
    print(f"    tokens_used: {mock_response.tokens_used}")
    print(f"    retries: {mock_response.retries}")

    return {
        "status": "success",
        "configs": results["configs_shown"],
        "feature_count": len(results["features"]),
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m2_async_llm_client())
