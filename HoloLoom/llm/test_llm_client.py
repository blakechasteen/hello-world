#!/usr/bin/env python3
"""
LLM Client Test Script
======================

Test multi-model LLM support with various providers.

Usage:
    PYTHONPATH=. python HoloLoom/llm/test_llm_client.py

Created: 2025-01-20
"""

import asyncio
import logging
import os
from HoloLoom.llm import UnifiedLLMClient, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ollama():
    """Test Ollama (local, free)"""
    print("\n=== Test 1: Ollama (Local) ===")

    try:
        config = LLMConfig(provider="ollama", model="llama3.2:3b")
        client = UnifiedLLMClient(primary=config)

        response = await client.complete(
            "What is 2+2? Answer in one sentence.",
            max_tokens=50
        )

        print(f"✓ Response: {response.content}")
        print(f"  Model: {response.provider}/{response.model}")
        print(f"  Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

    except Exception as e:
        print(f"✗ Ollama test failed: {e}")


async def test_anthropic():
    """Test Anthropic Claude"""
    print("\n=== Test 2: Anthropic Claude ===")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not set. Skipping test.")
        print("  Set with: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    try:
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        client = UnifiedLLMClient(primary=config)

        response = await client.complete(
            "What is Thompson Sampling in one sentence?",
            max_tokens=100
        )

        print(f"✓ Response: {response.content}")
        print(f"  Model: {response.provider}/{response.model}")
        print(f"  Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

    except Exception as e:
        print(f"✗ Anthropic test failed: {e}")


async def test_openai():
    """Test OpenAI GPT"""
    print("\n=== Test 3: OpenAI GPT ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set. Skipping test.")
        print("  Set with: export OPENAI_API_KEY='sk-...'")
        return

    try:
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        client = UnifiedLLMClient(primary=config)

        response = await client.complete(
            "What is recursion in one sentence?",
            max_tokens=100
        )

        print(f"✓ Response: {response.content}")
        print(f"  Model: {response.provider}/{response.model}")
        print(f"  Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

    except Exception as e:
        print(f"✗ OpenAI test failed: {e}")


async def test_gemini():
    """Test Google Gemini"""
    print("\n=== Test 4: Google Gemini ===")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ GOOGLE_API_KEY not set. Skipping test.")
        print("  Set with: export GOOGLE_API_KEY='...'")
        return

    try:
        config = LLMConfig(provider="google", model="gemini-1.5-flash")
        client = UnifiedLLMClient(primary=config)

        response = await client.complete(
            "What is the capital of France?",
            max_tokens=50
        )

        print(f"✓ Response: {response.content}")
        print(f"  Model: {response.provider}/{response.model}")
        print(f"  Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

    except Exception as e:
        print(f"✗ Gemini test failed: {e}")


async def test_fallback():
    """Test automatic fallback"""
    print("\n=== Test 5: Automatic Fallback ===")

    try:
        # Primary: expensive model (may not be available)
        primary = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")

        # Fallback: free local model (always available)
        fallback = LLMConfig(provider="ollama", model="llama3.2:3b")

        client = UnifiedLLMClient(
            primary=primary,
            fallbacks=[fallback],
            enable_cost_tracking=True
        )

        response = await client.complete("Hello!")

        print(f"✓ Response: {response.content[:100]}...")
        print(f"  Used model: {response.provider}/{response.model}")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

        if response.provider == "ollama":
            print("  → Fell back to Ollama (Claude not available)")
        else:
            print("  → Used primary model (Claude)")

    except Exception as e:
        print(f"✗ Fallback test failed: {e}")


async def test_comparison():
    """Test model comparison"""
    print("\n=== Test 6: Model Comparison (A/B Testing) ===")

    try:
        # Use default client (tries Ollama first)
        client = UnifiedLLMClient.create_default()

        # Available models
        available = client.list_available_models()
        print(f"Available models: {available}")

        if len(available) < 2:
            print("⚠ Need at least 2 models for comparison. Skipping.")
            return

        # Compare first 2 available models
        models = available[:2]
        print(f"Comparing: {models}")

        results = await client.compare_models(
            prompt="What is 5+3? Answer with just the number.",
            models=models,
            max_tokens=10
        )

        for model, response in results.items():
            print(f"\n{model}:")
            print(f"  Response: {response.content}")
            print(f"  Tokens: {response.input_tokens} + {response.output_tokens}")
            print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

    except Exception as e:
        print(f"✗ Comparison test failed: {e}")


async def test_cost_tracking():
    """Test cost tracking"""
    print("\n=== Test 7: Cost Tracking ===")

    try:
        client = UnifiedLLMClient.create_default()

        # Run several queries
        queries = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?"
        ]

        for query in queries:
            await client.complete(query, max_tokens=20)

        # Get statistics
        stats = client.get_cost_statistics()

        print(f"✓ Statistics:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
        print(f"  Avg cost/call: ${stats['avg_cost_per_call']:.4f}")

        print(f"\n  By model:")
        for model, model_stats in stats['by_model'].items():
            print(f"    {model}: {model_stats['calls']} calls, "
                  f"${model_stats['total_cost_usd']:.4f}")

    except Exception as e:
        print(f"✗ Cost tracking test failed: {e}")


async def test_budget_limit():
    """Test budget limit"""
    print("\n=== Test 8: Budget Limit ===")

    try:
        # Set very low budget (will force cheap model)
        expensive = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        cheap = LLMConfig(provider="ollama", model="llama3.2:3b")

        client = UnifiedLLMClient(
            primary=expensive,
            fallbacks=[cheap],
            max_cost_per_query=0.001  # $0.001 limit (very low)
        )

        response = await client.complete("Hello!")

        print(f"✓ Response: {response.content[:100]}...")
        print(f"  Used model: {response.provider}/{response.model}")
        print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")

        if response.provider == "ollama":
            print("  → Budget limit enforced (used free model)")
        else:
            print("  → Primary model was within budget")

    except Exception as e:
        print(f"✗ Budget limit test failed: {e}")


async def main():
    """Run all tests"""
    print("======================")
    print("LLM Client Test Suite")
    print("======================")

    tests = [
        test_ollama,
        test_anthropic,
        test_openai,
        test_gemini,
        test_fallback,
        test_comparison,
        test_cost_tracking,
        test_budget_limit,
    ]

    for test in tests:
        try:
            await test()
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}", exc_info=True)

    print("\n======================")
    print("Tests Complete")
    print("======================")


if __name__ == "__main__":
    asyncio.run(main())
