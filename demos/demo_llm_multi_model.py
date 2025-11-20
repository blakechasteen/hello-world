#!/usr/bin/env python3
"""
Multi-Model LLM Demo
====================

Demonstrates the unified LLM client with multiple providers:
- Ollama (local, free)
- Anthropic Claude (paid)
- OpenAI GPT (paid)
- Google Gemini (paid)

Features shown:
1. Simple query with default config
2. Custom configuration with fallback
3. Model comparison (A/B testing)
4. Cost tracking
5. Budget limits

Usage:
    PYTHONPATH=. python demos/demo_llm_multi_model.py

Created: 2025-01-20
"""

import asyncio
import logging
import os
from HoloLoom.llm import UnifiedLLMClient, LLMConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


async def demo_1_simple_query():
    """Demo 1: Simple query with default config"""
    print_section("Demo 1: Simple Query (Default Config)")

    # Create client with defaults (Ollama first, Claude fallback if available)
    client = UnifiedLLMClient.create_default()

    # Simple query
    query = "What is Thompson Sampling? Answer in 2 sentences."
    print(f"Query: {query}\n")

    response = await client.complete(query, max_tokens=100)

    print(f"Response: {response.content}\n")
    print(f"Model: {response.provider}/{response.model}")
    print(f"Tokens: {response.input_tokens} input, {response.output_tokens} output")

    if response.cost_estimate:
        print(f"Cost: ${response.cost_estimate.total_cost_usd:.4f}")
        if response.cost_estimate.is_free:
            print("  → Free! (local model)")
    else:
        print("Cost: $0.00 (free)")


async def demo_2_custom_config():
    """Demo 2: Custom configuration with fallback"""
    print_section("Demo 2: Custom Config with Fallback")

    # Primary: Claude (if available)
    primary = LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )

    # Fallback: Ollama (always available)
    fallback = LLMConfig(
        provider="ollama",
        model="llama3.2:3b"
    )

    client = UnifiedLLMClient(
        primary=primary,
        fallbacks=[fallback],
        enable_cost_tracking=True
    )

    print("Config:")
    print(f"  Primary: {primary.full_name}")
    print(f"  Fallback: {fallback.full_name}")
    print()

    # Query
    query = "What is recursion? One sentence only."
    print(f"Query: {query}\n")

    response = await client.complete(query, max_tokens=50)

    print(f"Response: {response.content}\n")
    print(f"Used model: {response.provider}/{response.model}")

    if response.provider == "anthropic":
        print("  → Primary model (Claude)")
    else:
        print("  → Fallback model (Ollama)")

    if response.cost_estimate:
        print(f"Cost: ${response.cost_estimate.total_cost_usd:.4f}")


async def demo_3_model_comparison():
    """Demo 3: Model comparison (A/B testing)"""
    print_section("Demo 3: Model Comparison (A/B Testing)")

    client = UnifiedLLMClient.create_default()

    # Get available models
    available = client.list_available_models()
    print(f"Available models: {available}\n")

    if len(available) < 1:
        print("⚠ No models available. Install Ollama or set API keys.")
        return

    # Compare available models (up to 2)
    models_to_compare = available[:min(2, len(available))]
    print(f"Comparing: {models_to_compare}\n")

    query = "What is 5 + 3? Just the number."
    print(f"Query: {query}\n")

    results = await client.compare_models(
        prompt=query,
        models=models_to_compare,
        max_tokens=10
    )

    print("Results:")
    for model, response in results.items():
        print(f"\n  {model}:")
        print(f"    Response: {response.content.strip()}")
        print(f"    Tokens: {response.input_tokens} + {response.output_tokens}")
        if response.cost_estimate:
            print(f"    Cost: ${response.cost_estimate.total_cost_usd:.4f}")


async def demo_4_cost_tracking():
    """Demo 4: Cost tracking"""
    print_section("Demo 4: Cost Tracking")

    client = UnifiedLLMClient.create_default()

    print("Running 3 queries...\n")

    queries = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
        response = await client.complete(query, max_tokens=10)
        print(f"   → {response.content.strip()}\n")

    # Get statistics
    stats = client.get_cost_statistics()

    print("Cost Statistics:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Avg cost/call: ${stats['avg_cost_per_call']:.4f}")

    if stats['by_model']:
        print("\n  By model:")
        for model, model_stats in stats['by_model'].items():
            print(f"    {model}:")
            print(f"      Calls: {model_stats['calls']}")
            print(f"      Tokens: {model_stats['input_tokens']:,} + {model_stats['output_tokens']:,}")
            print(f"      Cost: ${model_stats['total_cost_usd']:.4f}")


async def demo_5_budget_limit():
    """Demo 5: Budget limits"""
    print_section("Demo 5: Budget Limits")

    # Expensive model
    expensive = LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )

    # Cheap model
    cheap = LLMConfig(
        provider="ollama",
        model="llama3.2:3b"
    )

    # Very low budget (forces cheap model)
    client = UnifiedLLMClient(
        primary=expensive,
        fallbacks=[cheap],
        max_cost_per_query=0.001  # $0.001 limit
    )

    print("Config:")
    print(f"  Primary: {expensive.full_name}")
    print(f"  Fallback: {cheap.full_name}")
    print(f"  Budget limit: $0.001 per query")
    print()

    query = "Hello, world!"
    print(f"Query: {query}\n")

    response = await client.complete(query, max_tokens=20)

    print(f"Response: {response.content}\n")
    print(f"Used model: {response.provider}/{response.model}")

    if response.provider == "ollama":
        print("  → Budget enforced! Used free model.")
    else:
        print("  → Primary model was within budget.")

    if response.cost_estimate:
        print(f"Cost: ${response.cost_estimate.total_cost_usd:.4f}")


async def demo_6_provider_availability():
    """Demo 6: Check provider availability"""
    print_section("Demo 6: Provider Availability")

    providers = [
        ("ollama", "llama3.2:3b"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("openai", "gpt-4o-mini"),
        ("google", "gemini-1.5-flash")
    ]

    print("Checking providers...\n")

    for provider, model in providers:
        try:
            config = LLMConfig(provider=provider, model=model)
            client = UnifiedLLMClient(primary=config)

            # Try to list available models (checks availability)
            available = client.list_available_models()

            if f"{provider}/{model}" in available:
                print(f"✓ {provider}/{model} - Available")

                # Show pricing
                from HoloLoom.llm.cost_tracker import ModelPricing
                pricing = ModelPricing.get_pricing(model)
                if pricing["input"] == 0:
                    print(f"  Cost: Free (local)")
                else:
                    print(f"  Cost: ${pricing['input']:.2f}/${pricing['output']:.2f} per 1M tokens")
            else:
                print(f"✗ {provider}/{model} - Not available")

                # Show why
                if provider == "ollama":
                    print(f"  → Install Ollama from https://ollama.ai")
                else:
                    print(f"  → Set {provider.upper()}_API_KEY environment variable")

        except Exception as e:
            print(f"✗ {provider}/{model} - Error: {e}")

        print()


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  Multi-Model LLM System Demo")
    print("  HoloLoom - January 2025")
    print("=" * 60)

    demos = [
        demo_1_simple_query,
        demo_2_custom_config,
        demo_3_model_comparison,
        demo_4_cost_tracking,
        demo_5_budget_limit,
        demo_6_provider_availability,
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            logger.error(f"Demo {demo.__name__} failed: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("  1. Set API keys for paid providers (optional):")
    print("     export ANTHROPIC_API_KEY='sk-ant-...'")
    print("     export OPENAI_API_KEY='sk-...'")
    print("     export GOOGLE_API_KEY='...'")
    print()
    print("  2. Try model comparison with multiple providers")
    print()
    print("  3. Integrate into dashboard:")
    print("     See HoloLoom/llm/DASHBOARD_INTEGRATION.md")
    print()
    print("  4. Read full documentation:")
    print("     See HoloLoom/llm/README.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
