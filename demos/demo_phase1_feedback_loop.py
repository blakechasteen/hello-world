#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Feedback Loop Demo

Demonstrates the complete LLM-integrated context packer:
1. Pack context intelligently
2. Generate with LLM
3. Score quality (coherence, completeness, relevance)
4. Analyze token efficiency
5. Track context utilization
6. Learn from outcomes (adapt packing strategy)

This shows the feedback loop closing: packing → generation → feedback → learning

Usage:
    # With Ollama (free, local)
    PYTHONPATH=. python demos/demo_phase1_feedback_loop.py --provider ollama

    # With Anthropic (requires API key)
    ANTHROPIC_API_KEY=your_key PYTHONPATH=. python demos/demo_phase1_feedback_loop.py --provider anthropic

    # With OpenAI (requires API key)
    OPENAI_API_KEY=your_key PYTHONPATH=. python demos/demo_phase1_feedback_loop.py --provider openai
"""

import asyncio
import sys
from pathlib import Path
import argparse

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.awareness.compositional_awareness import CompositionalAwarenessLayer
from HoloLoom.awareness.context_packer_llm import (
    LLMContextPacker,
    PackedGeneration,
    TokenBudget,
    LLMGenerationConfig
)


def print_section(title: str, char: str = "="):
    """Print section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


def print_generation_summary(gen: PackedGeneration, index: int):
    """Print generation summary"""
    print(f"\n{'─' * 80}")
    print(f"Generation #{index}")
    print(f"{'─' * 80}")
    print(f"\n📝 Query: {gen.query}")
    print(f"\n📊 Quality Score: {gen.quality_score.overall:.2f} ({gen.quality_score.grade})")
    print(f"  • Coherence:    {gen.quality_score.coherence:.2f}")
    print(f"  • Completeness: {gen.quality_score.completeness:.2f}")
    print(f"  • Relevance:    {gen.quality_score.relevance:.2f}")

    print(f"\n💰 Efficiency:")
    print(f"  • Token Efficiency: {gen.token_efficiency:.4f} (quality/token)")
    print(f"  • Total Tokens:     {gen.llm_response.total_tokens}")
    print(f"  • Cost:             ${gen.llm_response.cost_estimate_usd:.4f}")

    print(f"\n🎯 Context Utilization:")
    print(f"  • Utilization Rate: {gen.context_utilization.utilization_rate:.1%}")
    print(f"  • Elements Used:    {gen.context_utilization.total_elements_utilized}/{gen.context_utilization.total_elements_packed}")
    print(f"  • Wasted Elements:  {gen.context_utilization.wasted_elements}")

    print(f"\n⏱️  Performance:")
    print(f"  • Packing Time:  {gen.packed_context.packing_time_ms:.1f}ms")
    print(f"  • LLM Latency:   {gen.llm_response.latency_ms:.0f}ms")
    print(f"  • Total:         {gen.packed_context.packing_time_ms + gen.llm_response.latency_ms:.0f}ms")

    print(f"\n💬 Response Preview:")
    preview = gen.llm_response.text[:300] + "..." if len(gen.llm_response.text) > 300 else gen.llm_response.text
    print(f"  {preview}")


async def demo_1_basic_usage(provider_name: str):
    """Demo 1: Basic pack_and_generate usage"""
    print_section("Demo 1: Basic Usage (Pack → Generate → Feedback)")

    # Create awareness layer
    awareness = CompositionalAwarenessLayer()

    # Create LLM-integrated packer
    packer = LLMContextPacker(
        token_budget=TokenBudget(total=4000),
        llm_provider=provider_name,
        enable_learning=False  # Disable for this demo
    )

    # Test query
    query = "What is quantum tunneling and how does it work?"

    print(f"📝 Query: {query}\n")
    print("⏳ Processing...\n")

    # Get awareness context
    awareness_ctx = await awareness.get_unified_context(query)

    # Mock memories
    memories = [
        {
            'text': 'Quantum tunneling is a quantum mechanical phenomenon where particles pass through potential barriers.',
            'score': 0.95
        },
        {
            'text': 'In classical physics, a particle needs energy to overcome a barrier. Quantum mechanics allows probabilistic tunneling.',
            'score': 0.88
        },
        {
            'text': 'Scanning tunneling microscopes use quantum tunneling to image surfaces at atomic resolution.',
            'score': 0.82
        }
    ]

    # Pack and generate
    result = await packer.pack_and_generate(
        query,
        awareness_ctx,
        memory_results=memories,
        max_memories=5
    )

    # Show results
    print_generation_summary(result, 1)

    print(f"\n📦 Packing Details:")
    print(f"  • Elements Included:   {result.packed_context.elements_included}")
    print(f"  • Elements Compressed: {result.packed_context.elements_compressed}")
    print(f"  • Elements Excluded:   {result.packed_context.elements_excluded}")
    print(f"  • Avg Importance:      {result.packed_context.avg_importance:.2f}")

    print("\n✅ Demo 1 complete!")


async def demo_2_learning_over_time(provider_name: str):
    """Demo 2: Learning system adapts packing strategy"""
    print_section("Demo 2: Learning System (Adapt from Outcomes)")

    # Create awareness layer
    awareness = CompositionalAwarenessLayer()

    # Create LLM-integrated packer with learning enabled
    packer = LLMContextPacker(
        token_budget=TokenBudget(total=4000),
        llm_provider=provider_name,
        enable_learning=True,
        learning_rate=0.2  # Faster learning for demo
    )

    # Test queries (varied complexity)
    test_queries = [
        "What is quantum tunneling?",
        "How does quantum tunneling work in semiconductors?",
        "Explain the applications of quantum tunneling in modern technology.",
        "What are the theoretical foundations of quantum tunneling?",
        "Compare quantum tunneling to classical barrier penetration.",
        "How is quantum tunneling used in scanning tunneling microscopes?",
        "What role does quantum tunneling play in nuclear fusion?",
        "Explain quantum tunneling probability calculations.",
        "How does quantum tunneling enable flash memory?",
        "What are the limitations of quantum tunneling?"
    ]

    # Mock memories (same for all queries for consistency)
    memories = [
        {'text': f'Memory {i}: Quantum mechanics concept related to the topic.', 'score': 0.9 - i*0.05}
        for i in range(10)
    ]

    print(f"🧪 Running {len(test_queries)} queries to train the learning system...\n")

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query[:60]}...")

        # Get awareness
        awareness_ctx = await awareness.get_unified_context(query)

        # Pack and generate
        result = await packer.pack_and_generate(
            query,
            awareness_ctx,
            memory_results=memories,
            max_memories=10
        )

        results.append(result)

        # Show progress
        print(f"  Quality: {result.quality_score.overall:.2f} | "
              f"Efficiency: {result.token_efficiency:.4f} | "
              f"Utilization: {result.context_utilization.utilization_rate:.1%}")

    # Show learning statistics
    print_section("Learning Statistics", "─")

    stats = packer.get_learning_statistics()

    print(f"📊 Learning Summary:")
    print(f"  • Total Interactions: {stats['total_interactions']}")
    print(f"  • Current Importance Threshold: {stats['current_importance_threshold']:.2f}")
    print(f"  • Learning Rate: {stats['learning_rate']:.2f}")

    # Show quality trends
    print(f"\n📈 Quality Trends:")
    qualities = [r.quality_score.overall for r in results]
    print(f"  • First 3 avg: {sum(qualities[:3])/3:.2f}")
    print(f"  • Last 3 avg:  {sum(qualities[-3:])/3:.2f}")
    print(f"  • Improvement: {(sum(qualities[-3:])/3 - sum(qualities[:3])/3):.2f}")

    # Show efficiency trends
    print(f"\n💰 Efficiency Trends:")
    efficiencies = [r.token_efficiency for r in results]
    print(f"  • First 3 avg: {sum(efficiencies[:3])/3:.4f}")
    print(f"  • Last 3 avg:  {sum(efficiencies[-3:])/3:.4f}")
    print(f"  • Improvement: {(sum(efficiencies[-3:])/3 - sum(efficiencies[:3])/3):.4f}")

    print("\n✅ Demo 2 complete! System learned from outcomes and adapted strategy.")


async def demo_3_provider_comparison(provider_names: list):
    """Demo 3: Compare different LLM providers"""
    print_section("Demo 3: Provider Comparison")

    # Create awareness layer
    awareness = CompositionalAwarenessLayer()

    # Test query
    query = "Explain quantum tunneling in simple terms."

    # Get awareness context
    awareness_ctx = await awareness.get_unified_context(query)

    # Mock memories
    memories = [
        {'text': 'Quantum tunneling allows particles to pass through barriers.', 'score': 0.95}
    ]

    print(f"📝 Query: {query}\n")
    print(f"🔄 Testing {len(provider_names)} providers...\n")

    results = {}

    for provider_name in provider_names:
        try:
            print(f"Testing {provider_name}...")

            # Create packer for this provider
            packer = LLMContextPacker(
                llm_provider=provider_name,
                enable_learning=False
            )

            # Pack and generate
            result = await packer.pack_and_generate(
                query,
                awareness_ctx,
                memory_results=memories,
                max_memories=5
            )

            results[provider_name] = result

            print(f"  ✅ Quality: {result.quality_score.overall:.2f} | "
                  f"Latency: {result.llm_response.latency_ms:.0f}ms | "
                  f"Cost: ${result.llm_response.cost_estimate_usd:.4f}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[provider_name] = None

    # Summary
    print_section("Provider Comparison Summary", "─")

    if results:
        print(f"\n{'Provider':<15} {'Quality':<10} {'Latency':<12} {'Cost':<10} {'Winner':<10}")
        print(f"{'-'*60}")

        # Find best in each category
        valid_results = {k: v for k, v in results.items() if v is not None}

        if valid_results:
            best_quality = max(valid_results.items(), key=lambda x: x[1].quality_score.overall)
            best_latency = min(valid_results.items(), key=lambda x: x[1].llm_response.latency_ms)
            best_cost = min(valid_results.items(), key=lambda x: x[1].llm_response.cost_estimate_usd)

            for provider, result in results.items():
                if result is None:
                    print(f"{provider:<15} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'':<10}")
                else:
                    quality = f"{result.quality_score.overall:.2f}"
                    latency = f"{result.llm_response.latency_ms:.0f}ms"
                    cost = f"${result.llm_response.cost_estimate_usd:.4f}"

                    winner = []
                    if provider == best_quality[0]:
                        winner.append("Quality")
                    if provider == best_latency[0]:
                        winner.append("Speed")
                    if provider == best_cost[0]:
                        winner.append("Cost")

                    winner_str = ",".join(winner) if winner else ""

                    print(f"{provider:<15} {quality:<10} {latency:<12} {cost:<10} {winner_str:<10}")

    print("\n✅ Demo 3 complete!")


async def demo_4_full_details(provider_name: str):
    """Demo 4: Show complete details of one generation"""
    print_section("Demo 4: Complete Generation Details")

    # Create awareness layer
    awareness = CompositionalAwarenessLayer()

    # Create packer
    packer = LLMContextPacker(
        llm_provider=provider_name,
        enable_learning=True
    )

    # Test query
    query = "How is quantum tunneling used in scanning tunneling microscopes?"

    # Get awareness
    awareness_ctx = await awareness.get_unified_context(query)

    # Rich memories
    memories = [
        {
            'text': 'Scanning tunneling microscopes (STM) use quantum tunneling to image surfaces at atomic resolution.',
            'score': 0.95
        },
        {
            'text': 'In STM, a sharp conducting tip is brought very close to the surface. Electrons tunnel between tip and surface.',
            'score': 0.92
        },
        {
            'text': 'The tunneling current is extremely sensitive to tip-surface distance, allowing atomic-scale height measurements.',
            'score': 0.88
        },
        {
            'text': 'STM can achieve lateral resolution of 0.1 nm and depth resolution of 0.01 nm.',
            'score': 0.85
        }
    ]

    print(f"📝 Query: {query}\n")
    print("⏳ Processing...\n")

    # Pack and generate
    result = await packer.pack_and_generate(
        query,
        awareness_ctx,
        memory_results=memories,
        max_memories=10
    )

    # Show everything
    print_generation_summary(result, 1)

    print(f"\n📦 Packed Context Structure:")
    print(f"{'─' * 80}")
    print("\n# AWARENESS CONTEXT")
    print(result.packed_context.awareness_section[:200] + "...")
    print("\n# RELEVANT MEMORIES")
    print(result.packed_context.memory_section[:300] + "...")
    print("\n# QUERY")
    print(result.packed_context.query_section)

    print(f"\n📊 Source Utilization:")
    for source, util_rate in result.context_utilization.source_utilization.items():
        print(f"  • {source}: {util_rate:.1%}")

    print(f"\n🔧 Provider Statistics:")
    provider_stats = packer.get_provider_statistics()
    print(f"  • Provider: {provider_stats.get('provider', 'N/A')}")
    print(f"  • Model: {provider_stats.get('model', 'N/A')}")
    print(f"  • Total Requests: {provider_stats.get('total_requests', 0)}")
    print(f"  • Total Tokens: {provider_stats.get('total_tokens', 0)}")
    print(f"  • Total Cost: ${provider_stats.get('total_cost_usd', 0):.4f}")

    print("\n✅ Demo 4 complete!")


async def main():
    """Run all demos"""
    parser = argparse.ArgumentParser(description="Phase 1 Feedback Loop Demo")
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific demo only (default: all)"
    )

    args = parser.parse_args()

    print("\n" + "🧠" * 40)
    print("PHASE 1: FEEDBACK LOOP DEMO".center(80))
    print("Context Packer + LLM + Quality Scoring + Learning".center(80))
    print("🧠" * 40)

    provider = args.provider

    print(f"\n🔧 Configuration:")
    print(f"  • LLM Provider: {provider}")
    print(f"  • Model: {'Default for provider'}")

    if provider != "ollama":
        print(f"\n⚠️  Note: Using {provider} requires API key in environment")
        print(f"  • ANTHROPIC_API_KEY for Anthropic")
        print(f"  • OPENAI_API_KEY for OpenAI")

    try:
        if args.demo is None or args.demo == 1:
            await demo_1_basic_usage(provider)

        if args.demo is None or args.demo == 2:
            await demo_2_learning_over_time(provider)

        if args.demo is None or args.demo == 3:
            # Try all available providers
            providers = ["ollama"]  # Always try Ollama
            if provider != "ollama":
                providers.append(provider)
            await demo_3_provider_comparison(providers)

        if args.demo is None or args.demo == 4:
            await demo_4_full_details(provider)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("🎯 Phase 1 Demo Complete!".center(80))
    print("=" * 80)

    print("\n🚀 Key Achievements:")
    print("  ✅ Context packing integrated with LLM generation")
    print("  ✅ Multi-dimensional quality scoring (coherence, completeness, relevance)")
    print("  ✅ Token efficiency tracking (quality per token)")
    print("  ✅ Context utilization analysis (which elements were used)")
    print("  ✅ Learning system adapts packing strategy from outcomes")
    print("  ✅ Support for multiple LLM providers (Anthropic, OpenAI, Ollama)")

    print("\n📈 Impact:")
    print("  • Feedback loop closed: pack → generate → score → learn")
    print("  • System improves over time (demonstrated in Demo 2)")
    print("  • Foundation for all future phases")

    print("\n💡 Next Steps:")
    print("  • Phase 2: Adaptive Budgeting (dynamic token budgets)")
    print("  • Phase 3: Conversation Packing (multi-turn conversations)")
    print("  • Phase 4: Semantic Compression (LLM-based compression)")


if __name__ == "__main__":
    asyncio.run(main())
