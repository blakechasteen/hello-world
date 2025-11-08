"""
Week 3 Demo: LLM Integration for Semantic Fact Extraction
===========================================================

This demo shows how to use production LLM integration for high-quality
semantic fact extraction from episodic memories.

Features demonstrated:
- Multi-provider support (OpenAI, Anthropic, Ollama, vLLM)
- Cost tracking
- Graceful fallback
- Integration with Week 2 consolidation
"""

import asyncio
import os
from datetime import datetime

# Week 3 imports
from HoloLoom.memory.llm_consolidator import (
    create_production_consolidator,
    create_llm_config,
    calculate_cost
)

# Week 2 imports
from HoloLoom.memory.consolidation import MemoryConsolidator, ConsolidationStrategy
from HoloLoom.memory.lifecycle_manager import ContextStreamManager, MemoryScope

# Week 1 imports
from HoloLoom.documentation.types import MemoryShard


# ============================================================================
# Sample Data
# ============================================================================

def create_sample_episodes():
    """Create sample episodic memories for demonstration."""
    return [
        MemoryShard(
            id="ep1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It balances exploration and exploitation by sampling from posterior distributions.",
            entities=["ThompsonSampling", "Bayesian", "MultiArmedBandit"],
            metadata={"timestamp": datetime.now().isoformat(), "type": "episode"}
        ),
        MemoryShard(
            id="ep2",
            text="Thompson Sampling uses Beta distributions to model arm probabilities. "
                 "When an arm succeeds, we update alpha. When it fails, we update beta.",
            entities=["ThompsonSampling", "BetaDistribution"],
            metadata={"timestamp": datetime.now().isoformat(), "type": "episode"}
        ),
        MemoryShard(
            id="ep3",
            text="Compared to epsilon-greedy, Thompson Sampling naturally balances exploration. "
                 "It explores more when uncertain and exploits when confident.",
            entities=["ThompsonSampling", "EpsilonGreedy"],
            metadata={"timestamp": datetime.now().isoformat(), "type": "episode"}
        ),
        MemoryShard(
            id="ep4",
            text="HoloLoom uses Thompson Sampling in its policy engine. "
                 "The bandit selects which tool to use based on historical success rates.",
            entities=["HoloLoom", "ThompsonSampling", "PolicyEngine"],
            metadata={"timestamp": datetime.now().isoformat(), "type": "episode"}
        ),
        MemoryShard(
            id="ep5",
            text="The key advantage of Thompson Sampling is probability matching. "
                 "It explores each arm proportional to its probability of being optimal.",
            entities=["ThompsonSampling"],
            metadata={"timestamp": datetime.now().isoformat(), "type": "episode"}
        )
    ]


# ============================================================================
# Demo 1: Rule-Based Fallback (No LLM)
# ============================================================================

async def demo_rule_based():
    """Demo 1: Rule-based fact extraction (no LLM required)."""
    print("\n" + "="*70)
    print("DEMO 1: Rule-Based Fact Extraction (No LLM)")
    print("="*70)

    # Create consolidator without LLM
    consolidator = create_production_consolidator(provider="none")

    episodes = create_sample_episodes()

    # Extract facts (rule-based)
    print(f"\nExtracting facts from {len(episodes)} episodes...")
    facts = await consolidator.extract_facts(episodes)

    print(f"\nExtracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")

    # Extract entities (rule-based)
    print(f"\nExtracting entity relationships...")
    edges = await consolidator.extract_entities(episodes)

    print(f"\nExtracted {len(edges)} relationships:")
    for src, dst, rel in edges[:10]:  # Show first 10
        print(f"  {src} - {rel} - {dst}")

    # Statistics
    stats = consolidator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Provider: {stats['provider']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")


# ============================================================================
# Demo 2: OpenAI Integration (with Mock)
# ============================================================================

async def demo_openai_mock():
    """Demo 2: OpenAI-style extraction (mocked for demo)."""
    print("\n" + "="*70)
    print("DEMO 2: OpenAI Integration (Mocked)")
    print("="*70)

    # Note: This would normally use real OpenAI API
    # For demo, we'll show how it would work with real credentials

    print("\nTo use real OpenAI:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  consolidator = create_production_consolidator(")
    print("      provider='openai',")
    print("      model='gpt-3.5-turbo'")
    print("  )")

    # Create config to show pricing
    config = create_llm_config(provider="openai", model="gpt-3.5-turbo")
    print(f"\nOpenAI Config:")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max tokens: {config.max_tokens}")

    # Show cost calculation
    print("\nCost Calculation (GPT-3.5-turbo):")
    cost = calculate_cost("gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=500)
    print(f"  1000 prompt + 500 completion tokens = ${cost:.4f}")

    print("\nWith 24 consolidations per day:")
    daily_cost = cost * 24
    monthly_cost = daily_cost * 30
    print(f"  Daily: ${daily_cost:.2f}")
    print(f"  Monthly: ${monthly_cost:.2f}")


# ============================================================================
# Demo 3: Anthropic Integration
# ============================================================================

async def demo_anthropic():
    """Demo 3: Anthropic Claude integration (conceptual)."""
    print("\n" + "="*70)
    print("DEMO 3: Anthropic Claude Integration (Conceptual)")
    print("="*70)

    print("\nTo use Anthropic Claude:")
    print("  export ANTHROPIC_API_KEY='sk-ant-...'")
    print("  consolidator = create_production_consolidator(")
    print("      provider='anthropic',")
    print("      model='claude-3-haiku-20240307'  # Fast, cheap")
    print("  )")

    # Show pricing comparison
    print("\nPricing Comparison (per 1M tokens):")
    models = [
        ("gpt-3.5-turbo", 0.50, 1.50),
        ("gpt-4-turbo", 10.00, 30.00),
        ("claude-3-haiku", 0.25, 1.25),
        ("claude-3-sonnet", 3.00, 15.00),
        ("claude-3-opus", 15.00, 75.00)
    ]

    for model, prompt_price, completion_price in models:
        # Calculate for typical consolidation (1000 prompt, 500 completion)
        cost = (1000/1_000_000) * prompt_price + (500/1_000_000) * completion_price
        print(f"  {model:20} ${cost:.4f}")

    print("\n  → claude-3-haiku is cheapest (5x cheaper than GPT-3.5)")


# ============================================================================
# Demo 4: Local Models (Ollama - Free)
# ============================================================================

async def demo_ollama():
    """Demo 4: Ollama local models (free, privacy-focused)."""
    print("\n" + "="*70)
    print("DEMO 4: Ollama Local Models (Free)")
    print("="*70)

    print("\nSetup:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Pull model: ollama pull llama2")
    print("  3. Start server: ollama serve")

    print("\nUsage:")
    print("  consolidator = create_production_consolidator(")
    print("      provider='ollama',")
    print("      model='llama2'")
    print("  )")

    print("\nPros:")
    print("  ✓ Free (no API costs)")
    print("  ✓ Privacy (runs locally)")
    print("  ✓ No internet required")

    print("\nCons:")
    print("  ✗ Slower than API (~2-3x)")
    print("  ✗ Requires local setup")
    print("  ✗ Lower quality than GPT-4")


# ============================================================================
# Demo 5: Integration with Week 2 (Background Consolidation)
# ============================================================================

async def demo_week2_integration():
    """Demo 5: Week 3 LLM integration with Week 2 consolidation."""
    print("\n" + "="*70)
    print("DEMO 5: Week 2 + Week 3 Integration")
    print("="*70)

    # Create multi-level memory manager (Week 1)
    stream_manager = ContextStreamManager()

    # Add episodes to SESSION scope
    episodes = create_sample_episodes()
    print(f"\nAdding {len(episodes)} episodes to SESSION scope...")
    for ep in episodes:
        await stream_manager.add_to_stream("session_state", ep)

    # Create consolidator with LLM (Week 3)
    print("\nCreating consolidator with LLM (rule-based fallback)...")
    consolidator = MemoryConsolidator(
        stream_manager=stream_manager,
        llm_provider=None,  # Use None for demo (no API keys required)
        consolidation_interval_minutes=60
    )

    # Consolidate (extract facts to AGENT scope)
    print("\nConsolidating episodes → semantic facts...")
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    print(f"\nConsolidation Result:")
    print(f"  Input episodes: {result.input_episodes}")
    print(f"  Output facts: {result.output_facts}")
    print(f"  Facts stored: {len(result.facts_stored)}")
    print(f"  Time: {result.consolidation_time_ms:.1f}ms")

    # Check facts in AGENT scope
    agent_memories = stream_manager.get_all_memories(scopes=[MemoryScope.AGENT])
    semantic_facts = [m for m in agent_memories if m.metadata.get("type") == "semantic_fact"]

    print(f"\nSemantic facts in AGENT scope: {len(semantic_facts)}")
    for i, fact in enumerate(semantic_facts, 1):
        print(f"  {i}. {fact.text[:80]}...")

    # Get statistics (includes LLM usage)
    stats = consolidator.get_statistics()
    print(f"\nConsolidation Statistics:")
    print(f"  Total consolidations: {stats['total_consolidations']}")
    print(f"  Total facts extracted: {stats['total_facts_extracted']}")
    print(f"  LLM provider: {stats['llm_provider'] or 'none'}")
    print(f"  LLM cost: ${stats['llm_usage']['total_cost_usd']:.4f}")

    print("\n✓ Demo 5 complete (Week 2 + Week 3 integration working)")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def main():
    """Run all Week 3 demos."""
    print("\n" + "="*70)
    print("WEEK 3: LLM INTEGRATION FOR SEMANTIC FACT EXTRACTION")
    print("="*70)
    print("\nThis demo shows how to use production LLM integration for")
    print("high-quality semantic fact extraction from episodic memories.")

    # Run demos
    await demo_rule_based()
    await demo_openai_mock()
    await demo_anthropic()
    await demo_ollama()
    await demo_week2_integration()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nWeek 3 provides:")
    print("  ✓ Multi-provider LLM support (OpenAI, Anthropic, Ollama, vLLM)")
    print("  ✓ Cost tracking and optimization")
    print("  ✓ Graceful fallback to rule-based extraction")
    print("  ✓ Seamless integration with Week 2 consolidation")
    print("  ✓ Production-ready error handling")

    print("\nNext: Week 4 - Hybrid Retrieval")
    print("  • Semantic search (sentence-transformers)")
    print("  • BM25 keyword search")
    print("  • Graph traversal")
    print("  • Reciprocal Rank Fusion")

    print("\n" + "="*70)
    print("\nDemo complete! 🎉")


if __name__ == "__main__":
    asyncio.run(main())
