"""
Demo: Elegant Beta Wave Context Packing

Shows how beta wave activation spreading replaces 506 lines of ad-hoc heuristics
with physics-based importance scoring.

Key Insight: "Activation IS Importance"
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.memory.spring_dynamics_engine import SpringDynamicsEngine, SpringEngineConfig
from hololoom.awareness.beta_wave_packer import BetaWaveContextPacker, TokenBudget


def print_section(title: str, char: str = "="):
    """Print section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


def create_test_memories():
    """Create test memory nodes about different topics"""
    memories = [
        # Core Thompson Sampling memories (should activate strongly)
        {
            'id': 'thompson_1',
            'content': 'Thompson Sampling is a Bayesian approach to the exploration-exploitation dilemma.',
            'embedding': np.random.randn(128),
            'k': 2.0  # Recently accessed
        },
        {
            'id': 'thompson_2',
            'content': 'Thompson Sampling maintains beta distributions for each bandit arm.',
            'embedding': np.random.randn(128),
            'k': 1.8
        },
        {
            'id': 'thompson_3',
            'content': 'The algorithm samples from posterior distributions to balance exploration and exploitation.',
            'embedding': np.random.randn(128),
            'k': 1.5
        },

        # Related reinforcement learning (should activate moderately)
        {
            'id': 'rl_1',
            'content': 'Multi-armed bandits are a class of reinforcement learning problems.',
            'embedding': np.random.randn(128),
            'k': 1.2
        },
        {
            'id': 'rl_2',
            'content': 'Upper Confidence Bound (UCB) is another exploration strategy.',
            'embedding': np.random.randn(128),
            'k': 1.0
        },

        # Distant concepts (should activate weakly or not at all)
        {
            'id': 'distant_1',
            'content': 'Neural networks use backpropagation to learn from data.',
            'embedding': np.random.randn(128),
            'k': 0.8
        },
        {
            'id': 'distant_2',
            'content': 'Quantum computing uses qubits for parallel computation.',
            'embedding': np.random.randn(128),
            'k': 0.5
        },

        # Old forgotten memory (low k)
        {
            'id': 'forgotten_1',
            'content': 'Bayesian inference updates beliefs based on evidence.',
            'embedding': np.random.randn(128),
            'k': 0.3  # Faded memory
        }
    ]

    return memories


async def demo_1_basic_packing():
    """Demo 1: Basic beta wave context packing"""
    print_section("Demo 1: Basic Beta Wave Context Packing")

    # 1. Create spring dynamics engine
    config = SpringEngineConfig(damping=0.95, dt=0.1)
    engine = SpringDynamicsEngine(config)

    # 2. Encode test memories
    memories = create_test_memories()
    for mem in memories:
        engine.encode_new_memory(
            node_id=mem['id'],
            content=mem['content'],
            embedding=mem['embedding'],
            initial_k=mem['k']
        )

    print(f"Encoded {len(memories)} memories into spring dynamics engine\n")

    # 3. Create query
    query_text = "How does Thompson Sampling work?"
    query_embedding = np.random.randn(128)  # Would be actual embedding in production

    # 4. Create beta wave context packer
    budget = TokenBudget(total=2000, reserved_for_query=200, reserved_for_response=500)
    packer = BetaWaveContextPacker(
        spring_engine=engine,
        token_budget=budget,
        activation_threshold=0.3,  # Exclude below 0.3
        compression_threshold=0.7  # Compress below 0.7
    )

    # 5. Pack context (SINGLE CALL - physics does all the work)
    packed = await packer.pack_context(
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=20
    )

    # 6. Display results
    print(f"Query: {query_text}\n")

    print("📊 Packing Statistics:")
    print(f"  Total tokens: {packed.total_tokens}/{budget.available_for_context}")
    print(f"  Elements: {packed.elements_included} included, "
          f"{packed.elements_compressed} compressed, "
          f"{packed.elements_excluded} excluded")
    print(f"  Activation: avg={packed.avg_activation:.3f}, "
          f"min={packed.min_activation:.3f}, "
          f"max={packed.max_activation:.3f}")
    print(f"  Packing time: {packed.packing_time_ms:.2f}ms")

    print("\n🧠 Beta Wave Statistics:")
    stats = packed.activation_stats
    print(f"  Beta wave iterations: {stats['beta_wave_iterations']}")
    print(f"  Seed nodes: {stats['seed_nodes']}")
    print(f"  Total activated: {stats['total_activated']}")
    print(f"  Creative insights: {stats['creative_insights']}")

    dist = stats['activation_distribution']
    print(f"\n📈 Activation Distribution:")
    print(f"  High (>0.7):    {dist.get('high (>0.7)', 0)} elements")
    print(f"  Medium (0.3-0.7): {dist.get('medium (0.3-0.7)', 0)} elements")
    print(f"  Low (<0.3):     {dist.get('low (<0.3)', 0)} elements")

    print(f"\n📝 Formatted Context (first 500 chars):")
    print("-" * 80)
    formatted = packed.format_for_llm(include_metadata=True)
    print(formatted[:500] + "...")
    print("-" * 80)


async def demo_2_budget_constraints():
    """Demo 2: How physics-based packing handles tight budgets"""
    print_section("Demo 2: Budget Constraints - Physics Adapts")

    # Setup
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    memories = create_test_memories()
    for mem in memories:
        engine.encode_new_memory(mem['id'], mem['content'], mem['embedding'], mem['k'])

    query_text = "Explain Thompson Sampling and exploration strategies"
    query_embedding = np.random.randn(128)

    # Test with different budgets
    budgets = [
        ("Tight", TokenBudget(total=1000, reserved_for_query=100, reserved_for_response=300)),
        ("Moderate", TokenBudget(total=2000, reserved_for_query=200, reserved_for_response=500)),
        ("Generous", TokenBudget(total=4000, reserved_for_query=300, reserved_for_response=800))
    ]

    for budget_name, budget_config in budgets:
        packer = BetaWaveContextPacker(engine, budget_config)
        packed = await packer.pack_context(query_text, query_embedding)

        print(f"\n🎯 {budget_name} Budget ({budget_config.total} tokens):")
        print(f"  Available for context: {budget_config.available_for_context}")
        print(f"  Actual usage: {packed.total_tokens}")
        print(f"  Elements: {packed.elements_included} included, "
              f"{packed.elements_compressed} compressed")
        print(f"  Avg activation: {packed.avg_activation:.3f}")


async def demo_3_activation_vs_heuristics():
    """Demo 3: Compare activation-based vs heuristic-based importance"""
    print_section("Demo 3: Physics vs Heuristics")

    # Setup
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    memories = create_test_memories()
    for mem in memories:
        engine.encode_new_memory(mem['id'], mem['content'], mem['embedding'], mem['k'])

    query_text = "What is Thompson Sampling?"
    query_embedding = np.random.randn(128)

    # Run beta wave retrieval
    result = engine.retrieve_memories(query_embedding, top_k=10)

    print("Activation-Based Importance (Physics):")
    print("-" * 80)
    for i, (node_id, activation) in enumerate(result.recalled_memories[:5], 1):
        node = engine.nodes[node_id]
        print(f"{i}. [{activation:.3f}] {node.content[:60]}...")
        print(f"   Spring k: {node.spring_constant:.2f} (freshness)")
        print()

    print("\nWhy This Works:")
    print("  ✓ Activation level = natural relevance (from semantic spreading)")
    print("  ✓ Spring constant k = recency (fresh memories conduct better)")
    print("  ✓ No magic numbers (no boost *= 1.2, boost *= 1.15, etc.)")
    print("  ✓ Physics handles complexity, packer just uses the results")

    print("\nOld Heuristic Approach (506 lines):")
    print("  ✗ Manual importance: if uncertainty > 0.7: boost *= 1.2")
    print("  ✗ Position decay: importance = 0.8 - (position * 0.05)")
    print("  ✗ Domain matching: if domain in content: boost *= 1.15")
    print("  ✗ Three separate packing passes (critical, high, medium/low)")
    print("  ✗ Brittle thresholds and magic numbers")


async def demo_4_creative_insights():
    """Demo 4: Cross-domain creative insights"""
    print_section("Demo 4: Creative Insights - Cross-Domain Bridges")

    # Setup with diverse domains
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    # Add memories from different domains
    diverse_memories = [
        {'id': 'ml_1', 'content': 'Thompson Sampling balances exploration and exploitation.',
         'embedding': np.array([1.0] + [0.0]*127), 'k': 2.0},
        {'id': 'bio_1', 'content': 'Foraging animals balance exploring new areas vs exploiting known food sources.',
         'embedding': np.array([0.8, 0.3] + [0.0]*126), 'k': 1.5},
        {'id': 'econ_1', 'content': 'Investors balance diversifying portfolios vs focusing on proven assets.',
         'embedding': np.array([0.7, 0.4] + [0.0]*126), 'k': 1.2},
        {'id': 'game_1', 'content': 'Game AI uses epsilon-greedy strategies to balance trying new moves.',
         'embedding': np.array([0.9, 0.2] + [0.0]*126), 'k': 1.0},
    ]

    for mem in diverse_memories:
        engine.encode_new_memory(mem['id'], mem['content'], mem['embedding'], mem['k'])

    query_text = "How do systems balance exploration and exploitation?"
    query_embedding = np.array([1.0] + [0.0]*127)

    # Pack with beta wave packer
    packer = BetaWaveContextPacker(engine, TokenBudget(total=3000))
    packed = await packer.pack_context(query_text, query_embedding)

    print(f"Query: {query_text}\n")
    print(f"Creative Insights Found: {packed.activation_stats['creative_insights']}")

    if packed.creative_section:
        print("\n🌟 Cross-Domain Bridges:")
        print("-" * 80)
        print(packed.creative_section)
        print("-" * 80)

    print("\nWhy This Works:")
    print("  ✓ Beta wave spreading activates semantically related nodes")
    print("  ✓ Creative insights = distant nodes that still activated")
    print("  ✓ Finds cross-domain analogies automatically")
    print("  ✓ No manual domain matching or analogy detection needed")


async def demo_5_comparison():
    """Demo 5: Side-by-side comparison with old approach"""
    print_section("Demo 5: Elegance Comparison")

    print("OLD APPROACH (SmartContextPacker):")
    print("-" * 80)
    print("Lines of code: 506")
    print("Passes: 3 (critical → high → medium/low)")
    print("Importance calculation: Ad-hoc heuristics")
    print("  - Manual boosts: *= 1.2, *= 1.15, *= 1.1")
    print("  - Position decay: 0.8 - (i * 0.05)")
    print("  - Domain matching: string matching")
    print("Compression: Manual (requires writing summary/detailed versions)")
    print("Complexity: O(3n) with brittle thresholds")
    print()

    print("NEW APPROACH (BetaWaveContextPacker):")
    print("-" * 80)
    print("Lines of code: ~350 (30% reduction)")
    print("Passes: 1 (single pass)")
    print("Importance calculation: Physics-based")
    print("  - Activation spreading determines relevance")
    print("  - Spring constant k encodes recency")
    print("  - Creative insights for cross-domain")
    print("Compression: Automatic (activation threshold)")
    print("Complexity: O(n) with natural thresholds")
    print()

    print("KEY INSIGHT:")
    print("  'Activation IS Importance'")
    print()
    print("  Instead of manually calculating importance,")
    print("  just trust the springs. Physics does the hard work.")


async def main():
    """Run all demos"""
    print("\n" + "🧲" * 40)
    print("BETA WAVE CONTEXT PACKING - ELEGANT SOLUTION".center(80))
    print("Physics Replaces Heuristics".center(80))
    print("🧲" * 40)

    demos = [
        ("Basic Packing", demo_1_basic_packing),
        ("Budget Constraints", demo_2_budget_constraints),
        ("Activation vs Heuristics", demo_3_activation_vs_heuristics),
        ("Creative Insights", demo_4_creative_insights),
        ("Elegance Comparison", demo_5_comparison)
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
            print(f"\n✅ {name} completed")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("🎯 Beta Wave Context Packing Complete!".center(80))
    print("Activation IS Importance".center(80))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
