"""
Complete Recursive Learning System Demo
========================================
Demonstrates all 5 phases of the recursive learning vision working together:

Phase 1: Scratchpad Integration (Provenance tracking)
Phase 2: Loop Engine Integration (Pattern learning)
Phase 3: Hot Pattern Feedback (Usage-based adaptation)
Phase 4: Advanced Refinement (Multiple strategies, quality tracking)
Phase 5: Full Learning Loop (Background learning, Thompson/policy updates)

This demo shows:
1. Basic query processing with provenance
2. Pattern learning from successful queries
3. Hot pattern detection and adaptive retrieval
4. Low-confidence refinement with strategy selection
5. Background learning and Thompson Sampling updates
6. Complete learning statistics
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config
from HoloLoom.recursive import FullLearningEngine


def create_demo_shards() -> list[MemoryShard]:
    """Create demonstration knowledge shards"""
    return [
        MemoryShard(
            id="thompson_sampling",
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                   "It maintains Beta distributions for each arm's success probability and samples "
                   "from these distributions to balance exploration and exploitation.",
            metadata={"domain": "reinforcement_learning", "topic": "bandits"}
        ),
        MemoryShard(
            id="attention_mechanism",
            content="Attention mechanisms in neural networks allow models to focus on relevant parts "
                   "of the input. The key innovation is the Query-Key-Value (QKV) formulation where "
                   "attention weights are computed as softmax(Q·K^T/√d)·V.",
            metadata={"domain": "deep_learning", "topic": "transformers"}
        ),
        MemoryShard(
            id="hololoom_architecture",
            content="HoloLoom implements a weaving orchestrator that coordinates multiple feature "
                   "extraction threads (motifs, embeddings, spectral) and uses a unified policy "
                   "engine for tool selection. It supports progressive complexity (LITE/FAST/FULL/RESEARCH).",
            metadata={"domain": "hololoom", "topic": "architecture"}
        ),
        MemoryShard(
            id="ppo_algorithm",
            content="Proximal Policy Optimization (PPO) is a policy gradient method that uses a "
                   "clipped surrogate objective to prevent large policy updates. It achieves stable "
                   "training by constraining the policy update to a trust region.",
            metadata={"domain": "reinforcement_learning", "topic": "policy_gradient"}
        ),
        MemoryShard(
            id="matryoshka_embeddings",
            content="Matryoshka embeddings are multi-scale representations where smaller dimensions "
                   "are nested within larger ones. This allows using different embedding sizes "
                   "(e.g., 96, 192, 384) without retraining, enabling efficient retrieval at multiple scales.",
            metadata={"domain": "embeddings", "topic": "multi_scale"}
        ),
    ]


async def demo_1_basic_processing():
    """Demo 1: Basic query processing with provenance (Phase 1)"""
    print("=" * 80)
    print("DEMO 1: Basic Query Processing with Provenance (Phase 1)")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=False  # Disable for this demo
    ) as engine:

        # Process a simple query
        query = Query(text="What is Thompson Sampling?")
        spacetime = await engine.weave(query, enable_refinement=False)

        print(f"Query: {query.text}")
        print(f"Tool: {spacetime.trace.tool_selected}")
        print(f"Confidence: {spacetime.trace.tool_confidence:.2f}")
        print(f"Threads: {len(spacetime.trace.threads_activated)}")
        print()

        # Show scratchpad history
        print("Scratchpad History:")
        print("-" * 80)
        print(engine.get_scratchpad_history())
        print()


async def demo_2_pattern_learning():
    """Demo 2: Pattern learning from successful queries (Phase 2)"""
    print("=" * 80)
    print("DEMO 2: Pattern Learning from Successful Queries (Phase 2)")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=False
    ) as engine:

        # Process multiple related queries to learn patterns
        queries = [
            "What is Thompson Sampling?",
            "Explain Thompson Sampling for bandits",
            "How does Thompson Sampling balance exploration and exploitation?",
        ]

        print("Processing queries to learn patterns...")
        print()

        for i, query_text in enumerate(queries, 1):
            query = Query(text=query_text)
            spacetime = await engine.weave(query, enable_refinement=False)

            print(f"{i}. Query: {query_text}")
            print(f"   Confidence: {spacetime.trace.tool_confidence:.2f}")
            print(f"   Motifs: {spacetime.trace.motifs_detected[:3]}")
            print()

        # Show learned patterns
        stats = engine.get_learning_statistics()
        print("Learned Patterns:")
        print("-" * 80)

        if stats.get('learned_patterns'):
            for i, pattern in enumerate(stats['learned_patterns'][:5], 1):
                print(f"{i}. Motifs: {', '.join(pattern['motifs'])}")
                print(f"   Tool: {pattern['tool']}, Type: {pattern['query_type']}")
                print(f"   Occurrences: {pattern['occurrences']}, "
                      f"Confidence: {pattern['avg_confidence']:.2f}")
                print()
        else:
            print("(No patterns learned yet - need high-confidence results)")
            print()


async def demo_3_hot_patterns():
    """Demo 3: Hot pattern detection and adaptive retrieval (Phase 3)"""
    print("=" * 80)
    print("DEMO 3: Hot Pattern Detection and Adaptive Retrieval (Phase 3)")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=False
    ) as engine:

        # Process same topic multiple times to create hot patterns
        print("Accessing 'attention' topic multiple times...")
        print()

        for i in range(5):
            query = Query(text=f"Explain attention mechanisms (query {i+1})")
            spacetime = await engine.weave(query, enable_refinement=False)
            print(f"{i+1}. Confidence: {spacetime.trace.tool_confidence:.2f}")

        print()

        # Show hot patterns
        stats = engine.get_learning_statistics()
        print("Hot Patterns (Most Accessed):")
        print("-" * 80)

        if stats.get('hot_patterns'):
            for i, pattern in enumerate(stats['hot_patterns'][:5], 1):
                print(f"{i}. {pattern['element_id']}")
                print(f"   Heat Score: {pattern['heat_score']:.1f}")
                print(f"   Access Count: {pattern['access_count']}")
                print(f"   Success Rate: {pattern['success_rate']:.2%}")
                print(f"   Avg Confidence: {pattern['avg_confidence']:.2f}")
                print()
        else:
            print("(No hot patterns yet - need more queries)")
            print()


async def demo_4_advanced_refinement():
    """Demo 4: Advanced refinement with strategy selection (Phase 4)"""
    print("=" * 80)
    print("DEMO 4: Advanced Refinement with Strategy Selection (Phase 4)")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=False
    ) as engine:

        # First, a query that might have low confidence
        query = Query(text="embedding")  # Ambiguous, likely low confidence

        print(f"Query: {query.text}")
        print("(Intentionally vague to trigger refinement)")
        print()

        spacetime = await engine.weave(
            query,
            enable_refinement=True,
            refinement_threshold=0.8,  # High threshold to trigger refinement
            max_refinement_iterations=3
        )

        print(f"Final Confidence: {spacetime.trace.tool_confidence:.2f}")
        print()

        # Show scratchpad with refinement history
        print("Refinement History (from Scratchpad):")
        print("-" * 80)
        history = engine.get_scratchpad_history()
        # Show last few entries (refinement iterations)
        entries = history.split("\n\n")
        for entry in entries[-3:]:  # Last 3 iterations
            print(entry)
            print()

        # Show refinement strategy statistics
        stats = engine.get_learning_statistics()
        if stats.get('refinement_strategies'):
            print("Refinement Strategy Performance:")
            print("-" * 80)
            for strategy, perf in stats['refinement_strategies'].items():
                print(f"{strategy}:")
                print(f"  Uses: {perf['uses']}")
                print(f"  Avg Improvement: {perf['avg_improvement']:.3f}")
                print(f"  Success Rate: {perf['success_rate']:.2%}")
                print()


async def demo_5_background_learning():
    """Demo 5: Background learning with Thompson Sampling updates (Phase 5)"""
    print("=" * 80)
    print("DEMO 5: Background Learning with Thompson Sampling (Phase 5)")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=True,
        learning_update_interval=5.0  # Update every 5 seconds for demo
    ) as engine:

        print("Processing queries while background learner runs...")
        print()

        # Process multiple queries
        queries = [
            "What is Thompson Sampling?",
            "Explain attention mechanisms",
            "How does HoloLoom work?",
            "What is PPO?",
            "Explain Matryoshka embeddings",
        ]

        for i, query_text in enumerate(queries, 1):
            query = Query(text=query_text)
            spacetime = await engine.weave(query, enable_refinement=False)

            print(f"{i}. {query_text}")
            print(f"   Tool: {spacetime.trace.tool_selected}, "
                  f"Confidence: {spacetime.trace.tool_confidence:.2f}")

        print()
        print("Waiting for background learning update...")
        await asyncio.sleep(6)  # Wait for background update
        print()

        # Show learning statistics
        stats = engine.get_learning_statistics()

        print("Learning Statistics:")
        print("=" * 80)
        print()

        print(f"Queries Processed: {stats['queries_processed']}")
        print(f"Average Confidence: {stats['avg_confidence']:.2f}")
        print()

        # Thompson Sampling priors
        print("Thompson Sampling Priors:")
        print("-" * 80)
        if stats.get('thompson_priors'):
            for tool, priors in list(stats['thompson_priors'].items())[:3]:
                print(f"{tool}:")
                print(f"  Expected Reward: {priors['expected_reward']:.3f}")
                print(f"  Uncertainty: {priors['uncertainty']:.4f}")
                print(f"  Alpha: {priors['alpha']:.2f}, Beta: {priors['beta']:.2f}")
                print()

        # Policy adapter weights
        print("Policy Adapter Weights:")
        print("-" * 80)
        if stats.get('policy_weights'):
            for adapter, weights in stats['policy_weights'].items():
                print(f"{adapter}:")
                print(f"  Weight: {weights['weight']:.3f}")
                print(f"  Success Rate: {weights['success_rate']:.2%}")
                print(f"  Total Uses: {weights['total_uses']}")
                print()

        # Background learning info
        print("Background Learning:")
        print("-" * 80)
        bg = stats.get('background_learning', {})
        print(f"Enabled: {bg.get('enabled', False)}")
        print(f"Update Interval: {bg.get('update_interval', 0):.1f}s")
        print(f"Recent Experiences: {bg.get('recent_experiences', 0)}")
        print()


async def demo_6_complete_integration():
    """Demo 6: Complete integration of all 5 phases"""
    print("=" * 80)
    print("DEMO 6: Complete Integration of All 5 Phases")
    print("=" * 80)
    print()

    config = Config.fused()  # Use FUSED mode for best results
    shards = create_demo_shards()

    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=True,
        learning_update_interval=5.0
    ) as engine:

        print("Full system demonstration with all phases active:")
        print("- Phase 1: Scratchpad provenance tracking")
        print("- Phase 2: Pattern learning from successful queries")
        print("- Phase 3: Hot pattern feedback and adaptive retrieval")
        print("- Phase 4: Advanced refinement with strategy selection")
        print("- Phase 5: Background learning with Thompson/policy updates")
        print()

        # Process a mix of queries
        queries = [
            ("What is Thompson Sampling?", False),  # No refinement needed
            ("explain", True),  # Vague, needs refinement
            ("How does attention work in transformers?", False),
            ("embeddings", True),  # Vague, needs refinement
            ("What is PPO and how does it work?", False),
        ]

        for i, (query_text, expect_refinement) in enumerate(queries, 1):
            print(f"Query {i}: {query_text}")

            query = Query(text=query_text)
            spacetime = await engine.weave(
                query,
                enable_refinement=True,
                refinement_threshold=0.75
            )

            print(f"  Tool: {spacetime.trace.tool_selected}")
            print(f"  Confidence: {spacetime.trace.tool_confidence:.2f}")
            print(f"  Threads: {len(spacetime.trace.threads_activated)}")
            print(f"  Refinement Expected: {expect_refinement}")
            print()

        # Wait for background learning
        print("Waiting for background learning update...")
        await asyncio.sleep(6)
        print()

        # Comprehensive statistics
        stats = engine.get_learning_statistics()

        print("COMPLETE SYSTEM STATISTICS")
        print("=" * 80)
        print()

        print(f"📊 Overall:")
        print(f"   Queries: {stats['queries_processed']}")
        print(f"   Avg Confidence: {stats['avg_confidence']:.2f}")
        print()

        print(f"🔥 Hot Patterns: {len(stats.get('hot_patterns', []))}")
        if stats.get('hot_patterns'):
            top_hot = stats['hot_patterns'][0]
            print(f"   Top: {top_hot['element_id']} "
                  f"(heat={top_hot['heat_score']:.1f})")
            print()

        print(f"🧠 Learned Patterns: {len(stats.get('learned_patterns', []))}")
        if stats.get('learned_patterns'):
            top_pattern = stats['learned_patterns'][0]
            print(f"   Top: {', '.join(top_pattern['motifs'][:2])} → "
                  f"{top_pattern['tool']} "
                  f"({top_pattern['occurrences']} uses)")
            print()

        print(f"🎯 Thompson Sampling: {len(stats.get('thompson_priors', {}))} tools")
        if stats.get('thompson_priors'):
            best_tool = max(
                stats['thompson_priors'].items(),
                key=lambda x: x[1]['expected_reward']
            )
            print(f"   Best: {best_tool[0]} "
                  f"(reward={best_tool[1]['expected_reward']:.3f})")
            print()

        print(f"⚙️  Policy Adapters: {len(stats.get('policy_weights', {}))}")
        if stats.get('policy_weights'):
            best_adapter = max(
                stats['policy_weights'].items(),
                key=lambda x: x[1]['weight']
            )
            print(f"   Best: {best_adapter[0]} "
                  f"(weight={best_adapter[1]['weight']:.3f})")
            print()

        print(f"🔄 Refinement Strategies:")
        if stats.get('refinement_strategies'):
            for strategy, perf in stats['refinement_strategies'].items():
                if perf['uses'] > 0:
                    print(f"   {strategy}: {perf['uses']} uses, "
                          f"{perf['avg_improvement']:.3f} avg improvement")
            print()

        # Save learning state
        print("Saving complete learning state...")
        engine.save_learning_state("./learning_state_demo")
        print("✓ Saved to ./learning_state_demo/")
        print()


async def main():
    """Run all demos"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     COMPLETE RECURSIVE LEARNING SYSTEM DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  All 5 Phases: Scratchpad | Pattern Learning | Hot Patterns |".center(78) + "║")
    print("║" + "                Advanced Refinement | Full Learning Loop".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    demos = [
        ("Basic Processing", demo_1_basic_processing),
        ("Pattern Learning", demo_2_pattern_learning),
        ("Hot Patterns", demo_3_hot_patterns),
        ("Advanced Refinement", demo_4_advanced_refinement),
        ("Background Learning", demo_5_background_learning),
        ("Complete Integration", demo_6_complete_integration),
    ]

    for name, demo_fn in demos:
        try:
            await demo_fn()
            print(f"✓ {name} demo complete")
            print()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"✗ {name} demo failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    print()
    print("=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print()
    print("The recursive learning system demonstrates:")
    print("  ✓ Complete provenance tracking (Phase 1)")
    print("  ✓ Automatic pattern learning (Phase 2)")
    print("  ✓ Usage-based adaptation (Phase 3)")
    print("  ✓ Intelligent refinement strategies (Phase 4)")
    print("  ✓ Continuous background learning (Phase 5)")
    print()
    print("This is a self-improving knowledge system that learns from every interaction.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
