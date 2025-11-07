"""
Demo: Thompson Sampling Bandit Orchestrator

Demonstrates the complete bandit system:
1. Shadow mode (training only)
2. A/B testing with traffic split
3. Metrics tracking (ECE, rewards, regret)
4. Safety integration
5. Learning over time

Run: PYTHONPATH=. python demos/demo_bandit_orchestrator.py
"""

import asyncio
import numpy as np
from HoloLoom.weaving_orchestrator_bandit import create_bandit_orchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard


def create_demo_shards(n=20):
    """Create demo memory shards about Thompson Sampling."""
    topics = [
        ("Thompson Sampling", "thompson_sampling", "Bayesian exploration strategy"),
        ("Multi-armed bandits", "bandit", "Sequential decision making under uncertainty"),
        ("Exploration vs Exploitation", "exploration", "Balancing known vs unknown options"),
        ("Beta distribution", "beta", "Conjugate prior for Bernoulli likelihood"),
        ("Contextual bandits", "context", "Bandits with side information"),
        ("Upper Confidence Bound", "ucb", "Optimistic exploration strategy"),
        ("Epsilon-greedy", "epsilon", "Simple random exploration"),
        ("Bayesian optimization", "bayes_opt", "Sample-efficient black-box optimization"),
        ("Gaussian Process", "gp", "Non-parametric regression with uncertainty"),
        ("Neural bandits", "neural", "Deep learning for contextual bandits"),
    ]

    shards = []
    for i, (text, motif, desc) in enumerate(topics * 2):  # Duplicate for 20 shards
        shards.append(
            MemoryShard(
                id=f"shard_{i}",
                text=f"{text}: {desc}",
                motifs=[motif, "ml", "decision_making"],
                entities=[text, "machine_learning"],
            )
        )
    return shards


async def demo_shadow_mode():
    """Demo 1: Shadow mode - training without affecting traffic."""
    print("\n" + "="*70)
    print("DEMO 1: Shadow Mode (0% traffic to bandit)")
    print("="*70)

    config = Config.fast()
    shards = create_demo_shards()

    # Shadow mode: 0% traffic
    orchestrator = create_bandit_orchestrator(
        cfg=config,
        shards=shards,
        enable_bandit=True,
        ab_test_ratio=0.0,  # 0% traffic - shadow only
        sampler_type="discrete",  # Simple for demo
    )

    # Run some queries
    queries = [
        "What is Thompson Sampling?",
        "How does exploration work?",
        "Explain contextual bandits",
    ]

    print("\nProcessing queries in shadow mode...")
    for query_text in queries:
        query = Query(text=query_text)
        spacetime = await orchestrator.weave(query)

        print(f"\n  Query: {query_text[:50]}...")
        print(f"  Tool: {spacetime.metadata.get('tool_used', 'N/A')}")
        print(f"  Bandit used: {spacetime.metadata.get('bandit_used', False)}")
        print(f"  Confidence: {spacetime.confidence:.3f}")

    # Check metrics
    metrics = orchestrator.get_bandit_metrics()
    print("\n" + "-"*70)
    print("Shadow Mode Metrics:")
    print(f"  Total decisions: {metrics['total_decisions']}")
    print(f"  Bandit decisions: {metrics['bandit_decisions']} (should be 0)")
    print(f"  Baseline decisions: {metrics['baseline_decisions']} (should be 3)")
    print(f"  Replay buffer size: {metrics.get('replay_size', 0)}")

    await orchestrator.close()


async def demo_ab_testing():
    """Demo 2: A/B testing with 50% traffic split."""
    print("\n" + "="*70)
    print("DEMO 2: A/B Testing (50% traffic split)")
    print("="*70)

    config = Config.fast()
    shards = create_demo_shards()

    # A/B test: 50% traffic
    orchestrator = create_bandit_orchestrator(
        cfg=config,
        shards=shards,
        enable_bandit=True,
        ab_test_ratio=0.5,  # 50% to bandit
        sampler_type="discrete",
    )

    # Run many queries to see split
    queries = [
        "What is Thompson Sampling?",
        "How does UCB work?",
        "Explain epsilon-greedy",
        "What is Bayesian optimization?",
        "How do neural bandits work?",
        "What is exploration?",
        "Explain GP-TS",
        "What are contextual bandits?",
    ]

    print("\nProcessing queries with A/B split...")
    bandit_count = 0
    baseline_count = 0

    for query_text in queries:
        query = Query(text=query_text)
        spacetime = await orchestrator.weave(query)

        bandit_used = spacetime.metadata.get('bandit_used', False)
        if bandit_used:
            bandit_count += 1
        else:
            baseline_count += 1

        print(f"\n  Query: {query_text[:50]}...")
        print(f"  Bandit: {'YES' if bandit_used else 'NO'}")
        print(f"  Tool: {spacetime.metadata.get('tool_used', 'N/A')}")
        print(f"  Confidence: {spacetime.confidence:.3f}")

    # Check A/B ratio
    metrics = orchestrator.get_bandit_metrics()
    print("\n" + "-"*70)
    print("A/B Testing Metrics:")
    print(f"  Total decisions: {metrics['total_decisions']}")
    print(f"  Bandit decisions: {metrics['bandit_decisions']} ({bandit_count} expected)")
    print(f"  Baseline decisions: {metrics['baseline_decisions']} ({baseline_count} expected)")
    print(f"  Actual ratio: {metrics.get('ab_ratio_actual', 0):.2%}")
    print(f"  Target ratio: {metrics.get('ab_ratio_target', 0.5):.2%}")
    print(f"  Variance: ±{abs(metrics.get('ab_ratio_actual', 0.5) - 0.5):.1%}")

    await orchestrator.close()


async def demo_metrics_tracking():
    """Demo 3: Comprehensive metrics tracking."""
    print("\n" + "="*70)
    print("DEMO 3: Metrics Tracking (ECE, Regret, Rewards)")
    print("="*70)

    config = Config.fast()
    shards = create_demo_shards()

    # Full bandit mode for metric tracking
    orchestrator = create_bandit_orchestrator(
        cfg=config,
        shards=shards,
        enable_bandit=True,
        ab_test_ratio=1.0,  # 100% to bandit
        sampler_type="discrete",
    )

    # Run queries and track metrics
    queries = [
        "What is Thompson Sampling?",
        "How does exploration work?",
        "Explain contextual bandits",
        "What is the Beta distribution?",
        "How do neural bandits work?",
    ]

    print("\nProcessing queries and tracking metrics...")
    for query_text in queries:
        query = Query(text=query_text)
        spacetime = await orchestrator.weave(query)

        print(f"\n  Query: {query_text[:50]}...")
        print(f"  Tool: {spacetime.metadata.get('tool_used', 'N/A')}")
        print(f"  Confidence: {spacetime.confidence:.3f}")

    # Get comprehensive metrics
    metrics = orchestrator.get_bandit_metrics()

    print("\n" + "-"*70)
    print("Comprehensive Metrics:")
    print(f"\n  Decisions:")
    print(f"    Total: {metrics['total_decisions']}")
    print(f"    Bandit: {metrics['bandit_decisions']}")
    print(f"    Safety blocks: {metrics['safety_blocks']}")

    if 'ece' in metrics and metrics['ece'] is not None:
        print(f"\n  Calibration:")
        print(f"    ECE: {metrics['ece']:.4f}")
        print(f"    Status: {'Good' if metrics['ece'] < 0.1 else 'Needs improvement'}")

    if 'mean_reward' in metrics and metrics['mean_reward'] is not None:
        print(f"\n  Rewards:")
        print(f"    Mean: {metrics['mean_reward']:.4f}")
        if 'reward_std' in metrics:
            print(f"    Std: {metrics['reward_std']:.4f}")

    if 'cumulative_regret' in metrics and metrics['cumulative_regret'] is not None:
        print(f"\n  Regret:")
        print(f"    Cumulative: {metrics['cumulative_regret']:.2f}")

    # Decision log
    decision_log = orchestrator.get_decision_log(limit=3)
    print(f"\n  Recent Decisions (last 3):")
    for i, decision in enumerate(decision_log, 1):
        print(f"    {i}. Tool: {decision.get('tool', 'N/A')}, "
              f"Reward: {decision.get('reward', 0):.3f}, "
              f"Bandit: {decision.get('bandit_used', False)}")

    await orchestrator.close()


async def demo_learning():
    """Demo 4: Learning over time."""
    print("\n" + "="*70)
    print("DEMO 4: Learning Over Time")
    print("="*70)

    config = Config.fast()
    shards = create_demo_shards()

    orchestrator = create_bandit_orchestrator(
        cfg=config,
        shards=shards,
        enable_bandit=True,
        ab_test_ratio=1.0,
        sampler_type="discrete",
    )

    # Same query repeated - bandit should learn
    query_text = "What is Thompson Sampling?"
    n_iterations = 5

    print(f"\nRepeating query {n_iterations} times to observe learning...")
    print(f"Query: {query_text}\n")

    tool_counts = {}

    for i in range(n_iterations):
        query = Query(text=query_text)
        spacetime = await orchestrator.weave(query)

        tool = spacetime.metadata.get('tool_used', 'N/A')
        conf = spacetime.confidence

        tool_counts[tool] = tool_counts.get(tool, 0) + 1

        print(f"  Iteration {i+1}: Tool={tool}, Confidence={conf:.3f}")

    print("\n" + "-"*70)
    print("Tool Selection Distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}/{n_iterations} ({count/n_iterations:.1%})")

    # Get statistics
    stats = orchestrator.get_bandit_statistics()
    if 'tool_counts' in stats:
        print("\n  Bandit Internal Statistics:")
        print(f"  Tool usage from bandit perspective:")
        for tool, count in stats['tool_counts'].items():
            print(f"    {tool}: {count} selections")

    await orchestrator.close()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Thompson Sampling Bandit Orchestrator - Demo Suite")
    print("="*70)
    print("\nDemonstrating complete bandit system:")
    print("  1. Shadow mode (training only)")
    print("  2. A/B testing (traffic split)")
    print("  3. Metrics tracking (ECE, rewards, regret)")
    print("  4. Learning over time")

    try:
        await demo_shadow_mode()
        await demo_ab_testing()
        await demo_metrics_tracking()
        await demo_learning()

        print("\n" + "="*70)
        print("Demo Complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review metrics output above")
        print("  2. Check THOMPSON_SAMPLING_PRODUCTION_READY.md for deployment guide")
        print("  3. Deploy to shadow mode (0% traffic) for 1 week")
        print("  4. Gradually roll out with A/B testing")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
