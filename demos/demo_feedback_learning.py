#!/usr/bin/env python3
"""
Feedback Learning Demo
======================
Demonstrates the complete reflection learning system with user feedback.

This demo shows:
1. Storing user feedback on query responses
2. Extracting learning signals (success rates, Thompson priors)
3. Updating Thompson Sampling bandits automatically
4. Visualizing learning trajectory over time
5. A/B testing different strategies

Author: Claude Code
Date: 2025-11-20
"""

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.reflection.feedback_store import FeedbackStore
from hololoom.policy.thompson_sampling import TSBandit


async def demo_basic_feedback():
    """Demo 1: Basic feedback storage and retrieval."""
    print("=" * 60)
    print("DEMO 1: Basic Feedback Storage")
    print("=" * 60)

    async with FeedbackStore(db_path="./data/demo_feedback.db") as store:
        # User queries and rates responses
        queries = [
            ("What is reinforcement learning?", "answer", 0.92, 1.0, "Excellent explanation!"),
            ("How does PPO work?", "answer", 0.88, 0.8, "Good but could be clearer"),
            ("Show me example code", "research", 0.75, 0.6, "Not quite what I needed"),
            ("Explain Thompson Sampling", "answer", 0.95, 1.0, "Perfect!"),
            ("Compare RL algorithms", "research", 0.82, 0.9, "Very comprehensive"),
        ]

        print(f"\nStoring {len(queries)} feedback records...")

        for query, tool, confidence, rating, text in queries:
            feedback_id = await store.store_feedback(
                query=query,
                response=f"[Response to: {query}]",
                tool_used=tool,
                confidence=confidence,
                user_rating=rating,
                feedback_type="explicit",
                explicit_feedback=text,
                user_id="@demo:local"
            )
            print(f"  ✓ {feedback_id[:8]}: {tool} (rating={rating:.1f})")

        # Get statistics
        print("\n" + "─" * 60)
        print("Feedback Statistics:")
        stats = await store.get_statistics()
        print(f"  Total feedback: {stats['total_feedback_count']}")
        print(f"  Recent (24h): {stats['recent_feedback_24h']}")

        print("\n  Per-tool statistics:")
        for tool, tool_stats in stats['tool_statistics'].items():
            print(f"    {tool}: {tool_stats['count']} queries, "
                  f"avg_rating={tool_stats['avg_rating']:.2f}, "
                  f"avg_confidence={tool_stats['avg_confidence']:.2f}")

        # Get learning signals
        print("\n" + "─" * 60)
        print("Learning Signals:")
        signals = await store.get_learning_signals(min_samples=1)

        for tool, signal in signals.items():
            print(f"\n  {tool}:")
            print(f"    Samples: {signal.total_samples} (✓{signal.successful_samples} / ✗{signal.failed_samples})")
            print(f"    Success Rate: {signal.success_rate:.1%}")
            print(f"    Avg Rating: {signal.avg_rating:.2f}")
            print(f"    Thompson Priors: α={signal.alpha:.2f}, β={signal.beta:.2f}")
            print(f"    Expected Reward: {signal.expected_reward:.3f}")


async def demo_thompson_sampling():
    """Demo 2: Thompson Sampling integration."""
    print("\n\n" + "=" * 60)
    print("DEMO 2: Thompson Sampling Updates")
    print("=" * 60)

    # Create bandit with 3 tools
    tools = ["answer", "research", "explore"]
    bandit = TSBandit(n_arms=len(tools))

    print(f"\nInitial Thompson Sampling priors:")
    for i, tool in enumerate(tools):
        stats = bandit.get_stats()[i]
        print(f"  {tool}: α={stats['success']:.2f}, β={stats['fail']:.2f}, "
              f"E[X]={stats['mean']:.3f}")

    # Simulate user feedback
    print("\n" + "─" * 60)
    print("Simulating user feedback...")

    feedback_data = [
        ("answer", [0.9, 0.95, 0.8, 0.85, 0.9]),  # High ratings
        ("research", [0.7, 0.75, 0.6, 0.65, 0.7]),  # Medium ratings
        ("explore", [0.5, 0.4, 0.45, 0.55, 0.5]),  # Low ratings
    ]

    for tool, ratings in feedback_data:
        tool_idx = tools.index(tool)
        print(f"\n  {tool} tool:")

        for rating in ratings:
            bandit.update_from_feedback(tool_idx, rating)
            print(f"    Rating {rating:.2f} → α={bandit.success[tool_idx]:.2f}, β={bandit.fail[tool_idx]:.2f}")

    # Final priors
    print("\n" + "─" * 60)
    print("Updated Thompson Sampling priors:")
    for i, tool in enumerate(tools):
        stats = bandit.get_stats()[i]
        print(f"  {tool}: α={stats['success']:.2f}, β={stats['fail']:.2f}, "
              f"E[X]={stats['mean']:.3f} ({stats['pulls']:.0f} pulls)")

    # Compare expected rewards
    print("\n" + "─" * 60)
    print("Expected Rewards (higher is better):")
    expected_rewards = [(tool, bandit.get_stats()[i]['mean']) for i, tool in enumerate(tools)]
    expected_rewards.sort(key=lambda x: x[1], reverse=True)

    for i, (tool, reward) in enumerate(expected_rewards, 1):
        print(f"  {i}. {tool}: {reward:.3f}")

    # Save bandit state
    bandit.save_state("./data/demo_bandit_state.json")
    print("\n✓ Bandit state saved to ./data/demo_bandit_state.json")


async def demo_learning_trajectory():
    """Demo 3: Learning trajectory over time."""
    print("\n\n" + "=" * 60)
    print("DEMO 3: Learning Trajectory Over Time")
    print("=" * 60)

    async with FeedbackStore(db_path="./data/demo_trajectory.db") as store:
        # Simulate 30 days of feedback
        print("\nSimulating 30 days of feedback...")
        print("(System improves over time as it learns)")

        start_date = datetime.now() - timedelta(days=30)

        for day in range(30):
            # Simulate 5 queries per day
            for _ in range(5):
                # Rating improves over time (system learns)
                base_rating = 0.5 + (day / 60)  # 0.5 → 1.0
                rating = min(1.0, base_rating + random.uniform(-0.1, 0.1))

                await store.store_feedback(
                    query=f"Query on day {day}",
                    response="Response",
                    tool_used="answer",
                    confidence=0.85,
                    user_rating=rating,
                    feedback_type="rating",
                    user_id="@demo:local"
                )

        # Get trajectory
        signals = await store.get_learning_signals(tool="answer", min_samples=1)
        signal = signals["answer"]

        print(f"\n" + "─" * 60)
        print("Final Learning Signals:")
        print(f"  Total samples: {signal.total_samples}")
        print(f"  Success rate: {signal.success_rate:.1%}")
        print(f"  Avg rating: {signal.avg_rating:.2f}")
        print(f"  Thompson priors: α={signal.alpha:.1f}, β={signal.beta:.1f}")
        print(f"  Expected reward: {signal.expected_reward:.3f}")

        # Show trajectory visualization (text-based)
        print(f"\n" + "─" * 60)
        print("Learning Trajectory (rolling 10-query window):")
        print()

        records = await store.get_feedback_history(tool="answer", limit=1000)

        window_size = 10
        trajectory_points = []

        for i in range(window_size, len(records), window_size // 2):
            window = records[i-window_size:i]
            ratings = [r.user_rating for r in window]
            avg_rating = sum(ratings) / len(ratings)
            trajectory_points.append(avg_rating)

        # ASCII bar chart
        max_rating = max(trajectory_points)
        for i, rating in enumerate(trajectory_points):
            bar_length = int((rating / max_rating) * 40)
            bar = "█" * bar_length
            print(f"  Window {i+1:2d}: {bar} {rating:.2f}")


async def demo_ab_testing():
    """Demo 4: A/B testing foundation."""
    print("\n\n" + "=" * 60)
    print("DEMO 4: A/B Testing Foundation")
    print("=" * 60)

    async with FeedbackStore(db_path="./data/demo_ab_test.db") as store:
        print("\nComparing two strategies:")
        print("  Strategy A: 'answer' tool")
        print("  Strategy B: 'research' tool")

        print("\nSimulating 50 queries per strategy...")

        # Strategy A: answer tool (high success rate)
        for i in range(50):
            await store.store_feedback(
                query=f"Query A{i}",
                response="Response A",
                tool_used="answer",
                confidence=0.90,
                user_rating=random.uniform(0.7, 1.0),  # High ratings
                feedback_type="rating",
                user_id="@demo:local",
                metadata={"strategy": "A"}
            )

        # Strategy B: research tool (lower success rate)
        for i in range(50):
            await store.store_feedback(
                query=f"Query B{i}",
                response="Response B",
                tool_used="research",
                confidence=0.85,
                user_rating=random.uniform(0.5, 0.8),  # Lower ratings
                feedback_type="rating",
                user_id="@demo:local",
                metadata={"strategy": "B"}
            )

        # Compare signals
        signals = await store.get_learning_signals(min_samples=1)

        print("\n" + "─" * 60)
        print("A/B Test Results:")

        print("\n  Strategy A ('answer' tool):")
        a_signal = signals['answer']
        print(f"    Success Rate: {a_signal.success_rate:.1%}")
        print(f"    Avg Rating: {a_signal.avg_rating:.2f}")
        print(f"    Expected Reward: {a_signal.expected_reward:.3f}")

        print("\n  Strategy B ('research' tool):")
        b_signal = signals['research']
        print(f"    Success Rate: {b_signal.success_rate:.1%}")
        print(f"    Avg Rating: {b_signal.avg_rating:.2f}")
        print(f"    Expected Reward: {b_signal.expected_reward:.3f}")

        # Statistical comparison
        print("\n" + "─" * 60)
        print("Winner Determination:")

        if a_signal.expected_reward > b_signal.expected_reward:
            improvement = (a_signal.expected_reward - b_signal.expected_reward) / b_signal.expected_reward
            print(f"  ✅ Strategy A wins by {improvement:.1%}")
            print(f"     Recommended: Use 'answer' tool for this query type")
        else:
            improvement = (b_signal.expected_reward - a_signal.expected_reward) / a_signal.expected_reward
            print(f"  ✅ Strategy B wins by {improvement:.1%}")
            print(f"     Recommended: Use 'research' tool for this query type")


async def main():
    """Run all demos."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Reflection Learning from User Feedback - Demo Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    # Create data directory
    Path("./data").mkdir(exist_ok=True)

    # Run demos
    await demo_basic_feedback()
    await demo_thompson_sampling()
    await demo_learning_trajectory()
    await demo_ab_testing()

    print("\n\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. ✓ User feedback stored with complete context")
    print("  2. ✓ Thompson Sampling priors updated automatically")
    print("  3. ✓ Learning signals track success rates over time")
    print("  4. ✓ System improves as it learns from feedback")
    print("  5. ✓ A/B testing infrastructure ready for experiments")

    print("\nNext Steps:")
    print("  - Integrate feedback UI into dashboard")
    print("  - Connect to production HoloLoom orchestrator")
    print("  - Monitor learning trajectory in real-time")
    print("  - Run A/B tests on different strategies")

    print("\nSee FEEDBACK_LEARNING_README.md for complete documentation.")


if __name__ == "__main__":
    asyncio.run(main())
