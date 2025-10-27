#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Reflection Loop Demo
==============================
Demonstrates the self-improving weaving system with reflection and learning.

This demo shows:
1. Weaving with reflection enabled
2. User feedback collection
3. Learning signal generation
4. System adaptation based on outcomes
5. Performance improvement over time

Run: python -c "import sys; sys.path.insert(0, '.'); from demos import reflection_demo; import asyncio; asyncio.run(reflection_demo.main())"
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard


# ============================================================================
# Demo Data
# ============================================================================

def create_sample_memory() -> list[MemoryShard]:
    """Create sample memory shards for testing."""
    return [
        MemoryShard(
            id="thompson_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
            episode="ml_docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit", "exploration", "exploitation"],
            motifs=["ALGORITHM", "OPTIMIZATION", "PROBABILITY"]
        ),
        MemoryShard(
            id="thompson_002",
            text="The algorithm samples from posterior distributions to naturally balance trying new options with exploiting known good ones.",
            episode="ml_docs",
            entities=["algorithm", "posterior distributions", "exploration", "exploitation"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
        MemoryShard(
            id="python_001",
            text="Python is a high-level programming language known for its readability and extensive libraries for data science and ML.",
            episode="programming_docs",
            entities=["Python", "programming", "data science", "ML"],
            motifs=["PROGRAMMING", "LANGUAGE"]
        ),
        MemoryShard(
            id="math_001",
            text="The derivative of a function measures its instantaneous rate of change. It's fundamental to calculus and optimization.",
            episode="math_docs",
            entities=["derivative", "calculus", "optimization", "rate of change"],
            motifs=["MATH", "CALCULUS", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="hive_001",
            text="Hive Jodi has 8 frames of brood and is very active. The goldenrod flow is providing excellent nectar.",
            episode="inspection_2025_10_13",
            entities=["Hive Jodi", "brood", "goldenrod", "nectar"],
            motifs=["HIVE_INSPECTION", "SEASONAL", "BEEKEEPING"]
        ),
    ]


# ============================================================================
# Query Scenarios
# ============================================================================

QUERY_SCENARIOS = [
    {
        "query": "What is Thompson Sampling?",
        "expected_tool": "answer",
        "user_helpful": True
    },
    {
        "query": "Search for beekeeping tips",
        "expected_tool": "search",
        "user_helpful": True
    },
    {
        "query": "Calculate 25 * 17",
        "expected_tool": "calc",
        "user_helpful": True
    },
    {
        "query": "Write this to my Notion database",
        "expected_tool": "notion_write",
        "user_helpful": False  # Let's say user didn't find this helpful
    },
    {
        "query": "Explain Python programming",
        "expected_tool": "answer",
        "user_helpful": True
    },
    {
        "query": "What is a derivative in calculus?",
        "expected_tool": "answer",
        "user_helpful": True
    },
    {
        "query": "Find information about optimization",
        "expected_tool": "search",
        "user_helpful": True
    },
    {
        "query": "How many frames of brood does Hive Jodi have?",
        "expected_tool": "answer",
        "user_helpful": True
    },
]


# ============================================================================
# Demo Functions
# ============================================================================

def simulate_user_feedback(spacetime, scenario: dict) -> dict:
    """
    Simulate user feedback based on scenario.

    In a real system, this would come from actual user input.
    """
    # Base feedback on whether tool matches expected and confidence
    helpful = scenario.get("user_helpful", True)
    tool_match = spacetime.tool_used == scenario.get("expected_tool", spacetime.tool_used)

    # Adjust helpfulness based on tool match
    if not tool_match:
        helpful = False

    # Generate rating (1-5)
    if helpful and tool_match:
        rating = 4 + (1 if spacetime.confidence > 0.6 else 0)
    elif helpful:
        rating = 3
    else:
        rating = 2

    return {
        "helpful": helpful,
        "rating": rating,
        "comment": "Good" if helpful else "Could be better"
    }


async def run_weaving_session(
    shuttle: WeavingShuttle,
    scenarios: list,
    session_name: str
) -> None:
    """Run a session of weaving with multiple queries."""
    print(f"\n{'='*80}")
    print(f"Session: {session_name}")
    print(f"{'='*80}")

    for i, scenario in enumerate(scenarios, 1):
        query_text = scenario["query"]
        query = Query(text=query_text)

        print(f"\n[{i}/{len(scenarios)}] Query: '{query_text}'")
        print("-" * 80)

        # Weave
        spacetime = await shuttle.weave(query)

        # Simulate user feedback
        feedback = simulate_user_feedback(spacetime, scenario)

        # Reflect
        await shuttle.reflect(spacetime, feedback=feedback)

        # Display results
        print(f"  Tool Used: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2f}")
        print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")
        print(f"  User Feedback: {feedback['rating']}/5 {'(helpful)' if feedback['helpful'] else '(not helpful)'}")

    # Analyze and learn
    print(f"\n{'='*80}")
    print("Analyzing Session for Learning...")
    print(f"{'='*80}\n")

    signals = await shuttle.learn(force=True)

    if signals:
        print(f"Generated {len(signals)} learning signals:\n")
        for i, signal in enumerate(signals, 1):
            print(f"{i}. [{signal.signal_type.upper()}] {signal.recommendation}")
            print(f"   Priority: {signal.priority:.2f}")
            if signal.tool:
                print(f"   Tool: {signal.tool}")
            if signal.pattern:
                print(f"   Pattern: {signal.pattern}")
            if signal.evidence:
                print(f"   Evidence: {signal.evidence}")
            print()

        # Apply signals
        await shuttle.apply_learning_signals(signals)
    else:
        print("No learning signals generated (not enough data yet)")


def display_metrics(shuttle: WeavingShuttle) -> None:
    """Display reflection metrics."""
    metrics = shuttle.get_reflection_metrics()

    if not metrics:
        print("Reflection not enabled")
        return

    print(f"\n{'='*80}")
    print("Reflection Metrics")
    print(f"{'='*80}")

    print(f"\nOverall Performance:")
    print(f"  Total Cycles: {metrics['total_cycles']}")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")

    print(f"\nTool Performance:")
    for tool, rate in metrics['tool_success_rates'].items():
        print(f"  {tool:15s}: {rate:.1%} success")

    print(f"\nTool Recommendations (0-1):")
    for tool, score in sorted(
        metrics['tool_recommendations'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {tool:15s}: {score:.2f}")

    if metrics['pattern_success_rates']:
        print(f"\nPattern Performance:")
        for pattern, rate in metrics['pattern_success_rates'].items():
            print(f"  {pattern:15s}: {rate:.1%}")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run comprehensive reflection loop demo."""
    print("\n" + "="*80)
    print("HoloLoom Reflection Loop - Comprehensive Demo")
    print("="*80)
    print("\nThis demo shows the system learning from outcomes and improving over time.")

    # Create memory
    print("\nInitializing system with sample memory...")
    shards = create_sample_memory()

    # Create config
    config = Config.fused()

    # Create shuttle with reflection enabled
    print("Creating WeavingShuttle with reflection enabled...")
    shuttle = WeavingShuttle(
        cfg=config,
        shards=shards,
        enable_reflection=True,
        reflection_capacity=1000
    )
    print("Shuttle ready!\n")

    # Run first session (initial learning)
    await run_weaving_session(
        shuttle,
        QUERY_SCENARIOS,
        "Session 1: Initial Learning"
    )

    # Display metrics after first session
    display_metrics(shuttle)

    # Run second session (with adapted behavior)
    print("\n\n")
    print("="*80)
    print("Running Second Session with Adapted Behavior...")
    print("="*80)

    await run_weaving_session(
        shuttle,
        QUERY_SCENARIOS[:4],  # Subset of queries
        "Session 2: Adapted Behavior"
    )

    # Final metrics
    display_metrics(shuttle)

    # Show improvement
    print(f"\n{'='*80}")
    print("Learning Summary")
    print(f"{'='*80}\n")

    buffer = shuttle.reflection_buffer
    print(f"Total episodes stored: {len(buffer)}")
    print(f"Success rate: {buffer.get_success_rate():.1%}")
    print(f"\nThe system has:")
    print("  1. Stored all weaving outcomes")
    print("  2. Analyzed patterns for success/failure")
    print("  3. Generated learning signals")
    print("  4. Adapted tool selection based on feedback")
    print("\nThis is continuous improvement in action!")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
