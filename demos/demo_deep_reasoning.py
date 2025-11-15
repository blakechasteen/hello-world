#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEEP Reasoning Mode Demo
=========================
Demonstrates Phase 3: Planning, Backtracking, Thompson Sampling.

Author: Claude Code
Date: 2025-11-15
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HoloLoom.documentation.types import Query, Features, Context, MemoryShard
from HoloLoom.reasoning import (
    ReasoningEngine,
    ReasoningMode,
    StepType,
)
from HoloLoom.reasoning.bandit import ReasoningModeBandit


# ============================================================================
# Demo Utilities
# ============================================================================

def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_reasoning_chain(result):
    """Print reasoning chain in readable format."""
    print(f"Mode: {result.mode.value.upper()}")
    print(f"Total Confidence: {result.total_confidence:.2f}")
    print(f"Duration: {result.duration_ms:.1f}ms")
    print(f"Steps: {len(result.chain)}\n")

    # Step type icons
    icons = {
        StepType.PLANNING: "📋",
        StepType.UNDERSTANDING: "🧠",
        StepType.EVIDENCE: "🔍",
        StepType.SYNTHESIS: "🔗",
        StepType.VERIFICATION: "✓",
        StepType.BACKTRACK: "↩",
        StepType.CORRECTION: "🔧",
    }

    for i, step in enumerate(result.chain, 1):
        icon = icons.get(step.step_type, "•")
        print(f"{i}. {icon} [{step.confidence:.2f}] {step.step_type.value.upper()}")
        print(f"   Thought: {step.thought[:100]}{'...' if len(step.thought) > 100 else ''}")
        if step.evidence and len(step.evidence) > 10:
            print(f"   Evidence: {step.evidence[:100]}{'...' if len(step.evidence) > 100 else ''}")
        print()


def create_sample_context(topic: str = "Thompson Sampling") -> Context:
    """Create sample context for demos."""
    shards = [
        MemoryShard(
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It uses probability matching to balance exploration and exploitation by sampling "
                 "from posterior distributions of arm rewards.",
            entities=["thompson_sampling", "bayesian", "multi_armed_bandit", "exploration", "exploitation"],
            timestamp=1.0
        ),
        MemoryShard(
            text="The algorithm maintains Beta distributions for each arm, updating the alpha "
                 "parameter on success and beta parameter on failure. At each step, it samples "
                 "from each arm's distribution and selects the arm with the highest sample.",
            entities=["beta_distribution", "alpha", "beta", "sampling", "algorithm"],
            timestamp=1.0
        ),
        MemoryShard(
            text="In non-stationary environments where arm rewards change over time, Thompson Sampling "
                 "naturally adapts because the posterior distributions track recent performance. "
                 "Epsilon-greedy with fixed epsilon may explore too much or too little.",
            entities=["nonstationary", "adaptation", "epsilon_greedy", "exploration_rate"],
            timestamp=1.0
        ),
        MemoryShard(
            text="With delayed rewards, Thompson Sampling can credit earlier actions through "
                 "Bayesian updating. The posterior propagates information about which actions "
                 "led to later rewards, enabling better credit assignment than simpler methods.",
            entities=["delayed_rewards", "credit_assignment", "bayesian_updating", "posterior"],
            timestamp=1.0
        ),
        MemoryShard(
            text="UCB (Upper Confidence Bound) is a deterministic algorithm that uses confidence "
                 "intervals to balance exploration and exploitation. While it has strong theoretical "
                 "guarantees, it can be more conservative than Thompson Sampling in practice.",
            entities=["ucb", "upper_confidence_bound", "deterministic", "confidence_intervals"],
            timestamp=1.0
        ),
        MemoryShard(
            text="Empirical studies show Thompson Sampling often outperforms UCB and epsilon-greedy "
                 "in real-world scenarios with complex reward structures. However, UCB has better "
                 "worst-case theoretical bounds for regret minimization.",
            entities=["empirical_studies", "performance", "regret", "theoretical_bounds"],
            timestamp=1.0
        ),
    ]
    return Context(shards=shards, retrieved_at=None)


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_fast_vs_standard_vs_deep():
    """Compare FAST, STANDARD, and DEEP modes on the same query."""
    print_section("Demo 1: Comparing FAST vs STANDARD vs DEEP Modes")

    query = Query(text="How does Thompson Sampling balance exploration and exploitation?")
    features = Features(
        motifs=["thompson", "sampling", "exploration", "exploitation"],
        embeddings=[[0.1] * 96],
        spectral=None
    )
    context = create_sample_context()

    print(f"Query: {query.text}\n")

    for mode in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]:
        print(f"\n{'-'*70}")
        print(f"Running {mode.value.upper()} mode...")
        print(f"{'-'*70}")

        engine = ReasoningEngine(mode=mode, max_thinking_steps=8)
        result = await engine.reason(query, features, context)

        print_reasoning_chain(result)


async def demo_complex_analytical_query():
    """Demonstrate DEEP mode on complex analytical query."""
    print_section("Demo 2: DEEP Mode on Complex Analytical Query")

    query = Query(
        text="Why does Thompson Sampling work better than epsilon-greedy "
             "in non-stationary environments with delayed rewards?"
    )
    features = Features(
        motifs=["thompson", "sampling", "epsilon", "greedy", "nonstationary", "delayed", "rewards"],
        embeddings=[[0.1] * 96, [0.2] * 192],
        spectral={"eigenvalues": [0.6, 0.3, 0.1]}
    )
    context = create_sample_context()

    print(f"Query: {query.text}\n")

    engine = ReasoningEngine(mode=ReasoningMode.DEEP, max_thinking_steps=10)
    result = await engine.reason(query, features, context)

    print_reasoning_chain(result)

    # Show metadata
    print(f"{'-'*70}")
    print("DEEP Mode Metadata:")
    print(f"{'-'*70}")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


async def demo_comparative_planning():
    """Demonstrate query planning for comparative question."""
    print_section("Demo 3: Query Planning for Comparative Analysis")

    query = Query(text="Compare Thompson Sampling and UCB algorithms in detail")
    features = Features(
        motifs=["thompson", "sampling", "ucb", "compare"],
        embeddings=[[0.1] * 96],
        spectral=None
    )
    context = create_sample_context()

    print(f"Query: {query.text}\n")

    from HoloLoom.reasoning.planner import QueryPlanner
    planner = QueryPlanner()

    # Analyze intent
    intent = planner.analyze_intent(query, features)
    print(f"Query Type: {intent.type.value}")
    print(f"Complexity: {intent.complexity:.2f}")
    print(f"Key Concepts: {', '.join(intent.key_concepts[:5])}")
    print(f"Requirements: {', '.join(intent.requirements[:3])}\n")

    # Create plan
    plan = planner.create_plan(query, features, context)
    print(f"Query Plan ({len(plan.steps)} steps):")
    print(f"{'-'*70}")

    for i, step in enumerate(plan.steps):
        deps = plan.dependencies.get(i, [])
        deps_str = f" (depends on: {deps})" if deps else ""
        print(f"{i}. [{step.complexity:.1f}] {step.question}{deps_str}")
        print(f"   Required for: {step.required_for}")
        print()

    # Execute with DEEP mode
    print(f"{'-'*70}")
    print("Executing plan with DEEP mode...")
    print(f"{'-'*70}\n")

    engine = ReasoningEngine(mode=ReasoningMode.DEEP, max_thinking_steps=12)
    result = await engine.reason(query, features, context)

    print_reasoning_chain(result)


async def demo_thompson_sampling_mode_selection():
    """Demonstrate Thompson Sampling for mode selection."""
    print_section("Demo 4: Thompson Sampling for Adaptive Mode Selection")

    bandit = ReasoningModeBandit()

    queries = [
        ("What is Thompson Sampling?", 0.3, 0.8),  # Simple, good context
        ("Compare A and B", 0.5, 0.7),  # Medium complexity
        ("Why does X happen in complex scenarios?", 0.8, 0.6),  # Complex
        ("Is Y true?", 0.4, 0.75),  # Medium
    ]

    print("Simulating adaptive mode selection across queries...\n")

    for i, (query_text, complexity, quality) in enumerate(queries, 1):
        print(f"{i}. Query: \"{query_text}\"")
        print(f"   Complexity: {complexity:.2f}, Context Quality: {quality:.2f}")

        # Select mode
        selected_mode = bandit.select_mode(complexity, quality)
        print(f"   Selected Mode: {selected_mode.value.upper()}")

        # Simulate outcome (simplified)
        confidence = 0.85 if selected_mode != ReasoningMode.DEEP else 0.9
        duration = {
            ReasoningMode.FAST: 40.0,
            ReasoningMode.STANDARD: 180.0,
            ReasoningMode.DEEP: 450.0
        }[selected_mode]

        # Update bandit
        success = confidence >= 0.75
        bandit.update(selected_mode, success, confidence, duration)

        print(f"   Outcome: Confidence={confidence:.2f}, Duration={duration:.0f}ms\n")

    # Show final statistics
    print(f"{'-'*70}")
    print("Final Bandit Statistics:")
    print(f"{'-'*70}\n")

    stats = bandit.get_stats()
    for mode, mode_stats in stats.items():
        print(f"{mode.upper()}:")
        print(f"  Uses: {mode_stats['total_uses']}")
        print(f"  Success Rate: {mode_stats['success_rate']:.2f}")
        print(f"  Avg Confidence: {mode_stats['avg_confidence']:.2f}")
        print(f"  Avg Duration: {mode_stats['avg_duration_ms']:.1f}ms")
        print(f"  Expected Reward: {mode_stats['expected_reward']:.2f}")
        print()


async def demo_backtracking():
    """Demonstrate backtracking on contradictory evidence."""
    print_section("Demo 5: Backtracking on Contradictory Evidence")

    # Create contradictory context
    contradictory_shards = [
        MemoryShard(
            text="Thompson Sampling always outperforms all other algorithms in every scenario.",
            entities=["thompson", "always", "best"],
            timestamp=1.0
        ),
        MemoryShard(
            text="Thompson Sampling never works well and always fails in practical applications.",
            entities=["thompson", "never", "fails"],
            timestamp=1.0
        ),
        MemoryShard(
            text="Empirical studies show Thompson Sampling is sometimes better, sometimes worse "
                 "than alternatives depending on the problem structure.",
            entities=["thompson", "empirical", "sometimes"],
            timestamp=1.0
        ),
    ]
    context = Context(shards=contradictory_shards, retrieved_at=None)

    query = Query(text="Is Thompson Sampling always the best choice?")
    features = Features(
        motifs=["thompson", "sampling", "best"],
        embeddings=[[0.1] * 96],
        spectral=None
    )

    print(f"Query: {query.text}")
    print("\nNote: Context contains contradictory statements\n")

    engine = ReasoningEngine(mode=ReasoningMode.DEEP, max_thinking_steps=10)
    result = await engine.reason(query, features, context)

    print_reasoning_chain(result)

    # Check for backtracking
    backtrack_steps = [s for s in result.chain if s.step_type == StepType.BACKTRACK]
    if backtrack_steps:
        print(f"{'-'*70}")
        print(f"Backtracking occurred: {len(backtrack_steps)} backtrack step(s)")
        print(f"{'-'*70}")
        for step in backtrack_steps:
            print(f"\n{step.thought}")
            print(f"Evidence: {step.evidence}")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def main():
    """Run all demos."""
    print("\n")
    print("="*70)
    print("  HoloLoom Phase 3: DEEP Reasoning Mode Demonstration")
    print("="*70)

    demos = [
        ("FAST vs STANDARD vs DEEP", demo_fast_vs_standard_vs_deep),
        ("Complex Analytical Query", demo_complex_analytical_query),
        ("Comparative Query Planning", demo_comparative_planning),
        ("Thompson Sampling Mode Selection", demo_thompson_sampling_mode_selection),
        ("Backtracking on Contradictions", demo_backtracking),
    ]

    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n0. Run all demos")

    try:
        choice = input("\nSelect demo (0-5): ").strip()

        if choice == "0":
            # Run all demos
            for name, demo_func in demos:
                await demo_func()
                input("\nPress Enter to continue to next demo...")
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            # Run selected demo
            name, demo_func = demos[int(choice) - 1]
            await demo_func()
        else:
            print("Invalid choice. Running demo 1 by default.")
            await demos[0][1]()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return

    print("\n" + "="*70)
    print("  Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
