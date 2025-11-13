"""
Demo: Ultra-Fast Optimizer for Agentic Reasoning

Demonstrates <1ms decision-making for:
1. Sub-query routing (which agent to invoke)
2. Token gating (streaming confidence)
3. Verification triggering (fact-checking)
4. Sub-query prioritization (execution order)

Philosophy:
"Not every decision needs deep reasoning. Ultra-fast decisions enable
fluid agentic reasoning and natural conversation flow."
"""

import asyncio
import torch
import time
from typing import List

# Import Phase 6 components
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from HoloLoom.nested.ultra_fast import (
    UltraFastOptimizer,
    AgentType,
    DecisionType,
    route_subquery_fast,
    should_verify_fast
)
from HoloLoom.nested.hierarchy import (
    NestedLearningHierarchy,
    NestedLearningContext,
    OptimizationLevel
)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


async def demo_subquery_routing():
    """Demo 1: Sub-query routing"""
    print_section("Demo 1: Sub-Query Routing (<1ms)")

    optimizer = UltraFastOptimizer()

    # Test queries with different routing needs
    test_cases = [
        ("Is this claim accurate?", "What is Thompson Sampling?"),
        ("Show me Python code examples", "How to implement A/B testing?"),
        ("What is 2 + 2?", "Math problem"),
        ("Can you summarize this article?", "Long text processing"),
        ("Compare these two algorithms", "Algorithm analysis")
    ]

    print("Routing sub-queries to appropriate agents:\n")

    total_time = 0.0
    for subquery, parent in test_cases:
        # Mock working memory (in production, this comes from embeddings)
        working_memory = torch.randn(1, 244)

        start = time.perf_counter()
        decision = await optimizer.route_subquery(
            text=subquery,
            parent_query=parent,
            working_memory_state=working_memory
        )
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        print(f"Sub-query: '{subquery[:50]}'")
        print(f"  -> Routed to: {decision.agent_type.value}")
        print(f"  -> Confidence: {decision.confidence:.2f}")
        print(f"  -> Latency: {elapsed:.3f}ms")
        print()

    print(f"Average routing latency: {total_time / len(test_cases):.3f}ms")
    print(f"Target: <1ms per decision [OK]" if total_time / len(test_cases) < 1.0 else "Target: <1ms per decision [FAIL]")


async def demo_token_gating():
    """Demo 2: Token-level confidence gating"""
    print_section("Demo 2: Token Gating for Streaming (<0.1ms)")

    optimizer = UltraFastOptimizer()

    # Simulate streaming tokens
    tokens = [
        "Thompson", "Sampling", "is", "a", "Bayesian",
        "approach", "to", "the", "multi-armed", "bandit",
        "problem", ".", "It", "samples", "from", "posterior",
        "distributions", "..."
    ]

    print("Gating tokens during streaming response:\n")

    total_time = 0.0
    passed = 0
    blocked = 0

    for i, token in enumerate(tokens):
        # Mock working memory (confidence varies)
        confidence_factor = 0.5 + 0.5 * (i / len(tokens))  # Increase confidence over time
        working_memory = torch.randn(1, 244) * confidence_factor

        start = time.perf_counter()
        decision = await optimizer.gate_token(
            token=token,
            working_memory_state=working_memory,
            threshold=0.7
        )
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        if decision.confidence >= 0.7:
            status = "PASS"
            passed += 1
        else:
            status = "BLOCK"
            blocked += 1

        print(f"Token '{token:15}' -> {status:5} (conf={decision.confidence:.2f}, {elapsed:.3f}ms)")

    print(f"\nTotal: {passed} passed, {blocked} blocked")
    print(f"Average gating latency: {total_time / len(tokens):.3f}ms")
    print(f"Target: <0.1ms per token [OK]" if total_time / len(tokens) < 0.1 else "Target: <0.5ms per token")


async def demo_verification_triggering():
    """Demo 3: Verification triggering"""
    print_section("Demo 3: Verification Triggering (<1ms)")

    optimizer = UltraFastOptimizer()

    claims = [
        "Python was created in 1991",
        "I think neural networks are cool",  # Opinion, skip verification
        "Thompson Sampling always outperforms epsilon-greedy",  # Needs verification
        "2 + 2 = 4",  # Obviously true, skip
        "Paris is the capital of France",  # Known fact, skip
    ]

    print("Deciding which claims need verification:\n")

    total_time = 0.0
    for claim in claims:
        working_memory = torch.randn(1, 244)

        start = time.perf_counter()
        decision = await optimizer.should_verify(
            claim=claim,
            working_memory_state=working_memory
        )
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        needs_verification = decision.agent_type == AgentType.VERIFICATION
        status = "VERIFY" if needs_verification else "SKIP"

        print(f"Claim: '{claim[:60]}'")
        print(f"  -> {status} (conf={decision.confidence:.2f}, {elapsed:.3f}ms)")
        print()

    print(f"Average verification decision: {total_time / len(claims):.3f}ms")


async def demo_subquery_prioritization():
    """Demo 4: Sub-query prioritization"""
    print_section("Demo 4: Sub-Query Prioritization")

    optimizer = UltraFastOptimizer()

    # Multiple sub-queries to prioritize
    subqueries = [
        "What is the definition?",
        "Show me an example",
        "What are the tradeoffs?",
        "How does it compare to alternatives?",
        "Can I see the math behind it?"
    ]

    print(f"Prioritizing {len(subqueries)} sub-queries:\n")

    working_memory = torch.randn(1, 244)

    start = time.perf_counter()
    priorities = await optimizer.prioritize_subqueries(
        subqueries=subqueries,
        working_memory_state=working_memory
    )
    elapsed = (time.perf_counter() - start) * 1000

    print("Execution order (by priority):\n")
    for i, (subquery, score) in enumerate(priorities, 1):
        print(f"{i}. [{score:.2f}] {subquery}")

    print(f"\nPrioritization latency: {elapsed:.3f}ms for {len(subqueries)} queries")
    print(f"Average: {elapsed / len(subqueries):.3f}ms per query")


async def demo_hierarchy_integration():
    """Demo 5: Full hierarchy with all 5 levels"""
    print_section("Demo 5: Full Nested Learning Hierarchy")

    print("Initializing 5-level hierarchy...\n")

    async with NestedLearningContext() as hierarchy:
        print(hierarchy.summary())

        print("\nRunning ultra-fast decisions:")

        # Ultra-fast decision 1
        decision1 = await hierarchy.ultra_fast_decision(
            text="Should I verify this claim?",
            parent_query="What is Thompson Sampling?"
        )
        print(f"Decision 1: {decision1.agent_type.value if decision1.agent_type else 'None'} "
              f"(conf={decision1.confidence:.2f}, {decision1.latency_ms:.3f}ms)")

        # Ultra-fast decision 2
        decision2 = await hierarchy.ultra_fast_decision(
            text="Show me code examples",
            parent_query="How to implement PPO?"
        )
        print(f"Decision 2: {decision2.agent_type.value if decision2.agent_type else 'None'} "
              f"(conf={decision2.confidence:.2f}, {decision2.latency_ms:.3f}ms)")

        # Ultra-fast decision 3
        decision3 = await hierarchy.ultra_fast_decision(
            text="Explain the math",
            parent_query="What is Bayes' theorem?"
        )
        print(f"Decision 3: {decision3.agent_type.value if decision3.agent_type else 'None'} "
              f"(conf={decision3.confidence:.2f}, {decision3.latency_ms:.3f}ms)")

        print("\nHierarchy statistics:")
        print("-" * 60)
        stats = hierarchy.get_statistics()
        print(f"Total updates by level:")
        for level, count in stats['updates'].items():
            latency = stats['latencies'][level]
            print(f"  {level.upper()}: {count} updates, avg latency: {latency}")


async def demo_convenience_functions():
    """Demo 6: Convenience functions"""
    print_section("Demo 6: Convenience Functions")

    print("Using convenience functions for quick decisions:\n")

    # Quick routing
    print("1. Quick sub-query routing:")
    decision = await route_subquery_fast(
        "Is this accurate?",
        "Parent query"
    )
    print(f"   -> {decision.agent_type.value} ({decision.latency_ms:.3f}ms)\n")

    # Quick verification
    print("2. Quick verification check:")
    should_verify = await should_verify_fast("Python was created in 1991")
    print(f"   -> Verification needed: {should_verify}\n")


async def demo_statistics_tracking():
    """Demo 7: Statistics tracking"""
    print_section("Demo 7: Statistics Tracking")

    optimizer = UltraFastOptimizer()

    # Make various decisions
    print("Making 20 decisions to gather statistics...\n")

    for i in range(20):
        working_memory = torch.randn(1, 244)
        await optimizer.route_subquery(
            f"Query {i}",
            "Parent",
            working_memory
        )

        if i % 5 == 0:
            await optimizer.gate_token(
                f"token_{i}",
                working_memory
            )

    print(optimizer.summary())


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  Phase 6: Ultra-Fast Optimizer Demo")
    print("  <1ms decisions for fluid agentic reasoning")
    print("=" * 60)

    # Run demos
    await demo_subquery_routing()
    await demo_token_gating()
    await demo_verification_triggering()
    await demo_subquery_prioritization()
    await demo_hierarchy_integration()
    await demo_convenience_functions()
    await demo_statistics_tracking()

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
