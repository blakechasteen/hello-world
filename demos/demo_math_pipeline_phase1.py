#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Pipeline Phase 1 Demo
===========================
Demonstrates the Smart Math Pipeline integration with basic operations.

Phase 1 Features:
- Intent classification
- Operation selection (inner_product, metric_distance, norm)
- Basic execution (mock results for now)
- Meaning synthesis (numbers → natural language)
- Clean integration pattern

This demo shows how the math pipeline enriches query understanding by:
1. Classifying query intent (SIMILARITY, OPTIMIZATION, etc.)
2. Selecting appropriate mathematical operations
3. Executing operations (mock in Phase 1)
4. Synthesizing natural language explanations

Author: HoloLoom Team
Date: 2025-10-29
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.warp.math_pipeline_integration import (
    create_math_integration_lite,
    create_math_integration_fast,
    MathPipelineIntegration
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def demo_basic_integration():
    """Demo: Basic integration (LITE mode)."""
    print_section("DEMO 1: Basic Integration (LITE Mode)")

    # Create integration
    integration = create_math_integration_lite()
    print(f"Integration: {integration}\n")

    # Test query
    query = "Find documents similar to quantum computing"
    context = {"has_embeddings": True}
    embedding = np.random.randn(384)  # Mock embedding

    print(f"Query: {query}")
    print(f"Context: {context}\n")

    # Analyze
    result = integration.analyze(
        query_text=query,
        query_embedding=embedding,
        context=context
    )

    # Show results
    if result:
        print("RESULT:")
        print(f"  Summary: {result.summary}")
        print(f"\n  Insights:")
        for insight in result.insights:
            print(f"    • {insight}")
        print(f"\n  Operations used: {', '.join(result.operations_used)}")
        print(f"  Total cost: {result.total_cost}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")

        # Show detailed meaning
        if result.meaning:
            print(f"\n  Detailed explanation:")
            print("  " + "-"*76)
            detailed = result.meaning.to_text(style="detailed")
            for line in detailed.split("\n"):
                print(f"  {line}")
    else:
        print("Math pipeline returned no result (disabled or error)")


def demo_multiple_queries():
    """Demo: Multiple queries showing different intents."""
    print_section("DEMO 2: Multiple Query Intents")

    integration = create_math_integration_lite()

    test_cases = [
        ("Find documents similar to machine learning", {"has_embeddings": True}),
        ("What is the distance between concept A and B?", {}),
        ("Analyze the structure of the knowledge graph", {}),
        ("Verify that the metric space is valid", {"needs_verification": True}),
        ("How similar are these two documents?", {"has_embeddings": True}),
    ]

    for i, (query, context) in enumerate(test_cases, 1):
        print(f"{i}. Query: {query}")

        result = integration.analyze(
            query_text=query,
            query_embedding=np.random.randn(384),
            context=context
        )

        if result:
            print(f"   Intent detected: {result.metadata.get('intent', 'unknown')}")
            print(f"   Operations: {', '.join(result.operations_used)}")
            print(f"   Summary: {result.summary}")
        else:
            print("   (No result)")

        print()


def demo_fast_mode():
    """Demo: FAST mode with higher budget."""
    print_section("DEMO 3: FAST Mode (Higher Budget)")

    # Create FAST mode integration (budget: 50)
    integration = create_math_integration_fast()
    print(f"Integration: {integration}\n")

    query = "Optimize the retrieval algorithm for better performance"
    context = {"requires_optimization": True}

    print(f"Query: {query}")
    print(f"Context: {context}\n")

    result = integration.analyze(
        query_text=query,
        query_embedding=np.random.randn(384),
        context=context
    )

    if result:
        print("RESULT:")
        print(f"  Summary: {result.summary}")
        print(f"\n  Operations used: {', '.join(result.operations_used)}")
        print(f"  Total cost: {result.total_cost} (budget: 50)")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")

        # Note about RL
        print("\n  NOTE: FAST mode has RL enabled (Thompson Sampling)")
        print("        Operation selection will improve over time with feedback")


def demo_statistics():
    """Demo: Statistics tracking."""
    print_section("DEMO 4: Statistics Tracking")

    integration = create_math_integration_lite()

    # Run multiple analyses
    queries = [
        "Find similar documents",
        "What is the distance?",
        "How similar are these?",
        "Analyze the structure",
        "Find closest match"
    ]

    print("Running 5 queries...\n")

    for query in queries:
        integration.analyze(
            query_text=query,
            query_embedding=np.random.randn(384),
            context={}
        )

    # Show statistics
    stats = integration.get_statistics()

    print("STATISTICS:")
    print(f"  Total analyses: {stats['total_analyses']}")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Avg operations/analysis: {stats.get('avg_operations_per_analysis', 0):.1f}")
    print(f"  Avg cost/analysis: {stats.get('avg_cost_per_analysis', 0):.1f}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2%}")

    if "avg_execution_time_ms" in stats:
        print(f"  Avg execution time: {stats['avg_execution_time_ms']:.1f}ms")
        print(f"  P95 execution time: {stats['p95_execution_time_ms']:.1f}ms")

    # Show operations by intent
    print(f"\n  Operations by intent:")
    for intent, data in stats["operations_by_intent"].items():
        print(f"    {intent}: {data['count']} queries")
        for op, count in data['operations'].items():
            print(f"      - {op}: {count} times")


def demo_graceful_degradation():
    """Demo: Graceful degradation when disabled."""
    print_section("DEMO 5: Graceful Degradation")

    # Create disabled integration
    integration = MathPipelineIntegration(enabled=False)
    print(f"Integration: {integration}\n")

    query = "Find similar documents"

    result = integration.analyze(
        query_text=query,
        query_embedding=np.random.randn(384),
        context={}
    )

    if result is None:
        print("✓ Math pipeline gracefully degraded (returned None)")
        print("  Orchestrator can continue without math analysis")
    else:
        print("✗ Expected None, got result")


def demo_meaning_synthesis_styles():
    """Demo: Different output styles."""
    print_section("DEMO 6: Meaning Synthesis Styles")

    integration = create_math_integration_lite()

    query = "Find documents similar to artificial intelligence"
    context = {"has_embeddings": True}

    print(f"Query: {query}\n")

    result = integration.analyze(
        query_text=query,
        query_embedding=np.random.randn(384),
        context=context
    )

    if result and result.meaning:
        print("CONCISE style:")
        print("-" * 80)
        print(result.meaning.to_text(style="concise"))
        print()

        print("\nDETAILED style:")
        print("-" * 80)
        print(result.meaning.to_text(style="detailed"))
        print()

        print("\nTECHNICAL style:")
        print("-" * 80)
        print(result.meaning.to_text(style="technical"))


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos."""
    print("="*80)
    print("MATH PIPELINE PHASE 1 DEMO")
    print("Basic Integration: Intent Classification → Operation Selection → Meaning")
    print("="*80)

    demos = [
        ("Basic Integration", demo_basic_integration),
        ("Multiple Query Intents", demo_multiple_queries),
        ("FAST Mode", demo_fast_mode),
        ("Statistics Tracking", demo_statistics),
        ("Graceful Degradation", demo_graceful_degradation),
        ("Meaning Synthesis Styles", demo_meaning_synthesis_styles),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo {i} ({name}) failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print_section("PHASE 1 SUMMARY")

    print("✓ Phase 1 Features Demonstrated:")
    print("  1. Intent classification (7 intent types)")
    print("  2. Operation selection (basic operations)")
    print("  3. Mock execution (Phase 2+ will use actual warp/math/ modules)")
    print("  4. Meaning synthesis (natural language explanations)")
    print("  5. Statistics tracking")
    print("  6. Graceful degradation (no crashes if disabled)")
    print("  7. Multiple output styles (concise/detailed/technical)")
    print()
    print("Next Steps:")
    print("  Phase 2: Enable RL learning (Thompson Sampling)")
    print("  Phase 3: Add operation composition + rigorous testing")
    print("  Phase 4: Activate advanced operations (eigenvalues, Ricci flow)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()