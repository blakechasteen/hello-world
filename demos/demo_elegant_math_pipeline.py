#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elegant Math Pipeline Demo - Showcasing Beauty & Power
=======================================================
Demonstrates the elegant, sexy math pipeline with:
- Fluent API (method chaining)
- Beautiful terminal UI (Rich library)
- Interactive HTML dashboard
- Async/parallel execution
- Real-time RL learning visualization

Author: HoloLoom Team
Date: 2025-10-29
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.warp.math_pipeline_elegant import (
    ElegantMathPipeline,
    analyze,
    analyze_sync
)
from HoloLoom.warp.math_dashboard_generator import generate_math_dashboard


async def demo_fluent_api():
    """Demo: Fluent API with method chaining."""
    print("\n" + "="*80)
    print("DEMO 1: Fluent API - Elegant Method Chaining")
    print("="*80)
    print()
    print("# Create pipeline with fluent API")
    print("pipeline = (ElegantMathPipeline()")
    print("    .fast()")
    print("    .beautiful_output())")
    print()

    # Create pipeline with method chaining
    async with (ElegantMathPipeline()
        .fast()
        .beautiful_output()
    ) as pipeline:

        print("# Analyze query")
        print('await pipeline.analyze("Find documents similar to quantum computing")')
        print()

        result = await pipeline.analyze(
            "Find documents similar to quantum computing",
            context={"has_embeddings": True}
        )

        print("\n✨ Beautiful output above showcases:")
        print("  - Colored intent detection")
        print("  - Operation tree visualization")
        print("  - Cost progress bar")
        print("  - Rich result panel with insights")


async def demo_one_liner():
    """Demo: One-liner analysis function."""
    print("\n" + "="*80)
    print("DEMO 2: One-Liner Analysis - Maximum Convenience")
    print("="*80)
    print()

    print("# Single line to analyze any query:")
    print('result = await analyze("Optimize the retrieval algorithm", mode="fast")')
    print()

    result = await analyze(
        "Optimize the retrieval algorithm",
        mode="fast",
        beautiful=True,
        requires_optimization=True
    )

    print("\n✨ One function call does everything!")
    print("  - Intent classification")
    print("  - Smart operation selection")
    print("  - Execution with RL learning")
    print("  - Beautiful terminal output")


async def demo_batch_processing():
    """Demo: Batch processing with parallel execution."""
    print("\n" + "="*80)
    print("DEMO 3: Batch Processing - Parallel Execution")
    print("="*80)
    print()

    queries = [
        "Find similar documents to artificial intelligence",
        "Optimize the search algorithm for speed",
        "Analyze the knowledge graph structure",
        "Verify that the metric space is valid",
        "Transform embeddings to hyperbolic space"
    ]

    print(f"# Analyze {len(queries)} queries in parallel:")
    print("results = await pipeline.analyze_batch(queries)")
    print()

    async with (ElegantMathPipeline()
        .fast()
        .beautiful_output()
    ) as pipeline:

        results = await pipeline.analyze_batch(queries, show_progress=True)

        print(f"\n✨ Processed {len(results)} queries in parallel!")
        print("  - Async/await for non-blocking execution")
        print("  - Smart caching (repeated queries instant)")
        print("  - Beautiful progress indication")


async def demo_mode_comparison():
    """Demo: Different modes (LITE → FAST → FULL → RESEARCH)."""
    print("\n" + "="*80)
    print("DEMO 4: Mode Comparison - Progressive Enhancement")
    print("="*80)
    print()

    query = "Find the shortest path between concepts A and B"

    modes = ["lite", "fast", "full", "research"]

    print(f"Query: \"{query}\"\n")

    for mode in modes:
        pipeline = ElegantMathPipeline()

        if mode == "lite":
            pipeline = pipeline.lite()
        elif mode == "fast":
            pipeline = pipeline.fast()
        elif mode == "full":
            pipeline = pipeline.full()
        elif mode == "research":
            pipeline = pipeline.research()

        result = await pipeline.analyze(query)

        if result:
            print(f"{mode.upper():10} | Ops: {len(result.operations_used):2} | "
                  f"Cost: {result.total_cost:3} | "
                  f"Time: {result.execution_time_ms:5.1f}ms | "
                  f"Confidence: {result.confidence:.0%}")

    print("\n✨ Progressive enhancement:")
    print("  - LITE (10): Basic ops only, fastest")
    print("  - FAST (50): + RL learning")
    print("  - FULL (100): + Composition + Testing")
    print("  - RESEARCH (999): + Expensive ops (Ricci flow, etc.)")


async def demo_statistics_and_trends():
    """Demo: Statistics tracking with beautiful visualization."""
    print("\n" + "="*80)
    print("DEMO 5: Statistics & Trends - Beautiful Insights")
    print("="*80)
    print()

    async with (ElegantMathPipeline()
        .fast()
        .beautiful_output()
    ) as pipeline:

        # Run several queries
        queries = [
            "Find similar documents",
            "Optimize retrieval",
            "Analyze structure",
            "Verify correctness",
            "Transform embeddings"
        ]

        print("# Analyzing 5 queries...")
        for query in queries:
            await pipeline.analyze(query)

        print("\n# Show statistics with beautiful table:")
        print("pipeline.show_statistics()")
        print()

        pipeline.show_statistics()

        print("\n# Show trends with sparklines:")
        print("pipeline.show_trends()")
        print()

        pipeline.show_trends()

        print("\n✨ Beautiful statistics visualization:")
        print("  - Rich tables with color")
        print("  - RL leaderboard (top operations)")
        print("  - Sparklines for trends")
        print("  - Real-time metrics")


async def demo_interactive_dashboard():
    """Demo: Generate interactive HTML dashboard."""
    print("\n" + "="*80)
    print("DEMO 6: Interactive HTML Dashboard - Visualization")
    print("="*80)
    print()

    # Create pipeline and run queries
    async with (ElegantMathPipeline()
        .fast()
        .enable_rl()
    ) as pipeline:

        # Run multiple queries to generate data
        print("# Running queries to generate dashboard data...")
        queries = [
            "Find similar documents to AI",
            "Find similar documents to ML",
            "Optimize search performance",
            "Optimize retrieval speed",
            "Analyze graph topology",
            "Verify metric properties",
            "Transform feature space"
        ]

        results = []
        for query in queries:
            result = await pipeline.analyze(query)
            if result:
                results.append({
                    "execution_time_ms": result.execution_time_ms,
                    "confidence": result.confidence,
                    "total_cost": result.total_cost,
                    "operations_used": result.operations_used
                })

        # Get statistics
        stats = pipeline.statistics()

        # Generate dashboard
        print("\n# Generating interactive HTML dashboard...")
        print("generate_math_dashboard(stats, results)")
        print()

        output_path = generate_math_dashboard(stats, results)

        print(f"\n✨ Interactive dashboard generated!")
        print(f"  - File: {output_path}")
        print("  - Real-time charts with Plotly.js")
        print("  - RL leaderboard visualization")
        print("  - Operation flow diagram")
        print("  - Beautiful gradients & animations")
        print(f"\n  Open in browser: file://{Path(output_path).absolute()}")


async def demo_elegant_features():
    """Demo: All elegant features in one showcase."""
    print("\n" + "="*80)
    print("DEMO 7: Elegant Features Showcase")
    print("="*80)
    print()

    # Create the most elegant pipeline
    async with (ElegantMathPipeline()
        .fast()
        .enable_composition()
        .beautiful_output()
    ) as pipeline:

        print("✨ Elegant pipeline created with:")
        print(f"  - {pipeline}")
        print()

        # Single query
        print("# Elegant one-liner analysis:")
        result = await pipeline.analyze(
            "Find documents similar to machine learning and verify the results",
            context={"has_embeddings": True, "needs_verification": True}
        )

        # Show cache
        print("\n# Cache demonstration (instant results):")
        print("# Analyzing same query again...")
        import time
        start = time.time()
        cached_result = await pipeline.analyze(
            "Find documents similar to machine learning and verify the results",
            use_cache=True
        )
        elapsed = (time.time() - start) * 1000
        print(f"⚡ Cache hit! Time: {elapsed:.2f}ms (vs {result.execution_time_ms:.1f}ms first time)")

        # Clear cache
        print("\n# Clear cache with fluent API:")
        print("pipeline.clear_cache().save_state()")
        pipeline.clear_cache().save_state()
        print("✓ Cache cleared, RL state saved")

        print("\n✨ Elegant features demonstrated:")
        print("  - Fluent API (method chaining)")
        print("  - Beautiful terminal UI (Rich)")
        print("  - Smart caching (instant repeated queries)")
        print("  - RL state persistence")
        print("  - Async/await support")
        print("  - Context manager (auto cleanup)")


async def main():
    """Run all demos."""
    print("="*80)
    print("ELEGANT MATH PIPELINE - Beauty & Power Showcase")
    print("="*80)
    print()
    print("Demonstrating:")
    print("  1. Fluent API (method chaining)")
    print("  2. One-liner convenience")
    print("  3. Batch parallel processing")
    print("  4. Progressive enhancement (LITE → RESEARCH)")
    print("  5. Beautiful statistics & trends")
    print("  6. Interactive HTML dashboard")
    print("  7. Complete elegant features")
    print()
    input("Press Enter to start demos...")

    demos = [
        demo_fluent_api,
        demo_one_liner,
        demo_batch_processing,
        demo_mode_comparison,
        demo_statistics_and_trends,
        demo_interactive_dashboard,
        demo_elegant_features
    ]

    for demo in demos:
        try:
            await demo()
            print("\n" + "-"*80)
            input("Press Enter for next demo...")
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ELEGANT MATH PIPELINE - All Demos Complete!")
    print("="*80)
    print()
    print("Key Takeaways:")
    print("  ✓ Fluent API makes code beautiful and readable")
    print("  ✓ Beautiful terminal UI enhances developer experience")
    print("  ✓ Interactive dashboards provide powerful insights")
    print("  ✓ Async/parallel execution for performance")
    print("  ✓ Progressive enhancement (LITE → RESEARCH)")
    print("  ✓ Smart caching and RL learning")
    print()
    print("✨ Beauty is a feature, not a luxury! ✨")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
