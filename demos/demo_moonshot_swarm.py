#!/usr/bin/env python3
"""
Demo: Moonshot Swarm - Parallel Validation Framework

This demo showcases HoloLoom's Moonshot Swarm validation framework:
- ValidationPipeline: Composable validators with plugin architecture
- MoonshotSwarm: Parallel validation across multiple dimensions

Run with:
    PYTHONPATH=. python demos/demo_moonshot_swarm.py

Created: December 2025
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import validation framework
from HoloLoom.prompting.validation.pipeline import (
    ValidationPipeline,
    ValidationPhase,
    QualityValidator,
    LatencyValidator,
    UserSatisfactionValidator,
    StatisticalValidator,
)
from HoloLoom.prompting.validation.swarm import (
    MoonshotSwarm,
    SwarmResults,
)


def create_synthetic_data(
    n_queries: int = 100,
    baseline_quality_range: tuple = (0.72, 0.81),
    mrf_quality_range: tuple = (0.91, 1.00),
) -> tuple:
    """
    Create synthetic validation data.

    Args:
        n_queries: Number of queries to generate
        baseline_quality_range: (min, max) quality scores for baseline
        mrf_quality_range: (min, max) quality scores for MRF

    Returns:
        (baseline_queries, mrf_queries) tuple
    """
    baseline_queries = [
        {
            "quality_score": baseline_quality_range[0] + (i % 10) * 0.01,
            "latency_ms": 140 + (i % 5) * 2,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["recursive", "skills", "rag", "memory"][i % 4],
            "complexity": ["simple", "moderate", "complex"][i % 3],
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(n_queries)
    ]

    mrf_queries = [
        {
            "quality_score": mrf_quality_range[0] + (i % 10) * 0.01,
            "latency_ms": 155 + (i % 5) * 2,
            "user_rating": 4 + (i % 2),
            "thumbs_up": (i % 3) > 0,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["recursive", "skills", "rag", "memory"][i % 4],
            "complexity": ["simple", "moderate", "complex"][i % 3],
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(n_queries)
    ]

    return baseline_queries, mrf_queries


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}\n")


async def demo_validation_pipeline(baseline_queries, mrf_queries):
    """Demo 1: ValidationPipeline with multiple validators."""
    print_section("DEMO 1: Validation Pipeline")

    print("Creating pipeline with 4 validators:")
    print("  - QualityValidator (min 20% improvement)")
    print("  - LatencyValidator (max 20% regression)")
    print("  - UserSatisfactionValidator (min 4.0 rating)")
    print("  - StatisticalValidator (p < 0.05)")

    # Create pipeline with elegant chaining
    pipeline = (
        ValidationPipeline()
        .add_validator(QualityValidator(min_improvement_pct=20.0))
        .add_validator(LatencyValidator(max_regression_pct=20.0))
        .add_validator(UserSatisfactionValidator(min_rating=4.0))
        .add_validator(StatisticalValidator(significance_level=0.05))
    )

    print(f"\nRunning pipeline with {len(baseline_queries)} baseline and {len(mrf_queries)} MRF queries...")

    # Run validation
    results = await pipeline.run(
        baseline_queries=baseline_queries,
        mrf_queries=mrf_queries,
        phase=ValidationPhase.ANALYSIS,
    )

    # Display results
    print(f"\nOverall: {'PASS' if results.passed else 'FAIL'}")
    print(f"Summary: {results.summary}")
    print(f"Recommendation: {results.get_recommendation()}")

    print("\nValidator Results:")
    for name, result in results.validator_results.items():
        status = result.get("status", "unknown").upper()
        print(f"  {name}: {status}")

        if name == "quality":
            improvement = result.get("improvement_pct", 0)
            print(f"    Improvement: {improvement:+.1f}%")
        elif name == "latency":
            change = result.get("change_pct", 0)
            print(f"    Latency change: {change:+.1f}%")
        elif name == "user_satisfaction":
            rating = result.get("avg_rating", 0)
            print(f"    Average rating: {rating:.2f}")
        elif name == "statistical":
            p_value = result.get("p_value", 1.0)
            print(f"    P-value: {p_value:.4f}")

    return results


async def demo_moonshot_swarm(baseline_queries, mrf_queries):
    """Demo 2: MoonshotSwarm with parallel validation across dimensions."""
    print_section("DEMO 2: Moonshot Swarm")

    print("Creating swarm with 14 agents across 4 dimensions:")
    print("  - MODEL: claude, gemini, gpt (3 agents)")
    print("  - SYSTEM: recursive, skills, rag, memory (4 agents)")
    print("  - COMPLEXITY: simple, moderate, complex (3 agents)")
    print("  - TIMELINE: week_1 through week_4 (4 agents)")

    # Create swarm with elegant chaining
    swarm = (
        MoonshotSwarm()
        .add_model_agents(["claude", "gemini", "gpt"])
        .add_system_agents(["recursive", "skills", "rag", "memory"])
        .add_complexity_agents(["simple", "moderate", "complex"])
        .add_timeline_agents(weeks=4)
    )

    print(f"\nRunning {len(swarm.agents)} agents (max 5 concurrent)...")
    print()

    # Run swarm
    results = await swarm.run(
        baseline_queries=baseline_queries,
        mrf_queries=mrf_queries,
        max_concurrent=5,
    )

    # Print summary
    swarm.print_summary(results)

    return results


async def demo_custom_swarm():
    """Demo 3: Custom swarm with specific thresholds."""
    print_section("DEMO 3: Custom Swarm Configuration")

    # Create data with lower improvement to test graduated thresholds
    print("Creating data with ~15% quality improvement (borderline)...")
    baseline_queries, _ = create_synthetic_data(
        n_queries=100,
        baseline_quality_range=(0.80, 0.89),
        mrf_quality_range=(0.92, 1.00),  # ~15% improvement
    )
    mrf_queries = [
        {
            **q,
            "quality_score": q["quality_score"] + 0.12,  # 12-15% improvement
        }
        for q in baseline_queries
    ]

    print("\nTesting complexity agents with graduated thresholds:")
    print("  - simple: 15% quality, 15% latency (should pass)")
    print("  - moderate: 18% quality, 20% latency (may fail)")
    print("  - complex: 20% quality, 30% latency (should fail)")

    swarm = MoonshotSwarm()
    swarm.add_complexity_agents(["simple", "moderate", "complex"])

    results = await swarm.run(
        baseline_queries=baseline_queries,
        mrf_queries=mrf_queries,
        max_concurrent=3,
    )

    print(f"\nResults:")
    for agent in results.agents:
        status = "PASS" if agent.results and agent.results.passed else "FAIL"
        quality_result = agent.results.validator_results.get("quality", {}) if agent.results else {}
        improvement = quality_result.get("improvement_pct", 0)
        print(f"  {agent.agent_id}: {status} (improvement: {improvement:.1f}%)")

    return results


async def main():
    """Run all demos."""
    print()
    print("=" * 60)
    print(" MOONSHOT SWARM - Parallel Validation Framework Demo")
    print("=" * 60)
    print()
    print("This demo showcases HoloLoom's validation infrastructure:")
    print("- ValidationPipeline: Composable validators")
    print("- MoonshotSwarm: Parallel validation across dimensions")
    print("- Automatic recommendations based on pass rates")

    # Create shared synthetic data
    print("\n[*] Generating synthetic validation data (100 queries)...")
    baseline_queries, mrf_queries = create_synthetic_data(n_queries=100)
    print(f"    Baseline quality range: 0.72-0.81")
    print(f"    MRF quality range: 0.91-1.00")
    print(f"    Expected improvement: ~25%")

    # Run demos
    await demo_validation_pipeline(baseline_queries, mrf_queries)
    await demo_moonshot_swarm(baseline_queries, mrf_queries)
    await demo_custom_swarm()

    # Final summary
    print_section("DEMO COMPLETE")
    print("Key takeaways:")
    print("  1. ValidationPipeline enables composable, reusable validators")
    print("  2. MoonshotSwarm provides parallel validation across dimensions")
    print("  3. Graduated thresholds allow different expectations by complexity")
    print("  4. 75% pass rate threshold for deployment recommendations")
    print()
    print("For production use, replace synthetic data with real A/B test results.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
