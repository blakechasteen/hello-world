#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Pattern Demo
====================

Demonstrates the DS-STAR (Data Science with Self-Taught Autonomous Reasoning)
pattern for autonomous data analysis.

Usage:
    PYTHONPATH=. python demos/demo_dsstar.py
"""

import asyncio
import pandas as pd
from pathlib import Path

# Import DS-STAR components
from src.hololoom.patterns.dsstar import (
    DSStarOrchestrator,
    DataAnalyzer,
    analyze_data,
    create_plan
)


def create_sample_data():
    """Create sample dataset for demo."""
    data = {
        'region': ['North', 'South', 'East', 'West'] * 25,
        'sales': [100, 150, 120, 180, 200, 220, 190, 210] * 12 + [150, 160, 140, 170],
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 8 + ['Jan', 'Feb', 'Mar', 'Apr'],
        'year': [2023] * 50 + [2024] * 50,
    }
    df = pd.DataFrame(data)

    output_path = Path('demos/output/sample_sales.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return str(output_path)


async def demo_basic_analysis():
    """Demo 1: Basic data analysis."""
    print("=" * 60)
    print("DEMO 1: Basic Data Analysis")
    print("=" * 60)

    # Create sample data
    data_file = create_sample_data()
    print(f"Created sample data: {data_file}")
    print()

    # Analyze data
    print("Step 1: Analyzing data...")
    profile = analyze_data(data_file)
    print(f"  → {profile.row_count} rows, {profile.column_count} columns")
    print()

    # Show anchors
    print("Data Anchors:")
    for anchor in profile.anchors:
        print(f"  {anchor.name} ({anchor.type}):")
        print(f"    - Unique: {anchor.unique_count}")
        print(f"    - Missing: {anchor.missing_count}")
        if anchor.sample_values:
            print(f"    - Samples: {anchor.sample_values[:3]}")
    print()


async def demo_plan_creation():
    """Demo 2: Execution plan creation."""
    print("=" * 60)
    print("DEMO 2: Execution Plan Creation")
    print("=" * 60)

    data_file = create_sample_data()
    analyzer = DataAnalyzer()
    profile = analyzer.analyze_file(data_file)

    # Create plan for query
    query = "What is the average sales by region?"
    print(f"Query: {query}")
    print()

    plan = create_plan(query, profile)

    print("Execution Plan:")
    print(f"  Complexity: {plan.estimated_complexity}")
    print(f"  Steps: {len(plan.steps)}")
    print()

    for step in plan.steps:
        print(f"  Step {step.step_id}: {step.action.value}")
        print(f"    Description: {step.description}")
        print(f"    Code: {step.code_template}")
        if step.dependencies:
            print(f"    Depends on: {step.dependencies}")
        print()


async def demo_full_dsstar():
    """Demo 3: Full DS-STAR iterative loop."""
    print("=" * 60)
    print("DEMO 3: Full DS-STAR Iterative Loop")
    print("=" * 60)

    data_file = create_sample_data()

    # Create orchestrator
    orchestrator = DSStarOrchestrator(
        max_iterations=3,
        verification_threshold=0.7,
        enable_refinement=True
    )

    # Process query
    query = "Calculate the average sales by region and create a summary"
    print(f"Query: {query}")
    print()

    print("Processing...")
    result = await orchestrator.process(query, data_file)
    print()

    # Show results
    print(result.summary())
    print()

    # Show iteration details
    print("Iteration History:")
    for iteration in result.iterations:
        print(f"\n  Iteration {iteration.iteration + 1}:")
        print(f"    Plan: {len(iteration.plan.steps)} steps")
        print(f"    Execution: {'✓ Success' if iteration.execution_result.success else '✗ Failed'}")
        print(f"    Confidence: {iteration.verification.confidence:.1%}")

        if iteration.verification.issues:
            print(f"    Issues: {iteration.verification.issues}")

    # Explain process
    print("\n" + "=" * 60)
    print("Process Explanation:")
    print("=" * 60)
    print(orchestrator.explain_process(result))


async def demo_with_visualization():
    """Demo 4: DS-STAR with visualization."""
    print("=" * 60)
    print("DEMO 4: DS-STAR with Visualization")
    print("=" * 60)

    data_file = create_sample_data()

    orchestrator = DSStarOrchestrator(max_iterations=2)

    query = "Plot sales by region"
    print(f"Query: {query}")
    print()

    result = await orchestrator.process(query, data_file)

    print(result.summary())

    if 'plot' in result.final_outputs:
        print("\n✓ Visualization created successfully!")
    else:
        print("\n⚠ Visualization not created")


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "DS-STAR PATTERN DEMO" + " " * 23 + "║")
    print("║" + " " * 10 + "Data Science with Self-Taught" + " " * 19 + "║")
    print("║" + " " * 13 + "Autonomous Reasoning" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    try:
        # Run demos
        await demo_basic_analysis()
        input("Press Enter to continue...")

        await demo_plan_creation()
        input("Press Enter to continue...")

        await demo_full_dsstar()
        input("Press Enter to continue...")

        await demo_with_visualization()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
