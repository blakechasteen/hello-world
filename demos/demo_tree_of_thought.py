"""
Tree-of-Thought Planning Demo
==============================
Demonstrates systematic solution space exploration using tree search.

Usage:
    PYTHONPATH=. python demos/demo_tree_of_thought.py

**Implementation Date**: 2025-11-16
"""

import asyncio
from HoloLoom.alignment.tree_of_thought import TreeOfThought


async def demo_authentication_planning():
    """Plan authentication system design."""
    print("=" * 80)
    print("DEMO 1: Authentication System Planning")
    print("=" * 80)

    planner = TreeOfThought(
        max_depth=4,
        beam_width=3,
        min_value_threshold=0.3,
    )

    result = await planner.plan(
        problem="Design user authentication system",
        constraints=["Secure", "User-friendly", "Scalable"],
        context={"users": 10000, "compliance": "GDPR"},
    )

    print(f"\n{result.get_summary()}")

    print("\n" + "-" * 80)
    print("BEST SOLUTION PATH:")
    for i, node in enumerate(result.best_path):
        indent = "  " * node.depth
        marker = "✓" if node.complete else "→"
        print(f"{indent}{marker} {node.thought} (value={node.value:.2f})")

    print("\n" + "-" * 80)
    print("EXPLORATION STATISTICS:")
    stats = result.exploration_stats
    print(f"  Nodes Explored: {stats['nodes_explored']}")
    print(f"  Max Depth Reached: {stats['max_depth_reached']}")
    print(f"  Solutions Found: {stats['solutions_found']}")
    print(f"  Complete Solutions: {stats['complete_solutions']}")
    print(f"  Best Value: {stats['best_value']:.2f}")

    print("=" * 80)


async def demo_api_design():
    """Plan API architecture."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 2: API Design Planning")
    print("=" * 80)

    planner = TreeOfThought(
        max_depth=3,
        beam_width=2,
    )

    result = await planner.plan(
        problem="Design RESTful API architecture",
        constraints=["RESTful", "Scalable", "Well-documented"],
        context={"endpoints": 50, "traffic": "high"},
    )

    print(f"\n{result.get_summary()}")

    print("\n" + "-" * 80)
    print("ALL SOLUTIONS:")
    for i, solution_path in enumerate(result.all_solutions, 1):
        print(f"\nSolution {i} (value={solution_path[-1].value:.2f}):")
        path_text = " → ".join(node.thought for node in solution_path)
        print(f"  {path_text}")

    if not result.all_solutions:
        print("\n(No complete solutions found - showing best partial path)")
        print(f"\nBest partial: {result.get_best_solution()}")

    print("=" * 80)


async def demo_database_selection():
    """Plan database architecture."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 3: Database Selection Planning")
    print("=" * 80)

    planner = TreeOfThought(
        max_depth=5,
        beam_width=3,
    )

    result = await planner.plan(
        problem="Choose database architecture for social network",
        constraints=["Scalable", "Fast reads", "ACID compliance"],
        context={"data_size": "petabytes", "read_write_ratio": "90:10"},
    )

    print(f"\n{result.get_summary()}")

    print("\n" + "-" * 80)
    print("REASONING TRACE (first 15 lines):")
    for trace in result.reasoning_trace[:15]:
        print(f"  {trace}")

    print("\n" + "-" * 80)
    print(f"BEST SOLUTION: {result.get_best_solution()}")

    print("=" * 80)


async def demo_tree_visualization():
    """Visualize solution tree."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 4: Tree Visualization")
    print("=" * 80)

    planner = TreeOfThought(
        max_depth=3,
        beam_width=2,
    )

    result = await planner.plan(
        problem="Optimize system performance",
        constraints=["Fast", "Resource-efficient"],
    )

    print("\nTREE STRUCTURE:")
    print(planner.visualize_tree(result, max_width=70))

    print("=" * 80)


async def demo_planning_statistics():
    """Show planning statistics."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 5: Planning Statistics")
    print("=" * 80)

    planner = TreeOfThought(max_depth=3, beam_width=2)

    # Run multiple planning sessions
    await planner.plan("Design authentication", ["Secure"])
    await planner.plan("Design API", ["Scalable"])
    await planner.plan("Choose database", ["Fast"])

    stats = planner.get_statistics()

    print("\nPLANNING STATISTICS:")
    print(f"  Total Plans: {stats['total_plans']}")
    print(f"  Avg Nodes Explored: {stats['avg_nodes_explored']:.1f}")
    print(f"  Avg Solutions Found: {stats['avg_solutions_found']:.1f}")
    print(f"  Avg Best Value: {stats['avg_best_value']:.2f}")
    print(f"  Completion Rate: {stats['completion_rate']:.1%}")

    print("\n" + "-" * 80)
    print("PLANNING HISTORY:")
    for i, result in enumerate(planner.get_planning_history(), 1):
        print(f"\n{i}. {result.problem}")
        print(f"   Nodes: {result.exploration_stats['nodes_explored']}")
        print(f"   Solutions: {result.exploration_stats['solutions_found']}")
        print(f"   Best Value: {result.exploration_stats['best_value']:.2f}")

    print("=" * 80)


async def demo_custom_evaluation():
    """Demo with custom evaluation function."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 6: Custom Evaluation Function")
    print("=" * 80)

    async def security_focused_eval(node, problem, constraints, context):
        """Custom evaluator that prioritizes security."""
        score = 0.5

        thought_lower = node.thought.lower()

        # High bonus for security keywords
        if any(word in thought_lower for word in ["encrypt", "secure", "auth", "validate"]):
            score += 0.3

        # Bonus for depth
        score += min(0.2, node.depth * 0.05)

        return min(1.0, score)

    planner = TreeOfThought(
        max_depth=3,
        beam_width=2,
        evaluation_fn=security_focused_eval,
    )

    result = await planner.plan(
        problem="Design secure file storage",
        constraints=["Secure", "Encrypted"],
    )

    print(f"\n{result.get_summary()}")
    print(f"\nBest solution (security-focused): {result.get_best_solution()}")

    print("=" * 80)


async def main():
    """Run all demos."""
    await demo_authentication_planning()
    await demo_api_design()
    await demo_database_selection()
    await demo_tree_visualization()
    await demo_planning_statistics()
    await demo_custom_evaluation()

    print("\n" * 2)
    print("=" * 80)
    print("DEMOS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
