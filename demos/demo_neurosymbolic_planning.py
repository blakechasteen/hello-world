"""
Neurosymbolic Planning Demo

Shows Layer 1 (Causal) + Layer 2 (Planning) integration.

Neurosymbolic Architecture:
- Symbolic: Causal DAG structure + HTN planning rules
- Neural: Can integrate learned models (future enhancement)

Key Innovation: Planning uses causal knowledge to find HOW to achieve goals!
"""

import sys
sys.path.insert(0, '.')

from HoloLoom.causal import CausalDAG, CausalNode, CausalEdge, NodeType
from HoloLoom.planning import HierarchicalPlanner, Goal


def print_section(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def main():
    print_section("NEUROSYMBOLIC PLANNING DEMO")

    print("The Power of Integration:")
    print("-" * 80)
    print("Layer 1 (Causal) knows: treatment → recovery")
    print("Layer 2 (Planning) uses this to achieve goals!")
    print("")

    # ========================================================================
    # Build Causal Knowledge (Layer 1 - Symbolic)
    # ========================================================================

    print_section("LAYER 1: Causal Knowledge (Symbolic)")

    dag = CausalDAG()

    # Add nodes
    dag.add_node(CausalNode("age", NodeType.OBSERVABLE))
    dag.add_node(CausalNode("treatment", NodeType.OBSERVABLE))
    dag.add_node(CausalNode("recovery", NodeType.OBSERVABLE))

    # Add causal edges (domain knowledge)
    dag.add_edge(CausalEdge(
        "age", "treatment",
        strength=0.3,
        mechanism="Elderly more likely to get treatment"
    ))

    dag.add_edge(CausalEdge(
        "age", "recovery",
        strength=0.2,
        mechanism="Age affects recovery rate"
    ))

    dag.add_edge(CausalEdge(
        "treatment", "recovery",
        strength=0.6,
        mechanism="Treatment improves recovery"
    ))

    print("Causal Model:")
    for (src, tgt), edge in dag.edges.items():
        print(f"  {src} → {tgt} (strength={edge.strength:.2f})")
    print("")

    # ========================================================================
    # Initialize Planner (Layer 2)
    # ========================================================================

    print_section("LAYER 2: Hierarchical Planning")

    planner = HierarchicalPlanner(dag)
    print(f"✓ Initialized {planner}")
    print("")

    # ========================================================================
    # Planning Scenario 1: Achieve Recovery
    # ========================================================================

    print_section("SCENARIO 1: Achieve Patient Recovery")

    current_state = {
        "age": 50,
        "treatment": 0,  # Not treated
        "recovery": 0    # Not recovered
    }

    goal = Goal(
        desired_state={"recovery": 1},
        description="Make patient recover"
    )

    print(f"Current State: {current_state}")
    print(f"Goal: {goal}")
    print("")

    # Generate plan
    plan = planner.plan(goal, current_state)

    if plan:
        print("✓ Plan Generated!")
        print(f"  Cost: {plan.expected_cost:.1f}")
        print(f"  Steps: {len(plan.actions)}")
        print("")

        print(plan.explanation)
        print("")

        print("Execution Trace:")
        for i, action in enumerate(plan.actions, 1):
            print(f"  Step {i}: {action}")
    else:
        print("✗ No plan found!")

    print("")

    # ========================================================================
    # Scenario 2: Goal Already Satisfied
    # ========================================================================

    print_section("SCENARIO 2: Goal Already Achieved")

    already_recovered = {
        "age": 30,
        "treatment": 1,
        "recovery": 1  # Already recovered!
    }

    plan2 = planner.plan(goal, already_recovered)

    if plan2:
        print(f"Plan: {plan2}")
        print(f"Actions: {len(plan2.actions)}")
        print(f"Explanation: {plan2.explanation}")
    print("")

    # ========================================================================
    # Scenario 3: Multiple Goals
    # ========================================================================

    print_section("SCENARIO 3: Multiple Variables to Change")

    complex_goal = Goal(
        desired_state={"treatment": 1, "recovery": 1},
        description="Apply treatment AND achieve recovery"
    )

    initial_state = {
        "age": 40,
        "treatment": 0,
        "recovery": 0
    }

    plan3 = planner.plan(complex_goal, initial_state)

    if plan3:
        print(f"Plan for complex goal: {plan3}")
        print("")
        print(plan3.explanation)
    print("")

    # ========================================================================
    # Show Causal Chain Finding
    # ========================================================================

    print_section("UNDER THE HOOD: Causal Chain Finding")

    from HoloLoom.planning import CausalChainFinder

    finder = CausalChainFinder(dag)

    print("Finding all paths to 'recovery':")
    paths = finder.find_paths_to_goal("recovery")

    for i, path in enumerate(paths, 1):
        print(f"\n  Path {i}: {path}")
        print(f"    {finder.explain_path(path)}")

    print("")

    # ========================================================================
    # Summary
    # ========================================================================

    print_section("NEUROSYMBOLIC INTEGRATION SUMMARY")

    print("What We Demonstrated:")
    print("")

    print("1. ✅ Symbolic Causal Knowledge (Layer 1)")
    print("   - Explicit causal DAG")
    print("   - Interpretable structure")
    print("   - Domain expertise encoded")
    print("")

    print("2. ✅ Symbolic Planning Rules (Layer 2)")
    print("   - HTN decomposition")
    print("   - Goal → Subgoal → Action")
    print("   - Causal chain finding")
    print("")

    print("3. ✅ Integration (Neurosymbolic)")
    print("   - Planning USES causal knowledge")
    print("   - Finds paths via causal reasoning")
    print("   - Verifiable, explainable plans")
    print("")

    print("Future: Add Neural Components")
    print("  - Learned action models")
    print("  - Value functions for action selection")
    print("  - Neural-symbolic hybrid complete!")
    print("")

    print("=" * 80)
    print("✅ Neurosymbolic Planning Demo Complete!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
