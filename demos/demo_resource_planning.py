"""
Resource-Constrained Project Planning Demo

Scenario: Software project with budget and deadline constraints.
Demonstrates:
- Consumable resources (budget/money)
- Reusable resources (developers, equipment)
- Producible resources (features, code)
- Budget and deadline constraints
- Resource feasibility checking
- Plan repair when constraints violated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.causal import CausalDAG, CausalNode, CausalEdge
from HoloLoom.planning.planner import HierarchicalPlanner, Goal, Plan, Action, ActionType
from HoloLoom.planning.resources import (
    Resource, ResourceType, ResourceRequirement,
    ResourceTracker, ResourceAwarePlanner
)

# ============================================================================
# Setup
# ============================================================================

def create_project_dag():
    """Create causal model for software project."""
    dag = CausalDAG()

    # Software project causal structure
    # design → code → test → deploy → project_complete
    dag.add_node(CausalNode("design"))
    dag.add_node(CausalNode("code"))
    dag.add_node(CausalNode("test"))
    dag.add_node(CausalNode("deploy"))
    dag.add_node(CausalNode("project_complete"))

    dag.add_edge(CausalEdge("design", "code", strength=0.9))
    dag.add_edge(CausalEdge("code", "test", strength=0.8))
    dag.add_edge(CausalEdge("test", "deploy", strength=0.85))
    dag.add_edge(CausalEdge("deploy", "project_complete", strength=0.95))

    return dag


def create_project_resources():
    """Create project resources with constraints."""

    resources = [
        # Consumable: Budget (money)
        Resource(
            name="budget",
            resource_type=ResourceType.CONSUMABLE,
            initial_amount=10000.0,  # $10,000 total budget
            cost_per_unit=1.0  # $1 per dollar (identity)
        ),

        # Reusable: Developers (can work on one task at a time)
        Resource(
            name="developers",
            resource_type=ResourceType.REUSABLE,
            initial_amount=3.0,  # 3 developers available
            capacity=3.0
        ),

        # Reusable: Test servers
        Resource(
            name="test_servers",
            resource_type=ResourceType.REUSABLE,
            initial_amount=2.0,
            capacity=2.0
        ),

        # Producible: Features completed
        Resource(
            name="features",
            resource_type=ResourceType.PRODUCIBLE,
            initial_amount=0.0
        ),

        # Consumable: Developer hours
        Resource(
            name="dev_hours",
            resource_type=ResourceType.CONSUMABLE,
            initial_amount=480.0,  # 3 devs × 40 hours/week × 4 weeks
            cost_per_unit=50.0  # $50/hour developer cost
        ),
    ]

    return resources


def create_project_plan_with_resources():
    """Create a project plan with resource requirements."""

    # Define actions with resource requirements
    actions = []

    # Action 1: Design phase
    design_action = Action(
        action_type=ActionType.INTERVENE,
        parameters={"variable": "design", "value": 1},
        description="Design phase"
    )
    design_action.variable = "design"  # For display
    design_action.duration = 40.0  # 40 hours (1 week)
    design_action.requirements = [
        ResourceRequirement("developers", 2.0, when="start"),  # Need 2 devs
        ResourceRequirement("developers", 2.0, when="end"),     # Release 2 devs
        ResourceRequirement("dev_hours", 80.0, when="duration"), # 2 devs × 40 hours
        ResourceRequirement("budget", 4000.0, when="duration"),  # 80 hours × $50
    ]
    actions.append(design_action)

    # Action 2: Code phase
    code_action = Action(
        action_type=ActionType.INTERVENE,
        parameters={"variable": "code", "value": 1},
        description="Code phase"
    )
    code_action.variable = "code"  # For display
    code_action.duration = 80.0  # 80 hours (2 weeks)
    code_action.requirements = [
        ResourceRequirement("developers", 3.0, when="start"),   # All 3 devs
        ResourceRequirement("developers", 3.0, when="end"),
        ResourceRequirement("dev_hours", 240.0, when="duration"), # 3 devs × 80 hours
        ResourceRequirement("budget", 12000.0, when="duration"),  # 240 hours × $50
        ResourceRequirement("features", 1.0, when="end"),       # Produce 1 feature
    ]
    actions.append(code_action)

    # Action 3: Test phase
    test_action = Action(
        action_type=ActionType.INTERVENE,
        parameters={"variable": "test", "value": 1},
        description="Test phase"
    )
    test_action.variable = "test"  # For display
    test_action.duration = 40.0  # 40 hours (1 week)
    test_action.requirements = [
        ResourceRequirement("developers", 1.0, when="start"),
        ResourceRequirement("developers", 1.0, when="end"),
        ResourceRequirement("test_servers", 2.0, when="start"),  # Need both servers
        ResourceRequirement("test_servers", 2.0, when="end"),
        ResourceRequirement("dev_hours", 40.0, when="duration"),
        ResourceRequirement("budget", 2000.0, when="duration"),
    ]
    actions.append(test_action)

    # Action 4: Deploy phase
    deploy_action = Action(
        action_type=ActionType.INTERVENE,
        parameters={"variable": "deploy", "value": 1},
        description="Deploy phase"
    )
    deploy_action.variable = "deploy"  # For display
    deploy_action.duration = 8.0  # 8 hours (1 day)
    deploy_action.requirements = [
        ResourceRequirement("developers", 1.0, when="start"),
        ResourceRequirement("developers", 1.0, when="end"),
        ResourceRequirement("dev_hours", 8.0, when="duration"),
        ResourceRequirement("budget", 400.0, when="duration"),
    ]
    actions.append(deploy_action)

    # Create plan
    goal = Goal(
        desired_state={"project_complete": 1},
        description="Complete software project"
    )
    plan = Plan(actions=actions, goal=goal, expected_cost=1.0)

    return plan


# ============================================================================
# Demo
# ============================================================================

def main():
    """Run resource planning demo."""

    print("=" * 80)
    print("RESOURCE-CONSTRAINED PROJECT PLANNING DEMO".center(80))
    print("=" * 80)
    print()

    print("Scenario: Software project with budget and deadline constraints")
    print("-" * 80)
    print()

    # Setup
    print("1. PROJECT SETUP")
    print("-" * 80)

    dag = create_project_dag()
    print(f"✓ Created project causal model ({len(dag.nodes)} phases)")
    print("   Phases: design → code → test → deploy → complete")
    print()

    resources = create_project_resources()
    print(f"✓ Created {len(resources)} resource types:")
    for resource in resources:
        print(f"   - {resource.name}: {resource.initial_amount:.0f} "
              f"({resource.resource_type.value})")
    print()

    # Create planner
    base_planner = HierarchicalPlanner(dag)
    constraints = {
        "budget": 15000.0,    # $15,000 budget limit
        "deadline": 160.0,    # 160 hours (4 weeks) deadline
    }

    resource_planner = ResourceAwarePlanner(
        base_planner=base_planner,
        resources=resources,
        constraints=constraints
    )
    print(f"✓ Created resource-aware planner")
    print(f"   Budget limit: ${constraints['budget']:.0f}")
    print(f"   Deadline: {constraints['deadline']:.0f} hours")
    print()

    # Create plan with resources
    print("2. CREATE PROJECT PLAN")
    print("-" * 80)

    plan = create_project_plan_with_resources()
    print(f"✓ Created plan with {len(plan.actions)} phases:")
    print()

    for i, action in enumerate(plan.actions, 1):
        duration = getattr(action, 'duration', 0)
        reqs = getattr(action, 'requirements', [])

        print(f"   Phase {i}: {action.variable}")
        print(f"      Duration: {duration:.0f} hours")
        print(f"      Requirements:")
        for req in reqs:
            print(f"         - {req.resource}: {req.amount:.0f} ({req.when})")
        print()

    # Check feasibility
    print("3. RESOURCE FEASIBILITY CHECK")
    print("-" * 80)

    tracker = resource_planner.tracker
    violations = tracker.find_violations(plan, constraints)

    if not violations:
        print("✓ Plan is resource-feasible!")
        print()
    else:
        print(f"✗ Plan has {len(violations)} resource violations:")
        print()
        for v in violations:
            print(f"   {v}")
        print()

    # Resource timeline
    print("4. RESOURCE TIMELINE SIMULATION")
    print("-" * 80)

    timeline = tracker.simulate_plan(plan)
    print(f"Simulated {len(timeline)} time steps:")
    print()

    for i, state in enumerate(timeline):
        print(f"   t={state.time:.0f}h:")
        print(f"      Budget: ${state.available.get('budget', 0):.0f} available")
        print(f"      Developers: {state.allocated.get('developers', 0):.0f} working, "
              f"{state.available.get('developers', 0):.0f} free")
        print(f"      Features: {state.produced.get('features', 0):.0f} completed")
        print()

    # Cost analysis
    print("5. COST & TIME ANALYSIS")
    print("-" * 80)

    total_cost = tracker.calculate_total_cost(plan)
    total_time = tracker.calculate_total_time(plan)

    print(f"Total cost: ${total_cost:.0f}")
    print(f"Budget limit: ${constraints['budget']:.0f}")
    print(f"Budget status: {'✓ UNDER' if total_cost <= constraints['budget'] else '✗ OVER'} budget")
    print()

    print(f"Total time: {total_time:.0f} hours ({total_time/40:.1f} weeks)")
    print(f"Deadline: {constraints['deadline']:.0f} hours ({constraints['deadline']/40:.1f} weeks)")
    print(f"Schedule status: {'✓ ON TIME' if total_time <= constraints['deadline'] else '✗ LATE'}")
    print()

    # Resource usage
    print("6. RESOURCE USAGE SUMMARY")
    print("-" * 80)

    usage = tracker.compute_resource_usage(plan)
    for resource_name, stats in usage.items():
        print(f"{resource_name}:")
        print(f"   Consumed: {stats['consumed']:.0f}")
        print(f"   Peak allocated: {stats['peak_allocated']:.0f}")
        print(f"   Produced: {stats['produced']:.0f}")
        print(f"   Net change: {stats['net_change']:.0f}")
        print()

    # Summary
    print("7. PROJECT SUMMARY")
    print("=" * 80)
    print()

    print(f"Project: {len(plan.actions)} phases, {total_time:.0f} hours, ${total_cost:.0f}")
    print()

    if not violations:
        print("✅ PROJECT IS FEASIBLE")
        print(f"   - Budget: ${total_cost:.0f} / ${constraints['budget']:.0f} "
              f"({total_cost/constraints['budget']*100:.0f}% used)")
        print(f"   - Time: {total_time:.0f}h / {constraints['deadline']:.0f}h "
              f"({total_time/constraints['deadline']*100:.0f}% used)")
        print(f"   - Features delivered: {usage['features']['produced']:.0f}")
    else:
        print("❌ PROJECT HAS RESOURCE VIOLATIONS")
        for v in violations:
            print(f"   - {v}")

    print()

    print("=" * 80)
    print("RESOURCE PLANNING DEMONSTRATION COMPLETE".center(80))
    print("=" * 80)
    print()

    print("What We Demonstrated:")
    print()
    print("1. ✅ Resource Types")
    print("   - Consumable: Budget, dev hours (used up)")
    print("   - Reusable: Developers, servers (held temporarily)")
    print("   - Producible: Features (generated)")
    print()
    print("2. ✅ Resource Constraints")
    print("   - Budget limits ($15,000)")
    print("   - Deadline constraints (160 hours)")
    print("   - Capacity constraints (3 developers max)")
    print()
    print("3. ✅ Feasibility Checking")
    print("   - Simulate resource usage over time")
    print("   - Detect violations (budget, deadline, capacity)")
    print("   - Report detailed violations")
    print()
    print("4. ✅ Resource Timeline")
    print("   - Track resource state at each time step")
    print("   - Show allocation and availability")
    print("   - Monitor production (features)")
    print()

    print("Research Alignment:")
    print("   - Ghallab et al. (2004): Resource-constrained planning")
    print("   - Coles et al. (2009): COLIN numeric planning")
    print("   - Fox & Long (2003): PDDL2.1 numeric constraints")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
