"""
Multi-Agent Warehouse Demo

Scenario: Warehouse with multiple robots picking and packing orders.
Demonstrates:
- Contract Net Protocol for task allocation
- Coalition formation for complex tasks
- Cooperative agent behavior
- Joint plan execution

Robots have different capabilities:
- Picker: Can pick items from shelves
- Packer: Can pack items into boxes
- Hybrid: Can both pick and pack (slower)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.causal import CausalDAG, CausalNode, CausalEdge
from HoloLoom.planning.multi_agent import (
    Agent, AgentType, MultiAgentCoordinator,
    NegotiationProtocol, Task, Capability, create_agent
)
from HoloLoom.planning.planner import Goal

# ============================================================================
# Setup
# ============================================================================

def create_warehouse_dag():
    """Create causal model for warehouse operations."""
    dag = CausalDAG()

    # Warehouse causal structure
    # pick_item → item_ready → pack_item → order_complete
    dag.add_node(CausalNode("pick_item"))
    dag.add_node(CausalNode("item_ready"))
    dag.add_node(CausalNode("pack_item"))
    dag.add_node(CausalNode("order_complete"))

    dag.add_edge(CausalEdge("pick_item", "item_ready", strength=0.9))
    dag.add_edge(CausalEdge("item_ready", "pack_item", strength=0.8))
    dag.add_edge(CausalEdge("pack_item", "order_complete", strength=0.95))

    return dag


def create_warehouse_agents(dag: CausalDAG):
    """Create warehouse robots with different capabilities."""

    agents = [
        # Picker robots (fast picking, can't pack)
        create_agent(
            agent_id="picker_1",
            agent_type=AgentType.COOPERATIVE,
            dag=dag,
            capabilities=["pick"],
            proficiency=0.9
        ),
        create_agent(
            agent_id="picker_2",
            agent_type=AgentType.COOPERATIVE,
            dag=dag,
            capabilities=["pick"],
            proficiency=0.85
        ),

        # Packer robots (fast packing, can't pick)
        create_agent(
            agent_id="packer_1",
            agent_type=AgentType.COOPERATIVE,
            dag=dag,
            capabilities=["pack"],
            proficiency=0.88
        ),

        # Hybrid robot (can do both, but slower)
        create_agent(
            agent_id="hybrid_1",
            agent_type=AgentType.COOPERATIVE,
            dag=dag,
            capabilities=["pick", "pack"],
            proficiency=0.7  # Jack of all trades
        ),
    ]

    return agents


def create_warehouse_tasks():
    """Create warehouse tasks (customer orders)."""

    tasks = [
        Task(
            task_id="order_A",
            goal=Goal(desired_state={"order_complete": 1},
                     description="Complete order A"),
            required_capabilities={"pick"},
            priority=2.0,  # High priority
            difficulty=1.0
        ),
        Task(
            task_id="order_B",
            goal=Goal(desired_state={"order_complete": 1},
                     description="Complete order B"),
            required_capabilities={"pack"},
            priority=1.0,  # Normal priority
            difficulty=0.8
        ),
        Task(
            task_id="order_C",
            goal=Goal(desired_state={"order_complete": 1},
                     description="Complete order C"),
            required_capabilities={"pick", "pack"},  # Needs both!
            priority=3.0,  # Urgent!
            difficulty=1.5
        ),
    ]

    return tasks


# ============================================================================
# Demo
# ============================================================================

def main():
    """Run warehouse demo."""

    print("=" * 80)
    print("MULTI-AGENT WAREHOUSE DEMO".center(80))
    print("=" * 80)
    print()

    print("Scenario: Warehouse with multiple robots fulfilling orders")
    print("-" * 80)
    print()

    # Setup
    print("1. SETUP")
    print("-" * 80)

    dag = create_warehouse_dag()
    print(f"✓ Created warehouse causal model ({len(dag.nodes)} nodes)")

    agents = create_warehouse_agents(dag)
    print(f"✓ Created {len(agents)} warehouse robots:")
    for agent in agents:
        caps = ", ".join(agent.capabilities.keys())
        print(f"   - {agent.agent_id}: [{caps}] (proficiency: "
              f"{list(agent.capabilities.values())[0].proficiency:.2f})")

    tasks = create_warehouse_tasks()
    print(f"✓ Created {len(tasks)} customer orders:")
    for task in tasks:
        caps = ", ".join(task.required_capabilities)
        print(f"   - {task.task_id}: Needs [{caps}], priority={task.priority:.1f}")
    print()

    # Initialize coordinator
    print("2. INITIALIZE COORDINATOR")
    print("-" * 80)

    coordinator = MultiAgentCoordinator(
        agents=agents,
        protocol=NegotiationProtocol.CONTRACT_NET
    )
    print(f"✓ Using {coordinator.protocol.value} protocol")
    print()

    # Task allocation
    print("3. TASK ALLOCATION (Contract Net Protocol)")
    print("-" * 80)
    print()

    current_state = {
        "pick_item": 0,
        "item_ready": 0,
        "pack_item": 0,
        "order_complete": 0
    }

    print("Broadcasting tasks to agents...")
    print()

    allocation = coordinator.allocate_tasks(tasks, current_state)

    print("\nAllocation Results:")
    print("-" * 80)
    for task_id, agent_id in allocation.items():
        task = next(t for t in tasks if t.task_id == task_id)
        agent = agents[[a.agent_id for a in agents].index(agent_id)]

        caps_needed = ", ".join(task.required_capabilities)
        caps_has = ", ".join(agent.capabilities.keys())

        print(f"✓ {task_id} → {agent_id}")
        print(f"   Needs: [{caps_needed}]")
        print(f"   Has: [{caps_has}]")
        print(f"   Priority: {task.priority:.1f}")
        print()

    # Agent commitments
    print("4. AGENT COMMITMENTS")
    print("-" * 80)
    for agent in agents:
        if agent.commitments:
            print(f"✓ {agent.agent_id} committed to:")
            for task_id, agreement in agent.commitments.items():
                print(f"   - {task_id} ({len(agreement.joint_plan.actions)} actions)")
        else:
            print(f"  {agent.agent_id}: No commitments")
    print()

    # Coalition formation for complex task
    print("5. COALITION FORMATION")
    print("-" * 80)

    # Find task that needs multiple capabilities
    complex_task = next(t for t in tasks if len(t.required_capabilities) > 1)
    print(f"Task '{complex_task.task_id}' requires multiple capabilities: "
          f"{', '.join(complex_task.required_capabilities)}")
    print()

    coalition = coordinator.form_coalition(complex_task, current_state)

    if coalition:
        print(f"✓ Coalition formed!")
        print(f"   Members: {', '.join(coalition.members)}")
        print(f"   Coalition value: {coalition.value:.2f}")
        print(f"   Stable: {coalition.is_stable}")
        print()

        # Show member capabilities
        print("   Member capabilities:")
        for member_id in coalition.members:
            agent = next(a for a in agents if a.agent_id == member_id)
            caps = ", ".join(agent.capabilities.keys())
            prof = list(agent.capabilities.values())[0].proficiency
            print(f"   - {member_id}: [{caps}] (proficiency: {prof:.2f})")
    else:
        print("✗ No stable coalition could be formed")
    print()

    # Summary
    print("6. SUMMARY")
    print("=" * 80)
    print()

    print(f"Total tasks: {len(tasks)}")
    print(f"Tasks allocated: {len(allocation)}")
    print(f"Tasks unallocated: {len(tasks) - len(allocation)}")
    print()

    print("Agent Utilization:")
    for agent in agents:
        count = sum(1 for task_id, agent_id in allocation.items()
                   if agent_id == agent.agent_id)
        util = (count / len(tasks)) * 100 if tasks else 0
        print(f"   {agent.agent_id}: {count} tasks ({util:.0f}%)")
    print()

    print("Protocol Effectiveness:")
    print(f"   ✓ Contract Net enabled efficient bidding")
    print(f"   ✓ Agents selected based on capabilities & cost")
    print(f"   ✓ Coalitions formed for complex tasks")
    print()

    print("=" * 80)
    print("MULTI-AGENT COORDINATION DEMONSTRATION COMPLETE".center(80))
    print("=" * 80)
    print()

    print("What We Demonstrated:")
    print()
    print("1. ✅ Contract Net Protocol")
    print("   - Manager announces tasks")
    print("   - Agents submit bids (proposals)")
    print("   - Best bid wins (cost/confidence tradeoff)")
    print()
    print("2. ✅ Agent Specialization")
    print("   - Picker robots: Fast at picking, can't pack")
    print("   - Packer robots: Fast at packing, can't pick")
    print("   - Hybrid robots: Can do both, but slower")
    print()
    print("3. ✅ Coalition Formation")
    print("   - Complex tasks need multiple capabilities")
    print("   - Agents form coalitions (teams)")
    print("   - Shapley value calculation")
    print("   - Stability checking (no agent wants to leave)")
    print()
    print("4. ✅ Cooperative Behavior")
    print("   - All agents maximize joint utility")
    print("   - Efficient task allocation")
    print("   - Resource sharing")
    print()

    print("Research Alignment:")
    print("   - Smith (1980): Contract Net Protocol")
    print("   - Wooldridge (2009): Multiagent Systems")
    print("   - Rahwan (2009): Coalition Formation")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
