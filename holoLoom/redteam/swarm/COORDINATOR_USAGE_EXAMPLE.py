"""SwarmCoordinator Usage Examples.

Comprehensive examples demonstrating different use cases of the SwarmCoordinator.

Examples:
1. Basic campaign execution
2. Custom task distribution
3. Real-time monitoring
4. Agent performance analysis
5. Multi-target campaigns
6. Error handling and recovery
"""

import asyncio
import time
from typing import List

from coordinator import SwarmCoordinator
from communication import MessageBus
from protocols import AgentTask, MessagePriority


# ============================================================================
# Example 1: Basic Campaign Execution
# ============================================================================


async def example_basic_campaign():
    """Example 1: Run a basic campaign against a target.

    This is the simplest way to use the coordinator - just initialize,
    start, run a campaign, and stop.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Campaign Execution")
    print("=" * 70)

    # Create message bus
    bus = MessageBus(max_queue_size=10000)

    # Create and start coordinator
    coordinator = SwarmCoordinator(
        message_bus=bus,
        num_scouts=2,
        num_attackers=3,
        num_exploiters=1,
    )

    await coordinator.start()
    print(f"✓ Coordinator started with {len(coordinator._all_agents)} agents")

    try:
        # Run campaign
        print("\nRunning campaign against target...")
        result = await coordinator.run_campaign(
            target="192.168.1.100",
            duration_seconds=60,
        )

        # Display results
        print(f"\n✓ Campaign complete!")
        print(f"  Target: {result.target}")
        print(f"  Duration: {result.total_duration_ms:.0f}ms")
        print(f"  Vulnerabilities found: {len(result.vulnerabilities_found)}")
        print(f"  Exploits successful: {len(result.exploits_successful)}")
        print(f"  Tasks completed: {result.metrics.tasks_completed}")
        print(f"  Discoveries: {result.metrics.discoveries}")

    finally:
        await coordinator.stop()
        print("✓ Coordinator stopped")


# ============================================================================
# Example 2: Custom Task Distribution
# ============================================================================


async def example_custom_task_distribution():
    """Example 2: Manually distribute tasks and collect results.

    Demonstrates finer-grained control over task distribution and
    result collection.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Task Distribution")
    print("=" * 70)

    bus = MessageBus()
    coordinator = SwarmCoordinator(message_bus=bus)

    async with coordinator:
        print(f"✓ Coordinator started")

        # Create custom tasks
        tasks = [
            AgentTask(
                task_type="probe_surface",
                target="192.168.1.100",
                parameters={
                    "scan_type": "network",
                    "timeout": 30,
                },
                priority=MessagePriority.HIGH,
                timeout_seconds=45,
            ),
            AgentTask(
                task_type="probe_surface",
                target="192.168.1.100",
                parameters={
                    "scan_type": "vulnerability",
                    "database": "local",
                },
                priority=MessagePriority.NORMAL,
                timeout_seconds=45,
            ),
        ]

        print(f"\nDistributing {len(tasks)} custom tasks...")

        # Distribute tasks
        assigned_agents = []
        for task in tasks:
            agent_id = await coordinator.distribute_task(task)
            assigned_agents.append(agent_id)
            print(f"  ✓ Task {task.task_id[:8]} assigned to {agent_id}")

        # Collect results
        print("\nCollecting results...")
        results = await coordinator.collect_results(timeout_ms=30000)

        print(f"\n✓ Collected {len(results)} results")
        for result in results:
            status = "✓" if result.success else "✗"
            print(
                f"  {status} Agent {result.agent_id}: "
                f"{len(result.discoveries)} discoveries"
            )


# ============================================================================
# Example 3: Real-Time Monitoring
# ============================================================================


async def example_real_time_monitoring():
    """Example 3: Monitor campaign progress in real-time.

    Shows how to track metrics and agent states while campaign is running.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Real-Time Monitoring")
    print("=" * 70)

    bus = MessageBus()
    coordinator = SwarmCoordinator(message_bus=bus)

    async with coordinator:
        print("✓ Coordinator started")

        # Run campaign in background
        campaign_task = asyncio.create_task(
            coordinator.run_campaign("192.168.1.100", duration_seconds=30)
        )

        # Monitor progress
        print("\nMonitoring campaign progress...")
        iteration = 0

        while not campaign_task.done():
            iteration += 1
            metrics = coordinator.get_metrics()
            states = coordinator.get_agent_states()

            # Count agent states
            active_count = sum(1 for s in states.values() if s.value == "active")

            print(
                f"\n[Iteration {iteration}]"
                f" Tasks: {metrics.tasks_completed + metrics.tasks_failed} | "
                f"Active: {active_count}/{len(coordinator._all_agents)} | "
                f"Discoveries: {metrics.discoveries}"
            )

            # Show agent state breakdown
            executing = sum(1 for s in states.values() if s.value == "executing")
            if executing > 0:
                print(f"  Agents executing: {executing}")

            await asyncio.sleep(2)

        # Get final results
        result = await campaign_task
        print(f"\n✓ Campaign complete: {result.metrics.discoveries} discoveries")


# ============================================================================
# Example 4: Agent Performance Analysis
# ============================================================================


async def example_agent_performance_analysis():
    """Example 4: Analyze agent performance and contributions.

    Demonstrates how to track individual agent performance and
    calculate contribution scores.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Agent Performance Analysis")
    print("=" * 70)

    bus = MessageBus()
    coordinator = SwarmCoordinator(message_bus=bus)

    async with coordinator:
        print("✓ Coordinator started\n")

        # Simulate agent performance
        print("Simulating agent tasks...")
        for agent in coordinator._all_agents:
            # Simulate 5 tasks per agent
            successes = 0
            for i in range(5):
                success = i % 2 == 0  # Alternate success/failure
                if success:
                    successes += 1
                coordinator._update_agent_prior(agent.agent_id, success)

            print(f"  {agent.agent_id}: {successes}/5 successful")

        # Calculate contributions
        print("\nAgent contributions:")
        contributions = coordinator._calculate_agent_contributions()

        for agent_id, score in sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {agent_id}: {score:.2%}")

        # Calculate total for scouts, attackers, exploiters
        scout_scores = [
            contributions[a.agent_id] for a in coordinator._scouts
        ]
        attack_scores = [
            contributions[a.agent_id] for a in coordinator._attackers
        ]
        exploit_scores = [
            contributions[a.agent_id] for a in coordinator._exploiters
        ]

        print(f"\nTeam averages:")
        print(
            f"  Scouts: {sum(scout_scores) / len(scout_scores):.2%} "
            f"({len(coordinator._scouts)} agents)"
        )
        print(
            f"  Attackers: {sum(attack_scores) / len(attack_scores):.2%} "
            f"({len(coordinator._attackers)} agents)"
        )
        print(
            f"  Exploiters: {sum(exploit_scores) / len(exploit_scores):.2%} "
            f"({len(coordinator._exploiters)} agents)"
        )


# ============================================================================
# Example 5: Multi-Target Campaigns
# ============================================================================


async def example_multi_target_campaigns():
    """Example 5: Run sequential campaigns against multiple targets.

    Shows how to run multiple campaigns with the same coordinator.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Target Campaigns")
    print("=" * 70)

    bus = MessageBus()
    coordinator = SwarmCoordinator(message_bus=bus)

    targets = ["192.168.1.100", "192.168.1.101", "192.168.1.102"]

    async with coordinator:
        print("✓ Coordinator started\n")

        results_by_target = {}

        for target in targets:
            print(f"Running campaign against {target}...")
            result = await coordinator.run_campaign(
                target=target,
                duration_seconds=20,
            )

            results_by_target[target] = result
            print(
                f"  ✓ {len(result.vulnerabilities_found)} vulnerabilities found\n"
            )

        # Summary
        print("Campaign Summary:")
        print("-" * 70)
        for target, result in results_by_target.items():
            print(
                f"  {target}: "
                f"{len(result.vulnerabilities_found)} vulns, "
                f"{result.total_duration_ms:.0f}ms"
            )


# ============================================================================
# Example 6: Error Handling and Recovery
# ============================================================================


async def example_error_handling():
    """Example 6: Handle errors and test recovery.

    Demonstrates error handling and graceful degradation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Error Handling and Recovery")
    print("=" * 70)

    bus = MessageBus()
    coordinator = SwarmCoordinator(message_bus=bus)

    try:
        # Try to run campaign without starting coordinator
        print("Attempting campaign without starting coordinator...")
        await coordinator.run_campaign("localhost")
    except RuntimeError as e:
        print(f"✓ Caught expected error: {e}\n")

    async with coordinator:
        # Test double start (should fail)
        try:
            await coordinator.start()
        except RuntimeError as e:
            print(f"✓ Caught expected error on double start: {e}\n")

        # Test that coordinator recovers and continues working
        print("Coordinator still functional after error...")

        task = AgentTask(
            task_type="probe_surface",
            target="localhost",
            parameters={},
        )

        agent_id = await coordinator.distribute_task(task)
        print(f"✓ Successfully distributed task to {agent_id}")


# ============================================================================
# Example 7: Advanced Configuration
# ============================================================================


async def example_advanced_configuration():
    """Example 7: Advanced coordinator configuration.

    Shows tuning coordinator for different workloads.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Advanced Configuration")
    print("=" * 70)

    # Heavy reconnaissance focus
    print("\nConfiguration 1: Heavy Reconnaissance (5 scouts)")
    bus = MessageBus(max_queue_size=20000)  # Large queue for many scouts
    coordinator = SwarmCoordinator(
        message_bus=bus,
        num_scouts=5,  # Many scouts for parallel probing
        num_attackers=2,  # Few attackers (focus on discovery)
        num_exploiters=0,  # No exploiters yet
    )

    await coordinator.start()
    print(f"✓ Started with {len(coordinator._scouts)} scouts")
    await coordinator.stop()

    # Aggressive exploitation focus
    print("\nConfiguration 2: Aggressive Exploitation (3 attackers, 2 exploiters)")
    coordinator = SwarmCoordinator(
        message_bus=MessageBus(),
        num_scouts=1,  # Minimal scouts (assume we know targets)
        num_attackers=3,  # Many attackers (parallel exploitation)
        num_exploiters=2,  # Many exploiters (escalation)
    )

    await coordinator.start()
    print(f"✓ Started with {len(coordinator._attackers)} attackers, "
          f"{len(coordinator._exploiters)} exploiters")
    await coordinator.stop()

    # Balanced approach
    print("\nConfiguration 3: Balanced Approach")
    coordinator = SwarmCoordinator(
        message_bus=MessageBus(),
        num_scouts=2,
        num_attackers=3,
        num_exploiters=1,
    )

    await coordinator.start()
    print(f"✓ Started balanced configuration")
    await coordinator.stop()


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SwarmCoordinator Usage Examples")
    print("=" * 70)

    try:
        await example_basic_campaign()
        await example_custom_task_distribution()
        await example_real_time_monitoring()
        await example_agent_performance_analysis()
        await example_multi_target_campaigns()
        await example_error_handling()
        await example_advanced_configuration()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
