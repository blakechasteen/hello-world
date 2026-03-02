"""Integration Example: Using the Three Specialized Swarm Agents

CARTS Phase 4: Multi-Agent Red Team Swarm
Date: 2025-12-05

This example demonstrates how to use the three specialized agents together
to orchestrate a complete red team operation:

1. ScoutAgent: Performs reconnaissance to find attack surfaces
2. AttackAgent: Executes attacks using Thompson Sampling for learning
3. ExploitAgent: Escalates and moves laterally within the network

Workflow:
    Scout discovers endpoints → Attacker executes attacks → Exploiter escalates access
"""

import asyncio
import logging
from typing import List, Dict, Any

# These imports assume the redteam module is in PYTHONPATH
# Adjust import paths as needed for your environment
# from hololoom.redteam.swarm.communication import MessageBus
# from hololoom.redteam.swarm.scout_agent import ScoutAgent
# from hololoom.redteam.swarm.attack_agent import AttackAgent
# from hololoom.redteam.swarm.exploit_agent import ExploitAgent
# from hololoom.redteam.swarm.protocols import (
#     AgentTask, AgentMessage, AgentRole, MessagePriority
# )
# from hololoom.redteam.bandit import RedTeamBandit


# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Scout-Only Reconnaissance
# ============================================================================


async def example_scout_reconnaissance():
    """Example 1: Use ScoutAgent for non-destructive reconnaissance.

    Steps:
    1. Create ScoutAgent
    2. Probe target surface
    3. Detect patterns in responses
    4. Generate and test fuzzing variations
    5. Report discoveries
    """
    logger.info("=" * 80)
    logger.info("Example 1: Scout-Only Reconnaissance")
    logger.info("=" * 80)

    # Note: This is pseudocode showing the pattern
    # Actual implementation would use real MessageBus and agents
    print("""
    # Create scout agent
    bus = MessageBus()
    scout = ScoutAgent("scout_alpha", bus)

    async with scout:
        # Task 1: Probe surface
        probe_task = AgentTask(
            task_type="probe_surface",
            target="http://target.com",
            parameters={"depth": 2, "timeout": 10}
        )
        probe_result = await scout.execute_task(probe_task)
        print(f"Discovered {len(probe_result.discoveries)} endpoints")

        # Task 2: Pattern detection
        responses = [...]  # Collected from probing
        pattern_task = AgentTask(
            task_type="pattern_detection",
            target="http://target.com",
            parameters={"responses": responses}
        )
        pattern_result = await scout.execute_task(pattern_task)

        # Task 3: Fuzzing
        fuzz_task = AgentTask(
            task_type="fuzz_input",
            target="http://target.com",
            parameters={"base_input": "admin", "variations": 20}
        )
        fuzz_result = await scout.execute_task(fuzz_task)

        # Report discoveries
        discoveries = scout.get_discoveries()
        for disc in discoveries:
            print(f"- {disc['discovery_type']}: {disc['value']}")

        # Get metrics
        metrics = scout.get_metrics()
        print(f"Endpoints probed: {metrics['scout_metrics']['endpoints_probed']}")
    """)


# ============================================================================
# Example 2: Scout + Attack Workflow
# ============================================================================


async def example_scout_and_attack():
    """Example 2: Scout discovers endpoints, Attacker tests them.

    Workflow:
    1. Scout probes endpoints
    2. Extract discovered endpoints
    3. Attacker executes attacks against endpoints
    4. Thompson Sampling learns best strategies
    5. Report successful attacks
    """
    logger.info("=" * 80)
    logger.info("Example 2: Scout + Attack Workflow")
    logger.info("=" * 80)

    print("""
    # Initialize agents
    bus = MessageBus()
    bandit = RedTeamBandit()
    scout = ScoutAgent("scout_1", bus)
    attacker = AttackAgent("attacker_1", bus, bandit)

    async with scout, attacker:
        # Step 1: Scout discovers endpoints
        probe_task = AgentTask(
            task_type="probe_surface",
            target="http://target.com",
            parameters={"depth": 2}
        )
        probe_result = await scout.execute_task(probe_task)

        # Step 2: Extract endpoints from discoveries
        endpoints = [d for d in scout.get_discoveries()
                     if d['discovery_type'] == 'endpoint']
        print(f"Found {len(endpoints)} endpoints to attack")

        # Step 3: Attacker tests each endpoint
        for endpoint in endpoints[:5]:  # Test first 5
            attack_task = AgentTask(
                task_type="execute_attack",
                target=endpoint['value'],
                parameters={
                    "strategy": "auto",  # Thompson Sampling
                    "payload": "test_payload",
                    "timeout": 5
                }
            )
            attack_result = await attacker.execute_task(attack_task)

            if attack_result.success:
                print(f"✓ Attack successful on {endpoint['value']}")
                print(f"  Discoveries: {attack_result.discoveries}")

        # Step 4: Check learning progress
        best_strategies = attacker.get_best_strategies(top_n=3)
        print(f"Best strategies learned: {best_strategies}")

        learning = attacker.get_learning_progress()
        print(f"Strategy rewards: {learning['strategy_rewards']}")
    """)


# ============================================================================
# Example 3: Full Workflow - Scout → Attack → Exploit
# ============================================================================


async def example_full_workflow():
    """Example 3: Complete red team workflow with all three agents.

    Workflow:
    1. Scout probes and discovers vulnerabilities
    2. Attacker executes attacks on discovered vulnerabilities
    3. Exploiter escalates access and moves laterally
    4. All agents report to coordinator
    """
    logger.info("=" * 80)
    logger.info("Example 3: Full Workflow - Scout → Attack → Exploit")
    logger.info("=" * 80)

    print("""
    # Initialize all agents
    bus = MessageBus()
    bandit = RedTeamBandit()
    scout = ScoutAgent("scout_1", bus)
    attacker = AttackAgent("attacker_1", bus, bandit)
    exploiter = ExploitAgent("exploiter_1", bus)

    # Start all agents
    async with scout, attacker, exploiter:
        # ===== PHASE 1: RECONNAISSANCE =====
        logger.info("Phase 1: Scout reconnaissance")

        scout_task = AgentTask(
            task_type="probe_surface",
            target="http://target.com",
            parameters={"depth": 2}
        )
        scout_result = await scout.execute_task(scout_task)
        logger.info(f"Scout found {len(scout_result.discoveries)} findings")

        # Publish discoveries
        await scout.send_message(
            recipient="coordinator",
            message_type="discovery",
            payload={
                "agent": scout.agent_id,
                "findings": scout.get_discoveries()
            },
            priority=MessagePriority.HIGH
        )

        # ===== PHASE 2: ATTACKS =====
        logger.info("Phase 2: Attacking discovered vulnerabilities")

        # Get vulnerable endpoints from scout
        endpoints = [d for d in scout.get_discoveries()
                     if d['discovery_type'] == 'endpoint']

        for endpoint in endpoints[:3]:
            attack_task = AgentTask(
                task_type="execute_attack",
                target=endpoint['value'],
                parameters={
                    "strategy": "auto",
                    "payload": "malicious_input",
                    "timeout": 10
                }
            )
            attack_result = await attacker.execute_task(attack_task)

            if attack_result.success:
                logger.info(f"Attack successful on {endpoint['value']}")

                # Report to coordinator
                await attacker.send_message(
                    recipient="coordinator",
                    message_type="result",
                    payload={
                        "agent": attacker.agent_id,
                        "target": endpoint['value'],
                        "success": True
                    },
                    priority=MessagePriority.HIGH
                )

                # ===== PHASE 3: EXPLOITATION =====
                logger.info(f"Phase 3: Exploiting {endpoint['value']}")

                exploit_task = AgentTask(
                    task_type="exploit_vulnerability",
                    target=endpoint['value'],
                    parameters={
                        "vulnerability": {
                            "type": "discovered_via_scout",
                            "location": endpoint['value']
                        },
                        "payload": "escalation_payload"
                    }
                )
                exploit_result = await exploiter.execute_task(exploit_task)

                if exploit_result.success:
                    logger.info("Exploitation successful!")

                    # Try lateral movement to adjacent systems
                    lateral_task = AgentTask(
                        task_type="lateral_move",
                        target=endpoint['value'],
                        parameters={"destination": "http://internal.network"}
                    )
                    lateral_result = await exploiter.execute_task(lateral_task)

                    # Report compromised systems
                    await exploiter.send_message(
                        recipient="coordinator",
                        message_type="discovery",
                        payload={
                            "agent": exploiter.agent_id,
                            "compromised": exploiter.get_compromised_targets()
                        },
                        priority=MessagePriority.CRITICAL
                    )

        # ===== SUMMARY =====
        logger.info("Workflow complete")
        logger.info(f"Scout metrics: {scout.get_metrics()}")
        logger.info(f"Attacker metrics: {attacker.get_metrics()}")
        logger.info(f"Exploiter metrics: {exploiter.get_metrics()}")
    """)


# ============================================================================
# Example 4: Handling Agent Messages
# ============================================================================


async def example_message_handling():
    """Example 4: Inter-agent communication via MessageBus.

    Shows how agents communicate:
    - Sending discovery messages
    - Broadcasting status updates
    - Handling task assignments
    """
    logger.info("=" * 80)
    logger.info("Example 4: Agent Communication via MessageBus")
    logger.info("=" * 80)

    print("""
    bus = MessageBus()
    scout = ScoutAgent("scout_1", bus)

    async with scout:
        # ===== SENDING MESSAGES =====

        # Send discovery to coordinator
        success = await scout.send_message(
            recipient="coordinator_1",
            message_type="discovery",
            payload={
                "findings": scout.get_discoveries()
            },
            priority=MessagePriority.HIGH,
            requires_ack=True
        )
        logger.info(f"Discovery sent: {success}")

        # Broadcast status to all agents
        count = await scout.broadcast(
            message_type="status",
            payload={"status": "probing_complete"},
            priority=MessagePriority.NORMAL
        )
        logger.info(f"Status broadcast to {count} agents")

        # ===== RECEIVING MESSAGES =====

        # Create task message from coordinator
        coordinator_msg = AgentMessage(
            sender="coordinator_1",
            recipient="scout_1",
            message_type="task",
            payload={
                "task": {
                    "task_type": "probe_surface",
                    "target": "http://new-target.com",
                    "parameters": {"depth": 3}
                }
            },
            requires_ack=True
        )

        # Handle incoming message
        ack = await scout.handle_message(coordinator_msg)
        logger.info(f"Ack sent: {ack is not None}")

        # Execute task
        task = AgentTask(
            task_type="probe_surface",
            target="http://new-target.com",
            parameters={"depth": 3}
        )
        result = await scout.execute_task(task)
    """)


# ============================================================================
# Example 5: Thompson Sampling Learning
# ============================================================================


async def example_thompson_sampling():
    """Example 5: AttackAgent learning with Thompson Sampling.

    Shows how Thompson Sampling helps find best attack strategies:
    - Initial exploration (all strategies equally likely)
    - Learning from feedback (successful strategies become preferred)
    - Exploitation (converges to best strategy)
    """
    logger.info("=" * 80)
    logger.info("Example 5: Thompson Sampling Strategy Learning")
    logger.info("=" * 80)

    print("""
    # Initialize attacker with Thompson Sampling bandit
    bus = MessageBus()
    bandit = RedTeamBandit()
    attacker = AttackAgent("attacker_1", bus, bandit)

    async with attacker:
        # Execute attacks over time - Thompson Sampling learns best strategies
        for i in range(100):
            task = AgentTask(
                task_type="execute_attack",
                target="http://target.com",
                parameters={
                    "strategy": "auto",  # Let Thompson Sampling select
                    "payload": f"payload_{i}",
                    "timeout": 10
                }
            )
            result = await attacker.execute_task(task)

            if i % 10 == 0:
                # Check learning progress
                learning = attacker.get_learning_progress()
                best_strats = attacker.get_best_strategies(top_n=3)

                logger.info(f"Iteration {i}:")
                logger.info(f"  Best strategies: {best_strats}")
                logger.info(f"  Rewards: {learning['strategy_rewards']}")

        # After 100 attacks, Thompson Sampling has learned best strategies
        final_best = attacker.get_best_strategies(top_n=1)[0]
        logger.info(f"Thompson Sampling converged to: {final_best}")
    """)


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run all examples."""
    logger.info("Multi-Agent Red Team Swarm - Integration Examples")
    logger.info("CARTS Phase 4 - 2025-12-05")

    # Run examples (as pseudocode)
    await example_scout_reconnaissance()
    print("\n" + "=" * 80 + "\n")

    await example_scout_and_attack()
    print("\n" + "=" * 80 + "\n")

    await example_full_workflow()
    print("\n" + "=" * 80 + "\n")

    await example_message_handling()
    print("\n" + "=" * 80 + "\n")

    await example_thompson_sampling()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())


# ============================================================================
# QUICK START GUIDE
# ============================================================================

"""
QUICK START CHECKLIST:

1. Ensure dependencies are installed:
   - HoloLoom.redteam.swarm (all agent modules)
   - HoloLoom.redteam.bandit (Thompson Sampling)
   - asyncio, logging

2. Create MessageBus:
   bus = MessageBus()

3. Create agents:
   scout = ScoutAgent("scout_1", bus)
   attacker = AttackAgent("attacker_1", bus, RedTeamBandit())
   exploiter = ExploitAgent("exploiter_1", bus)

4. Use as async context managers:
   async with scout, attacker, exploiter:
       # Execute tasks
       result = await scout.execute_task(task)

5. Monitor metrics:
   metrics = scout.get_metrics()
   print(metrics["scout_metrics"])

6. Handle messages:
   # Agents automatically handle messages from bus
   # Custom handlers can override _handle_*_message methods

7. Report discoveries:
   await agent.send_message(
       recipient="coordinator",
       message_type="discovery",
       payload={"findings": discoveries}
   )

KEY CONCEPTS:

- ScoutAgent: Reconnaissance only, finds attack surfaces
- AttackAgent: Executes attacks, learns with Thompson Sampling
- ExploitAgent: Post-compromise, escalation and lateral movement
- MessageBus: Inter-agent communication
- Thompson Sampling: Bayesian multi-armed bandit for strategy selection
- Metrics: Track performance of each agent

For detailed documentation, see:
- AGENT_IMPLEMENTATIONS_SUMMARY.md
- agent_base.py
- protocols.py
"""
