#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Background Agents Demo
=================================

Demonstrates tiny recursive loops running in background for each system.

Created: 2025-01-20

Scenario:
- Start 4 persistent agents (one per system)
- Process requests through each system
- Agents learn continuously in background
- Show insights, Thompson priors, state changes
"""

import asyncio
from pathlib import Path

from HoloLoom.agents.persistent_agent import (
    PersistentBackgroundAgent,
    AgentManager,
)


async def demo_single_agent():
    """Demo 1: Single persistent agent."""
    print("\n" + "="*70)
    print("Demo 1: Single Persistent Agent")
    print("="*70)

    async with PersistentBackgroundAgent(
        agent_id="chain_orchestrator_01",
        agent_type="chain",
        loop_interval=5.0  # Fast loop for demo
    ) as agent:
        # Set up callbacks
        def on_insight(insight: str):
            print(f"\n💡 New Insight: {insight}")

        def on_state_change(state):
            print(f"\n📊 State Update: {state.requests_processed} requests, "
                  f"avg confidence: {state.avg_confidence:.2f}")

        agent.on_insight = on_insight
        agent.on_state_change = on_state_change

        print(f"\n✅ Agent started: {agent.agent_id}")
        print(f"   Loop interval: {agent.loop_interval}s")

        # Simulate processing requests
        print("\n📝 Simulating 10 requests...\n")
        for i in range(10):
            # Simulate request
            query = f"Query {i+1}"
            confidence = 0.5 + (i * 0.05)  # Increasing confidence
            duration_ms = 150 - (i * 5)  # Decreasing latency

            agent.record_request(
                query=query,
                result=f"Result for {query}",
                confidence=confidence,
                duration_ms=duration_ms,
                metadata={"strategy": "default"}
            )

            print(f"   {i+1}. {query} → confidence: {confidence:.2f}, latency: {duration_ms:.0f}ms")
            await asyncio.sleep(0.5)

        # Wait for learning cycle
        print("\n⏳ Waiting for learning cycle...")
        await asyncio.sleep(6)  # Wait for one cycle

        # Show final state
        state = agent.get_state()
        print(f"\n📈 Final State:")
        print(f"   Requests processed: {state.requests_processed}")
        print(f"   Avg confidence: {state.avg_confidence:.2f}")
        print(f"   Learning cycles: {state.learning_cycles}")
        print(f"   Insights generated: {len(state.insights)}")

        # Show Thompson priors
        print(f"\n🎲 Thompson Sampling Priors:")
        for strategy, priors in state.thompson_priors.items():
            alpha = priors['alpha']
            beta = priors['beta']
            expected = alpha / (alpha + beta)
            print(f"   {strategy}: α={alpha:.2f}, β={beta:.2f}, E[X]={expected:.2f}")

        # Show insights
        if state.insights:
            print(f"\n💡 Insights:")
            for insight in state.insights:
                print(f"   - {insight}")


async def demo_multi_agent():
    """Demo 2: Multiple agents (one per system)."""
    print("\n" + "="*70)
    print("Demo 2: Multiple Persistent Agents (All 4 Systems)")
    print("="*70)

    async with AgentManager(loop_interval=5.0) as manager:
        # Create 4 agents (one per system)
        print("\n🚀 Starting 4 persistent agents...")

        chain_agent = await manager.create_agent(
            agent_id="chain_orchestrator",
            agent_type="chain"
        )
        print("   ✅ Chain Orchestrator agent started")

        recursive_agent = await manager.create_agent(
            agent_id="recursive_reasoner",
            agent_type="recursive"
        )
        print("   ✅ Recursive Reasoner agent started")

        workflow_agent = await manager.create_agent(
            agent_id="agentic_workflow",
            agent_type="workflow"
        )
        print("   ✅ Agentic Workflow agent started")

        scratchpad_agent = await manager.create_agent(
            agent_id="hofstadter_scratchpad",
            agent_type="scratchpad"
        )
        print("   ✅ Hofstadter Scratchpad agent started")

        # Simulate requests to different agents
        print("\n📝 Simulating requests to all 4 agents...\n")

        for i in range(5):
            # Chain agent
            chain_agent.record_request(
                query=f"Chain query {i+1}",
                result="Chain result",
                confidence=0.8 + (i * 0.02),
                duration_ms=150,
                metadata={"strategy": "verified_query"}
            )

            # Recursive agent
            recursive_agent.record_request(
                query=f"Recursive query {i+1}",
                result="Recursive result",
                confidence=0.6 + (i * 0.05),  # Improving
                duration_ms=300,
                metadata={"strategy": "expand_search"}
            )

            # Workflow agent
            workflow_agent.record_request(
                query=f"Workflow query {i+1}",
                result="Workflow result",
                confidence=0.75,
                duration_ms=500,
                metadata={"strategy": "recursive_research"}
            )

            # Scratchpad agent
            scratchpad_agent.record_request(
                query=f"Scratchpad query {i+1}",
                result="Scratchpad result",
                confidence=0.7 + (i * 0.03),
                duration_ms=200,
                metadata={"strategy": "hofstadter"}
            )

            print(f"   Batch {i+1}: 4 requests across all agents")
            await asyncio.sleep(1)

        # Wait for learning cycles
        print("\n⏳ Waiting for learning cycles...")
        await asyncio.sleep(6)

        # Show all states
        print(f"\n📊 Agent States:\n")
        all_states = manager.get_all_states()

        for agent_id, state in all_states.items():
            print(f"   {agent_id}:")
            print(f"      Requests: {state.requests_processed}")
            print(f"      Avg Confidence: {state.avg_confidence:.2f}")
            print(f"      Learning Cycles: {state.learning_cycles}")
            print(f"      Insights: {len(state.insights)}")


async def demo_insight_generation():
    """Demo 3: Insight generation from patterns."""
    print("\n" + "="*70)
    print("Demo 3: Insight Generation from Patterns")
    print("="*70)

    async with PersistentBackgroundAgent(
        agent_id="insight_demo",
        agent_type="chain",
        loop_interval=5.0
    ) as agent:
        # Track insights
        insights = []

        def on_insight(insight: str):
            insights.append(insight)
            print(f"\n💡 Pattern Detected: {insight}")

        agent.on_insight = on_insight

        print("\n📝 Creating pattern: Low-confidence requests...\n")

        # Create pattern: Low-confidence requests
        for i in range(5):
            agent.record_request(
                query=f"Difficult query {i+1}",
                result="Uncertain result",
                confidence=0.4 + (i * 0.02),  # All low
                duration_ms=150,
                metadata={"strategy": "default"}
            )
            print(f"   {i+1}. Low confidence: {0.4 + (i * 0.02):.2f}")
            await asyncio.sleep(0.5)

        # Wait for learning cycle to detect pattern
        print("\n⏳ Waiting for pattern detection...")
        await asyncio.sleep(6)

        print(f"\n📈 Insights Generated: {len(insights)}")
        for insight in insights:
            print(f"   - {insight}")


async def demo_thompson_adaptation():
    """Demo 4: Thompson Sampling adaptation."""
    print("\n" + "="*70)
    print("Demo 4: Thompson Sampling Adaptation")
    print("="*70)

    async with PersistentBackgroundAgent(
        agent_id="thompson_demo",
        agent_type="recursive",
        loop_interval=5.0
    ) as agent:
        print("\n📊 Testing 3 strategies with different success rates...\n")

        strategies = {
            "strategy_A": 0.9,  # High success
            "strategy_B": 0.6,  # Medium success
            "strategy_C": 0.3,  # Low success
        }

        # Simulate requests with different strategies
        for round_num in range(3):
            print(f"\n   Round {round_num + 1}:")
            for strategy, success_rate in strategies.items():
                # Simulate 5 requests per strategy
                for i in range(5):
                    confidence = success_rate + (0.1 * (i % 2))  # Add some variance
                    agent.record_request(
                        query=f"Query with {strategy}",
                        result="Result",
                        confidence=confidence,
                        duration_ms=150,
                        metadata={"strategy": strategy}
                    )

                print(f"      {strategy}: {success_rate:.1%} success rate")

            await asyncio.sleep(1)

        # Wait for learning
        print("\n⏳ Waiting for Thompson Sampling updates...")
        await asyncio.sleep(6)

        # Show Thompson priors
        priors = agent.get_thompson_priors()
        print(f"\n🎲 Thompson Sampling Results:\n")

        for strategy, prior in sorted(priors.items()):
            alpha = prior['alpha']
            beta = prior['beta']
            expected = alpha / (alpha + beta)

            print(f"   {strategy}:")
            print(f"      α (successes): {alpha:.2f}")
            print(f"      β (failures): {beta:.2f}")
            print(f"      E[X] (expected reward): {expected:.2f}")
            print()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("PERSISTENT BACKGROUND AGENTS DEMO")
    print("="*70)
    print("\nTiny recursive loops running in background for continuous learning")

    await demo_single_agent()
    await asyncio.sleep(2)

    await demo_multi_agent()
    await asyncio.sleep(2)

    await demo_insight_generation()
    await asyncio.sleep(2)

    await demo_thompson_adaptation()

    print("\n" + "="*70)
    print("✅ All Demos Complete")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Background learning loops (60s intervals)")
    print("  ✓ Persistent internal dialogue via scratchpad")
    print("  ✓ Pattern detection and insight generation")
    print("  ✓ Thompson Sampling adaptation")
    print("  ✓ Continuous state tracking")
    print("  ✓ Multi-agent management")
    print("\nEach agent continuously learns even when not processing requests!")


if __name__ == "__main__":
    asyncio.run(main())
