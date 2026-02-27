#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collaborative Agents Demo
=========================

Demonstrates multi-agent communication with budget limits and safety guardrails.

Created: 2025-01-20

Scenarios:
1. Question & Answer - One agent asks another
2. Help Request - Agent requests help from multiple agents
3. Insight Sharing - Agents share discoveries
4. Budget Limits - Conversations stop when budget exceeded
5. Safety Guardrails - Loop detection and productivity checks
"""

import asyncio
from hololoom.agents.collaborative_agents import (
    CollaborativeAgentManager,
    Budget
)
from hololoom.agents.multi_agent_communication import MessageType


async def demo_question_answer():
    """Demo 1: One agent asks another a question."""
    print("\n" + "="*70)
    print("Demo 1: Question & Answer Between Agents")
    print("="*70)

    async with CollaborativeAgentManager(loop_interval=60.0) as manager:
        # Create 2 agents
        chain_agent = await manager.create_agent("chain_orchestrator", "chain")
        recursive_agent = await manager.create_agent("recursive_reasoner", "recursive")

        print("\n✅ Created 2 agents:")
        print("   - Chain Orchestrator")
        print("   - Recursive Reasoner")

        # Chain agent asks Recursive agent a question
        print("\n💬 Chain Orchestrator asks Recursive Reasoner...")
        answer = await chain_agent.ask_question(
            to_agent="recursive_reasoner",
            question="When should I use iterative refinement vs simple chains?",
            topic="Strategy Selection",
            timeout=10.0
        )

        if answer:
            print(f"\n💡 Recursive Reasoner answered:")
            print(f"   {answer}")
        else:
            print("\n⚠️  No answer received")

        # Show statistics
        stats = manager.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Conversations: {stats['conversations']['total_conversations']}")
        print(f"   Messages: {stats['conversations']['total_messages']}")


async def demo_help_request():
    """Demo 2: Agent requests help from multiple agents."""
    print("\n" + "="*70)
    print("Demo 2: Help Request to Multiple Agents")
    print("="*70)

    async with CollaborativeAgentManager(loop_interval=60.0) as manager:
        # Create 4 agents
        print("\n✅ Creating 4 agents...\n")
        workflow_agent = await manager.create_agent("workflow", "workflow")
        chain_agent = await manager.create_agent("chain", "chain")
        recursive_agent = await manager.create_agent("recursive", "recursive")
        scratchpad_agent = await manager.create_agent("scratchpad", "scratchpad")

        # Workflow agent needs help
        print("💬 Workflow agent requests help from all others...")
        print("   Request: How to handle a complex multi-step query?")

        responses = await workflow_agent.request_help(
            from_agents=["chain", "recursive", "scratchpad"],
            request="I have a complex multi-step query that needs sequential processing, refinement, and meta-reasoning. Which approach should I use?",
            topic="Complex Query Handling",
            timeout=5.0
        )

        print(f"\n📨 Received {len(responses)} response(s):\n")
        for agent_id, response in responses:
            print(f"   {agent_id}:")
            print(f"      {response}\n")


async def demo_insight_sharing():
    """Demo 3: Agents share insights with each other."""
    print("\n" + "="*70)
    print("Demo 3: Insight Sharing Between Agents")
    print("="*70)

    async with CollaborativeAgentManager(loop_interval=60.0) as manager:
        # Create agents
        print("\n✅ Creating agents...\n")
        chain_agent = await manager.create_agent("chain", "chain")
        recursive_agent = await manager.create_agent("recursive", "recursive")
        scratchpad_agent = await manager.create_agent("scratchpad", "scratchpad")

        # Chain agent discovers something
        print("💡 Chain Orchestrator discovered:")
        insight = "Pattern detected: Verified queries have 20% higher confidence than simple queries"
        print(f"   {insight}")

        # Share with others
        print("\n📤 Sharing insight with Recursive and Scratchpad agents...")
        await chain_agent.share_insight(
            with_agents=["recursive", "scratchpad"],
            insight=insight,
            topic="Performance Patterns"
        )

        # Wait for acknowledgments
        await asyncio.sleep(3)

        # Check if others received it
        print("\n✅ Other agents received insight:")
        recursive_insights = recursive_agent.state.insights
        scratchpad_insights = scratchpad_agent.state.insights

        print(f"   Recursive agent: {len(recursive_insights)} insights")
        print(f"   Scratchpad agent: {len(scratchpad_insights)} insights")


async def demo_budget_limits():
    """Demo 4: Conversation stops when budget exceeded."""
    print("\n" + "="*70)
    print("Demo 4: Budget Limits Enforcement")
    print("="*70)

    # Create manager with tight budget
    tight_budget = Budget(
        max_messages=5,  # Only 5 messages allowed
        max_duration_seconds=30.0,
        max_conversations_per_hour=2
    )

    async with CollaborativeAgentManager(
        loop_interval=60.0,
        budget=tight_budget
    ) as manager:
        print("\n⚙️  Budget limits:")
        print(f"   Max messages: {tight_budget.max_messages}")
        print(f"   Max duration: {tight_budget.max_duration_seconds}s")
        print(f"   Max conversations/hour: {tight_budget.max_conversations_per_hour}")

        # Create agents
        agent1 = await manager.create_agent("agent1", "chain")
        agent2 = await manager.create_agent("agent2", "recursive")

        # Set up budget exceeded callback
        exceeded_count = [0]

        def on_budget_exceeded(reason):
            exceeded_count[0] += 1
            print(f"\n⚠️  Budget exceeded: {reason}")

        manager.conversation_manager.on_budget_exceeded = on_budget_exceeded

        # Start conversation
        print("\n💬 Starting conversation (will hit budget limit)...")
        conversation_id = await manager.conversation_manager.start_conversation(
            topic="Test",
            initiator="agent1",
            participants=["agent1", "agent2"]
        )

        if conversation_id:
            # Send messages until budget exceeded
            for i in range(10):  # Try to send 10 messages (budget only allows 5)
                msg = await agent1.send_message(
                    conversation_id=conversation_id,
                    to_agent="agent2",
                    message_type=MessageType.QUESTION,
                    content=f"Question {i+1}"
                )

                if not msg:
                    print(f"\n🛑 Stopped at message {i+1} (budget limit reached)")
                    break

                await asyncio.sleep(0.5)

        print(f"\n📊 Budget exceeded {exceeded_count[0]} time(s)")


async def demo_safety_guardrails():
    """Demo 5: Safety guardrails prevent infinite loops."""
    print("\n" + "="*70)
    print("Demo 5: Safety Guardrails (Loop Detection)")
    print("="*70)

    async with CollaborativeAgentManager(loop_interval=60.0) as manager:
        # Create agents
        agent1 = await manager.create_agent("agent1", "chain")
        agent2 = await manager.create_agent("agent2", "recursive")

        # Set up safety violation callback
        violations = []

        def on_safety_violation(reason):
            violations.append(reason)
            print(f"\n⚠️  Safety violation: {reason}")

        manager.conversation_manager.on_safety_violation = on_safety_violation

        print("\n💬 Starting conversation with repetitive messages...")

        conversation_id = await manager.conversation_manager.start_conversation(
            topic="Test",
            initiator="agent1",
            participants=["agent1", "agent2"]
        )

        if conversation_id:
            # Send repetitive messages (should trigger loop detection)
            repetitive_content = "Tell me about Thompson Sampling strategies"

            for i in range(6):
                msg = await agent1.send_message(
                    conversation_id=conversation_id,
                    to_agent="agent2",
                    message_type=MessageType.QUESTION,
                    content=repetitive_content  # Same content each time!
                )

                if not msg:
                    print(f"\n🛑 Stopped at message {i+1} (loop detected)")
                    break

                print(f"   Message {i+1}: {repetitive_content[:40]}...")
                await asyncio.sleep(0.5)

        print(f"\n🛡️  Safety violations detected: {len(violations)}")
        for violation in violations:
            print(f"   - {violation}")


async def demo_multi_agent_collaboration():
    """Demo 6: Full collaboration scenario."""
    print("\n" + "="*70)
    print("Demo 6: Full Multi-Agent Collaboration")
    print("="*70)

    async with CollaborativeAgentManager(loop_interval=60.0) as manager:
        # Create 4 agents
        print("\n✅ Creating 4-agent system...\n")
        chain = await manager.create_agent("chain", "chain")
        recursive = await manager.create_agent("recursive", "recursive")
        workflow = await manager.create_agent("workflow", "workflow")
        scratchpad = await manager.create_agent("scratchpad", "scratchpad")

        # Scenario: Complex query arrives
        print("📝 Scenario: Complex research query arrives")
        print("   Query: 'Compare Thompson Sampling vs UCB vs Epsilon-Greedy'")

        # Step 1: Workflow agent asks Chain agent
        print("\n1️⃣  Workflow asks Chain: How to structure this?")
        answer1 = await workflow.ask_question(
            to_agent="chain",
            question="How should I structure a comparison query?",
            topic="Query Structuring",
            timeout=5.0
        )
        if answer1:
            print(f"   Chain: {answer1[:80]}...")

        await asyncio.sleep(1)

        # Step 2: Chain agent asks Recursive agent
        print("\n2️⃣  Chain asks Recursive: How to ensure quality?")
        answer2 = await chain.ask_question(
            to_agent="recursive",
            question="How should I refine low-confidence comparisons?",
            topic="Quality Refinement",
            timeout=5.0
        )
        if answer2:
            print(f"   Recursive: {answer2[:80]}...")

        await asyncio.sleep(1)

        # Step 3: Recursive agent shares insight
        print("\n3️⃣  Recursive shares insight with all:")
        await recursive.share_insight(
            with_agents=["chain", "workflow", "scratchpad"],
            insight="For comparisons, use EXPAND_SEARCH strategy first, then MULTI_PERSPECTIVE for synthesis",
            topic="Comparison Strategies"
        )

        await asyncio.sleep(2)

        # Step 4: Scratchpad agent reflects
        print("\n4️⃣  Scratchpad reflects on conversation:")
        print("   'This collaboration showed sequential refinement works best'")

        # Final stats
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Active agents: {stats['agents']}")
        print(f"   Total conversations: {stats['conversations']['total_conversations']}")
        print(f"   Total messages: {stats['conversations']['total_messages']}")
        print(f"   Avg messages/conversation: {stats['conversations']['avg_messages_per_conversation']:.1f}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("COLLABORATIVE AGENTS DEMO")
    print("="*70)
    print("\nMulti-agent communication with budget limits and safety guardrails")

    await demo_question_answer()
    await asyncio.sleep(2)

    await demo_help_request()
    await asyncio.sleep(2)

    await demo_insight_sharing()
    await asyncio.sleep(2)

    await demo_budget_limits()
    await asyncio.sleep(2)

    await demo_safety_guardrails()
    await asyncio.sleep(2)

    await demo_multi_agent_collaboration()

    print("\n" + "="*70)
    print("✅ All Demos Complete")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Question & Answer between agents")
    print("  ✓ Multi-agent help requests")
    print("  ✓ Insight sharing")
    print("  ✓ Budget limits (messages, time, conversations)")
    print("  ✓ Safety guardrails (loop detection, productivity)")
    print("  ✓ Full multi-agent collaboration")
    print("\nAgents can now learn from each other!")


if __name__ == "__main__":
    asyncio.run(main())
