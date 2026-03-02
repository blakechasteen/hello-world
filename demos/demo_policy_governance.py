#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy & Governance Demo
========================

Demonstrates policy-based decision making for multi-agent systems.

Shows:
1. Role-Based Access Control (RBAC)
2. Topic restrictions
3. Policy templates (development, production, enterprise)
4. Audit trail
5. Escalation workflow

Created: 2025-01-20
"""

import asyncio
import logging
from typing import Dict, Any

from hololoom.agents.policy_governance import (
    PolicyEngine,
    RoleBasedAccessControl,
    TopicGovernance,
    CommunicationRequest,
    PolicyDecision,
    Priority,
    AgentRole,
    PolicyTemplates,
    GovernancePolicy,
    PolicyRule
)
from hololoom.agents.collaborative_agents import (
    CollaborativeAgentManager,
    MessageType
)
from hololoom.agents.multi_agent_communication import Budget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def print_decision(request: CommunicationRequest, decision: PolicyDecision, reason: str) -> None:
    """Print policy decision."""
    status = "✅ ALLOWED" if decision == PolicyDecision.ALLOW else "❌ DENIED" if decision == PolicyDecision.DENY else "⚠️  ESCALATED"
    print(f"{status}")
    print(f"  From: {request.from_agent} ({request.message_type})")
    print(f"  To: {request.to_agent}")
    print(f"  Topic: {request.topic}")
    print(f"  Priority: {request.priority.value}")
    print(f"  Reason: {reason}")
    print()


async def demo_rbac():
    """Demo 1: Role-Based Access Control."""
    print_section("Demo 1: Role-Based Access Control (RBAC)")

    rbac = RoleBasedAccessControl()

    # Assign roles
    rbac.assign_role("admin_agent", AgentRole.ADMIN)
    rbac.assign_role("coordinator_agent", AgentRole.COORDINATOR)
    rbac.assign_role("worker_agent", AgentRole.WORKER)
    rbac.assign_role("observer_agent", AgentRole.OBSERVER)
    rbac.assign_role("restricted_agent", AgentRole.RESTRICTED)

    print("Roles assigned:")
    print(f"  admin_agent → ADMIN")
    print(f"  coordinator_agent → COORDINATOR")
    print(f"  worker_agent → WORKER")
    print(f"  observer_agent → OBSERVER")
    print(f"  restricted_agent → RESTRICTED\n")

    # Test communication
    test_cases = [
        ("admin_agent", "worker_agent", "query"),
        ("coordinator_agent", "worker_agent", "help_request"),
        ("worker_agent", "coordinator_agent", "answer"),
        ("observer_agent", "worker_agent", "query"),  # Should fail
        ("restricted_agent", "admin_agent", "query"),  # Should fail
        ("worker_agent", "observer_agent", "insight"),  # Should fail
    ]

    for from_agent, to_agent, message_type in test_cases:
        can_communicate, reason = rbac.can_communicate(from_agent, to_agent, message_type)
        status = "✅" if can_communicate else "❌"
        print(f"{status} {from_agent} → {to_agent} ({message_type}): {reason}")


async def demo_topic_governance():
    """Demo 2: Topic Governance."""
    print_section("Demo 2: Topic Governance")

    topic_gov = TopicGovernance()

    # Configure topics
    topic_gov.allow_topic("research")
    topic_gov.allow_topic("development")
    topic_gov.forbid_topic("security")  # Globally forbidden
    topic_gov.restrict_topic("confidential", ["admin_agent", "coordinator_agent"])

    print("Topics configured:")
    print(f"  Allowed: research, development")
    print(f"  Forbidden: security")
    print(f"  Restricted: confidential → [admin_agent, coordinator_agent]\n")

    # Test topic access
    test_cases = [
        ("research", "worker_agent"),
        ("development", "worker_agent"),
        ("security", "admin_agent"),  # Forbidden for everyone
        ("confidential", "admin_agent"),  # Allowed
        ("confidential", "worker_agent"),  # Denied
        ("random_topic", "worker_agent"),  # Not in allowed list
    ]

    for topic, agent_id in test_cases:
        allowed, reason = topic_gov.is_topic_allowed(topic, agent_id)
        status = "✅" if allowed else "❌"
        print(f"{status} {agent_id} can discuss '{topic}': {reason}")


async def demo_policy_templates():
    """Demo 3: Policy Templates."""
    print_section("Demo 3: Policy Templates (Development, Production, Enterprise)")

    # Setup RBAC and topic governance
    rbac = RoleBasedAccessControl()
    rbac.assign_role("worker1", AgentRole.WORKER)
    rbac.assign_role("worker2", AgentRole.WORKER)

    topic_gov = TopicGovernance()
    topic_gov.allow_topic("general")

    # Test each template
    templates = {
        "Development": PolicyTemplates.development(),
        "Production": PolicyTemplates.production(),
        "Enterprise": PolicyTemplates.enterprise()
    }

    for template_name, policy in templates.items():
        print(f"\n{template_name} Policy:")
        print(f"  Rules: {len(policy.rules)}")
        print(f"  Default: {policy.default_decision.value}\n")

        engine = PolicyEngine(rbac, topic_gov)
        engine.register_policy(policy)

        # Test critical priority
        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="general",
            content="Critical issue",
            priority=Priority.CRITICAL
        )
        decision, reason = engine.evaluate_communication_request(request)
        print_decision(request, decision, reason)

        # Test low priority
        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="general",
            content="Low priority question",
            priority=Priority.LOW
        )
        decision, reason = engine.evaluate_communication_request(request)
        print_decision(request, decision, reason)


async def demo_custom_policy():
    """Demo 4: Custom Policy Rules."""
    print_section("Demo 4: Custom Policy Rules")

    # Setup
    rbac = RoleBasedAccessControl()
    rbac.assign_role("worker1", AgentRole.WORKER)
    rbac.assign_role("worker2", AgentRole.WORKER)

    topic_gov = TopicGovernance()
    topic_gov.allow_topic("general")

    # Create custom policy
    custom_policy = GovernancePolicy(
        policy_id="custom_policy",
        name="Custom Policy",
        rules=[
            # Rule 1: Block messages containing "urgent" in low priority
            PolicyRule(
                rule_id="urgent_mismatch",
                name="Urgent content must be high priority",
                description="Block low-priority messages with urgent content",
                condition=lambda ctx: (
                    "urgent" in ctx.get("content", "").lower() and
                    ctx.get("priority") == Priority.LOW
                ),
                decision=PolicyDecision.DENY,
                priority=100
            ),

            # Rule 2: Escalate large messages
            PolicyRule(
                rule_id="large_message",
                name="Escalate large messages",
                description="Messages >500 chars need approval",
                condition=lambda ctx: len(ctx.get("content", "")) > 500,
                decision=PolicyDecision.ESCALATE,
                priority=90
            ),

            # Rule 3: Audit all help requests
            PolicyRule(
                rule_id="audit_help",
                name="Audit help requests",
                description="Log all help requests",
                condition=lambda ctx: ctx.get("message_type") == "help_request",
                decision=PolicyDecision.AUDIT_ONLY,
                priority=50
            )
        ],
        default_decision=PolicyDecision.ALLOW
    )

    engine = PolicyEngine(rbac, topic_gov)
    engine.register_policy(custom_policy)

    print("Custom policy registered:")
    print(f"  Rules: {len(custom_policy.rules)}")
    print(f"    1. Urgent content must be high priority")
    print(f"    2. Escalate large messages (>500 chars)")
    print(f"    3. Audit all help requests\n")

    # Test cases
    test_cases = [
        {
            "content": "This is URGENT!",
            "priority": Priority.LOW,
            "description": "Urgent content + low priority (should DENY)"
        },
        {
            "content": "This is URGENT!",
            "priority": Priority.HIGH,
            "description": "Urgent content + high priority (should ALLOW)"
        },
        {
            "content": "x" * 600,  # 600 chars
            "priority": Priority.MEDIUM,
            "description": "Large message (should ESCALATE)"
        },
        {
            "content": "Normal question",
            "priority": Priority.MEDIUM,
            "description": "Normal message (should ALLOW)"
        }
    ]

    for test_case in test_cases:
        print(f"Test: {test_case['description']}")
        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="general",
            content=test_case["content"],
            priority=test_case["priority"]
        )
        decision, reason = engine.evaluate_communication_request(request)
        print_decision(request, decision, reason)


async def demo_audit_trail():
    """Demo 5: Audit Trail."""
    print_section("Demo 5: Audit Trail")

    # Setup
    rbac = RoleBasedAccessControl()
    rbac.assign_role("worker1", AgentRole.WORKER)
    rbac.assign_role("worker2", AgentRole.WORKER)
    rbac.assign_role("observer1", AgentRole.OBSERVER)

    topic_gov = TopicGovernance()
    topic_gov.allow_topic("general")
    topic_gov.forbid_topic("security")

    engine = PolicyEngine(rbac, topic_gov)
    engine.register_policy(PolicyTemplates.production())

    print("Making several requests...\n")

    # Make several requests
    requests = [
        ("worker1", "worker2", "general", Priority.MEDIUM),
        ("worker1", "worker2", "security", Priority.HIGH),  # Forbidden topic
        ("observer1", "worker1", "general", Priority.LOW),  # Observer can't send
        ("worker1", "worker2", "general", Priority.CRITICAL),
    ]

    for from_agent, to_agent, topic, priority in requests:
        request = CommunicationRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type="query",
            topic=topic,
            content=f"Message about {topic}",
            priority=priority
        )
        decision, reason = engine.evaluate_communication_request(request)
        print_decision(request, decision, reason)

    # View audit trail
    print("\nAudit Trail:")
    print(f"{'='*60}")
    audit = engine.get_audit_trail(limit=10)

    for entry in audit:
        status = "✅" if entry.decision == PolicyDecision.ALLOW else "❌"
        print(f"{status} {entry.request.from_agent} → {entry.request.to_agent}")
        print(f"   Topic: {entry.request.topic}")
        print(f"   Decision: {entry.decision.value}")
        print(f"   Reason: {entry.reason}")
        print(f"   Time: {entry.timestamp.strftime('%H:%M:%S')}")
        print()

    # Statistics
    stats = engine.get_statistics()
    print("\nStatistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Allow rate: {stats['allow_rate']:.1%}")
    print(f"  Deny rate: {stats['deny_rate']:.1%}")
    print(f"  Escalate rate: {stats['escalate_rate']:.1%}")


async def demo_collaborative_agents_with_policy():
    """Demo 6: Collaborative Agents with Policy Enforcement."""
    print_section("Demo 6: Collaborative Agents with Policy Enforcement")

    # Setup policy engine
    rbac = RoleBasedAccessControl()
    rbac.assign_role("chain_agent", AgentRole.COORDINATOR)
    rbac.assign_role("recursive_agent", AgentRole.WORKER)
    rbac.assign_role("workflow_agent", AgentRole.WORKER)

    topic_gov = TopicGovernance()
    topic_gov.allow_topic("optimization")
    topic_gov.allow_topic("research")
    topic_gov.forbid_topic("security")

    policy_engine = PolicyEngine(rbac, topic_gov)
    policy_engine.register_policy(PolicyTemplates.production())

    print("Policy configured:")
    print(f"  Roles:")
    print(f"    chain_agent → COORDINATOR")
    print(f"    recursive_agent → WORKER")
    print(f"    workflow_agent → WORKER")
    print(f"  Topics:")
    print(f"    Allowed: optimization, research")
    print(f"    Forbidden: security\n")

    # Create manager with policy
    async with CollaborativeAgentManager(
        loop_interval=60.0,
        budget=Budget(max_messages=5),
        policy_engine=policy_engine
    ) as manager:
        # Create agents
        chain = await manager.create_agent("chain_agent", "chain")
        recursive = await manager.create_agent("recursive_agent", "recursive")
        workflow = await manager.create_agent("workflow_agent", "workflow")

        print("Agents created with policy enforcement\n")

        # Test 1: Allowed communication (coordinator → worker)
        print("Test 1: Coordinator → Worker (optimization topic)")
        question = await chain.ask_question(
            to_agent="recursive_agent",
            question="Can you help optimize this query?",
            topic="optimization",
            timeout=5.0
        )
        if question:
            print(f"✅ Question sent and answered: {question[:50]}...")
        else:
            print(f"❌ Question blocked by policy")
        print()

        # Test 2: Forbidden topic
        print("Test 2: Worker → Worker (security topic - FORBIDDEN)")
        question = await recursive.ask_question(
            to_agent="workflow_agent",
            question="Security vulnerability found",
            topic="security",
            timeout=5.0
        )
        if question:
            print(f"✅ Question sent and answered: {question[:50]}...")
        else:
            print(f"❌ Question blocked by policy (expected)")
        print()

        # Test 3: Topic not in allowed list
        print("Test 3: Worker → Worker (random topic - NOT ALLOWED)")
        question = await recursive.ask_question(
            to_agent="workflow_agent",
            question="Random question",
            topic="random",
            timeout=5.0
        )
        if question:
            print(f"✅ Question sent and answered: {question[:50]}...")
        else:
            print(f"❌ Question blocked by policy (expected)")
        print()

    # View audit trail
    print("\nFinal Audit Trail:")
    print(f"{'='*60}")
    audit = policy_engine.get_audit_trail(limit=10)

    for entry in audit:
        status = "✅" if entry.decision == PolicyDecision.ALLOW else "❌"
        print(f"{status} {entry.request.from_agent} → {entry.request.to_agent}")
        print(f"   Topic: {entry.request.topic}")
        print(f"   Decision: {entry.decision.value}")
        print(f"   Reason: {entry.reason}")
        print()

    stats = policy_engine.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Allow rate: {stats['allow_rate']:.1%}")
    print(f"  Deny rate: {stats['deny_rate']:.1%}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("POLICY & GOVERNANCE SYSTEM DEMO")
    print("="*60)

    await demo_rbac()
    await demo_topic_governance()
    await demo_policy_templates()
    await demo_custom_policy()
    await demo_audit_trail()
    await demo_collaborative_agents_with_policy()

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✅ Role-Based Access Control (RBAC)")
    print("  ✅ Topic Governance (allowed/forbidden/restricted)")
    print("  ✅ Policy Templates (dev/prod/enterprise)")
    print("  ✅ Custom Policy Rules")
    print("  ✅ Audit Trail with Statistics")
    print("  ✅ Integration with Collaborative Agents")


if __name__ == "__main__":
    asyncio.run(main())
