#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Policy & Governance System
=====================================

Tests:
1. Role-Based Access Control (RBAC)
2. Topic Governance
3. Policy Engine
4. Policy Templates
5. Audit Trail
6. Integration with Collaborative Agents

Created: 2025-01-20
"""

import pytest
from datetime import datetime

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


class TestRBAC:
    """Test Role-Based Access Control."""

    def test_assign_role(self):
        """Test role assignment."""
        rbac = RoleBasedAccessControl()

        rbac.assign_role("agent1", AgentRole.ADMIN)
        rbac.assign_role("agent2", AgentRole.WORKER)

        assert rbac.get_role("agent1") == AgentRole.ADMIN
        assert rbac.get_role("agent2") == AgentRole.WORKER

    def test_default_role(self):
        """Test default role for unassigned agent."""
        rbac = RoleBasedAccessControl()

        # Unassigned agent should get WORKER role
        assert rbac.get_role("unknown_agent") == AgentRole.WORKER

    def test_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("admin", AgentRole.ADMIN)

        # Admin should have any permission
        has_perm, _ = rbac.has_permission("admin", "any_permission")
        assert has_perm

    def test_worker_permissions(self):
        """Test worker permissions."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker", AgentRole.WORKER)

        # Worker should have query permission
        has_perm, _ = rbac.has_permission("worker", "query")
        assert has_perm

        # Worker should have answer permission
        has_perm, _ = rbac.has_permission("worker", "answer")
        assert has_perm

        # Worker should NOT have help_request permission
        has_perm, _ = rbac.has_permission("worker", "help_request")
        assert not has_perm

    def test_observer_cannot_send(self):
        """Test observer cannot send messages."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("observer", AgentRole.OBSERVER)

        can_communicate, _ = rbac.can_communicate("observer", "worker", "query")
        assert not can_communicate

    def test_restricted_cannot_communicate(self):
        """Test restricted agent cannot communicate."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("restricted", AgentRole.RESTRICTED)

        can_communicate, _ = rbac.can_communicate("restricted", "worker", "query")
        assert not can_communicate

    def test_coordinator_can_talk_to_anyone(self):
        """Test coordinator can talk to anyone."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("coordinator", AgentRole.COORDINATOR)
        rbac.assign_role("worker", AgentRole.WORKER)

        can_communicate, _ = rbac.can_communicate("coordinator", "worker", "query")
        assert can_communicate

    def test_worker_to_worker_allowed(self):
        """Test worker-to-worker communication allowed."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)
        rbac.assign_role("worker2", AgentRole.WORKER)

        can_communicate, _ = rbac.can_communicate("worker1", "worker2", "query")
        assert can_communicate

    def test_worker_to_observer_denied(self):
        """Test worker cannot message observer."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker", AgentRole.WORKER)
        rbac.assign_role("observer", AgentRole.OBSERVER)

        can_communicate, _ = rbac.can_communicate("worker", "observer", "insight")
        assert not can_communicate


class TestTopicGovernance:
    """Test Topic Governance."""

    def test_allow_topic(self):
        """Test allowing a topic."""
        topic_gov = TopicGovernance()

        topic_gov.allow_topic("research")

        allowed, _ = topic_gov.is_topic_allowed("research", "agent1")
        assert allowed

    def test_forbid_topic(self):
        """Test forbidding a topic."""
        topic_gov = TopicGovernance()

        topic_gov.forbid_topic("security")

        allowed, _ = topic_gov.is_topic_allowed("security", "agent1")
        assert not allowed

    def test_restrict_topic(self):
        """Test restricting topic to specific agents."""
        topic_gov = TopicGovernance()

        topic_gov.restrict_topic("confidential", ["admin", "coordinator"])

        # Allowed agents
        allowed, _ = topic_gov.is_topic_allowed("confidential", "admin")
        assert allowed

        # Denied agents
        allowed, _ = topic_gov.is_topic_allowed("confidential", "worker")
        assert not allowed

    def test_no_whitelist_allows_all(self):
        """Test when no whitelist, all topics allowed."""
        topic_gov = TopicGovernance()

        # No topics configured - should allow
        allowed, _ = topic_gov.is_topic_allowed("random", "agent1")
        assert allowed

    def test_whitelist_restricts(self):
        """Test whitelist restricts topics."""
        topic_gov = TopicGovernance()

        topic_gov.allow_topic("research")

        # Allowed topic
        allowed, _ = topic_gov.is_topic_allowed("research", "agent1")
        assert allowed

        # Not in whitelist
        allowed, _ = topic_gov.is_topic_allowed("random", "agent1")
        assert not allowed

    def test_forbidden_overrides_whitelist(self):
        """Test forbidden topics override whitelist."""
        topic_gov = TopicGovernance()

        topic_gov.allow_topic("security")
        topic_gov.forbid_topic("security")  # Forbid after allowing

        # Should still be forbidden
        allowed, _ = topic_gov.is_topic_allowed("security", "agent1")
        assert not allowed


class TestPolicyEngine:
    """Test Policy Engine."""

    def test_rbac_check(self):
        """Test RBAC check blocks restricted agents."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("restricted", AgentRole.RESTRICTED)

        topic_gov = TopicGovernance()
        engine = PolicyEngine(rbac, topic_gov)

        request = CommunicationRequest(
            from_agent="restricted",
            to_agent="worker",
            message_type="query",
            topic="general",
            content="Test",
            priority=Priority.MEDIUM
        )

        decision, reason = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY
        assert "RBAC" in reason

    def test_topic_check(self):
        """Test topic check blocks forbidden topics."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)

        topic_gov = TopicGovernance()
        topic_gov.forbid_topic("security")

        engine = PolicyEngine(rbac, topic_gov)

        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="security",
            content="Test",
            priority=Priority.MEDIUM
        )

        decision, reason = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY
        assert "Topic" in reason

    def test_policy_evaluation(self):
        """Test policy rule evaluation."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)

        topic_gov = TopicGovernance()
        topic_gov.allow_topic("general")

        # Create policy that denies low priority
        policy = GovernancePolicy(
            policy_id="test_policy",
            name="Test Policy",
            rules=[
                PolicyRule(
                    rule_id="deny_low",
                    name="Deny low priority",
                    description="Test rule",
                    condition=lambda ctx: ctx.get("priority") == Priority.LOW,
                    decision=PolicyDecision.DENY,
                    priority=100
                )
            ]
        )

        engine = PolicyEngine(rbac, topic_gov)
        engine.register_policy(policy)

        # Test low priority (should deny)
        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="general",
            content="Test",
            priority=Priority.LOW
        )

        decision, reason = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY

        # Test high priority (should allow)
        request.priority = Priority.HIGH
        decision, reason = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.ALLOW

    def test_policy_priority(self):
        """Test policy rules evaluated by priority."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)

        topic_gov = TopicGovernance()
        topic_gov.allow_topic("general")

        # Create policy with conflicting rules
        policy = GovernancePolicy(
            policy_id="test_policy",
            name="Test Policy",
            rules=[
                PolicyRule(
                    rule_id="allow_all",
                    name="Allow all",
                    description="Allow everything",
                    condition=lambda ctx: True,
                    decision=PolicyDecision.ALLOW,
                    priority=50  # Lower priority
                ),
                PolicyRule(
                    rule_id="deny_critical",
                    name="Deny critical",
                    description="Deny critical priority",
                    condition=lambda ctx: ctx.get("priority") == Priority.CRITICAL,
                    decision=PolicyDecision.DENY,
                    priority=100  # Higher priority
                )
            ]
        )

        engine = PolicyEngine(rbac, topic_gov)
        engine.register_policy(policy)

        # Critical priority should be denied (higher priority rule)
        request = CommunicationRequest(
            from_agent="worker1",
            to_agent="worker2",
            message_type="query",
            topic="general",
            content="Test",
            priority=Priority.CRITICAL
        )

        decision, reason = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY

    def test_audit_trail(self):
        """Test audit trail records decisions."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)

        topic_gov = TopicGovernance()
        topic_gov.allow_topic("general")

        engine = PolicyEngine(rbac, topic_gov)

        # Make several requests
        for i in range(5):
            request = CommunicationRequest(
                from_agent="worker1",
                to_agent="worker2",
                message_type="query",
                topic="general",
                content=f"Test {i}",
                priority=Priority.MEDIUM
            )
            engine.evaluate_communication_request(request)

        # Check audit trail
        audit = engine.get_audit_trail(limit=10)
        assert len(audit) == 5

        # Check filtering
        audit_filtered = engine.get_audit_trail(from_agent="worker1", limit=10)
        assert len(audit_filtered) == 5

    def test_statistics(self):
        """Test statistics calculation."""
        rbac = RoleBasedAccessControl()
        rbac.assign_role("worker1", AgentRole.WORKER)
        rbac.assign_role("restricted", AgentRole.RESTRICTED)

        topic_gov = TopicGovernance()
        topic_gov.allow_topic("general")

        engine = PolicyEngine(rbac, topic_gov)

        # Make 3 allowed requests
        for i in range(3):
            request = CommunicationRequest(
                from_agent="worker1",
                to_agent="worker2",
                message_type="query",
                topic="general",
                content=f"Test {i}",
                priority=Priority.MEDIUM
            )
            engine.evaluate_communication_request(request)

        # Make 2 denied requests (restricted agent)
        for i in range(2):
            request = CommunicationRequest(
                from_agent="restricted",
                to_agent="worker1",
                message_type="query",
                topic="general",
                content=f"Test {i}",
                priority=Priority.MEDIUM
            )
            engine.evaluate_communication_request(request)

        stats = engine.get_statistics()
        assert stats["total_decisions"] == 5
        assert stats["allow_count"] == 3
        assert stats["deny_count"] == 2
        assert stats["allow_rate"] == 0.6
        assert stats["deny_rate"] == 0.4


class TestPolicyTemplates:
    """Test Policy Templates."""

    def test_development_template(self):
        """Test development template allows everything."""
        policy = PolicyTemplates.development()

        assert policy.policy_id == "dev_policy"
        assert policy.default_decision == PolicyDecision.ALLOW
        assert len(policy.rules) == 1

        # Development policy should allow everything
        context = {
            "from_agent": "worker1",
            "to_agent": "worker2",
            "priority": Priority.LOW
        }
        decision, _ = policy.evaluate(context)
        assert decision == PolicyDecision.ALLOW

    def test_production_template(self):
        """Test production template."""
        policy = PolicyTemplates.production()

        assert policy.policy_id == "prod_policy"
        assert len(policy.rules) == 3

        # Critical priority should be allowed
        context = {
            "from_agent": "worker1",
            "to_agent": "worker2",
            "priority": Priority.CRITICAL,
            "topic": "general"
        }
        decision, _ = policy.evaluate(context)
        assert decision == PolicyDecision.ALLOW

        # Sensitive topics should escalate
        context = {
            "from_agent": "worker1",
            "to_agent": "worker2",
            "priority": Priority.MEDIUM,
            "topic": "security vulnerability"
        }
        decision, _ = policy.evaluate(context)
        assert decision == PolicyDecision.ESCALATE

    def test_enterprise_template(self):
        """Test enterprise template."""
        policy = PolicyTemplates.enterprise()

        assert policy.policy_id == "enterprise_policy"
        assert policy.default_decision == PolicyDecision.DENY  # Deny by default
        assert len(policy.rules) == 5

        # Admin should be allowed
        context = {
            "from_agent": "admin",
            "to_agent": "worker",
            "from_role": AgentRole.ADMIN,
            "priority": Priority.MEDIUM,
            "topic": "general"
        }
        decision, _ = policy.evaluate(context)
        assert decision == PolicyDecision.ALLOW

        # Restricted should be denied
        context = {
            "from_agent": "restricted",
            "to_agent": "worker",
            "from_role": AgentRole.RESTRICTED,
            "priority": Priority.MEDIUM,
            "topic": "general"
        }
        decision, _ = policy.evaluate(context)
        assert decision == PolicyDecision.DENY


class TestIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self):
        """Test complete policy evaluation workflow."""
        # Setup
        rbac = RoleBasedAccessControl()
        rbac.assign_role("admin", AgentRole.ADMIN)
        rbac.assign_role("coordinator", AgentRole.COORDINATOR)
        rbac.assign_role("worker", AgentRole.WORKER)
        rbac.assign_role("observer", AgentRole.OBSERVER)

        topic_gov = TopicGovernance()
        topic_gov.allow_topic("research")
        topic_gov.allow_topic("development")
        topic_gov.forbid_topic("security")
        topic_gov.restrict_topic("confidential", ["admin", "coordinator"])

        engine = PolicyEngine(rbac, topic_gov)
        engine.register_policy(PolicyTemplates.production())

        # Test 1: Admin on any topic
        request = CommunicationRequest(
            from_agent="admin",
            to_agent="worker",
            message_type="query",
            topic="research",
            content="Test",
            priority=Priority.MEDIUM
        )
        decision, _ = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.ALLOW

        # Test 2: Worker on allowed topic
        request = CommunicationRequest(
            from_agent="worker",
            to_agent="coordinator",
            message_type="query",
            topic="development",
            content="Test",
            priority=Priority.MEDIUM
        )
        decision, _ = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.ALLOW

        # Test 3: Worker on forbidden topic
        request = CommunicationRequest(
            from_agent="worker",
            to_agent="coordinator",
            message_type="query",
            topic="security",
            content="Test",
            priority=Priority.MEDIUM
        )
        decision, _ = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY

        # Test 4: Worker on restricted topic
        request = CommunicationRequest(
            from_agent="worker",
            to_agent="coordinator",
            message_type="query",
            topic="confidential",
            content="Test",
            priority=Priority.MEDIUM
        )
        decision, _ = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY

        # Test 5: Observer trying to send
        request = CommunicationRequest(
            from_agent="observer",
            to_agent="worker",
            message_type="query",
            topic="research",
            content="Test",
            priority=Priority.MEDIUM
        )
        decision, _ = engine.evaluate_communication_request(request)
        assert decision == PolicyDecision.DENY

        # Verify statistics
        stats = engine.get_statistics()
        assert stats["total_decisions"] == 5
        assert stats["allow_count"] == 2
        assert stats["deny_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
