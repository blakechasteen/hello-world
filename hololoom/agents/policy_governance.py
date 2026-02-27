#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy & Governance System
==========================

Policy-based decision making and governance for multi-agent systems.

Created: 2025-01-20

Philosophy:
"Every agent decision must follow clear policies, be auditable, and respect governance rules."

Features:
- Policy-based decision making (when to communicate, who to ask)
- Role-Based Access Control (RBAC)
- Topic-based permissions
- Audit trail for all decisions
- Compliance enforcement
- Policy templates (development, production, enterprise)

Policies Control:
1. Communication decisions (when to ask, who to ask)
2. Resource allocation (budget, priority)
3. Topic restrictions (allowed/forbidden topics)
4. Access control (who can talk to whom)
5. Escalation rules (when to escalate to human)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Callable
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Policy decision outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"  # Escalate to human
    DEFER = "defer"        # Defer to another agent
    AUDIT_ONLY = "audit_only"  # Allow but audit


class AgentRole(Enum):
    """Agent roles for RBAC."""
    ADMIN = "admin"           # Full access
    COORDINATOR = "coordinator"  # Can coordinate others
    WORKER = "worker"         # Basic agent
    OBSERVER = "observer"     # Read-only
    RESTRICTED = "restricted"  # Limited access


class Priority(Enum):
    """Request priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PolicyRule:
    """Individual policy rule."""
    rule_id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    decision: PolicyDecision
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernancePolicy:
    """Complete governance policy."""
    policy_id: str
    name: str
    rules: List[PolicyRule]
    default_decision: PolicyDecision = PolicyDecision.DENY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Dict[str, Any]) -> tuple[PolicyDecision, str]:
        """
        Evaluate policy against context.

        Args:
            context: Decision context

        Returns:
            (decision, reason)
        """
        # Sort rules by priority
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        # Evaluate each rule
        for rule in sorted_rules:
            try:
                if rule.condition(context):
                    logger.debug(f"Rule matched: {rule.name}")
                    return rule.decision, f"Rule: {rule.name}"
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                continue

        # No rule matched
        return self.default_decision, "Default policy"


@dataclass
class CommunicationRequest:
    """Request to communicate with another agent."""
    from_agent: str
    to_agent: str
    message_type: str
    topic: str
    content: str
    priority: Priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyAuditEntry:
    """Audit entry for policy decision."""
    timestamp: datetime
    request: CommunicationRequest
    decision: PolicyDecision
    reason: str
    policy_id: str
    enforced_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoleBasedAccessControl:
    """
    Role-Based Access Control (RBAC) for agents.

    Controls who can talk to whom based on roles.
    """

    def __init__(self):
        """Initialize RBAC."""
        self.agent_roles: Dict[str, AgentRole] = {}
        self.role_permissions: Dict[AgentRole, Set[str]] = {
            AgentRole.ADMIN: {"*"},  # All permissions
            AgentRole.COORDINATOR: {"query", "help_request", "insight"},
            AgentRole.WORKER: {"query", "answer"},
            AgentRole.OBSERVER: {"insight"},
            AgentRole.RESTRICTED: set()
        }

    def assign_role(self, agent_id: str, role: AgentRole) -> None:
        """Assign role to agent."""
        self.agent_roles[agent_id] = role
        logger.info(f"Assigned role {role.value} to {agent_id}")

    def get_role(self, agent_id: str) -> AgentRole:
        """Get agent role."""
        return self.agent_roles.get(agent_id, AgentRole.WORKER)

    def has_permission(
        self,
        agent_id: str,
        permission: str
    ) -> tuple[bool, str]:
        """
        Check if agent has permission.

        Args:
            agent_id: Agent ID
            permission: Permission to check

        Returns:
            (has_permission, reason)
        """
        role = self.get_role(agent_id)
        permissions = self.role_permissions.get(role, set())

        # Admin has all permissions
        if "*" in permissions:
            return True, f"Admin role"

        # Check specific permission
        if permission in permissions:
            return True, f"Role {role.value} has permission"

        return False, f"Role {role.value} lacks permission"

    def can_communicate(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str
    ) -> tuple[bool, str]:
        """
        Check if from_agent can send message_type to to_agent.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            message_type: Message type

        Returns:
            (allowed, reason)
        """
        from_role = self.get_role(from_agent)
        to_role = self.get_role(to_agent)

        # Observers can't send messages
        if from_role == AgentRole.OBSERVER:
            return False, "Observer role cannot send messages"

        # Restricted agents can't communicate
        if from_role == AgentRole.RESTRICTED:
            return False, "Restricted role cannot communicate"

        # Check message type permission
        has_perm, reason = self.has_permission(from_agent, message_type.lower())
        if not has_perm:
            return False, reason

        # Coordinators can talk to anyone
        if from_role == AgentRole.COORDINATOR:
            return True, "Coordinator privilege"

        # Workers can talk to each other and coordinators
        if from_role == AgentRole.WORKER:
            if to_role in [AgentRole.WORKER, AgentRole.COORDINATOR, AgentRole.ADMIN]:
                return True, "Worker-to-worker/coordinator allowed"
            return False, "Workers cannot message observers/restricted"

        return True, "Default allow"


class TopicGovernance:
    """
    Topic-based governance.

    Controls which topics can be discussed.
    """

    def __init__(self):
        """Initialize topic governance."""
        self.allowed_topics: Set[str] = set()
        self.forbidden_topics: Set[str] = set()
        self.restricted_topics: Dict[str, Set[str]] = {}  # topic → allowed agents

    def allow_topic(self, topic: str) -> None:
        """Allow topic globally."""
        self.allowed_topics.add(topic.lower())
        logger.info(f"Topic allowed: {topic}")

    def forbid_topic(self, topic: str) -> None:
        """Forbid topic globally."""
        self.forbidden_topics.add(topic.lower())
        logger.info(f"Topic forbidden: {topic}")

    def restrict_topic(self, topic: str, allowed_agents: List[str]) -> None:
        """Restrict topic to specific agents."""
        self.restricted_topics[topic.lower()] = set(allowed_agents)
        logger.info(f"Topic restricted: {topic} → {allowed_agents}")

    def is_topic_allowed(
        self,
        topic: str,
        agent_id: str
    ) -> tuple[bool, str]:
        """
        Check if agent can discuss topic.

        Args:
            topic: Topic to check
            agent_id: Agent ID

        Returns:
            (allowed, reason)
        """
        topic_lower = topic.lower()

        # Check forbidden topics
        if topic_lower in self.forbidden_topics:
            return False, f"Topic '{topic}' is forbidden"

        # Check restricted topics
        if topic_lower in self.restricted_topics:
            allowed_agents = self.restricted_topics[topic_lower]
            if agent_id not in allowed_agents:
                return False, f"Topic '{topic}' restricted to {allowed_agents}"

        # Check allowed topics (if whitelist exists)
        if self.allowed_topics and topic_lower not in self.allowed_topics:
            return False, f"Topic '{topic}' not in allowed list"

        return True, "Topic allowed"


class PolicyEngine:
    """
    Policy engine for decision making.

    Evaluates policies and makes decisions.
    """

    def __init__(
        self,
        rbac: RoleBasedAccessControl,
        topic_governance: TopicGovernance
    ):
        """
        Initialize policy engine.

        Args:
            rbac: Role-based access control
            topic_governance: Topic governance
        """
        self.rbac = rbac
        self.topic_governance = topic_governance
        self.policies: Dict[str, GovernancePolicy] = {}
        self.audit_log: List[PolicyAuditEntry] = []

        # Callbacks
        self.on_decision: Optional[Callable[[PolicyDecision, str], None]] = None
        self.on_escalation: Optional[Callable[[CommunicationRequest, str], None]] = None

    def register_policy(self, policy: GovernancePolicy) -> None:
        """Register governance policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered policy: {policy.name}")

    def evaluate_communication_request(
        self,
        request: CommunicationRequest
    ) -> tuple[PolicyDecision, str]:
        """
        Evaluate communication request against all policies.

        Args:
            request: Communication request

        Returns:
            (decision, reason)
        """
        # 1. RBAC check
        can_communicate, rbac_reason = self.rbac.can_communicate(
            request.from_agent,
            request.to_agent,
            request.message_type
        )

        if not can_communicate:
            decision = PolicyDecision.DENY
            reason = f"RBAC: {rbac_reason}"
            self._audit(request, decision, reason)
            return decision, reason

        # 2. Topic governance check
        topic_allowed, topic_reason = self.topic_governance.is_topic_allowed(
            request.topic,
            request.from_agent
        )

        if not topic_allowed:
            decision = PolicyDecision.DENY
            reason = f"Topic: {topic_reason}"
            self._audit(request, decision, reason)
            return decision, reason

        # 3. Evaluate registered policies
        context = {
            "from_agent": request.from_agent,
            "to_agent": request.to_agent,
            "message_type": request.message_type,
            "topic": request.topic,
            "content": request.content,
            "priority": request.priority,
            "metadata": request.metadata,
            "from_role": self.rbac.get_role(request.from_agent),
            "to_role": self.rbac.get_role(request.to_agent),
        }

        # Evaluate each policy
        for policy in self.policies.values():
            decision, reason = policy.evaluate(context)

            # If any policy denies, deny immediately
            if decision == PolicyDecision.DENY:
                self._audit(request, decision, f"Policy {policy.name}: {reason}")
                return decision, f"Policy {policy.name}: {reason}"

            # If policy escalates, escalate
            if decision == PolicyDecision.ESCALATE:
                self._audit(request, decision, f"Policy {policy.name}: {reason}")
                if self.on_escalation:
                    self.on_escalation(request, reason)
                return decision, f"Policy {policy.name}: {reason}"

        # All checks passed
        decision = PolicyDecision.ALLOW
        reason = "All policies passed"
        self._audit(request, decision, reason)

        # Callback
        if self.on_decision:
            self.on_decision(decision, reason)

        return decision, reason

    def _audit(
        self,
        request: CommunicationRequest,
        decision: PolicyDecision,
        reason: str
    ) -> None:
        """Audit policy decision."""
        entry = PolicyAuditEntry(
            timestamp=datetime.now(),
            request=request,
            decision=decision,
            reason=reason,
            policy_id="policy_engine",
            enforced_by="system"
        )

        self.audit_log.append(entry)

        # Keep only recent entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_trail(
        self,
        from_agent: Optional[str] = None,
        limit: int = 100
    ) -> List[PolicyAuditEntry]:
        """
        Get audit trail.

        Args:
            from_agent: Filter by agent (None = all)
            limit: Max entries

        Returns:
            Audit entries
        """
        entries = self.audit_log

        if from_agent:
            entries = [e for e in entries if e.request.from_agent == from_agent]

        return entries[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy statistics."""
        total = len(self.audit_log)
        if total == 0:
            return {
                "total_decisions": 0,
                "allow_rate": 0.0,
                "deny_rate": 0.0,
                "escalate_rate": 0.0
            }

        allow_count = sum(1 for e in self.audit_log if e.decision == PolicyDecision.ALLOW)
        deny_count = sum(1 for e in self.audit_log if e.decision == PolicyDecision.DENY)
        escalate_count = sum(1 for e in self.audit_log if e.decision == PolicyDecision.ESCALATE)

        return {
            "total_decisions": total,
            "allow_count": allow_count,
            "deny_count": deny_count,
            "escalate_count": escalate_count,
            "allow_rate": allow_count / total,
            "deny_rate": deny_count / total,
            "escalate_rate": escalate_count / total
        }


class PolicyTemplates:
    """
    Pre-built policy templates for common scenarios.

    Templates:
    - Development: Permissive (allow most communication)
    - Production: Balanced (reasonable restrictions)
    - Enterprise: Strict (heavy restrictions + auditing)
    """

    @staticmethod
    def development() -> GovernancePolicy:
        """Development policy (permissive)."""
        return GovernancePolicy(
            policy_id="dev_policy",
            name="Development Policy",
            rules=[
                PolicyRule(
                    rule_id="dev_allow_all",
                    name="Allow all communication",
                    description="Development mode - allow everything",
                    condition=lambda ctx: True,
                    decision=PolicyDecision.ALLOW,
                    priority=100
                )
            ],
            default_decision=PolicyDecision.ALLOW
        )

    @staticmethod
    def production() -> GovernancePolicy:
        """Production policy (balanced)."""
        return GovernancePolicy(
            policy_id="prod_policy",
            name="Production Policy",
            rules=[
                # Rule 1: Critical priority always allowed
                PolicyRule(
                    rule_id="prod_critical",
                    name="Allow critical priority",
                    description="Critical messages always allowed",
                    condition=lambda ctx: ctx.get("priority") == Priority.CRITICAL,
                    decision=PolicyDecision.ALLOW,
                    priority=100
                ),

                # Rule 2: Escalate if sensitive topic
                PolicyRule(
                    rule_id="prod_sensitive",
                    name="Escalate sensitive topics",
                    description="Sensitive topics require approval",
                    condition=lambda ctx: any(
                        word in ctx.get("topic", "").lower()
                        for word in ["security", "private", "confidential", "pii"]
                    ),
                    decision=PolicyDecision.ESCALATE,
                    priority=90
                ),

                # Rule 3: Rate limit low priority
                PolicyRule(
                    rule_id="prod_rate_limit",
                    name="Limit low priority",
                    description="Audit low priority messages",
                    condition=lambda ctx: ctx.get("priority") == Priority.LOW,
                    decision=PolicyDecision.AUDIT_ONLY,
                    priority=50
                )
            ],
            default_decision=PolicyDecision.ALLOW
        )

    @staticmethod
    def enterprise() -> GovernancePolicy:
        """Enterprise policy (strict)."""
        return GovernancePolicy(
            policy_id="enterprise_policy",
            name="Enterprise Policy",
            rules=[
                # Rule 1: Admin always allowed
                PolicyRule(
                    rule_id="ent_admin",
                    name="Allow admin",
                    description="Admin role always allowed",
                    condition=lambda ctx: ctx.get("from_role") == AgentRole.ADMIN,
                    decision=PolicyDecision.ALLOW,
                    priority=100
                ),

                # Rule 2: Deny restricted agents
                PolicyRule(
                    rule_id="ent_restricted",
                    name="Deny restricted",
                    description="Restricted agents cannot communicate",
                    condition=lambda ctx: ctx.get("from_role") == AgentRole.RESTRICTED,
                    decision=PolicyDecision.DENY,
                    priority=95
                ),

                # Rule 3: Escalate cross-department
                PolicyRule(
                    rule_id="ent_cross_dept",
                    name="Escalate cross-department",
                    description="Cross-department requires approval",
                    condition=lambda ctx: (
                        ctx.get("metadata", {}).get("from_department") !=
                        ctx.get("metadata", {}).get("to_department")
                    ),
                    decision=PolicyDecision.ESCALATE,
                    priority=90
                ),

                # Rule 4: Audit all insights
                PolicyRule(
                    rule_id="ent_audit_insights",
                    name="Audit insights",
                    description="All insights must be audited",
                    condition=lambda ctx: ctx.get("message_type") == "insight",
                    decision=PolicyDecision.AUDIT_ONLY,
                    priority=80
                ),

                # Rule 5: Deny low priority during peak hours
                PolicyRule(
                    rule_id="ent_peak_hours",
                    name="Deny low priority in peak hours",
                    description="No low priority during peak",
                    condition=lambda ctx: (
                        ctx.get("priority") == Priority.LOW and
                        9 <= datetime.now().hour <= 17  # Business hours
                    ),
                    decision=PolicyDecision.DENY,
                    priority=70
                )
            ],
            default_decision=PolicyDecision.DENY  # Deny by default!
        )
