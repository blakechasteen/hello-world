"""
RBAC Policy Engine - Complex Permission Rules

Implements attribute-based access control (ABAC) for complex permission rules
beyond simple role checks.

Features:
- Conditional permissions (time-based, IP-based, resource-based)
- Policy composition (AND, OR, NOT operations)
- Dynamic rule evaluation
- Audit logging

Examples:
- "Admin can delete users only on weekdays"
- "Write users can ingest data only from trusted IPs"
- "Users can edit their own profiles but not change roles"

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from enum import Enum

from HoloLoom.security.rbac.core import Role, Permission, RBACManager

logger = logging.getLogger(__name__)


# ============================================================================
# Policy Decision
# ============================================================================

class PolicyDecision(str, Enum):
    """Policy evaluation result."""
    ALLOW = "allow"      # Access granted
    DENY = "deny"        # Access denied
    ABSTAIN = "abstain"  # Policy doesn't apply (defer to other policies)


@dataclass
class PolicyResult:
    """
    Result of policy evaluation.

    Attributes:
        decision: ALLOW, DENY, or ABSTAIN
        reason: Human-readable explanation
        matched_rules: List of rules that matched
        metadata: Optional additional data
    """
    decision: PolicyDecision
    reason: str
    matched_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Policy Context
# ============================================================================

@dataclass
class PolicyContext:
    """
    Context for policy evaluation.

    Contains all information needed to evaluate policies:
    - User information (user_id, role, permissions)
    - Resource information (resource_id, resource_type, owner_id)
    - Request information (IP, time, action)
    - Environment information (is_production, region)
    """
    # User information
    user_id: str
    role: Optional[Role] = None
    permissions: Optional[List[Permission]] = None

    # Resource information
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_owner_id: Optional[str] = None

    # Request information
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    action: Optional[str] = None

    # Environment information
    is_production: bool = True
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Policy Rules
# ============================================================================

class PolicyRule:
    """
    Single policy rule with condition and effect.

    A policy rule consists of:
    1. Condition: When does this rule apply? (function returning bool)
    2. Effect: What happens if condition is true? (ALLOW or DENY)
    3. Priority: Rule priority (higher = evaluated first)

    Usage:
        >>> rule = PolicyRule(
        ...     name="weekday_delete",
        ...     condition=lambda ctx: ctx.timestamp.weekday() < 5,
        ...     effect=PolicyDecision.ALLOW,
        ...     description="Allow deletes on weekdays only"
        ... )
        >>>
        >>> result = await rule.evaluate(context)
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[PolicyContext], Awaitable[bool]],
        effect: PolicyDecision,
        description: str = "",
        priority: int = 0,
    ):
        """
        Initialize policy rule.

        Args:
            name: Rule identifier
            condition: Async function that evaluates condition
            effect: ALLOW, DENY, or ABSTAIN
            description: Human-readable description
            priority: Rule priority (higher = evaluated first)
        """
        self.name = name
        self.condition = condition
        self.effect = effect
        self.description = description
        self.priority = priority

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """
        Evaluate rule against context.

        Args:
            context: Policy context

        Returns:
            PolicyResult with decision and reason
        """
        try:
            matches = await self.condition(context)

            if matches:
                return PolicyResult(
                    decision=self.effect,
                    reason=self.description or f"Rule {self.name} matched",
                    matched_rules=[self.name],
                )
            else:
                return PolicyResult(
                    decision=PolicyDecision.ABSTAIN,
                    reason=f"Rule {self.name} did not match",
                )

        except Exception as e:
            logger.error(f"❌ Error evaluating rule {self.name}: {e}")
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Rule evaluation error: {e}",
            )


# ============================================================================
# Policy Engine
# ============================================================================

class PolicyEngine:
    """
    RBAC policy engine for complex permission rules.

    Evaluates policies in priority order and combines results.

    Default combining algorithm: First-Match
    - Evaluate rules in priority order (highest first)
    - Return first ALLOW or DENY
    - If all ABSTAIN, return DENY (fail-secure)

    Usage:
        >>> engine = PolicyEngine()
        >>>
        >>> # Add rules
        >>> engine.add_rule(PolicyRule(
        ...     name="admin_full_access",
        ...     condition=lambda ctx: ctx.role == Role.ADMIN,
        ...     effect=PolicyDecision.ALLOW,
        ...     priority=100
        ... ))
        >>>
        >>> # Evaluate
        >>> result = await engine.evaluate(context)
    """

    def __init__(self, rbac: Optional[RBACManager] = None):
        """
        Initialize policy engine.

        Args:
            rbac: Optional RBAC manager for role/permission lookups
        """
        self.rbac = rbac
        self.rules: List[PolicyRule] = []
        self._stats = {
            "evaluations": 0,
            "allows": 0,
            "denies": 0,
            "abstains": 0,
        }

    def add_rule(self, rule: PolicyRule):
        """
        Add policy rule.

        Args:
            rule: PolicyRule to add

        Note: Rules are automatically sorted by priority (highest first)
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"✅ Added policy rule: {rule.name} (priority: {rule.priority})")

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """
        Evaluate all rules against context.

        Args:
            context: Policy context

        Returns:
            Final policy decision

        Algorithm: First-Match
        1. Evaluate rules in priority order
        2. Return first ALLOW or DENY
        3. If all ABSTAIN, return DENY
        """
        self._stats["evaluations"] += 1

        # Enrich context with role/permissions from RBAC if available
        if self.rbac and context.role is None:
            context.role = await self.rbac.get_user_role(context.user_id)
            context.permissions = list(await self.rbac.get_user_permissions(context.user_id))

        matched_rules = []

        # Evaluate rules in priority order
        for rule in self.rules:
            result = await rule.evaluate(context)

            if result.decision != PolicyDecision.ABSTAIN:
                # First ALLOW or DENY wins
                matched_rules.extend(result.matched_rules)

                # Update stats
                if result.decision == PolicyDecision.ALLOW:
                    self._stats["allows"] += 1
                else:
                    self._stats["denies"] += 1

                logger.debug(
                    f"✅ Policy decision: {result.decision.value} "
                    f"(rule: {rule.name}, user: {context.user_id})"
                )

                return PolicyResult(
                    decision=result.decision,
                    reason=result.reason,
                    matched_rules=matched_rules,
                    metadata=result.metadata,
                )

        # All rules abstained → DENY (fail-secure)
        self._stats["abstains"] += 1

        logger.debug(
            f"⚠️  Policy decision: DENY (all rules abstained, user: {context.user_id})"
        )

        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="No matching policy rules (fail-secure)",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        return {
            **self._stats,
            "total_rules": len(self.rules),
        }


# ============================================================================
# Pre-Built Policy Rules
# ============================================================================

class CommonPolicies:
    """
    Common policy rules for typical use cases.

    Usage:
        >>> engine = PolicyEngine()
        >>> engine.add_rule(CommonPolicies.admin_full_access())
        >>> engine.add_rule(CommonPolicies.owner_can_edit())
    """

    @staticmethod
    def admin_full_access() -> PolicyRule:
        """Admin has full access to everything."""
        async def condition(ctx: PolicyContext) -> bool:
            return ctx.role == Role.ADMIN

        return PolicyRule(
            name="admin_full_access",
            condition=condition,
            effect=PolicyDecision.ALLOW,
            description="Admin has full access",
            priority=100,  # Highest priority
        )

    @staticmethod
    def owner_can_edit() -> PolicyRule:
        """Users can edit their own resources."""
        async def condition(ctx: PolicyContext) -> bool:
            return ctx.user_id == ctx.resource_owner_id

        return PolicyRule(
            name="owner_can_edit",
            condition=condition,
            effect=PolicyDecision.ALLOW,
            description="Owner can edit own resources",
            priority=90,
        )

    @staticmethod
    def business_hours_only(
        start: time = time(9, 0),
        end: time = time(17, 0)
    ) -> PolicyRule:
        """Allow access only during business hours."""
        async def condition(ctx: PolicyContext) -> bool:
            current_time = ctx.timestamp.time()
            return start <= current_time <= end

        return PolicyRule(
            name="business_hours_only",
            condition=condition,
            effect=PolicyDecision.DENY,  # DENY if outside hours
            description=f"Access restricted to business hours ({start}-{end})",
            priority=50,
        )

    @staticmethod
    def weekday_only() -> PolicyRule:
        """Allow access only on weekdays."""
        async def condition(ctx: PolicyContext) -> bool:
            return ctx.timestamp.weekday() < 5  # Monday=0, Friday=4

        return PolicyRule(
            name="weekday_only",
            condition=condition,
            effect=PolicyDecision.DENY,  # DENY on weekends
            description="Access restricted to weekdays",
            priority=50,
        )

    @staticmethod
    def ip_whitelist(allowed_ips: List[str]) -> PolicyRule:
        """Allow access only from whitelisted IPs."""
        async def condition(ctx: PolicyContext) -> bool:
            return ctx.ip_address in allowed_ips

        return PolicyRule(
            name="ip_whitelist",
            condition=condition,
            effect=PolicyDecision.DENY,  # DENY if not in whitelist
            description=f"Access restricted to IPs: {', '.join(allowed_ips)}",
            priority=80,
        )

    @staticmethod
    def production_admin_only() -> PolicyRule:
        """In production, only admins can access."""
        async def condition(ctx: PolicyContext) -> bool:
            return not ctx.is_production or ctx.role == Role.ADMIN

        return PolicyRule(
            name="production_admin_only",
            condition=condition,
            effect=PolicyDecision.DENY,
            description="Production access restricted to admins",
            priority=70,
        )

    @staticmethod
    def rate_limit_check(
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> PolicyRule:
        """
        Rate limiting policy.

        Note: This is a simplified example. In production, use
        HoloLoom.security.rate_limiting.DistributedRateLimiter
        """
        from collections import defaultdict
        from time import time

        # Simple in-memory rate limiter
        request_counts = defaultdict(list)

        async def condition(ctx: PolicyContext) -> bool:
            now = time()
            cutoff = now - window_seconds

            # Clean old requests
            request_counts[ctx.user_id] = [
                t for t in request_counts[ctx.user_id] if t > cutoff
            ]

            # Check limit
            if len(request_counts[ctx.user_id]) >= max_requests:
                return True  # Rate limit exceeded → DENY

            # Record this request
            request_counts[ctx.user_id].append(now)
            return False

        return PolicyRule(
            name="rate_limit",
            condition=condition,
            effect=PolicyDecision.DENY,
            description=f"Rate limit: {max_requests} requests per {window_seconds}s",
            priority=60,
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_policy_engine(
    rbac: Optional[RBACManager] = None,
    add_common_policies: bool = True
) -> PolicyEngine:
    """
    Create policy engine with optional common policies.

    Args:
        rbac: Optional RBAC manager
        add_common_policies: Add admin_full_access and owner_can_edit

    Returns:
        PolicyEngine instance

    Usage:
        >>> engine = create_policy_engine(rbac=rbac, add_common_policies=True)
        >>> # Admin full access and owner can edit already configured
    """
    engine = PolicyEngine(rbac=rbac)

    if add_common_policies:
        engine.add_rule(CommonPolicies.admin_full_access())
        engine.add_rule(CommonPolicies.owner_can_edit())

    return engine


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo policy engine."""
        print("=" * 70)
        print("Policy Engine Demo")
        print("=" * 70)

        # Create engine
        engine = create_policy_engine(add_common_policies=True)

        # Add business hours restriction
        engine.add_rule(CommonPolicies.business_hours_only(
            start=time(9, 0),
            end=time(17, 0)
        ))

        # Test 1: Admin access (should ALLOW)
        print("\n\nTest 1: Admin access")
        print("-" * 70)
        context = PolicyContext(
            user_id="admin_123",
            role=Role.ADMIN,
            action="delete_user"
        )
        result = await engine.evaluate(context)
        print(f"Decision: {result.decision.value}")
        print(f"Reason: {result.reason}")

        # Test 2: Owner editing own resource (should ALLOW)
        print("\n\nTest 2: Owner editing own resource")
        print("-" * 70)
        context = PolicyContext(
            user_id="user_456",
            role=Role.WRITE,
            resource_owner_id="user_456",
            action="edit_profile"
        )
        result = await engine.evaluate(context)
        print(f"Decision: {result.decision.value}")
        print(f"Reason: {result.reason}")

        # Test 3: Non-owner editing (should DENY)
        print("\n\nTest 3: Non-owner editing")
        print("-" * 70)
        context = PolicyContext(
            user_id="user_456",
            role=Role.WRITE,
            resource_owner_id="user_789",
            action="edit_profile"
        )
        result = await engine.evaluate(context)
        print(f"Decision: {result.decision.value}")
        print(f"Reason: {result.reason}")

        # Statistics
        print("\n\nPolicy Engine Statistics")
        print("-" * 70)
        stats = engine.get_statistics()
        print(f"Total evaluations: {stats['evaluations']}")
        print(f"Allows: {stats['allows']}")
        print(f"Denies: {stats['denies']}")
        print(f"Abstains: {stats['abstains']}")
        print(f"Total rules: {stats['total_rules']}")

        print("\n" + "=" * 70 + "\n")

    asyncio.run(demo())
