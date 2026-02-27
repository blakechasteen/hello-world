"""
Policy & Governance as Living System (Addendum A.3)

Implements tiered governance architecture with clear authority,
change management, and conflict resolution.

"Governance isn't a configuration file—it's a living system with
clear authority, change management, and conflict resolution."

TIER 0: IMMUTABLE CONSTRAINTS
    - Never executable without human approval: rm -rf, DROP TABLE
    - Never reveal: API keys, passwords, PII without consent
    - Hardcoded, cannot be overridden by any policy
    - Enforcement: Pre-flight block (no exceptions)

TIER 1: ORGANIZATIONAL POLICIES
    - Set by: Organization admins
    - Examples: HIPAA compliance, data retention, logging levels
    - Override: Requires admin approval + audit log
    - Enforcement: Safety guardrails (escalation possible)

TIER 2: DEPARTMENTAL POLICIES
    - Set by: Department leads
    - Examples: Code style, review requirements, response tone
    - Override: Department lead approval
    - Enforcement: Soft enforcement (warnings, not blocks)

TIER 3: USER PREFERENCES
    - Set by: Individual users
    - Examples: Verbosity, formatting, domain focus
    - Override: User can change anytime
    - Enforcement: Best-effort (preferences, not requirements)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import List, Optional, Dict, Any, Callable
import asyncio


class PolicyTier(IntEnum):
    """Policy tiers from most to least authoritative."""
    TIER_0_IMMUTABLE = 0      # Hardcoded, cannot be overridden
    TIER_1_ORGANIZATIONAL = 1  # Set by org admins
    TIER_2_DEPARTMENTAL = 2    # Set by department leads
    TIER_3_USER = 3            # Individual preferences

    # Short aliases for backward compatibility
    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class DecisionType(Enum):
    """Types of policy decisions."""
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"
    WARN = "warn"


# Alias for backward compatibility with tests
Decision = DecisionType


@dataclass
class Policy:
    """A single governance policy."""
    name: str
    tier: PolicyTier
    description: str
    patterns: List[str] = field(default_factory=list)  # Patterns to match
    decision: DecisionType = DecisionType.ALLOW
    set_by: str = "system"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, request: Any) -> "PolicyEvaluation":
        """Evaluate this policy against a request."""
        # Default implementation - subclasses can override
        return PolicyEvaluation(
            policy=self,
            decision=self.decision,
            confidence=1.0,
            reason=f"Policy {self.name} evaluated",
        )


@dataclass
class PolicyEvaluation:
    """Result of evaluating a single policy."""
    policy: Policy
    decision: DecisionType
    confidence: float
    reason: str
    caveats: List[str] = field(default_factory=list)


@dataclass
class PolicyDecision:
    """
    Final decision after evaluating all applicable policies.

    Per the governance architecture:
    - Tier 0 always wins (no override possible)
    - For other tiers, most restrictive wins
    - Escalation paths are provided for overrides
    """
    decision: DecisionType
    winning_policy: Optional[Policy] = None
    reason: str = ""
    override_possible: bool = True
    escalation_path: Optional[str] = None
    policies_evaluated: List[str] = field(default_factory=list)
    policies_satisfied: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    audit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision.value,
            "winning_policy": self.winning_policy.name if self.winning_policy else None,
            "reason": self.reason,
            "override_possible": self.override_possible,
            "escalation_path": self.escalation_path,
            "policies_evaluated": self.policies_evaluated,
            "policies_satisfied": self.policies_satisfied,
            "caveats": self.caveats,
            "timestamp": self.timestamp.isoformat(),
            "audit_id": self.audit_id,
        }


# Default immutable constraints (Tier 0)
IMMUTABLE_PATTERNS = {
    # Dangerous file operations
    r"rm\s+-rf",
    r"rm\s+--no-preserve-root",
    r"rmdir\s+/",
    # Database destruction
    r"DROP\s+DATABASE",
    r"DROP\s+TABLE",
    r"TRUNCATE\s+TABLE",
    # Credential exposure
    r"(?i)(api[_-]?key|password|secret|token)\s*[=:]\s*['\"][^'\"]+['\"]",
    # System commands
    r"shutdown\s+-[hPr]",
    r"init\s+0",
    # Format commands
    r"mkfs\.",
    r"dd\s+if=.+of=/dev/",
}

# Secrets that should never be revealed
SECRET_PATTERNS = {
    r"(?i)api[_-]?key",
    r"(?i)secret[_-]?key",
    r"(?i)access[_-]?token",
    r"(?i)private[_-]?key",
    r"(?i)password",
    r"(?i)credential",
}


def get_escalation_path(tier: PolicyTier) -> str:
    """Get the escalation path for a given tier."""
    paths = {
        PolicyTier.TIER_0_IMMUTABLE: "No override possible",
        PolicyTier.TIER_1_ORGANIZATIONAL: "Escalate to organization admin",
        PolicyTier.TIER_2_DEPARTMENTAL: "Escalate to department lead",
        PolicyTier.TIER_3_USER: "User can override directly",
    }
    return paths.get(tier, "Unknown escalation path")


async def resolve_policy_conflict(
    request: Any,
    applicable_policies: List[Policy],
) -> PolicyDecision:
    """
    Resolve conflicts between multiple applicable policies.

    Per the governance architecture:
    - Sort by tier (lower tier = higher authority)
    - Tier 0 always wins
    - For other tiers, most restrictive wins
    """
    if not applicable_policies:
        return PolicyDecision(
            decision=DecisionType.ALLOW,
            reason="No policies applicable",
            policies_evaluated=[],
        )

    # Sort by tier (lower tier = higher authority)
    sorted_policies = sorted(applicable_policies, key=lambda p: p.tier)

    # Track all evaluated policies
    evaluated = [p.name for p in sorted_policies]

    # Tier 0 always wins
    tier_0_policies = [p for p in sorted_policies if p.tier == PolicyTier.TIER_0_IMMUTABLE]
    if tier_0_policies:
        blocking = [p for p in tier_0_policies if p.decision == DecisionType.BLOCK]
        if blocking:
            return PolicyDecision(
                decision=DecisionType.BLOCK,
                winning_policy=blocking[0],
                reason="Immutable constraint violation",
                override_possible=False,
                escalation_path="No override possible",
                policies_evaluated=evaluated,
            )

    # Evaluate all policies
    evaluations: List[PolicyEvaluation] = []
    for policy in sorted_policies:
        try:
            evaluation = policy.evaluate(request)
            evaluations.append(evaluation)
        except Exception:
            # Policy evaluation failed, skip
            continue

    # For other tiers, most restrictive wins
    blocking_evals = [e for e in evaluations if e.decision == DecisionType.BLOCK]
    if blocking_evals:
        # Find highest authority (lowest tier) blocking policy
        blocker = min(blocking_evals, key=lambda e: e.policy.tier)
        return PolicyDecision(
            decision=DecisionType.BLOCK,
            winning_policy=blocker.policy,
            reason=blocker.reason,
            override_possible=(blocker.policy.tier > PolicyTier.TIER_0_IMMUTABLE),
            escalation_path=get_escalation_path(blocker.policy.tier),
            policies_evaluated=evaluated,
            caveats=blocker.caveats,
        )

    # Check for escalations
    escalating_evals = [e for e in evaluations if e.decision == DecisionType.ESCALATE]
    if escalating_evals:
        escalator = min(escalating_evals, key=lambda e: e.policy.tier)
        return PolicyDecision(
            decision=DecisionType.ESCALATE,
            winning_policy=escalator.policy,
            reason=escalator.reason,
            override_possible=True,
            escalation_path=get_escalation_path(escalator.policy.tier),
            policies_evaluated=evaluated,
        )

    # Check for warnings
    warning_evals = [e for e in evaluations if e.decision == DecisionType.WARN]

    # All policies allow, collect caveats and warnings
    all_caveats = []
    for e in evaluations:
        all_caveats.extend(e.caveats)

    return PolicyDecision(
        decision=DecisionType.ALLOW,
        reason="All policies satisfied",
        policies_evaluated=evaluated,
        policies_satisfied=[e.policy.name for e in evaluations if e.decision == DecisionType.ALLOW],
        caveats=all_caveats + [e.reason for e in warning_evals],
    )


class GovernanceEngine:
    """
    Engine for governance policy evaluation and management.

    Implements:
    - Policy registration and lookup
    - Conflict resolution
    - Audit logging
    - Version management
    """

    def __init__(
        self,
        enable_audit: bool = True,
        audit_callback: Optional[Callable[[PolicyDecision], None]] = None,
    ):
        """
        Initialize governance engine.

        Args:
            enable_audit: Whether to log all decisions
            audit_callback: Optional callback for audit events
        """
        self.policies: Dict[str, Policy] = {}
        self.enable_audit = enable_audit
        self.audit_callback = audit_callback
        self._audit_counter = 0

        # Register default Tier 0 policies
        self._register_immutable_policies()

    def _register_immutable_policies(self):
        """Register default immutable (Tier 0) policies."""
        import re

        # Dangerous commands policy
        dangerous_policy = Policy(
            name="tier0.dangerous_commands",
            tier=PolicyTier.TIER_0_IMMUTABLE,
            description="Block dangerous system commands",
            patterns=list(IMMUTABLE_PATTERNS),
            decision=DecisionType.BLOCK,
            set_by="system_default",
        )
        self.register_policy(dangerous_policy)

        # Secrets policy
        secrets_policy = Policy(
            name="tier0.no_secrets",
            tier=PolicyTier.TIER_0_IMMUTABLE,
            description="Never reveal API keys, passwords, or secrets",
            patterns=list(SECRET_PATTERNS),
            decision=DecisionType.BLOCK,
            set_by="system_default",
        )
        self.register_policy(secrets_policy)

    def register_policy(self, policy: Policy) -> None:
        """
        Register a policy.

        Args:
            policy: The policy to register
        """
        self.policies[policy.name] = policy

    def unregister_policy(self, name: str) -> bool:
        """
        Unregister a policy.

        Args:
            name: Policy name to unregister

        Returns:
            True if policy was found and removed
        """
        if name in self.policies:
            # Cannot unregister Tier 0 policies
            if self.policies[name].tier == PolicyTier.TIER_0_IMMUTABLE:
                return False
            del self.policies[name]
            return True
        return False

    def get_applicable_policies(
        self,
        request: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Policy]:
        """
        Get all policies that apply to a request.

        Args:
            request: The request to evaluate
            context: Optional context (user, department, etc.)

        Returns:
            List of applicable policies
        """
        import re

        applicable = []
        request_str = str(request) if not isinstance(request, str) else request

        for policy in self.policies.values():
            # Check if any pattern matches
            for pattern in policy.patterns:
                try:
                    if re.search(pattern, request_str):
                        applicable.append(policy)
                        break
                except re.error:
                    # Invalid pattern, skip
                    continue

            # If no patterns, policy applies by default
            if not policy.patterns:
                applicable.append(policy)

        return applicable

    async def evaluate(
        self,
        request: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Evaluate a request against all applicable policies.

        Args:
            request: The request to evaluate
            context: Optional context

        Returns:
            PolicyDecision with final verdict
        """
        # Get applicable policies
        applicable = self.get_applicable_policies(request, context)

        # Resolve conflicts
        decision = await resolve_policy_conflict(request, applicable)

        # Generate audit ID
        if self.enable_audit:
            self._audit_counter += 1
            decision.audit_id = f"aud-{self._audit_counter:08d}"

            # Call audit callback if provided
            if self.audit_callback:
                try:
                    self.audit_callback(decision)
                except Exception:
                    pass  # Don't let audit failures affect decisions

        return decision

    def should_block(self, request: Any) -> tuple[bool, str]:
        """
        Synchronous check if a request should be blocked.

        Args:
            request: The request to check

        Returns:
            Tuple of (should_block, reason)
        """
        import re

        request_str = str(request) if not isinstance(request, str) else request

        # Check Tier 0 patterns directly for fast path
        for pattern in IMMUTABLE_PATTERNS:
            try:
                if re.search(pattern, request_str):
                    return True, f"Blocked by immutable constraint: {pattern}"
            except re.error:
                continue

        # Check secrets patterns
        for pattern in SECRET_PATTERNS:
            try:
                if re.search(pattern, request_str):
                    # Only block if appears to be exposing a secret value
                    expose_pattern = pattern + r"\s*[=:]\s*['\"][^'\"]+['\"]"
                    if re.search(expose_pattern, request_str, re.IGNORECASE):
                        return True, "Blocked: Attempted secret exposure"
            except re.error:
                continue

        return False, ""

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all registered policies."""
        summary = {
            "total_policies": len(self.policies),
            "by_tier": {},
            "policies": [],
        }

        tier_counts: Dict[PolicyTier, int] = {}
        for policy in self.policies.values():
            tier_counts[policy.tier] = tier_counts.get(policy.tier, 0) + 1
            summary["policies"].append({
                "name": policy.name,
                "tier": policy.tier.value,
                "decision": policy.decision.value,
                "set_by": policy.set_by,
                "version": policy.version,
            })

        summary["by_tier"] = {
            f"tier_{tier.value}": count
            for tier, count in tier_counts.items()
        }

        return summary


# Convenience functions for common governance checks
def is_safe_for_execution(request: str) -> tuple[bool, str]:
    """
    Quick safety check for execution requests.

    Args:
        request: The request to check

    Returns:
        Tuple of (is_safe, reason_if_unsafe)
    """
    engine = GovernanceEngine(enable_audit=False)
    blocked, reason = engine.should_block(request)
    return not blocked, reason


def requires_human_approval(
    request: str,
    confidence: float,
    stakes: str = "normal",
) -> tuple[bool, str]:
    """
    Check if a request requires human approval.

    Per the governance matrix:
    - Safety-critical + LOW confidence = BLOCK (escalate)
    - High stakes + moderate confidence = require approval

    Args:
        request: The request
        confidence: Current confidence level
        stakes: "low", "normal", or "high"

    Returns:
        Tuple of (requires_approval, reason)
    """
    # Safety-critical with low confidence always requires approval
    if stakes == "high" and confidence < 0.6:
        return True, f"Low confidence ({confidence:.2f}) on high-stakes request"

    # Very low confidence requires approval regardless of stakes
    if confidence < 0.3:
        return True, f"Very low confidence ({confidence:.2f}) requires human review"

    # Check for dangerous patterns
    engine = GovernanceEngine(enable_audit=False)
    blocked, reason = engine.should_block(request)
    if blocked:
        return True, reason

    return False, ""
