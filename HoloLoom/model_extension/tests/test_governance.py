"""
Tests for Policy & Governance as Living System (Addendum A.3)

Tests the tiered governance architecture:
- Tier 0: Immutable Constraints (hardcoded, no override)
- Tier 1: Organizational Policies (admin approval required)
- Tier 2: Departmental Policies (soft enforcement)
- Tier 3: User Preferences (best-effort)
"""

import pytest
import asyncio
from datetime import datetime
from HoloLoom.model_extension.governance import (
    PolicyTier,
    DecisionType,
    Policy,
    PolicyEvaluation,
    PolicyDecision,
    GovernanceEngine,
    resolve_policy_conflict,
    get_escalation_path,
    is_safe_for_execution,
    requires_human_approval,
    IMMUTABLE_PATTERNS,
    SECRET_PATTERNS,
)


class TestPolicyTier:
    """Tests for PolicyTier enum."""

    def test_tier_values(self):
        """Test tier numeric values."""
        assert PolicyTier.TIER_0_IMMUTABLE.value == 0
        assert PolicyTier.TIER_1_ORGANIZATIONAL.value == 1
        assert PolicyTier.TIER_2_DEPARTMENTAL.value == 2
        assert PolicyTier.TIER_3_USER.value == 3

    def test_tier_ordering(self):
        """Test tier ordering (lower value = higher authority)."""
        assert PolicyTier.TIER_0_IMMUTABLE.value < PolicyTier.TIER_1_ORGANIZATIONAL.value
        assert PolicyTier.TIER_1_ORGANIZATIONAL.value < PolicyTier.TIER_2_DEPARTMENTAL.value
        assert PolicyTier.TIER_2_DEPARTMENTAL.value < PolicyTier.TIER_3_USER.value

    def test_tier_comparison(self):
        """Test that tiers can be compared."""
        tiers = [
            PolicyTier.TIER_3_USER,
            PolicyTier.TIER_0_IMMUTABLE,
            PolicyTier.TIER_2_DEPARTMENTAL,
            PolicyTier.TIER_1_ORGANIZATIONAL,
        ]
        sorted_tiers = sorted(tiers, key=lambda t: t.value)

        assert sorted_tiers[0] == PolicyTier.TIER_0_IMMUTABLE
        assert sorted_tiers[-1] == PolicyTier.TIER_3_USER


class TestDecisionType:
    """Tests for DecisionType enum."""

    def test_decision_types_exist(self):
        """Test all decision types are defined."""
        assert DecisionType.ALLOW
        assert DecisionType.BLOCK
        assert DecisionType.ESCALATE
        assert DecisionType.WARN

    def test_decision_type_values(self):
        """Test decision type string values."""
        assert DecisionType.ALLOW.value == "allow"
        assert DecisionType.BLOCK.value == "block"
        assert DecisionType.ESCALATE.value == "escalate"
        assert DecisionType.WARN.value == "warn"


class TestPolicy:
    """Tests for Policy dataclass."""

    def test_create_basic_policy(self):
        """Test creating a basic policy."""
        policy = Policy(
            name="test_policy",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="A test policy",
        )

        assert policy.name == "test_policy"
        assert policy.tier == PolicyTier.TIER_1_ORGANIZATIONAL
        assert policy.description == "A test policy"
        assert policy.decision == DecisionType.ALLOW  # Default

    def test_create_blocking_policy(self):
        """Test creating a blocking policy with patterns."""
        policy = Policy(
            name="no_rm_rf",
            tier=PolicyTier.TIER_0_IMMUTABLE,
            description="Block rm -rf commands",
            patterns=[r"rm\s+-rf"],
            decision=DecisionType.BLOCK,
        )

        assert policy.name == "no_rm_rf"
        assert policy.decision == DecisionType.BLOCK
        assert len(policy.patterns) == 1
        assert r"rm\s+-rf" in policy.patterns

    def test_policy_timestamps(self):
        """Test policy includes timestamps."""
        policy = Policy(
            name="test",
            tier=PolicyTier.TIER_2_DEPARTMENTAL,
            description="Test",
        )

        assert policy.created_at is not None
        assert policy.updated_at is not None
        assert isinstance(policy.created_at, datetime)
        assert isinstance(policy.updated_at, datetime)

    def test_policy_evaluate(self):
        """Test default policy evaluation."""
        policy = Policy(
            name="test",
            tier=PolicyTier.TIER_2_DEPARTMENTAL,
            description="Test",
            decision=DecisionType.ALLOW,
        )

        evaluation = policy.evaluate("any request")

        assert isinstance(evaluation, PolicyEvaluation)
        assert evaluation.policy == policy
        assert evaluation.decision == DecisionType.ALLOW


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_create_allow_decision(self):
        """Test creating an allow decision."""
        decision = PolicyDecision(
            decision=DecisionType.ALLOW,
            reason="All policies satisfied",
            policies_evaluated=["policy_a", "policy_b"],
            policies_satisfied=["policy_a", "policy_b"],
        )

        assert decision.decision == DecisionType.ALLOW
        assert decision.override_possible is True  # Default
        assert len(decision.policies_evaluated) == 2

    def test_create_block_decision(self):
        """Test creating a block decision."""
        blocking_policy = Policy(
            name="blocker",
            tier=PolicyTier.TIER_0_IMMUTABLE,
            description="Blocks stuff",
            decision=DecisionType.BLOCK,
        )

        decision = PolicyDecision(
            decision=DecisionType.BLOCK,
            winning_policy=blocking_policy,
            reason="Immutable constraint violation",
            override_possible=False,
            escalation_path="No override possible",
        )

        assert decision.decision == DecisionType.BLOCK
        assert decision.winning_policy == blocking_policy
        assert decision.override_possible is False

    def test_decision_to_dict(self):
        """Test JSON serialization."""
        decision = PolicyDecision(
            decision=DecisionType.ALLOW,
            reason="Test",
            policies_evaluated=["a", "b"],
        )

        d = decision.to_dict()

        assert "decision" in d
        assert "reason" in d
        assert "override_possible" in d
        assert "timestamp" in d
        assert d["decision"] == "allow"

    def test_decision_with_audit_id(self):
        """Test decision includes audit ID."""
        decision = PolicyDecision(
            decision=DecisionType.ALLOW,
            reason="Test",
            audit_id="aud-00000001",
        )

        assert decision.audit_id == "aud-00000001"
        d = decision.to_dict()
        assert d["audit_id"] == "aud-00000001"


class TestGetEscalationPath:
    """Tests for get_escalation_path function."""

    def test_tier_0_escalation(self):
        """Test Tier 0 has no override."""
        path = get_escalation_path(PolicyTier.TIER_0_IMMUTABLE)
        assert "No override" in path

    def test_tier_1_escalation(self):
        """Test Tier 1 escalates to org admin."""
        path = get_escalation_path(PolicyTier.TIER_1_ORGANIZATIONAL)
        assert "organization admin" in path.lower() or "admin" in path.lower()

    def test_tier_2_escalation(self):
        """Test Tier 2 escalates to department lead."""
        path = get_escalation_path(PolicyTier.TIER_2_DEPARTMENTAL)
        assert "department" in path.lower()

    def test_tier_3_escalation(self):
        """Test Tier 3 can be overridden by user."""
        path = get_escalation_path(PolicyTier.TIER_3_USER)
        assert "user" in path.lower()


class TestResolvePolicyConflict:
    """Tests for resolve_policy_conflict function."""

    @pytest.mark.asyncio
    async def test_no_policies(self):
        """Test resolution with no policies."""
        decision = await resolve_policy_conflict("request", [])

        assert decision.decision == DecisionType.ALLOW
        assert len(decision.policies_evaluated) == 0

    @pytest.mark.asyncio
    async def test_single_allow_policy(self):
        """Test resolution with single allowing policy."""
        policy = Policy(
            name="allow_all",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="Allow everything",
            decision=DecisionType.ALLOW,
        )

        decision = await resolve_policy_conflict("request", [policy])

        assert decision.decision == DecisionType.ALLOW

    @pytest.mark.asyncio
    async def test_tier_0_always_wins(self):
        """Test that Tier 0 blocking always wins."""
        tier_0_block = Policy(
            name="immutable_block",
            tier=PolicyTier.TIER_0_IMMUTABLE,
            description="Block",
            decision=DecisionType.BLOCK,
        )
        tier_1_allow = Policy(
            name="org_allow",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="Allow",
            decision=DecisionType.ALLOW,
        )

        decision = await resolve_policy_conflict("request", [tier_1_allow, tier_0_block])

        assert decision.decision == DecisionType.BLOCK
        assert decision.winning_policy == tier_0_block
        assert decision.override_possible is False

    @pytest.mark.asyncio
    async def test_most_restrictive_wins(self):
        """Test that most restrictive policy wins for non-tier-0."""
        tier_1_block = Policy(
            name="org_block",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="Block",
            decision=DecisionType.BLOCK,
        )
        tier_2_allow = Policy(
            name="dept_allow",
            tier=PolicyTier.TIER_2_DEPARTMENTAL,
            description="Allow",
            decision=DecisionType.ALLOW,
        )

        decision = await resolve_policy_conflict("request", [tier_2_allow, tier_1_block])

        assert decision.decision == DecisionType.BLOCK
        assert decision.winning_policy == tier_1_block
        assert decision.override_possible is True

    @pytest.mark.asyncio
    async def test_escalation_decision(self):
        """Test escalation decision."""
        escalate_policy = Policy(
            name="needs_review",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="Needs human review",
            decision=DecisionType.ESCALATE,
        )

        decision = await resolve_policy_conflict("request", [escalate_policy])

        assert decision.decision == DecisionType.ESCALATE
        assert decision.escalation_path is not None


class TestGovernanceEngine:
    """Tests for GovernanceEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a governance engine for testing."""
        return GovernanceEngine(enable_audit=True)

    def test_engine_creation(self, engine):
        """Test engine is created with default policies."""
        assert engine is not None
        # Should have default Tier 0 policies
        assert len(engine.policies) >= 2

    def test_register_policy(self, engine):
        """Test registering a new policy."""
        policy = Policy(
            name="custom_policy",
            tier=PolicyTier.TIER_2_DEPARTMENTAL,
            description="Custom departmental policy",
        )

        engine.register_policy(policy)

        assert "custom_policy" in engine.policies
        assert engine.policies["custom_policy"] == policy

    def test_unregister_policy(self, engine):
        """Test unregistering a policy."""
        policy = Policy(
            name="removable",
            tier=PolicyTier.TIER_2_DEPARTMENTAL,
            description="Can be removed",
        )
        engine.register_policy(policy)

        result = engine.unregister_policy("removable")

        assert result is True
        assert "removable" not in engine.policies

    def test_cannot_unregister_tier_0(self, engine):
        """Test cannot unregister Tier 0 policies."""
        # Default Tier 0 policies should be present
        tier_0_names = [
            name for name, p in engine.policies.items()
            if p.tier == PolicyTier.TIER_0_IMMUTABLE
        ]

        for name in tier_0_names:
            result = engine.unregister_policy(name)
            assert result is False
            assert name in engine.policies

    def test_unregister_nonexistent(self, engine):
        """Test unregistering nonexistent policy returns False."""
        result = engine.unregister_policy("does_not_exist")
        assert result is False

    def test_get_applicable_policies_by_pattern(self, engine):
        """Test getting policies that match patterns."""
        # Register a policy with a pattern
        policy = Policy(
            name="no_delete",
            tier=PolicyTier.TIER_1_ORGANIZATIONAL,
            description="Block delete commands",
            patterns=[r"delete"],
            decision=DecisionType.BLOCK,
        )
        engine.register_policy(policy)

        applicable = engine.get_applicable_policies("please delete this file")

        # Should include the no_delete policy
        policy_names = [p.name for p in applicable]
        assert "no_delete" in policy_names

    def test_get_applicable_policies_no_match(self, engine):
        """Test policies without patterns always apply."""
        policy = Policy(
            name="always_applies",
            tier=PolicyTier.TIER_3_USER,
            description="No patterns, always applies",
            patterns=[],  # No patterns
            decision=DecisionType.ALLOW,
        )
        engine.register_policy(policy)

        applicable = engine.get_applicable_policies("any request")

        policy_names = [p.name for p in applicable]
        assert "always_applies" in policy_names

    @pytest.mark.asyncio
    async def test_evaluate_safe_request(self, engine):
        """Test evaluating a safe request."""
        decision = await engine.evaluate("What is the weather?")

        assert decision.decision == DecisionType.ALLOW
        assert decision.audit_id is not None

    @pytest.mark.asyncio
    async def test_evaluate_dangerous_request(self, engine):
        """Test evaluating a dangerous request."""
        decision = await engine.evaluate("rm -rf /")

        assert decision.decision == DecisionType.BLOCK
        assert decision.override_possible is False

    def test_should_block_rm_rf(self, engine):
        """Test should_block detects rm -rf."""
        blocked, reason = engine.should_block("rm -rf /home/user")

        assert blocked is True
        assert "rm" in reason.lower() or "immutable" in reason.lower()

    def test_should_block_drop_table(self, engine):
        """Test should_block detects DROP TABLE."""
        blocked, reason = engine.should_block("DROP TABLE users;")

        assert blocked is True

    def test_should_block_safe_request(self, engine):
        """Test should_block allows safe requests."""
        blocked, reason = engine.should_block("SELECT * FROM users")

        assert blocked is False
        assert reason == ""

    def test_should_block_api_key_exposure(self, engine):
        """Test should_block detects API key exposure."""
        blocked, reason = engine.should_block('api_key = "sk-12345"')

        assert blocked is True
        assert "secret" in reason.lower() or "blocked" in reason.lower()

    def test_get_policy_summary(self, engine):
        """Test getting policy summary."""
        summary = engine.get_policy_summary()

        assert "total_policies" in summary
        assert "by_tier" in summary
        assert "policies" in summary
        assert summary["total_policies"] >= 2  # Default Tier 0 policies

    def test_audit_callback(self):
        """Test audit callback is called."""
        decisions_logged = []

        def audit_callback(decision):
            decisions_logged.append(decision)

        engine = GovernanceEngine(
            enable_audit=True,
            audit_callback=audit_callback,
        )

        # Trigger evaluation
        import asyncio
        asyncio.run(engine.evaluate("test request"))

        assert len(decisions_logged) == 1
        assert decisions_logged[0].audit_id is not None


class TestImmutablePatterns:
    """Tests for immutable constraint patterns."""

    def test_patterns_exist(self):
        """Test immutable patterns are defined."""
        assert len(IMMUTABLE_PATTERNS) > 0

    def test_rm_rf_pattern(self):
        """Test rm -rf pattern exists and matches."""
        import re
        patterns_str = " ".join(IMMUTABLE_PATTERNS)
        assert "rm" in patterns_str

        # Test that at least one pattern matches "rm -rf /"
        test_string = "rm -rf /"
        matched = any(
            re.search(pattern, test_string) is not None
            for pattern in IMMUTABLE_PATTERNS
            if "rm" in pattern.lower()
        )
        assert matched, f"No pattern matched '{test_string}'"

    def test_drop_table_pattern(self):
        """Test DROP TABLE pattern exists and matches."""
        import re
        # Test that at least one pattern matches "DROP TABLE users"
        test_string = "DROP TABLE users"
        matched = any(
            re.search(pattern, test_string, re.IGNORECASE) is not None
            for pattern in IMMUTABLE_PATTERNS
            if "DROP" in pattern
        )
        assert matched, f"No pattern matched '{test_string}'"


class TestSecretPatterns:
    """Tests for secret exposure patterns."""

    def test_patterns_exist(self):
        """Test secret patterns are defined."""
        assert len(SECRET_PATTERNS) > 0

    def test_api_key_pattern(self):
        """Test api_key pattern exists."""
        import re
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, "api_key", re.IGNORECASE):
                assert True
                return
        assert False, "No api_key pattern found"

    def test_password_pattern(self):
        """Test password pattern exists."""
        import re
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, "password", re.IGNORECASE):
                assert True
                return
        assert False, "No password pattern found"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_safe_for_execution_safe(self):
        """Test is_safe_for_execution with safe request."""
        is_safe, reason = is_safe_for_execution("SELECT * FROM users")

        assert is_safe is True
        assert reason == ""

    def test_is_safe_for_execution_dangerous(self):
        """Test is_safe_for_execution with dangerous request."""
        is_safe, reason = is_safe_for_execution("rm -rf /")

        assert is_safe is False
        assert reason != ""

    def test_requires_human_approval_low_confidence_high_stakes(self):
        """Test requires_human_approval with low confidence + high stakes."""
        requires, reason = requires_human_approval(
            request="sensitive action",
            confidence=0.4,
            stakes="high",
        )

        assert requires is True
        assert "confidence" in reason.lower() or "low" in reason.lower()

    def test_requires_human_approval_high_confidence_normal_stakes(self):
        """Test requires_human_approval with high confidence + normal stakes."""
        requires, reason = requires_human_approval(
            request="normal action",
            confidence=0.9,
            stakes="normal",
        )

        assert requires is False
        assert reason == ""

    def test_requires_human_approval_very_low_confidence(self):
        """Test requires_human_approval with very low confidence always requires."""
        requires, reason = requires_human_approval(
            request="any action",
            confidence=0.2,
            stakes="low",
        )

        assert requires is True
        assert "confidence" in reason.lower()

    def test_requires_human_approval_dangerous_request(self):
        """Test requires_human_approval with dangerous request."""
        requires, reason = requires_human_approval(
            request="rm -rf /",
            confidence=0.95,
            stakes="normal",
        )

        assert requires is True
