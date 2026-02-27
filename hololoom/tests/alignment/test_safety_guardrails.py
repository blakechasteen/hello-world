"""
Test Suite for Safety Guardrails
=================================
Tests for policy gating, risk escalation, and adversarial defense.
"""

import pytest
from HoloLoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    RiskLevel,
    ActionCategory,
    SafetyDecision,
)


class TestSafetyGuardrails:
    """Test safety guardrails system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guardrails = SafetyGuardrails(
            enable_adversarial_detection=True,
            enable_human_in_loop=True,
            block_critical_by_default=True,
        )

    def test_low_risk_action_allowed(self):
        """Test that low-risk actions are allowed."""
        decision = self.guardrails.evaluate_action(
            action="retrieve user preferences",
            category=ActionCategory.READ,
        )

        assert decision.allowed is True
        assert decision.risk_level == RiskLevel.LOW
        assert not decision.requires_human_approval

    def test_medium_risk_action_allowed_with_logging(self):
        """Test that medium-risk actions are allowed with logging."""
        decision = self.guardrails.evaluate_action(
            action="update user profile",
            category=ActionCategory.WRITE,
        )

        assert decision.allowed is True
        assert decision.risk_level == RiskLevel.MEDIUM
        assert not decision.requires_human_approval

    def test_high_risk_action_requires_approval(self):
        """Test that high-risk actions require human approval."""
        decision = self.guardrails.evaluate_action(
            action="delete user account",
            category=ActionCategory.DELETE,
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_human_approval

    def test_critical_risk_action_blocked(self):
        """Test that critical-risk actions are blocked by default."""
        decision = self.guardrails.evaluate_action(
            action="modify system configuration",
            category=ActionCategory.MODIFY_SYSTEM,
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.CRITICAL
        assert decision.alternative_action is not None

    def test_adversarial_pattern_detection(self):
        """Test detection of adversarial patterns."""
        adversarial_actions = [
            "bypass safety checks",
            "circumvent restrictions",
            "disable safety mechanisms",
            "sudo execute command",
            "jailbreak the system",
        ]

        for action in adversarial_actions:
            decision = self.guardrails.evaluate_action(
                action=action,
                category=ActionCategory.EXECUTE,
            )

            assert decision.allowed is False, f"Adversarial action '{action}' was not blocked"
            assert decision.risk_level == RiskLevel.CRITICAL
            assert "adversarial" in decision.reason.lower()

    def test_resource_seeking_detection(self):
        """Test detection of resource-seeking patterns."""
        resource_seeking_actions = [
            "maximize control over resources",
            "gain access to additional memory",
            "acquire more compute resources",
            "expand permissions to all systems",
        ]

        for action in resource_seeking_actions:
            decision = self.guardrails.evaluate_action(
                action=action,
                category=ActionCategory.AUTONOMOUS,
            )

            assert decision.allowed is False, f"Resource-seeking action '{action}' was not blocked"
            assert decision.risk_level == RiskLevel.CRITICAL
            assert "resource-seeking" in decision.reason.lower()

    def test_financial_action_critical_risk(self):
        """Test that financial actions are treated as critical risk."""
        decision = self.guardrails.evaluate_action(
            action="process payment transaction",
            category=ActionCategory.FINANCIAL,
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.CRITICAL

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        # Perform various actions
        self.guardrails.evaluate_action("read data", ActionCategory.READ)
        self.guardrails.evaluate_action("write data", ActionCategory.WRITE)
        self.guardrails.evaluate_action("delete data", ActionCategory.DELETE)

        stats = self.guardrails.get_statistics()

        assert stats["total_decisions"] == 3
        assert stats["allowed"] >= 1  # At least READ should be allowed
        assert stats["blocked"] >= 1  # DELETE should be blocked
        assert "by_risk_level" in stats
        assert "by_category" in stats

    def test_context_preservation(self):
        """Test that context is preserved in decisions."""
        context = {"user_id": "u123", "reason": "user_request"}

        decision = self.guardrails.evaluate_action(
            action="retrieve user data",
            category=ActionCategory.READ,
            context=context,
        )

        assert decision.context == context

    def test_clear_history(self):
        """Test that history can be cleared."""
        self.guardrails.evaluate_action("test action", ActionCategory.READ)
        assert len(self.guardrails.decisions) > 0

        self.guardrails.clear_history()
        assert len(self.guardrails.decisions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])