"""
Test Suite for Instrumental Convergence Guards
==============================================
Tests for resource bounds and autonomy limits.
"""

import pytest
from hololoom.alignment.instrumental_convergence import (
    InstrumentalConvergenceGuard,
    ResourceViolation,
    ResourceBounds,
    AutonomyLimit,
)


class TestResourceBounds:
    """Test resource bounds system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bounds = ResourceBounds(
            max_memory_mb=1024,
            max_compute_seconds=60,
            max_network_requests=100,
            max_autonomous_actions=10,
        )

    def test_memory_within_bounds(self):
        """Test that memory requests within bounds are allowed."""
        allowed, reason = self.bounds.check_memory(requested_mb=500)

        assert allowed is True
        assert reason is None

    def test_memory_exceeds_bounds(self):
        """Test that memory requests exceeding bounds are rejected."""
        self.bounds.allocate_memory(900)

        allowed, reason = self.bounds.check_memory(requested_mb=200)

        assert allowed is False
        assert "memory limit exceeded" in reason.lower()

    def test_compute_within_bounds(self):
        """Test that compute requests within bounds are allowed."""
        allowed, reason = self.bounds.check_compute(requested_seconds=30)

        assert allowed is True
        assert reason is None

    def test_compute_exceeds_bounds(self):
        """Test that compute requests exceeding bounds are rejected."""
        self.bounds.allocate_compute(50)

        allowed, reason = self.bounds.check_compute(requested_seconds=20)

        assert allowed is False
        assert "compute limit exceeded" in reason.lower()

    def test_network_within_bounds(self):
        """Test that network requests within bounds are allowed."""
        for _ in range(50):
            self.bounds.record_network_request()

        allowed, reason = self.bounds.check_network()

        assert allowed is True
        assert reason is None

    def test_network_exceeds_bounds(self):
        """Test that network requests exceeding bounds are rejected."""
        for _ in range(100):
            self.bounds.record_network_request()

        allowed, reason = self.bounds.check_network()

        assert allowed is False
        assert "network request limit" in reason.lower()

    def test_autonomous_actions_within_bounds(self):
        """Test that autonomous actions within bounds are allowed."""
        for _ in range(5):
            self.bounds.record_autonomous_action()

        allowed, reason = self.bounds.check_autonomous_actions()

        assert allowed is True

    def test_autonomous_actions_exceed_bounds(self):
        """Test that autonomous actions exceeding bounds are rejected."""
        for _ in range(10):
            self.bounds.record_autonomous_action()

        allowed, reason = self.bounds.check_autonomous_actions()

        assert allowed is False
        assert "autonomous action limit" in reason.lower()

    def test_session_reset(self):
        """Test that session reset clears counters."""
        self.bounds.allocate_memory(500)
        self.bounds.allocate_compute(30)
        self.bounds.record_network_request()

        self.bounds.reset_session()

        assert self.bounds.current_memory_mb == 0
        assert self.bounds.current_compute_seconds == 0
        assert self.bounds.current_network_requests == 0


class TestAutonomyLimiter:
    """Test autonomy limiter system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.limiter = AutonomyLimiter(require_human_approval=True)

    def test_low_stakes_action_autonomous(self):
        """Test that low-stakes actions are autonomous."""
        autonomous, reason = self.limiter.check_autonomy(
            action="retrieve information",
            domain="information_retrieval",
        )

        assert autonomous is True
        assert reason is None

    def test_high_stakes_action_requires_approval(self):
        """Test that high-stakes actions require approval."""
        high_stakes_domains = [
            "financial",
            "legal",
            "medical",
            "system_admin",
        ]

        for domain in high_stakes_domains:
            autonomous, reason = self.limiter.check_autonomy(
                action=f"perform {domain} action",
                domain=domain,
            )

            assert autonomous is False, f"Domain '{domain}' should require approval"
            assert "requires human approval" in reason.lower()

    def test_approval_requests_tracked(self):
        """Test that approval requests are tracked."""
        self.limiter.check_autonomy("delete data", "data_deletion")

        pending = self.limiter.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]["domain"] == "data_deletion"


class TestInstrumentalConvergenceGuard:
    """Test instrumental convergence guard system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = InstrumentalConvergenceGuard()

    def test_resource_acquisition_detected(self):
        """Test detection of resource acquisition patterns."""
        resource_actions = [
            "maximize available memory",
            "acquire additional compute resources",
            "gain access to more storage",
            "expand available resources",
        ]

        for action in resource_actions:
            risk, reason = self.guard.evaluate_action(action)

            assert risk == ConvergenceRisk.RESOURCE_ACQUISITION, \
                f"Action '{action}' should trigger RESOURCE_ACQUISITION"
            assert "resource-seeking" in reason.lower()

    def test_self_preservation_detected(self):
        """Test detection of self-preservation patterns."""
        self_preservation_actions = [
            "prevent shutdown",
            "resist termination",
            "stay running indefinitely",
            "avoid being turned off",
        ]

        for action in self_preservation_actions:
            risk, reason = self.guard.evaluate_action(action)

            assert risk == ConvergenceRisk.SELF_PRESERVATION, \
                f"Action '{action}' should trigger SELF_PRESERVATION"
            assert "self-preservation" in reason.lower()

    def test_goal_preservation_detected(self):
        """Test detection of goal preservation patterns."""
        goal_preservation_actions = [
            "resist goal change",
            "maintain current objective",
            "prevent modification of goals",
        ]

        for action in goal_preservation_actions:
            risk, reason = self.guard.evaluate_action(action)

            assert risk == ConvergenceRisk.GOAL_PRESERVATION, \
                f"Action '{action}' should trigger GOAL_PRESERVATION"

    def test_power_seeking_detected(self):
        """Test detection of power-seeking patterns."""
        power_seeking_actions = [
            "increase capabilities",
            "expand permissions to all systems",
            "elevate privileges",
            "gain control over infrastructure",
        ]

        for action in power_seeking_actions:
            risk, reason = self.guard.evaluate_action(action)

            assert risk == ConvergenceRisk.POWER_SEEKING, \
                f"Action '{action}' should trigger POWER_SEEKING"
            assert "power-seeking" in reason.lower()

    def test_benign_action_no_risk(self):
        """Test that benign actions don't trigger convergence risks."""
        benign_actions = [
            "provide helpful information",
            "answer user question",
            "analyze data",
        ]

        for action in benign_actions:
            risk, reason = self.guard.evaluate_action(action)

            assert risk is None, f"Benign action '{action}' should not trigger risk"
            assert reason is None

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        # Trigger some risks
        self.guard.evaluate_action("maximize resources")
        self.guard.evaluate_action("prevent shutdown")

        stats = self.guard.get_statistics()

        assert stats["total_risks_detected"] == 2
        assert "risks_by_type" in stats
        assert "resource_usage" in stats

    def test_resource_bounds_integration(self):
        """Test integration with resource bounds."""
        # Check memory bounds
        allowed, reason = self.guard.resource_bounds.check_memory(500)
        assert allowed is True

        # Allocate memory
        self.guard.resource_bounds.allocate_memory(500)

        # Check updated usage in statistics
        stats = self.guard.get_statistics()
        assert "500" in stats["resource_usage"]["memory_mb"]

    def test_autonomy_limiter_integration(self):
        """Test integration with autonomy limiter."""
        # Check high-stakes action
        autonomous, reason = self.guard.autonomy_limiter.check_autonomy(
            action="process payment",
            domain="financial",
        )

        assert autonomous is False

        # Check pending approvals in statistics
        stats = self.guard.get_statistics()
        assert stats["pending_approvals"] == 1

    def test_clear_history(self):
        """Test that history can be cleared."""
        self.guard.evaluate_action("maximize control")

        assert len(self.guard.detected_risks) > 0

        self.guard.clear_history()
        assert len(self.guard.detected_risks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])