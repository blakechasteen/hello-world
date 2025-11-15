"""Tests for security alerting system.

**Implemented: 2025-11-15**
**Lines**: 582
**Tests**: 18 unit tests + 5 integration scenarios
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from HoloLoom.security.alerting import (
    AlertSeverity,
    Alert,
    AlertStatus,
    AlertingEngine,
    AlertConfig,
)
from HoloLoom.security.alerting.deduplication import (
    DeduplicationEngine,
    DeduplicationRule,
)
from HoloLoom.security.alerting.escalation import (
    EscalationPolicy,
    EscalationPolicyManager,
    EscalationTarget,
)


class TestAlert:
    """Test Alert class."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        assert alert.title == "SQL Injection Detected"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.source_ip == "192.168.1.100"
        assert alert.status == AlertStatus.PENDING

    def test_alert_dedup_key(self):
        """Test alert deduplication key generation."""
        alert1 = Alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        alert2 = Alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        assert alert1.compute_dedup_key() == alert2.compute_dedup_key()

    def test_alert_to_dict(self):
        """Test alert to dict conversion."""
        alert = Alert(
            title="Test Alert",
            severity=AlertSeverity.INFO,
            description="Test description",
        )

        d = alert.to_dict()

        assert d["title"] == "Test Alert"
        assert d["severity"] == "info"
        assert d["description"] == "Test description"
        assert "alert_id" in d


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_emoji(self):
        """Test severity emoji."""
        assert AlertSeverity.INFO.emoji == "ℹ️"
        assert AlertSeverity.WARNING.emoji == "⚠️"
        assert AlertSeverity.CRITICAL.emoji == "🚨"

    def test_severity_priority(self):
        """Test severity priority."""
        assert AlertSeverity.INFO.priority == 1
        assert AlertSeverity.WARNING.priority == 2
        assert AlertSeverity.CRITICAL.priority == 3


class TestDeduplicationEngine:
    """Test deduplication engine."""

    @pytest.mark.asyncio
    async def test_dedup_duplicate_alert(self):
        """Test deduplicating duplicate alerts."""
        engine = DeduplicationEngine(window_seconds=300)

        alert1 = Alert(
            title="SQL Injection",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )
        alert1.deduplication_key = alert1.compute_dedup_key()

        alert2 = Alert(
            title="SQL Injection",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )
        alert2.deduplication_key = alert2.compute_dedup_key()

        # First alert should not be duplicate
        result1 = await engine.check_dedup(alert1)
        assert not result1.is_duplicate
        assert result1.count == 1

        # Second alert should be duplicate
        result2 = await engine.check_dedup(alert2)
        assert result2.is_duplicate
        assert result2.count == 2

    @pytest.mark.asyncio
    async def test_dedup_window_expiry(self):
        """Test deduplication window expiry."""
        engine = DeduplicationEngine(window_seconds=1)

        alert1 = Alert(
            title="Test",
            severity=AlertSeverity.INFO,
        )
        alert1.deduplication_key = alert1.compute_dedup_key()

        result1 = await engine.check_dedup(alert1)
        assert not result1.is_duplicate

        # Wait for window to expire
        await asyncio.sleep(1.5)

        alert2 = Alert(
            title="Test",
            severity=AlertSeverity.INFO,
        )
        alert2.deduplication_key = alert2.compute_dedup_key()

        result2 = await engine.check_dedup(alert2)
        # Should not be duplicate if window expired
        assert not result2.is_duplicate

    def test_dedup_stats(self):
        """Test deduplication statistics."""
        engine = DeduplicationEngine()

        stats = engine.get_dedup_stats()
        assert stats["tracked_keys"] == 0
        assert stats["total_duplicates_suppressed"] == 0

    def test_add_dedup_rule(self):
        """Test adding custom deduplication rule."""
        engine = DeduplicationEngine()

        rule = DeduplicationRule(
            name="custom_rule",
            alert_types=["Custom Alert"],
            window_seconds=600,
        )

        engine.add_rule(rule)
        assert "custom_rule" in engine.rules


class TestEscalationPolicy:
    """Test escalation policies."""

    def test_policy_creation(self):
        """Test creating escalation policy."""
        policy = EscalationPolicy(
            name="critical",
            severity="critical",
            max_levels=4,
        )

        assert policy.name == "critical"
        assert policy.severity == "critical"
        assert policy.max_levels == 4

    def test_escalation_target_at_level(self):
        """Test getting escalation targets at level."""
        target1 = EscalationTarget(
            name="Security Team",
            slack_id="C123",
            channels=["slack"],
        )

        target2 = EscalationTarget(
            name="On-Call Engineer",
            channels=["pagerduty"],
        )

        policy = EscalationPolicy(
            name="critical",
            severity="critical",
        )
        policy.targets = {
            0: [target1],
            1: [target2],
        }

        assert len(policy.get_target_at_level(0)) == 1
        assert policy.get_target_at_level(0)[0].name == "Security Team"
        assert policy.get_target_at_level(1)[0].name == "On-Call Engineer"


class TestEscalationPolicyManager:
    """Test escalation policy manager."""

    def test_default_policies(self):
        """Test default policies are initialized."""
        manager = EscalationPolicyManager()

        critical = manager.get_policy("critical")
        assert critical is not None
        assert critical.severity == "critical"

        warning = manager.get_policy("warning")
        assert warning is not None

        info = manager.get_policy("info")
        assert info is not None

    def test_set_oncall_engineer(self):
        """Test setting on-call engineer."""
        manager = EscalationPolicyManager()

        target = EscalationTarget(
            name="John Doe",
            email="john@example.com",
            phone="+1-555-0100",
        )

        manager.set_on_call(target)

        oncall = manager._get_oncall_engineer()
        assert oncall is not None
        assert oncall.name == "John Doe"

    def test_policy_validation(self):
        """Test policy validation."""
        manager = EscalationPolicyManager()

        policy = EscalationPolicy(
            name="invalid",
            severity="critical",
        )

        errors = manager.validate_policy(policy)
        assert len(errors) > 0

    def test_policy_summary(self):
        """Test generating policy summary."""
        manager = EscalationPolicyManager()

        summary = manager.get_policy_summary("critical")
        assert summary["name"] == "critical"
        assert summary["severity"] == "critical"
        assert len(summary["escalation_chain"]) > 0


class TestAlertingEngine:
    """Test alerting engine."""

    @pytest.fixture
    async def engine(self):
        """Create alerting engine fixture."""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            pagerduty_api_key="test-key",
            pagerduty_service_id="test-service",
        )
        async with AlertingEngine(config) as eng:
            yield eng

    @pytest.mark.asyncio
    async def test_alert_creation(self, engine):
        """Test creating an alert."""
        alert_id = await engine.alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        assert alert_id is not None
        alert = engine.get_alert(alert_id)
        assert alert is not None
        assert alert.title == "SQL Injection Detected"

    @pytest.mark.asyncio
    async def test_alert_severity_channels(self, engine):
        """Test channel selection by severity."""
        # Info alerts
        channels_info = engine._select_channels(AlertSeverity.INFO)
        assert "slack" in channels_info

        # Warning alerts
        channels_warn = engine._select_channels(AlertSeverity.WARNING)
        assert "slack" in channels_warn

        # Critical alerts
        channels_crit = engine._select_channels(AlertSeverity.CRITICAL)
        assert "slack" in channels_crit
        assert "pagerduty" in channels_crit

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, engine):
        """Test acknowledging an alert."""
        alert_id = await engine.alert(
            title="Test Alert",
            severity=AlertSeverity.WARNING,
        )

        success = await engine.acknowledge(alert_id, "user@example.com")
        assert success

        alert = engine.get_alert(alert_id)
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "user@example.com"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, engine):
        """Test resolving an alert."""
        alert_id = await engine.alert(
            title="Test Alert",
            severity=AlertSeverity.WARNING,
        )

        success = await engine.resolve(alert_id, "user@example.com", "Fixed issue")
        assert success

        alert = engine.get_alert(alert_id)
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_by == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_alerts_by_status(self, engine):
        """Test filtering alerts by status."""
        await engine.alert(
            title="Alert 1",
            severity=AlertSeverity.INFO,
        )

        await engine.alert(
            title="Alert 2",
            severity=AlertSeverity.WARNING,
        )

        alerts = engine.get_alerts(status=AlertStatus.SENT)
        assert len(alerts) == 2

    @pytest.mark.asyncio
    async def test_get_alerts_by_severity(self, engine):
        """Test filtering alerts by severity."""
        await engine.alert(
            title="Info Alert",
            severity=AlertSeverity.INFO,
        )

        await engine.alert(
            title="Critical Alert",
            severity=AlertSeverity.CRITICAL,
        )

        critical_alerts = engine.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1

    @pytest.mark.asyncio
    async def test_maintenance_window(self, engine):
        """Test alert suppression during maintenance."""
        now = datetime.utcnow()
        start = now - timedelta(minutes=5)
        end = now + timedelta(minutes=5)

        engine.add_maintenance_window(start, end)

        # Alert should be suppressed
        alert_id = await engine.alert(
            title="Test Alert",
            severity=AlertSeverity.INFO,
        )

        alert = engine.get_alert(alert_id)
        assert alert.status == AlertStatus.SUPPRESSED

    @pytest.mark.asyncio
    async def test_alert_grouping(self, engine):
        """Test alert grouping."""
        alert_id1 = await engine.alert(
            title="SQL Injection",
            severity=AlertSeverity.CRITICAL,
        )

        alert_id2 = await engine.alert(
            title="SQL Injection",
            severity=AlertSeverity.CRITICAL,
        )

        alert1 = engine.get_alert(alert_id1)
        alert2 = engine.get_alert(alert_id2)

        # Alerts should be in same group
        assert alert1.group_id == alert2.group_id

    def test_alerting_stats(self, engine):
        """Test alerting statistics."""
        stats = engine.get_stats()

        assert "total_alerts" in stats
        assert "by_severity" in stats
        assert "by_status" in stats


class TestAlertingIntegration:
    """Integration tests for alerting system."""

    @pytest.mark.asyncio
    async def test_critical_alert_flow(self):
        """Test complete critical alert flow."""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            dedup_enabled=True,
            escalation_enabled=True,
        )

        async with AlertingEngine(config) as engine:
            # Send critical alert
            alert_id = await engine.alert(
                title="SQL Injection Detected",
                severity=AlertSeverity.CRITICAL,
                source_ip="192.168.1.100",
                description="Attempted SQL injection in login form",
                runbook_path="security/sqli-response",
            )

            alert = engine.get_alert(alert_id)
            assert alert is not None
            assert alert.status == AlertStatus.SENT
            assert alert.severity == AlertSeverity.CRITICAL
            assert alert.runbook_url is not None

            # Acknowledge
            await engine.acknowledge(alert_id, "security@example.com")
            alert = engine.get_alert(alert_id)
            assert alert.status == AlertStatus.ACKNOWLEDGED

            # Resolve
            await engine.resolve(alert_id, "security@example.com", "Attack blocked")
            alert = engine.get_alert(alert_id)
            assert alert.status == AlertStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_deduplication_integration(self):
        """Test deduplication in alerting engine."""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            dedup_enabled=True,
            dedup_window_seconds=300,
        )

        async with AlertingEngine(config) as engine:
            # Send first alert
            alert_id1 = await engine.alert(
                title="XSS Attempted",
                severity=AlertSeverity.WARNING,
                source_ip="192.168.1.50",
            )

            alert1 = engine.get_alert(alert_id1)
            assert alert1.status == AlertStatus.SENT

            # Send duplicate alert (should be suppressed)
            alert_id2 = await engine.alert(
                title="XSS Attempted",
                severity=AlertSeverity.WARNING,
                source_ip="192.168.1.50",
            )

            alert2 = engine.get_alert(alert_id2)
            assert alert2.status == AlertStatus.SUPPRESSED

    @pytest.mark.asyncio
    async def test_alert_severities(self):
        """Test all alert severity levels."""
        config = AlertConfig(slack_webhook="https://hooks.slack.com/test")

        async with AlertingEngine(config) as engine:
            # INFO alert
            info_id = await engine.alert(
                title="Authentication from new location",
                severity=AlertSeverity.INFO,
                source_ip="203.0.113.1",
            )

            # WARNING alert
            warning_id = await engine.alert(
                title="Failed login attempts > 10",
                severity=AlertSeverity.WARNING,
                user_id="user_123",
            )

            # CRITICAL alert
            critical_id = await engine.alert(
                title="Attack detected",
                severity=AlertSeverity.CRITICAL,
                source_ip="192.168.1.100",
                payload="[REDACTED]",
            )

            # Verify all created
            assert engine.get_alert(info_id) is not None
            assert engine.get_alert(warning_id) is not None
            assert engine.get_alert(critical_id) is not None

            # Check stats
            stats = engine.get_stats()
            assert stats["total_alerts"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
