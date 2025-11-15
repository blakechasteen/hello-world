"""
Demo of HoloLoom security alerting system with all severity levels.

**Implemented: 2025-11-15**
**Demonstrates**: Multi-channel alerting, deduplication, escalation, grouping

Run with:
    PYTHONPATH=. python demos/demo_alerting.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.security.alerting import (
    AlertSeverity,
    Alert,
    AlertStatus,
    AlertingEngine,
    AlertConfig,
)
from HoloLoom.security.alerting.escalation import EscalationPolicyManager, EscalationTarget


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_alert_summary(alert: Alert):
    """Print alert summary."""
    print(f"  Alert ID:       {alert.alert_id}")
    print(f"  Title:          {alert.title}")
    print(f"  Severity:       {alert.severity.emoji} {alert.severity.value.upper()}")
    print(f"  Status:         {alert.status.value}")
    print(f"  Time:           {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if alert.source_ip:
        print(f"  Source IP:      {alert.source_ip}")

    if alert.user_id:
        print(f"  User:           {alert.user_id}")

    if alert.description:
        print(f"  Description:    {alert.description[:60]}...")

    if alert.channels_sent:
        print(f"  Channels:       {', '.join(alert.channels_sent)}")

    print()


async def demo_info_alerts():
    """Demo INFO level alerts."""
    print_section("Demo 1: INFO Level Alerts")

    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/test",
        dedup_enabled=True,
    )

    async with AlertingEngine(config) as engine:
        print("Sending INFO alerts (routine security events)...\n")

        # New location login
        alert_id = await engine.alert(
            title="Authentication from new location",
            severity=AlertSeverity.INFO,
            source_ip="203.0.113.1",
            description="New IP address detected for user account",
            metadata={"event_type": "login", "country": "United States"},
        )

        print("✓ Alert 1 - New Location Login:")
        print_alert_summary(engine.get_alert(alert_id))

        # New API key
        alert_id = await engine.alert(
            title="New API key created",
            severity=AlertSeverity.INFO,
            user_id="user_abc123",
            description="New API key generated for service account",
            metadata={"key_id": "key_xyz789"},
        )

        print("✓ Alert 2 - New API Key:")
        print_alert_summary(engine.get_alert(alert_id))

        stats = engine.get_stats()
        print(f"Total alerts: {stats['total_alerts']}")
        print(f"By status: {stats['by_status']}")


async def demo_warning_alerts():
    """Demo WARNING level alerts."""
    print_section("Demo 2: WARNING Level Alerts")

    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/test",
        dedup_enabled=True,
        dedup_window_seconds=300,
    )

    async with AlertingEngine(config) as engine:
        print("Sending WARNING alerts (potential issues)...\n")

        # Failed login attempts
        alert_id = await engine.alert(
            title="Failed login attempts > 10",
            severity=AlertSeverity.WARNING,
            user_id="user_def456",
            description="User account has 12 failed login attempts in last 15 minutes",
            metadata={
                "failed_attempts": 12,
                "time_window_minutes": 15,
                "last_attempt": datetime.utcnow().isoformat(),
            },
            tags={"event_type": "authentication", "risk_score": "0.65"},
        )

        print("✓ Alert 1 - Failed Login Attempts:")
        print_alert_summary(engine.get_alert(alert_id))

        # Rate limit exceeded
        alert_id = await engine.alert(
            title="Rate limit exceeded",
            severity=AlertSeverity.WARNING,
            source_ip="192.168.1.200",
            description="API endpoint exceeded rate limit threshold",
            metadata={
                "endpoint": "/api/users",
                "requests_per_minute": 450,
                "limit": 400,
            },
        )

        print("✓ Alert 2 - Rate Limit Exceeded:")
        print_alert_summary(engine.get_alert(alert_id))

        # Anomaly detection
        alert_id = await engine.alert(
            title="Anomaly score > 0.7",
            severity=AlertSeverity.WARNING,
            user_id="user_ghi789",
            description="Unusual behavior detected for user account",
            metadata={
                "anomaly_score": 0.72,
                "anomaly_type": "unusual_access_pattern",
            },
        )

        print("✓ Alert 3 - Anomaly Detected:")
        print_alert_summary(engine.get_alert(alert_id))

        stats = engine.get_stats()
        print(f"\nTotal alerts: {stats['total_alerts']}")
        print(f"By severity: {stats['by_severity']}")


async def demo_critical_alerts():
    """Demo CRITICAL level alerts."""
    print_section("Demo 3: CRITICAL Level Alerts")

    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/test",
        pagerduty_api_key="test-key",
        pagerduty_service_id="test-service",
        dedup_enabled=True,
        escalation_enabled=True,
        grouping_enabled=True,
    )

    async with AlertingEngine(config) as engine:
        print("Sending CRITICAL alerts (immediate action required)...\n")

        # SQL Injection
        alert_id = await engine.alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
            payload="[REDACTED]",
            description="Attempted SQL injection in login form",
            metadata={
                "waf_action": "blocked",
                "pattern": "OR 1=1",
            },
            tags={
                "attack_type": "sql_injection",
                "severity": "critical",
            },
            runbook_path="security/sqli-response",
        )

        print("✓ Alert 1 - SQL Injection Detected:")
        alert1 = engine.get_alert(alert_id)
        print_alert_summary(alert1)

        # XSS Attack
        alert_id = await engine.alert(
            title="XSS Attempted",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.101",
            description="Cross-site scripting attempt in user input",
            metadata={
                "payload": "<script>alert('xss')</script>",
                "blocked": True,
            },
            runbook_path="security/xss-response",
        )

        print("✓ Alert 2 - XSS Attempted:")
        alert2 = engine.get_alert(alert_id)
        print_alert_summary(alert2)

        # Breach Suspected
        alert_id = await engine.alert(
            title="Potential system breach detected",
            severity=AlertSeverity.CRITICAL,
            description="Multiple security indicators suggest possible unauthorized access",
            metadata={
                "anomaly_score": 0.95,
                "indicators": [
                    "Unusual database queries",
                    "Privilege escalation attempt",
                    "Mass file access",
                ],
            },
            runbook_path="security/breach-response",
        )

        print("✓ Alert 3 - Breach Suspected:")
        alert3 = engine.get_alert(alert_id)
        print_alert_summary(alert3)

        stats = engine.get_stats()
        print(f"\nTotal alerts: {stats['total_alerts']}")
        print(f"By status: {stats['by_status']}")


async def demo_deduplication():
    """Demo alert deduplication."""
    print_section("Demo 4: Alert Deduplication")

    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/test",
        dedup_enabled=True,
        dedup_window_seconds=300,
    )

    async with AlertingEngine(config) as engine:
        print("Sending duplicate alerts to test deduplication...\n")

        # First alert
        alert_id1 = await engine.alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        alert1 = engine.get_alert(alert_id1)
        print(f"✓ Alert 1 sent - Status: {alert1.status.value}")
        print(f"  ID: {alert_id1}\n")

        # Duplicate alert (should be suppressed)
        alert_id2 = await engine.alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )

        alert2 = engine.get_alert(alert_id2)
        print(f"✓ Alert 2 (duplicate) - Status: {alert2.status.value}")
        print(f"  ID: {alert_id2}")
        print(f"  Reason: Same title + source IP within {config.dedup_window_seconds}s\n")

        # Different source IP (not a duplicate)
        alert_id3 = await engine.alert(
            title="SQL Injection Detected",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.101",
        )

        alert3 = engine.get_alert(alert_id3)
        print(f"✓ Alert 3 (different source) - Status: {alert3.status.value}")
        print(f"  ID: {alert_id3}")
        print(f"  Reason: Different source IP, not a duplicate\n")

        if engine.dedup_engine:
            dedup_stats = engine.dedup_engine.get_dedup_stats()
            print(f"Deduplication Statistics:")
            print(f"  Tracked keys: {dedup_stats['tracked_keys']}")
            print(f"  Duplicates suppressed: {dedup_stats['total_duplicates_suppressed']}")


async def demo_grouping():
    """Demo alert grouping."""
    print_section("Demo 5: Alert Grouping")

    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/test",
        grouping_enabled=True,
        grouping_window_seconds=300,
    )

    async with AlertingEngine(config) as engine:
        print("Sending related alerts to test grouping...\n")

        # Multiple SQL injection attempts
        alert_ids = []
        for i in range(3):
            alert_id = await engine.alert(
                title="SQL Injection Detected",
                severity=AlertSeverity.CRITICAL,
                source_ip=f"192.168.1.{100 + i}",
                description=f"SQL injection attempt #{i+1}",
            )
            alert_ids.append(alert_id)

        # Check grouping
        alerts = [engine.get_alert(aid) for aid in alert_ids]
        group_id = alerts[0].group_id

        print(f"✓ Created {len(alerts)} related alerts")
        print(f"  Group ID: {group_id}\n")

        for i, alert in enumerate(alerts, 1):
            print(f"  Alert {i}: {alert.alert_id}")
            print(f"    Group: {alert.group_id}")
            print(f"    Grouped: {alert.group_id == group_id}")

        print(f"\n  → All related alerts grouped together for easier management")


async def demo_escalation():
    """Demo escalation policies."""
    print_section("Demo 6: Escalation Policies")

    manager = EscalationPolicyManager()

    # Set on-call engineer
    oncall = EscalationTarget(
        name="Alice Johnson",
        email="alice@example.com",
        phone="+1-555-0100",
    )
    manager.set_on_call(oncall)

    print("Escalation Policy Configuration:\n")

    # Get critical policy
    critical_policy = manager.get_policy("critical")
    summary = manager.get_policy_summary("critical")

    print(f"Policy: {summary['name'].upper()}")
    print(f"Support: {'24/7' if summary['support_24h'] else 'Business hours'}\n")

    for step in summary["escalation_chain"]:
        level = step["level"]
        wait = step["wait_seconds"]
        targets = step["targets"]

        if level == 0:
            print(f"Level {level} (Immediate):")
        else:
            print(f"Level {level} (After {wait//60} minutes):")

        for target in targets:
            print(f"  → {target['name']}")
            print(f"    Channels: {', '.join(target['channels'])}")

        print()

    print("Escalation Timeline:")
    print("  0 min   : Alert sent to Security Team (Slack)")
    print("  15 min  : Escalate to On-Call Engineer (PagerDuty + Slack)")
    print("  30 min  : Escalate to Security Lead (PagerDuty + Email + SMS)")
    print("  60 min  : Escalate to CTO (PagerDuty + Email + SMS)")


async def demo_acknowledgment():
    """Demo alert acknowledgment and resolution."""
    print_section("Demo 7: Alert Acknowledgment & Resolution")

    config = AlertConfig(slack_webhook="https://hooks.slack.com/test")

    async with AlertingEngine(config) as engine:
        print("Creating and managing alert lifecycle...\n")

        # Create alert
        alert_id = await engine.alert(
            title="Rate limit exceeded",
            severity=AlertSeverity.WARNING,
            source_ip="192.168.1.200",
            description="API endpoint exceeded rate limit",
        )

        alert = engine.get_alert(alert_id)
        print(f"✓ Alert created")
        print(f"  Status: {alert.status.value}\n")

        # Acknowledge
        await engine.acknowledge(alert_id, "bob@example.com")
        alert = engine.get_alert(alert_id)
        print(f"✓ Alert acknowledged by {alert.acknowledged_by}")
        print(f"  Status: {alert.status.value}")
        print(f"  Time: {alert.acknowledged_at.strftime('%H:%M:%S UTC')}\n")

        # Resolve
        await engine.resolve(
            alert_id,
            "bob@example.com",
            "Fixed rate limiting configuration"
        )
        alert = engine.get_alert(alert_id)
        print(f"✓ Alert resolved by {alert.resolved_by}")
        print(f"  Status: {alert.status.value}")
        print(f"  Time: {alert.resolved_at.strftime('%H:%M:%S UTC')}")
        print(f"  Reason: {alert.metadata.get('resolution_reason')}")


async def demo_maintenance_windows():
    """Demo maintenance window suppression."""
    print_section("Demo 8: Maintenance Windows")

    config = AlertConfig(slack_webhook="https://hooks.slack.com/test")

    async with AlertingEngine(config) as engine:
        print("Testing alert suppression during maintenance...\n")

        # Add maintenance window
        now = datetime.utcnow()
        start = now - timedelta(minutes=5)
        end = now + timedelta(minutes=5)

        engine.add_maintenance_window(start, end)
        print(f"✓ Maintenance window added")
        print(f"  Start: {start.strftime('%H:%M:%S UTC')}")
        print(f"  End:   {end.strftime('%H:%M:%S UTC')}\n")

        # Try to send alert (should be suppressed)
        alert_id = await engine.alert(
            title="Test alert during maintenance",
            severity=AlertSeverity.INFO,
            description="This alert is sent during maintenance window",
        )

        alert = engine.get_alert(alert_id)
        print(f"✓ Alert attempted during maintenance")
        print(f"  Status: {alert.status.value}")
        print(f"  Reason: Inside maintenance window\n")

        # Remove maintenance window
        engine.remove_maintenance_window(start, end)
        print(f"✓ Maintenance window removed\n")

        # Alert now works
        alert_id = await engine.alert(
            title="Test alert after maintenance",
            severity=AlertSeverity.INFO,
            description="This alert is sent after maintenance",
        )

        alert = engine.get_alert(alert_id)
        print(f"✓ Alert after maintenance")
        print(f"  Status: {alert.status.value}")


async def demo_summary():
    """Print summary of alerting system capabilities."""
    print_section("Alerting System Summary")

    print("""
Multi-Channel Alerting Capabilities:
  ✓ Slack      - Real-time notifications in chat
  ✓ PagerDuty  - Incident management platform
  ✓ Email      - HTML-formatted email alerts
  ✓ SMS        - Twilio-based SMS alerts

Alert Severity Levels:
  ℹ️  INFO      - Routine security events
  ⚠️  WARNING   - Potential issues requiring attention
  🚨 CRITICAL  - Immediate action required

Key Features:
  ✓ Alert deduplication (prevents alert spam)
  ✓ Alert grouping (groups related alerts)
  ✓ Escalation policies (time-based escalation)
  ✓ Alert acknowledgment (track response)
  ✓ Alert resolution (close tickets)
  ✓ Maintenance windows (suppress during downtime)
  ✓ On-call rotation (route to right person)
  ✓ Complete audit trail (compliance tracking)

Example Alert Response Times:
  • Level 0 (Immediate)  : Slack notification
  • Level 1 (15 min)     : PagerDuty incident creation
  • Level 2 (30 min)     : Email + SMS to security lead
  • Level 3 (60 min)     : All channels to CTO
    """)


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  HoloLoom Security Alerting System - Comprehensive Demo")
    print("=" * 70)

    try:
        await demo_info_alerts()
        await demo_warning_alerts()
        await demo_critical_alerts()
        await demo_deduplication()
        await demo_grouping()
        await demo_escalation()
        await demo_acknowledgment()
        await demo_maintenance_windows()
        await demo_summary()

        print("\n" + "=" * 70)
        print("  ✓ Demo Complete")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
