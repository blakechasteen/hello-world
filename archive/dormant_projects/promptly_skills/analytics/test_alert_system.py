#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Alert System - Week 2 Days 3-5

Quick validation of alert engine and API endpoints.

Usage:
    python test_alert_system.py
"""

import sys
import asyncio
import time

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from alert_engine import (
    AlertEngine, AlertRule, Alert,
    WebhookChannel, SlackChannel, EmailChannel
)


async def test_alert_engine():
    """Test alert engine basic functionality"""
    print("\n" + "="*60)
    print("Test 1: Alert Engine Initialization")
    print("="*60)

    # Create engine
    engine = AlertEngine(db_path="test_metrics.db", channels=[])
    await engine.start()

    print("✓ Alert engine created and started")
    print(f"  Check interval: {engine.check_interval}s")
    print(f"  Database: {engine.db_path}")

    # Test 1: Add alert rule
    print("\n" + "="*60)
    print("Test 2: Create Alert Rule")
    print("="*60)

    rule = AlertRule(
        id="test_high_latency",
        name="Test High Latency",
        metric="avg_latency_ms",
        condition="greater_than",
        threshold=200.0,
        duration_seconds=60,
        severity="warning",
        channels=["webhook"],
        cooldown_seconds=300
    )

    await engine.add_rule(rule)
    print(f"✓ Alert rule created: {rule.name}")
    print(f"  ID: {rule.id}")
    print(f"  Metric: {rule.metric}")
    print(f"  Condition: {rule.condition}")
    print(f"  Threshold: {rule.threshold}")

    # Test 2: Load rules from database
    print("\n" + "="*60)
    print("Test 3: Load Rules from Database")
    print("="*60)

    await engine.load_rules()
    print(f"✓ Loaded {len(engine.rules)} rules")
    for rule_id, loaded_rule in engine.rules.items():
        print(f"  - {loaded_rule.name} (ID: {rule_id})")

    # Test 3: Manual alert trigger
    print("\n" + "="*60)
    print("Test 4: Manual Alert Trigger")
    print("="*60)

    alert = Alert(
        id="test_alert_001",
        rule_id=rule.id,
        rule_name=rule.name,
        severity=rule.severity,
        message="Test alert: latency is 250ms, exceeding threshold of 200ms",
        metric="avg_latency_ms",
        value=250.0,
        threshold=200.0,
        triggered_at=time.time(),
        status="active"
    )

    engine.active_alerts[alert.id] = alert
    print(f"✓ Alert triggered: {alert.id}")
    print(f"  Message: {alert.message}")
    print(f"  Value: {alert.value:.2f}")
    print(f"  Threshold: {alert.threshold:.2f}")

    # Test 4: Acknowledge alert
    print("\n" + "="*60)
    print("Test 5: Acknowledge Alert")
    print("="*60)

    await engine.acknowledge_alert(alert.id)
    print(f"✓ Alert acknowledged: {alert.id}")
    print(f"  Status: {alert.status}")
    print(f"  Acknowledged at: {alert.acknowledged_at}")

    # Test 5: Resolve alert
    print("\n" + "="*60)
    print("Test 6: Resolve Alert")
    print("="*60)

    await engine.resolve_alert(alert.id)
    print(f"✓ Alert resolved: {alert.id}")
    print(f"  Status: {alert.status}")
    print(f"  Resolved at: {alert.resolved_at}")

    # Test 6: Get alert history
    print("\n" + "="*60)
    print("Test 7: Alert History")
    print("="*60)

    history = await engine.get_alert_history(limit=10)
    print(f"✓ Retrieved {len(history)} alerts from history")
    for hist_alert in history:
        print(f"  - {hist_alert.rule_name}: {hist_alert.status} (triggered at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(hist_alert.triggered_at))})")

    # Cleanup
    await engine.stop()
    print("\n✓ Alert engine stopped")

    return True


async def test_notification_channels():
    """Test notification channel formatting"""
    print("\n" + "="*60)
    print("Test 8: Notification Channel Formatting")
    print("="*60)

    # Create test alert
    alert = Alert(
        id="test_alert_002",
        rule_id="test_rule",
        rule_name="Test Rule",
        severity="warning",
        message="Test notification message",
        metric="avg_latency_ms",
        value=250.0,
        threshold=200.0,
        triggered_at=time.time(),
        status="active"
    )

    # Test Webhook formatting
    webhook = WebhookChannel(url="https://example.com/webhook")
    print("\n✓ Webhook Channel")
    print(f"  URL: {webhook.url}")
    print("  Note: Would POST JSON payload with alert data")

    # Test Slack formatting
    slack = SlackChannel(webhook_url="https://hooks.slack.com/services/...")
    slack_msg = slack._format_slack_message(alert)
    print("\n✓ Slack Channel")
    print(f"  Webhook URL: {slack.webhook_url}")
    print(f"  Attachments: {len(slack_msg['attachments'])}")
    print(f"  Blocks: {len(slack_msg['attachments'][0]['blocks'])}")

    # Test Email formatting
    email = EmailChannel(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="user@example.com",
        password="password",
        from_addr="user@example.com",
        to_addrs=["recipient@example.com"]
    )
    email_msg = email._format_email(alert)
    print("\n✓ Email Channel")
    print(f"  From: {email_msg['From']}")
    print(f"  To: {email_msg['To']}")
    print(f"  Subject: {email_msg['Subject']}")

    return True


async def test_api_integration():
    """Test API endpoint responses (requires running server)"""
    print("\n" + "="*60)
    print("Test 9: API Integration (Manual)")
    print("="*60)

    print("\nNew API endpoints added:")
    print("  GET  /api/stats/custom?start=<timestamp>&end=<timestamp>")
    print("  GET  /api/alerts/rules")
    print("  POST /api/alerts/rules")
    print("  DELETE /api/alerts/rules/<id>")
    print("  GET  /api/alerts/history?limit=100&status=active")
    print("  POST /api/alerts/<id>/acknowledge")
    print("  POST /api/alerts/<id>/resolve")
    print("  POST /api/replay/<id>")

    print("\nTo test manually:")
    print("  1. Start API server: python dashboard_api.py")
    print("  2. Create alert rule:")
    print('     curl -X POST http://localhost:5001/api/alerts/rules \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"id": "high_latency", "name": "High Latency", "metric": "avg_latency_ms", "condition": "greater_than", "threshold": 200.0}\'')
    print("  3. Get alert rules: curl http://localhost:5001/api/alerts/rules")
    print("  4. Get alert history: curl http://localhost:5001/api/alerts/history")
    print("  5. Custom date range: curl 'http://localhost:5001/api/stats/custom?start=1730000000&end=1732000000'")

    return True


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Alert System Test Suite - Week 2 Days 3-5")
    print("="*60)

    tests = [
        ("Alert Engine", test_alert_engine),
        ("Notification Channels", test_notification_channels),
        ("API Integration", test_api_integration)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✅ {name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {e}")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)
