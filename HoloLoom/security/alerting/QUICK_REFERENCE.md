# Alerting System Quick Reference

## Import

```python
from HoloLoom.security.alerting import (
    AlertingEngine,
    AlertConfig,
    AlertSeverity,
    Alert,
    AlertStatus,
)
```

## Basic Usage

### Create Alerting Engine

```python
config = AlertConfig(
    slack_webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    dedup_enabled=True,
)

async with AlertingEngine(config) as engine:
    # Use engine
```

### Send Alert

```python
alert_id = await engine.alert(
    title="SQL Injection Detected",
    severity=AlertSeverity.CRITICAL,
    source_ip="192.168.1.100",
    user_id="user_123",
    payload="[REDACTED]",
    description="Attempted SQL injection in login form",
    tags={"attack_type": "sql_injection"},
    runbook_path="security/sqli-response",
)
```

### Acknowledge Alert

```python
success = await engine.acknowledge(alert_id, "user@example.com")
```

### Resolve Alert

```python
success = await engine.resolve(
    alert_id,
    "user@example.com",
    reason="Fixed WAF rule"
)
```

### Get Alert

```python
alert = engine.get_alert(alert_id)
print(alert.title)           # Alert title
print(alert.severity.value)  # "critical", "warning", "info"
print(alert.status.value)    # "sent", "acknowledged", "resolved"
print(alert.timestamp)       # datetime when created
```

### Filter Alerts

```python
# By status
sent_alerts = engine.get_alerts(status=AlertStatus.SENT)

# By severity
critical = engine.get_alerts(severity=AlertSeverity.CRITICAL)

# By both
unresolved_critical = engine.get_alerts(
    status=AlertStatus.SENT,
    severity=AlertSeverity.CRITICAL
)
```

### Maintenance Windows

```python
from datetime import datetime, timedelta

now = datetime.utcnow()
engine.add_maintenance_window(
    start=now,
    end=now + timedelta(hours=2)
)

# Alerts automatically suppressed during window
```

### Get Statistics

```python
stats = engine.get_stats()
print(stats["total_alerts"])      # Total alerts
print(stats["by_severity"])       # Dict of severity -> count
print(stats["by_status"])         # Dict of status -> count
print(stats["active_escalations"]) # Number of ongoing escalations
```

## Severity Levels

```python
AlertSeverity.INFO       # ℹ️  Routine events
AlertSeverity.WARNING    # ⚠️  Potential issues
AlertSeverity.CRITICAL   # 🚨 Immediate action required
```

## Configuration Examples

### Slack Only

```python
config = AlertConfig(
    slack_webhook="https://hooks.slack.com/...",
    slack_channel="#security-alerts",
)
```

### Slack + PagerDuty

```python
config = AlertConfig(
    slack_webhook="https://hooks.slack.com/...",
    pagerduty_api_key="YOUR_API_KEY",
    pagerduty_service_id="YOUR_SERVICE_ID",
)
```

### All Channels

```python
config = AlertConfig(
    slack_webhook="https://hooks.slack.com/...",
    slack_channel="#security-alerts",
    pagerduty_api_key="YOUR_API_KEY",
    pagerduty_service_id="YOUR_SERVICE_ID",
    email_config={
        "host": "smtp.gmail.com",
        "port": 587,
        "sender": "alerts@example.com",
        "recipients": ["security@example.com"],
        "user": "your_email@gmail.com",
        "password": "your_app_password",
    },
    sms_config={
        "account_sid": "YOUR_ACCOUNT_SID",
        "auth_token": "YOUR_AUTH_TOKEN",
        "from_number": "+1-555-0123",
        "recipients": ["+1-555-0100", "+1-555-0101"],
    },
)
```

### With Deduplication & Escalation

```python
config = AlertConfig(
    slack_webhook="https://hooks.slack.com/...",
    pagerduty_api_key="YOUR_API_KEY",
    pagerduty_service_id="YOUR_SERVICE_ID",

    # Deduplication
    dedup_enabled=True,
    dedup_window_seconds=300,  # 5 minutes

    # Escalation
    escalation_enabled=True,
    escalation_windows={
        0: 900,    # 15 min
        1: 900,    # 15 min
        2: 1800,   # 30 min
    },

    # Grouping
    grouping_enabled=True,
    grouping_window_seconds=300,

    # Suppression
    suppression_enabled=True,
    low_priority_environments={"dev", "staging"},
)
```

## Alert Channels

### Select Specific Channels

```python
alert_id = await engine.alert(
    title="Custom Alert",
    severity=AlertSeverity.CRITICAL,
    channels=["slack", "email"],  # Only Slack and Email
)
```

## Deduplication

### How It Works

```
Alert 1: "SQL Injection" from 192.168.1.100 → SENT
Alert 2: "SQL Injection" from 192.168.1.100 (1 min later) → SUPPRESSED (duplicate)
Alert 3: "SQL Injection" from 192.168.1.101 (different IP) → SENT

# After resolution:
Alert 4: "SQL Injection" from 192.168.1.100 → SENT (dedup reset)
```

## Escalation Timeline

```
CRITICAL Alerts Escalation:
  0 min   : Slack #security-alerts
  15 min  : If not acknowledged → PagerDuty incident (on-call engineer)
  30 min  : If still not acked → Email + SMS (security lead)
  60 min  : If still not acked → All channels (CTO)
```

## Alert Status Flow

```
PENDING → SENT → ACKNOWLEDGED → RESOLVED
  ↑           ↓
  └──────────SUPPRESSED
```

## API Error Handling

```python
try:
    alert_id = await engine.alert(
        title="Test Alert",
        severity=AlertSeverity.CRITICAL,
    )
except Exception as e:
    logger.error(f"Failed to send alert: {e}")
```

## Common Patterns

### Wrap in Async Context Manager

```python
async with AlertingEngine(config) as engine:
    alert_id = await engine.alert(...)
    # Resources automatically cleaned up
```

### Async Task Pattern

```python
async def send_critical_alert(title, description):
    async with AlertingEngine(config) as engine:
        await engine.alert(
            title=title,
            severity=AlertSeverity.CRITICAL,
            description=description,
        )

# Later:
asyncio.create_task(send_critical_alert(...))
```

### Batch Alerts

```python
for ip in suspicious_ips:
    await engine.alert(
        title="Suspicious IP Access",
        severity=AlertSeverity.WARNING,
        source_ip=ip,
        metadata={"risk_score": 0.75},
    )
```

## Alert Templates

### Authentication

```python
await engine.alert(
    title="Authentication from new location",
    severity=AlertSeverity.INFO,
    source_ip=new_ip,
    user_id=user_id,
    metadata={
        "event_type": "login",
        "country": "country_name",
    },
)
```

### Attack Attempt

```python
await engine.alert(
    title="SQL Injection Detected",
    severity=AlertSeverity.CRITICAL,
    source_ip=attacker_ip,
    payload="[REDACTED]",
    description="Attempted SQL injection in login form",
    tags={"attack_type": "sql_injection"},
    runbook_path="security/sqli-response",
)
```

### Rate Limit

```python
await engine.alert(
    title="Rate limit exceeded",
    severity=AlertSeverity.WARNING,
    source_ip=client_ip,
    description="API endpoint exceeded rate limit threshold",
    metadata={
        "endpoint": "/api/users",
        "requests_per_minute": 450,
        "limit": 400,
    },
)
```

### Anomaly Detected

```python
await engine.alert(
    title="Anomaly detected",
    severity=AlertSeverity.WARNING,
    user_id=user_id,
    description="Unusual behavior detected for user account",
    metadata={
        "anomaly_score": 0.72,
        "anomaly_type": "unusual_access_pattern",
    },
)
```

## Testing

### Unit Test Template

```python
@pytest.mark.asyncio
async def test_alert_deduplication():
    config = AlertConfig(slack_webhook="https://test")
    async with AlertingEngine(config) as engine:
        # First alert
        alert_id1 = await engine.alert(
            title="Test",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )
        assert engine.get_alert(alert_id1).status == AlertStatus.SENT

        # Duplicate
        alert_id2 = await engine.alert(
            title="Test",
            severity=AlertSeverity.CRITICAL,
            source_ip="192.168.1.100",
        )
        assert engine.get_alert(alert_id2).status == AlertStatus.SUPPRESSED
```

## Troubleshooting

### Alert Not Sending

Check configuration:
```python
# Verify webhook URL
print(config.slack_webhook)

# Check alert status
alert = engine.get_alert(alert_id)
print(alert.status)           # Should be "sent"
print(alert.channels_sent)    # Should have channel names
```

### Alerts Stuck in Escalation

Check if acknowledged:
```python
alert = engine.get_alert(alert_id)
if alert.status == AlertStatus.SENT and alert.escalation_level > 0:
    # Alert is escalating
    # Acknowledge to stop escalation
    await engine.acknowledge(alert_id, "user@example.com")
```

### Deduplication Too Aggressive

Adjust window:
```python
config = AlertConfig(
    slack_webhook="...",
    dedup_window_seconds=60,  # Shorter window = less deduplication
)
```

Or disable:
```python
config.dedup_enabled = False
```

## Environment Variables

```bash
# Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# PagerDuty
export PAGERDUTY_API_KEY="..."
export PAGERDUTY_SERVICE_ID="..."

# Email
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SENDER_EMAIL="alerts@example.com"

# SMS
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
export TWILIO_PHONE="+1-555-0123"
```

## Performance Tips

1. **Use Deduplication**: Prevents alert spam
2. **Group Related Alerts**: Easier to manage
3. **Set Maintenance Windows**: Avoid false positives during deploys
4. **Batch Alerts**: If sending many at once
5. **Monitor Escalation Task Count**: Don't let escalation tasks pile up

## See Also

- `ALERTING_SYSTEM_README.md` - Complete documentation
- `demos/demo_alerting.py` - Full working examples
- `HoloLoom/security/alerting/tests/test_alerting.py` - Test examples
