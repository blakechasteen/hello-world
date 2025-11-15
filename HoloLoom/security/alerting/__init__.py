"""
HoloLoom Security Alerting System

Automated multi-channel alerting with deduplication, escalation, and suppression.

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/security/alerting/`

Features:
- Multi-channel alerting (Slack, PagerDuty, email, SMS)
- Alert severity levels (INFO, WARNING, CRITICAL)
- Alert deduplication and grouping
- Escalation policies with time-based escalation
- On-call rotation integration
- Maintenance window suppression
- Complete audit trail

Quick Start:
```python
from HoloLoom.security.alerting import AlertingEngine, AlertSeverity

engine = AlertingEngine(
    slack_webhook="https://hooks.slack.com/...",
    pagerduty_api_key="YOUR_API_KEY",
    email_config={"host": "smtp.gmail.com", ...}
)

# Send alert
alert_id = await engine.alert(
    title="SQL Injection Detected",
    severity=AlertSeverity.CRITICAL,
    source_ip="192.168.1.100",
    payload="[REDACTED]"
)
```
"""

from .core import (
    AlertSeverity,
    Alert,
    AlertStatus,
    AlertingEngine,
    AlertConfig,
)
from .escalation import EscalationPolicy, EscalationLevel
from .deduplication import DeduplicationEngine, DeduplicationRule

__all__ = [
    "AlertSeverity",
    "Alert",
    "AlertStatus",
    "AlertingEngine",
    "AlertConfig",
    "EscalationPolicy",
    "EscalationLevel",
    "DeduplicationEngine",
    "DeduplicationRule",
]
