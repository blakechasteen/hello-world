"""Threshold-based alerting system for metrics.

Provides:
- AlertSeverity levels (INFO, WARNING, CRITICAL)
- ThresholdRule for defining alert conditions
- AlertingSystem with cooldown and notification handlers
- Default rules for common metrics
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metrics_collector import MetricType


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThresholdRule:
    """Define when to trigger an alert.

    Attributes:
        metric_type: Type of metric to monitor
        threshold: Threshold value
        comparison: Comparison operator ("gt", "lt", "gte", "lte", "eq")
        severity: Alert severity level
        message_template: Template for alert message (supports {metric}, {threshold}, {actual})
        cooldown_seconds: Minimum time between alerts for same rule (default: 300)
    """

    metric_type: "MetricType"
    threshold: float
    comparison: str  # "gt", "lt", "gte", "lte", "eq"
    severity: AlertSeverity
    message_template: str = "{metric} {comparison} {threshold}"
    cooldown_seconds: int = 300  # 5 minutes default


@dataclass
class Alert:
    """A triggered alert.

    Attributes:
        rule: The ThresholdRule that triggered this alert
        actual_value: The value that triggered the alert
        timestamp: When the alert was triggered
        acknowledged: Whether the alert has been acknowledged
        metadata: Additional context about the alert
    """

    rule: ThresholdRule
    actual_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def message(self) -> str:
        """Generate alert message from template."""
        return self.rule.message_template.format(
            metric=self.rule.metric_type.value,
            comparison=self.rule.comparison,
            threshold=self.rule.threshold,
            actual=self.actual_value
        )


class AlertingSystem:
    """Threshold-based alerting for metrics.

    Features:
    - Multiple threshold rules per metric type
    - Cooldown to prevent alert spam
    - Pluggable notification handlers
    - Alert acknowledgment tracking
    """

    def __init__(self):
        """Initialize alerting system."""
        self.rules: list[ThresholdRule] = []
        self.alerts: list[Alert] = []
        self.handlers: list[Callable[[Alert], None]] = []
        self._last_alert_time: dict[str, datetime] = {}

    def add_rule(self, rule: ThresholdRule) -> None:
        """Register a threshold rule.

        Args:
            rule: ThresholdRule to add
        """
        self.rules.append(rule)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register alert notification handler.

        Args:
            handler: Callable that receives Alert objects (e.g., Slack, email)
        """
        self.handlers.append(handler)

    def check(
        self,
        metric_type: "MetricType",
        value: float,
        metadata: dict[str, Any] | None = None
    ) -> Alert | None:
        """Check value against rules, return alert if triggered.

        Args:
            metric_type: Type of metric being checked
            value: Current metric value
            metadata: Optional context to include in alert

        Returns:
            Alert if triggered, None otherwise
        """
        for rule in self.rules:
            if rule.metric_type != metric_type:
                continue

            triggered = self._compare(value, rule.threshold, rule.comparison)
            if not triggered:
                continue

            # Check cooldown
            rule_key = f"{rule.metric_type.value}:{rule.comparison}:{rule.threshold}"
            last_time = self._last_alert_time.get(rule_key)
            now = datetime.now()

            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue  # Still in cooldown

            # Create and store alert
            alert = Alert(
                rule=rule,
                actual_value=value,
                timestamp=now,
                metadata=metadata or {}
            )
            self.alerts.append(alert)
            self._last_alert_time[rule_key] = now

            # Notify handlers
            for handler in self.handlers:
                try:
                    handler(alert)
                except Exception:
                    pass  # Don't break on handler failure

            return alert

        return None

    def get_active_alerts(self) -> list[Alert]:
        """Get unacknowledged alerts.

        Returns:
            List of alerts that have not been acknowledged
        """
        return [a for a in self.alerts if not a.acknowledged]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts filtered by severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of alerts with matching severity
        """
        return [a for a in self.alerts if a.rule.severity == severity]

    def acknowledge(self, alert: Alert) -> None:
        """Mark alert as acknowledged.

        Args:
            alert: Alert to acknowledge
        """
        alert.acknowledged = True

    def acknowledge_all(self) -> int:
        """Acknowledge all active alerts.

        Returns:
            Number of alerts acknowledged
        """
        count = 0
        for alert in self.alerts:
            if not alert.acknowledged:
                alert.acknowledged = True
                count += 1
        return count

    def clear_old_alerts(self, max_age_hours: int = 24) -> int:
        """Remove alerts older than specified age.

        Args:
            max_age_hours: Maximum age of alerts to keep

        Returns:
            Number of alerts removed
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        before = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff]
        return before - len(self.alerts)

    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        """Compare value against threshold.

        Args:
            value: Current value
            threshold: Threshold to compare against
            comparison: Comparison operator

        Returns:
            True if comparison is satisfied
        """
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        return ops.get(comparison, lambda v, t: False)(value, threshold)


def get_default_rules() -> list[ThresholdRule]:
    """Get default threshold rules for common metrics.

    Returns:
        List of sensible default rules
    """
    from .metrics_collector import MetricType

    return [
        ThresholdRule(
            metric_type=MetricType.ERROR_RATE,
            threshold=0.05,  # 5%
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            message_template="Error rate {actual:.1%} exceeds {threshold:.1%}"
        ),
        ThresholdRule(
            metric_type=MetricType.LATENCY_MS,
            threshold=500,
            comparison="gt",
            severity=AlertSeverity.WARNING,
            message_template="Latency {actual:.0f}ms exceeds {threshold:.0f}ms"
        ),
        ThresholdRule(
            metric_type=MetricType.PASS_RATE,
            threshold=0.95,
            comparison="lt",
            severity=AlertSeverity.WARNING,
            message_template="Pass rate {actual:.1%} below {threshold:.1%}"
        ),
        ThresholdRule(
            metric_type=MetricType.SECURITY_VIOLATIONS,
            threshold=0,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            message_template="Security violations detected: {actual:.0f}"
        ),
        ThresholdRule(
            metric_type=MetricType.SAFETY_RISK_LEVEL,
            threshold=0.5,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            message_template="Safety risk level {actual:.2f} exceeds {threshold:.2f}"
        ),
        ThresholdRule(
            metric_type=MetricType.FABRIC_CONFIDENCE,
            threshold=0.6,
            comparison="lt",
            severity=AlertSeverity.WARNING,
            message_template="Fabric confidence {actual:.2f} below {threshold:.2f}"
        ),
    ]


def create_alerting_system(
    with_defaults: bool = True
) -> AlertingSystem:
    """Factory function for AlertingSystem.

    Args:
        with_defaults: Whether to include default rules

    Returns:
        New AlertingSystem instance
    """
    system = AlertingSystem()
    if with_defaults:
        for rule in get_default_rules():
            system.add_rule(rule)
    return system
