"""
Core alerting engine with multi-channel support, deduplication, and escalation.

**Implemented: 2025-11-15**
**Lines**: 748
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        """Get emoji for severity level."""
        mapping = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }
        return mapping[self]

    @property
    def priority(self) -> int:
        """Get priority (higher = more severe)."""
        mapping = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
        }
        return mapping[self]


class AlertStatus(Enum):
    """Alert status."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""

    alert_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    payload: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.PENDING
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    channels_sent: List[str] = field(default_factory=list)
    escalation_level: int = 0
    deduplication_key: Optional[str] = None
    group_id: Optional[str] = None
    runbook_url: Optional[str] = None

    def compute_dedup_key(self, include_source: bool = True) -> str:
        """Compute deduplication key."""
        parts = [self.title, self.severity.value]
        if include_source and self.source_ip:
            parts.append(self.source_ip)
        key_str = ":".join(parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "channels_sent": self.channels_sent,
            "escalation_level": self.escalation_level,
            "deduplication_key": self.deduplication_key,
            "group_id": self.group_id,
            "runbook_url": self.runbook_url,
        }


@dataclass
class AlertConfig:
    """Alerting configuration."""

    # Channels
    slack_webhook: Optional[str] = None
    slack_channel: str = "#security-alerts"
    pagerduty_api_key: Optional[str] = None
    pagerduty_service_id: Optional[str] = None
    email_config: Optional[Dict[str, Any]] = None
    sms_config: Optional[Dict[str, Any]] = None

    # Deduplication
    dedup_window_seconds: int = 300  # 5 minutes
    dedup_enabled: bool = True

    # Escalation
    escalation_enabled: bool = True
    escalation_windows: Dict[int, int] = field(default_factory=lambda: {
        0: 900,      # 15 minutes to escalation level 1
        1: 900,      # 15 minutes to escalation level 2
        2: 1800,     # 30 minutes to escalation level 3
    })

    # Grouping
    grouping_enabled: bool = True
    grouping_window_seconds: int = 300

    # Suppression
    suppression_enabled: bool = True
    low_priority_environments: Set[str] = field(default_factory=lambda: {"dev", "staging"})

    # Runbook
    runbook_base_url: str = "https://wiki.hololoom.com/security"


class AlertingEngine:
    """Multi-channel alerting with deduplication and escalation."""

    def __init__(self, config: AlertConfig):
        """Initialize alerting engine."""
        self.config = config
        self.alerts: Dict[str, Alert] = {}
        self.dedup_engine = None
        self.escalation_tasks: Dict[str, asyncio.Task] = {}
        self.alert_groups: Dict[str, List[Alert]] = {}
        self.maintenance_windows: List[tuple] = []
        self.handlers: Dict[str, Callable] = {
            "slack": self._send_slack,
            "pagerduty": self._send_pagerduty,
            "email": self._send_email,
            "sms": self._send_sms,
        }

    async def __aenter__(self):
        """Context manager entry."""
        from .deduplication import DeduplicationEngine
        self.dedup_engine = DeduplicationEngine(
            window_seconds=self.config.dedup_window_seconds
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def close(self):
        """Cleanup resources."""
        for task in self.escalation_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def alert(
        self,
        title: str,
        severity: AlertSeverity,
        description: str = "",
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        payload: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
        runbook_path: Optional[str] = None,
    ) -> str:
        """Send an alert."""

        # Create alert
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            payload=payload,
            tags=tags or {},
            metadata=metadata or {},
        )

        # Compute deduplication key
        alert.deduplication_key = alert.compute_dedup_key(include_source=True)

        # Set runbook URL
        if runbook_path:
            alert.runbook_url = f"{self.config.runbook_base_url}/{runbook_path}"

        # Check suppression
        if self._is_suppressed(alert):
            alert.status = AlertStatus.SUPPRESSED
            logger.info(f"Alert suppressed: {alert.alert_id}")
            return alert.alert_id

        # Check deduplication
        if self.config.dedup_enabled and self.dedup_engine:
            dedup_result = await self.dedup_engine.check_dedup(alert)
            if dedup_result.is_duplicate:
                alert.status = AlertStatus.SUPPRESSED
                logger.info(f"Alert deduplicated: {alert.alert_id}")
                return alert.alert_id

        # Check grouping
        if self.config.grouping_enabled:
            group_id = self._find_or_create_group(alert)
            alert.group_id = group_id

        # Store alert
        self.alerts[alert.alert_id] = alert

        # Determine channels
        if channels is None:
            channels = self._select_channels(severity)

        # Send through channels
        alert.status = AlertStatus.SENT
        for channel in channels:
            try:
                handler = self.handlers.get(channel)
                if handler:
                    await handler(alert)
                    alert.channels_sent.append(channel)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")

        # Start escalation
        if self.config.escalation_enabled and severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
            task = asyncio.create_task(self._escalate_alert(alert))
            self.escalation_tasks[alert.alert_id] = task

        logger.info(f"Alert sent: {alert.alert_id} ({severity.value}) via {alert.channels_sent}")
        return alert.alert_id

    async def acknowledge(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by

        # Cancel escalation task
        if alert_id in self.escalation_tasks:
            self.escalation_tasks[alert_id].cancel()
            del self.escalation_tasks[alert_id]

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    async def resolve(self, alert_id: str, resolved_by: str, reason: str = "") -> bool:
        """Resolve an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            logger.warning(f"Alert not found: {alert_id}")
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.resolved_by = resolved_by

        if reason:
            alert.metadata["resolution_reason"] = reason

        # Cancel escalation task
        if alert_id in self.escalation_tasks:
            self.escalation_tasks[alert_id].cancel()
            del self.escalation_tasks[alert_id]

        # Reset deduplication window
        if self.dedup_engine:
            await self.dedup_engine.reset_dedup(alert.deduplication_key)

        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True

    async def _escalate_alert(self, alert: Alert):
        """Escalate alert after timeout."""
        try:
            for level, window in self.config.escalation_windows.items():
                await asyncio.sleep(window)

                # Check if alert is acknowledged
                current = self.alerts.get(alert.alert_id)
                if current and current.status == AlertStatus.ACKNOWLEDGED:
                    break

                # Escalate
                alert.escalation_level = level + 1
                channels = self._get_escalation_channels(alert.escalation_level)

                logger.warning(f"Escalating alert {alert.alert_id} to level {alert.escalation_level}")

                for channel in channels:
                    try:
                        handler = self.handlers.get(channel)
                        if handler:
                            await handler(alert, is_escalation=True)
                    except Exception as e:
                        logger.error(f"Failed to escalate alert via {channel}: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Escalation task error: {e}")

    def _select_channels(self, severity: AlertSeverity) -> List[str]:
        """Select channels based on severity."""
        channels = []

        if self.config.slack_webhook:
            channels.append("slack")

        if severity == AlertSeverity.CRITICAL:
            if self.config.pagerduty_api_key:
                channels.append("pagerduty")
            if self.config.email_config:
                channels.append("email")
        elif severity == AlertSeverity.WARNING:
            if self.config.slack_webhook:
                channels.append("slack")

        return channels if channels else ["slack"]

    def _get_escalation_channels(self, level: int) -> List[str]:
        """Get channels for escalation level."""
        escalation_map = {
            0: ["slack"],
            1: ["pagerduty", "slack"],
            2: ["pagerduty", "email", "sms"],
            3: ["pagerduty", "email", "sms"],
        }
        return escalation_map.get(level, ["slack"])

    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        # Check maintenance windows
        now = datetime.utcnow()
        for start, end in self.maintenance_windows:
            if start <= now <= end:
                return True

        # Check low-priority environment
        env = alert.metadata.get("environment", "").lower()
        if env in self.config.low_priority_environments and alert.severity == AlertSeverity.INFO:
            return True

        return False

    def _find_or_create_group(self, alert: Alert) -> str:
        """Find or create alert group."""
        # Simple grouping by severity + alert type prefix
        group_key = f"{alert.severity.value}:{alert.title.split()[0]}"

        # Check if group exists within grouping window
        now = datetime.utcnow()
        if group_key in self.alert_groups:
            group = self.alert_groups[group_key]
            if group:
                last_alert = group[-1]
                if (now - last_alert.timestamp).total_seconds() < self.config.grouping_window_seconds:
                    group_id = group[0].group_id or str(uuid4())
                    group.append(alert)
                    return group_id

        # Create new group
        group_id = str(uuid4())
        self.alert_groups[group_key] = [alert]
        return group_id

    async def _send_slack(self, alert: Alert, is_escalation: bool = False):
        """Send alert via Slack."""
        if not self.config.slack_webhook:
            return

        try:
            import aiohttp

            # Build message
            escalation_suffix = f" [ESCALATED - Level {alert.escalation_level}]" if is_escalation else ""
            text = f"{alert.severity.emoji} {alert.title}{escalation_suffix}"

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": text,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{alert.severity.value.upper()}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                        },
                    ] + (
                        [
                            {
                                "type": "mrkdwn",
                                "text": f"*Source IP:*\n{alert.source_ip}",
                            },
                        ] if alert.source_ip else []
                    ) + (
                        [
                            {
                                "type": "mrkdwn",
                                "text": f"*User:*\n{alert.user_id}",
                            },
                        ] if alert.user_id else []
                    ),
                },
            ]

            if alert.description:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:*\n{alert.description}",
                    },
                })

            # Action buttons
            actions = []
            if alert.status == AlertStatus.SENT:
                actions = [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Acknowledge",
                        },
                        "value": alert.alert_id,
                        "action_id": f"ack_{alert.alert_id}",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Resolve",
                        },
                        "value": alert.alert_id,
                        "action_id": f"resolve_{alert.alert_id}",
                    },
                ]

            if actions:
                blocks.append({
                    "type": "actions",
                    "elements": actions,
                })

            payload = {
                "channel": self.config.slack_channel,
                "blocks": blocks,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Slack send failed: {resp.status}")

        except ImportError:
            logger.warning("aiohttp not installed, skipping Slack notification")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_pagerduty(self, alert: Alert, is_escalation: bool = False):
        """Send alert via PagerDuty."""
        if not self.config.pagerduty_api_key or not self.config.pagerduty_service_id:
            return

        try:
            import aiohttp

            headers = {
                "Authorization": f"Token token={self.config.pagerduty_api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "routing_key": self.config.pagerduty_service_id,
                "event_action": "trigger",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "severity": alert.severity.value,
                    "source": alert.source_ip or "HoloLoom",
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": {
                        "description": alert.description,
                        "user": alert.user_id,
                        "tags": alert.tags,
                    },
                },
            }

            if alert.runbook_url:
                payload["links"] = [{"href": alert.runbook_url, "text": "Runbook"}]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status != 202:
                        logger.error(f"PagerDuty send failed: {resp.status}")

        except ImportError:
            logger.warning("aiohttp not installed, skipping PagerDuty notification")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

    async def _send_email(self, alert: Alert, is_escalation: bool = False):
        """Send alert via email."""
        if not self.config.email_config:
            return

        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            config = self.config.email_config

            # Build email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg["From"] = config.get("sender", "alerts@hololoom.com")
            msg["To"] = ", ".join(config.get("recipients", []))

            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2>{alert.severity.emoji} {alert.title}</h2>
                    <table border="1" cellpadding="5">
                        <tr><th>Field</th><th>Value</th></tr>
                        <tr><td>Severity</td><td>{alert.severity.value.upper()}</td></tr>
                        <tr><td>Time</td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
                        {f'<tr><td>Source IP</td><td>{alert.source_ip}</td></tr>' if alert.source_ip else ''}
                        {f'<tr><td>User</td><td>{alert.user_id}</td></tr>' if alert.user_id else ''}
                        <tr><td>Alert ID</td><td>{alert.alert_id}</td></tr>
                    </table>
                    <p><strong>Description:</strong></p>
                    <p>{alert.description}</p>
                    {f'<p><a href="{alert.runbook_url}">View Runbook</a></p>' if alert.runbook_url else ''}
                </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Send email
            async with aiosmtplib.SMTP(hostname=config.get("host"), port=config.get("port", 587)) as smtp:
                await smtp.login(config.get("user"), config.get("password"))
                await smtp.send_message(msg)

        except ImportError:
            logger.warning("aiosmtplib not installed, skipping email notification")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_sms(self, alert: Alert, is_escalation: bool = False):
        """Send alert via SMS (Twilio)."""
        if not self.config.sms_config:
            return

        try:
            from twilio.rest import Client

            config = self.config.sms_config
            client = Client(config.get("account_sid"), config.get("auth_token"))

            message = f"{alert.severity.emoji} {alert.title} - {alert.alert_id}"

            for phone in config.get("recipients", []):
                message_obj = client.messages.create(
                    body=message,
                    from_=config.get("from_number"),
                    to=phone
                )
                logger.info(f"SMS sent: {message_obj.sid}")

        except ImportError:
            logger.warning("twilio not installed, skipping SMS notification")
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)

    def get_alerts(self, status: Optional[AlertStatus] = None, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alerts by status and/or severity."""
        alerts = list(self.alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def add_maintenance_window(self, start: datetime, end: datetime):
        """Add maintenance window (alerts suppressed)."""
        self.maintenance_windows.append((start, end))
        logger.info(f"Maintenance window added: {start} - {end}")

    def remove_maintenance_window(self, start: datetime, end: datetime):
        """Remove maintenance window."""
        self.maintenance_windows = [w for w in self.maintenance_windows if w != (start, end)]
        logger.info(f"Maintenance window removed: {start} - {end}")

    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        alerts = list(self.alerts.values())

        return {
            "total_alerts": len(alerts),
            "by_severity": {
                severity.value: len([a for a in alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "by_status": {
                status.value: len([a for a in alerts if a.status == status])
                for status in AlertStatus
            },
            "active_escalations": len(self.escalation_tasks),
            "maintenance_windows": len(self.maintenance_windows),
        }
