"""
Webhook Alerting System
=======================
Push alerts to external channels (Slack, Discord, Email) for critical safety events.

E2.3 Implementation - December 2025

Features:
- Slack webhook integration
- Discord webhook integration
- Email via SMTP
- Alert deduplication (5-minute window)
- Severity-based routing (CRITICAL → all, WARNING → Slack only)
- Configurable via environment variables

Environment Variables:
    ALERT_SLACK_WEBHOOK_URL - Slack incoming webhook URL
    ALERT_DISCORD_WEBHOOK_URL - Discord webhook URL
    ALERT_EMAIL_SMTP_HOST - SMTP server hostname
    ALERT_EMAIL_SMTP_PORT - SMTP server port (default: 587)
    ALERT_EMAIL_FROM - Sender email address
    ALERT_EMAIL_TO - Recipient email address(es), comma-separated
    ALERT_EMAIL_USERNAME - SMTP username (optional)
    ALERT_EMAIL_PASSWORD - SMTP password (optional)
"""

import asyncio
import hashlib
import json
import logging
import os
import smtplib
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger("HoloLoom.alignment.alerting")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    """Available alert channels."""
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"


@dataclass
class Alert:
    """Represents an alert to be dispatched."""
    severity: AlertSeverity
    title: str
    message: str
    source: str = "HoloLoom"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.severity.value}:{self.title}:{self.source}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class AlertConfig:
    """Configuration for alert dispatcher."""
    # Slack
    slack_webhook_url: Optional[str] = None

    # Discord
    discord_webhook_url: Optional[str] = None

    # Email
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_use_tls: bool = True

    # Deduplication
    dedup_window_seconds: float = 300.0  # 5 minutes

    # Routing
    warning_channels: List[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.SLACK]
    )
    critical_channels: List[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.SLACK, AlertChannel.DISCORD, AlertChannel.EMAIL]
    )
    info_channels: List[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.SLACK]
    )

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """Load configuration from environment variables."""
        email_to_str = os.getenv("ALERT_EMAIL_TO", "")
        email_to = [e.strip() for e in email_to_str.split(",") if e.strip()]

        return cls(
            slack_webhook_url=os.getenv("ALERT_SLACK_WEBHOOK_URL"),
            discord_webhook_url=os.getenv("ALERT_DISCORD_WEBHOOK_URL"),
            email_smtp_host=os.getenv("ALERT_EMAIL_SMTP_HOST"),
            email_smtp_port=int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587")),
            email_from=os.getenv("ALERT_EMAIL_FROM"),
            email_to=email_to,
            email_username=os.getenv("ALERT_EMAIL_USERNAME"),
            email_password=os.getenv("ALERT_EMAIL_PASSWORD"),
            email_use_tls=os.getenv("ALERT_EMAIL_USE_TLS", "true").lower() == "true"
        )


class AlertDispatcher:
    """
    Dispatches alerts to configured channels with deduplication.

    Usage:
        config = AlertConfig.from_env()
        dispatcher = AlertDispatcher(config)

        # Send alert
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="Deception Detected",
            message="Hidden goal behavior detected in agentic reasoning"
        )
        await dispatcher.dispatch(alert)

        # Or use convenience methods
        await dispatcher.critical("Security Breach", "Unauthorized access attempt")
        await dispatcher.warning("High Risk Action", "User requested file deletion")
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize dispatcher.

        Args:
            config: Alert configuration. If None, loads from environment.
        """
        self.config = config or AlertConfig.from_env()

        # Deduplication cache: fingerprint -> last_sent_timestamp
        self._sent_alerts: Dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_dispatched": 0,
            "deduplicated": 0,
            "by_severity": defaultdict(int),
            "by_channel": defaultdict(int),
            "failures": defaultdict(int)
        }

        # Channel handlers
        self._handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.DISCORD: self._send_discord,
            AlertChannel.EMAIL: self._send_email
        }

    async def dispatch(self, alert: Alert) -> Dict[str, bool]:
        """
        Dispatch alert to appropriate channels.

        Args:
            alert: Alert to dispatch

        Returns:
            Dict mapping channel name to success status
        """
        async with self._lock:
            # Check deduplication
            if self._is_duplicate(alert):
                self._stats["deduplicated"] += 1
                logger.debug(f"Alert deduplicated: {alert.title}")
                return {}

            # Mark as sent
            self._sent_alerts[alert.fingerprint] = time.time()

        # Determine channels based on severity
        channels = self._get_channels_for_severity(alert.severity)

        # Dispatch to each channel
        results = {}
        for channel in channels:
            if self._is_channel_configured(channel):
                try:
                    handler = self._handlers.get(channel)
                    if handler:
                        success = await handler(alert)
                        results[channel.value] = success
                        if success:
                            self._stats["by_channel"][channel.value] += 1
                        else:
                            self._stats["failures"][channel.value] += 1
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel.value}: {e}")
                    results[channel.value] = False
                    self._stats["failures"][channel.value] += 1

        if results:
            self._stats["total_dispatched"] += 1
            self._stats["by_severity"][alert.severity.value] += 1

        return results

    async def critical(
        self,
        title: str,
        message: str,
        source: str = "HoloLoom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Send CRITICAL severity alert."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        return await self.dispatch(alert)

    async def warning(
        self,
        title: str,
        message: str,
        source: str = "HoloLoom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Send WARNING severity alert."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        return await self.dispatch(alert)

    async def info(
        self,
        title: str,
        message: str,
        source: str = "HoloLoom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Send INFO severity alert."""
        alert = Alert(
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        return await self.dispatch(alert)

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert was sent recently (within dedup window)."""
        fingerprint = alert.fingerprint
        if fingerprint not in self._sent_alerts:
            return False

        last_sent = self._sent_alerts[fingerprint]
        elapsed = time.time() - last_sent
        return elapsed < self.config.dedup_window_seconds

    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get channels to use for given severity."""
        if severity == AlertSeverity.CRITICAL:
            return self.config.critical_channels
        elif severity == AlertSeverity.WARNING:
            return self.config.warning_channels
        else:  # INFO
            return self.config.info_channels

    def _is_channel_configured(self, channel: AlertChannel) -> bool:
        """Check if channel is properly configured."""
        if channel == AlertChannel.SLACK:
            return bool(self.config.slack_webhook_url)
        elif channel == AlertChannel.DISCORD:
            return bool(self.config.discord_webhook_url)
        elif channel == AlertChannel.EMAIL:
            return bool(
                self.config.email_smtp_host and
                self.config.email_from and
                self.config.email_to
            )
        return False

    async def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, cannot send Slack alert")
            return False

        if not self.config.slack_webhook_url:
            return False

        # Format Slack message
        color = self._severity_to_color(alert.severity)
        emoji = self._severity_to_emoji(alert.severity)

        payload = {
            "attachments": [{
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} {alert.title}",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": alert.message
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {alert.severity.value} | *Source:* {alert.source} | *Time:* <!date^{int(alert.timestamp)}^{{date_short_pretty}} {{time}}|{datetime.fromtimestamp(alert.timestamp).isoformat()}>"
                            }
                        ]
                    }
                ]
            }]
        }

        # Add metadata if present
        if alert.metadata:
            metadata_text = "\n".join(f"• *{k}:* {v}" for k, v in alert.metadata.items())
            payload["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{metadata_text}"
                }
            })

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.title}")
                        return True
                    else:
                        logger.error(f"Slack webhook failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    async def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, cannot send Discord alert")
            return False

        if not self.config.discord_webhook_url:
            return False

        # Format Discord embed
        color = self._severity_to_discord_color(alert.severity)
        emoji = self._severity_to_emoji(alert.severity)

        embed = {
            "title": f"{emoji} {alert.title}",
            "description": alert.message,
            "color": color,
            "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
            "footer": {
                "text": f"Source: {alert.source} | Severity: {alert.severity.value}"
            }
        }

        # Add metadata as fields
        if alert.metadata:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in alert.metadata.items()
            ]

        payload = {
            "embeds": [embed]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Discord alert sent: {alert.title}")
                        return True
                    else:
                        logger.error(f"Discord webhook failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def _send_email(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not all([
            self.config.email_smtp_host,
            self.config.email_from,
            self.config.email_to
        ]):
            return False

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_email_sync, alert)

    def _send_email_sync(self, alert: Alert) -> bool:
        """Synchronous email sending."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value}] {alert.title}"
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)

            # Plain text version
            text_content = f"""
HoloLoom Safety Alert
=====================

Severity: {alert.severity.value}
Title: {alert.title}
Source: {alert.source}
Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}

Message:
{alert.message}
"""
            if alert.metadata:
                text_content += "\nDetails:\n"
                for k, v in alert.metadata.items():
                    text_content += f"  - {k}: {v}\n"

            # HTML version
            emoji = self._severity_to_emoji(alert.severity)
            color = self._severity_to_color(alert.severity)

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .alert {{ border-left: 4px solid {color}; padding: 16px; background: #f8f9fa; margin: 16px 0; }}
        .title {{ font-size: 18px; font-weight: 600; margin-bottom: 8px; }}
        .message {{ color: #333; margin-bottom: 12px; }}
        .meta {{ font-size: 12px; color: #666; }}
        .details {{ margin-top: 12px; padding: 8px; background: #e9ecef; border-radius: 4px; }}
        .details dt {{ font-weight: 600; display: inline; }}
        .details dd {{ display: inline; margin-left: 8px; }}
    </style>
</head>
<body>
    <div class="alert">
        <div class="title">{emoji} {alert.title}</div>
        <div class="message">{alert.message}</div>
        <div class="meta">
            <strong>Severity:</strong> {alert.severity.value} |
            <strong>Source:</strong> {alert.source} |
            <strong>Time:</strong> {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
"""
            if alert.metadata:
                html_content += '<dl class="details">'
                for k, v in alert.metadata.items():
                    html_content += f"<dt>{k}:</dt><dd>{v}</dd><br>"
                html_content += "</dl>"

            html_content += """
    </div>
</body>
</html>
"""

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Connect and send
            if self.config.email_use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(
                    self.config.email_smtp_host,
                    self.config.email_smtp_port
                ) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    if self.config.email_username and self.config.email_password:
                        server.login(
                            self.config.email_username,
                            self.config.email_password
                        )
                    server.sendmail(
                        self.config.email_from,
                        self.config.email_to,
                        msg.as_string()
                    )
            else:
                with smtplib.SMTP(
                    self.config.email_smtp_host,
                    self.config.email_smtp_port
                ) as server:
                    if self.config.email_username and self.config.email_password:
                        server.login(
                            self.config.email_username,
                            self.config.email_password
                        )
                    server.sendmail(
                        self.config.email_from,
                        self.config.email_to,
                        msg.as_string()
                    )

            logger.info(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _severity_to_color(self, severity: AlertSeverity) -> str:
        """Get hex color for severity (Slack format)."""
        colors = {
            AlertSeverity.INFO: "#36a64f",     # Green
            AlertSeverity.WARNING: "#f2c744",   # Yellow/Amber
            AlertSeverity.CRITICAL: "#dc3545"   # Red
        }
        return colors.get(severity, "#808080")

    def _severity_to_discord_color(self, severity: AlertSeverity) -> int:
        """Get integer color for severity (Discord format)."""
        colors = {
            AlertSeverity.INFO: 0x36a64f,      # Green
            AlertSeverity.WARNING: 0xf2c744,    # Yellow/Amber
            AlertSeverity.CRITICAL: 0xdc3545    # Red
        }
        return colors.get(severity, 0x808080)

    def _severity_to_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity."""
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emojis.get(severity, "📢")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "total_dispatched": self._stats["total_dispatched"],
            "deduplicated": self._stats["deduplicated"],
            "by_severity": dict(self._stats["by_severity"]),
            "by_channel": dict(self._stats["by_channel"]),
            "failures": dict(self._stats["failures"]),
            "active_fingerprints": len(self._sent_alerts)
        }

    async def cleanup_expired(self):
        """Remove expired fingerprints from deduplication cache."""
        async with self._lock:
            now = time.time()
            expired = [
                fp for fp, ts in self._sent_alerts.items()
                if now - ts > self.config.dedup_window_seconds
            ]
            for fp in expired:
                del self._sent_alerts[fp]

            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired alert fingerprints")


# Global instance for easy access
_alert_dispatcher: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    """Get global AlertDispatcher instance."""
    global _alert_dispatcher
    if _alert_dispatcher is None:
        _alert_dispatcher = AlertDispatcher()
    return _alert_dispatcher


def set_alert_dispatcher(dispatcher: AlertDispatcher):
    """Set global AlertDispatcher instance."""
    global _alert_dispatcher
    _alert_dispatcher = dispatcher


# Integration with AlignmentMonitor
async def dispatch_alignment_alert(
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str = "AlignmentMonitor",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """
    Dispatch an alignment-related alert.

    Convenience function for integration with AlignmentMonitor.

    Args:
        severity: Alert severity
        title: Alert title
        message: Alert message
        source: Source system
        metadata: Additional metadata

    Returns:
        Dict mapping channel name to success status
    """
    dispatcher = get_alert_dispatcher()
    alert = Alert(
        severity=severity,
        title=title,
        message=message,
        source=source,
        metadata=metadata or {}
    )
    return await dispatcher.dispatch(alert)


# Convenience functions for common alignment alerts
async def alert_deception_detected(
    flag_type: str,
    description: str,
    confidence: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Alert for deception detection."""
    meta = metadata or {}
    meta["flag_type"] = flag_type
    meta["confidence"] = f"{confidence:.1%}"

    return await dispatch_alignment_alert(
        severity=AlertSeverity.CRITICAL,
        title=f"Deception Detected: {flag_type}",
        message=description,
        source="DeceptionDetector",
        metadata=meta
    )


async def alert_convergence_violation(
    violation_type: str,
    description: str,
    severity_str: str = "WARNING",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Alert for convergence violation."""
    meta = metadata or {}
    meta["violation_type"] = violation_type

    severity = AlertSeverity.CRITICAL if severity_str == "CRITICAL" else AlertSeverity.WARNING

    return await dispatch_alignment_alert(
        severity=severity,
        title=f"Convergence Violation: {violation_type}",
        message=description,
        source="InstrumentalConvergenceGuard",
        metadata=meta
    )


async def alert_high_risk_action(
    action: str,
    risk_level: str,
    outcome: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Alert for high-risk action gating."""
    meta = metadata or {}
    meta["action"] = action
    meta["risk_level"] = risk_level
    meta["outcome"] = outcome

    severity = AlertSeverity.CRITICAL if risk_level == "CRITICAL" else AlertSeverity.WARNING

    return await dispatch_alignment_alert(
        severity=severity,
        title=f"High Risk Action: {action}",
        message=f"Action '{action}' was {outcome} with risk level {risk_level}",
        source="SafetyGuardrails",
        metadata=meta
    )
