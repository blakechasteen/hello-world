"""
Alert Engine for Promptly Performance Dashboard

Monitors metrics and triggers alerts when thresholds are exceeded.
Supports multiple notification backends: webhook, Slack, email.

Features:
- Real-time threshold monitoring
- Configurable alert rules
- Multiple notification channels
- Alert history and acknowledgment
- Rate limiting to prevent alert fatigue

Usage:
    from analytics.alert_engine import AlertEngine, AlertRule, NotificationChannel

    # Create alert rule
    rule = AlertRule(
        id="high_latency",
        name="High Latency Alert",
        metric="avg_latency_ms",
        condition="greater_than",
        threshold=200.0,
        duration_seconds=300,  # Must persist for 5 minutes
        severity="warning"
    )

    # Create engine with notification channels
    engine = AlertEngine(
        db_path="metrics.db",
        channels=[
            WebhookChannel(url="https://example.com/webhook"),
            SlackChannel(webhook_url="https://hooks.slack.com/..."),
            EmailChannel(smtp_host="smtp.gmail.com", ...)
        ]
    )

    # Add rule
    await engine.add_rule(rule)

    # Start monitoring
    await engine.start()
"""

import asyncio
import json
import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Alert condition types"""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    BETWEEN = "between"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    metric: str  # e.g., "avg_latency_ms", "error_rate", "cache_hit_rate"
    condition: str  # "greater_than", "less_than", etc.
    threshold: float
    duration_seconds: int = 60  # How long condition must persist
    severity: str = "warning"
    enabled: bool = True
    channels: List[str] = field(default_factory=lambda: ["webhook"])  # Which channels to notify
    cooldown_seconds: int = 300  # Min time between repeated alerts
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_id: str
    rule_name: str
    severity: str
    message: str
    metric: str
    value: float
    threshold: float
    triggered_at: float
    status: str = "active"
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationChannel:
    """Base class for notification channels"""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Returns True if successful."""
        raise NotImplementedError


class WebhookChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    async def send(self, alert: Alert) -> bool:
        """Send alert via HTTP POST"""
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity,
            "message": alert.message,
            "metric": alert.metric,
            "value": alert.value,
            "threshold": alert.threshold,
            "triggered_at": alert.triggered_at,
            "status": alert.status
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Webhook notification failed: {e}")
            return False


class SlackChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def _format_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Format alert as Slack block message"""
        # Severity emoji and color
        severity_map = {
            "info": {"emoji": "ℹ️", "color": "#36a64f"},
            "warning": {"emoji": "⚠️", "color": "#ff9900"},
            "error": {"emoji": "❌", "color": "#ff0000"},
            "critical": {"emoji": "🚨", "color": "#cc0000"}
        }
        severity_info = severity_map.get(alert.severity, severity_map["info"])

        return {
            "attachments": [
                {
                    "color": severity_info["color"],
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{severity_info['emoji']} {alert.rule_name}"
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Severity:*\n{alert.severity.upper()}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Metric:*\n{alert.metric}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Current Value:*\n{alert.value:.2f}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Threshold:*\n{alert.threshold:.2f}"
                                }
                            ]
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Message:*\n{alert.message}"
                            }
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Alert ID: `{alert.id}` | Triggered: <t:{int(alert.triggered_at)}:F>"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        payload = self._format_slack_message(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Slack notification failed: {e}")
            return False


class EmailChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def _format_email(self, alert: Alert) -> MIMEMultipart:
        """Format alert as HTML email"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.rule_name}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        # Plain text version
        text = f"""
Promptly Performance Alert

Severity: {alert.severity.upper()}
Rule: {alert.rule_name}
Metric: {alert.metric}
Current Value: {alert.value:.2f}
Threshold: {alert.threshold:.2f}

{alert.message}

Alert ID: {alert.id}
Triggered: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.triggered_at))}
"""

        # HTML version
        severity_colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "critical": "#cc0000"
        }
        color = severity_colors.get(alert.severity, "#36a64f")

        html = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
        <h1 style="margin: 0;">Promptly Performance Alert</h1>
        <p style="margin: 5px 0 0 0;">{alert.rule_name}</p>
    </div>
    <div style="background: #f5f5f5; padding: 20px; border-radius: 0 0 8px 8px;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; font-weight: bold;">Severity:</td>
                <td style="padding: 8px;">{alert.severity.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold;">Metric:</td>
                <td style="padding: 8px;">{alert.metric}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold;">Current Value:</td>
                <td style="padding: 8px;">{alert.value:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold;">Threshold:</td>
                <td style="padding: 8px;">{alert.threshold:.2f}</td>
            </tr>
        </table>
        <div style="margin-top: 20px; padding: 15px; background: white; border-left: 4px solid {color};">
            <p style="margin: 0;"><strong>Message:</strong></p>
            <p style="margin: 10px 0 0 0;">{alert.message}</p>
        </div>
        <div style="margin-top: 20px; font-size: 12px; color: #666;">
            <p>Alert ID: {alert.id}</p>
            <p>Triggered: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.triggered_at))}</p>
        </div>
    </div>
</body>
</html>
"""

        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        msg.attach(part1)
        msg.attach(part2)

        return msg

    async def send(self, alert: Alert) -> bool:
        """Send alert via email (runs in thread pool to avoid blocking)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_sync, alert)

    def _send_sync(self, alert: Alert) -> bool:
        """Synchronous email sending"""
        try:
            msg = self._format_email(alert)

            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)

            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Email notification failed: {e}")
            return False


class AlertEngine:
    """Main alert monitoring engine"""

    def __init__(
        self,
        db_path: str = "test_metrics.db",
        channels: Optional[List[NotificationChannel]] = None,
        check_interval: float = 10.0
    ):
        self.db_path = db_path
        self.channels = channels or []
        self.check_interval = check_interval

        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.last_alert_time: Dict[str, float] = {}  # For cooldown

        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize alert database
        self._init_alert_db()

    def _init_alert_db(self):
        """Create alert tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Alert rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metric TEXT NOT NULL,
                condition TEXT NOT NULL,
                threshold REAL NOT NULL,
                duration_seconds INTEGER DEFAULT 60,
                severity TEXT DEFAULT 'warning',
                enabled INTEGER DEFAULT 1,
                channels TEXT DEFAULT '["webhook"]',
                cooldown_seconds INTEGER DEFAULT 300,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Alert history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                triggered_at REAL NOT NULL,
                status TEXT DEFAULT 'active',
                acknowledged_at REAL,
                resolved_at REAL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (rule_id) REFERENCES alert_rules(id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_triggered ON alert_history(triggered_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_status ON alert_history(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_rule ON alert_history(rule_id)")

        conn.commit()
        conn.close()

    async def add_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.rules[rule.id] = rule

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO alert_rules
            (id, name, metric, condition, threshold, duration_seconds, severity, enabled, channels, cooldown_seconds, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.id,
            rule.name,
            rule.metric,
            rule.condition,
            rule.threshold,
            rule.duration_seconds,
            rule.severity,
            1 if rule.enabled else 0,
            json.dumps(rule.channels),
            rule.cooldown_seconds,
            json.dumps(rule.metadata)
        ))

        conn.commit()
        conn.close()

    async def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
        conn.commit()
        conn.close()

    async def load_rules(self):
        """Load rules from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM alert_rules WHERE enabled = 1")
        rows = cursor.fetchall()

        for row in rows:
            rule = AlertRule(
                id=row[0],
                name=row[1],
                metric=row[2],
                condition=row[3],
                threshold=row[4],
                duration_seconds=row[5],
                severity=row[6],
                enabled=bool(row[7]),
                channels=json.loads(row[8]),
                cooldown_seconds=row[9],
                metadata=json.loads(row[10])
            )
            self.rules[rule.id] = rule

        conn.close()

    async def check_metrics(self):
        """Check current metrics against alert rules"""
        # Get recent metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get metrics from last 5 minutes
        cutoff_time = time.time() - 300
        cursor.execute("""
            SELECT tags_json, values_json, timestamp
            FROM events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff_time,))

        events = cursor.fetchall()
        conn.close()

        if not events:
            return

        # Aggregate metrics
        metrics = self._aggregate_metrics(events)

        # Check each rule
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            last_alert = self.last_alert_time.get(rule_id, 0)
            if time.time() - last_alert < rule.cooldown_seconds:
                continue

            # Get metric value
            value = metrics.get(rule.metric)
            if value is None:
                continue

            # Check condition
            triggered = self._check_condition(value, rule.condition, rule.threshold)

            if triggered:
                await self._trigger_alert(rule, value)

    def _aggregate_metrics(self, events: List[tuple]) -> Dict[str, float]:
        """Aggregate metrics from events"""
        from collections import defaultdict

        metrics = defaultdict(list)

        for tags_json, values_json, timestamp in events:
            values = json.loads(values_json)
            for key, value in values.items():
                metrics[key].append(value)

        # Average each metric
        return {
            key: sum(values) / len(values)
            for key, values in metrics.items()
        }

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.01
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.01
        return False

    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert"""
        alert_id = f"{rule.id}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.metric} is {value:.2f}, exceeding threshold of {rule.threshold:.2f}",
            metric=rule.metric,
            value=value,
            threshold=rule.threshold,
            triggered_at=time.time(),
            status="active"
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.last_alert_time[rule.id] = time.time()

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alert_history
            (id, rule_id, rule_name, severity, message, metric, value, threshold, triggered_at, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.rule_id,
            alert.rule_name,
            alert.severity,
            alert.message,
            alert.metric,
            alert.value,
            alert.threshold,
            alert.triggered_at,
            alert.status,
            json.dumps(alert.metadata)
        ))

        conn.commit()
        conn.close()

        # Send notifications
        for channel in self.channels:
            if any(ch_type in rule.channels for ch_type in ["webhook", "slack", "email"]):
                success = await channel.send(alert)
                if not success:
                    print(f"Failed to send alert {alert_id} via {channel.__class__.__name__}")

    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "acknowledged"
            alert.acknowledged_at = time.time()

            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alert_history
                SET status = 'acknowledged', acknowledged_at = ?
                WHERE id = ?
            """, (alert.acknowledged_at, alert_id))
            conn.commit()
            conn.close()

    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            alert.resolved_at = time.time()

            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alert_history
                SET status = 'resolved', resolved_at = ?
                WHERE id = ?
            """, (alert.resolved_at, alert_id))
            conn.commit()
            conn.close()

            # Remove from active alerts
            del self.active_alerts[alert_id]

    async def get_alert_history(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        rule_id: Optional[str] = None
    ) -> List[Alert]:
        """Get alert history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM alert_history WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if rule_id:
            query += " AND rule_id = ?"
            params.append(rule_id)

        query += " ORDER BY triggered_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert = Alert(
                id=row[0],
                rule_id=row[1],
                rule_name=row[2],
                severity=row[3],
                message=row[4],
                metric=row[5],
                value=row[6],
                threshold=row[7],
                triggered_at=row[8],
                status=row[9],
                acknowledged_at=row[10],
                resolved_at=row[11],
                metadata=json.loads(row[12])
            )
            alerts.append(alert)

        return alerts

    async def start(self):
        """Start the alert monitoring loop"""
        if self._running:
            return

        self._running = True
        await self.load_rules()
        self._task = asyncio.create_task(self._monitoring_loop())

    async def stop(self):
        """Stop the alert monitoring loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self.check_metrics()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
