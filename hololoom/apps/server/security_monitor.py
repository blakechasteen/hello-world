"""
Security Monitoring and Alerting System

Comprehensive security monitoring with:
- Real-time event tracking
- Threshold-based alerting
- Email and Slack notifications
- Prometheus metrics export
- Security dashboard generation

Created: 2025-11-26
"""

import asyncio
import json
import logging
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Dict, List, Optional, Set, Deque, Any, Tuple
import aiohttp
import hashlib
from fastapi import WebSocket
import re

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events"""
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    WAF_VIOLATION = "waf_violation"
    INVALID_SIGNATURE = "invalid_signature"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BRUTE_FORCE = "brute_force"
    DOS_ATTACK = "dos_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALY = "anomaly"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data"""
    event_type: SecurityEventType
    severity: AlertSeverity
    timestamp: datetime
    client_ip: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    request_path: Optional[str] = None
    rule_id: Optional[str] = None
    fingerprint: Optional[str] = None  # For deduplication


@dataclass
class Alert:
    """Security alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    events: List[SecurityEvent]
    action_required: bool = False
    auto_resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class AttackerProfile:
    """Profile of potential attacker"""
    ip_address: str
    first_seen: datetime
    last_seen: datetime
    total_events: int = 0
    event_types: Dict[str, int] = field(default_factory=dict)
    blocked: bool = False
    risk_score: float = 0.0
    geographic_location: Optional[str] = None
    user_agents: Set[str] = field(default_factory=set)


class SecurityMonitor:
    """Main security monitoring system"""

    def __init__(
        self,
        alert_threshold_auth_failures: int = 5,
        alert_threshold_rate_limit: int = 10,
        alert_threshold_waf_violations: int = 3,
        time_window_seconds: int = 60,
        enable_email_alerts: bool = False,
        enable_slack_alerts: bool = False,
        enable_prometheus: bool = True,
        smtp_config: Optional[Dict] = None,
        slack_webhook_url: Optional[str] = None
    ):
        # Configuration
        self.alert_threshold_auth_failures = alert_threshold_auth_failures
        self.alert_threshold_rate_limit = alert_threshold_rate_limit
        self.alert_threshold_waf_violations = alert_threshold_waf_violations
        self.time_window_seconds = time_window_seconds
        self.enable_email_alerts = enable_email_alerts
        self.enable_slack_alerts = enable_slack_alerts
        self.enable_prometheus = enable_prometheus
        self.smtp_config = smtp_config or {}
        self.slack_webhook_url = slack_webhook_url

        # Event storage
        self.events: Deque[SecurityEvent] = deque(maxlen=10000)
        self.events_by_ip: Dict[str, Deque[SecurityEvent]] = defaultdict(lambda: deque(maxlen=1000))
        self.events_by_type: Dict[SecurityEventType, Deque[SecurityEvent]] = defaultdict(lambda: deque(maxlen=1000))

        # Attacker tracking
        self.attacker_profiles: Dict[str, AttackerProfile] = {}
        self.blocked_ips: Set[str] = set()

        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: Deque[Alert] = deque(maxlen=1000)
        self.alert_counter = 0

        # Metrics for Prometheus
        self.metrics = {
            "security_events_total": defaultdict(int),
            "active_alerts_count": 0,
            "blocked_ips_count": 0,
            "auth_failures_per_minute": 0,
            "waf_violations_per_minute": 0,
            "rate_limits_per_minute": 0,
            "high_risk_ips": 0
        }

        # WebSocket connections for real-time dashboard
        self.websocket_clients: Set[WebSocket] = set()

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

        logger.info("SecurityMonitor initialized")

    async def start(self):
        """Start background monitoring tasks"""

        # Start cleanup task
        self.background_tasks.append(
            asyncio.create_task(self._cleanup_old_events())
        )

        # Start metrics calculation task
        self.background_tasks.append(
            asyncio.create_task(self._calculate_metrics())
        )

        # Start anomaly detection task
        self.background_tasks.append(
            asyncio.create_task(self._detect_anomalies())
        )

        logger.info("Security monitoring started")

    async def stop(self):
        """Stop background monitoring tasks"""

        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        logger.info("Security monitoring stopped")

    async def log_event(
        self,
        event_type: SecurityEventType,
        client_ip: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        details: Optional[Dict] = None,
        user_id: Optional[str] = None,
        request_path: Optional[str] = None,
        rule_id: Optional[str] = None
    ):
        """Log a security event"""

        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            client_ip=client_ip,
            details=details or {},
            user_id=user_id,
            request_path=request_path,
            rule_id=rule_id,
            fingerprint=self._generate_fingerprint(event_type, client_ip, rule_id)
        )

        # Store event
        self.events.append(event)
        self.events_by_ip[client_ip].append(event)
        self.events_by_type[event_type].append(event)

        # Update attacker profile
        await self._update_attacker_profile(event)

        # Check thresholds and generate alerts
        await self._check_alert_thresholds(event)

        # Update metrics
        self.metrics["security_events_total"][event_type.value] += 1

        # Broadcast to WebSocket clients
        await self._broadcast_event(event)

        logger.debug(f"Security event logged: {event_type.value} from {client_ip}")

    async def log_auth_failure(
        self,
        client_ip: str,
        username: Optional[str] = None,
        reason: str = "Invalid credentials"
    ):
        """Log authentication failure"""

        await self.log_event(
            event_type=SecurityEventType.AUTH_FAILURE,
            client_ip=client_ip,
            severity=AlertSeverity.WARNING,
            details={
                "username": username,
                "reason": reason
            },
            user_id=username
        )

    async def log_rate_limit(
        self,
        client_ip: str,
        endpoint: str,
        limit: int,
        window: str
    ):
        """Log rate limit violation"""

        await self.log_event(
            event_type=SecurityEventType.RATE_LIMIT,
            client_ip=client_ip,
            severity=AlertSeverity.WARNING,
            details={
                "endpoint": endpoint,
                "limit": limit,
                "window": window
            },
            request_path=endpoint
        )

    async def log_waf_violation(
        self,
        client_ip: str,
        rule_id: str,
        reason: str,
        timestamp: datetime
    ):
        """Log WAF violation"""

        await self.log_event(
            event_type=SecurityEventType.WAF_VIOLATION,
            client_ip=client_ip,
            severity=AlertSeverity.HIGH,
            details={
                "reason": reason,
                "timestamp": timestamp.isoformat()
            },
            rule_id=rule_id
        )

    async def log_detailed_violation(self, log_entry: Dict[str, Any]):
        """Log detailed WAF violation with full context"""

        await self.log_event(
            event_type=SecurityEventType.WAF_VIOLATION,
            client_ip=log_entry.get("client_ip", "unknown"),
            severity=AlertSeverity[log_entry.get("severity", "HIGH")],
            details=log_entry,
            request_path=log_entry.get("url"),
            rule_id=log_entry.get("rule_id")
        )

    async def _update_attacker_profile(self, event: SecurityEvent):
        """Update or create attacker profile"""

        ip = event.client_ip

        if ip not in self.attacker_profiles:
            self.attacker_profiles[ip] = AttackerProfile(
                ip_address=ip,
                first_seen=event.timestamp,
                last_seen=event.timestamp
            )

        profile = self.attacker_profiles[ip]
        profile.last_seen = event.timestamp
        profile.total_events += 1
        profile.event_types[event.event_type.value] = \
            profile.event_types.get(event.event_type.value, 0) + 1

        # Calculate risk score
        profile.risk_score = self._calculate_risk_score(profile)

        # Auto-block high-risk IPs
        if profile.risk_score >= 80 and not profile.blocked:
            await self._block_ip(ip, "High risk score")

    def _calculate_risk_score(self, profile: AttackerProfile) -> float:
        """Calculate risk score for attacker profile"""

        score = 0.0

        # Factor in event types
        critical_events = [
            SecurityEventType.BRUTE_FORCE,
            SecurityEventType.DOS_ATTACK,
            SecurityEventType.DATA_EXFILTRATION,
            SecurityEventType.PRIVILEGE_ESCALATION
        ]

        for event_type in critical_events:
            count = profile.event_types.get(event_type.value, 0)
            if count > 0:
                score += min(count * 10, 30)

        # Factor in WAF violations
        waf_count = profile.event_types.get(SecurityEventType.WAF_VIOLATION.value, 0)
        score += min(waf_count * 5, 20)

        # Factor in auth failures
        auth_failures = profile.event_types.get(SecurityEventType.AUTH_FAILURE.value, 0)
        score += min(auth_failures * 3, 15)

        # Factor in rate limiting
        rate_limits = profile.event_types.get(SecurityEventType.RATE_LIMIT.value, 0)
        score += min(rate_limits * 2, 10)

        # Factor in total events
        if profile.total_events > 100:
            score += 10
        elif profile.total_events > 50:
            score += 5

        # Factor in time span (rapid attacks score higher)
        time_span = (profile.last_seen - profile.first_seen).total_seconds()
        if time_span > 0 and time_span < 60:  # All events within 1 minute
            score += 15

        return min(score, 100.0)

    async def _check_alert_thresholds(self, event: SecurityEvent):
        """Check if event thresholds trigger alerts"""

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.time_window_seconds)

        # Get recent events for this IP
        recent_events = [
            e for e in self.events_by_ip[event.client_ip]
            if e.timestamp >= window_start
        ]

        # Check auth failure threshold
        auth_failures = [
            e for e in recent_events
            if e.event_type == SecurityEventType.AUTH_FAILURE
        ]
        if len(auth_failures) >= self.alert_threshold_auth_failures:
            await self._create_alert(
                severity=AlertSeverity.HIGH,
                title=f"Brute Force Attack Detected",
                description=f"IP {event.client_ip} has {len(auth_failures)} auth failures in {self.time_window_seconds} seconds",
                events=auth_failures
            )

        # Check WAF violation threshold
        waf_violations = [
            e for e in recent_events
            if e.event_type == SecurityEventType.WAF_VIOLATION
        ]
        if len(waf_violations) >= self.alert_threshold_waf_violations:
            await self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Multiple WAF Violations",
                description=f"IP {event.client_ip} triggered {len(waf_violations)} WAF violations",
                events=waf_violations,
                action_required=True
            )

        # Check rate limit threshold
        rate_limits = [
            e for e in recent_events
            if e.event_type == SecurityEventType.RATE_LIMIT
        ]
        if len(rate_limits) >= self.alert_threshold_rate_limit:
            await self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Excessive Rate Limiting",
                description=f"IP {event.client_ip} hit rate limits {len(rate_limits)} times",
                events=rate_limits
            )

    async def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        events: List[SecurityEvent],
        action_required: bool = False
    ):
        """Create and send security alert"""

        self.alert_counter += 1
        alert_id = f"ALERT-{self.alert_counter:05d}"

        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            events=events,
            action_required=action_required
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update metrics
        self.metrics["active_alerts_count"] = len(self.active_alerts)

        # Send notifications
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            await self._send_alert_notifications(alert)

        logger.warning(f"Security alert created: {alert_id} - {title}")

        return alert

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via email and Slack"""

        if self.enable_email_alerts:
            await self._send_email_alert(alert)

        if self.enable_slack_alerts:
            await self._send_slack_alert(alert)

    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification"""

        if not self.smtp_config:
            logger.warning("Email alerts enabled but no SMTP configuration provided")
            return

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.smtp_config.get('from_address', 'security@hololoom.io')
            msg['To'] = self.smtp_config.get('to_address', 'admin@hololoom.io')

            # Create HTML body
            html = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <h2 style="color: {'#dc3545' if alert.severity == AlertSeverity.CRITICAL else '#ffc107'};">
                  Security Alert: {alert.title}
                </h2>
                <p><strong>Alert ID:</strong> {alert.alert_id}</p>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
                <p><strong>Description:</strong> {alert.description}</p>

                <h3>Event Details:</h3>
                <ul>
                  {"".join([f"<li>{e.timestamp.isoformat()} - {e.event_type.value} from {e.client_ip}</li>" for e in alert.events[:5]])}
                </ul>

                {'<p style="color: red;"><strong>ACTION REQUIRED</strong></p>' if alert.action_required else ''}

                <hr>
                <p style="font-size: 12px; color: #666;">
                  This is an automated security alert from hololoom Security Monitor.
                </p>
              </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))

            # Send email
            with smtplib.SMTP(
                self.smtp_config.get('host', 'localhost'),
                self.smtp_config.get('port', 587)
            ) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()
                if self.smtp_config.get('username'):
                    server.login(
                        self.smtp_config['username'],
                        self.smtp_config['password']
                    )
                server.send_message(msg)

            logger.info(f"Email alert sent for {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert notification"""

        if not self.slack_webhook_url:
            logger.warning("Slack alerts enabled but no webhook URL provided")
            return

        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.HIGH: "#ff6600",
                AlertSeverity.CRITICAL: "#ff0000"
            }

            # Create Slack message
            message = {
                "attachments": [{
                    "fallback": f"{alert.severity.value.upper()}: {alert.title}",
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f":warning: {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.alert_id,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Events",
                            "value": f"{len(alert.events)} security events",
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ],
                    "footer": "HoloLoom Security Monitor",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            if alert.action_required:
                message["attachments"][0]["fields"].append({
                    "title": ":rotating_light: Action Required",
                    "value": "Immediate investigation needed",
                    "short": False
                })

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=message
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.alert_id}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _block_ip(self, ip: str, reason: str):
        """Block an IP address"""

        self.blocked_ips.add(ip)
        if ip in self.attacker_profiles:
            self.attacker_profiles[ip].blocked = True

        self.metrics["blocked_ips_count"] = len(self.blocked_ips)

        # Create critical alert
        await self._create_alert(
            severity=AlertSeverity.CRITICAL,
            title=f"IP Address Blocked: {ip}",
            description=f"IP {ip} has been automatically blocked. Reason: {reason}",
            events=[],
            action_required=True
        )

        logger.warning(f"Blocked IP address: {ip} (Reason: {reason})")

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips

    def unblock_ip(self, ip: str):
        """Unblock an IP address"""

        self.blocked_ips.discard(ip)
        if ip in self.attacker_profiles:
            self.attacker_profiles[ip].blocked = False

        self.metrics["blocked_ips_count"] = len(self.blocked_ips)

        logger.info(f"Unblocked IP address: {ip}")

    async def _cleanup_old_events(self):
        """Background task to clean up old events"""

        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff = datetime.utcnow() - timedelta(days=7)

                # Clean up old events
                self.events = deque(
                    (e for e in self.events if e.timestamp > cutoff),
                    maxlen=10000
                )

                # Clean up attacker profiles
                for ip in list(self.attacker_profiles.keys()):
                    if self.attacker_profiles[ip].last_seen < cutoff:
                        del self.attacker_profiles[ip]

                # Clean up resolved alerts
                for alert_id in list(self.active_alerts.keys()):
                    alert = self.active_alerts[alert_id]
                    if alert.auto_resolved and alert.resolution_time and \
                       alert.resolution_time < cutoff:
                        del self.active_alerts[alert_id]

                logger.debug("Cleaned up old security events")

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _calculate_metrics(self):
        """Background task to calculate metrics"""

        while True:
            try:
                await asyncio.sleep(60)  # Calculate every minute

                now = datetime.utcnow()
                one_minute_ago = now - timedelta(seconds=60)

                # Calculate per-minute rates
                recent_events = [e for e in self.events if e.timestamp >= one_minute_ago]

                self.metrics["auth_failures_per_minute"] = len([
                    e for e in recent_events
                    if e.event_type == SecurityEventType.AUTH_FAILURE
                ])

                self.metrics["waf_violations_per_minute"] = len([
                    e for e in recent_events
                    if e.event_type == SecurityEventType.WAF_VIOLATION
                ])

                self.metrics["rate_limits_per_minute"] = len([
                    e for e in recent_events
                    if e.event_type == SecurityEventType.RATE_LIMIT
                ])

                # Calculate high-risk IPs
                self.metrics["high_risk_ips"] = len([
                    p for p in self.attacker_profiles.values()
                    if p.risk_score >= 70
                ])

                logger.debug("Metrics calculated")

            except Exception as e:
                logger.error(f"Error in metrics calculation: {e}")

    async def _detect_anomalies(self):
        """Background task for anomaly detection"""

        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Detect suspicious patterns
                await self._detect_suspicious_patterns()

                # Detect potential DoS attacks
                await self._detect_dos_attacks()

                # Detect data exfiltration attempts
                await self._detect_data_exfiltration()

            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")

    async def _detect_suspicious_patterns(self):
        """Detect suspicious activity patterns"""

        now = datetime.utcnow()
        five_minutes_ago = now - timedelta(seconds=300)

        for ip, events in self.events_by_ip.items():
            recent = [e for e in events if e.timestamp >= five_minutes_ago]

            if len(recent) < 10:
                continue

            # Check for scanning behavior (multiple 404s)
            not_found_count = sum(
                1 for e in recent
                if e.details.get('status_code') == 404
            )

            if not_found_count > 5:
                await self.log_event(
                    event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                    client_ip=ip,
                    severity=AlertSeverity.HIGH,
                    details={
                        "pattern": "scanning",
                        "404_count": not_found_count
                    }
                )

    async def _detect_dos_attacks(self):
        """Detect potential DoS attacks"""

        now = datetime.utcnow()
        one_minute_ago = now - timedelta(seconds=60)

        # Check request rate per IP
        for ip, events in self.events_by_ip.items():
            recent = [e for e in events if e.timestamp >= one_minute_ago]

            if len(recent) > 100:  # More than 100 requests per minute
                await self.log_event(
                    event_type=SecurityEventType.DOS_ATTACK,
                    client_ip=ip,
                    severity=AlertSeverity.CRITICAL,
                    details={
                        "request_count": len(recent),
                        "time_window": "1 minute"
                    }
                )

    async def _detect_data_exfiltration(self):
        """Detect potential data exfiltration attempts"""

        now = datetime.utcnow()
        ten_minutes_ago = now - timedelta(seconds=600)

        for ip, events in self.events_by_ip.items():
            recent = [e for e in events if e.timestamp >= ten_minutes_ago]

            # Calculate total response size
            total_bytes = sum(
                e.details.get('response_size', 0)
                for e in recent
            )

            # Alert on large data transfers
            if total_bytes > 100 * 1024 * 1024:  # 100MB
                await self.log_event(
                    event_type=SecurityEventType.DATA_EXFILTRATION,
                    client_ip=ip,
                    severity=AlertSeverity.CRITICAL,
                    details={
                        "total_bytes": total_bytes,
                        "request_count": len(recent),
                        "time_window": "10 minutes"
                    }
                )

    async def add_websocket_client(self, websocket: WebSocket):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.add(websocket)
        logger.debug(f"WebSocket client added. Total clients: {len(self.websocket_clients)}")

    async def remove_websocket_client(self, websocket: WebSocket):
        """Remove WebSocket client"""
        self.websocket_clients.discard(websocket)
        logger.debug(f"WebSocket client removed. Total clients: {len(self.websocket_clients)}")

    async def _broadcast_event(self, event: SecurityEvent):
        """Broadcast event to WebSocket clients"""

        if not self.websocket_clients:
            return

        message = {
            "type": "security_event",
            "data": {
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "timestamp": event.timestamp.isoformat(),
                "client_ip": event.client_ip,
                "details": event.details
            }
        }

        disconnected = set()

        for websocket in self.websocket_clients:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            await self.remove_websocket_client(websocket)

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""

        lines = []

        # Event counters
        lines.append("# HELP security_events_total Total security events by type")
        lines.append("# TYPE security_events_total counter")
        for event_type, count in self.metrics["security_events_total"].items():
            lines.append(f'security_events_total{{type="{event_type}"}} {count}')

        # Active alerts
        lines.append("# HELP active_alerts_count Number of active security alerts")
        lines.append("# TYPE active_alerts_count gauge")
        lines.append(f"active_alerts_count {self.metrics['active_alerts_count']}")

        # Blocked IPs
        lines.append("# HELP blocked_ips_count Number of blocked IP addresses")
        lines.append("# TYPE blocked_ips_count gauge")
        lines.append(f"blocked_ips_count {self.metrics['blocked_ips_count']}")

        # Per-minute rates
        lines.append("# HELP auth_failures_per_minute Authentication failures in the last minute")
        lines.append("# TYPE auth_failures_per_minute gauge")
        lines.append(f"auth_failures_per_minute {self.metrics['auth_failures_per_minute']}")

        lines.append("# HELP waf_violations_per_minute WAF violations in the last minute")
        lines.append("# TYPE waf_violations_per_minute gauge")
        lines.append(f"waf_violations_per_minute {self.metrics['waf_violations_per_minute']}")

        lines.append("# HELP rate_limits_per_minute Rate limit violations in the last minute")
        lines.append("# TYPE rate_limits_per_minute gauge")
        lines.append(f"rate_limits_per_minute {self.metrics['rate_limits_per_minute']}")

        # High-risk IPs
        lines.append("# HELP high_risk_ips Number of high-risk IP addresses")
        lines.append("# TYPE high_risk_ips gauge")
        lines.append(f"high_risk_ips {self.metrics['high_risk_ips']}")

        return "\n".join(lines)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard"""

        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        # Get recent events
        recent_events_hour = [e for e in self.events if e.timestamp >= one_hour_ago]
        recent_events_day = [e for e in self.events if e.timestamp >= one_day_ago]

        # Get top attackers
        top_attackers = sorted(
            self.attacker_profiles.values(),
            key=lambda x: x.risk_score,
            reverse=True
        )[:10]

        # Event type distribution
        event_distribution = defaultdict(int)
        for event in recent_events_day:
            event_distribution[event.event_type.value] += 1

        return {
            "summary": {
                "total_events_hour": len(recent_events_hour),
                "total_events_day": len(recent_events_day),
                "active_alerts": len(self.active_alerts),
                "blocked_ips": len(self.blocked_ips),
                "high_risk_ips": self.metrics["high_risk_ips"]
            },
            "metrics": self.metrics,
            "top_attackers": [
                {
                    "ip": p.ip_address,
                    "risk_score": p.risk_score,
                    "total_events": p.total_events,
                    "blocked": p.blocked,
                    "event_types": p.event_types
                }
                for p in top_attackers
            ],
            "event_distribution": dict(event_distribution),
            "recent_alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity.value,
                    "title": a.title,
                    "timestamp": a.timestamp.isoformat(),
                    "action_required": a.action_required
                }
                for a in list(self.alert_history)[-10:]
            ],
            "timeline": self._generate_timeline_data(recent_events_hour)
        }

    def _generate_timeline_data(self, events: List[SecurityEvent]) -> List[Dict]:
        """Generate timeline data for visualization"""

        # Group events by 5-minute intervals
        timeline = defaultdict(lambda: defaultdict(int))

        for event in events:
            # Round to 5-minute interval
            interval = event.timestamp.replace(
                minute=(event.timestamp.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            timeline[interval.isoformat()][event.event_type.value] += 1

        return [
            {
                "timestamp": ts,
                "events": dict(counts)
            }
            for ts, counts in sorted(timeline.items())
        ]

    def _generate_fingerprint(
        self,
        event_type: SecurityEventType,
        client_ip: str,
        rule_id: Optional[str]
    ) -> str:
        """Generate fingerprint for event deduplication"""

        data = f"{event_type.value}:{client_ip}:{rule_id or ''}"
        return hashlib.md5(data.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""

        return {
            "events": {
                "total": len(self.events),
                "by_type": {
                    event_type.value: len(events)
                    for event_type, events in self.events_by_type.items()
                },
                "unique_ips": len(self.events_by_ip)
            },
            "attackers": {
                "total_profiles": len(self.attacker_profiles),
                "blocked": len(self.blocked_ips),
                "high_risk": self.metrics["high_risk_ips"]
            },
            "alerts": {
                "active": len(self.active_alerts),
                "total_generated": self.alert_counter,
                "by_severity": {
                    severity.value: len([
                        a for a in self.active_alerts.values()
                        if a.severity == severity
                    ])
                    for severity in AlertSeverity
                }
            },
            "metrics": self.metrics
        }