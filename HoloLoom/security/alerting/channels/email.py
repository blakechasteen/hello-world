"""Email notification channel.

**Implemented: 2025-11-15**
**Lines**: 215
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class EmailChannel:
    """Email notification channel."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ):
        """Initialize email channel.

        Args:
            smtp_host: SMTP server host.
            smtp_port: SMTP server port.
            sender: Sender email address.
            username: SMTP username.
            password: SMTP password.
            use_tls: Whether to use TLS.
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.username = username
        self.password = password
        self.use_tls = use_tls

    async def send(
        self,
        alert: "Alert",
        recipients: Optional[List[str]] = None,
        is_escalation: bool = False,
    ) -> bool:
        """Send alert via email.

        Args:
            alert: Alert object.
            recipients: List of recipient emails.
            is_escalation: Whether this is an escalation.

        Returns:
            True if successful.
        """
        if not recipients:
            return False

        try:
            import aiosmtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            # Build message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._build_subject(alert, is_escalation)
            msg["From"] = self.sender
            msg["To"] = ", ".join(recipients)

            # Build HTML body
            html = self._build_html_body(alert, is_escalation)
            msg.attach(MIMEText(html, "html"))

            # Send email
            async with aiosmtplib.SMTP(hostname=self.smtp_host, port=self.smtp_port) as smtp:
                if self.use_tls:
                    await smtp.starttls()

                if self.username and self.password:
                    await smtp.login(self.username, self.password)

                await smtp.send_message(msg)

            logger.info(f"Email alert sent to {recipients}: {alert.alert_id}")
            return True

        except ImportError:
            logger.warning("aiosmtplib not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _build_subject(self, alert: "Alert", is_escalation: bool = False) -> str:
        """Build email subject.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            Email subject.
        """
        escalation = " [ESCALATED]" if is_escalation else ""
        return f"[{alert.severity.value.upper()}] {alert.title}{escalation}"

    def _build_html_body(self, alert: "Alert", is_escalation: bool = False) -> str:
        """Build HTML email body.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            HTML body.
        """
        escalation_notice = ""
        if is_escalation:
            escalation_notice = f"""
            <div style="background-color: #fff3cd; padding: 10px; margin-bottom: 20px; border-left: 4px solid #ff9800;">
                <strong>⚠️ This alert has been ESCALATED</strong> (Level {alert.escalation_level})
            </div>
            """

        tags_html = ""
        if alert.tags:
            tags_html = "<tr><td><strong>Tags</strong></td><td>"
            tags_html += ", ".join([f"<code>{k}:{v}</code>" for k, v in alert.tags.items()])
            tags_html += "</td></tr>"

        runbook_html = ""
        if alert.runbook_url:
            runbook_html = f'<tr><td><strong>Runbook</strong></td><td><a href="{alert.runbook_url}">{alert.runbook_url}</a></td></tr>'

        return f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2 style="color: {self._get_severity_color(alert.severity.value)};">
                    {alert.severity.emoji} {alert.title}
                </h2>

                {escalation_notice}

                <table style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold; width: 20%;">Severity</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{alert.severity.value.upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Time</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                    </tr>
                    {f'<tr style="background-color: #f5f5f5;"><td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Source IP</td><td style="padding: 10px; border: 1px solid #ddd;"><code>{alert.source_ip}</code></td></tr>' if alert.source_ip else ''}
                    {f'<tr><td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">User</td><td style="padding: 10px; border: 1px solid #ddd;"><code>{alert.user_id}</code></td></tr>' if alert.user_id else ''}
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Alert ID</td>
                        <td style="padding: 10px; border: 1px solid #ddd;"><code>{alert.alert_id}</code></td>
                    </tr>
                    {tags_html}
                    {runbook_html}
                </table>

                {f'<h3 style="margin-top: 20px;">Description</h3><p style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #2196F3;">{alert.description}</p>' if alert.description else ''}

                <div style="margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px;">
                    <p style="margin: 0; font-size: 12px; color: #666;">
                        This is an automated security alert from HoloLoom.
                        Please review immediately and take appropriate action.
                    </p>
                </div>
            </body>
        </html>
        """

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level.

        Args:
            severity: Severity level string.

        Returns:
            Hex color code.
        """
        colors = {
            "info": "#2196F3",
            "warning": "#FF9800",
            "critical": "#F44336",
        }
        return colors.get(severity, "#333333")

    async def send_batch(
        self,
        alerts: List["Alert"],
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Send multiple alerts in one email.

        Args:
            alerts: List of alert objects.
            recipients: List of recipient emails.

        Returns:
            Dict mapping alert_id to send status.
        """
        results = {}
        for alert in alerts:
            success = await self.send(alert, recipients=recipients)
            results[alert.alert_id] = success

        return results
