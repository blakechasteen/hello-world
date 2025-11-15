"""SMS notification channel (Twilio).

**Implemented: 2025-11-15**
**Lines**: 168
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SMSChannel:
    """SMS notification channel using Twilio."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
    ):
        """Initialize SMS channel.

        Args:
            account_sid: Twilio account SID.
            auth_token: Twilio auth token.
            from_number: Twilio phone number to send from.
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.client = None

    def _get_client(self):
        """Get or create Twilio client.

        Returns:
            Twilio client.
        """
        if self.client is None:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
            except ImportError:
                logger.warning("twilio not installed")
                return None

        return self.client

    async def send(
        self,
        alert: "Alert",
        recipients: Optional[List[str]] = None,
        is_escalation: bool = False,
    ) -> bool:
        """Send alert via SMS.

        Args:
            alert: Alert object.
            recipients: List of phone numbers.
            is_escalation: Whether this is an escalation.

        Returns:
            True if successful.
        """
        if not recipients:
            logger.warning("No SMS recipients provided")
            return False

        client = self._get_client()
        if not client:
            return False

        try:
            message_body = self._build_message(alert, is_escalation)

            results = []
            for phone in recipients:
                try:
                    msg = client.messages.create(
                        body=message_body,
                        from_=self.from_number,
                        to=phone
                    )
                    results.append(True)
                    logger.info(f"SMS sent to {phone}: {msg.sid}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {phone}: {e}")
                    results.append(False)

            return all(results)

        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False

    async def send_batch(
        self,
        alerts: List["Alert"],
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Send alerts via SMS (one per recipient).

        Args:
            alerts: List of alert objects.
            recipients: List of phone numbers.

        Returns:
            Dict mapping alert_id to send status.
        """
        results = {}
        for alert in alerts:
            success = await self.send(alert, recipients=recipients)
            results[alert.alert_id] = success

        return results

    def _build_message(self, alert: "Alert", is_escalation: bool = False) -> str:
        """Build SMS message.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            SMS message (max 160 chars).
        """
        escalation = " [ESC]" if is_escalation else ""
        severity_prefix = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }.get(alert.severity.value, "●")

        # Build short message
        message = f"{severity_prefix} {alert.title[:50]}{escalation}"

        if alert.source_ip:
            message += f" from {alert.source_ip.split('.')[-1]}"

        # Ensure under SMS limit
        if len(message) > 160:
            message = message[:157] + "..."

        return message

    async def send_critical_alert(
        self,
        title: str,
        source_ip: Optional[str],
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """Send critical alert via SMS.

        Args:
            title: Alert title.
            source_ip: Source IP address.
            recipients: List of phone numbers.

        Returns:
            True if successful.
        """
        client = self._get_client()
        if not client or not recipients:
            return False

        try:
            message = f"🚨 {title}"
            if source_ip:
                message += f" from {source_ip}"

            if len(message) > 160:
                message = message[:157] + "..."

            results = []
            for phone in recipients:
                try:
                    msg = client.messages.create(
                        body=message,
                        from_=self.from_number,
                        to=phone
                    )
                    results.append(True)
                except Exception as e:
                    logger.error(f"Failed to send critical SMS: {e}")
                    results.append(False)

            return all(results)

        except Exception as e:
            logger.error(f"Failed to send critical SMS alert: {e}")
            return False
