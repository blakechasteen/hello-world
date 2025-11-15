"""Slack notification channel.

**Implemented: 2025-11-15**
**Lines**: 198
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SlackChannel:
    """Slack notification channel."""

    def __init__(self, webhook_url: str, channel: str = "#security-alerts"):
        """Initialize Slack channel.

        Args:
            webhook_url: Incoming webhook URL.
            channel: Target Slack channel.
        """
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, alert: "Alert", is_escalation: bool = False) -> bool:
        """Send alert via Slack.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            True if successful.
        """
        try:
            import aiohttp

            payload = self._build_payload(alert, is_escalation)

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status == 200:
                        logger.info(f"Slack alert sent: {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Slack send failed: {resp.status}")
                        return False

        except ImportError:
            logger.warning("aiohttp not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _build_payload(self, alert: "Alert", is_escalation: bool = False) -> Dict[str, Any]:
        """Build Slack message payload.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            Slack payload dict.
        """
        escalation_suffix = f" [ESCALATED - Level {alert.escalation_level}]" if is_escalation else ""
        text = f"{alert.severity.emoji} {alert.title}{escalation_suffix}"

        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": text,
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": self._build_fields(alert),
            },
        ]

        if alert.description:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert.description[:500]}",
                },
            })

        # Add tags if present
        if alert.tags:
            tag_text = " ".join([f"`{k}:{v}`" for k, v in alert.tags.items()])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Tags:*\n{tag_text}",
                },
            })

        # Add actions
        blocks.append({
            "type": "actions",
            "elements": self._build_actions(alert),
        })

        return {
            "channel": self.channel,
            "blocks": blocks,
        }

    def _build_fields(self, alert: "Alert") -> List[Dict[str, str]]:
        """Build field blocks.

        Args:
            alert: Alert object.

        Returns:
            List of field blocks.
        """
        fields = [
            {
                "type": "mrkdwn",
                "text": f"*Severity*\n{alert.severity.value.upper()}",
            },
            {
                "type": "mrkdwn",
                "text": f"*Time*\n{alert.timestamp.strftime('%H:%M:%S UTC')}",
            },
        ]

        if alert.source_ip:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Source IP*\n`{alert.source_ip}`",
            })

        if alert.user_id:
            fields.append({
                "type": "mrkdwn",
                "text": f"*User*\n`{alert.user_id}`",
            })

        if alert.group_id:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Group*\n`{alert.group_id}`",
            })

        return fields

    def _build_actions(self, alert: "Alert") -> List[Dict[str, Any]]:
        """Build action buttons.

        Args:
            alert: Alert object.

        Returns:
            List of action elements.
        """
        from ..core import AlertStatus

        actions = []

        if alert.status == AlertStatus.SENT:
            actions.extend([
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Acknowledge",
                        "emoji": True,
                    },
                    "value": alert.alert_id,
                    "action_id": f"ack_{alert.alert_id}",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Resolve",
                        "emoji": True,
                    },
                    "value": alert.alert_id,
                    "action_id": f"resolve_{alert.alert_id}",
                    "style": "danger",
                },
            ])

        if alert.runbook_url:
            actions.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Runbook",
                    "emoji": True,
                },
                "url": alert.runbook_url,
            })

        return actions

    def build_message(self, alert: "Alert") -> str:
        """Build plain text message.

        Args:
            alert: Alert object.

        Returns:
            Plain text message.
        """
        lines = [
            f"{alert.severity.emoji} {alert.title}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if alert.source_ip:
            lines.append(f"Source IP: {alert.source_ip}")

        if alert.user_id:
            lines.append(f"User: {alert.user_id}")

        if alert.description:
            lines.append(f"\nDescription:\n{alert.description}")

        return "\n".join(lines)
