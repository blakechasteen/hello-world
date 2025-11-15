"""PagerDuty notification channel.

**Implemented: 2025-11-15**
**Lines**: 187
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class PagerDutyChannel:
    """PagerDuty notification channel."""

    def __init__(self, api_key: str, service_id: str):
        """Initialize PagerDuty channel.

        Args:
            api_key: PagerDuty API key.
            service_id: PagerDuty service ID.
        """
        self.api_key = api_key
        self.service_id = service_id
        self.base_url = "https://events.pagerduty.com/v2"

    async def send(self, alert: "Alert", is_escalation: bool = False) -> bool:
        """Send alert via PagerDuty.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            True if successful.
        """
        try:
            import aiohttp

            payload = self._build_payload(alert, is_escalation)
            headers = self._build_headers()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/enqueue",
                    json=payload,
                    headers=headers
                ) as resp:
                    if resp.status == 202:
                        logger.info(f"PagerDuty event sent: {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"PagerDuty send failed: {resp.status}")
                        return False

        except ImportError:
            logger.warning("aiohttp not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers.

        Returns:
            Headers dict.
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token token={self.api_key}",
        }

    def _build_payload(self, alert: "Alert", is_escalation: bool = False) -> Dict[str, Any]:
        """Build PagerDuty event payload.

        Args:
            alert: Alert object.
            is_escalation: Whether this is an escalation.

        Returns:
            PagerDuty event payload.
        """
        severity_map = {
            "info": "info",
            "warning": "warning",
            "critical": "critical",
        }

        custom_details = {
            "description": alert.description,
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
        }

        if alert.source_ip:
            custom_details["source_ip"] = alert.source_ip

        if alert.user_id:
            custom_details["user_id"] = alert.user_id

        if alert.tags:
            custom_details["tags"] = alert.tags

        if is_escalation:
            custom_details["escalation_level"] = alert.escalation_level

        payload = {
            "routing_key": self.service_id,
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": alert.title,
                "severity": severity_map.get(alert.severity.value, "critical"),
                "source": alert.source_ip or "HoloLoom Security",
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": custom_details,
            },
        }

        # Add links
        if alert.runbook_url:
            payload["links"] = [
                {
                    "href": alert.runbook_url,
                    "text": "Runbook",
                }
            ]

        return payload

    async def trigger(
        self,
        summary: str,
        severity: str = "critical",
        source: str = "HoloLoom",
        details: Optional[Dict[str, Any]] = None,
        dedup_key: Optional[str] = None,
    ) -> bool:
        """Trigger a PagerDuty incident.

        Args:
            summary: Incident summary.
            severity: Incident severity.
            source: Incident source.
            details: Custom details dict.
            dedup_key: Deduplication key.

        Returns:
            True if successful.
        """
        try:
            import aiohttp

            payload = {
                "routing_key": self.service_id,
                "event_action": "trigger",
                "payload": {
                    "summary": summary,
                    "severity": severity,
                    "source": source,
                    "custom_details": details or {},
                },
            }

            if dedup_key:
                payload["dedup_key"] = dedup_key

            headers = self._build_headers()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/enqueue",
                    json=payload,
                    headers=headers
                ) as resp:
                    return resp.status == 202

        except Exception as e:
            logger.error(f"PagerDuty trigger failed: {e}")
            return False

    async def resolve(self, dedup_key: str) -> bool:
        """Resolve a PagerDuty incident.

        Args:
            dedup_key: Incident dedup key.

        Returns:
            True if successful.
        """
        try:
            import aiohttp

            payload = {
                "routing_key": self.service_id,
                "event_action": "resolve",
                "dedup_key": dedup_key,
            }

            headers = self._build_headers()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/enqueue",
                    json=payload,
                    headers=headers
                ) as resp:
                    return resp.status == 202

        except Exception as e:
            logger.error(f"PagerDuty resolve failed: {e}")
            return False
