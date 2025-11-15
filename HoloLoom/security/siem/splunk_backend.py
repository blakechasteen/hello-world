"""
Splunk SIEM Backend Integration

Integrates with Splunk HTTP Event Collector (HEC).

Created: 2025-11-15
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp

from .taxonomy import SecurityEvent


logger = logging.getLogger(__name__)


class SplunkBackend:
    """
    Splunk HTTP Event Collector (HEC) backend

    Uses Splunk's HEC API for event ingestion.

    Configuration:
        hec_url: Splunk HEC endpoint (e.g., "https://splunk.example.com:8088")
        hec_token: HEC authentication token
        index: Splunk index name (default: "hololoom_security")
        source: Event source (default: "hololoom")
        sourcetype: Event sourcetype (default: "_json")
        verify_ssl: Verify SSL certificates (default: True)
    """

    def __init__(
        self,
        hec_url: str,
        hec_token: str,
        index: str = "hololoom_security",
        source: str = "hololoom",
        sourcetype: str = "_json",
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ):
        self.hec_url = hec_url.rstrip('/') + '/services/collector/event'
        self.hec_token = hec_token
        self.index = index
        self.source = source
        self.sourcetype = sourcetype
        self.verify_ssl = verify_ssl
        self.timeout = timeout

        self.headers = {
            'Authorization': f'Splunk {hec_token}',
            'Content-Type': 'application/json',
        }

    async def send_events(self, events: List[SecurityEvent]) -> bool:
        """
        Send events to Splunk HEC

        Splunk HEC expects format:
        {
            "time": epoch_time,
            "host": hostname,
            "source": source,
            "sourcetype": sourcetype,
            "index": index,
            "event": {...}
        }

        Args:
            events: List of security events

        Returns:
            True if successful, False otherwise
        """
        if not events:
            return True

        # Convert to Splunk HEC format
        hec_events = []
        for event in events:
            hec_event = {
                "time": event.timestamp.timestamp(),
                "host": "hololoom",
                "source": self.source,
                "sourcetype": self.sourcetype,
                "index": self.index,
                "event": event.to_dict(),
            }
            hec_events.append(hec_event)

        # Send to Splunk (batch format - newline-separated JSON)
        payload = '\n'.join([self._json_dumps(e) for e in hec_events])

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.hec_url,
                    data=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=self.verify_ssl,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Sent {len(events)} events to Splunk")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Splunk HEC error ({response.status}): {error_text}")
                        return False

        except aiohttp.ClientError as e:
            logger.error(f"Splunk HEC connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Splunk HEC unexpected error: {e}")
            return False

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query events from Splunk

        Uses Splunk's search API. Note: Requires separate search credentials.

        Example SPL query:
            index=hololoom_security earliest=<start> latest=<end>
            | search category=ATTACK
            | table timestamp, category, subcategory, risk_score

        Args:
            start_time: Start of time range
            end_time: End of time range
            filters: Optional filters (category, severity, etc.)

        Returns:
            List of matching events
        """
        # Build SPL query
        spl_query = f'search index={self.index}'

        # Add time range
        spl_query += f' earliest="{start_time.isoformat()}" latest="{end_time.isoformat()}"'

        # Add filters
        if filters:
            for key, value in filters.items():
                spl_query += f' {key}="{value}"'

        logger.info(f"Splunk query: {spl_query}")

        # Note: Actual search API implementation would require additional
        # credentials and search endpoint. This is a placeholder.
        logger.warning("Splunk search API not implemented - use Splunk UI for queries")
        return []

    async def health_check(self) -> bool:
        """
        Check Splunk HEC health

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Send empty event to test connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.hec_url.replace('/event', '/health'),
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                    ssl=self.verify_ssl,
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Splunk health check failed: {e}")
            return False

    @staticmethod
    def _json_dumps(obj: Dict[str, Any]) -> str:
        """JSON dumps without extra whitespace"""
        import json
        return json.dumps(obj, separators=(',', ':'))


def create_splunk_backend(config: Dict[str, Any]) -> SplunkBackend:
    """
    Create Splunk backend from config dict

    Required config keys:
        - hec_url: Splunk HEC endpoint
        - hec_token: HEC authentication token

    Optional config keys:
        - index: Splunk index (default: "hololoom_security")
        - source: Event source (default: "hololoom")
        - sourcetype: Event sourcetype (default: "_json")
        - verify_ssl: Verify SSL (default: True)
        - timeout: Request timeout (default: 30.0)

    Example config:
        {
            "hec_url": "https://splunk.example.com:8088",
            "hec_token": "your-hec-token-here",
            "index": "security_events",
            "verify_ssl": False  # For testing only!
        }

    Args:
        config: Configuration dict

    Returns:
        SplunkBackend instance
    """
    return SplunkBackend(
        hec_url=config["hec_url"],
        hec_token=config["hec_token"],
        index=config.get("index", "hololoom_security"),
        source=config.get("source", "hololoom"),
        sourcetype=config.get("sourcetype", "_json"),
        verify_ssl=config.get("verify_ssl", True),
        timeout=config.get("timeout", 30.0),
    )
