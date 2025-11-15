"""
Datadog SIEM Backend Integration

Integrates with Datadog Logs API for security event logging.

Created: 2025-11-15
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp

from .taxonomy import SecurityEvent


logger = logging.getLogger(__name__)


class DatadogBackend:
    """
    Datadog Logs API backend for SIEM integration

    Uses Datadog's HTTP Logs API for event ingestion.

    Configuration:
        api_key: Datadog API key
        site: Datadog site (e.g., "datadoghq.com", "datadoghq.eu")
        service: Service name (default: "hololoom")
        source: Log source (default: "security")
        tags: Additional tags (default: ["env:production"])
    """

    def __init__(
        self,
        api_key: str,
        site: str = "datadoghq.com",
        service: str = "hololoom",
        source: str = "security",
        tags: Optional[List[str]] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.site = site
        self.service = service
        self.source = source
        self.tags = tags or ["env:production"]
        self.timeout = timeout

        # Datadog Logs API endpoint
        self.logs_url = f"https://http-intake.logs.{site}/api/v2/logs"

        self.headers = {
            'DD-API-KEY': api_key,
            'Content-Type': 'application/json',
        }

    async def send_events(self, events: List[SecurityEvent]) -> bool:
        """
        Send events to Datadog Logs API

        Datadog expects format:
        [
            {
                "ddsource": "source",
                "ddtags": "tag1:value1,tag2:value2",
                "hostname": "hostname",
                "message": "message text",
                "service": "service-name",
                "timestamp": epoch_ms,
                "custom_field": "value"
            }
        ]

        Args:
            events: List of security events

        Returns:
            True if successful, False otherwise
        """
        if not events:
            return True

        # Convert to Datadog format
        dd_logs = []
        for event in events:
            # Build tags
            tags = [
                f"category:{event.category.value}",
                f"subcategory:{event.subcategory.value}",
                f"severity:{event.level.value}",
                f"blocked:{event.blocked}",
                *self.tags,
            ]

            dd_log = {
                "ddsource": self.source,
                "ddtags": ",".join(tags),
                "hostname": "hololoom",
                "service": self.service,
                "timestamp": int(event.timestamp.timestamp() * 1000),  # Epoch milliseconds
                "message": f"{event.category.value}/{event.subcategory.value}: {event.action}",
                # Include full event data
                **event.to_dict(),
            }
            dd_logs.append(dd_log)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.logs_url,
                    json=dd_logs,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status in (200, 202):
                        logger.info(f"Sent {len(events)} events to Datadog")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Datadog Logs API error ({response.status}): {error_text}")
                        return False

        except aiohttp.ClientError as e:
            logger.error(f"Datadog connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Datadog unexpected error: {e}")
            return False

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query events from Datadog

        Uses Datadog's Log Search API.

        Args:
            start_time: Start of time range
            end_time: End of time range
            filters: Optional filters (category, severity, etc.)

        Returns:
            List of matching events
        """
        # Build query string
        query_parts = [f"service:{self.service}", f"source:{self.source}"]

        # Add filters
        if filters:
            for key, value in filters.items():
                query_parts.append(f"{key}:{value}")

        query_string = " AND ".join(query_parts)

        # Build request
        search_url = f"https://api.{self.site}/api/v2/logs/events/search"
        payload = {
            "filter": {
                "query": query_string,
                "from": start_time.isoformat(),
                "to": end_time.isoformat(),
            },
            "page": {
                "limit": 1000,  # Max results
            },
            "sort": "timestamp",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    search_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logs = result.get('data', [])
                        events = [log.get('attributes', {}) for log in logs]
                        logger.info(f"Retrieved {len(events)} events from Datadog")
                        return events
                    else:
                        error_text = await response.text()
                        logger.error(f"Datadog query error ({response.status}): {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Datadog query failed: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Check Datadog API health

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple API validation
            validation_url = f"https://api.{self.site}/api/v1/validate"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    validation_url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('valid', False)
                    return False
        except Exception as e:
            logger.debug(f"Datadog health check failed: {e}")
            return False


def create_datadog_backend(config: Dict[str, Any]) -> DatadogBackend:
    """
    Create Datadog backend from config dict

    Required config keys:
        - api_key: Datadog API key

    Optional config keys:
        - site: Datadog site (default: "datadoghq.com")
        - service: Service name (default: "hololoom")
        - source: Log source (default: "security")
        - tags: Additional tags (default: ["env:production"])
        - timeout: Request timeout (default: 30.0)

    Example config:
        {
            "api_key": "your-datadog-api-key",
            "site": "datadoghq.com",
            "service": "hololoom-prod",
            "tags": ["env:production", "team:security"]
        }

    Args:
        config: Configuration dict

    Returns:
        DatadogBackend instance
    """
    return DatadogBackend(
        api_key=config["api_key"],
        site=config.get("site", "datadoghq.com"),
        service=config.get("service", "hololoom"),
        source=config.get("source", "security"),
        tags=config.get("tags", ["env:production"]),
        timeout=config.get("timeout", 30.0),
    )
