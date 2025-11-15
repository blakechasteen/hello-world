"""
ELK (Elasticsearch) SIEM Backend Integration

Integrates with Elasticsearch for log storage and Kibana for visualization.

Created: 2025-11-15
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
import base64

from .taxonomy import SecurityEvent


logger = logging.getLogger(__name__)


class ELKBackend:
    """
    Elasticsearch backend for SIEM integration

    Uses Elasticsearch's Bulk API for efficient event ingestion.

    Configuration:
        es_url: Elasticsearch endpoint (e.g., "https://es.example.com:9200")
        username: Elasticsearch username (optional)
        password: Elasticsearch password (optional)
        api_key: Elasticsearch API key (alternative to username/password)
        index_pattern: Index pattern (default: "hololoom-security-{YYYY.MM.DD}")
        verify_ssl: Verify SSL certificates (default: True)
    """

    def __init__(
        self,
        es_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        index_pattern: str = "hololoom-security-{date}",
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ):
        self.es_url = es_url.rstrip('/')
        self.index_pattern = index_pattern
        self.verify_ssl = verify_ssl
        self.timeout = timeout

        # Authentication
        self.headers = {'Content-Type': 'application/x-ndjson'}
        if api_key:
            self.headers['Authorization'] = f'ApiKey {api_key}'
        elif username and password:
            auth = base64.b64encode(f'{username}:{password}'.encode()).decode()
            self.headers['Authorization'] = f'Basic {auth}'

    def _get_index_name(self, timestamp: datetime) -> str:
        """Get index name with date formatting"""
        date_str = timestamp.strftime("%Y.%m.%d")
        return self.index_pattern.replace('{date}', date_str)

    async def send_events(self, events: List[SecurityEvent]) -> bool:
        """
        Send events to Elasticsearch using Bulk API

        Elasticsearch Bulk API format:
            {"index": {"_index": "my-index"}}
            {"field1": "value1"}
            {"index": {"_index": "my-index"}}
            {"field2": "value2"}

        Args:
            events: List of security events

        Returns:
            True if successful, False otherwise
        """
        if not events:
            return True

        # Build bulk request (newline-delimited JSON)
        bulk_lines = []
        for event in events:
            # Index metadata
            index_name = self._get_index_name(event.timestamp)
            index_meta = {"index": {"_index": index_name}}
            bulk_lines.append(self._json_dumps(index_meta))

            # Event data
            event_data = event.to_dict()
            # Add @timestamp for Elasticsearch time-series
            event_data["@timestamp"] = event.timestamp.isoformat()
            bulk_lines.append(self._json_dumps(event_data))

        # Add final newline
        payload = '\n'.join(bulk_lines) + '\n'

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.es_url}/_bulk',
                    data=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=self.verify_ssl,
                ) as response:
                    if response.status in (200, 201):
                        result = await response.json()
                        if result.get('errors'):
                            logger.warning(f"Elasticsearch bulk indexing had errors: {result}")
                        else:
                            logger.info(f"Sent {len(events)} events to Elasticsearch")
                        return not result.get('errors', False)
                    else:
                        error_text = await response.text()
                        logger.error(f"Elasticsearch error ({response.status}): {error_text}")
                        return False

        except aiohttp.ClientError as e:
            logger.error(f"Elasticsearch connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Elasticsearch unexpected error: {e}")
            return False

    async def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query events from Elasticsearch

        Uses Elasticsearch Query DSL.

        Args:
            start_time: Start of time range
            end_time: End of time range
            filters: Optional filters (category, severity, etc.)

        Returns:
            List of matching events
        """
        # Build query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat(),
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"timestamp": "desc"}],
            "size": 1000,  # Max results
        }

        # Add filters
        if filters:
            for key, value in filters.items():
                query["query"]["bool"]["must"].append({"term": {key: value}})

        try:
            # Search across all security indices
            search_url = f'{self.es_url}/hololoom-security-*/_search'

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    search_url,
                    json=query,
                    headers={'Content-Type': 'application/json', **self.headers},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=self.verify_ssl,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        hits = result.get('hits', {}).get('hits', [])
                        events = [hit['_source'] for hit in hits]
                        logger.info(f"Retrieved {len(events)} events from Elasticsearch")
                        return events
                    else:
                        error_text = await response.text()
                        logger.error(f"Elasticsearch query error ({response.status}): {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Elasticsearch query failed: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Check Elasticsearch cluster health

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{self.es_url}/_cluster/health',
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                    ssl=self.verify_ssl,
                ) as response:
                    if response.status == 200:
                        health = await response.json()
                        status = health.get('status', 'red')
                        # Green or yellow is acceptable
                        return status in ('green', 'yellow')
                    return False
        except Exception as e:
            logger.debug(f"Elasticsearch health check failed: {e}")
            return False

    @staticmethod
    def _json_dumps(obj: Dict[str, Any]) -> str:
        """JSON dumps without extra whitespace"""
        import json
        return json.dumps(obj, separators=(',', ':'))


def create_elk_backend(config: Dict[str, Any]) -> ELKBackend:
    """
    Create ELK backend from config dict

    Required config keys:
        - es_url: Elasticsearch endpoint

    Optional config keys:
        - username: Elasticsearch username
        - password: Elasticsearch password
        - api_key: Elasticsearch API key (alternative to username/password)
        - index_pattern: Index pattern (default: "hololoom-security-{date}")
        - verify_ssl: Verify SSL (default: True)
        - timeout: Request timeout (default: 30.0)

    Example config:
        {
            "es_url": "https://elasticsearch.example.com:9200",
            "username": "elastic",
            "password": "your-password",
            "index_pattern": "security-{date}"
        }

    Args:
        config: Configuration dict

    Returns:
        ELKBackend instance
    """
    return ELKBackend(
        es_url=config["es_url"],
        username=config.get("username"),
        password=config.get("password"),
        api_key=config.get("api_key"),
        index_pattern=config.get("index_pattern", "hololoom-security-{date}"),
        verify_ssl=config.get("verify_ssl", True),
        timeout=config.get("timeout", 30.0),
    )
