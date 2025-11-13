"""
Time-Series Database Adapter - Phase 5

Provides abstraction layer for time-series storage.
Start with SQLite, easy to migrate to InfluxDB/Prometheus later.

Features:
- Batch writes (efficient)
- Time-range queries
- Aggregation (sum, avg, min, max, count)
- Tag-based filtering
- Automatic schema creation
"""

import sqlite3
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

try:
    from .metrics_collector import MetricEvent, MetricType
except ImportError:
    from metrics_collector import MetricEvent, MetricType

logger = logging.getLogger(__name__)


class AggregationFunction(Enum):
    """Supported aggregation functions."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


@dataclass
class Query:
    """
    Time-series query.

    Examples:
        # Get all query metrics from last hour
        Query(
            metric_type=MetricType.QUERY,
            start_time=time.time() - 3600,
            end_time=time.time()
        )

        # Get deep strategy metrics, average confidence
        Query(
            metric_type=MetricType.STRATEGY,
            tags={'strategy': 'deep'},
            aggregation=AggregationFunction.AVG,
            aggregation_field='avg_confidence'
        )
    """
    metric_type: MetricType
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: Dict[str, str] = None
    aggregation: Optional[AggregationFunction] = None
    aggregation_field: Optional[str] = None
    limit: int = 1000


class TimeSeriesDB:
    """
    Time-series database adapter.

    Uses SQLite for simplicity, can be swapped for InfluxDB later.

    Schema:
        events (
            id INTEGER PRIMARY KEY,
            type TEXT,
            timestamp REAL,
            tags_json TEXT,
            values_json TEXT
        )

    Usage:
        db = TimeSeriesDB('metrics.db')
        await db.initialize()

        # Write events
        await db.write_batch(events)

        # Query events
        results = await db.query(Query(
            metric_type=MetricType.QUERY,
            start_time=time.time() - 3600
        ))
    """

    def __init__(self, db_path: str = 'promptly_metrics.db'):
        """
        Initialize time-series database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    async def initialize(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Create events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tags_json TEXT,
                values_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for fast queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type_timestamp
            ON events(type, timestamp)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON events(timestamp)
        """)

        self.conn.commit()
        logger.info(f"Time-series DB initialized: {self.db_path}")

    async def write_batch(self, events: List[MetricEvent]):
        """
        Write batch of events to database.

        Args:
            events: List of metric events
        """
        if not self.conn:
            await self.initialize()

        try:
            cursor = self.conn.cursor()

            for event in events:
                cursor.execute("""
                    INSERT INTO events (type, timestamp, tags_json, values_json)
                    VALUES (?, ?, ?, ?)
                """, (
                    event.type.value,
                    event.timestamp,
                    json.dumps(event.tags),
                    json.dumps(event.values)
                ))

            self.conn.commit()
            logger.debug(f"Wrote {len(events)} events to database")

        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            self.conn.rollback()
            raise

    async def query(self, query: Query) -> List[Dict[str, Any]]:
        """
        Query events from database.

        Args:
            query: Query specification

        Returns:
            List of events matching query
        """
        if not self.conn:
            await self.initialize()

        # Build SQL query
        sql = "SELECT * FROM events WHERE type = ?"
        params = [query.metric_type.value]

        # Add time range
        if query.start_time is not None:
            sql += " AND timestamp >= ?"
            params.append(query.start_time)

        if query.end_time is not None:
            sql += " AND timestamp <= ?"
            params.append(query.end_time)

        # Add tag filters
        if query.tags:
            for key, value in query.tags.items():
                sql += " AND json_extract(tags_json, ?) = ?"
                params.extend([f'$.{key}', value])

        # Add ordering and limit
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(query.limit)

        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(sql, params)

        # Parse results
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'type': row['type'],
                'timestamp': row['timestamp'],
                'tags': json.loads(row['tags_json']) if row['tags_json'] else {},
                'values': json.loads(row['values_json']),
                'created_at': row['created_at']
            })

        return results

    async def aggregate(
        self,
        query: Query
    ) -> Optional[float]:
        """
        Run aggregation query.

        Args:
            query: Query with aggregation function

        Returns:
            Aggregated value
        """
        if not query.aggregation or not query.aggregation_field:
            raise ValueError("Aggregation query requires aggregation and aggregation_field")

        if not self.conn:
            await self.initialize()

        # Build SQL query
        agg_func = query.aggregation.value.upper()
        sql = f"""
            SELECT {agg_func}(CAST(json_extract(values_json, ?) AS REAL)) as result
            FROM events
            WHERE type = ?
        """
        params = [f'$.{query.aggregation_field}', query.metric_type.value]

        # Add time range
        if query.start_time is not None:
            sql += " AND timestamp >= ?"
            params.append(query.start_time)

        if query.end_time is not None:
            sql += " AND timestamp <= ?"
            params.append(query.end_time)

        # Add tag filters
        if query.tags:
            for key, value in query.tags.items():
                sql += " AND json_extract(tags_json, ?) = ?"
                params.extend([f'$.{key}', value])

        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchone()

        return result['result'] if result else None

    async def get_top_strategies(
        self,
        metric: str = 'confidence',
        limit: int = 10,
        start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top strategies by metric.

        Args:
            metric: Metric to sort by (confidence, latency_ms, etc.)
            limit: Number of results
            start_time: Optional time filter

        Returns:
            List of (strategy, avg_metric) tuples
        """
        if not self.conn:
            await self.initialize()

        sql = f"""
            SELECT
                json_extract(tags_json, '$.strategy') as strategy,
                AVG(CAST(json_extract(values_json, ?) AS REAL)) as avg_metric,
                COUNT(*) as count
            FROM events
            WHERE type = 'query'
        """
        params = [f'$.{metric}']

        if start_time is not None:
            sql += " AND timestamp >= ?"
            params.append(start_time)

        sql += """
            GROUP BY strategy
            ORDER BY avg_metric DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'strategy': row['strategy'],
                'avg_metric': row['avg_metric'],
                'count': row['count']
            })

        return results

    async def get_time_series(
        self,
        metric_type: MetricType,
        field: str,
        interval_seconds: int = 60,
        duration_seconds: int = 3600,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time-series data bucketed by interval.

        Args:
            metric_type: Type of metric
            field: Field to aggregate
            interval_seconds: Bucket size in seconds
            duration_seconds: How far back to look
            tags: Optional tag filters

        Returns:
            List of {timestamp, value} dicts
        """
        if not self.conn:
            await self.initialize()

        end_time = datetime.now().timestamp()
        start_time = end_time - duration_seconds

        sql = f"""
            SELECT
                CAST((timestamp / ?) AS INTEGER) * ? as bucket,
                AVG(CAST(json_extract(values_json, ?) AS REAL)) as avg_value
            FROM events
            WHERE type = ?
                AND timestamp >= ?
                AND timestamp <= ?
        """
        params = [
            interval_seconds,
            interval_seconds,
            f'$.{field}',
            metric_type.value,
            start_time,
            end_time
        ]

        # Add tag filters
        if tags:
            for key, value in tags.items():
                sql += " AND json_extract(tags_json, ?) = ?"
                params.extend([f'$.{key}', value])

        sql += """
            GROUP BY bucket
            ORDER BY bucket
        """

        cursor = self.conn.cursor()
        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'timestamp': row['bucket'],
                'value': row['avg_value']
            })

        return results

    async def get_cache_stats(
        self,
        duration_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Args:
            duration_seconds: Time window

        Returns:
            Cache statistics
        """
        start_time = datetime.now().timestamp() - duration_seconds

        # Get total hits and misses
        query = Query(
            metric_type=MetricType.QUERY,
            start_time=start_time
        )

        events = await self.query(query)

        total_hits = sum(
            1 for e in events
            if e['tags'].get('cache_hit') == 'True'
        )
        total_queries = len(events)
        total_misses = total_queries - total_hits

        hit_rate = total_hits / total_queries if total_queries > 0 else 0.0

        # Calculate average speedup
        cached_latencies = [
            e['values']['latency_ms'] for e in events
            if e['tags'].get('cache_hit') == 'True'
        ]
        uncached_latencies = [
            e['values']['latency_ms'] for e in events
            if e['tags'].get('cache_hit') == 'False'
        ]

        avg_cached = sum(cached_latencies) / len(cached_latencies) if cached_latencies else 0
        avg_uncached = sum(uncached_latencies) / len(uncached_latencies) if uncached_latencies else 0
        speedup = avg_uncached / avg_cached if avg_cached > 0 else 1.0

        return {
            'hit_rate': hit_rate,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_queries': total_queries,
            'avg_cached_latency_ms': avg_cached,
            'avg_uncached_latency_ms': avg_uncached,
            'speedup': speedup
        }

    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Time-series DB closed")
