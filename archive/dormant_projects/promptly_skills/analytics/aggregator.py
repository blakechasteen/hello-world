"""
Metrics Aggregator - Phase 5

Pre-computes aggregated statistics for fast dashboard queries.

Features:
- Pre-aggregate on write (faster reads)
- Multiple time windows (1h, 24h, 7d, 30d)
- Rolling statistics
- Efficient caching
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging

try:
    from .time_series_db import TimeSeriesDB, Query, AggregationFunction
    from .metrics_collector import MetricType
except ImportError:
    from time_series_db import TimeSeriesDB, Query, AggregationFunction
    from metrics_collector import MetricType

logger = logging.getLogger(__name__)


class AggregationPeriod(Enum):
    """Time periods for aggregation."""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"


@dataclass
class AggregatedStats:
    """Aggregated statistics for a time period."""
    period: AggregationPeriod
    start_time: float
    end_time: float

    # Query stats
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    median_confidence: float = 0.0

    # Strategy stats
    strategy_distribution: Dict[str, int] = None
    strategy_performance: Dict[str, Dict[str, float]] = None

    # Cache stats
    cache_hit_rate: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0

    # Learning stats
    exploration_rate: float = 0.0
    optimal_strategy_rate: float = 0.0

    def __post_init__(self):
        if self.strategy_distribution is None:
            self.strategy_distribution = {}
        if self.strategy_performance is None:
            self.strategy_performance = {}


class MetricsAggregator:
    """
    Aggregates metrics for fast dashboard queries.

    Usage:
        aggregator = MetricsAggregator(db)
        await aggregator.compute_all()

        # Get aggregated stats
        stats = await aggregator.get_stats(AggregationPeriod.DAY)
    """

    def __init__(self, db: TimeSeriesDB):
        """
        Initialize aggregator.

        Args:
            db: Time-series database
        """
        self.db = db
        self._cache: Dict[AggregationPeriod, AggregatedStats] = {}
        self._cache_ttl = 60  # Cache for 60 seconds

    async def compute_all(self):
        """Compute aggregations for all periods."""
        for period in AggregationPeriod:
            await self.compute(period)

    async def compute(self, period: AggregationPeriod) -> AggregatedStats:
        """
        Compute aggregations for a time period.

        Args:
            period: Time period to aggregate

        Returns:
            Aggregated statistics
        """
        # Calculate time range
        end_time = time.time()
        duration = self._get_duration_seconds(period)
        start_time = end_time - duration

        # Get query events
        query = Query(
            metric_type=MetricType.QUERY,
            start_time=start_time,
            end_time=end_time,
            limit=100000  # Get all events
        )

        events = await self.db.query(query)

        if not events:
            logger.warning(f"No events found for period {period.value}")
            return AggregatedStats(
                period=period,
                start_time=start_time,
                end_time=end_time
            )

        # Compute statistics
        stats = AggregatedStats(
            period=period,
            start_time=start_time,
            end_time=end_time
        )

        # Total queries
        stats.total_queries = len(events)

        # Latency statistics
        latencies = [e['values']['latency_ms'] for e in events]
        latencies.sort()

        stats.avg_latency_ms = sum(latencies) / len(latencies)
        stats.p50_latency_ms = latencies[len(latencies) // 2]
        stats.p95_latency_ms = latencies[int(len(latencies) * 0.95)]
        stats.p99_latency_ms = latencies[int(len(latencies) * 0.99)]

        # Confidence statistics
        confidences = [e['values']['confidence'] for e in events]
        confidences.sort()

        stats.avg_confidence = sum(confidences) / len(confidences)
        stats.median_confidence = confidences[len(confidences) // 2]

        # Strategy distribution
        strategy_counts: Dict[str, int] = {}
        for event in events:
            strategy = event['tags'].get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        stats.strategy_distribution = strategy_counts

        # Strategy performance
        strategy_metrics: Dict[str, List[Dict[str, float]]] = {}
        for event in events:
            strategy = event['tags'].get('strategy', 'unknown')
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = []

            strategy_metrics[strategy].append({
                'latency_ms': event['values']['latency_ms'],
                'confidence': event['values']['confidence']
            })

        stats.strategy_performance = {}
        for strategy, metrics in strategy_metrics.items():
            stats.strategy_performance[strategy] = {
                'avg_latency_ms': sum(m['latency_ms'] for m in metrics) / len(metrics),
                'avg_confidence': sum(m['confidence'] for m in metrics) / len(metrics),
                'total_uses': len(metrics),
                'success_rate': sum(1 for m in metrics if m['confidence'] >= 0.75) / len(metrics)
            }

        # Cache statistics
        cache_hits = sum(
            1 for e in events
            if e['tags'].get('cache_hit') == 'True'
        )
        stats.total_cache_hits = cache_hits
        stats.total_cache_misses = stats.total_queries - cache_hits
        stats.cache_hit_rate = cache_hits / stats.total_queries if stats.total_queries > 0 else 0.0

        # Cache in memory
        self._cache[period] = stats

        logger.info(f"Computed aggregations for {period.value}: {stats.total_queries} queries")

        return stats

    async def get_stats(
        self,
        period: AggregationPeriod,
        force_refresh: bool = False
    ) -> AggregatedStats:
        """
        Get aggregated statistics (cached).

        Args:
            period: Time period
            force_refresh: Force recomputation

        Returns:
            Aggregated statistics
        """
        # Check cache
        if not force_refresh and period in self._cache:
            cached = self._cache[period]
            age = time.time() - cached.end_time

            if age < self._cache_ttl:
                return cached

        # Compute fresh stats
        return await self.compute(period)

    async def get_top_strategies(
        self,
        period: AggregationPeriod = AggregationPeriod.DAY,
        metric: str = 'avg_confidence',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top strategies by metric.

        Args:
            period: Time period
            metric: Metric to sort by
            limit: Number of results

        Returns:
            List of top strategies
        """
        stats = await self.get_stats(period)

        # Sort strategies by metric
        sorted_strategies = sorted(
            stats.strategy_performance.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )[:limit]

        results = []
        for strategy, performance in sorted_strategies:
            results.append({
                'strategy': strategy,
                'performance': performance
            })

        return results

    async def get_trend(
        self,
        metric: str,
        period: AggregationPeriod = AggregationPeriod.DAY,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Get time-series trend for a metric.

        Args:
            metric: Metric name (latency_ms, confidence, etc.)
            period: Time period
            interval_minutes: Bucket size in minutes

        Returns:
            List of {timestamp, value} dicts
        """
        duration = self._get_duration_seconds(period)
        interval_seconds = interval_minutes * 60

        return await self.db.get_time_series(
            metric_type=MetricType.QUERY,
            field=metric,
            interval_seconds=interval_seconds,
            duration_seconds=duration
        )

    async def get_cache_trend(
        self,
        period: AggregationPeriod = AggregationPeriod.DAY,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Get cache hit rate trend.

        Args:
            period: Time period
            interval_minutes: Bucket size

        Returns:
            List of {timestamp, hit_rate} dicts
        """
        duration = self._get_duration_seconds(period)
        interval_seconds = interval_minutes * 60

        # Get all queries bucketed by interval
        end_time = time.time()
        start_time = end_time - duration

        query = Query(
            metric_type=MetricType.QUERY,
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )

        events = await self.db.query(query)

        # Bucket by interval
        buckets: Dict[int, Dict[str, int]] = {}
        for event in events:
            bucket = int(event['timestamp'] // interval_seconds) * interval_seconds

            if bucket not in buckets:
                buckets[bucket] = {'hits': 0, 'total': 0}

            buckets[bucket]['total'] += 1
            if event['tags'].get('cache_hit') == 'True':
                buckets[bucket]['hits'] += 1

        # Calculate hit rate for each bucket
        results = []
        for bucket, counts in sorted(buckets.items()):
            hit_rate = counts['hits'] / counts['total'] if counts['total'] > 0 else 0.0
            results.append({
                'timestamp': bucket,
                'hit_rate': hit_rate,
                'hits': counts['hits'],
                'total': counts['total']
            })

        return results

    def _get_duration_seconds(self, period: AggregationPeriod) -> int:
        """Get duration in seconds for a period."""
        durations = {
            AggregationPeriod.HOUR: 3600,
            AggregationPeriod.DAY: 86400,
            AggregationPeriod.WEEK: 604800,
            AggregationPeriod.MONTH: 2592000
        }
        return durations[period]

    def clear_cache(self):
        """Clear cached aggregations."""
        self._cache.clear()
        logger.info("Aggregation cache cleared")
