"""
Promptly Analytics - Phase 5: Learning & Analytics

This module provides comprehensive analytics and learning capabilities:
- Metrics collection and time-series storage
- Performance dashboard
- A/B testing framework
- Advanced learning algorithms (contextual bandits, neural bandits)
"""

from .metrics_collector import MetricsCollector, MetricEvent, MetricType
from .time_series_db import TimeSeriesDB, Query as DBQuery
from .aggregator import MetricsAggregator, AggregationPeriod

__all__ = [
    'MetricsCollector',
    'MetricEvent',
    'MetricType',
    'TimeSeriesDB',
    'DBQuery',
    'MetricsAggregator',
    'AggregationPeriod',
]

__version__ = '1.0.0'
