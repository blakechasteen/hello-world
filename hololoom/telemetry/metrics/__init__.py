"""
Metrics module - Prometheus metrics collection.

Provides counters, gauges, histograms, and summaries for HoloLoom systems.
"""

from hololoom.telemetry.metrics.collectors import (
    HoloLoomCollector,
    create_default_collector,
)
from hololoom.telemetry.metrics.prometheus import (
    Counter,
    Gauge,
    Histogram,
    PrometheusRegistry,
    Summary,
    # Convenience functions
    counter,
    create_registry,
    gauge,
    get_registry,
    histogram,
    summary,
)

__all__ = [
    # Registry
    "PrometheusRegistry",
    "create_registry",
    "get_registry",
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Functions
    "counter",
    "gauge",
    "histogram",
    "summary",
    # Collectors
    "HoloLoomCollector",
    "create_default_collector",
]
