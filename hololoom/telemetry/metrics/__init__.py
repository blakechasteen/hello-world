"""
Metrics module - Prometheus metrics collection.

Provides counters, gauges, histograms, and summaries for HoloLoom systems.
"""

from HoloLoom.telemetry.metrics.prometheus import (
    PrometheusRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    create_registry,
    get_registry,
    # Convenience functions
    counter,
    gauge,
    histogram,
    summary,
)
from HoloLoom.telemetry.metrics.collectors import (
    HoloLoomCollector,
    create_default_collector,
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
