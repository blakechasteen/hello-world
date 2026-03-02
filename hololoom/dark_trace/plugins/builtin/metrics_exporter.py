"""
Metrics Exporter Plugin
=======================

TRUSTED-level plugin that exports Dark Trace metrics to Prometheus format.

This plugin provides:
- Analysis latency metrics
- Feature activation statistics
- Plugin performance tracking
- Safety alert counters
- Memory and resource usage

Trust Level: TRUSTED (official plugin, full capabilities)
Capabilities: READ_ACTIVATIONS, EXTERNAL_NETWORK

Date: 2025-12-22
Phase: 11.7 - Built-in Plugins
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.plugins.safety_gate import PluginSafetyGate
    from hololoom.dark_trace.result import TraceResult

from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
    MonitorPlugin,
)
from hololoom.dark_trace.plugins.safety_gate import (
    PluginCapability,
    TrustLevel,
)

logger = logging.getLogger("hololoom.dark_trace.plugins.builtin.metrics_exporter")


# =============================================================================
# Prometheus Metric Types
# =============================================================================

@dataclass
class PrometheusMetric:
    """A single Prometheus metric."""
    name: str
    value: float
    metric_type: str  # counter, gauge, histogram, summary
    help_text: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_prometheus_format(self) -> str:
        """Format as Prometheus exposition format."""
        lines = []

        # Help line
        lines.append(f"# HELP {self.name} {self.help_text}")

        # Type line
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        # Value line with labels
        if self.labels:
            label_str = ",".join(
                f'{k}="{v}"' for k, v in self.labels.items()
            )
            metric_line = f"{self.name}{{{label_str}}} {self.value}"
        else:
            metric_line = f"{self.name} {self.value}"

        if self.timestamp:
            metric_line += f" {int(self.timestamp * 1000)}"

        lines.append(metric_line)
        return "\n".join(lines)


@dataclass
class MetricsBatch:
    """A batch of metrics for export."""
    metrics: List[PrometheusMetric] = field(default_factory=list)
    collection_time: datetime = field(default_factory=datetime.now)

    def to_prometheus_format(self) -> str:
        """Format all metrics in Prometheus exposition format."""
        return "\n\n".join(m.to_prometheus_format() for m in self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "collection_time": self.collection_time.isoformat(),
            "metrics_count": len(self.metrics),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "type": m.metric_type,
                    "labels": m.labels,
                }
                for m in self.metrics
            ],
        }


# =============================================================================
# Metrics Exporter Plugin
# =============================================================================

class MetricsExporterPlugin(MonitorPlugin):
    """
    TRUSTED plugin that exports Dark Trace metrics to Prometheus format.

    This plugin collects metrics from the Dark Trace engine and exports
    them in Prometheus exposition format for monitoring and alerting.

    Metrics Exported:
    - dark_trace_analysis_total (counter): Total analyses performed
    - dark_trace_analysis_duration_seconds (histogram): Analysis latency
    - dark_trace_feature_activations (gauge): Current feature activations
    - dark_trace_safety_alerts_total (counter): Safety alerts by level
    - dark_trace_plugin_count (gauge): Number of loaded plugins
    - dark_trace_lens_results_total (counter): Lens results by type
    """

    # Metric prefix for all Dark Trace metrics
    METRIC_PREFIX = "dark_trace"

    def __init__(
        self,
        export_interval_seconds: float = 60.0,
        enable_histogram: bool = True,
    ) -> None:
        """
        Initialize metrics exporter.

        Args:
            export_interval_seconds: How often to collect metrics
            enable_histogram: Whether to collect histogram metrics
        """
        self._engine: Optional["DarkTraceEngine"] = None
        self._safety_gate: Optional["PluginSafetyGate"] = None
        self._export_interval = export_interval_seconds
        self._enable_histogram = enable_histogram

        # Counters (monotonically increasing)
        self._counters: Dict[str, float] = {
            "analysis_total": 0,
            "safety_alerts_info": 0,
            "safety_alerts_warning": 0,
            "safety_alerts_high": 0,
            "safety_alerts_critical": 0,
            "lens_results_total": 0,
        }

        # Gauges (point-in-time values)
        self._gauges: Dict[str, float] = {
            "plugin_count": 0,
            "active_features": 0,
            "memory_usage_bytes": 0,
        }

        # Histogram buckets for latency
        self._latency_buckets = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]
        self._latency_bucket_counts: Dict[float, int] = {
            b: 0 for b in self._latency_buckets
        }
        self._latency_bucket_counts[float("inf")] = 0
        self._latency_sum = 0.0
        self._latency_count = 0

        # Export callbacks
        self._export_callbacks: List[Callable[[MetricsBatch], None]] = []

        self._initialized = False
        self._background_task: Optional[asyncio.Task] = None

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        return PluginMetadata(
            name="metrics_exporter",
            version="1.0.0",
            author="HoloLoom Team",
            description=(
                "TRUSTED-level plugin that exports Dark Trace metrics to "
                "Prometheus format for monitoring and alerting"
            ),
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.EXTERNAL_NETWORK,  # For push gateway
            ],
            signature="HOLOLOOM_BUILTIN",
            min_dark_trace_version="1.0.0",
            tags=["metrics", "prometheus", "monitoring", "builtin"],
        )

    async def initialize(
        self,
        engine: "DarkTraceEngine",
        safety_gate: "PluginSafetyGate",
    ) -> None:
        """Initialize the metrics exporter."""
        self._engine = engine
        self._safety_gate = safety_gate
        self._initialized = True
        logger.info("MetricsExporterPlugin initialized")

    async def shutdown(self) -> None:
        """Clean shutdown."""
        # Cancel background task if running
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("MetricsExporterPlugin shutting down")

    def describe_behavior(self) -> str:
        """Describe plugin behavior for transparency."""
        return (
            "This plugin collects and exports metrics from the Dark Trace "
            "engine in Prometheus exposition format. It tracks analysis counts, "
            "latency histograms, feature activations, safety alerts, and plugin "
            "counts. Metrics can be exported via HTTP endpoint or push gateway. "
            "The plugin only reads engine state and does not modify behavior."
        )

    # -------------------------------------------------------------------------
    # MonitorPlugin Interface
    # -------------------------------------------------------------------------

    async def on_analysis_complete(
        self,
        trace_result: "TraceResult",
    ) -> None:
        """
        Called after each analysis completes.

        Updates metrics based on the trace result.
        """
        if not self._initialized:
            return

        # Update counters
        self._counters["analysis_total"] += 1

        # Update latency histogram
        duration = getattr(trace_result, 'duration_seconds', 0.0)
        if self._enable_histogram and duration > 0:
            self._record_latency(duration)

        # Count lens results
        lens_results = getattr(trace_result, 'lens_results', [])
        self._counters["lens_results_total"] += len(lens_results)

        # Count active features
        total_features = 0
        for lens_result in lens_results:
            features = getattr(lens_result, 'active_features', [])
            total_features += len(features)
        self._gauges["active_features"] = total_features

    def _record_latency(self, duration: float) -> None:
        """Record latency in histogram buckets."""
        self._latency_sum += duration
        self._latency_count += 1

        for bucket in self._latency_buckets:
            if duration <= bucket:
                self._latency_bucket_counts[bucket] += 1
        self._latency_bucket_counts[float("inf")] += 1

    def record_safety_alert(self, level: str) -> None:
        """Record a safety alert by level."""
        counter_key = f"safety_alerts_{level.lower()}"
        if counter_key in self._counters:
            self._counters[counter_key] += 1

    def update_plugin_count(self, count: int) -> None:
        """Update the plugin count gauge."""
        self._gauges["plugin_count"] = count

    def update_memory_usage(self, bytes_used: int) -> None:
        """Update memory usage gauge."""
        self._gauges["memory_usage_bytes"] = bytes_used

    # -------------------------------------------------------------------------
    # Metrics Collection
    # -------------------------------------------------------------------------

    def collect_metrics(self) -> MetricsBatch:
        """Collect all current metrics."""
        metrics = []

        # Counter metrics
        metrics.append(PrometheusMetric(
            name=f"{self.METRIC_PREFIX}_analysis_total",
            value=self._counters["analysis_total"],
            metric_type="counter",
            help_text="Total number of Dark Trace analyses performed",
        ))

        metrics.append(PrometheusMetric(
            name=f"{self.METRIC_PREFIX}_lens_results_total",
            value=self._counters["lens_results_total"],
            metric_type="counter",
            help_text="Total number of lens results generated",
        ))

        # Safety alerts by level
        for level in ["info", "warning", "high", "critical"]:
            metrics.append(PrometheusMetric(
                name=f"{self.METRIC_PREFIX}_safety_alerts_total",
                value=self._counters[f"safety_alerts_{level}"],
                metric_type="counter",
                help_text="Total safety alerts by level",
                labels={"level": level},
            ))

        # Gauge metrics
        metrics.append(PrometheusMetric(
            name=f"{self.METRIC_PREFIX}_plugin_count",
            value=self._gauges["plugin_count"],
            metric_type="gauge",
            help_text="Number of currently loaded plugins",
        ))

        metrics.append(PrometheusMetric(
            name=f"{self.METRIC_PREFIX}_active_features",
            value=self._gauges["active_features"],
            metric_type="gauge",
            help_text="Number of currently active features",
        ))

        metrics.append(PrometheusMetric(
            name=f"{self.METRIC_PREFIX}_memory_usage_bytes",
            value=self._gauges["memory_usage_bytes"],
            metric_type="gauge",
            help_text="Memory usage in bytes",
        ))

        # Histogram metrics (if enabled)
        if self._enable_histogram and self._latency_count > 0:
            # Bucket counts
            cumulative = 0
            for bucket in self._latency_buckets:
                cumulative += self._latency_bucket_counts[bucket]
                metrics.append(PrometheusMetric(
                    name=f"{self.METRIC_PREFIX}_analysis_duration_seconds_bucket",
                    value=cumulative,
                    metric_type="histogram",
                    help_text="Analysis duration histogram",
                    labels={"le": str(bucket)},
                ))

            # +Inf bucket
            metrics.append(PrometheusMetric(
                name=f"{self.METRIC_PREFIX}_analysis_duration_seconds_bucket",
                value=self._latency_count,
                metric_type="histogram",
                help_text="Analysis duration histogram",
                labels={"le": "+Inf"},
            ))

            # Sum and count
            metrics.append(PrometheusMetric(
                name=f"{self.METRIC_PREFIX}_analysis_duration_seconds_sum",
                value=self._latency_sum,
                metric_type="histogram",
                help_text="Sum of analysis durations",
            ))

            metrics.append(PrometheusMetric(
                name=f"{self.METRIC_PREFIX}_analysis_duration_seconds_count",
                value=self._latency_count,
                metric_type="histogram",
                help_text="Count of analyses for histogram",
            ))

        return MetricsBatch(metrics=metrics)

    def get_prometheus_output(self) -> str:
        """Get metrics in Prometheus exposition format."""
        batch = self.collect_metrics()
        return batch.to_prometheus_format()

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        batch = self.collect_metrics()
        return batch.to_dict()

    # -------------------------------------------------------------------------
    # Export Callbacks
    # -------------------------------------------------------------------------

    def add_export_callback(
        self,
        callback: Callable[[MetricsBatch], None],
    ) -> None:
        """Add a callback for metrics export."""
        self._export_callbacks.append(callback)

    def remove_export_callback(
        self,
        callback: Callable[[MetricsBatch], None],
    ) -> None:
        """Remove an export callback."""
        if callback in self._export_callbacks:
            self._export_callbacks.remove(callback)

    async def _export_loop(self) -> None:
        """Background loop for periodic metric export."""
        while True:
            try:
                await asyncio.sleep(self._export_interval)
                batch = self.collect_metrics()

                for callback in self._export_callbacks:
                    try:
                        callback(batch)
                    except Exception as e:
                        logger.warning(f"Export callback failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")

    def start_background_export(self) -> None:
        """Start background metrics export."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._export_loop())
            logger.info("Started background metrics export")

    def stop_background_export(self) -> None:
        """Stop background metrics export."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            logger.info("Stopped background metrics export")

    def get_statistics(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_enabled": self._enable_histogram,
            "latency_count": self._latency_count,
            "latency_sum": self._latency_sum,
            "export_interval": self._export_interval,
            "callback_count": len(self._export_callbacks),
        }


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "MetricsExporterPlugin",
    "PrometheusMetric",
    "MetricsBatch",
]
