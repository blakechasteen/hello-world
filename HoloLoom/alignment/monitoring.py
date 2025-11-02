"""
Alignment Framework Production Monitoring

Tracks latency metrics (P50, P95, P99) for all alignment components
with real-time alerting and dashboard integration.

Usage:
    from HoloLoom.alignment.monitoring import AlignmentMonitor

    monitor = AlignmentMonitor()

    with monitor.track("guardrails"):
        # ... guardrails.evaluate() ...

    # View metrics
    print(monitor.get_summary())

    # Check alerts
    alerts = monitor.check_alerts()
"""

import time
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import json
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LatencyMetrics:
    """Latency statistics for a component."""
    component: str
    samples: List[float] = field(default_factory=list)
    window_size: int = 1000  # Keep last 1000 measurements

    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.samples.append(latency_ms)

        # Sliding window
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]

    def get_percentile(self, p: float) -> float:
        """Get percentile (0.0-1.0)."""
        if not self.samples:
            return 0.0

        sorted_samples = sorted(self.samples)
        index = int(p * len(sorted_samples))
        index = min(index, len(sorted_samples) - 1)
        return sorted_samples[index]

    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive statistics."""
        if not self.samples:
            return {
                "count": 0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "count": len(self.samples),
            "p50": self.get_percentile(0.50),
            "p95": self.get_percentile(0.95),
            "p99": self.get_percentile(0.99),
            "mean": statistics.mean(self.samples),
            "std": statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0,
            "min": min(self.samples),
            "max": max(self.samples),
        }


@dataclass
class Alert:
    """Monitoring alert."""
    level: AlertLevel
    component: str
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "component": self.component,
            "metric": self.metric,
            "value": value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class AlignmentMonitor:
    """
    Production monitoring for alignment framework.

    Tracks latency metrics with configurable alerting thresholds.
    """

    # Default P99 thresholds (ms)
    DEFAULT_THRESHOLDS = {
        "guardrails": 1.0,
        "detector": 2.0,
        "guard": 0.5,
        "audit": 10.0,
        "pipeline": 20.0,
    }

    # Warning thresholds (50% of critical)
    WARNING_MULTIPLIER = 0.5

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1000,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize monitor.

        Args:
            thresholds: Custom P99 thresholds (ms) per component
            window_size: Number of samples to keep per component
            persist_path: Optional path to persist metrics
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.window_size = window_size
        self.persist_path = persist_path

        # Metrics storage
        self.metrics: Dict[str, LatencyMetrics] = {}

        # Alert history
        self.alerts: List[Alert] = []
        self.alert_cooldown: Dict[str, datetime] = {}  # Prevent spam
        self.cooldown_duration = timedelta(minutes=5)

        # Session tracking
        self.session_start = datetime.now()

    def _get_or_create_metrics(self, component: str) -> LatencyMetrics:
        """Get or create metrics for component."""
        if component not in self.metrics:
            self.metrics[component] = LatencyMetrics(
                component=component,
                window_size=self.window_size
            )
        return self.metrics[component]

    @contextmanager
    def track(self, component: str):
        """
        Context manager to track component latency.

        Usage:
            with monitor.track("guardrails"):
                guardrails.evaluate(request)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            self.record(component, latency_ms)

    def record(self, component: str, latency_ms: float):
        """Manually record a latency measurement."""
        metrics = self._get_or_create_metrics(component)
        metrics.record(latency_ms)

        # Check for threshold violations
        self._check_thresholds(component)

    def _check_thresholds(self, component: str):
        """Check if component exceeds P99 threshold."""
        if component not in self.thresholds:
            return

        metrics = self.metrics[component]
        stats = metrics.get_stats()

        if stats["count"] < 100:  # Need minimum samples
            return

        p99 = stats["p99"]
        critical_threshold = self.thresholds[component]
        warning_threshold = critical_threshold * self.WARNING_MULTIPLIER

        # Check for alerts
        if p99 > critical_threshold:
            self._create_alert(
                level=AlertLevel.CRITICAL,
                component=component,
                metric="p99",
                value=p99,
                threshold=critical_threshold,
                message=f"P99 latency {p99:.2f}ms exceeds critical threshold {critical_threshold:.2f}ms"
            )
        elif p99 > warning_threshold:
            self._create_alert(
                level=AlertLevel.WARNING,
                component=component,
                metric="p99",
                value=p99,
                threshold=warning_threshold,
                message=f"P99 latency {p99:.2f}ms exceeds warning threshold {warning_threshold:.2f}ms"
            )

    def _create_alert(
        self,
        level: AlertLevel,
        component: str,
        metric: str,
        value: float,
        threshold: float,
        message: str,
    ):
        """Create alert with cooldown."""
        alert_key = f"{component}_{metric}_{level.value}"

        # Check cooldown
        if alert_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[alert_key]
            if datetime.now() - last_alert < self.cooldown_duration:
                return  # Skip - in cooldown period

        # Create alert
        alert = Alert(
            level=level,
            component=component,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message,
        )

        self.alerts.append(alert)
        self.alert_cooldown[alert_key] = datetime.now()

        # Print to console (can be replaced with logging)
        symbol = "🔴" if level == AlertLevel.CRITICAL else "⚠️"
        print(f"{symbol} ALERT [{level.value.upper()}]: {message}")

    def get_stats(self, component: str) -> Dict[str, float]:
        """Get statistics for a component."""
        if component not in self.metrics:
            return {}
        return self.metrics[component].get_stats()

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all components."""
        return {
            component: metrics.get_stats()
            for component, metrics in self.metrics.items()
        }

    def get_summary(self) -> str:
        """Get formatted summary of all metrics."""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("ALIGNMENT FRAMEWORK MONITORING SUMMARY")
        lines.append("="*70)
        lines.append(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Uptime: {datetime.now() - self.session_start}")
        lines.append("")

        # Component stats
        lines.append(f"{'Component':<20} {'Count':<8} {'P50':<10} {'P95':<10} {'P99':<10} {'Status':<10}")
        lines.append("-" * 70)

        for component in sorted(self.metrics.keys()):
            stats = self.get_stats(component)
            if stats["count"] == 0:
                continue

            # Status check
            status = "✅ OK"
            if component in self.thresholds:
                threshold = self.thresholds[component]
                if stats["p99"] > threshold:
                    status = "🔴 CRITICAL"
                elif stats["p99"] > threshold * self.WARNING_MULTIPLIER:
                    status = "⚠️  WARNING"

            lines.append(
                f"{component:<20} {stats['count']:<8} {stats['p50']:<10.3f} "
                f"{stats['p95']:<10.3f} {stats['p99']:<10.3f} {status:<10}"
            )

        lines.append("="*70)

        # Recent alerts
        if self.alerts:
            lines.append(f"\nRecent Alerts ({len(self.alerts)} total):")
            lines.append("-" * 70)

            # Show last 5 alerts
            for alert in self.alerts[-5:]:
                symbol = "🔴" if alert.level == AlertLevel.CRITICAL else "⚠️"
                lines.append(
                    f"{symbol} [{alert.level.value.upper()}] {alert.component}: {alert.message}"
                )
                lines.append(f"   Timestamp: {alert.timestamp.strftime('%H:%M:%S')}")

        lines.append("")

        return "\n".join(lines)

    def check_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        Get alerts, optionally filtered by level.

        Args:
            level: Filter by alert level (None = all)

        Returns:
            List of alerts
        """
        if level is None:
            return self.alerts.copy()

        return [a for a in self.alerts if a.level == level]

    def clear_alerts(self):
        """Clear alert history."""
        self.alerts.clear()
        self.alert_cooldown.clear()

    def persist_metrics(self):
        """Persist metrics to disk."""
        if not self.persist_path:
            return

        data = {
            "session_start": self.session_start.isoformat(),
            "metrics": {
                component: {
                    "stats": metrics.get_stats(),
                    "samples": metrics.samples,
                }
                for component, metrics in self.metrics.items()
            },
            "alerts": [
                {
                    "level": alert.level.value,
                    "component": alert.component,
                    "metric": alert.metric,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self.alerts
            ],
        }

        self.persist_path.write_text(json.dumps(data, indent=2))
        print(f"📊 Metrics persisted to {self.persist_path}")

    def load_metrics(self):
        """Load metrics from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        data = json.loads(self.persist_path.read_text())

        # Restore session start
        self.session_start = datetime.fromisoformat(data["session_start"])

        # Restore metrics
        for component, metric_data in data["metrics"].items():
            metrics = LatencyMetrics(component=component)
            metrics.samples = metric_data["samples"]
            self.metrics[component] = metrics

        # Restore alerts
        for alert_data in data["alerts"]:
            alert = Alert(
                level=AlertLevel(alert_data["level"]),
                component=alert_data["component"],
                metric=alert_data["metric"],
                value=alert_data["value"],
                threshold=alert_data["threshold"],
                message=alert_data["message"],
                timestamp=datetime.fromisoformat(alert_data["timestamp"]),
            )
            self.alerts.append(alert)

        print(f"📊 Metrics loaded from {self.persist_path}")

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        lines = []

        for component, metrics in self.metrics.items():
            stats = metrics.get_stats()

            if stats["count"] == 0:
                continue

            # Latency percentiles
            lines.append(f'alignment_latency_p50{{component="{component}"}} {stats["p50"]}')
            lines.append(f'alignment_latency_p95{{component="{component}"}} {stats["p95"]}')
            lines.append(f'alignment_latency_p99{{component="{component}"}} {stats["p99"]}')

            # Sample count
            lines.append(f'alignment_samples_total{{component="{component}"}} {stats["count"]}')

        # Alert counts
        for level in AlertLevel:
            count = len([a for a in self.alerts if a.level == level])
            lines.append(f'alignment_alerts_total{{level="{level.value}"}} {count}')

        return "\n".join(lines)


# Global monitor instance (singleton pattern)
_global_monitor: Optional[AlignmentMonitor] = None


def get_global_monitor() -> AlignmentMonitor:
    """Get global monitor instance (singleton)."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AlignmentMonitor()
    return _global_monitor


def set_global_monitor(monitor: AlignmentMonitor):
    """Set global monitor instance."""
    global _global_monitor
    _global_monitor = monitor
