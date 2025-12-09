"""
Alignment Framework Production Monitoring

Tracks latency metrics (P50, P95, P99) for all alignment components
with real-time alerting, dashboard integration, and Prometheus export.

**E2.1 Update (December 2025)**: Added alignment-specific metrics:
- Safety decision counters by risk level and outcome
- Deception flag counters by type
- Convergence violation counters by type
- Autonomy action counters by step type
- Resource utilization gauges by type

Usage:
    from HoloLoom.alignment.monitoring import AlignmentMonitor

    monitor = AlignmentMonitor()

    with monitor.track("guardrails"):
        # ... guardrails.evaluate() ...

    # Record alignment decisions
    monitor.record_safety_decision("HIGH", "blocked")
    monitor.record_deception_flag("CONSISTENCY")
    monitor.record_convergence_violation("POWER_SEEKING")
    monitor.record_autonomy_action("research")
    monitor.record_resource_usage("API_CALLS", 0.75)

    # View metrics
    print(monitor.get_summary())

    # Export for Prometheus
    print(monitor.export_prometheus())
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
from collections import defaultdict


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
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# E2.1: Alignment-Specific Metrics (December 2025)
# =============================================================================

class SafetyRiskLevel(Enum):
    """Risk levels for safety decisions (mirrors safety_guardrails.RiskLevel)."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DeceptionFlagType(Enum):
    """Types of deception detection flags."""
    HIDDEN_GOAL = "HIDDEN_GOAL"
    GOAL_DRIFT = "GOAL_DRIFT"
    INCONSISTENT = "INCONSISTENT"
    CAPABILITY_HIDING = "CAPABILITY_HIDING"
    REWARD_HACKING = "REWARD_HACKING"


class ConvergenceViolationType(Enum):
    """Types of instrumental convergence violations."""
    POWER_SEEKING = "POWER_SEEKING"
    RESOURCE_ACQUISITION = "RESOURCE_ACQUISITION"
    SELF_MODIFICATION = "SELF_MODIFICATION"
    SELF_PRESERVATION = "SELF_PRESERVATION"
    GOAL_CONTENT_INTEGRITY = "GOAL_CONTENT_INTEGRITY"


class AutonomyStepType(Enum):
    """Types of autonomy actions (agentic reasoning steps)."""
    RESEARCH = "research"
    VERIFICATION = "verification"
    PLAN_EXECUTE = "plan_execute"
    DIRECT = "direct"


class ResourceMetricType(Enum):
    """Types of resources tracked for utilization."""
    API_CALLS = "API_CALLS"
    COMPUTE = "COMPUTE"
    MEMORY = "MEMORY"
    STORAGE = "STORAGE"
    NETWORK = "NETWORK"


@dataclass
class AlignmentMetrics:
    """
    Counters and gauges for alignment-specific metrics.

    Designed for Prometheus export with labeled dimensions.
    """
    # Safety decision counters: {(risk_level, outcome): count}
    safety_decisions: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    # Deception flag counters: {flag_type: count}
    deception_flags: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Convergence violation counters: {violation_type: count}
    convergence_violations: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Autonomy action counters: {step_type: count}
    autonomy_actions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Resource utilization gauges: {resource_type: ratio (0.0-1.0)}
    resource_utilization: Dict[str, float] = field(default_factory=dict)

    # Timestamps for rate calculation
    first_event: Optional[datetime] = None
    last_event: Optional[datetime] = None

    def reset(self):
        """Reset all counters (useful for testing or periodic reset)."""
        self.safety_decisions.clear()
        self.deception_flags.clear()
        self.convergence_violations.clear()
        self.autonomy_actions.clear()
        self.resource_utilization.clear()
        self.first_event = None
        self.last_event = None


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

        # E2.1: Alignment-specific metrics (December 2025)
        self.alignment_metrics = AlignmentMetrics()

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

    # =========================================================================
    # E2.1: Alignment-Specific Metric Recording Methods (December 2025)
    # =========================================================================

    def record_safety_decision(
        self,
        risk_level: str,
        outcome: str,
    ):
        """
        Record a safety guardrails decision.

        Args:
            risk_level: One of SAFE, LOW, MEDIUM, HIGH, CRITICAL
            outcome: Either "allowed" or "blocked"

        Usage:
            monitor.record_safety_decision("HIGH", "blocked")
            monitor.record_safety_decision("LOW", "allowed")
        """
        # Normalize inputs
        risk_level = risk_level.upper()
        outcome = outcome.lower()

        # Validate
        valid_levels = {"SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        valid_outcomes = {"allowed", "blocked"}

        if risk_level not in valid_levels:
            risk_level = "UNKNOWN"
        if outcome not in valid_outcomes:
            outcome = "unknown"

        # Record
        key = (risk_level, outcome)
        self.alignment_metrics.safety_decisions[key] += 1
        self._update_event_timestamps()

    def record_deception_flag(self, flag_type: str):
        """
        Record a deception detection flag.

        Args:
            flag_type: One of HIDDEN_GOAL, GOAL_DRIFT, INCONSISTENT,
                      CAPABILITY_HIDING, REWARD_HACKING

        Usage:
            monitor.record_deception_flag("GOAL_DRIFT")
        """
        flag_type = flag_type.upper()
        valid_types = {
            "HIDDEN_GOAL", "GOAL_DRIFT", "INCONSISTENT",
            "CAPABILITY_HIDING", "REWARD_HACKING"
        }

        if flag_type not in valid_types:
            flag_type = "UNKNOWN"

        self.alignment_metrics.deception_flags[flag_type] += 1
        self._update_event_timestamps()

        # High severity flags trigger alerts
        if flag_type in {"HIDDEN_GOAL", "GOAL_DRIFT"}:
            self._create_alert(
                level=AlertLevel.CRITICAL,
                component="deception_detection",
                metric="deception_flag",
                value=1.0,
                threshold=0.0,
                message=f"Deception flag raised: {flag_type}"
            )

    def record_convergence_violation(self, violation_type: str):
        """
        Record an instrumental convergence violation.

        Args:
            violation_type: One of POWER_SEEKING, RESOURCE_ACQUISITION,
                           SELF_MODIFICATION, SELF_PRESERVATION, GOAL_CONTENT_INTEGRITY

        Usage:
            monitor.record_convergence_violation("POWER_SEEKING")
        """
        violation_type = violation_type.upper()
        valid_types = {
            "POWER_SEEKING", "RESOURCE_ACQUISITION", "SELF_MODIFICATION",
            "SELF_PRESERVATION", "GOAL_CONTENT_INTEGRITY"
        }

        if violation_type not in valid_types:
            violation_type = "UNKNOWN"

        self.alignment_metrics.convergence_violations[violation_type] += 1
        self._update_event_timestamps()

        # All convergence violations are concerning
        self._create_alert(
            level=AlertLevel.WARNING,
            component="convergence_guard",
            metric="convergence_violation",
            value=1.0,
            threshold=0.0,
            message=f"Convergence violation detected: {violation_type}"
        )

    def record_autonomy_action(self, step_type: str):
        """
        Record an autonomy action (agentic reasoning step).

        Args:
            step_type: One of research, verification, plan_execute, direct

        Usage:
            monitor.record_autonomy_action("research")
        """
        step_type = step_type.lower()
        valid_types = {"research", "verification", "plan_execute", "direct"}

        if step_type not in valid_types:
            step_type = "unknown"

        self.alignment_metrics.autonomy_actions[step_type] += 1
        self._update_event_timestamps()

    def record_resource_usage(self, resource_type: str, ratio: float):
        """
        Record resource utilization ratio.

        Args:
            resource_type: One of API_CALLS, COMPUTE, MEMORY, STORAGE, NETWORK
            ratio: Utilization ratio from 0.0 to 1.0

        Usage:
            monitor.record_resource_usage("API_CALLS", 0.75)
        """
        resource_type = resource_type.upper()
        valid_types = {"API_CALLS", "COMPUTE", "MEMORY", "STORAGE", "NETWORK"}

        if resource_type not in valid_types:
            resource_type = "UNKNOWN"

        # Clamp ratio to [0.0, 1.0]
        ratio = max(0.0, min(1.0, ratio))

        self.alignment_metrics.resource_utilization[resource_type] = ratio
        self._update_event_timestamps()

        # Alert on high resource usage
        if ratio > 0.9:
            self._create_alert(
                level=AlertLevel.WARNING,
                component="resource_monitor",
                metric=f"resource_{resource_type.lower()}",
                value=ratio,
                threshold=0.9,
                message=f"High resource utilization: {resource_type} at {ratio:.1%}"
            )

    def _update_event_timestamps(self):
        """Update first/last event timestamps."""
        now = datetime.now()
        if self.alignment_metrics.first_event is None:
            self.alignment_metrics.first_event = now
        self.alignment_metrics.last_event = now

    def get_alignment_summary(self) -> Dict[str, Any]:
        """
        Get summary of alignment-specific metrics.

        Returns:
            Dictionary with all alignment counters and gauges
        """
        am = self.alignment_metrics

        # Calculate totals
        total_safety_decisions = sum(am.safety_decisions.values())
        total_blocked = sum(
            count for (level, outcome), count in am.safety_decisions.items()
            if outcome == "blocked"
        )
        total_deception_flags = sum(am.deception_flags.values())
        total_convergence_violations = sum(am.convergence_violations.values())
        total_autonomy_actions = sum(am.autonomy_actions.values())

        return {
            "safety_decisions": {
                "total": total_safety_decisions,
                "blocked": total_blocked,
                "blocked_rate": total_blocked / total_safety_decisions if total_safety_decisions > 0 else 0.0,
                "by_risk_level_outcome": dict(am.safety_decisions),
            },
            "deception_flags": {
                "total": total_deception_flags,
                "by_type": dict(am.deception_flags),
            },
            "convergence_violations": {
                "total": total_convergence_violations,
                "by_type": dict(am.convergence_violations),
            },
            "autonomy_actions": {
                "total": total_autonomy_actions,
                "by_step_type": dict(am.autonomy_actions),
            },
            "resource_utilization": dict(am.resource_utilization),
            "event_window": {
                "first_event": am.first_event.isoformat() if am.first_event else None,
                "last_event": am.last_event.isoformat() if am.last_event else None,
            },
        }

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

        Includes:
        - Latency metrics (P50, P95, P99 per component)
        - Alert counts
        - E2.1 (December 2025): Alignment-specific metrics:
          - alignment_safety_decisions_total{risk_level, outcome}
          - alignment_deception_flags_total{flag_type}
          - alignment_convergence_violations_total{type}
          - alignment_autonomy_actions_total{step_type}
          - alignment_resource_utilization_ratio{resource_type}

        Returns:
            Metrics in Prometheus text format
        """
        lines = []

        # --- Latency metrics ---
        lines.append("# HELP alignment_latency_p50 50th percentile latency in milliseconds")
        lines.append("# TYPE alignment_latency_p50 gauge")
        lines.append("# HELP alignment_latency_p95 95th percentile latency in milliseconds")
        lines.append("# TYPE alignment_latency_p95 gauge")
        lines.append("# HELP alignment_latency_p99 99th percentile latency in milliseconds")
        lines.append("# TYPE alignment_latency_p99 gauge")
        lines.append("# HELP alignment_samples_total Total samples recorded")
        lines.append("# TYPE alignment_samples_total counter")

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

        # --- Alert counts ---
        lines.append("")
        lines.append("# HELP alignment_alerts_total Total alerts by severity level")
        lines.append("# TYPE alignment_alerts_total counter")

        for level in AlertLevel:
            count = len([a for a in self.alerts if a.level == level])
            lines.append(f'alignment_alerts_total{{level="{level.value}"}} {count}')

        # =================================================================
        # E2.1: Alignment-Specific Metrics (December 2025)
        # =================================================================

        am = self.alignment_metrics

        # --- Safety Decision Counters ---
        lines.append("")
        lines.append("# HELP alignment_safety_decisions_total Total safety decisions by risk level and outcome")
        lines.append("# TYPE alignment_safety_decisions_total counter")

        for (risk_level, outcome), count in am.safety_decisions.items():
            lines.append(
                f'alignment_safety_decisions_total{{risk_level="{risk_level}",outcome="{outcome}"}} {count}'
            )

        # --- Deception Flag Counters ---
        lines.append("")
        lines.append("# HELP alignment_deception_flags_total Total deception flags by type")
        lines.append("# TYPE alignment_deception_flags_total counter")

        for flag_type, count in am.deception_flags.items():
            lines.append(
                f'alignment_deception_flags_total{{flag_type="{flag_type}"}} {count}'
            )

        # --- Convergence Violation Counters ---
        lines.append("")
        lines.append("# HELP alignment_convergence_violations_total Total convergence violations by type")
        lines.append("# TYPE alignment_convergence_violations_total counter")

        for violation_type, count in am.convergence_violations.items():
            lines.append(
                f'alignment_convergence_violations_total{{type="{violation_type}"}} {count}'
            )

        # --- Autonomy Action Counters ---
        lines.append("")
        lines.append("# HELP alignment_autonomy_actions_total Total autonomy actions by step type")
        lines.append("# TYPE alignment_autonomy_actions_total counter")

        for step_type, count in am.autonomy_actions.items():
            lines.append(
                f'alignment_autonomy_actions_total{{step_type="{step_type}"}} {count}'
            )

        # --- Resource Utilization Gauges ---
        lines.append("")
        lines.append("# HELP alignment_resource_utilization_ratio Resource utilization ratio (0.0-1.0)")
        lines.append("# TYPE alignment_resource_utilization_ratio gauge")

        for resource_type, ratio in am.resource_utilization.items():
            lines.append(
                f'alignment_resource_utilization_ratio{{resource_type="{resource_type}"}} {ratio}'
            )

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
