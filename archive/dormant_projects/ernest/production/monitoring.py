"""
Ernest Production Monitoring
=============================
Health checks and monitoring for Ernest creative writing system.

Features:
- Learning engine health monitoring
- Hemingway score trend tracking
- Mode preference convergence
- Performance metrics
- Prometheus-compatible export

Philosophy: "Know what's happening. See what's working."
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


class HealthStatus(Enum):
    """Overall health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class HemingwayScoreTrend:
    """Tracks Hemingway score trends over time"""
    scores: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_score(self, score: float):
        """Add a score to the trend"""
        self.scores.append(score)
        self.timestamps.append(time.time())

    def get_average(self, window_minutes: int = 60) -> Optional[float]:
        """Get average score over last N minutes"""
        if not self.scores:
            return None

        cutoff = time.time() - (window_minutes * 60)
        recent_scores = [
            score for score, ts in zip(self.scores, self.timestamps)
            if ts >= cutoff
        ]

        if not recent_scores:
            return None

        return sum(recent_scores) / len(recent_scores)

    def get_trend_direction(self) -> str:
        """Get trend direction: improving, declining, stable"""
        if len(self.scores) < 10:
            return "insufficient_data"

        # Compare first half vs second half
        mid = len(self.scores) // 2
        first_half_avg = sum(list(self.scores)[:mid]) / mid
        second_half_avg = sum(list(self.scores)[mid:]) / (len(self.scores) - mid)

        diff = second_half_avg - first_half_avg
        if diff > 2.0:
            return "improving"
        elif diff < -2.0:
            return "declining"
        else:
            return "stable"


@dataclass
class ModeConvergenceTracker:
    """Tracks mode preference convergence"""
    mode_selections: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))

    def add_selection(self, mode: str):
        """Record a mode selection"""
        self.mode_selections.append(mode)
        self.timestamps.append(time.time())

    def is_converged(self, threshold: float = 0.7) -> bool:
        """Check if mode preferences have converged"""
        if len(self.mode_selections) < 20:
            return False

        # Count mode frequencies
        mode_counts = {}
        for mode in self.mode_selections:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Check if any mode dominates
        total = len(self.mode_selections)
        for count in mode_counts.values():
            if count / total >= threshold:
                return True

        return False

    def get_dominant_mode(self) -> Optional[str]:
        """Get the most frequently selected mode"""
        if not self.mode_selections:
            return None

        mode_counts = {}
        for mode in self.mode_selections:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return max(mode_counts.items(), key=lambda x: x[1])[0]


@dataclass
class PerformanceMetrics:
    """Performance metrics tracker"""
    refinement_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_update_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_refinement_latency(self, latency_ms: float):
        """Record refinement latency"""
        self.refinement_latencies.append(latency_ms)
        self.timestamps.append(time.time())

    def record_learning_update_latency(self, latency_ms: float):
        """Record learning update latency"""
        self.learning_update_latencies.append(latency_ms)

    def get_average_refinement_latency(self) -> Optional[float]:
        """Get average refinement latency"""
        if not self.refinement_latencies:
            return None
        return sum(self.refinement_latencies) / len(self.refinement_latencies)

    def get_p95_refinement_latency(self) -> Optional[float]:
        """Get 95th percentile refinement latency"""
        if not self.refinement_latencies:
            return None
        sorted_latencies = sorted(self.refinement_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    def get_average_learning_latency(self) -> Optional[float]:
        """Get average learning update latency"""
        if not self.learning_update_latencies:
            return None
        return sum(self.learning_update_latencies) / len(self.learning_update_latencies)


class ErnestHealthMonitor:
    """Health monitor for Ernest system"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ErnestHealthMonitor")
        self.score_trend = HemingwayScoreTrend()
        self.mode_convergence = ModeConvergenceTracker()
        self.performance = PerformanceMetrics()
        self.start_time = time.time()
        self.last_health_check = time.time()

        # Health check state
        self.background_learner_alive = True
        self.last_learning_update = time.time()
        self.last_refinement = time.time()

    def record_refinement(
        self,
        before_score: float,
        after_score: float,
        mode: str,
        latency_ms: float
    ):
        """Record a refinement for monitoring"""
        self.score_trend.add_score(after_score)
        self.mode_convergence.add_selection(mode)
        self.performance.record_refinement_latency(latency_ms)
        self.last_refinement = time.time()

    def record_learning_update(self, latency_ms: float):
        """Record a learning update"""
        self.performance.record_learning_update_latency(latency_ms)
        self.last_learning_update = time.time()
        self.background_learner_alive = True

    def check_background_learner_health(self, timeout_seconds: float = 120.0) -> bool:
        """Check if background learner is alive"""
        elapsed = time.time() - self.last_learning_update
        self.background_learner_alive = elapsed < timeout_seconds
        return self.background_learner_alive

    def get_health_status(self) -> HealthStatus:
        """Get overall health status"""
        issues = []

        # Check background learner
        if not self.check_background_learner_health():
            issues.append("background_learner_timeout")

        # Check recent activity
        time_since_refinement = time.time() - self.last_refinement
        if time_since_refinement > 300:  # 5 minutes
            issues.append("no_recent_refinements")

        # Check score trend
        trend = self.score_trend.get_trend_direction()
        if trend == "declining":
            issues.append("declining_scores")

        # Check performance
        avg_latency = self.performance.get_average_refinement_latency()
        if avg_latency and avg_latency > 200:  # 200ms threshold
            issues.append("high_latency")

        # Determine status
        if not issues:
            return HealthStatus.HEALTHY
        elif "background_learner_timeout" in issues:
            return HealthStatus.CRITICAL
        elif len(issues) >= 3:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.WARNING

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        uptime = time.time() - self.start_time
        health_status = self.get_health_status()

        # Score metrics
        avg_score_1h = self.score_trend.get_average(window_minutes=60)
        avg_score_24h = self.score_trend.get_average(window_minutes=1440)
        trend_direction = self.score_trend.get_trend_direction()

        # Mode convergence
        is_converged = self.mode_convergence.is_converged()
        dominant_mode = self.mode_convergence.get_dominant_mode()

        # Performance metrics
        avg_refinement_latency = self.performance.get_average_refinement_latency()
        p95_refinement_latency = self.performance.get_p95_refinement_latency()
        avg_learning_latency = self.performance.get_average_learning_latency()

        # Background learner
        time_since_learning_update = time.time() - self.last_learning_update
        background_learner_healthy = self.background_learner_alive

        return {
            "health_status": health_status.value,
            "uptime_seconds": uptime,
            "hemingway_scores": {
                "average_1h": avg_score_1h,
                "average_24h": avg_score_24h,
                "trend": trend_direction,
                "sample_count": len(self.score_trend.scores)
            },
            "mode_convergence": {
                "converged": is_converged,
                "dominant_mode": dominant_mode,
                "sample_count": len(self.mode_convergence.mode_selections)
            },
            "performance": {
                "avg_refinement_latency_ms": avg_refinement_latency,
                "p95_refinement_latency_ms": p95_refinement_latency,
                "avg_learning_latency_ms": avg_learning_latency
            },
            "background_learner": {
                "healthy": background_learner_healthy,
                "time_since_update_seconds": time_since_learning_update
            },
            "timestamp": datetime.now().isoformat()
        }

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        report = self.get_comprehensive_report()

        metrics = []

        # Health status (0=healthy, 1=warning, 2=degraded, 3=critical)
        health_map = {
            "healthy": 0,
            "warning": 1,
            "degraded": 2,
            "critical": 3
        }
        metrics.append(
            f'ernest_health_status {health_map[report["health_status"]]}'
        )

        # Uptime
        metrics.append(f'ernest_uptime_seconds {report["uptime_seconds"]:.1f}')

        # Hemingway scores
        if report["hemingway_scores"]["average_1h"]:
            metrics.append(
                f'ernest_hemingway_score_1h {report["hemingway_scores"]["average_1h"]:.2f}'
            )
        if report["hemingway_scores"]["average_24h"]:
            metrics.append(
                f'ernest_hemingway_score_24h {report["hemingway_scores"]["average_24h"]:.2f}'
            )

        # Trend (0=insufficient, 1=declining, 2=stable, 3=improving)
        trend_map = {
            "insufficient_data": 0,
            "declining": 1,
            "stable": 2,
            "improving": 3
        }
        metrics.append(
            f'ernest_score_trend {trend_map[report["hemingway_scores"]["trend"]]}'
        )

        # Mode convergence
        metrics.append(
            f'ernest_mode_converged {1 if report["mode_convergence"]["converged"] else 0}'
        )

        # Performance
        if report["performance"]["avg_refinement_latency_ms"]:
            metrics.append(
                f'ernest_refinement_latency_avg_ms {report["performance"]["avg_refinement_latency_ms"]:.1f}'
            )
        if report["performance"]["p95_refinement_latency_ms"]:
            metrics.append(
                f'ernest_refinement_latency_p95_ms {report["performance"]["p95_refinement_latency_ms"]:.1f}'
            )
        if report["performance"]["avg_learning_latency_ms"]:
            metrics.append(
                f'ernest_learning_latency_avg_ms {report["performance"]["avg_learning_latency_ms"]:.1f}'
            )

        # Background learner
        metrics.append(
            f'ernest_background_learner_healthy {1 if report["background_learner"]["healthy"] else 0}'
        )
        metrics.append(
            f'ernest_time_since_learning_update_seconds {report["background_learner"]["time_since_update_seconds"]:.1f}'
        )

        return "\n".join(metrics)


class ErnestAlertManager:
    """Alert manager for Ernest monitoring"""

    def __init__(self, health_monitor: ErnestHealthMonitor):
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(f"{__name__}.ErnestAlertManager")
        self.last_alert_time: Dict[str, float] = {}
        self.alert_cooldown = 300.0  # 5 minutes

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        now = time.time()

        # Alert 1: Background learner timeout
        if not self.health_monitor.background_learner_alive:
            if self._should_alert("background_learner_timeout", now):
                alerts.append({
                    "severity": "critical",
                    "type": "background_learner_timeout",
                    "message": "Background learner has not updated in >2 minutes",
                    "timestamp": datetime.now().isoformat()
                })

        # Alert 2: Declining scores
        trend = self.health_monitor.score_trend.get_trend_direction()
        if trend == "declining":
            if self._should_alert("declining_scores", now):
                alerts.append({
                    "severity": "warning",
                    "type": "declining_scores",
                    "message": "Hemingway scores are declining over time",
                    "timestamp": datetime.now().isoformat()
                })

        # Alert 3: High latency
        avg_latency = self.health_monitor.performance.get_average_refinement_latency()
        if avg_latency and avg_latency > 200:
            if self._should_alert("high_latency", now):
                alerts.append({
                    "severity": "warning",
                    "type": "high_latency",
                    "message": f"Average refinement latency is {avg_latency:.1f}ms (threshold: 200ms)",
                    "timestamp": datetime.now().isoformat()
                })

        # Alert 4: Mode not converging
        if len(self.health_monitor.mode_convergence.mode_selections) >= 50:
            if not self.health_monitor.mode_convergence.is_converged():
                if self._should_alert("mode_not_converging", now):
                    alerts.append({
                        "severity": "info",
                        "type": "mode_not_converging",
                        "message": "Mode preferences have not converged after 50 refinements",
                        "timestamp": datetime.now().isoformat()
                    })

        return alerts

    def _should_alert(self, alert_type: str, now: float) -> bool:
        """Check if we should send this alert (respects cooldown)"""
        last_alert = self.last_alert_time.get(alert_type, 0)
        if now - last_alert >= self.alert_cooldown:
            self.last_alert_time[alert_type] = now
            return True
        return False


# Quick helpers
def create_health_monitor() -> ErnestHealthMonitor:
    """Create health monitor with sensible defaults"""
    return ErnestHealthMonitor()


def create_alert_manager(health_monitor: ErnestHealthMonitor) -> ErnestAlertManager:
    """Create alert manager"""
    return ErnestAlertManager(health_monitor)
