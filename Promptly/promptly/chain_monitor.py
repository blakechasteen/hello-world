#!/usr/bin/env python3
"""
Chain Monitor - Real-time metrics and monitoring

Provides:
- Real-time metrics dashboard
- Cost tracking per chain
- Success/failure rates
- Average execution time
- Anomaly detection and alerts
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics


@dataclass
class ChainMetrics:
    """Metrics for a single chain"""
    chain_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    total_cost: float = 0.0
    durations: deque = field(default_factory=lambda: deque(maxlen=100))
    costs: deque = field(default_factory=lambda: deque(maxlen=100))
    errors: List[str] = field(default_factory=list)
    last_execution: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def average_duration(self) -> float:
        if not self.durations:
            return 0.0
        return statistics.mean(self.durations)

    @property
    def average_cost(self) -> float:
        if not self.costs:
            return 0.0
        return statistics.mean(self.costs)


@dataclass
class Alert:
    """Monitoring alert"""
    timestamp: datetime
    severity: str  # critical, warning, info
    chain_name: str
    message: str
    metric: str
    value: Any


class ChainMonitor:
    """
    Monitor chain executions in real-time.

    Features:
    - Track execution metrics
    - Calculate success rates
    - Monitor performance trends
    - Detect anomalies
    - Generate alerts
    """

    def __init__(self):
        self.metrics: Dict[str, ChainMetrics] = {}
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Anomaly detection thresholds
        self.duration_threshold_multiplier = 2.0  # Alert if duration > 2x average
        self.cost_threshold_multiplier = 2.0
        self.failure_rate_threshold = 0.3  # Alert if > 30% failures
        self.min_samples_for_anomaly = 10

    def record_execution(
        self,
        chain_name: str,
        duration: float,
        cost: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Record a chain execution.

        Args:
            chain_name: Name of the chain
            duration: Execution duration in seconds
            cost: Execution cost in dollars
            success: Whether execution was successful
            error: Error message if failed
        """
        # Get or create metrics
        if chain_name not in self.metrics:
            self.metrics[chain_name] = ChainMetrics(chain_name=chain_name)

        metrics = self.metrics[chain_name]

        # Update metrics
        metrics.total_executions += 1
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
            if error:
                metrics.errors.append(error)

        metrics.total_duration += duration
        metrics.total_cost += cost
        metrics.durations.append(duration)
        metrics.costs.append(cost)
        metrics.last_execution = datetime.now()

        # Check for anomalies
        self._detect_anomalies(chain_name)

    def _detect_anomalies(self, chain_name: str) -> None:
        """Detect anomalies in chain execution"""
        metrics = self.metrics[chain_name]

        # Need minimum samples for meaningful anomaly detection
        if len(metrics.durations) < self.min_samples_for_anomaly:
            return

        # Check duration anomaly
        avg_duration = statistics.mean(list(metrics.durations)[:-1])  # Exclude latest
        latest_duration = metrics.durations[-1]

        if latest_duration > avg_duration * self.duration_threshold_multiplier:
            self._create_alert(
                severity="warning",
                chain_name=chain_name,
                message=f"Execution time ({latest_duration:.2f}s) is {latest_duration/avg_duration:.1f}x the average ({avg_duration:.2f}s)",
                metric="duration",
                value=latest_duration
            )

        # Check cost anomaly
        avg_cost = statistics.mean(list(metrics.costs)[:-1])
        latest_cost = metrics.costs[-1]

        if latest_cost > avg_cost * self.cost_threshold_multiplier:
            self._create_alert(
                severity="warning",
                chain_name=chain_name,
                message=f"Execution cost (${latest_cost:.4f}) is {latest_cost/avg_cost:.1f}x the average (${avg_cost:.4f})",
                metric="cost",
                value=latest_cost
            )

        # Check failure rate
        if metrics.success_rate < (1 - self.failure_rate_threshold):
            self._create_alert(
                severity="critical",
                chain_name=chain_name,
                message=f"High failure rate: {(1-metrics.success_rate)*100:.1f}%",
                metric="failure_rate",
                value=1 - metrics.success_rate
            )

    def _create_alert(
        self,
        severity: str,
        chain_name: str,
        message: str,
        metric: str,
        value: Any
    ) -> None:
        """Create and dispatch alert"""
        alert = Alert(
            timestamp=datetime.now(),
            severity=severity,
            chain_name=chain_name,
            message=message,
            metric=metric,
            value=value
        )

        self.alerts.append(alert)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass  # Don't let callback errors break monitoring

    def register_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)

    def get_metrics(self, chain_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a chain or all chains.

        Args:
            chain_name: Optional chain name (None for all chains)

        Returns:
            Metrics dictionary
        """
        if chain_name:
            if chain_name not in self.metrics:
                return {}

            metrics = self.metrics[chain_name]
            return {
                "chain_name": metrics.chain_name,
                "total_executions": metrics.total_executions,
                "successful_executions": metrics.successful_executions,
                "failed_executions": metrics.failed_executions,
                "success_rate": metrics.success_rate,
                "total_duration": metrics.total_duration,
                "average_duration": metrics.average_duration,
                "total_cost": metrics.total_cost,
                "average_cost": metrics.average_cost,
                "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None,
                "recent_errors": metrics.errors[-5:] if metrics.errors else []
            }
        else:
            return {
                name: self.get_metrics(name)
                for name in self.metrics.keys()
            }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        total_executions = sum(m.total_executions for m in self.metrics.values())
        total_successful = sum(m.successful_executions for m in self.metrics.values())
        total_failed = sum(m.failed_executions for m in self.metrics.values())
        total_cost = sum(m.total_cost for m in self.metrics.values())

        return {
            "summary": {
                "total_chains": len(self.metrics),
                "total_executions": total_executions,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "overall_success_rate": total_successful / total_executions if total_executions > 0 else 0,
                "total_cost": total_cost,
                "active_alerts": len([a for a in self.alerts if a.severity in ["critical", "warning"]])
            },
            "chains": {
                name: {
                    "executions": m.total_executions,
                    "success_rate": m.success_rate,
                    "avg_duration": m.average_duration,
                    "avg_cost": m.average_cost,
                    "last_execution": m.last_execution.isoformat() if m.last_execution else None
                }
                for name, m in self.metrics.items()
            },
            "recent_alerts": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "severity": a.severity,
                    "chain": a.chain_name,
                    "message": a.message
                }
                for a in sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            "cost_breakdown": {
                name: m.total_cost
                for name, m in sorted(self.metrics.items(), key=lambda x: x[1].total_cost, reverse=True)[:10]
            }
        }

    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        dashboard_data = self.get_dashboard_data()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chain Monitor Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid; }}
        .alert-critical {{ background: #ffebee; border-color: #f44336; }}
        .alert-warning {{ background: #fff3e0; border-color: #ff9800; }}
        .alert-info {{ background: #e3f2fd; border-color: #2196f3; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: bold; }}
        .success-rate {{ color: #4caf50; }}
        .failure-rate {{ color: #f44336; }}
    </style>
    <script>
        setTimeout(() => location.reload(), 30000); // Auto-refresh every 30s
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chain Monitor Dashboard</h1>
            <p>Real-time execution monitoring and metrics</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['summary']['total_chains']}</div>
                <div class="metric-label">Active Chains</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['summary']['total_executions']}</div>
                <div class="metric-label">Total Executions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['summary']['overall_success_rate']*100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${dashboard_data['summary']['total_cost']:.2f}</div>
                <div class="metric-label">Total Cost</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['summary']['active_alerts']}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
        </div>

        <div class="section">
            <h2>Recent Alerts</h2>
            {''.join(f'''
                <div class="alert alert-{alert['severity']}">
                    <strong>{alert['severity'].upper()}</strong> - {alert['chain']} - {alert['message']}
                    <br><small>{alert['timestamp']}</small>
                </div>
            ''' for alert in dashboard_data['recent_alerts'][:5]) or '<p>No recent alerts</p>'}
        </div>

        <div class="section">
            <h2>Chain Performance</h2>
            <table>
                <tr>
                    <th>Chain</th>
                    <th>Executions</th>
                    <th>Success Rate</th>
                    <th>Avg Duration</th>
                    <th>Avg Cost</th>
                    <th>Last Execution</th>
                </tr>
                {''.join(f'''
                <tr>
                    <td>{name}</td>
                    <td>{data['executions']}</td>
                    <td class="success-rate">{data['success_rate']*100:.1f}%</td>
                    <td>{data['avg_duration']:.2f}s</td>
                    <td>${data['avg_cost']:.4f}</td>
                    <td>{data['last_execution'] or 'N/A'}</td>
                </tr>
                ''' for name, data in dashboard_data['chains'].items())}
            </table>
        </div>

        <div class="section">
            <h2>Cost Breakdown (Top 10)</h2>
            {''.join(f'''
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span><strong>{name}</strong></span>
                        <span>${cost:.4f}</span>
                    </div>
                    <div style="background: #e0e0e0; height: 20px; border-radius: 4px;">
                        <div style="background: #667eea; height: 100%; width: {min(100, cost/max(dashboard_data['cost_breakdown'].values() or [1])*100):.1f}%; border-radius: 4px;"></div>
                    </div>
                </div>
            ''' for name, cost in list(dashboard_data['cost_breakdown'].items())[:10])}
        </div>
    </div>
</body>
</html>"""
        return html

    def export_json(self) -> str:
        """Export all metrics as JSON"""
        return json.dumps(self.get_dashboard_data(), indent=2)

    def reset_metrics(self, chain_name: Optional[str] = None) -> None:
        """Reset metrics for a chain or all chains"""
        if chain_name:
            if chain_name in self.metrics:
                del self.metrics[chain_name]
        else:
            self.metrics.clear()
            self.alerts.clear()


# Global monitor instance
_global_monitor = None


def get_global_monitor() -> ChainMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ChainMonitor()
    return _global_monitor


__all__ = ['ChainMonitor', 'ChainMetrics', 'Alert', 'get_global_monitor']
