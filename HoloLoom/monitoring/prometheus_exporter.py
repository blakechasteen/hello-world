"""
HoloLoom Prometheus Exporter
=============================

Prometheus integration for HoloLoom performance monitoring.

Features:
- FastAPI middleware for automatic request tracking
- /metrics endpoint for Prometheus scraping
- Graceful degradation if prometheus_client unavailable
- Background system metrics collection

Created: 2025-11-16
Integration: Prometheus + Grafana
"""

import time
import asyncio
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from HoloLoom.monitoring.performance_metrics import (
    get_metrics_collector,
    PROMETHEUS_AVAILABLE
)

if PROMETHEUS_AVAILABLE:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


# ============================================================================
# Prometheus Metrics Middleware
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic Prometheus metrics collection.

    Tracks:
    - Request count (by endpoint, method, status)
    - Request latency (by endpoint)
    - Error count (by endpoint, error type)
    - System metrics (CPU, memory)
    """

    def __init__(
        self,
        app: ASGIApp,
        update_system_metrics: bool = True,
        system_metrics_interval: float = 5.0
    ):
        super().__init__(app)
        self.collector = get_metrics_collector()
        self.update_system_metrics = update_system_metrics
        self.system_metrics_interval = system_metrics_interval
        self._background_task: Optional[asyncio.Task] = None

        # Start background system metrics collection
        if self.update_system_metrics:
            asyncio.create_task(self._collect_system_metrics())

    async def dispatch(self, request: Request, call_next):
        """Process request and record metrics."""
        start_time = time.time()
        status_code = 500  # Default to error

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            # Record error
            self.collector.record_error(
                endpoint=request.url.path,
                error_type=type(e).__name__
            )
            raise
        finally:
            # Record request metrics
            duration = time.time() - start_time
            self.collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                status=status_code,
                duration_seconds=duration
            )

    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        while True:
            try:
                self.collector.update_system_metrics()
                await asyncio.sleep(self.system_metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore errors in background collection
                await asyncio.sleep(self.system_metrics_interval)


# ============================================================================
# Metrics Endpoint Handler
# ============================================================================

async def metrics_endpoint() -> Response:
    """
    Prometheus /metrics endpoint handler.

    Returns:
        Response with Prometheus metrics in text format

    Usage:
        @app.get("/metrics")
        async def metrics():
            return await metrics_endpoint()
    """
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="# Prometheus client not available\n"
                    "# Install with: pip install prometheus-client\n",
            media_type="text/plain"
        )

    # Update system metrics one final time before export
    collector = get_metrics_collector()
    collector.update_system_metrics()

    # Generate Prometheus-format metrics
    metrics_output = generate_latest()

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Alert Checking
# ============================================================================

ALERT_THRESHOLDS = {
    "p95_latency_ms": 1000,  # Alert if p95 > 1s
    "p99_latency_ms": 5000,  # Alert if p99 > 5s
    "error_rate": 0.05,      # Alert if error rate > 5%
    "cpu_percent": 80,       # Alert if CPU > 80%
    "memory_percent": 80,    # Alert if memory > 80%
    "memory_mb": 2048,       # Alert if memory > 2GB
    "cache_hit_rate_query": 0.5,      # Alert if query cache hit rate < 50%
    "cache_hit_rate_embedding": 0.6,  # Alert if embedding cache hit rate < 60%
}


async def check_alerts(custom_thresholds: Optional[dict] = None) -> list:
    """
    Check current metrics against alert thresholds.

    Args:
        custom_thresholds: Optional custom thresholds to override defaults

    Returns:
        List of alert dictionaries with structure:
        {
            "severity": "warning" | "critical",
            "metric": str,
            "value": float,
            "threshold": float,
            "message": str
        }
    """
    thresholds = ALERT_THRESHOLDS.copy()
    if custom_thresholds:
        thresholds.update(custom_thresholds)

    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()
    alerts = []

    # P95 latency check
    if snapshot.p95_latency_ms > thresholds["p95_latency_ms"]:
        alerts.append({
            "severity": "warning",
            "metric": "p95_latency",
            "value": snapshot.p95_latency_ms,
            "threshold": thresholds["p95_latency_ms"],
            "message": f"P95 latency ({snapshot.p95_latency_ms:.0f}ms) exceeds threshold ({thresholds['p95_latency_ms']:.0f}ms)"
        })

    # P99 latency check
    if snapshot.p99_latency_ms > thresholds["p99_latency_ms"]:
        alerts.append({
            "severity": "critical",
            "metric": "p99_latency",
            "value": snapshot.p99_latency_ms,
            "threshold": thresholds["p99_latency_ms"],
            "message": f"P99 latency ({snapshot.p99_latency_ms:.0f}ms) exceeds threshold ({thresholds['p99_latency_ms']:.0f}ms)"
        })

    # CPU check
    if snapshot.cpu_percent > thresholds["cpu_percent"]:
        alerts.append({
            "severity": "warning",
            "metric": "cpu_percent",
            "value": snapshot.cpu_percent,
            "threshold": thresholds["cpu_percent"],
            "message": f"CPU usage ({snapshot.cpu_percent:.1f}%) exceeds threshold ({thresholds['cpu_percent']}%)"
        })

    # Memory percent check
    if snapshot.memory_percent > thresholds["memory_percent"]:
        alerts.append({
            "severity": "warning",
            "metric": "memory_percent",
            "value": snapshot.memory_percent,
            "threshold": thresholds["memory_percent"],
            "message": f"Memory usage ({snapshot.memory_percent:.1f}%) exceeds threshold ({thresholds['memory_percent']}%)"
        })

    # Memory absolute check
    if snapshot.memory_mb > thresholds["memory_mb"]:
        alerts.append({
            "severity": "critical",
            "metric": "memory_mb",
            "value": snapshot.memory_mb,
            "threshold": thresholds["memory_mb"],
            "message": f"Memory usage ({snapshot.memory_mb:.0f}MB) exceeds threshold ({thresholds['memory_mb']}MB)"
        })

    # Cache hit rate checks
    for cache_type in ['query', 'embedding']:
        threshold_key = f"cache_hit_rate_{cache_type}"
        hit_rate = snapshot.cache_hit_rates.get(cache_type, 1.0)

        if hit_rate < thresholds[threshold_key]:
            alerts.append({
                "severity": "warning",
                "metric": f"cache_hit_rate_{cache_type}",
                "value": hit_rate,
                "threshold": thresholds[threshold_key],
                "message": f"{cache_type.capitalize()} cache hit rate ({hit_rate:.1%}) below threshold ({thresholds[threshold_key]:.1%})"
            })

    return alerts


# ============================================================================
# Alert Formatter
# ============================================================================

def format_alert_for_slack(alert: dict) -> dict:
    """
    Format alert for Slack webhook.

    Args:
        alert: Alert dictionary from check_alerts()

    Returns:
        Slack-compatible message dict
    """
    emoji = "🚨" if alert["severity"] == "critical" else "⚠️"
    color = "danger" if alert["severity"] == "critical" else "warning"

    return {
        "attachments": [
            {
                "color": color,
                "title": f"{emoji} HoloLoom Alert: {alert['metric']}",
                "text": alert["message"],
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert["severity"].upper(),
                        "short": True
                    },
                    {
                        "title": "Current Value",
                        "value": f"{alert['value']:.2f}",
                        "short": True
                    },
                    {
                        "title": "Threshold",
                        "value": f"{alert['threshold']:.2f}",
                        "short": True
                    }
                ],
                "footer": "HoloLoom Performance Monitoring",
                "ts": int(time.time())
            }
        ]
    }


def format_alert_for_email(alert: dict) -> dict:
    """
    Format alert for email notification.

    Args:
        alert: Alert dictionary from check_alerts()

    Returns:
        Email-compatible dict with subject and body
    """
    subject = f"[{alert['severity'].upper()}] HoloLoom Alert: {alert['metric']}"

    body = f"""
HoloLoom Performance Alert
==========================

Severity: {alert['severity'].upper()}
Metric: {alert['metric']}
Current Value: {alert['value']:.2f}
Threshold: {alert['threshold']:.2f}

Message:
{alert['message']}

Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

---
HoloLoom Performance Monitoring System
"""

    return {
        "subject": subject,
        "body": body.strip()
    }


# ============================================================================
# Helper Functions
# ============================================================================

def is_prometheus_available() -> bool:
    """Check if Prometheus client is available."""
    return PROMETHEUS_AVAILABLE


def get_prometheus_config_example() -> str:
    """
    Get example Prometheus scrape configuration.

    Returns:
        YAML configuration string
    """
    return """
# Prometheus scrape configuration for HoloLoom
# Add to prometheus.yml

scrape_configs:
  - job_name: 'hololoom-dashboard'
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'hololoom'
          environment: 'production'

  # Optional: Add alert rules
  - job_name: 'hololoom-alerts'
    scrape_interval: 30s
    metrics_path: '/api/v1/monitoring/alerts'
    static_configs:
      - targets: ['localhost:8000']
"""


def get_grafana_dashboard_json() -> dict:
    """
    Get Grafana dashboard JSON template.

    Returns:
        Grafana dashboard configuration dict
    """
    return {
        "dashboard": {
            "title": "HoloLoom Performance Dashboard",
            "tags": ["hololoom", "performance"],
            "timezone": "browser",
            "panels": [
                {
                    "title": "Request Rate",
                    "targets": [
                        {
                            "expr": "rate(hololoom_requests_total[5m])",
                            "legendFormat": "{{endpoint}}"
                        }
                    ],
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
                },
                {
                    "title": "Request Latency (P95)",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(hololoom_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{endpoint}}"
                        }
                    ],
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
                },
                {
                    "title": "System Resources",
                    "targets": [
                        {
                            "expr": "hololoom_system_cpu_percent",
                            "legendFormat": "CPU %"
                        },
                        {
                            "expr": "hololoom_system_memory_percent",
                            "legendFormat": "Memory %"
                        }
                    ],
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
                },
                {
                    "title": "Cache Hit Rate",
                    "targets": [
                        {
                            "expr": "hololoom_cache_hit_rate",
                            "legendFormat": "{{cache_type}}"
                        }
                    ],
                    "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
                }
            ]
        }
    }
