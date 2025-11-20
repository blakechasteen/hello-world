"""
Prometheus Metrics Server for Alignment Framework

Exposes alignment metrics in Prometheus format for scraping by Prometheus server.

Features:
- Standard /metrics endpoint
- Health check endpoint
- Automatic metric updates from global monitor
- Optional basic auth for security

Usage:
    # Simple usage
    python HoloLoom/alignment/prometheus_server.py

    # Or programmatically
    from HoloLoom.alignment.prometheus_server import start_metrics_server

    start_metrics_server(port=9090, host='0.0.0.0')

Configuration:
    Set via environment variables:
    - PROMETHEUS_PORT: Port to listen on (default: 9090)
    - PROMETHEUS_HOST: Host to bind to (default: 0.0.0.0)
    - PROMETHEUS_AUTH_USER: Basic auth username (optional)
    - PROMETHEUS_AUTH_PASS: Basic auth password (optional)
"""

import os
import logging
from typing import Optional
from functools import wraps
from flask import Flask, Response, request, jsonify

from HoloLoom.alignment.monitoring import get_global_monitor, AlignmentMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)


def check_auth(username: str, password: str) -> bool:
    """Check if username/password is valid."""
    auth_user = os.getenv('PROMETHEUS_AUTH_USER')
    auth_pass = os.getenv('PROMETHEUS_AUTH_PASS')

    # If auth not configured, allow all
    if not auth_user or not auth_pass:
        return True

    return username == auth_user and password == auth_pass


def requires_auth(f):
    """Decorator to require basic auth if configured."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization

        # Check if auth is configured
        if os.getenv('PROMETHEUS_AUTH_USER') and os.getenv('PROMETHEUS_AUTH_PASS'):
            if not auth or not check_auth(auth.username, auth.password):
                return Response(
                    'Authentication required',
                    401,
                    {'WWW-Authenticate': 'Basic realm="Prometheus Metrics"'}
                )

        return f(*args, **kwargs)

    return decorated


@app.route('/')
def index():
    """Root endpoint - redirect to metrics."""
    return jsonify({
        "service": "HoloLoom Alignment Metrics",
        "version": "1.0.0",
        "endpoints": {
            "/metrics": "Prometheus metrics (text format)",
            "/health": "Health check",
            "/stats": "JSON stats (human-readable)"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        monitor = get_global_monitor()
        stats = monitor.get_all_stats()

        return jsonify({
            "status": "healthy",
            "components": len(stats),
            "total_samples": sum(s["count"] for s in stats.values()),
            "alerts": len(monitor.alerts)
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/metrics')
@requires_auth
def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format:
    - alignment_latency_p50{component="..."}
    - alignment_latency_p95{component="..."}
    - alignment_latency_p99{component="..."}
    - alignment_samples_total{component="..."}
    - alignment_alerts_total{level="..."}
    """
    try:
        monitor = get_global_monitor()
        prometheus_text = monitor.export_prometheus()

        return Response(
            prometheus_text,
            mimetype='text/plain; version=0.0.4'
        )
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        return Response(
            f"# ERROR: {str(e)}\n",
            mimetype='text/plain',
            status=500
        )


@app.route('/stats')
@requires_auth
def stats():
    """JSON stats endpoint (human-readable)."""
    try:
        monitor = get_global_monitor()
        stats = monitor.get_all_stats()

        return jsonify({
            "components": stats,
            "alerts": {
                "total": len(monitor.alerts),
                "recent": [
                    {
                        "level": a.level.value,
                        "component": a.component,
                        "message": a.message,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in monitor.alerts[-5:]  # Last 5 alerts
                ]
            }
        })
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({"error": str(e)}), 500


def start_metrics_server(
    port: Optional[int] = None,
    host: Optional[str] = None,
    monitor: Optional[AlignmentMonitor] = None,
    debug: bool = False
):
    """
    Start Prometheus metrics server.

    Args:
        port: Port to listen on (default: from env PROMETHEUS_PORT or 9090)
        host: Host to bind to (default: from env PROMETHEUS_HOST or 0.0.0.0)
        monitor: AlignmentMonitor instance (default: global monitor)
        debug: Enable Flask debug mode

    Environment Variables:
        PROMETHEUS_PORT: Port to listen on
        PROMETHEUS_HOST: Host to bind to
        PROMETHEUS_AUTH_USER: Basic auth username (optional)
        PROMETHEUS_AUTH_PASS: Basic auth password (optional)
    """
    # Get config from env or params
    port = port or int(os.getenv('PROMETHEUS_PORT', '9090'))
    host = host or os.getenv('PROMETHEUS_HOST', '0.0.0.0')

    # Set global monitor if provided
    if monitor:
        from HoloLoom.alignment.monitoring import set_global_monitor
        set_global_monitor(monitor)

    # Log startup
    logger.info(f"Starting Prometheus metrics server on {host}:{port}")

    if os.getenv('PROMETHEUS_AUTH_USER'):
        logger.info("Basic authentication enabled")

    # Start server
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    """
    Run standalone metrics server.

    Usage:
        python HoloLoom/alignment/prometheus_server.py

    With authentication:
        export PROMETHEUS_AUTH_USER=admin
        export PROMETHEUS_AUTH_PASS=secret
        python HoloLoom/alignment/prometheus_server.py

    Custom port:
        export PROMETHEUS_PORT=8080
        python HoloLoom/alignment/prometheus_server.py
    """
    import sys

    # Add parent to path for imports
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("="*80)
    print("PROMETHEUS METRICS SERVER")
    print("="*80)
    print()
    print("Starting server...")
    print()
    print("Endpoints:")
    print(f"  http://localhost:{os.getenv('PROMETHEUS_PORT', '9090')}/metrics")
    print(f"  http://localhost:{os.getenv('PROMETHEUS_PORT', '9090')}/health")
    print(f"  http://localhost:{os.getenv('PROMETHEUS_PORT', '9090')}/stats")
    print()
    print("Press Ctrl+C to stop")
    print("="*80)
    print()

    start_metrics_server()
