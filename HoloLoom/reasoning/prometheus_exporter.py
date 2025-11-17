#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prometheus Metrics Exporter
=============================
Export reasoning metrics in Prometheus format.

Phase 4: Advanced Observability

Author: Claude Code
Date: 2025-11-17
"""

import logging
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from HoloLoom.reasoning.metrics import get_reasoning_metrics

logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Format Exporter
# ============================================================================

class PrometheusExporter:
    """
    Export metrics in Prometheus text format.

    Converts our internal metrics to Prometheus exposition format.
    """

    def __init__(self):
        self.metrics = get_reasoning_metrics()
        self.logger = logging.getLogger(f"{__name__}.PrometheusExporter")

    def export_counter(self, name: str, help_text: str, value: int, labels: Optional[Dict[str, int]] = None) -> str:
        """
        Export counter metric in Prometheus format.

        Format:
            # HELP metric_name Help text
            # TYPE metric_name counter
            metric_name{label="value"} 123
        """
        lines = []

        # HELP
        lines.append(f"# HELP {name} {help_text}")

        # TYPE
        lines.append(f"# TYPE {name} counter")

        # Metrics
        if labels:
            for label_str, count in labels.items():
                lines.append(f"{name}{{{label_str}}} {count}")
        else:
            lines.append(f"{name} {value}")

        return "\n".join(lines)

    def export_histogram(self, name: str, help_text: str, stats: Dict[str, float], buckets: Optional[Dict[float, int]] = None) -> str:
        """
        Export histogram metric in Prometheus format.

        Format:
            # HELP metric_name Help text
            # TYPE metric_name histogram
            metric_name_bucket{le="10"} 5
            metric_name_bucket{le="25"} 12
            metric_name_bucket{le="+Inf"} 15
            metric_name_sum 1234.5
            metric_name_count 15
        """
        lines = []

        # HELP
        lines.append(f"# HELP {name} {help_text}")

        # TYPE
        lines.append(f"# TYPE {name} histogram")

        # Buckets (if available)
        if buckets:
            cumulative = 0
            for bucket, count in sorted(buckets.items()):
                cumulative += count
                bucket_str = "+Inf" if bucket == float('inf') else str(bucket)
                lines.append(f'{name}_bucket{{le="{bucket_str}"}} {cumulative}')

        # Summary stats
        lines.append(f"{name}_sum {stats.get('sum', 0)}")
        lines.append(f"{name}_count {stats.get('count', 0)}")

        return "\n".join(lines)

    def export_all(self) -> str:
        """
        Export all metrics in Prometheus format.

        Returns:
            Complete Prometheus exposition format text
        """
        all_metrics = self.metrics.registry.get_all_metrics()
        lines = []

        for metric_name, metric_data in all_metrics.items():
            metric_type = metric_data["type"]

            if metric_type == "counter":
                lines.append(self.export_counter(
                    name=metric_name,
                    help_text=metric_data["help"],
                    value=metric_data["value"],
                    labels=metric_data.get("labels")
                ))

            elif metric_type == "histogram":
                lines.append(self.export_histogram(
                    name=metric_name,
                    help_text=metric_data["help"],
                    stats=metric_data["stats"],
                    buckets=metric_data.get("buckets")
                ))

            lines.append("")  # Blank line between metrics

        return "\n".join(lines)


# ============================================================================
# HTTP Server for Metrics Endpoint
# ============================================================================

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint."""

    def do_GET(self):
        """Handle GET request."""
        if self.path == "/metrics":
            # Export metrics
            exporter = PrometheusExporter()
            metrics_text = exporter.export_all()

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(metrics_text.encode('utf-8'))

        elif self.path == "/health":
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        else:
            # 404
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Override to use logging instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")


class PrometheusHTTPServer:
    """
    HTTP server for Prometheus metrics endpoint.

    Runs in background thread, serves metrics on /metrics endpoint.
    """

    def __init__(self, port: int = 9090, host: str = "localhost"):
        self.port = port
        self.host = host
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.PrometheusHTTPServer")

    def start(self):
        """Start HTTP server in background thread."""
        if self.server is not None:
            self.logger.warning("Server already running")
            return

        # Create server
        self.server = HTTPServer((self.host, self.port), MetricsHandler)

        # Start in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        self.logger.info(f"Prometheus metrics server started on http://{self.host}:{self.port}/metrics")

    def stop(self):
        """Stop HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            self.thread = None
            self.logger.info("Prometheus metrics server stopped")


# ============================================================================
# Convenience Functions
# ============================================================================

_global_server: Optional[PrometheusHTTPServer] = None


def start_metrics_server(port: int = 9090, host: str = "localhost"):
    """
    Start global Prometheus metrics server.

    Args:
        port: HTTP port (default: 9090)
        host: Host to bind to (default: localhost)

    Usage:
        from HoloLoom.reasoning.prometheus_exporter import start_metrics_server

        # Start metrics server
        start_metrics_server(port=9090)

        # Metrics available at http://localhost:9090/metrics
        # Health check at http://localhost:9090/health

        # Prometheus scrape config:
        # scrape_configs:
        #   - job_name: 'hololoom-reasoning'
        #     static_configs:
        #       - targets: ['localhost:9090']
    """
    global _global_server

    if _global_server is not None:
        logger.warning("Metrics server already running")
        return

    _global_server = PrometheusHTTPServer(port=port, host=host)
    _global_server.start()


def stop_metrics_server():
    """Stop global Prometheus metrics server."""
    global _global_server

    if _global_server:
        _global_server.stop()
        _global_server = None


def export_metrics_to_file(file_path: str):
    """
    Export current metrics to file in Prometheus format.

    Useful for testing or batch export.

    Args:
        file_path: Output file path

    Usage:
        export_metrics_to_file("./metrics.prom")
    """
    exporter = PrometheusExporter()
    metrics_text = exporter.export_all()

    with open(file_path, 'w') as f:
        f.write(metrics_text)

    logger.info(f"Metrics exported to {file_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import time

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Start metrics server
    start_metrics_server(port=9090)

    print("="*70)
    print("Prometheus Metrics Server Running")
    print("="*70)
    print()
    print("Endpoints:")
    print("  Metrics: http://localhost:9090/metrics")
    print("  Health:  http://localhost:9090/health")
    print()
    print("Prometheus scrape config:")
    print("  scrape_configs:")
    print("    - job_name: 'hololoom-reasoning'")
    print("      static_configs:")
    print("        - targets: ['localhost:9090']")
    print()
    print("Press Ctrl+C to stop...")
    print("="*70)

    try:
        # Simulate some metrics
        metrics = get_reasoning_metrics()
        for i in range(10):
            metrics.track_operation(
                mode="standard",
                duration_ms=150.0 + (i * 10),
                confidence=0.85,
                chain_length=5,
                success=True
            )
            time.sleep(1)

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping server...")
        stop_metrics_server()
        print("Stopped.")
