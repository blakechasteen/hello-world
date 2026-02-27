"""
Prometheus HTTP Endpoint - Expose metrics for scraping.

Provides an HTTP server that exposes Prometheus metrics at /metrics.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from hololoom.telemetry.metrics.prometheus import PrometheusRegistry, get_registry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS HTTP HANDLER
# ═══════════════════════════════════════════════════════════════════════════════


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    registry: Optional[PrometheusRegistry] = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404, "Not Found")

    def _serve_metrics(self) -> None:
        """Serve Prometheus metrics."""
        registry = self.registry or get_registry()
        metrics_output = registry.export()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(metrics_output)))
        self.end_headers()
        self.wfile.write(metrics_output.encode("utf-8"))

    def _serve_health(self) -> None:
        """Serve health check."""
        response = b'{"status": "healthy"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass  # Use our own logging if needed


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMETHEUS HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PrometheusHTTPServer:
    """HTTP server for exposing Prometheus metrics.

    Runs in a background thread and exposes metrics at /metrics.
    """

    host: str = "0.0.0.0"
    port: int = 9090
    registry: Optional[PrometheusRegistry] = None

    # Internal state
    _server: Optional[HTTPServer] = field(default=None, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._running:
            logger.warning("Metrics server already running")
            return

        # Set registry on handler class
        MetricsHandler.registry = self.registry

        try:
            self._server = HTTPServer((self.host, self.port), MetricsHandler)
            self._thread = threading.Thread(
                target=self._serve_forever,
                daemon=True,
                name="prometheus-metrics-server",
            )
            self._thread.start()
            self._running = True
            logger.info(f"Prometheus metrics server started at http://{self.host}:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def _serve_forever(self) -> None:
        """Serve requests until shutdown."""
        if self._server:
            self._server.serve_forever()

    def stop(self) -> None:
        """Stop the metrics server."""
        if not self._running:
            return

        if self._server:
            self._server.shutdown()
            self._server.server_close()

        if self._thread:
            self._thread.join(timeout=5.0)

        self._running = False
        logger.info("Prometheus metrics server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def metrics_url(self) -> str:
        """Get the metrics URL."""
        return f"http://{self.host}:{self.port}/metrics"


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_prometheus_server(
    port: int = 9090,
    registry: Optional[PrometheusRegistry] = None,
) -> PrometheusHTTPServer:
    """Create a Prometheus HTTP server."""
    return PrometheusHTTPServer(
        host="0.0.0.0",
        port=port,
        registry=registry,
    )


def start_metrics_server(
    port: int = 9090,
    registry: Optional[PrometheusRegistry] = None,
) -> PrometheusHTTPServer:
    """Create and start a Prometheus HTTP server."""
    server = create_prometheus_server(port, registry)
    server.start()
    return server


# ═══════════════════════════════════════════════════════════════════════════════
#  ASYNC WRAPPER (Optional)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_metrics_server_async(
    port: int = 9090,
    registry: Optional[PrometheusRegistry] = None,
) -> PrometheusHTTPServer:
    """Start metrics server and return immediately (async wrapper).

    The server runs in a background thread, so this is safe to call
    from async code.
    """
    server = start_metrics_server(port, registry)
    # Give server time to start
    await asyncio.sleep(0.1)
    return server
