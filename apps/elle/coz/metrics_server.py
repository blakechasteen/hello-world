#!/usr/bin/env python3
"""
COZ Daily Brief Metrics Server

Exposes health check metrics for Prometheus scraping.

Usage:
    python metrics_server.py [--port PORT] [--interval SECONDS]

    --port PORT: Port to listen on (default: 9090)
    --interval SECONDS: Health check interval (default: 60)

Created: 2025-11-22
Part of: Production Deployment (Monitoring Setup)
"""

import sys
import time
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.health_check import HealthCheck


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics"""

    # Shared state (updated by background thread)
    latest_metrics = ""
    metrics_lock = Lock()

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            # Serve Prometheus metrics
            with self.metrics_lock:
                metrics = self.latest_metrics

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics.encode('utf-8'))

        elif self.path == '/health':
            # Serve health check (JSON)
            health_check = HealthCheck()
            health = health_check.run_all_checks()

            import json
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health, indent=2).encode('utf-8'))

        elif self.path == '/':
            # Serve index page
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>COZ Automation Metrics</title>
    <style>
        body {
            font-family: monospace;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 {
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        .endpoint {
            margin: 20px 0;
            padding: 10px;
            background: #f5f5f5;
            border-left: 3px solid #007bff;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        code {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>COZ Daily Brief - Metrics Server</h1>

    <div class="endpoint">
        <h3><a href="/metrics">/metrics</a></h3>
        <p>Prometheus metrics (text format)</p>
        <p><code>Content-Type: text/plain; version=0.0.4</code></p>
    </div>

    <div class="endpoint">
        <h3><a href="/health">/health</a></h3>
        <p>Health check (JSON format)</p>
        <p><code>Content-Type: application/json</code></p>
    </div>

    <p style="margin-top: 40px; color: #6c757d;">
        Server running on port {port}<br>
        Health check interval: {interval} seconds<br>
        Last update: {timestamp}
    </p>
</body>
</html>
""".format(
                port=self.server.server_port,
                interval=getattr(self.server, 'check_interval', 60),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        else:
            # 404 Not Found
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'404 Not Found')

    def log_message(self, format, *args):
        """Override to suppress request logging (too noisy)"""
        # Comment out to enable request logging
        pass


def update_metrics_loop(interval: int):
    """
    Background thread that updates metrics periodically

    Args:
        interval: Update interval in seconds
    """
    health_check = HealthCheck()

    while True:
        try:
            # Run health check and export Prometheus metrics
            metrics = health_check.export_prometheus()

            # Update shared state
            with MetricsHandler.metrics_lock:
                MetricsHandler.latest_metrics = metrics

            print(f"[{time.strftime('%H:%M:%S')}] Metrics updated")

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error updating metrics: {e}")

        # Sleep for interval
        time.sleep(interval)


def main():
    """Run metrics server"""
    parser = argparse.ArgumentParser(
        description="COZ Daily Brief Metrics Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--port',
        type=int,
        default=9090,
        help='Port to listen on (default: 9090)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Health check interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )

    args = parser.parse_args()

    # Initial metrics update
    print("Performing initial health check...")
    health_check = HealthCheck()
    metrics = health_check.export_prometheus()
    with MetricsHandler.metrics_lock:
        MetricsHandler.latest_metrics = metrics
    print("Initial health check complete")

    # Start background metrics update thread
    print(f"Starting background metrics update (every {args.interval}s)...")
    update_thread = Thread(
        target=update_metrics_loop,
        args=(args.interval,),
        daemon=True
    )
    update_thread.start()

    # Start HTTP server
    server = HTTPServer((args.host, args.port), MetricsHandler)
    server.check_interval = args.interval

    print("=" * 70)
    print("COZ Daily Brief Metrics Server")
    print("=" * 70)
    print(f"Listening on: http://{args.host}:{args.port}")
    print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")
    print(f"Health endpoint: http://{args.host}:{args.port}/health")
    print(f"Update interval: {args.interval}s")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
