#!/usr/bin/env python3
"""
Prometheus Metrics Server for HoloLoom
======================================
Standalone HTTP server exposing Prometheus metrics on port 8001.

This server runs independently and exposes metrics that are tracked
by the WeavingOrchestrator, ChronoTrigger, and other components.

Usage:
    # Run directly
    python -m HoloLoom.performance.metrics_server

    # Or import and start
    from HoloLoom.performance.metrics_server import run_metrics_server
    run_metrics_server(port=8001)

Metrics Endpoint:
    http://localhost:8001/metrics

Monitored Metrics:
    - hololoom_query_duration_seconds: Query latency histogram
    - hololoom_queries_total: Total queries by pattern/complexity
    - hololoom_breathing_cycles_total: Breathing cycles by phase
    - hololoom_cache_hits_total: Cache hit counter
    - hololoom_cache_misses_total: Cache miss counter
    - hololoom_pattern_selections_total: Pattern selections
    - hololoom_backend_status: Backend health gauges
    - hololoom_errors_total: Error counter by type/stage
"""

import logging
import time
import signal
import sys
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import metrics module
try:
    from HoloLoom.performance.prometheus_metrics import metrics, start_metrics_server, PROMETHEUS_AVAILABLE
    if not PROMETHEUS_AVAILABLE:
        logger.error("prometheus_client not installed")
        logger.error("Install with: pip install prometheus-client")
        sys.exit(1)
except ImportError as e:
    logger.error(f"Failed to import prometheus_metrics: {e}")
    logger.error("Make sure you're in the repository root with PYTHONPATH set")
    sys.exit(1)

# Global server instance
_server_running = False


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("\nShutdown signal received, stopping metrics server...")
    global _server_running
    _server_running = False
    sys.exit(0)


def run_metrics_server(port: int = 8001, host: str = '0.0.0.0'):
    """
    Run the Prometheus metrics server.

    Args:
        port: Port to listen on (default: 8001)
        host: Host to bind to (default: 0.0.0.0 for all interfaces)

    The server will run until interrupted (Ctrl+C) or killed.
    """
    global _server_running

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("="*80)
    logger.info("HoloLoom Prometheus Metrics Server")
    logger.info("="*80)
    logger.info(f"Starting metrics server on {host}:{port}...")

    try:
        # Start the HTTP server
        start_metrics_server(port=port)
        _server_running = True

        logger.info(f"✓ Metrics server started successfully!")
        logger.info(f"✓ Metrics endpoint: http://localhost:{port}/metrics")
        logger.info("")
        logger.info("Available Metrics:")
        logger.info("  - hololoom_query_duration_seconds")
        logger.info("  - hololoom_queries_total")
        logger.info("  - hololoom_breathing_cycles_total")
        logger.info("  - hololoom_cache_hits_total")
        logger.info("  - hololoom_cache_misses_total")
        logger.info("  - hololoom_pattern_selections_total")
        logger.info("  - hololoom_backend_status")
        logger.info("  - hololoom_errors_total")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("="*80)

        # Keep server running
        while _server_running:
            time.sleep(1)

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"✗ Port {port} is already in use")
            logger.error(f"  Either stop the other process or use a different port")
            logger.error(f"  Example: python -m HoloLoom.performance.metrics_server --port 8002")
        else:
            logger.error(f"✗ Failed to start server: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='HoloLoom Prometheus Metrics Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on default port 8001
  python -m HoloLoom.performance.metrics_server

  # Run on custom port
  python -m HoloLoom.performance.metrics_server --port 9090

  # Bind to specific host
  python -m HoloLoom.performance.metrics_server --host 127.0.0.1 --port 8001

For Docker deployment:
  # The metrics endpoint is automatically exposed by docker-compose.production.yml
  # on port 8001 of the hololoom container

For Prometheus scraping:
  # Add to prometheus.yml:
  scrape_configs:
    - job_name: 'hololoom'
      static_configs:
        - targets: ['localhost:8001']
        """
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port to listen on (default: 8001)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )

    args = parser.parse_args()

    # Run the server
    run_metrics_server(port=args.port, host=args.host)


if __name__ == '__main__':
    main()
