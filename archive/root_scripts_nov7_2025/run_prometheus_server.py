#!/usr/bin/env python3
"""
Prometheus Metrics Server Launcher
===================================
Convenience wrapper to run the Prometheus metrics server with correct PYTHONPATH.

Usage:
    python run_prometheus_server.py [--port PORT] [--host HOST]

Examples:
    python run_prometheus_server.py
    python run_prometheus_server.py --port 8080
    python run_prometheus_server.py --host 127.0.0.1 --port 9090
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the server
from HoloLoom.alignment.prometheus_server import start_metrics_server

if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "PROMETHEUS METRICS SERVER")
    print("="*80)
    print()
    print("Starting Prometheus metrics server for HoloLoom Alignment Framework")
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
