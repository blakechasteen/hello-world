#!/usr/bin/env python3
"""
Elle Core Dashboard Demo

Starts the FastAPI server and opens the dashboard in a browser.

Usage:
    python demos/demo_dashboard.py

Features:
- Starts FastAPI server on port 8000
- Opens browser to main dashboard
- Displays real-time operational intelligence
- Ctrl+C to stop

Created: 2025-11-15
Author: Blake Chasteen (Agent C)
"""

import asyncio
import webbrowser
import time
import subprocess
import sys
from pathlib import Path

# Add elle to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run the dashboard demo"""
    print("\n" + "="*70)
    print("ELLE CORE DASHBOARD DEMO")
    print("="*70 + "\n")

    print("Starting FastAPI server on http://localhost:8000")
    print("\nAvailable dashboards:")
    print("  • Main Dashboard:    http://localhost:8000/")
    print("  • Budget Tracking:   http://localhost:8000/budget")
    print("  • Product ROI:       http://localhost:8000/products")
    print("  • Weekly Planner:    http://localhost:8000/planner")
    print("  • Task Tracker:      http://localhost:8000/tracker")
    print("\nPress Ctrl+C to stop the server\n")

    # Start server
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:8000/')

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

        # Start FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "elle.dashboard.api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("Dashboard demo complete!\n")


if __name__ == "__main__":
    main()
