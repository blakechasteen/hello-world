#!/usr/bin/env python3
"""
Quick Launcher for Agentic Dashboard

Just run:
    python start_agentic_dashboard.py

Then open: http://localhost:8001
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    server_path = Path(__file__).parent / "HoloLoom" / "web_dashboard" / "agentic_server.py"

    print("\n" + "="*70)
    print("  >> Launching HoloLoom Agentic Dashboard")
    print("="*70)
    print("\n  The dashboard will open at: http://localhost:8001")
    print("\n  Features:")
    print("    - 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)")
    print("    - LLM-activated intelligent search")
    print("    - Persistent memory (Neo4j + Qdrant)")
    print("    - Real-time provenance tracking")
    print("\n  Press Ctrl+C to stop the server")
    print("="*70 + "\n")

    # Run the server
    subprocess.run([sys.executable, str(server_path)])
