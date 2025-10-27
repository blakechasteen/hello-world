#!/usr/bin/env python3
"""
Quick launcher for Promptly Terminal UI with WeavingShuttle
============================================================
Just run this. It works. I promise.
"""

import sys
from pathlib import Path

# Add paths - be explicit
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# Direct import to avoid __init__.py issues
from promptly.ui.terminal_app_wired import HoloLoomTerminalApp

if __name__ == "__main__":
    print("\n🚀 Launching HoloLoom Terminal UI...")
    print("   Controls:")
    print("   - Ctrl+W: Weave query")
    print("   - Ctrl+M: Add memory")
    print("   - Ctrl+S: Search")
    print("   - Ctrl+Q: Quit")
    print("\n")

    app = HoloLoomTerminalApp()
    app.run()
