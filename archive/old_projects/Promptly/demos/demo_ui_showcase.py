#!/usr/bin/env python3
"""
Promptly UI Showcase - Complete UI Demo
========================================
Demonstrates both terminal and web interfaces.

Run with:
    python demos/demo_ui_showcase.py --mode [terminal|web|both]
"""

import sys
import os
import argparse
import threading
import time

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def demo_terminal_ui():
    """Launch terminal UI demo"""
    print("\n" + "="*60)
    print("PROMPTLY TERMINAL UI")
    print("="*60 + "\n")

    try:
        from promptly.ui.terminal_app import run_tui
        print("Launching Terminal UI...")
        print("Press 'q' to quit\n")
        time.sleep(1)
        run_tui()
    except ImportError as e:
        print(f"Terminal UI not available: {e}")
        print("\nInstall with: pip install textual")
    except Exception as e:
        print(f"Error: {e}")


def demo_web_ui(port=5000):
    """Launch web UI demo"""
    print("\n" + "="*60)
    print("PROMPTLY WEB DASHBOARD")
    print("="*60 + "\n")

    try:
        from promptly.ui.web_app import run_web_app
        print(f"Launching Web Dashboard on port {port}...")
        print(f"Open browser to: http://localhost:{port}")
        print("Press Ctrl+C to stop\n")
        time.sleep(1)
        run_web_app(port=port, debug=False)
    except ImportError as e:
        print(f"Web UI not available: {e}")
        print("\nInstall with: pip install flask flask-socketio")
    except Exception as e:
        print(f"Error: {e}")


def demo_both():
    """Run both UIs simultaneously"""
    print("\n" + "="*60)
    print("PROMPTLY DUAL UI MODE")
    print("="*60 + "\n")

    print("Starting both Terminal and Web interfaces...")
    print("- Terminal UI: Interactive TUI")
    print("- Web Dashboard: http://localhost:5000")
    print("\nPress Ctrl+C to stop both\n")

    # Start web in background thread
    web_thread = threading.Thread(target=demo_web_ui, args=(5000,), daemon=True)
    web_thread.start()

    # Give web time to start
    time.sleep(2)

    # Run terminal in foreground
    demo_terminal_ui()


def show_features():
    """Display feature showcase"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║             PROMPTLY UI FEATURES SHOWCASE                    ║
╚══════════════════════════════════════════════════════════════╝

🖥️  TERMINAL UI (Textual-based TUI)
────────────────────────────────────
✓ Interactive prompt composer
✓ Live execution with results display
✓ Real-time analytics dashboard
✓ Loop execution visualizer
✓ Skill library browser
✓ Cost tracking
✓ Keyboard shortcuts (Ctrl+E, Ctrl+C, etc.)
✓ Beautiful tables and panels
✓ Multi-tab interface

🌐 WEB DASHBOARD (Flask + SocketIO)
────────────────────────────────────
✓ Modern, responsive design
✓ Real-time updates via WebSockets
✓ Live prompt execution
✓ Interactive charts (Chart.js)
✓ Execution history
✓ Cost breakdown visualization
✓ Skill management
✓ Export capabilities
✓ Mobile-friendly

🎨 VISUALIZATION FEATURES
────────────────────────────────────
✓ Loop execution flow diagrams
✓ Token usage charts
✓ Cost trends over time
✓ Skill usage analytics
✓ Performance metrics
✓ Success/failure rates

⌨️  TERMINAL UI SHORTCUTS
────────────────────────────────────
Ctrl+E  : Execute prompt
Ctrl+C  : Clear inputs
Ctrl+S  : Save results
Ctrl+L  : Load skill
F1      : Help
Q       : Quit

🚀 WEB DASHBOARD FEATURES
────────────────────────────────────
• Live execution status
• Real-time cost tracking
• Historical analytics
• Skill library
• Export to JSON/CSV
• Dark/light themes (coming soon)

═══════════════════════════════════════════════════════════════

INSTALLATION REQUIREMENTS:
──────────────────────────
Terminal UI:  pip install textual rich
Web Dashboard: pip install flask flask-socketio

USAGE:
──────
Terminal only:  python demos/demo_ui_showcase.py --mode terminal
Web only:       python demos/demo_ui_showcase.py --mode web
Both:           python demos/demo_ui_showcase.py --mode both
Features:       python demos/demo_ui_showcase.py --features

═══════════════════════════════════════════════════════════════
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Promptly UI Showcase',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['terminal', 'web', 'both'],
        default='terminal',
        help='Which UI to launch (default: terminal)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for web dashboard (default: 5000)'
    )

    parser.add_argument(
        '--features',
        action='store_true',
        help='Show feature list instead of launching UI'
    )

    args = parser.parse_args()

    # Show features and exit
    if args.features:
        show_features()
        return

    # Launch requested UI(s)
    try:
        if args.mode == 'terminal':
            demo_terminal_ui()
        elif args.mode == 'web':
            demo_web_ui(args.port)
        elif args.mode == 'both':
            demo_both()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
