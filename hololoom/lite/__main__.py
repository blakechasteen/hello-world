"""
CLI entry point for HoloLoom Lite.

Usage:
    python -m HoloLoom.lite repl       # Simple REPL
    python -m HoloLoom.lite terminal   # Rich terminal
    python -m HoloLoom.lite web        # Web chat (localhost:8080)
    python -m HoloLoom.lite desktop    # Gradio desktop app
"""

from hololoom.lite import main

if __name__ == "__main__":
    main()
