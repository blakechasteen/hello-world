"""
HoloLoom Lite UI - Unified launcher for all UI modes.

Available Modes:
    - repl: Simple REPL (no dependencies)
    - terminal: Rich terminal (requires: rich)
    - web: Web chat (requires: fastapi, uvicorn)
    - desktop: Gradio desktop app (requires: gradio)

Usage:
    from hololoom.lite.ui import launch
    launch(mode="repl")

    # Or directly:
    from hololoom.lite.ui.repl import start_repl
    start_repl()

Date: December 2025
"""

from typing import Any, Dict, Optional


def launch(mode: str = "repl", **kwargs) -> None:
    """
    Launch HoloLoom Lite UI.

    Args:
        mode: UI mode - "repl", "terminal", "web", or "desktop"
        **kwargs: Additional arguments passed to the launcher

    Example:
        >>> launch("repl")  # Start simple REPL
        >>> launch("web", port=8080)  # Start web chat
        >>> launch("desktop", share=True)  # Start Gradio with sharing
    """
    mode = mode.lower()

    launchers = {
        "repl": _launch_repl,
        "terminal": _launch_terminal,
        "web": _launch_web,
        "desktop": _launch_desktop,
    }

    if mode not in launchers:
        print(f"Unknown mode: {mode}")
        print(f"Available modes: {', '.join(launchers.keys())}")
        return

    launchers[mode](**kwargs)


def _launch_repl(**kwargs) -> None:
    """Launch simple REPL."""
    from hololoom.lite.ui.repl import start_repl
    start_repl(**kwargs)


def _launch_terminal(**kwargs) -> None:
    """Launch rich terminal."""
    try:
        from hololoom.lite.ui.terminal import start_terminal
        start_terminal(**kwargs)
    except ImportError:
        print("Rich terminal requires 'rich' package.")
        print("Install with: pip install rich")
        print()
        print("Falling back to simple REPL...")
        _launch_repl(**kwargs)


def _launch_web(**kwargs) -> None:
    """Launch web chat."""
    try:
        from hololoom.lite.ui.web import start_web
        start_web(**kwargs)
    except ImportError as e:
        print("Web chat requires 'fastapi' and 'uvicorn' packages.")
        print("Install with: pip install fastapi uvicorn")
        print(f"Error: {e}")


def _launch_desktop(**kwargs) -> None:
    """Launch Gradio desktop app."""
    try:
        from hololoom.lite.ui.desktop import start_desktop
        start_desktop(**kwargs)
    except ImportError as e:
        print("Desktop app requires 'gradio' package.")
        print("Install with: pip install gradio")
        print(f"Error: {e}")


__all__ = ['launch']
