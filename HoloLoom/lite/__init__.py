"""
HoloLoom Lite - Simplified HoloLoom for quick startup and minimal dependencies.

Usage:
    from HoloLoom import HoloLoomLite

    async with HoloLoomLite() as loom:
        await loom.experience("Thompson Sampling balances exploration")
        results = await loom.recall("What is Thompson Sampling?")

CLI:
    python -m HoloLoom.lite repl       # Simple REPL
    python -m HoloLoom.lite terminal   # Rich terminal
    python -m HoloLoom.lite web        # Web chat (localhost:8080)
    python -m HoloLoom.lite desktop    # Gradio desktop app

Agent Tools:
    python -m HoloLoom.lite.mcp_server     # MCP Server for Claude Desktop
    python -m HoloLoom.lite.openai_tools   # OpenAI function schemas

Date: December 2025
"""

from HoloLoom.lite.core import HoloLoomLite, LiteResult, SimpleLoom

# Agent tool exports (lazy loaded to avoid import errors if dependencies missing)
try:
    from HoloLoom.lite.openai_tools import (
        HOLOLOOM_LITE_TOOLS,
        execute_tool,
        get_tools_for_openai,
        get_tools_for_anthropic,
        get_tool_names,
        get_tool_descriptions
    )
    _OPENAI_TOOLS_AVAILABLE = True
except ImportError:
    _OPENAI_TOOLS_AVAILABLE = False

__all__ = [
    # Core
    'HoloLoomLite',
    'LiteResult',
    'SimpleLoom',
    # Agent tools (when available)
    'HOLOLOOM_LITE_TOOLS',
    'execute_tool',
    'get_tools_for_openai',
    'get_tools_for_anthropic',
    'get_tool_names',
    'get_tool_descriptions',
]


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for HoloLoom Lite UIs."""
    import sys

    if len(sys.argv) < 2:
        print("HoloLoom Lite - Choose a UI mode:")
        print()
        print("  python -m HoloLoom.lite repl      Simple REPL")
        print("  python -m HoloLoom.lite terminal  Rich terminal")
        print("  python -m HoloLoom.lite web       Web chat (localhost:8080)")
        print("  python -m HoloLoom.lite desktop   Gradio desktop app")
        print()
        sys.exit(0)

    mode = sys.argv[1].lower()

    try:
        from HoloLoom.lite.ui import launch
        launch(mode=mode)
    except ImportError as e:
        print(f"UI module not available: {e}")
        print("Install required dependencies:")
        print("  pip install rich     # For terminal mode")
        print("  pip install fastapi uvicorn  # For web mode")
        print("  pip install gradio   # For desktop mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
