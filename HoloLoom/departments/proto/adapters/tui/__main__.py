"""Proto TUI Module Entry Point.

Enables running the TUI with:
    python -m HoloLoom.departments.proto.adapters.tui

Options:
    --engine        Connect to ProtoEngine for real AI responses
    --theme THEME   Set theme (ocean, kanagawa, cyberpunk, minimal)
    --session ID    Resume a previous session
    --list-sessions List available sessions

Status: Phase 3 TUI (2025-12-05)
"""

from .app import main

if __name__ == "__main__":
    main()
