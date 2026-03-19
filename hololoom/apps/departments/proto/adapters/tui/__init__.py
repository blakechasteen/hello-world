"""Proto TUI Adapter - Phase 3 Complete.

Rich terminal user interface for Proto code agent with semantic wave animations.

Features:
    - Interactive REPL with syntax highlighting
    - Semantic wave animations at top/bottom (communicate system state)
    - Real-time streaming output display
    - Context file management sidebar
    - Session persistence and history
    - Theme support (Ocean, Kanagawa, Cyberpunk, Minimal)
    - Keyboard shortcuts and mode switching

Usage:
    python -m HoloLoom.apps.departments.proto.adapters.tui

    With engine:
        python -m HoloLoom.apps.departments.proto.adapters.tui --engine

    With theme:
        python -m HoloLoom.apps.departments.proto.adapters.tui --theme cyberpunk

Status: Phase 3 TUI (2025-12-05)
"""

# Wave animation system
# Main application
from .app import (
    ProtoTUI,
    WaveFooter,
    WaveHeader,
    main,
)

# Panels (from panels module for more detailed components)
from .panels import (
    CommandHistory,
    CommandInput,
    # Context panel
    ContextPanel,
    FileInfo,
    FileWidget,
    # Input panel
    InputPanel,
    Message,
    MessageType,
    MessageWidget,
    # Response panel
    ResponsePanel,
)

# Session management
from .session import (
    SessionContext,
    SessionData,
    SessionManager,
    SessionMessage,
    create_session_manager,
)
from .wave_animation import (
    WAVE_CHARS,
    WAVE_STATE_COLORS,
    SemanticWaveController,
    WaveConfig,
    WaveGenerator,
    WaveProcess,
    WaveState,
    WaveTheme,
    create_wave_system,
)

__all__ = [
    # Wave animation
    "WaveGenerator",
    "WaveConfig",
    "WaveTheme",
    "WaveState",
    "WaveProcess",
    "SemanticWaveController",
    "WAVE_CHARS",
    "WAVE_STATE_COLORS",
    "create_wave_system",
    # Main app
    "ProtoTUI",
    "WaveHeader",
    "WaveFooter",
    "main",
    # Panels
    "InputPanel",
    "CommandInput",
    "CommandHistory",
    "ResponsePanel",
    "MessageType",
    "Message",
    "MessageWidget",
    "ContextPanel",
    "FileInfo",
    "FileWidget",
    # Session
    "SessionManager",
    "SessionData",
    "SessionMessage",
    "SessionContext",
    "create_session_manager",
]
