"""
Proto Interface Adapters
=========================

User-facing interfaces for Proto code agent:

1. **CLI Adapter** (Primary)
   - Command-line interface via proto.py entry point
   - Commands: ask, review, repl, chat, suggest
   - Real-time streaming output support
   - Structured result formatting

2. **TUI Adapter** (Phase 3) [COMPLETE]
   - Rich terminal UI with Textual framework
   - Semantic wave animations (communicate system state)
   - Interactive REPL with syntax highlighting
   - Real-time session management
   - File browsing and context management
   - Theme support (Ocean, Kanagawa, Cyberpunk, Minimal)

3. **Matrix Adapter** (Phase 2) [COMPLETE]
   - Matrix/Element ChatOps integration
   - Bot commands: !proto <query>, !proto review, !proto test, etc.
   - Room-based persistent sessions
   - Async message handling with audit trail
   - Integration with Tier 2 abilities

All adapters route through ProtoEngine.process() for unified
request handling and response generation.

Status: CLI + Matrix + TUI production ready (2025-12-05)
"""

from hololoom.apps.departments.proto.adapters.cli import cli
from hololoom.apps.departments.proto.adapters.matrix import (
    ProtoMatrixHandlers,
    MatrixMessage,
    ProtoSession,
    create_proto_handlers,
)
from hololoom.apps.departments.proto.adapters.tui import (
    # Main app
    ProtoTUI,
    WaveHeader,
    WaveFooter,
    main as tui_main,
    # Wave animation
    WaveGenerator,
    WaveConfig,
    WaveTheme,
    WaveState,
    SemanticWaveController,
    create_wave_system,
    # Panels
    InputPanel,
    ResponsePanel,
    ContextPanel,
    # Session
    SessionManager,
    create_session_manager,
)

__all__ = [
    # CLI adapter
    "cli",
    # Matrix adapter
    "ProtoMatrixHandlers",
    "MatrixMessage",
    "ProtoSession",
    "create_proto_handlers",
    # TUI adapter
    "ProtoTUI",
    "WaveHeader",
    "WaveFooter",
    "tui_main",
    "WaveGenerator",
    "WaveConfig",
    "WaveTheme",
    "WaveState",
    "SemanticWaveController",
    "create_wave_system",
    "InputPanel",
    "ResponsePanel",
    "ContextPanel",
    "SessionManager",
    "create_session_manager",
]
