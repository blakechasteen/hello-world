# HoloLoom/ralph/__init__.py
"""
Ralph Loop Engine - Context-aware iterative execution with state preservation.

Named after the viral "Ralph Wiggum" technique by Geoff Huntley (2025):
"Iteration beats perfection" - Reset context, preserve progress.

Philosophy:
- Context resets aren't failures, they're features
- State lives in files/memory, not in the context window
- Each iteration builds on the last through persistent storage
- Embrace the loop, weaponize the reset

Usage:
    from HoloLoom.ralph import RalphEngine, RalphConfig, LoopTemplate

    # Create engine
    config = RalphConfig.default()
    engine = RalphEngine(config)

    # Run loop with template
    async with engine.loop(
        task="Implement feature X",
        template=LoopTemplate.ITERATIVE_REFINEMENT
    ) as loop:
        async for iteration in loop:
            result = await iteration.execute()
            if result.is_complete:
                break

    # Or use convenience function
    result = await ralph_loop(
        task="Build the thing",
        max_iterations=10,
        on_iteration=my_callback
    )

ChatOps Commands:
    !ralph start <task>  - Start a Ralph loop
    !ralph status        - Show loop status
    !ralph pause/resume  - Control execution
    !reset               - Manual context reset with state preservation

Automatic Reset:
    When context usage exceeds threshold (default 80%):
    1. Consolidate memories
    2. Save state checkpoint
    3. Generate handoff summary
    4. Signal for fresh context

Created: 2026-01-28
"""

from .config import (
    RalphConfig,
    ContextThresholds,
    CeilingStrategy,
    ContextCeilingConfig,
)
from .state import (
    RalphState,
    StateCheckpoint,
    save_state,
    load_state,
    create_handoff_summary,
)
from .engine import (
    RalphEngine,
    RalphIteration,
    RalphResult,
    IterationStatus,
)
from .templates import (
    LoopTemplate,
    TemplateRegistry,
    get_template,
    register_template,
)
from .hooks import (
    RalphHook,
    PreIterationHook,
    PostIterationHook,
    OnResetHook,
    HookRegistry,
)
from .context_monitor import (
    ContextMonitor,
    ContextUsage,
    ResetTrigger,
    AutoResetConfig,
    CeilingAction,
    create_ceiling_monitor,
)
from .convenience import (
    ralph_loop,
    reset_context,
    get_loop_status,
)

__all__ = [
    # Config
    "RalphConfig",
    "ContextThresholds",
    "CeilingStrategy",
    "ContextCeilingConfig",
    # State
    "RalphState",
    "StateCheckpoint",
    "save_state",
    "load_state",
    "create_handoff_summary",
    # Engine
    "RalphEngine",
    "RalphIteration",
    "RalphResult",
    "IterationStatus",
    # Templates
    "LoopTemplate",
    "TemplateRegistry",
    "get_template",
    "register_template",
    # Hooks
    "RalphHook",
    "PreIterationHook",
    "PostIterationHook",
    "OnResetHook",
    "HookRegistry",
    # Context Monitor
    "ContextMonitor",
    "ContextUsage",
    "ResetTrigger",
    "AutoResetConfig",
    "CeilingAction",
    "create_ceiling_monitor",
    # Convenience
    "ralph_loop",
    "reset_context",
    "get_loop_status",
]

__version__ = "1.0.0"
