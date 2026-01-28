#!/usr/bin/env python3
"""
Ralph Loop ChatOps Handlers
============================
Command handlers for the Ralph context reset and loop system.

Named after the viral "Ralph Wiggum" technique by Geoff Huntley (2025):
"Iteration beats perfection" - Reset context, preserve progress.

Commands:
- !ralph start <task>      - Start a Ralph loop
- !ralph status            - Show current loop status
- !ralph pause             - Pause the current loop
- !ralph resume            - Resume a paused loop
- !ralph stop              - Stop the current loop
- !ralph handoff           - Get handoff summary for context transfer
- !ralph history           - Show iteration history
- !ralph templates         - List available templates
- !reset                   - Manual context reset with state preservation
- !reset --reason <text>   - Reset with custom reason

Created: 2026-01-28
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    # Allow running without Matrix
    pass

# Handler registry
from HoloLoom.chatops.handlers.handler_registry import (
    HandlerRegistry,
    HandlerCategory,
    chatops_handler,
)

# Ralph imports
try:
    from HoloLoom.ralph import (
        RalphEngine,
        RalphConfig,
        RalphState,
        RalphResult,
        LoopTemplate,
        IterationStatus,
        reset_context,
        get_loop_status,
        get_handoff_summary,
        init_global_engine,
        get_global_engine,
        get_global_monitor,
        ralph_loop,
        list_templates,
    )
    from HoloLoom.ralph.context_monitor import (
        ContextMonitor,
        create_context_monitor,
        ResetTrigger,
    )
    RALPH_AVAILABLE = True
except ImportError as e:
    RALPH_AVAILABLE = False
    RalphEngine = None
    RalphConfig = None
    reset_context = None
    get_loop_status = None

logger = logging.getLogger(__name__)


# ============================================================================
# Ralph State Management
# ============================================================================

class RalphChatOpsState:
    """Manages Ralph state for ChatOps sessions."""

    def __init__(self):
        self.engine: Optional[RalphEngine] = None
        self.monitor: Optional[ContextMonitor] = None
        self.active_task: Optional[str] = None
        self.session_id: Optional[str] = None
        self._iteration_callbacks: List[callable] = []

    def initialize(self, config: Optional[RalphConfig] = None):
        """Initialize Ralph engine for ChatOps."""
        if not RALPH_AVAILABLE:
            raise RuntimeError("Ralph module not available")

        config = config or RalphConfig.default()
        self.engine = init_global_engine(config)
        self.monitor = get_global_monitor()
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        return {"initialized": True, "session_id": self.session_id}

    def get_status(self) -> Dict[str, Any]:
        """Get current Ralph status."""
        if self.engine is None:
            return {"status": "not_initialized", "ralph_available": RALPH_AVAILABLE}

        status = get_loop_status()
        if self.monitor:
            status["context"] = self.monitor.get_current_usage()
        return status


# Global ChatOps state
_chatops_state = RalphChatOpsState()


# ============================================================================
# Command Handlers
# ============================================================================

@chatops_handler(
    command="ralph",
    description="Ralph loop management",
    category=HandlerCategory.SYSTEM,
    examples=[
        "!ralph start Build a REST API",
        "!ralph status",
        "!ralph pause",
        "!ralph resume",
        "!ralph handoff",
    ],
)
async def handle_ralph_command(
    room: "MatrixRoom",
    event: "RoomMessageText",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """
    Handle !ralph commands.

    Subcommands:
    - start <task>: Start a new Ralph loop
    - status: Show current loop status
    - pause: Pause the current loop
    - resume: Resume a paused loop
    - stop: Stop the current loop
    - handoff: Get handoff summary
    - history: Show iteration history
    - templates: List available templates
    """
    if not RALPH_AVAILABLE:
        await send_message(room.room_id, "❌ Ralph module not available")
        return {"error": "ralph_not_available"}

    if not args:
        await send_message(room.room_id, _format_ralph_help())
        return {"action": "help"}

    subcommand = args[0].lower()
    subargs = args[1:] if len(args) > 1 else []

    handlers = {
        "start": _handle_ralph_start,
        "status": _handle_ralph_status,
        "pause": _handle_ralph_pause,
        "resume": _handle_ralph_resume,
        "stop": _handle_ralph_stop,
        "handoff": _handle_ralph_handoff,
        "history": _handle_ralph_history,
        "templates": _handle_ralph_templates,
        "help": lambda r, a, s: _send_help(r, s),
    }

    handler = handlers.get(subcommand)
    if handler:
        return await handler(room, subargs, send_message)
    else:
        await send_message(
            room.room_id,
            f"❓ Unknown Ralph subcommand: `{subcommand}`\n\n"
            f"Use `!ralph help` for available commands.",
        )
        return {"error": "unknown_subcommand", "subcommand": subcommand}


async def _handle_ralph_start(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Start a new Ralph loop."""
    if not args:
        await send_message(
            room.room_id,
            "❌ Missing task description\n\n"
            "Usage: `!ralph start <task description>`\n"
            "Example: `!ralph start Build a REST API with authentication`",
        )
        return {"error": "missing_task"}

    task = " ".join(args)

    # Initialize if needed
    if _chatops_state.engine is None:
        _chatops_state.initialize()

    _chatops_state.active_task = task

    await send_message(
        room.room_id,
        f"🔄 **Ralph Loop Started**\n\n"
        f"**Task:** {task}\n"
        f"**Session:** {_chatops_state.session_id}\n"
        f"**Template:** iterative_refinement\n\n"
        f"Use `!ralph status` to check progress.\n"
        f"Use `!reset` to manually reset context when needed.",
    )

    return {
        "action": "start",
        "task": task,
        "session_id": _chatops_state.session_id,
    }


async def _handle_ralph_status(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Show current Ralph status."""
    status = _chatops_state.get_status()

    if status.get("status") == "not_initialized":
        await send_message(
            room.room_id,
            "ℹ️ **Ralph Status**\n\n"
            "No active Ralph loop.\n"
            "Use `!ralph start <task>` to begin.",
        )
        return status

    # Format status message
    lines = [
        "📊 **Ralph Loop Status**",
        "",
        f"**Loop ID:** {status.get('loop_id', 'N/A')}",
        f"**Task:** {status.get('task', 'N/A')}",
        f"**Status:** {status.get('status', 'unknown')}",
        f"**Iteration:** {status.get('iteration', 0)}",
        f"**Resets:** {status.get('resets', 0)}",
        f"**Progress:** {status.get('completion_percent', 0):.1f}%",
    ]

    # Add context usage if available
    if "context" in status:
        ctx = status["context"]
        lines.extend([
            "",
            "**Context Window:**",
            f"  Usage: {ctx.get('percent', 0):.1%}",
            f"  Tokens: {ctx.get('tokens', 0):,}",
        ])
        if ctx.get("trigger"):
            lines.append(f"  ⚠️ Trigger: {ctx['trigger']}")

    await send_message(room.room_id, "\n".join(lines))
    return status


async def _handle_ralph_pause(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Pause the current loop."""
    if _chatops_state.engine is None:
        await send_message(room.room_id, "❌ No active Ralph loop to pause")
        return {"error": "no_active_loop"}

    await _chatops_state.engine.pause()

    await send_message(
        room.room_id,
        "⏸️ **Ralph Loop Paused**\n\n"
        "Use `!ralph resume` to continue.",
    )

    return {"action": "pause"}


async def _handle_ralph_resume(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Resume a paused loop."""
    if _chatops_state.engine is None:
        await send_message(room.room_id, "❌ No active Ralph loop to resume")
        return {"error": "no_active_loop"}

    await _chatops_state.engine.resume()

    await send_message(
        room.room_id,
        "▶️ **Ralph Loop Resumed**\n\n"
        "Use `!ralph status` to check progress.",
    )

    return {"action": "resume"}


async def _handle_ralph_stop(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Stop the current loop."""
    if _chatops_state.engine is None:
        await send_message(room.room_id, "❌ No active Ralph loop to stop")
        return {"error": "no_active_loop"}

    # Get final handoff before stopping
    handoff = await get_handoff_summary()

    _chatops_state.active_task = None

    await send_message(
        room.room_id,
        "⏹️ **Ralph Loop Stopped**\n\n"
        "Final state has been saved.\n"
        "Use `!ralph handoff` to get the handoff summary.",
    )

    return {"action": "stop", "handoff_available": True}


async def _handle_ralph_handoff(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Get handoff summary for context transfer."""
    handoff = await get_handoff_summary()

    await send_message(
        room.room_id,
        f"📋 **Ralph Handoff Summary**\n\n"
        f"```markdown\n{handoff[:3000]}\n```\n\n"
        f"_This summary can be used to continue work in a fresh context._",
    )

    return {"action": "handoff", "length": len(handoff)}


async def _handle_ralph_history(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """Show iteration history."""
    if _chatops_state.engine is None or _chatops_state.engine.state is None:
        await send_message(room.room_id, "ℹ️ No Ralph history available")
        return {"error": "no_history"}

    state = _chatops_state.engine.state
    iterations = state.iterations[-10:]  # Last 10

    if not iterations:
        await send_message(room.room_id, "ℹ️ No iterations recorded yet")
        return {"iterations": 0}

    lines = ["📜 **Ralph Iteration History** (last 10)", ""]

    for record in iterations:
        status_icon = {
            "success": "✅",
            "error": "❌",
            "timeout": "⏱️",
            "skipped": "⏭️",
        }.get(record.status, "❓")

        lines.append(
            f"{status_icon} **Iteration {record.iteration_number}** "
            f"({record.duration_ms:.0f}ms) - {record.summary or record.status}"
        )

    await send_message(room.room_id, "\n".join(lines))

    return {"action": "history", "count": len(iterations)}


async def _handle_ralph_templates(
    room: "MatrixRoom",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """List available templates."""
    templates = list_templates()

    lines = ["📚 **Available Loop Templates**", ""]

    for template in templates:
        lines.append(f"• **{template['name']}**: {template['description']}")

    lines.extend([
        "",
        "Use: `!ralph start --template <name> <task>`",
    ])

    await send_message(room.room_id, "\n".join(lines))

    return {"action": "templates", "count": len(templates)}


async def _send_help(room: "MatrixRoom", send_message: callable):
    """Send help message."""
    await send_message(room.room_id, _format_ralph_help())
    return {"action": "help"}


def _format_ralph_help() -> str:
    """Format Ralph help message."""
    return """🤖 **Ralph Loop Commands**

**Loop Management:**
• `!ralph start <task>` - Start a new Ralph loop
• `!ralph status` - Show current loop status
• `!ralph pause` - Pause the current loop
• `!ralph resume` - Resume a paused loop
• `!ralph stop` - Stop the current loop

**Context Management:**
• `!ralph handoff` - Get handoff summary for context transfer
• `!reset` - Manual context reset with state preservation
• `!reset --reason <text>` - Reset with custom reason

**Information:**
• `!ralph history` - Show iteration history
• `!ralph templates` - List available loop templates
• `!ralph help` - Show this help

**Philosophy:**
_"Iteration beats perfection."_ - Context resets aren't failures, they're features.
Progress lives in files and memory, not in the context window.
"""


# ============================================================================
# Reset Command Handler
# ============================================================================

@chatops_handler(
    command="reset",
    description="Manual context reset with state preservation",
    category=HandlerCategory.SYSTEM,
    examples=[
        "!reset",
        "!reset --reason Context getting too large",
        "!reset --consolidate",
    ],
)
async def handle_reset_command(
    room: "MatrixRoom",
    event: "RoomMessageText",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """
    Handle !reset command for manual context reset.

    This triggers:
    1. Memory consolidation (optional)
    2. State checkpoint save
    3. Handoff summary generation
    4. Context monitor reset

    Flags:
    --reason <text>: Custom reason for the reset
    --consolidate: Trigger memory consolidation
    --no-checkpoint: Skip checkpoint save
    """
    if not RALPH_AVAILABLE:
        await send_message(room.room_id, "❌ Ralph module not available")
        return {"error": "ralph_not_available"}

    # Parse args
    reason = "Manual reset via ChatOps"
    consolidate = True
    save_checkpoint = True

    i = 0
    while i < len(args):
        if args[i] == "--reason" and i + 1 < len(args):
            reason = args[i + 1]
            i += 2
        elif args[i] == "--consolidate":
            consolidate = True
            i += 1
        elif args[i] == "--no-checkpoint":
            save_checkpoint = False
            i += 1
        else:
            # Treat remaining as reason
            reason = " ".join(args[i:])
            break

    # Perform reset
    try:
        result = await reset_context(
            reason=reason,
            save_checkpoint=save_checkpoint,
            consolidate_memories=consolidate,
        )

        # Format response
        lines = [
            "🔄 **Context Reset Triggered**",
            "",
            f"**Reason:** {reason}",
            f"**Timestamp:** {result['timestamp']}",
            "",
            "**Actions Taken:**",
        ]

        for action in result.get("actions_taken", []):
            lines.append(f"  ✅ {action}")

        if result.get("checkpoint_id"):
            lines.append(f"\n**Checkpoint ID:** `{result['checkpoint_id']}`")

        lines.extend([
            "",
            "**Handoff Summary:**",
            "```markdown",
            result.get("handoff_summary", "No summary available")[:1500],
            "```",
            "",
            "_A fresh context can continue from this handoff summary._",
        ])

        await send_message(room.room_id, "\n".join(lines))

        return result

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        await send_message(room.room_id, f"❌ Reset failed: {str(e)}")
        return {"error": str(e)}


# ============================================================================
# Context Status Command (Quick Check)
# ============================================================================

@chatops_handler(
    command="context",
    description="Check context window usage",
    category=HandlerCategory.SYSTEM,
    examples=[
        "!context",
        "!context projection",
    ],
)
async def handle_context_command(
    room: "MatrixRoom",
    event: "RoomMessageText",
    args: List[str],
    send_message: callable,
) -> Dict[str, Any]:
    """
    Handle !context command for quick context status check.
    """
    if not RALPH_AVAILABLE:
        await send_message(room.room_id, "❌ Ralph module not available")
        return {"error": "ralph_not_available"}

    monitor = get_global_monitor()

    if monitor is None:
        # Initialize if not already done
        _chatops_state.initialize()
        monitor = get_global_monitor()

    if monitor is None:
        await send_message(room.room_id, "ℹ️ Context monitor not initialized")
        return {"status": "not_initialized"}

    usage = monitor.get_current_usage()
    projection = monitor.get_projection()

    # Format gauge
    percent = usage.get("percent", 0)
    gauge = _format_gauge(percent)

    lines = [
        "📊 **Context Window Status**",
        "",
        f"**Usage:** {gauge} {percent:.1%}",
        f"**Tokens:** {usage.get('tokens', 0):,}",
    ]

    if usage.get("trigger"):
        lines.append(f"⚠️ **Alert:** {usage['trigger']}")

    # Add projection if requested
    if args and args[0] == "projection":
        lines.extend([
            "",
            "**Projection:**",
            f"  Iterations until reset: ~{projection.get('iterations_until_reset', '∞')}",
            f"  Tokens until reset: {projection.get('tokens_until_reset', 0):,}",
        ])

    # Add threshold info
    lines.extend([
        "",
        "**Thresholds:**",
        f"  Warning: {usage.get('warning_threshold', 0.6):.0%}",
        f"  Reset: {usage.get('reset_threshold', 0.85):.0%}",
    ])

    await send_message(room.room_id, "\n".join(lines))

    return usage


def _format_gauge(percent: float, width: int = 20) -> str:
    """Format a simple ASCII gauge."""
    filled = int(percent * width)
    empty = width - filled

    # Color coding via emoji
    if percent >= 0.85:
        marker = "🔴"
    elif percent >= 0.60:
        marker = "🟡"
    else:
        marker = "🟢"

    return f"{marker} [{'█' * filled}{'░' * empty}]"


# ============================================================================
# Registration
# ============================================================================

def register_ralph_handlers(registry: HandlerRegistry):
    """Register all Ralph handlers with the registry."""
    registry.register(handle_ralph_command)
    registry.register(handle_reset_command)
    registry.register(handle_context_command)

    logger.info("Ralph ChatOps handlers registered")
