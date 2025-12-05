"""
Code Handlers for Matrix ChatOps
=================================

Matrix command handlers for Claude Code integration.

Commands:
- !code query <question> - Ask about code
- !code refactor <instruction> - Request refactoring
- !code explain [target] - Explain code or concept
- !code test [type] - Generate tests
- !code fix - Suggest fixes for issues
- !code context - Get current editor context
- !code status - Check VS Code connection

Usage:
    from HoloLoom.chatops.handlers.code_handlers import register_code_handlers

    # In run_chatops.py:
    register_code_handlers(bot, orchestrator, code_department)
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.chatops.core.matrix_bot import MatrixBot
    from HoloLoom.chatops.core.chatops_bridge import ChatOpsOrchestrator
    from HoloLoom.departments.claude_code import ClaudeCodeDepartment

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_code_query(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code query command

    Args:
        room: Matrix room
        event: Message event
        args: Query text
        department: Claude Code Department

    Returns:
        Response message
    """
    if not args.strip():
        return "❌ Usage: !code query <question>\nExample: !code query How does the WeavingOrchestrator work?"

    # Extract optional mode from args
    mode = "verify"  # default
    question = args.strip()

    if question.startswith("--research "):
        mode = "research"
        question = question.replace("--research ", "").strip()
    elif question.startswith("--verify "):
        mode = "verify"
        question = question.replace("--verify ", "").strip()
    elif question.startswith("--direct "):
        mode = "direct"
        question = question.replace("--direct ", "").strip()

    # Send typing indicator
    # await room.typing(True)

    try:
        # Call department
        result = await department.process({
            "action": "query",
            "params": {
                "question": question,
                "mode": mode,
                "includeContext": True
            },
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            metadata = result.get("metadata", {})

            # Format response with metadata
            formatted = f"🔍 **Claude Code Query**\n\n{response}"

            if metadata.get("duration_ms"):
                formatted += f"\n\n_Query took {metadata['duration_ms']:.0f}ms_"

            return formatted
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Query failed: {error}"

    except Exception as e:
        logger.error(f"Error in code query: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_refactor(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code refactor command

    Args:
        room: Matrix room
        event: Message event
        args: Refactoring instruction
        department: Claude Code Department

    Returns:
        Response message
    """
    if not args.strip():
        return "❌ Usage: !code refactor <instruction>\nExample: !code refactor Extract this function"

    try:
        result = await department.process({
            "action": "refactor",
            "params": {
                "instruction": args.strip()
            },
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            return f"🔧 **Code Refactoring**\n\n{response}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Refactoring failed: {error}"

    except Exception as e:
        logger.error(f"Error in code refactor: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_explain(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code explain command

    Args:
        room: Matrix room
        event: Message event
        args: (optional) Target to explain
        department: Claude Code Department

    Returns:
        Response message
    """
    # Parse depth option
    depth = "detailed"
    target = args.strip() if args else None

    if target and target.startswith("--brief "):
        depth = "brief"
        target = target.replace("--brief ", "").strip() or None
    elif target and target.startswith("--comprehensive "):
        depth = "comprehensive"
        target = target.replace("--comprehensive ", "").strip() or None

    try:
        params = {"depth": depth}
        if target:
            params["target"] = target

        result = await department.process({
            "action": "explain",
            "params": params,
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            return f"📖 **Code Explanation**\n\n{response}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Explanation failed: {error}"

    except Exception as e:
        logger.error(f"Error in code explain: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_test(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code test command

    Args:
        room: Matrix room
        event: Message event
        args: (optional) Test type (unit, integration, edge, all)
        department: Claude Code Department

    Returns:
        Response message
    """
    # Parse test type
    test_type = args.strip().lower() if args.strip() else "unit"

    if test_type not in ["unit", "integration", "edge", "all"]:
        return f"❌ Invalid test type: {test_type}\nValid options: unit, integration, edge, all"

    try:
        result = await department.process({
            "action": "test",
            "params": {
                "testType": test_type
            },
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            return f"🧪 **Test Generation ({test_type})**\n\n{response}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Test generation failed: {error}"

    except Exception as e:
        logger.error(f"Error in code test: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_fix(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code fix command

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        department: Claude Code Department

    Returns:
        Response message
    """
    try:
        result = await department.process({
            "action": "fix",
            "params": {
                "includeDiagnostics": True
            },
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            return f"🔨 **Code Fixes**\n\n{response}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Fix suggestions failed: {error}"

    except Exception as e:
        logger.error(f"Error in code fix: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_context(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code context command

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        department: Claude Code Department

    Returns:
        Response message
    """
    try:
        result = await department.process({
            "action": "context",
            "params": {
                "includeSelection": True,
                "includeDiagnostics": True
            },
            "user_id": event.sender,
            "room_id": room.room_id
        })

        if result["success"]:
            response = result["result"]
            return f"📋 **Editor Context**\n\n{response}"
        else:
            error = result.get("error", "Unknown error")
            return f"❌ Context retrieval failed: {error}"

    except Exception as e:
        logger.error(f"Error in code context: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    department: 'ClaudeCodeDepartment'
) -> str:
    """
    Handle !code status command

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        department: Claude Code Department

    Returns:
        Response message
    """
    try:
        healthy = await department.health_check()
        stats = department.get_stats()

        if healthy:
            status_icon = "✅"
            status_text = "Connected"
        else:
            status_icon = "❌"
            status_text = "Disconnected"

        response = f"{status_icon} **Claude Code Status**: {status_text}\n\n"
        response += f"**MCP Server**: {stats['server_url']}\n"
        response += f"**Requests Processed**: {stats['requests_processed']}\n"
        response += f"**Errors**: {stats['errors']}\n"

        if stats.get("uptime_seconds"):
            uptime = stats["uptime_seconds"]
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            response += f"**Uptime**: {hours}h {minutes}m\n"

        if not healthy:
            response += "\n⚠️ Make sure VS Code is running with the Squad extension."

        return response

    except Exception as e:
        logger.error(f"Error in code status: {e}")
        return f"❌ Error: {str(e)}"


async def handle_code_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !code help command

    Returns:
        Help message
    """
    return """📚 **Claude Code Commands**

**Basic Commands:**
• `!code query <question>` - Ask about code
• `!code refactor <instruction>` - Request refactoring
• `!code explain [target]` - Explain code or concept
• `!code test [type]` - Generate tests (unit/integration/edge/all)
• `!code fix` - Suggest fixes for issues
• `!code context` - Get current editor context
• `!code status` - Check VS Code connection

**Query Modes:**
• `!code query --research <question>` - Deep research mode
• `!code query --verify <question>` - Verification mode (default)
• `!code query --direct <question>` - Direct answer mode

**Explanation Depth:**
• `!code explain --brief [target]` - Brief explanation
• `!code explain [target]` - Detailed explanation (default)
• `!code explain --comprehensive [target]` - Comprehensive explanation

**Examples:**
```
!code query How does the WeavingOrchestrator work?
!code refactor Extract this into a separate function
!code explain --brief recursion
!code test unit
!code fix
!code status
```

**Requirements:**
- VS Code must be running
- Squad extension must be installed and active
- MCP Server must be running on port 9001
"""


# ============================================================================
# Registration
# ============================================================================

def register_code_handlers(
    bot: 'MatrixBot',
    orchestrator: Optional['ChatOpsOrchestrator'],
    department: 'ClaudeCodeDepartment'
) -> None:
    """
    Register all code command handlers

    Args:
        bot: Matrix bot instance
        orchestrator: ChatOps orchestrator (optional, for context)
        department: Claude Code Department instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, code handlers not registered")
        return

    logger.info("Registering Claude Code command handlers")

    # Define command map
    command_map = {
        "query": lambda room, event, args: handle_code_query(room, event, args, department),
        "refactor": lambda room, event, args: handle_code_refactor(room, event, args, department),
        "explain": lambda room, event, args: handle_code_explain(room, event, args, department),
        "test": lambda room, event, args: handle_code_test(room, event, args, department),
        "fix": lambda room, event, args: handle_code_fix(room, event, args, department),
        "context": lambda room, event, args: handle_code_context(room, event, args, department),
        "status": lambda room, event, args: handle_code_status(room, event, args, department),
        "help": handle_code_help
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(f"code {cmd}", handler)

    # Also register !code with no args as help
    bot.register_command("code", handle_code_help)

    logger.info(f"Registered {len(command_map)} code commands")


# ============================================================================
# Decorator-Based Handler Class (New ChatOps Pattern)
# ============================================================================

try:
    from HoloLoom.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class CodeHandlers:
    """
    Decorator-based ChatOps handlers for Claude Code integration.

    Usage:
        from HoloLoom.chatops.handlers.code_handlers import CodeHandlers

        handlers = CodeHandlers(department=code_department)
        registry.register_instance(handlers)
    """

    def __init__(self, department: 'ClaudeCodeDepartment'):
        """
        Initialize with Claude Code Department.

        Args:
            department: ClaudeCodeDepartment instance for code operations
        """
        self.department = department

    @HandlerRegistry.register(
        command="code query",
        description="Ask questions about code",
        usage="!code query <question>",
        category=HandlerCategory.QUERY,
        aliases=["cq"]
    )
    async def handle_query(self, room, event, args: str) -> str:
        """Handle !code query command."""
        return await handle_code_query(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code refactor",
        description="Request code refactoring",
        usage="!code refactor <instruction>",
        category=HandlerCategory.QUERY,
        aliases=["cr"]
    )
    async def handle_refactor(self, room, event, args: str) -> str:
        """Handle !code refactor command."""
        return await handle_code_refactor(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code explain",
        description="Explain code or concepts",
        usage="!code explain [--brief|--comprehensive] [target]",
        category=HandlerCategory.QUERY,
        aliases=["ce"]
    )
    async def handle_explain(self, room, event, args: str) -> str:
        """Handle !code explain command."""
        return await handle_code_explain(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code test",
        description="Generate tests for code",
        usage="!code test [unit|integration|edge|all]",
        category=HandlerCategory.QUERY,
        aliases=["ct"]
    )
    async def handle_test(self, room, event, args: str) -> str:
        """Handle !code test command."""
        return await handle_code_test(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code fix",
        description="Suggest fixes for code issues",
        usage="!code fix",
        category=HandlerCategory.QUERY,
        aliases=["cf"]
    )
    async def handle_fix(self, room, event, args: str) -> str:
        """Handle !code fix command."""
        return await handle_code_fix(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code context",
        description="Get current editor context",
        usage="!code context",
        category=HandlerCategory.SYSTEM,
        aliases=["cc"]
    )
    async def handle_context(self, room, event, args: str) -> str:
        """Handle !code context command."""
        return await handle_code_context(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code status",
        description="Check VS Code connection status",
        usage="!code status",
        category=HandlerCategory.SYSTEM,
        aliases=["cs"]
    )
    async def handle_status(self, room, event, args: str) -> str:
        """Handle !code status command."""
        return await handle_code_status(room, event, args, self.department)

    @HandlerRegistry.register(
        command="code help",
        description="Show Claude Code command help",
        usage="!code help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !code help command."""
        return await handle_code_help(room, event, args)
