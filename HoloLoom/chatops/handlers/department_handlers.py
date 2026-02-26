"""
Department Handlers for Matrix ChatOps
=======================================

Matrix command handlers for HoloLoom Department System - multi-department
coordination and routing.

Commands:
- !dept list - List all registered departments
- !dept status <name> - Check department health status
- !dept process <name> <task> [params] - Process request via department
- !dept capabilities <name> - Show department capabilities
- !dept help - Department command help

Usage:
    from HoloLoom.chatops.handlers.department_handlers import register_department_handlers

    # In run_chatops.py:
    register_department_handlers(bot, registry)

Created: 2025-12-06
"""

import logging
import json
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.departments.registry import DepartmentRegistry

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Department imports
try:
    from HoloLoom.departments.registry import (
        DepartmentRegistry,
        DepartmentInstance,
        get_global_registry,
        create_registry
    )
    from HoloLoom.protocols.department import (
        DepartmentRequest,
        DepartmentResponse,
        ConfidenceLevel,
        ConfidenceMetadata
    )
    DEPARTMENTS_AVAILABLE = True
except ImportError:
    DEPARTMENTS_AVAILABLE = False
    DepartmentRegistry = None
    DepartmentInstance = None
    DepartmentRequest = None
    DepartmentResponse = None
    ConfidenceLevel = None
    ConfidenceMetadata = None

# Handler registry
try:
    from HoloLoom.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_dept_list(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    registry: 'DepartmentRegistry'
) -> str:
    """
    Handle !dept list command - List all registered departments.

    Args:
        room: Matrix room
        event: Message event
        args: Optional filter (ignored for now)
        registry: DepartmentRegistry instance

    Returns:
        List of departments with status
    """
    if not DEPARTMENTS_AVAILABLE:
        return "X Department system not available"

    try:
        departments = registry.list_departments()

        if not departments:
            return "** Registered Departments**\n\n_No departments registered._"

        response = f"** Registered Departments** ({len(departments)} total)\n\n"

        # Group by domain if available
        for dept in departments:
            name = dept.name if hasattr(dept, 'name') else str(dept)
            version = getattr(dept, 'version', '1.0.0')
            domain = getattr(dept, 'domain', 'general')
            status = getattr(dept, 'health_status', 'unknown')

            # Status icon
            if status == 'healthy':
                icon = "[OK]"
            elif status == 'degraded':
                icon = "[!]"
            elif status == 'unhealthy':
                icon = "[X]"
            else:
                icon = "[?]"

            response += f"{icon} **{name}** (v{version})\n"
            response += f"   Domain: {domain}\n"

            # Show capabilities count if available
            capabilities = getattr(dept, 'capabilities', [])
            if capabilities:
                response += f"   Capabilities: {len(capabilities)}\n"

            response += "\n"

        response += "_Use `!dept status <name>` for details_"
        return response

    except Exception as e:
        logger.error(f"Error in dept list: {e}")
        return f"X Failed to list departments: {str(e)}"


async def handle_dept_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    registry: 'DepartmentRegistry'
) -> str:
    """
    Handle !dept status command - Check department health status.

    Args:
        room: Matrix room
        event: Message event
        args: Department name
        registry: DepartmentRegistry instance

    Returns:
        Department health status and metrics
    """
    if not DEPARTMENTS_AVAILABLE:
        return "X Department system not available"

    dept_name = args.strip().lower() if args.strip() else None

    if not dept_name:
        return "X Usage: !dept status <name>\nExample: !dept status qa"

    try:
        # Try to get department
        dept = registry.get_department(dept_name)

        if dept is None:
            # List available departments
            available = [d.name for d in registry.list_departments() if hasattr(d, 'name')]
            available_str = ', '.join(available[:5]) if available else 'none'
            return f"X Department `{dept_name}` not found\n\nAvailable: {available_str}"

        # Get department instance info
        instance = getattr(dept, '_instance', None)

        response = f"** Department: {dept_name}**\n\n"

        # Basic info
        version = getattr(dept, 'version', '1.0.0')
        domain = getattr(dept, 'domain', 'general')
        response += f"**Version:** {version}\n"
        response += f"**Domain:** {domain}\n"

        # Health status
        health_status = getattr(dept, 'health_status', 'unknown')
        if health_status == 'healthy':
            response += f"**Health:** [OK] Healthy\n"
        elif health_status == 'degraded':
            response += f"**Health:** [!] Degraded\n"
        elif health_status == 'unhealthy':
            response += f"**Health:** [X] Unhealthy\n"
        else:
            response += f"**Health:** [?] {health_status}\n"

        # Request metrics if available
        request_count = getattr(instance, 'request_count', 0) if instance else 0
        active_requests = getattr(instance, 'active_requests', 0) if instance else 0
        if request_count > 0 or active_requests > 0:
            response += f"\n**Metrics:**\n"
            response += f"  Total requests: {request_count}\n"
            response += f"  Active requests: {active_requests}\n"

        # Last health check
        last_check = getattr(instance, 'last_health_check', None) if instance else None
        if last_check:
            response += f"  Last check: {last_check}\n"

        # Description if available
        description = getattr(dept, 'description', None)
        if description:
            response += f"\n**Description:**\n_{description}_\n"

        return response

    except Exception as e:
        logger.error(f"Error in dept status: {e}")
        return f"X Failed to get department status: {str(e)}"


async def handle_dept_process(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    registry: 'DepartmentRegistry'
) -> str:
    """
    Handle !dept process command - Process request via department.

    Args:
        room: Matrix room
        event: Message event
        args: "<dept_name> <task> [params_json]"
        registry: DepartmentRegistry instance

    Returns:
        Processing result
    """
    if not DEPARTMENTS_AVAILABLE:
        return "X Department system not available"

    # Parse args: dept_name task [params_json]
    parts = args.strip().split(None, 2)

    if len(parts) < 2:
        return "X Usage: !dept process <name> <task> [params_json]\nExample: !dept process qa analyze_code"

    dept_name = parts[0].lower()
    task = parts[1]
    params_str = parts[2] if len(parts) > 2 else "{}"

    # Parse optional JSON params
    try:
        if params_str.startswith('{'):
            params = json.loads(params_str)
        else:
            # Treat as simple key=value pairs
            params = {"input": params_str}
    except json.JSONDecodeError:
        params = {"input": params_str}

    try:
        # Get department
        dept = registry.get_department(dept_name)

        if dept is None:
            available = [d.name for d in registry.list_departments() if hasattr(d, 'name')]
            available_str = ', '.join(available[:5]) if available else 'none'
            return f"X Department `{dept_name}` not found\n\nAvailable: {available_str}"

        # Create request
        request = DepartmentRequest(
            task_type=task,
            input_data=params,
            context={"source": "chatops", "room_id": room.room_id if room else None}
        ) if DepartmentRequest else {"task_type": task, "input_data": params}

        # Route request through registry
        response = await registry.route_request(request)

        # Format response
        result = "** Department Processing Result**\n\n"
        result += f"**Department:** {dept_name}\n"
        result += f"**Task:** {task}\n\n"

        if response:
            # Check for success
            success = getattr(response, 'success', True)
            if success:
                result += "[OK] **Status:** Success\n\n"
            else:
                result += "[X] **Status:** Failed\n\n"

            # Output data
            output = getattr(response, 'output_data', response) if hasattr(response, 'output_data') else response
            if isinstance(output, dict):
                result += "**Output:**\n```json\n"
                result += json.dumps(output, indent=2)[:500]  # Truncate long output
                result += "\n```\n"
            elif output:
                result += f"**Output:** {str(output)[:500]}\n"

            # Confidence if available
            confidence = getattr(response, 'confidence', None)
            if confidence:
                score = getattr(confidence, 'score', 0)
                level = getattr(confidence, 'level', 'UNKNOWN')
                result += f"\n**Confidence:** {score:.2f} ({level})\n"

            # Processing time if available
            processing_time = getattr(response, 'processing_time_ms', None)
            if processing_time:
                result += f"**Time:** {processing_time:.1f}ms\n"
        else:
            result += "[?] No response received\n"

        return result

    except Exception as e:
        logger.error(f"Error in dept process: {e}")
        return f"X Processing failed: {str(e)}"


async def handle_dept_capabilities(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    registry: 'DepartmentRegistry'
) -> str:
    """
    Handle !dept capabilities command - Show department capabilities.

    Args:
        room: Matrix room
        event: Message event
        args: Department name
        registry: DepartmentRegistry instance

    Returns:
        Department capabilities list
    """
    if not DEPARTMENTS_AVAILABLE:
        return "X Department system not available"

    dept_name = args.strip().lower() if args.strip() else None

    if not dept_name:
        return "X Usage: !dept capabilities <name>\nExample: !dept capabilities qa"

    try:
        dept = registry.get_department(dept_name)

        if dept is None:
            available = [d.name for d in registry.list_departments() if hasattr(d, 'name')]
            available_str = ', '.join(available[:5]) if available else 'none'
            return f"X Department `{dept_name}` not found\n\nAvailable: {available_str}"

        response = f"** Department Capabilities: {dept_name}**\n\n"

        # Get capabilities
        capabilities = getattr(dept, 'capabilities', [])

        if not capabilities:
            # Try to get from manifest
            manifest = getattr(dept, 'manifest', None)
            if manifest:
                capabilities = getattr(manifest, 'capabilities', [])

        if capabilities:
            response += f"**{len(capabilities)} Capabilities:**\n\n"
            for i, cap in enumerate(capabilities, 1):
                if isinstance(cap, dict):
                    name = cap.get('name', f'Capability {i}')
                    desc = cap.get('description', '')
                    response += f"{i}. **{name}**\n"
                    if desc:
                        response += f"   _{desc}_\n"
                elif isinstance(cap, str):
                    response += f"{i}. {cap}\n"
                else:
                    cap_name = getattr(cap, 'name', str(cap))
                    cap_desc = getattr(cap, 'description', '')
                    response += f"{i}. **{cap_name}**\n"
                    if cap_desc:
                        response += f"   _{cap_desc}_\n"
        else:
            response += "_No capabilities defined_\n"

        # Task types if available
        task_types = getattr(dept, 'supported_task_types', [])
        if task_types:
            response += f"\n**Supported Task Types:**\n"
            for task in task_types[:10]:
                response += f"- `{task}`\n"
            if len(task_types) > 10:
                response += f"_...and {len(task_types) - 10} more_\n"

        # Input/output schemas if available
        input_schema = getattr(dept, 'input_schema', None)
        if input_schema:
            response += f"\n**Input Schema:** Available (use API for details)\n"

        response += f"\n_Use `!dept process {dept_name} <task>` to execute_"
        return response

    except Exception as e:
        logger.error(f"Error in dept capabilities: {e}")
        return f"X Failed to get capabilities: {str(e)}"


async def handle_dept_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !dept help command.

    Returns:
        Help message
    """
    return """** Department Commands**

**Discovery:**
- `!dept list` - List all registered departments
- `!dept status <name>` - Check department health and metrics
- `!dept capabilities <name>` - Show department capabilities

**Processing:**
- `!dept process <name> <task> [params]` - Process request via department

**Parameters:**
- `<name>` - Department name (e.g., qa, planning, context)
- `<task>` - Task type to execute (e.g., analyze_code, generate_plan)
- `[params]` - Optional JSON parameters or plain text input

**Available Departments:**
Common departments include:
- `qa` - Quality assurance (code analysis, testing)
- `planning` - Task planning and decomposition
- `context` - Context management and retrieval
- `rag` - Retrieval-augmented generation
- `orchestration` - Multi-department coordination
- `infrastructure` - System infrastructure

**Examples:**
```
!dept list
!dept status qa
!dept capabilities planning
!dept process qa analyze_code {"file": "main.py"}
!dept process planning decompose "Build a REST API"
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_department_handlers(
    bot,
    registry: 'DepartmentRegistry'
) -> None:
    """
    Register all Department command handlers.

    Args:
        bot: Matrix bot instance
        registry: DepartmentRegistry instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, department handlers not registered")
        return

    if not DEPARTMENTS_AVAILABLE:
        logger.warning("Department system not available, handlers not registered")
        return

    logger.info("Registering Department command handlers")

    # Define command map
    command_map = {
        "list": lambda room, event, args: handle_dept_list(room, event, args, registry),
        "status": lambda room, event, args: handle_dept_status(room, event, args, registry),
        "process": lambda room, event, args: handle_dept_process(room, event, args, registry),
        "capabilities": lambda room, event, args: handle_dept_capabilities(room, event, args, registry),
        "help": handle_dept_help
    }

    # Register with bot under !dept prefix
    for cmd, handler in command_map.items():
        bot.register_command(f"dept {cmd}", handler)

    # Also register !dept with no args as help
    bot.register_command("dept", handle_dept_help)

    logger.info(f"Registered {len(command_map)} Department commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class DepartmentHandlers:
    """
    Decorator-based ChatOps handlers for Department operations.

    Usage:
        from HoloLoom.chatops.handlers.department_handlers import DepartmentHandlers

        handlers = DepartmentHandlers(registry=dept_registry)
        registry.register_instance(handlers)
    """

    def __init__(self, registry: 'DepartmentRegistry'):
        """
        Initialize with DepartmentRegistry instance.

        Args:
            registry: DepartmentRegistry instance for department coordination
        """
        self.registry = registry

    @HandlerRegistry.register(
        command="dept list",
        description="List all registered departments",
        usage="!dept list",
        category=HandlerCategory.SYSTEM,
        aliases=["dls"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_list(self, room, event, args: str) -> str:
        """Handle !dept list command."""
        return await handle_dept_list(room, event, args, self.registry)

    @HandlerRegistry.register(
        command="dept status",
        description="Check department health status",
        usage="!dept status <name>",
        category=HandlerCategory.SYSTEM,
        aliases=["dst"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_status(self, room, event, args: str) -> str:
        """Handle !dept status command."""
        return await handle_dept_status(room, event, args, self.registry)

    @HandlerRegistry.register(
        command="dept process",
        description="Process request via department",
        usage="!dept process <name> <task> [params]",
        category=HandlerCategory.QUERY,
        aliases=["dp"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_process(self, room, event, args: str) -> str:
        """Handle !dept process command."""
        return await handle_dept_process(room, event, args, self.registry)

    @HandlerRegistry.register(
        command="dept capabilities",
        description="Show department capabilities",
        usage="!dept capabilities <name>",
        category=HandlerCategory.SYSTEM,
        aliases=["dcap"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_capabilities(self, room, event, args: str) -> str:
        """Handle !dept capabilities command."""
        return await handle_dept_capabilities(room, event, args, self.registry)

    @HandlerRegistry.register(
        command="dept help",
        description="Show Department command help",
        usage="!dept help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !dept help command."""
        return await handle_dept_help(room, event, args)
