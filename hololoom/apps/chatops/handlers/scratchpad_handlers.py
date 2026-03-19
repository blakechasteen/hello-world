from __future__ import annotations
"""
Scratch Pad Handlers for Matrix ChatOps
========================================

Matrix command handlers for HoloLoom Scratch Pad artifact storage.

Commands:
- !scratch store <name> - Store last weaving result
- !scratch get <name|id> - Retrieve artifact by name or ID
- !scratch list [scope] - List artifacts (user/room/all)
- !scratch delete <name|id> - Delete artifact
- !scratch share <name|id> - Share artifact with room (promote USER→ROOM)
- !scratch export <name|id> - Upload artifact to Matrix as file
- !scratch help - Show scratch pad help

Usage:
    from hololoom.apps.chatops.handlers.scratchpad_handlers import (
        register_scratchpad_handlers,
        ScratchPadHandlers
    )

    # In run_chatops.py:
    register_scratchpad_handlers(bot, orchestrator)

Created: December 2025
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hololoom.apps.chatops.scratchpad.manager import ScratchPadManager
    from hololoom.fabric.spacetime import Spacetime

logger = logging.getLogger(__name__)


# ============================================================================
# Matrix Dependencies (optional)
# ============================================================================

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = Any
    RoomMessageText = Any


# ============================================================================
# Handler Registry (optional)
# ============================================================================

try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerCategory,
        HandlerRegistry,
        chatops_handler,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    HandlerRegistry = None
    HandlerCategory = None

    # Create a no-op decorator if registry unavailable
    def chatops_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Scratch Pad Dependencies
# ============================================================================

try:
    from hololoom.apps.chatops.scratchpad.manager import (
        ScratchPadManager,
        create_scratchpad_manager,
    )
    from hololoom.apps.chatops.scratchpad.types import (
        ArtifactScope,
        ArtifactType,
        ListResult,
        RetrieveResult,
        ScratchArtifact,
        SessionArtifactContext,
        StoreResult,
    )
    SCRATCHPAD_AVAILABLE = True
except ImportError:
    SCRATCHPAD_AVAILABLE = False
    ScratchPadManager = None
    create_scratchpad_manager = None
    ArtifactScope = None
    ArtifactType = None
    ScratchArtifact = None
    StoreResult = None
    RetrieveResult = None
    ListResult = None
    SessionArtifactContext = None


# ============================================================================
# Conversation Handler Integration (for last result access)
# ============================================================================

try:
    from hololoom.apps.chatops.handlers.conversation_handlers import get_session_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    get_session_manager = None


# ============================================================================
# HoloLoom Dependencies (optional)
# ============================================================================

try:
    from hololoom.fabric.spacetime import Spacetime
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    WeavingOrchestrator = Any
    Spacetime = None


# ============================================================================
# Global Instances (set by register_scratchpad_handlers)
# ============================================================================

_scratchpad_manager: Optional['ScratchPadManager'] = None
_last_results: dict[str, 'Spacetime'] = {}  # user_id -> last Spacetime result


def get_scratchpad_manager() -> Optional['ScratchPadManager']:
    """Get the global ScratchPadManager instance."""
    return _scratchpad_manager


def set_scratchpad_manager(manager: 'ScratchPadManager') -> None:
    """Set the global ScratchPadManager instance."""
    global _scratchpad_manager
    _scratchpad_manager = manager


def record_last_result(user_id: str, spacetime: 'Spacetime') -> None:
    """Record the last weaving result for a user (for !scratch store)."""
    global _last_results
    _last_results[user_id] = spacetime


def get_last_result(user_id: str) -> Optional['Spacetime']:
    """Get the last weaving result for a user."""
    return _last_results.get(user_id)


def clear_last_result(user_id: str) -> None:
    """Clear the last result for a user after storing."""
    if user_id in _last_results:
        del _last_results[user_id]


# ============================================================================
# Helper Functions
# ============================================================================

def format_artifact_summary(artifact: 'ScratchArtifact', include_content: bool = False) -> str:
    """Format an artifact for display."""
    size_kb = artifact.size_bytes / 1024
    scope_emoji = {
        'user': '👤',
        'room': '🚪',
        'global': '🌐'
    }.get(artifact.scope.value, '📦')

    type_emoji = {
        'code': '💻',
        'image': '🖼️',
        'document': '📄',
        'archive': '📦',
        'audio': '🎵',
        'video': '🎬',
        'data': '📊',
        'text': '📝',
        'embedding': '🧮',
        'context': '💬'
    }.get(artifact.artifact_type.value, '📁')

    lines = [
        f"**{artifact.name}** {type_emoji} {scope_emoji}",
        f"ID: `{artifact.artifact_id[:16]}...`",
        f"Type: {artifact.artifact_type.value} | Size: {size_kb:.1f}KB",
        f"Created: {artifact.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"Accessed: {artifact.access_count}x"
    ]

    if artifact.expires_at:
        days_left = (artifact.expires_at - datetime.now()).days
        if days_left > 0:
            lines.append(f"Expires: {days_left} days")
        else:
            lines.append("⚠️ Expired")

    if artifact.source_spacetime_id:
        lines.append(f"Source: Spacetime `{artifact.source_spacetime_id[:12]}...`")

    return "\n".join(lines)


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ============================================================================
# Scratch Pad Command Handlers
# ============================================================================

async def handle_scratch_store(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None
) -> str:
    """
    Handle !scratch store command - Store last weaving result.

    Args:
        room: Matrix room
        event: Message event
        args: Artifact name
        manager: ScratchPadManager instance (optional, uses global)

    Returns:
        Store result message
    """
    if not args.strip():
        return (
            "**Usage:** `!scratch store <name>`\n\n"
            "Stores the last weaving result as a scratch pad artifact.\n\n"
            "**Examples:**\n"
            "- `!scratch store my_analysis`\n"
            "- `!scratch store query_result_2025`\n"
            "- `!scratch store research_notes`"
        )

    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'
    name = args.strip()

    # Get last result for this user
    last_result = get_last_result(user_id)

    if not last_result:
        # Try to get from conversation handler if available
        if CONVERSATION_AVAILABLE and get_session_manager:
            session_mgr = get_session_manager()
            if session_mgr:
                session = session_mgr.get_session(user_id)
                if session and session.last_result:
                    last_result = session.last_result

    if not last_result:
        return (
            "❌ No recent weaving result to store.\n\n"
            "Run a `!weave` or `!loom` command first, then use `!scratch store <name>`."
        )

    try:
        # Extract content from spacetime result
        # Try response attribute first, then fallback to string representation
        content = getattr(last_result, 'response', None)
        if content is None:
            content = str(last_result)

        # Convert to bytes if string
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content

        # Get spacetime ID if available
        spacetime_id = getattr(last_result, 'spacetime_id', None)

        # Store with user-provided name using direct store() API
        result = await manager.store(
            name=name,
            content=content_bytes,
            user_id=user_id,
            room_id=room_id,
            artifact_type=ArtifactType.DATA,
            source_spacetime_id=spacetime_id,
            metadata={'source': 'scratch_store_command'}
        )

        if not result.success:
            return f"❌ Store failed: {result.error}"

        artifact = result.artifact
        dedup = " (deduplicated)" if result.deduplicated else ""

        clear_last_result(user_id)

        return (
            f"✅ **Artifact Stored**\n\n"
            f"- **Name:** {artifact.name}\n"
            f"- **ID:** `{artifact.artifact_id[:16]}...`\n"
            f"- **Size:** {artifact.size_bytes:,} bytes{dedup}\n\n"
            f"Use `!scratch get {name}` to retrieve."
        )

    except Exception as e:
        logger.exception(f"Error storing artifact: {e}")
        return f"❌ Error storing artifact: {str(e)}"


async def handle_scratch_get(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None
) -> str:
    """
    Handle !scratch get command - Retrieve artifact by name or ID.

    Args:
        room: Matrix room
        event: Message event
        args: Artifact name or ID
        manager: ScratchPadManager instance (optional, uses global)

    Returns:
        Artifact details
    """
    if not args.strip():
        return (
            "**Usage:** `!scratch get <name|id>`\n\n"
            "Retrieves an artifact by name or ID.\n\n"
            "**Examples:**\n"
            "- `!scratch get my_analysis`\n"
            "- `!scratch get artifact-abc123def456`"
        )

    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'
    identifier = args.strip()

    try:
        result = await manager.get(
            identifier=identifier,
            user_id=user_id,
            room_id=room_id
        )

        if not result.success:
            return f"❌ {result.error}"

        artifact = result.artifact
        content = result.content

        # Format response
        lines = ["## 📦 Artifact Retrieved\n"]
        lines.append(format_artifact_summary(artifact))

        # Show content preview (truncated for large content)
        if content:
            if isinstance(content, bytes):
                lines.append(f"\n**Content:** Binary ({len(content)} bytes)")
                # Show hex preview for small binary
                if len(content) <= 64:
                    lines.append(f"```\n{content.hex()}\n```")
            elif isinstance(content, str):
                preview = content[:500]
                if len(content) > 500:
                    preview += f"\n... ({len(content) - 500} more characters)"
                lines.append(f"\n**Content:**\n```\n{preview}\n```")
            else:
                lines.append(f"\n**Content Type:** {type(content).__name__}")

        return "\n".join(lines)

    except Exception as e:
        logger.exception(f"Error retrieving artifact: {e}")
        return f"❌ Error retrieving artifact: {str(e)}"


async def handle_scratch_list(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None
) -> str:
    """
    Handle !scratch list command - List artifacts.

    Args:
        room: Matrix room
        event: Message event
        args: Optional scope filter (user/room/all)
        manager: ScratchPadManager instance (optional, uses global)

    Returns:
        List of artifacts
    """
    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'

    # Parse scope argument
    scope_filter = None
    if args.strip().lower() in ('user', 'room', 'all'):
        scope_str = args.strip().lower()
        if scope_str == 'user':
            scope_filter = ArtifactScope.USER if SCRATCHPAD_AVAILABLE else None
        elif scope_str == 'room':
            scope_filter = ArtifactScope.ROOM if SCRATCHPAD_AVAILABLE else None
        # 'all' means no filter

    try:
        result = await manager.list(
            user_id=user_id,
            room_id=room_id,
            scope_filter=scope_filter
        )

        if not result.success:
            return f"❌ {result.error}"

        if not result.artifacts:
            return (
                "📭 No artifacts found.\n\n"
                "Use `!scratch store <name>` after a weave to save results."
            )

        # Format response
        lines = [
            f"## 📦 Scratch Pad ({result.total_count} artifacts, {format_size(result.total_size_bytes)})\n"
        ]

        for ref in result.artifacts[:20]:  # Limit to 20
            scope_emoji = {
                'user': '👤',
                'room': '🚪',
                'global': '🌐'
            }.get(ref.scope.value, '📦')

            type_emoji = {
                'code': '💻',
                'image': '🖼️',
                'document': '📄',
                'archive': '📦',
                'data': '📊',
                'text': '📝'
            }.get(ref.artifact_type.value, '📁')

            size_str = format_size(ref.size_bytes)
            lines.append(
                f"{type_emoji} **{ref.name}** {scope_emoji} - {size_str} "
                f"(`{ref.artifact_id[:12]}...`)"
            )

        if result.total_count > 20:
            lines.append(f"\n... and {result.total_count - 20} more")

        lines.append(
            "\n*Use `!scratch list user|room|all` to filter by scope*"
        )

        return "\n".join(lines)

    except Exception as e:
        logger.exception(f"Error listing artifacts: {e}")
        return f"❌ Error listing artifacts: {str(e)}"


async def handle_scratch_delete(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None
) -> str:
    """
    Handle !scratch delete command - Delete an artifact.

    Args:
        room: Matrix room
        event: Message event
        args: Artifact name or ID
        manager: ScratchPadManager instance (optional, uses global)

    Returns:
        Delete result message
    """
    if not args.strip():
        return (
            "**Usage:** `!scratch delete <name|id>`\n\n"
            "Deletes an artifact you own.\n\n"
            "**Examples:**\n"
            "- `!scratch delete my_analysis`\n"
            "- `!scratch delete artifact-abc123def456`"
        )

    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'
    identifier = args.strip()

    try:
        result = await manager.delete(
            identifier=identifier,
            user_id=user_id,
            room_id=room_id
        )

        if not result.success:
            return f"❌ {result.error}"

        if result.archived:
            return f"✅ Artifact archived: `{result.artifact_id}`"
        else:
            return f"✅ Artifact deleted: `{result.artifact_id}`"

    except Exception as e:
        logger.exception(f"Error deleting artifact: {e}")
        return f"❌ Error deleting artifact: {str(e)}"


async def handle_scratch_share(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None
) -> str:
    """
    Handle !scratch share command - Share artifact with room.

    Args:
        room: Matrix room
        event: Message event
        args: Artifact name or ID
        manager: ScratchPadManager instance (optional, uses global)

    Returns:
        Share result message
    """
    if not args.strip():
        return (
            "**Usage:** `!scratch share <name|id>`\n\n"
            "Shares a private artifact with the current room (USER → ROOM scope).\n\n"
            "**Examples:**\n"
            "- `!scratch share my_analysis`\n"
            "- `!scratch share artifact-abc123def456`"
        )

    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'
    identifier = args.strip()

    try:
        success, error = await manager.share(
            identifier=identifier,
            user_id=user_id,
            room_id=room_id
        )

        if not success:
            return f"❌ {error}"

        return (
            f"✅ Artifact **{identifier}** shared with room.\n\n"
            f"Room members can now access it with `!scratch get {identifier}`"
        )

    except Exception as e:
        logger.exception(f"Error sharing artifact: {e}")
        return f"❌ Error sharing artifact: {str(e)}"


async def handle_scratch_export(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    manager: Optional['ScratchPadManager'] = None,
    matrix_client: Any = None
) -> str:
    """
    Handle !scratch export command - Upload artifact to Matrix.

    Args:
        room: Matrix room
        event: Message event
        args: Artifact name or ID
        manager: ScratchPadManager instance (optional, uses global)
        matrix_client: Matrix client for file upload

    Returns:
        Export result message with file URL
    """
    if not args.strip():
        return (
            "**Usage:** `!scratch export <name|id>`\n\n"
            "Uploads an artifact to Matrix as a file attachment.\n\n"
            "**Examples:**\n"
            "- `!scratch export my_analysis`\n"
            "- `!scratch export artifact-abc123def456`"
        )

    manager = manager or get_scratchpad_manager()
    if not manager:
        if not SCRATCHPAD_AVAILABLE:
            return "❌ Scratch pad module not available"
        manager = await create_scratchpad_manager()
        set_scratchpad_manager(manager)

    user_id = getattr(event, 'sender', 'unknown')
    room_id = room.room_id if room else 'unknown'
    identifier = args.strip()

    try:
        # Get the artifact and content
        result = await manager.get(
            identifier=identifier,
            user_id=user_id,
            room_id=room_id
        )

        if not result.success:
            return f"❌ {result.error}"

        artifact = result.artifact
        content = result.content

        # Prepare for export
        export_data = manager.export(
            identifier=identifier,
            user_id=user_id,
            room_id=room_id
        )

        if not export_data:
            return "❌ Failed to export artifact"

        content_bytes, content_type, filename = export_data

        # If we have a Matrix client, upload the file
        if matrix_client and hasattr(matrix_client, 'upload'):
            try:
                # Upload to Matrix
                upload_resp = await matrix_client.upload(
                    content_bytes,
                    content_type=content_type,
                    filename=filename
                )

                if hasattr(upload_resp, 'content_uri'):
                    return (
                        f"✅ **Artifact Exported**\n\n"
                        f"**Name:** {artifact.name}\n"
                        f"**File:** {filename}\n"
                        f"**Size:** {format_size(len(content_bytes))}\n"
                        f"**Type:** {content_type}\n\n"
                        f"📎 [Download]({upload_resp.content_uri})"
                    )
            except Exception as upload_error:
                logger.warning(f"Matrix upload failed: {upload_error}")

        # Fallback: return download info without actual upload
        return (
            f"✅ **Artifact Ready for Export**\n\n"
            f"**Name:** {artifact.name}\n"
            f"**File:** {filename}\n"
            f"**Size:** {format_size(len(content_bytes))}\n"
            f"**Type:** {content_type}\n\n"
            f"*Matrix file upload not available. "
            f"Use `!scratch get {identifier}` to view content.*"
        )

    except Exception as e:
        logger.exception(f"Error exporting artifact: {e}")
        return f"❌ Error exporting artifact: {str(e)}"


async def handle_scratch_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !scratch help command - Show scratch pad help.

    Returns:
        Help text
    """
    return (
        "## 📦 Scratch Pad Help\n\n"
        "Persistent artifact storage for ChatOps workflows.\n\n"
        "### Commands\n\n"
        "| Command | Description |\n"
        "|---------|-------------|\n"
        "| `!scratch store <name>` | Store last weaving result |\n"
        "| `!scratch get <name\\|id>` | Retrieve artifact |\n"
        "| `!scratch list [scope]` | List artifacts (user/room/all) |\n"
        "| `!scratch delete <name\\|id>` | Delete artifact |\n"
        "| `!scratch share <name\\|id>` | Share with room |\n"
        "| `!scratch export <name\\|id>` | Upload to Matrix |\n"
        "| `!scratch help` | Show this help |\n\n"
        "### Scopes\n\n"
        "- 👤 **USER**: Private to you in this room\n"
        "- 🚪 **ROOM**: Shared with all room members\n"
        "- 🌐 **GLOBAL**: System-wide (admin only)\n\n"
        "### Workflow Example\n\n"
        "```\n"
        "!weave What is Thompson Sampling?\n"
        "!scratch store sampling_explanation\n"
        "!scratch share sampling_explanation\n"
        "!scratch list room\n"
        "```\n\n"
        "Artifacts expire after 7 days by default."
    )


# ============================================================================
# Handler Class with Decorators (for registry integration)
# ============================================================================

class ScratchPadHandlers:
    """
    Scratch Pad handlers class with decorator-based registration.

    Usage:
        handlers = ScratchPadHandlers()
        registry.register_instance(handlers)
    """

    def __init__(self, manager: Optional['ScratchPadManager'] = None):
        self.manager = manager

    @chatops_handler(
        command="scratch",
        description="Scratch pad artifact storage",
        usage="!scratch <store|get|list|delete|share|export|help> [args]",
        category=HandlerCategory.SYSTEM if REGISTRY_AVAILABLE else None,
        aliases=["sp"]
    )
    async def handle_scratch(
        self,
        room: 'MatrixRoom',
        event: 'RoomMessageText',
        args: str
    ) -> str:
        """
        Main scratch pad command dispatcher.

        Routes to sub-commands: store, get, list, delete, share, export, help.
        """
        parts = args.strip().split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "help"
        subargs = parts[1] if len(parts) > 1 else ""

        manager = self.manager or get_scratchpad_manager()

        if subcommand == "store":
            return await handle_scratch_store(room, event, subargs, manager)
        elif subcommand == "get":
            return await handle_scratch_get(room, event, subargs, manager)
        elif subcommand == "list":
            return await handle_scratch_list(room, event, subargs, manager)
        elif subcommand == "delete":
            return await handle_scratch_delete(room, event, subargs, manager)
        elif subcommand == "share":
            return await handle_scratch_share(room, event, subargs, manager)
        elif subcommand == "export":
            return await handle_scratch_export(room, event, subargs, manager)
        elif subcommand == "help":
            return await handle_scratch_help(room, event, subargs)
        else:
            return (
                f"Unknown subcommand: `{subcommand}`\n\n"
                f"Use `!scratch help` for available commands."
            )


# ============================================================================
# Registration Functions
# ============================================================================

async def register_scratchpad_handlers(
    bot: Any = None,
    manager: Optional['ScratchPadManager'] = None,
    registry: Optional['HandlerRegistry'] = None
) -> 'ScratchPadHandlers':
    """
    Register scratch pad handlers with a Matrix bot.

    Args:
        bot: Matrix bot instance (for command registration)
        manager: ScratchPadManager instance (optional, creates default)
        registry: Handler registry (optional, uses global)

    Returns:
        ScratchPadHandlers instance
    """
    # Create or use provided manager
    if manager is None and SCRATCHPAD_AVAILABLE:
        manager = await create_scratchpad_manager()

    set_scratchpad_manager(manager)

    # Create handlers instance
    handlers = ScratchPadHandlers(manager)

    # Register with registry if available
    if REGISTRY_AVAILABLE:
        if registry is None:
            from hololoom.apps.chatops.handlers.handler_registry import get_global_registry
            registry = get_global_registry()

        registry.register_instance(handlers)
        logger.info("Scratch pad handlers registered with global registry")

    # Register with bot if available
    if bot is not None:
        # Direct bot registration (depends on bot implementation)
        if hasattr(bot, 'register_command'):
            bot.register_command("scratch", handlers.handle_scratch)
            bot.register_command("sp", handlers.handle_scratch)
            logger.info("Scratch pad handlers registered with bot")

    return handlers


def get_handlers() -> Optional['ScratchPadHandlers']:
    """Get scratch pad handlers (creates if needed)."""
    manager = get_scratchpad_manager()
    return ScratchPadHandlers(manager) if manager else None


# ============================================================================
# Integration Hook for Weaving Results
# ============================================================================

def create_weave_result_hook(user_id_extractor=None):
    """
    Create a hook to capture weaving results for scratch storage.

    Usage in orchestrator integration:
        hook = create_weave_result_hook()
        spacetime = await orchestrator.weave(query)
        hook(user_id, spacetime)  # Records for !scratch store

    Args:
        user_id_extractor: Function to extract user_id from context

    Returns:
        Hook function
    """
    def hook(user_id: str, spacetime: 'Spacetime'):
        """Record weaving result for potential scratch storage."""
        record_last_result(user_id, spacetime)
        logger.debug(f"Recorded weaving result for user {user_id}")

    return hook
