from __future__ import annotations
"""
Conversation Handlers for Matrix ChatOps
=========================================

Multi-turn conversation support with session tracking and context continuation.

Commands:
- !continue [text] - Continue/expand previous query with optional context
- !context - Show current conversation context

Features:
- Per-user session state tracking
- Query history with timestamps
- Thread-aware context continuation
- Session metrics (queries, duration)
- Scratch pad artifact integration for context passing
- Auto-storage of weaving results (optional)

Usage:
    from hololoom.apps.chatops.handlers.conversation_handlers import (
        register_conversation_handlers,
        ConversationHandlers,
        record_weave_result,
        get_artifact_context_for_user
    )

    # In run_chatops.py:
    register_conversation_handlers(bot, orchestrator)

    # After weaving (auto-record for !scratch store):
    record_weave_result(user_id, room_id, spacetime)

Created: 2025-12-09
Updated: 2025-12-19 - Added scratch pad integration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
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


# ============================================================================
# HoloLoom Dependencies (optional)
# ============================================================================

try:
    from hololoom.protocols.types import Query
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    WeavingOrchestrator = Any
    Query = None


# ============================================================================
# Thread Handler (optional)
# ============================================================================

try:
    from hololoom.apps.chatops.handlers.thread_handler import ThreadContext, ThreadHandler
    THREAD_AVAILABLE = True
except ImportError:
    THREAD_AVAILABLE = False
    ThreadHandler = None
    ThreadContext = None


# ============================================================================
# Scratch Pad Integration (optional)
# ============================================================================

try:
    from hololoom.apps.chatops.handlers.scratchpad_handlers import (
        get_scratchpad_manager,
        record_last_result,
        set_scratchpad_manager,
    )
    from hololoom.apps.chatops.scratchpad import (
        ArtifactReference,
        ArtifactScope,
        ScratchArtifact,
        SessionArtifactContext,
    )
    from hololoom.apps.chatops.scratchpad.manager import ScratchPadManager
    SCRATCHPAD_AVAILABLE = True
except ImportError:
    SCRATCHPAD_AVAILABLE = False
    ScratchArtifact = None
    ArtifactScope = None
    ArtifactReference = None
    SessionArtifactContext = None
    ScratchPadManager = None
    record_last_result = None
    get_scratchpad_manager = None
    set_scratchpad_manager = None


# ============================================================================
# Session Data Structures
# ============================================================================

@dataclass
class UserSessionState:
    """
    Per-user session state for multi-turn conversations.

    Tracks query history, last result, and session metrics.
    """
    user_id: str
    session_start: datetime = field(default_factory=datetime.now)
    last_query: str | None = None
    last_query_time: datetime | None = None
    last_result: Optional['Spacetime'] = None
    last_confidence: float = 0.0
    query_history: list[dict[str, Any]] = field(default_factory=list)

    # Session metrics
    total_queries: int = 0
    successful_queries: int = 0

    def add_query(
        self,
        query: str,
        result: Optional['Spacetime'] = None,
        confidence: float = 0.0
    ) -> None:
        """Record a new query to the session."""
        now = datetime.now()

        self.last_query = query
        self.last_query_time = now
        self.last_result = result
        self.last_confidence = confidence
        self.total_queries += 1

        if confidence >= 0.7:
            self.successful_queries += 1

        # Keep history (last 20 queries)
        self.query_history.append({
            'query': query,
            'timestamp': now,
            'confidence': confidence,
            'has_result': result is not None
        })

        if len(self.query_history) > 20:
            self.query_history = self.query_history[-20:]

    def get_session_duration(self) -> timedelta:
        """Get total session duration."""
        return datetime.now() - self.session_start

    def get_context_window(self, window_minutes: int = 10) -> list[dict[str, Any]]:
        """Get queries within the context window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        return [
            q for q in self.query_history
            if q['timestamp'] > cutoff
        ]

    def summary(self) -> str:
        """Get human-readable session summary."""
        duration = self.get_session_duration()
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)

        success_rate = (
            self.successful_queries / self.total_queries * 100
            if self.total_queries > 0 else 0
        )

        return (
            f"{self.total_queries} queries in "
            f"{minutes}m {seconds}s "
            f"({success_rate:.0f}% success)"
        )


class SessionManager:
    """
    Manages user sessions across rooms.

    Session Key: (room_id, user_id)
    """

    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize session manager.

        Args:
            session_timeout_minutes: Session expiry time (default 30 min)
        """
        self._sessions: dict[tuple, UserSessionState] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

        logger.info("SessionManager initialized")

    def get_session(self, room_id: str, user_id: str) -> UserSessionState:
        """
        Get or create session for user in room.

        Args:
            room_id: Matrix room ID
            user_id: Matrix user ID

        Returns:
            UserSessionState for the user
        """
        key = (room_id, user_id)

        # Check for existing session
        if key in self._sessions:
            session = self._sessions[key]

            # Check if session expired
            if datetime.now() - session.session_start > self.session_timeout:
                # Reset session
                self._sessions[key] = UserSessionState(user_id=user_id)
                logger.debug(f"Session expired for {user_id}, created new")

            return self._sessions[key]

        # Create new session
        session = UserSessionState(user_id=user_id)
        self._sessions[key] = session
        logger.debug(f"Created new session for {user_id} in {room_id}")

        return session

    def clear_session(self, room_id: str, user_id: str) -> bool:
        """Clear session for user in room."""
        key = (room_id, user_id)
        if key in self._sessions:
            del self._sessions[key]
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get session statistics."""
        active = sum(
            1 for s in self._sessions.values()
            if datetime.now() - s.session_start < self.session_timeout
        )

        total_queries = sum(s.total_queries for s in self._sessions.values())

        return {
            'active_sessions': active,
            'total_sessions': len(self._sessions),
            'total_queries': total_queries
        }


# Global session manager
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# ============================================================================
# Scratch Pad Integration Helpers
# ============================================================================

async def record_weave_result(
    user_id: str,
    room_id: str,
    spacetime: 'Spacetime',
    auto_store: bool = False
) -> None:
    """
    Record a weaving result for scratch pad integration.

    This function should be called after every successful weave operation
    to enable:
    1. `!scratch store` command to store the last result
    2. Auto-storage of artifacts from Spacetime (if enabled)

    Args:
        user_id: Matrix user ID
        room_id: Matrix room ID
        spacetime: The Spacetime result from weaving
        auto_store: If True, automatically store any artifacts in Spacetime
    """
    if not SCRATCHPAD_AVAILABLE:
        return

    # Record for !scratch store command
    if record_last_result:
        record_last_result(user_id, spacetime)

    # Auto-store artifacts if enabled
    if auto_store:
        await _auto_store_artifacts(user_id, room_id, spacetime)


async def _auto_store_artifacts(
    user_id: str,
    room_id: str,
    spacetime: 'Spacetime'
) -> None:
    """
    Automatically store artifacts from a Spacetime result.

    Called when auto_store=True in record_weave_result().

    Args:
        user_id: Matrix user ID
        room_id: Matrix room ID
        spacetime: The Spacetime result containing artifacts
    """
    if not SCRATCHPAD_AVAILABLE:
        return

    manager = get_scratchpad_manager() if get_scratchpad_manager else None
    if not manager:
        return

    # Check if Spacetime has artifacts
    artifacts = getattr(spacetime, 'artifacts', None)
    if not artifacts:
        return

    # Store each artifact
    for artifact in artifacts:
        try:
            # Extract content and metadata
            content = getattr(artifact, 'content', None)
            if content is None:
                continue

            name = getattr(artifact, 'name', None)
            if not name:
                # Generate name from artifact type if available
                artifact_type = getattr(artifact, 'type', 'artifact')
                name = f"auto_{artifact_type}_{spacetime.spacetime_id[:8]}"

            # Get artifact type
            from hololoom.apps.chatops.scratchpad import ArtifactType
            artifact_type_enum = ArtifactType.DATA  # Default

            type_value = getattr(artifact, 'type', None)
            if type_value:
                type_str = type_value.value if hasattr(type_value, 'value') else str(type_value)
                try:
                    artifact_type_enum = ArtifactType(type_str.lower())
                except ValueError:
                    artifact_type_enum = ArtifactType.DATA

            # Store artifact
            result = await manager.store(
                name=name,
                content=content if isinstance(content, bytes) else str(content).encode('utf-8'),
                artifact_type=artifact_type_enum,
                owner_user_id=user_id,
                owner_room_id=room_id,
                source_spacetime_id=getattr(spacetime, 'spacetime_id', None),
                metadata={
                    'auto_stored': True,
                    'source': 'weave_result'
                }
            )

            if result.success:
                logger.debug(f"Auto-stored artifact: {name}")
            else:
                logger.warning(f"Failed to auto-store artifact {name}: {result.error}")

        except Exception as e:
            logger.warning(f"Error auto-storing artifact: {e}")


async def get_artifact_context_for_user(
    user_id: str,
    room_id: str,
    limit: int = 5
) -> str:
    """
    Get artifact context summary for a user's session.

    Returns a formatted string describing available artifacts
    that can be included in continuation prompts.

    Args:
        user_id: Matrix user ID
        room_id: Matrix room ID
        limit: Maximum number of artifacts to include

    Returns:
        Formatted string with artifact context, or empty string if none
    """
    if not SCRATCHPAD_AVAILABLE:
        return ""

    manager = get_scratchpad_manager() if get_scratchpad_manager else None
    if not manager:
        return ""

    try:
        # List user's artifacts
        result = await manager.list_artifacts(
            owner_user_id=user_id,
            owner_room_id=room_id,
            limit=limit
        )

        if not result.success or not result.artifacts:
            return ""

        # Build context string
        lines = ["Available artifacts from previous queries:"]
        for ref in result.artifacts[:limit]:
            size_kb = ref.size_bytes / 1024
            lines.append(f"  - {ref.name} ({ref.artifact_type.value}, {size_kb:.1f}KB)")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Error getting artifact context: {e}")
        return ""


def get_artifact_references_for_session(
    user_id: str,
    room_id: str,
    session_id: str | None = None
) -> list['ArtifactReference']:
    """
    Get artifact references for a user's session.

    Synchronous version for use in session context.

    Args:
        user_id: Matrix user ID
        room_id: Matrix room ID
        session_id: Optional session ID filter

    Returns:
        List of ArtifactReference objects
    """
    if not SCRATCHPAD_AVAILABLE:
        return []

    manager = get_scratchpad_manager() if get_scratchpad_manager else None
    if not manager:
        return []

    # Note: This is a synchronous wrapper - for async, use get_artifact_context_for_user
    # In a real implementation, you might want to cache this
    return []


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_continue(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: 'WeavingOrchestrator',
    include_artifacts: bool = True
) -> str:
    """
    Handle !continue command - Continue/expand previous query.

    Usage:
        !continue                    - Re-run last query with deeper context
        !continue tell me more       - Expand on previous topic
        !continue about the examples - Continue with specific focus

    Args:
        room: Matrix room
        event: Message event
        args: Optional continuation text
        orchestrator: WeavingOrchestrator instance
        include_artifacts: If True, include artifact context from scratch pad

    Returns:
        Continued/expanded response
    """
    if not ORCHESTRATOR_AVAILABLE:
        return "Orchestrator not available"

    # Get session
    user_id = getattr(event, 'sender', 'unknown')
    room_id = getattr(room, 'room_id', 'unknown')

    session_mgr = get_session_manager()
    session = session_mgr.get_session(room_id, user_id)

    # Check if we have a previous query
    if not session.last_query:
        return (
            "No previous query to continue from.\n\n"
            "Use `!continue` after asking a question with `!weave` or similar commands."
        )

    # Build continuation query
    continuation_text = args.strip() if args else ""

    if continuation_text:
        # User provided continuation context
        full_query = f"{session.last_query}\n\nContinuation: {continuation_text}"
    else:
        # Re-run with request for more detail
        full_query = f"{session.last_query}\n\nPlease provide more detail and depth."

    # Add artifact context if available
    artifact_context = ""
    if include_artifacts and SCRATCHPAD_AVAILABLE:
        try:
            artifact_context = await get_artifact_context_for_user(user_id, room_id, limit=5)
            if artifact_context:
                full_query = f"{full_query}\n\n{artifact_context}"
        except Exception as e:
            logger.debug(f"Could not get artifact context: {e}")

    try:
        # Create query with context from last result
        if Query:
            query = Query(text=full_query)
        else:
            query = type('Query', (), {'text': full_query})()

        # Get previous context if available
        context = {}
        if session.last_result and hasattr(session.last_result, 'response'):
            context['previous_response'] = session.last_result.response[:500]

        # Execute continuation
        spacetime = await orchestrator.weave(query)

        # Update session
        confidence = getattr(spacetime, 'confidence', 0.0)
        session.add_query(full_query, spacetime, confidence)

        # Record result for scratch pad integration
        await record_weave_result(user_id, room_id, spacetime, auto_store=False)

        # Format response
        response_text = getattr(spacetime, 'response', str(spacetime))

        response = "## Continuing Previous Query\n\n"
        response += f"**Original**: {session.query_history[-2]['query'][:100] if len(session.query_history) > 1 else session.last_query[:100]}...\n"

        if continuation_text:
            response += f"**Continuation**: {continuation_text}\n"

        if artifact_context:
            response += f"**Artifacts**: {len(artifact_context.split(chr(10))) - 1} available\n"

        response += f"\n### Extended Response\n\n{response_text}\n\n"
        response += f"**Confidence**: {confidence:.2f}\n"
        response += f"**Session**: {session.summary()}\n"

        return response

    except Exception as e:
        logger.error(f"Error in continue: {e}")
        return f"Continuation failed: {str(e)}"


async def handle_context(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !context command - Show current conversation context.

    Shows:
        - Session duration and statistics
        - Recent queries in context window
        - Last query and confidence
        - Thread context (if in thread)

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (e.g., --full for full history)

    Returns:
        Context information
    """
    # Get session
    user_id = getattr(event, 'sender', 'unknown')
    room_id = getattr(room, 'room_id', 'unknown')

    session_mgr = get_session_manager()
    session = session_mgr.get_session(room_id, user_id)

    # Parse args
    show_full = '--full' in args if args else False

    # Build response
    response = "## Current Conversation Context\n\n"

    # Session info
    duration = session.get_session_duration()
    minutes = int(duration.total_seconds() / 60)
    seconds = int(duration.total_seconds() % 60)

    response += f"**Session Duration**: {minutes}m {seconds}s\n"
    response += f"**Total Queries**: {session.total_queries}\n"

    if session.total_queries > 0:
        success_rate = session.successful_queries / session.total_queries * 100
        response += f"**Success Rate**: {success_rate:.0f}%\n"

    response += "\n"

    # Last query
    if session.last_query:
        response += "### Last Query\n"
        response += f"**Query**: {session.last_query[:200]}{'...' if len(session.last_query) > 200 else ''}\n"
        response += f"**Confidence**: {session.last_confidence:.2f}\n"

        if session.last_query_time:
            ago = datetime.now() - session.last_query_time
            response += f"**Time**: {int(ago.total_seconds())}s ago\n"

        response += "\n"

    # Context window (last 10 minutes)
    context_queries = session.get_context_window(10)

    if context_queries:
        response += "### Context Window (last 10 min)\n"
        response += f"_{len(context_queries)} queries_\n\n"

        # Show recent queries
        queries_to_show = context_queries[-5:] if not show_full else context_queries

        for i, q in enumerate(queries_to_show, 1):
            ago = datetime.now() - q['timestamp']
            conf_icon = "+" if q['confidence'] >= 0.7 else "~" if q['confidence'] >= 0.4 else "-"
            query_preview = q['query'][:60] + "..." if len(q['query']) > 60 else q['query']
            response += f"{i}. [{conf_icon}] {query_preview} _({int(ago.total_seconds())}s ago)_\n"

        if not show_full and len(context_queries) > 5:
            response += f"\n_Use `!context --full` to see all {len(context_queries)} queries_\n"
    else:
        response += "### Context Window\n"
        response += "_No recent queries in context window_\n"

    # Scratch pad artifacts
    if SCRATCHPAD_AVAILABLE:
        try:
            artifact_context = await get_artifact_context_for_user(user_id, room_id, limit=10)
            if artifact_context:
                response += "\n### Stored Artifacts\n"
                # Parse the artifact context (skip the header line)
                artifact_lines = artifact_context.split('\n')[1:]
                response += f"_{len(artifact_lines)} artifacts available_\n\n"
                for line in artifact_lines[:5 if not show_full else 10]:
                    response += f"{line}\n"
                if not show_full and len(artifact_lines) > 5:
                    response += f"\n_Use `!context --full` to see all {len(artifact_lines)} artifacts_\n"
                response += "\n_Use `!scratch get <name>` to retrieve an artifact_\n"
            else:
                response += "\n### Stored Artifacts\n"
                response += "_No artifacts stored. Use `!scratch store <name>` after a query._\n"
        except Exception as e:
            logger.debug(f"Could not get artifact context for !context: {e}")
            response += "\n### Stored Artifacts\n"
            response += "_Artifact storage not available._\n"

    # Manager stats
    manager_stats = session_mgr.get_statistics()
    response += "\n### Global Stats\n"
    response += f"**Active Sessions**: {manager_stats['active_sessions']}\n"
    response += f"**Total Queries (All Sessions)**: {manager_stats['total_queries']}\n"

    return response


async def handle_conversation_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """Handle !conversation help command."""
    return """## Multi-Turn Conversation Commands

**Commands:**
- `!continue` - Re-run last query with deeper context
- `!continue <text>` - Expand on previous topic with specific focus
- `!context` - Show current conversation context
- `!context --full` - Show full query history

**How It Works:**
1. Ask a question with `!weave <question>`
2. Use `!continue` to expand on the response
3. Use `!continue tell me more about X` for specific follow-up
4. Use `!context` to see your conversation state

**Session Features:**
- Per-user session tracking
- 30-minute session timeout
- Context window (last 10 minutes)
- Success rate tracking

**Examples:**
```
!weave What is Thompson Sampling?
!continue tell me more about the Bayesian aspects
!continue with code examples
!context
```

**Tips:**
- Use `!continue` multiple times for deeper exploration
- Check `!context` to see your query history
- Sessions reset after 30 minutes of inactivity
"""


# ============================================================================
# Registration
# ============================================================================

def register_conversation_handlers(
    bot,
    orchestrator: 'WeavingOrchestrator'
) -> None:
    """
    Register all conversation command handlers.

    Args:
        bot: Matrix bot instance
        orchestrator: WeavingOrchestrator instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, conversation handlers not registered")
        return

    logger.info("Registering conversation command handlers")

    # Define command map
    command_map = {
        "continue": lambda room, event, args: handle_continue(room, event, args, orchestrator),
        "context": handle_context,
        "conversation": handle_conversation_help,
        "conversation help": handle_conversation_help,
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(cmd, handler)

    logger.info(f"Registered {len(command_map)} conversation commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class ConversationHandlers:
    """
    Decorator-based ChatOps handlers for multi-turn conversations.

    Usage:
        from hololoom.apps.chatops.handlers.conversation_handlers import ConversationHandlers

        handlers = ConversationHandlers(orchestrator=weaving_orchestrator)
        registry.register_instance(handlers)
    """

    def __init__(self, orchestrator: 'WeavingOrchestrator'):
        """
        Initialize with WeavingOrchestrator instance.

        Args:
            orchestrator: WeavingOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.session_manager = get_session_manager()

    if REGISTRY_AVAILABLE:
        @HandlerRegistry.register(
            command="continue",
            description="Continue/expand previous query with optional context",
            usage="!continue [additional context]",
            category=HandlerCategory.QUERY,
            aliases=["cont", "more"]
        )
        async def handle_continue(self, room, event, args: str) -> str:
            """Handle !continue command."""
            return await handle_continue(room, event, args, self.orchestrator)

        @HandlerRegistry.register(
            command="context",
            description="Show current conversation context",
            usage="!context [--full]",
            category=HandlerCategory.SYSTEM,
            aliases=["ctx"]
        )
        async def handle_context(self, room, event, args: str) -> str:
            """Handle !context command."""
            return await handle_context(room, event, args)

        @HandlerRegistry.register(
            command="conversation help",
            description="Show conversation command help",
            usage="!conversation help",
            category=HandlerCategory.SYSTEM,
            hidden=True
        )
        async def handle_help(self, room, event, args: str) -> str:
            """Handle !conversation help command."""
            return await handle_conversation_help(room, event, args)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Conversation Handlers Demo")
    print("=" * 80)
    print()

    # Create session manager
    session_mgr = SessionManager()

    # Simulate user session
    session = session_mgr.get_session("!room:matrix.org", "@alice:matrix.org")

    # Add some queries
    print("Simulating conversation...")
    session.add_query("What is Thompson Sampling?", confidence=0.85)
    session.add_query("Tell me more about the Bayesian aspects", confidence=0.78)
    session.add_query("Show me a code example", confidence=0.92)

    print(f"\nSession summary: {session.summary()}")

    # Show context window
    context = session.get_context_window(10)
    print(f"\nContext window ({len(context)} queries):")
    for q in context:
        print(f"  - {q['query'][:50]}... (conf: {q['confidence']:.2f})")

    # Stats
    print(f"\nManager stats: {session_mgr.get_statistics()}")

    print("\n" + "=" * 80)
    print("Demo complete!")
