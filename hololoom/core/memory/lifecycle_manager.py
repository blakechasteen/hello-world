"""
Multi-Level Memory System with Lifecycle Management
===================================================

Implements Mem0-style multi-level memory architecture with lifecycle separation.

Based on research from:
- Mem0: USER/SESSION/AGENT scoping
- Transcript Principle 2: "Separate by lifecycle, not convenience"
- LangMem: Background consolidation

Architecture:
- USER level: Personal preferences (PERMANENT, never expire)
- SESSION level: Conversation state (EPHEMERAL, 1 hour TTL)
- AGENT level: System behaviors (TEMPORARY, 30 days TTL)
- WORKING level: Current task context (manual cleanup)

Key Features:
- Automatic TTL-based expiration
- Scoped retrieval (search specific levels only)
- Background summarization for large streams
- Lifecycle-aware persistence
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from hololoom.protocols.types import MemoryShard

logger = logging.getLogger(__name__)


# ============================================================================
# Lifecycle Types
# ============================================================================

class LifeCycle(Enum):
    """Memory lifecycle categories (from Mem0 research)."""
    PERMANENT = "permanent"      # Never expire (user preferences, style)
    TEMPORARY = "temporary"      # Expire after 30 days (project facts)
    EPHEMERAL = "ephemeral"      # Expire after 1 hour (session state)
    WORKING = "working"          # Current task, manual cleanup


class MemoryScope(Enum):
    """Memory scope levels (from Mem0 research)."""
    USER = "user"        # Personal preferences (PERMANENT lifecycle)
    SESSION = "session"  # Conversation state (EPHEMERAL lifecycle)
    AGENT = "agent"      # System behaviors (TEMPORARY lifecycle)
    WORKING = "working"  # Current task context (WORKING lifecycle)


# Default lifecycle for each scope
SCOPE_LIFECYCLE_MAP = {
    MemoryScope.USER: LifeCycle.PERMANENT,
    MemoryScope.SESSION: LifeCycle.EPHEMERAL,
    MemoryScope.AGENT: LifeCycle.TEMPORARY,
    MemoryScope.WORKING: LifeCycle.WORKING,
}


# ============================================================================
# Context Stream
# ============================================================================

@dataclass
class ContextStream:
    """
    Separate context stream with lifecycle and auto-summarization.

    From research:
    - Mem0: Multi-level memory (user/session/agent)
    - Community insight: "Different context windows, RAG-accessed automatically"
    """

    name: str
    scope: MemoryScope
    lifecycle: LifeCycle
    memories: list[MemoryShard] = field(default_factory=list)
    summary: str | None = None  # Auto-generated summary
    last_summarized: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def _get_ttl(self) -> timedelta | None:
        """Time-to-live for memory expiration."""
        if self.lifecycle == LifeCycle.PERMANENT:
            return None  # Never expire
        elif self.lifecycle == LifeCycle.TEMPORARY:
            return timedelta(days=30)
        elif self.lifecycle == LifeCycle.EPHEMERAL:
            return timedelta(hours=1)
        elif self.lifecycle == LifeCycle.WORKING:
            return None  # Manual cleanup only

    async def add_memory(self, memory: MemoryShard):
        """
        Add memory and trigger background summarization if needed.

        Auto-summarize when:
        - More than 100 memories accumulated
        - More than 24 hours since last summary
        """
        self.memories.append(memory)

        should_summarize = (
            len(self.memories) > 100 or
            (self.last_summarized and
             datetime.now() - self.last_summarized > timedelta(hours=24))
        )

        if should_summarize:
            logger.info(f"Stream '{self.name}' triggering background summarization ({len(self.memories)} memories)")
            # Note: Actual summarization would use LLM, implement in future iteration
            # For now, just mark as needing summary
            self.summary = f"[Auto-summary needed for {len(self.memories)} memories]"
            self.last_summarized = datetime.now()

    def prune_expired(self):
        """
        Remove expired memories based on TTL.

        From transcript: "Humans use forgetting as technology - lossy compression"
        """
        ttl = self._get_ttl()
        if ttl is None:
            return  # No expiration

        now = datetime.now()
        before_count = len(self.memories)

        def get_timestamp(m):
            """Extract timestamp from memory (metadata or default to now)."""
            ts_str = m.metadata.get("timestamp")
            if ts_str:
                return datetime.fromisoformat(ts_str)
            return now  # Default to now if no timestamp

        self.memories = [
            m for m in self.memories
            if now - get_timestamp(m) < ttl
        ]

        pruned_count = before_count - len(self.memories)
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} expired memories from stream '{self.name}'")

    def get_active_memories(self) -> list[MemoryShard]:
        """Get non-expired memories."""
        self.prune_expired()
        return self.memories

    def count(self) -> int:
        """Count active (non-expired) memories."""
        return len(self.get_active_memories())


# ============================================================================
# Context Stream Manager
# ============================================================================

class ContextStreamManager:
    """
    Manage multiple context streams with automatic organization.

    From research:
    - Mem0: Multi-level memory (user/session/agent scoping)
    - Transcript: "Multiple context streams with different lifecycles"
    - Community: "Tons of different context windows, RAG-accessed automatically"

    Architecture:
    - Separate streams by lifecycle (PERMANENT/TEMPORARY/EPHEMERAL)
    - Background pruning (every 5 minutes)
    - Automatic routing based on memory metadata
    - Scoped retrieval (search specific streams only)
    """

    def __init__(self):
        # Default streams (user can add custom)
        self.streams: dict[str, ContextStream] = {
            "personal_preferences": ContextStream(
                name="personal_preferences",
                scope=MemoryScope.USER,
                lifecycle=LifeCycle.PERMANENT
            ),
            "project_facts": ContextStream(
                name="project_facts",
                scope=MemoryScope.AGENT,
                lifecycle=LifeCycle.TEMPORARY
            ),
            "session_state": ContextStream(
                name="session_state",
                scope=MemoryScope.SESSION,
                lifecycle=LifeCycle.EPHEMERAL
            ),
            "working_memory": ContextStream(
                name="working_memory",
                scope=MemoryScope.WORKING,
                lifecycle=LifeCycle.WORKING
            ),
        }

        # Background tasks
        self._pruning_task: asyncio.Task | None = None
        self._running = False

    async def start_background_tasks(self):
        """Start background pruning and summarization."""
        if self._running:
            logger.warning("Background tasks already running")
            return

        self._running = True
        self._pruning_task = asyncio.create_task(self._prune_loop())
        logger.info("Started background memory pruning task")

    async def stop_background_tasks(self):
        """Stop background tasks (for cleanup)."""
        self._running = False

        if self._pruning_task:
            self._pruning_task.cancel()
            try:
                await self._pruning_task
            except asyncio.CancelledError:
                pass

            logger.info("Stopped background memory pruning task")

    async def _prune_loop(self):
        """
        Background loop: Prune expired memories every 5 minutes.

        From transcript: "Forgetting is a technology for us"
        """
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutes

                for stream in self.streams.values():
                    stream.prune_expired()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pruning loop: {e}", exc_info=True)

    def create_stream(
        self,
        name: str,
        scope: MemoryScope,
        lifecycle: LifeCycle | None = None
    ) -> ContextStream:
        """
        Create a custom context stream.

        Args:
            name: Stream identifier
            scope: Memory scope (USER/SESSION/AGENT/WORKING)
            lifecycle: Optional override (defaults to scope's default lifecycle)
        """
        if lifecycle is None:
            lifecycle = SCOPE_LIFECYCLE_MAP[scope]

        stream = ContextStream(
            name=name,
            scope=scope,
            lifecycle=lifecycle
        )

        self.streams[name] = stream
        logger.info(f"Created custom stream '{name}' (scope={scope.value}, lifecycle={lifecycle.value})")

        return stream

    async def route_memory(self, memory: MemoryShard) -> str:
        """
        Automatically route memory to appropriate stream.

        Routing rules (based on metadata):
        - metadata["type"] == "preference" → personal_preferences (USER)
        - metadata["project_id"] exists → project_facts (AGENT)
        - metadata["session_only"] == True → session_state (SESSION)
        - default → working_memory (WORKING)

        Returns:
            Stream name where memory was routed
        """
        # Check metadata for routing hints
        if memory.metadata.get("type") == "preference":
            stream_name = "personal_preferences"
        elif memory.metadata.get("project_id"):
            stream_name = "project_facts"
        elif memory.metadata.get("session_only"):
            stream_name = "session_state"
        else:
            stream_name = "working_memory"

        # Add to stream
        stream = self.streams[stream_name]
        await stream.add_memory(memory)

        logger.debug(f"Routed memory to stream '{stream_name}' (scope={stream.scope.value})")

        return stream_name

    async def add_to_stream(self, stream_name: str, memory: MemoryShard):
        """Manually add memory to specific stream."""
        if stream_name not in self.streams:
            raise ValueError(f"Stream '{stream_name}' does not exist")

        await self.streams[stream_name].add_memory(memory)

    def get_stream(self, name: str) -> ContextStream | None:
        """Get stream by name."""
        return self.streams.get(name)

    def get_streams_by_scope(self, scope: MemoryScope) -> list[ContextStream]:
        """Get all streams with given scope."""
        return [
            stream for stream in self.streams.values()
            if stream.scope == scope
        ]

    def get_streams_by_lifecycle(self, lifecycle: LifeCycle) -> list[ContextStream]:
        """Get all streams with given lifecycle."""
        return [
            stream for stream in self.streams.values()
            if stream.lifecycle == lifecycle
        ]

    def get_all_memories(
        self,
        stream_names: list[str] | None = None,
        scopes: list[MemoryScope] | None = None,
        lifecycles: list[LifeCycle] | None = None
    ) -> list[MemoryShard]:
        """
        Get memories from specified streams/scopes/lifecycles.

        Args:
            stream_names: Specific streams to search (None = all)
            scopes: Filter by memory scope (None = all)
            lifecycles: Filter by lifecycle (None = all)

        Returns:
            All active memories matching criteria
        """
        # Determine which streams to search
        if stream_names:
            streams = [self.streams[name] for name in stream_names if name in self.streams]
        elif scopes:
            streams = []
            for scope in scopes:
                streams.extend(self.get_streams_by_scope(scope))
        elif lifecycles:
            streams = []
            for lifecycle in lifecycles:
                streams.extend(self.get_streams_by_lifecycle(lifecycle))
        else:
            streams = list(self.streams.values())

        # Collect all active memories
        all_memories = []
        for stream in streams:
            all_memories.extend(stream.get_active_memories())

        return all_memories

    def get_statistics(self) -> dict:
        """
        Get statistics about memory streams.

        Returns:
            Dict with stream counts, memory counts, lifecycle breakdown
        """
        stats = {
            "total_streams": len(self.streams),
            "streams_by_scope": {},
            "streams_by_lifecycle": {},
            "total_memories": 0,
            "memories_by_scope": {},
            "memories_by_lifecycle": {},
        }

        for stream in self.streams.values():
            # Stream counts
            scope = stream.scope.value
            lifecycle = stream.lifecycle.value

            stats["streams_by_scope"][scope] = stats["streams_by_scope"].get(scope, 0) + 1
            stats["streams_by_lifecycle"][lifecycle] = stats["streams_by_lifecycle"].get(lifecycle, 0) + 1

            # Memory counts
            memory_count = stream.count()
            stats["total_memories"] += memory_count
            stats["memories_by_scope"][scope] = stats["memories_by_scope"].get(scope, 0) + memory_count
            stats["memories_by_lifecycle"][lifecycle] = stats["memories_by_lifecycle"].get(lifecycle, 0) + memory_count

        return stats

    async def clear_stream(self, stream_name: str):
        """Clear all memories from stream (for testing or manual cleanup)."""
        if stream_name not in self.streams:
            raise ValueError(f"Stream '{stream_name}' does not exist")

        self.streams[stream_name].memories.clear()
        logger.info(f"Cleared stream '{stream_name}'")

    async def clear_working_memory(self):
        """Clear working memory (for task completion)."""
        await self.clear_stream("working_memory")


# ============================================================================
# Context Manager Support
# ============================================================================

class ContextStreamManagerContext:
    """Async context manager for ContextStreamManager."""

    def __init__(self, manager: ContextStreamManager):
        self.manager = manager

    async def __aenter__(self):
        await self.manager.start_background_tasks()
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.stop_background_tasks()


# Make ContextStreamManager support async with
def __aenter__(self):
    return ContextStreamManagerContext(self).__aenter__()

def __aexit__(self, exc_type, exc_val, exc_tb):
    return ContextStreamManagerContext(self).__aexit__(exc_type, exc_val, exc_tb)

ContextStreamManager.__aenter__ = __aenter__
ContextStreamManager.__aexit__ = __aexit__
