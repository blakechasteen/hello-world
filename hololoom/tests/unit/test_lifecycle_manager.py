"""
Unit tests for multi-level memory system (lifecycle_manager.py).

Tests:
- Multi-level memory (USER/SESSION/AGENT scoping)
- Lifecycle management (PERMANENT/TEMPORARY/EPHEMERAL)
- TTL-based expiration
- Automatic routing
- Scoped retrieval
- Background pruning

Based on research from:
- Mem0: Multi-level memory architecture
- Transcript Principle 2: "Separate by lifecycle, not convenience"
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from hololoom.memory.lifecycle_manager import (
    ContextStreamManager,
    ContextStream,
    LifeCycle,
    MemoryScope,
    SCOPE_LIFECYCLE_MAP
)
from hololoom.protocols.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def manager():
    """Create fresh ContextStreamManager for each test."""
    return ContextStreamManager()


@pytest.fixture
def sample_memory():
    """Create sample MemoryShard."""
    return MemoryShard(
        id="test_001",
        text="User prefers concise responses",
        metadata={"type": "preference", "timestamp": datetime.now().isoformat()}
    )


# ============================================================================
# Test: Lifecycle Enum
# ============================================================================

def test_lifecycle_enum():
    """Test LifeCycle enum values."""
    assert LifeCycle.PERMANENT.value == "permanent"
    assert LifeCycle.TEMPORARY.value == "temporary"
    assert LifeCycle.EPHEMERAL.value == "ephemeral"
    assert LifeCycle.WORKING.value == "working"


def test_memory_scope_enum():
    """Test MemoryScope enum values."""
    assert MemoryScope.USER.value == "user"
    assert MemoryScope.SESSION.value == "session"
    assert MemoryScope.AGENT.value == "agent"
    assert MemoryScope.WORKING.value == "working"


def test_scope_lifecycle_mapping():
    """Test default lifecycle for each scope."""
    assert SCOPE_LIFECYCLE_MAP[MemoryScope.USER] == LifeCycle.PERMANENT
    assert SCOPE_LIFECYCLE_MAP[MemoryScope.SESSION] == LifeCycle.EPHEMERAL
    assert SCOPE_LIFECYCLE_MAP[MemoryScope.AGENT] == LifeCycle.TEMPORARY
    assert SCOPE_LIFECYCLE_MAP[MemoryScope.WORKING] == LifeCycle.WORKING


# ============================================================================
# Test: ContextStream
# ============================================================================

def test_context_stream_creation():
    """Test creating a context stream."""
    stream = ContextStream(
        name="test_stream",
        scope=MemoryScope.USER,
        lifecycle=LifeCycle.PERMANENT
    )

    assert stream.name == "test_stream"
    assert stream.scope == MemoryScope.USER
    assert stream.lifecycle == LifeCycle.PERMANENT
    assert len(stream.memories) == 0
    assert stream.summary is None


def test_context_stream_ttl():
    """Test TTL calculation for different lifecycles."""
    permanent_stream = ContextStream("test", MemoryScope.USER, LifeCycle.PERMANENT)
    assert permanent_stream._get_ttl() is None  # Never expire

    temporary_stream = ContextStream("test", MemoryScope.AGENT, LifeCycle.TEMPORARY)
    assert temporary_stream._get_ttl() == timedelta(days=30)

    ephemeral_stream = ContextStream("test", MemoryScope.SESSION, LifeCycle.EPHEMERAL)
    assert ephemeral_stream._get_ttl() == timedelta(hours=1)

    working_stream = ContextStream("test", MemoryScope.WORKING, LifeCycle.WORKING)
    assert working_stream._get_ttl() is None  # Manual cleanup


@pytest.mark.asyncio
async def test_context_stream_add_memory():
    """Test adding memory to stream."""
    stream = ContextStream("test", MemoryScope.USER, LifeCycle.PERMANENT)

    memory = MemoryShard(
        id="test_001",
        text="Test content",
        metadata={"timestamp": datetime.now().isoformat()}
    )

    await stream.add_memory(memory)

    assert len(stream.memories) == 1
    assert stream.memories[0] == memory


def test_context_stream_prune_permanent():
    """Test that PERMANENT memories never expire."""
    stream = ContextStream("test", MemoryScope.USER, LifeCycle.PERMANENT)

    # Add old memory (2 years ago)
    old_memory = MemoryShard(
        id="test_001",
        text="Old preference",
        metadata={"timestamp": (datetime.now() - timedelta(days=730)).isoformat()}
    )
    stream.memories.append(old_memory)

    # Prune
    stream.prune_expired()

    # Should still have memory (PERMANENT never expires)
    assert len(stream.memories) == 1


def test_context_stream_prune_ephemeral():
    """Test that EPHEMERAL memories expire after 1 hour."""
    stream = ContextStream("test", MemoryScope.SESSION, LifeCycle.EPHEMERAL)

    # Add recent memory (30 minutes ago)
    recent_memory = MemoryShard(
        id="test_001",
        text="Recent session state",
        metadata={"timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()}
    )
    stream.memories.append(recent_memory)

    # Add old memory (2 hours ago)
    old_memory = MemoryShard(
        id="test_002",
        text="Old session state",
        metadata={"timestamp": (datetime.now() - timedelta(hours=2)).isoformat()}
    )
    stream.memories.append(old_memory)

    # Prune
    stream.prune_expired()

    # Should only have recent memory
    assert len(stream.memories) == 1
    assert stream.memories[0].id == "test_001"


def test_context_stream_prune_temporary():
    """Test that TEMPORARY memories expire after 30 days."""
    stream = ContextStream("test", MemoryScope.AGENT, LifeCycle.TEMPORARY)

    # Add recent memory (15 days ago)
    recent_memory = MemoryShard(
        id="test_001",
        text="Recent project fact",
        metadata={"timestamp": (datetime.now() - timedelta(days=15)).isoformat()}
    )
    stream.memories.append(recent_memory)

    # Add old memory (45 days ago)
    old_memory = MemoryShard(
        id="test_002",
        text="Old project fact",
        metadata={"timestamp": (datetime.now() - timedelta(days=45)).isoformat()}
    )
    stream.memories.append(old_memory)

    # Prune
    stream.prune_expired()

    # Should only have recent memory
    assert len(stream.memories) == 1
    assert stream.memories[0].id == "test_001"


# ============================================================================
# Test: ContextStreamManager - Default Streams
# ============================================================================

def test_manager_default_streams(manager):
    """Test that manager creates default streams."""
    assert "personal_preferences" in manager.streams
    assert "project_facts" in manager.streams
    assert "session_state" in manager.streams
    assert "working_memory" in manager.streams

    # Check scopes
    assert manager.streams["personal_preferences"].scope == MemoryScope.USER
    assert manager.streams["project_facts"].scope == MemoryScope.AGENT
    assert manager.streams["session_state"].scope == MemoryScope.SESSION
    assert manager.streams["working_memory"].scope == MemoryScope.WORKING

    # Check lifecycles
    assert manager.streams["personal_preferences"].lifecycle == LifeCycle.PERMANENT
    assert manager.streams["project_facts"].lifecycle == LifeCycle.TEMPORARY
    assert manager.streams["session_state"].lifecycle == LifeCycle.EPHEMERAL
    assert manager.streams["working_memory"].lifecycle == LifeCycle.WORKING


# ============================================================================
# Test: ContextStreamManager - Automatic Routing
# ============================================================================

@pytest.mark.asyncio
async def test_routing_preference(manager):
    """Test routing memory with type=preference to personal_preferences."""
    memory = MemoryShard(
        id="test_001",
        text="User prefers concise responses",
        metadata={"type": "preference", "timestamp": datetime.now().isoformat()}
    )

    stream_name = await manager.route_memory(memory)

    assert stream_name == "personal_preferences"
    assert len(manager.streams["personal_preferences"].memories) == 1


@pytest.mark.asyncio
async def test_routing_project(manager):
    """Test routing memory with project_id to project_facts."""
    memory = MemoryShard(
        id="test_001",
        text="Project uses TypeScript",
        metadata={"project_id": "hololoom", "timestamp": datetime.now().isoformat()}
    )

    stream_name = await manager.route_memory(memory)

    assert stream_name == "project_facts"
    assert len(manager.streams["project_facts"].memories) == 1


@pytest.mark.asyncio
async def test_routing_session(manager):
    """Test routing memory with session_only to session_state."""
    memory = MemoryShard(
        id="test_001",
        text="User asked about Python",
        metadata={"session_only": True, "timestamp": datetime.now().isoformat()}
    )

    stream_name = await manager.route_memory(memory)

    assert stream_name == "session_state"
    assert len(manager.streams["session_state"].memories) == 1


@pytest.mark.asyncio
async def test_routing_default_to_working(manager):
    """Test routing memory without metadata to working_memory."""
    memory = MemoryShard(
        id="test_001",
        text="Some general fact",
        metadata={"timestamp": datetime.now().isoformat()}
    )

    stream_name = await manager.route_memory(memory)

    assert stream_name == "working_memory"
    assert len(manager.streams["working_memory"].memories) == 1


# ============================================================================
# Test: ContextStreamManager - Custom Streams
# ============================================================================

def test_create_custom_stream(manager):
    """Test creating a custom stream."""
    stream = manager.create_stream(
        name="custom_project",
        scope=MemoryScope.AGENT,
        lifecycle=LifeCycle.TEMPORARY
    )

    assert stream.name == "custom_project"
    assert "custom_project" in manager.streams
    assert stream.scope == MemoryScope.AGENT
    assert stream.lifecycle == LifeCycle.TEMPORARY


def test_create_custom_stream_default_lifecycle(manager):
    """Test creating custom stream with default lifecycle."""
    stream = manager.create_stream(
        name="custom_user",
        scope=MemoryScope.USER
    )

    # Should use default lifecycle for USER scope (PERMANENT)
    assert stream.lifecycle == LifeCycle.PERMANENT


# ============================================================================
# Test: ContextStreamManager - Scoped Retrieval
# ============================================================================

@pytest.mark.asyncio
async def test_get_all_memories_all_streams(manager):
    """Test retrieving memories from all streams."""
    # Add memory to each stream
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_001", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("project_facts", MemoryShard(
        id="test_002", text="Fact", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("session_state", MemoryShard(
        id="test_003", text="State", metadata={"timestamp": datetime.now().isoformat()}
    ))

    # Retrieve all
    all_memories = manager.get_all_memories()

    assert len(all_memories) == 3


@pytest.mark.asyncio
async def test_get_all_memories_specific_streams(manager):
    """Test retrieving memories from specific streams only."""
    # Add memories
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_001", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("project_facts", MemoryShard(
        id="test_002", text="Fact", metadata={"timestamp": datetime.now().isoformat()}
    ))

    # Retrieve only from personal_preferences
    memories = manager.get_all_memories(stream_names=["personal_preferences"])

    assert len(memories) == 1
    assert memories[0].id == "test_001"


@pytest.mark.asyncio
async def test_get_all_memories_by_scope(manager):
    """Test retrieving memories by scope."""
    # Add memories to different scopes
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_001", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("session_state", MemoryShard(
        id="test_002", text="State", metadata={"timestamp": datetime.now().isoformat()}
    ))

    # Retrieve only USER scope
    user_memories = manager.get_all_memories(scopes=[MemoryScope.USER])

    assert len(user_memories) == 1
    assert user_memories[0].id == "test_001"


@pytest.mark.asyncio
async def test_get_all_memories_by_lifecycle(manager):
    """Test retrieving memories by lifecycle."""
    # Add memories to different lifecycles
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_001", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("session_state", MemoryShard(
        id="test_002", text="State", metadata={"timestamp": datetime.now().isoformat()}
    ))

    # Retrieve only PERMANENT lifecycle
    permanent_memories = manager.get_all_memories(lifecycles=[LifeCycle.PERMANENT])

    assert len(permanent_memories) == 1
    assert permanent_memories[0].id == "test_001"


# ============================================================================
# Test: ContextStreamManager - Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_statistics(manager):
    """Test getting manager statistics."""
    # Add memories to different streams
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_001", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_002", text="Pref2", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("session_state", MemoryShard(
        id="test_003", text="State", metadata={"timestamp": datetime.now().isoformat()}
    ))

    stats = manager.get_statistics()

    assert stats["total_streams"] == 4  # 4 default streams
    assert stats["total_memories"] == 3
    assert stats["memories_by_scope"]["user"] == 2
    assert stats["memories_by_scope"]["session"] == 1
    assert stats["memories_by_lifecycle"]["permanent"] == 2
    assert stats["memories_by_lifecycle"]["ephemeral"] == 1


# ============================================================================
# Test: ContextStreamManager - Background Tasks
# ============================================================================

@pytest.mark.asyncio
async def test_background_pruning():
    """Test background pruning task."""
    manager = ContextStreamManager()

    # Add expired memory to session_state (2 hours old, EPHEMERAL = 1 hour TTL)
    old_memory = MemoryShard(
        id="test_001",
        text="Old session state",
        metadata={"timestamp": (datetime.now() - timedelta(hours=2)).isoformat()}
    )
    manager.streams["session_state"].memories.append(old_memory)

    # Start background tasks
    await manager.start_background_tasks()

    # Wait briefly (background task runs every 5 minutes, but prune_expired is called)
    await asyncio.sleep(0.1)

    # Manually trigger prune to test
    manager.streams["session_state"].prune_expired()

    # Should have pruned expired memory
    assert len(manager.streams["session_state"].memories) == 0

    # Stop background tasks
    await manager.stop_background_tasks()


@pytest.mark.asyncio
async def test_context_manager_support():
    """Test async context manager support."""
    async with ContextStreamManager() as manager:
        # Add memory
        await manager.add_to_stream("working_memory", MemoryShard(
            id="test_001", text="Test", metadata={"timestamp": datetime.now().isoformat()}
        ))

        assert len(manager.streams["working_memory"].memories) == 1

    # Background tasks should be stopped after exiting context


# ============================================================================
# Test: ContextStreamManager - Stream Operations
# ============================================================================

@pytest.mark.asyncio
async def test_clear_stream(manager):
    """Test clearing a stream."""
    # Add memories
    await manager.add_to_stream("working_memory", MemoryShard(
        id="test_001", text="Test", metadata={"timestamp": datetime.now().isoformat()}
    ))

    assert len(manager.streams["working_memory"].memories) == 1

    # Clear
    await manager.clear_stream("working_memory")

    assert len(manager.streams["working_memory"].memories) == 0


@pytest.mark.asyncio
async def test_clear_working_memory(manager):
    """Test clearing working memory (task completion)."""
    # Add memories to working and other streams
    await manager.add_to_stream("working_memory", MemoryShard(
        id="test_001", text="Working", metadata={"timestamp": datetime.now().isoformat()}
    ))
    await manager.add_to_stream("personal_preferences", MemoryShard(
        id="test_002", text="Pref", metadata={"timestamp": datetime.now().isoformat()}
    ))

    # Clear working memory
    await manager.clear_working_memory()

    # Only working memory should be cleared
    assert len(manager.streams["working_memory"].memories) == 0
    assert len(manager.streams["personal_preferences"].memories) == 1


def test_get_stream(manager):
    """Test getting stream by name."""
    stream = manager.get_stream("personal_preferences")

    assert stream is not None
    assert stream.name == "personal_preferences"
    assert stream.scope == MemoryScope.USER


def test_get_stream_not_found(manager):
    """Test getting non-existent stream."""
    stream = manager.get_stream("nonexistent")

    assert stream is None


def test_get_streams_by_scope(manager):
    """Test getting all streams with a given scope."""
    user_streams = manager.get_streams_by_scope(MemoryScope.USER)

    assert len(user_streams) == 1
    assert user_streams[0].name == "personal_preferences"


def test_get_streams_by_lifecycle(manager):
    """Test getting all streams with a given lifecycle."""
    permanent_streams = manager.get_streams_by_lifecycle(LifeCycle.PERMANENT)

    assert len(permanent_streams) == 1
    assert permanent_streams[0].name == "personal_preferences"
