"""
Unit tests for Agent Memory Tools (Week 2 implementation).

Tests:
- Agent memory operations (store/search/update/delete)
- Automatic routing and scoping
- Soft-delete archival
- Hybrid search strategies
- Statistics and metadata

Based on LangMem research: "Agents decide what to store, not passive accumulation"
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from hololoom.agentic.memory_tools import (
    AgentMemoryTools,
    MemoryStoreResult,
    MemorySearchResult,
    MemoryUpdateResult,
    MemoryDeleteResult
)
from hololoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from hololoom.memory.graph import KG


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def stream_manager():
    """Create fresh ContextStreamManager for each test."""
    return ContextStreamManager()


@pytest.fixture
def knowledge_graph():
    """Create fresh KG for each test."""
    return KG()


@pytest.fixture
def agent_tools(stream_manager, knowledge_graph):
    """Create AgentMemoryTools with manager and KG."""
    return AgentMemoryTools(
        stream_manager=stream_manager,
        knowledge_graph=knowledge_graph,
        enable_archival=True
    )


# ============================================================================
# Test: Store Operation
# ============================================================================

@pytest.mark.asyncio
async def test_store_with_explicit_scope(agent_tools):
    """Test storing memory with explicit scope."""
    result = await agent_tools.store(
        content="User prefers TypeScript",
        scope=MemoryScope.USER,
        metadata={"type": "preference"},
        importance=0.9
    )

    assert result.success is True
    assert result.scope == MemoryScope.USER
    assert result.lifecycle == LifeCycle.PERMANENT
    assert "personal_preferences" in result.stream_name


@pytest.mark.asyncio
async def test_store_with_automatic_routing(agent_tools):
    """Test storing memory with automatic routing (metadata-based)."""
    result = await agent_tools.store(
        content="Project uses Python 3.12",
        metadata={"project_id": "hololoom"},
        importance=0.7
    )

    assert result.success is True
    assert result.stream_name == "project_facts"
    assert result.scope == MemoryScope.AGENT


@pytest.mark.asyncio
async def test_store_with_entities(agent_tools, knowledge_graph):
    """Test storing memory with entities (adds to KG)."""
    result = await agent_tools.store(
        content="Alice works on HoloLoom with Bob",
        entities=["Alice", "hololoom", "Bob"],
        importance=0.8
    )

    assert result.success is True

    # Check KG has entities
    assert "Alice" in knowledge_graph.G.nodes
    assert "hololoom" in knowledge_graph.G.nodes
    assert "Bob" in knowledge_graph.G.nodes


@pytest.mark.asyncio
async def test_store_importance_in_metadata(agent_tools):
    """Test that importance is stored in metadata."""
    result = await agent_tools.store(
        content="Critical security update",
        importance=1.0
    )

    # Retrieve and check metadata
    memories = agent_tools.stream_manager.get_all_memories()
    stored_memory = [m for m in memories if m.id == result.memory_id][0]

    assert stored_memory.metadata.get("importance") == 1.0


# ============================================================================
# Test: Search Operation
# ============================================================================

@pytest.mark.asyncio
async def test_search_basic(agent_tools):
    """Test basic memory search."""
    # Store some memories
    await agent_tools.store("Python is a programming language", importance=0.8)
    await agent_tools.store("Java is also a programming language", importance=0.7)
    await agent_tools.store("TypeScript extends JavaScript", importance=0.6)

    # Search
    result = await agent_tools.search(
        query="programming language",
        limit=10
    )

    assert result.total_found >= 2
    assert len(result.memories) >= 2
    assert result.search_time_ms >= 0  # Search time can be 0ms for fast operations


@pytest.mark.asyncio
async def test_search_with_scope_filter(agent_tools):
    """Test search filtered by scope."""
    # Store in different scopes
    await agent_tools.store(
        "User prefers dark mode",
        scope=MemoryScope.USER,
        importance=0.9
    )
    await agent_tools.store(
        "Session started at 10am",
        metadata={"session_only": True},
        importance=0.5
    )

    # Search only USER scope
    result = await agent_tools.search(
        query="prefers",
        scopes=[MemoryScope.USER],
        limit=10
    )

    assert result.total_found >= 1
    assert all(m.text.find("dark mode") >= 0 for m in result.memories)


@pytest.mark.asyncio
async def test_search_with_importance_filter(agent_tools):
    """Test search filtered by minimum importance."""
    # Store with different importance scores
    await agent_tools.store("Low importance note", importance=0.3)
    await agent_tools.store("High importance alert", importance=0.9)

    # Search with min_importance filter
    result = await agent_tools.search(
        query="importance",
        min_importance=0.7,
        limit=10
    )

    assert result.total_found >= 1
    # Should only return high-importance memory
    assert all("alert" in m.text for m in result.memories)


@pytest.mark.asyncio
async def test_search_ranking_by_importance(agent_tools):
    """Test that search ranks by importance."""
    # Store memories with different importance
    await agent_tools.store("Keyword match low", importance=0.3)
    await agent_tools.store("Keyword match high", importance=0.9)

    result = await agent_tools.search(
        query="keyword match",
        limit=10
    )

    # Higher importance should rank first
    if len(result.memories) >= 2:
        first_importance = result.memories[0].metadata.get("importance", 0.0)
        second_importance = result.memories[1].metadata.get("importance", 0.0)
        assert first_importance >= second_importance


# ============================================================================
# Test: Update Operation
# ============================================================================

@pytest.mark.asyncio
async def test_update_memory(agent_tools):
    """Test updating existing memory (temporal invalidation)."""
    # Store original
    store_result = await agent_tools.store(
        "Original content",
        importance=0.7
    )
    original_id = store_result.memory_id

    # Update
    update_result = await agent_tools.update(
        memory_id=original_id,
        new_content="Updated content"
    )

    assert update_result.success is True
    assert update_result.old_version_archived is True
    assert update_result.new_version_id != original_id

    # Check new version exists
    memories = agent_tools.stream_manager.get_all_memories()
    new_memory = [m for m in memories if m.id == update_result.new_version_id]
    assert len(new_memory) == 1
    assert new_memory[0].text == "Updated content"


@pytest.mark.asyncio
async def test_update_preserves_metadata(agent_tools):
    """Test that update preserves previous_version in metadata."""
    # Store and update
    store_result = await agent_tools.store("Original", importance=0.8)
    update_result = await agent_tools.update(
        store_result.memory_id,
        "Updated"
    )

    # Check metadata
    memories = agent_tools.stream_manager.get_all_memories()
    new_memory = [m for m in memories if m.id == update_result.new_version_id][0]

    assert new_memory.metadata.get("previous_version") == store_result.memory_id


@pytest.mark.asyncio
async def test_update_archives_old_version(agent_tools):
    """Test that update archives old version."""
    store_result = await agent_tools.store("Original", importance=0.7)
    await agent_tools.update(store_result.memory_id, "Updated")

    # Check archive stream
    archive_stream = agent_tools.archive_stream
    archived = [
        m for m in archive_stream.memories
        if m.metadata.get("original_id") == store_result.memory_id
    ]

    assert len(archived) == 1
    assert "Original" in archived[0].text


@pytest.mark.asyncio
async def test_update_nonexistent_memory(agent_tools):
    """Test updating nonexistent memory returns error."""
    result = await agent_tools.update(
        memory_id="nonexistent_id_12345",
        new_content="Updated"
    )

    assert result.success is False
    assert "not found" in result.message.lower()


# ============================================================================
# Test: Delete Operation
# ============================================================================

@pytest.mark.asyncio
async def test_delete_memory(agent_tools):
    """Test deleting memory (soft delete with archival)."""
    # Store memory
    store_result = await agent_tools.store("To be deleted", importance=0.5)

    # Delete
    delete_result = await agent_tools.delete(
        memory_id=store_result.memory_id,
        reason="test_cleanup"
    )

    assert delete_result.success is True
    assert delete_result.archived is True

    # Check memory is gone from active streams
    active_memories = agent_tools.stream_manager.get_all_memories()
    assert not any(m.id == store_result.memory_id for m in active_memories)


@pytest.mark.asyncio
async def test_delete_archives_with_reason(agent_tools):
    """Test that delete archives memory with reason."""
    store_result = await agent_tools.store("To be deleted", importance=0.5)
    await agent_tools.delete(
        store_result.memory_id,
        reason="outdated_information"
    )

    # Check archive
    archive_stream = agent_tools.archive_stream
    archived = [
        m for m in archive_stream.memories
        if m.metadata.get("original_id") == store_result.memory_id
    ]

    assert len(archived) == 1
    assert archived[0].metadata.get("archived_reason") == "outdated_information"


@pytest.mark.asyncio
async def test_delete_nonexistent_memory(agent_tools):
    """Test deleting nonexistent memory returns error."""
    result = await agent_tools.delete("nonexistent_id_12345")

    assert result.success is False
    assert "not found" in result.message.lower()


# ============================================================================
# Test: Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_statistics_includes_archived(agent_tools):
    """Test that statistics include archived memory count."""
    # Store and delete some memories
    store1 = await agent_tools.store("Memory 1", importance=0.7)
    store2 = await agent_tools.store("Memory 2", importance=0.8)
    await agent_tools.delete(store1.memory_id)

    stats = agent_tools.get_statistics()

    assert "archived_memories" in stats
    assert stats["archived_memories"] >= 1


@pytest.mark.asyncio
async def test_statistics_tracks_scope_distribution(agent_tools):
    """Test that statistics show memory distribution by scope."""
    # Store in different scopes
    await agent_tools.store("User pref", scope=MemoryScope.USER, importance=0.9)
    await agent_tools.store("Session state", metadata={"session_only": True}, importance=0.5)

    stats = agent_tools.get_statistics()

    assert "memories_by_scope" in stats
    assert stats["memories_by_scope"].get("user", 0) >= 1
    assert stats["memories_by_scope"].get("session", 0) >= 1


# ============================================================================
# Test: Full Lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_full_memory_lifecycle(agent_tools):
    """Test complete memory lifecycle: store → search → update → delete."""
    # 1. Store
    store_result = await agent_tools.store(
        "Initial version of fact",
        importance=0.8,
        entities=["Fact", "Version1"]
    )
    assert store_result.success is True

    # 2. Search (should find it)
    search_result = await agent_tools.search("version")
    assert len(search_result.memories) >= 1

    # 3. Update
    update_result = await agent_tools.update(
        store_result.memory_id,
        "Updated version of fact"
    )
    assert update_result.success is True

    # 4. Search again (should find updated version)
    search_result2 = await agent_tools.search("updated")
    assert len(search_result2.memories) >= 1
    assert "Updated" in search_result2.memories[0].text

    # 5. Delete
    delete_result = await agent_tools.delete(update_result.new_version_id)
    assert delete_result.success is True

    # 6. Verify archived (not in active memories)
    active_memories = agent_tools.stream_manager.get_all_memories()
    assert not any(m.id == update_result.new_version_id for m in active_memories)
    assert agent_tools.archive_stream.count() >= 2  # Original + updated versions


@pytest.mark.asyncio
async def test_multiple_agents_concurrent_access(stream_manager, knowledge_graph):
    """Test multiple agent tool instances accessing same manager."""
    agent1 = AgentMemoryTools(stream_manager, knowledge_graph)
    agent2 = AgentMemoryTools(stream_manager, knowledge_graph)

    # Both agents store memories
    result1 = await agent1.store("Agent 1 memory", importance=0.7)
    result2 = await agent2.store("Agent 2 memory", importance=0.8)

    # Both should see all memories
    search1 = await agent1.search("memory")
    search2 = await agent2.search("memory")

    assert search1.total_found >= 2
    assert search2.total_found >= 2
