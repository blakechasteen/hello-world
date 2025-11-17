"""
Tests for Sleep-Based Memory Consolidation System

Tests human-like memory processing:
- Access pattern tracking
- Exponential decay algorithm
- Promotion to long-term storage
- Archival of low-importance memories
- Contradiction detection
- Idle-triggered consolidation

Created: 2025-11-17
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from HoloLoom.memory.consolidation import (
    SleepBasedConsolidation,
    ConsolidationConfig,
    MemoryAccessStats
)
from HoloLoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope,
    LifeCycle
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.Documentation.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def stream_manager():
    """Create stream manager with test data."""
    manager = ContextStreamManager()

    # Add some test memories to SESSION scope
    for i in range(10):
        memory = MemoryShard(
            id=f"memory_{i}",
            text=f"Test memory {i}",
            episode="test_episode",
            entities=[f"entity_{i}", f"entity_{i+1}"],
            motifs=[f"motif_{i}"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "importance": 1.0
            }
        )
        asyncio.run(manager.add_memory(memory, scope=MemoryScope.SESSION))

    return manager


@pytest.fixture
def knowledge_graph():
    """Create knowledge graph with test data."""
    kg = KG()

    # Add some test edges
    edges = [
        KGEdge("entity_0", "entity_1", "MENTIONS", 1.0),
        KGEdge("entity_1", "entity_2", "MENTIONS", 1.0),
        KGEdge("entity_2", "type_a", "IS_A", 1.0),
        KGEdge("entity_3", "type_b", "IS_A", 1.0),
    ]
    kg.add_edges(edges)

    return kg


@pytest.fixture
def config():
    """Create test configuration."""
    return ConsolidationConfig(
        enabled=True,
        idle_threshold_hours=1.0,  # 1 hour for tests
        decay_rate=0.95,
        promotion_threshold_accesses=3,  # Lower threshold for tests
        promotion_window_days=7,
        archive_threshold=0.1,
        contradiction_detection=True
    )


@pytest.fixture
def consolidator(stream_manager, knowledge_graph, config):
    """Create sleep-based consolidator."""
    return SleepBasedConsolidation(
        stream_manager=stream_manager,
        knowledge_graph=knowledge_graph,
        config=config
    )


# ============================================================================
# Test Access Pattern Tracking
# ============================================================================

def test_record_access(consolidator):
    """Test recording memory access patterns."""
    # Record access to memory
    consolidator.record_access("memory_0", importance=1.0)

    # Check stats created
    assert "memory_0" in consolidator.access_stats
    stats = consolidator.access_stats["memory_0"]

    assert stats.access_count == 1
    assert stats.base_importance == 1.0
    assert stats.first_access is not None
    assert stats.last_access is not None
    assert len(stats.access_times) == 1

    # Record another access
    consolidator.record_access("memory_0")

    assert stats.access_count == 2
    assert len(stats.access_times) == 2

    print(f"✅ Access tracking working: {stats.access_count} accesses recorded")


def test_multiple_memory_access(consolidator):
    """Test tracking multiple memories."""
    # Record accesses to different memories
    for i in range(5):
        consolidator.record_access(f"memory_{i}", importance=1.0)

    # Check all tracked
    assert len(consolidator.access_stats) == 5

    # Access one memory multiple times
    for _ in range(10):
        consolidator.record_access("memory_0")

    assert consolidator.access_stats["memory_0"].access_count == 11  # 1 + 10
    assert consolidator.access_stats["memory_1"].access_count == 1

    print(f"✅ Multiple memory tracking: {len(consolidator.access_stats)} memories tracked")


# ============================================================================
# Test Decay Algorithm
# ============================================================================

def test_importance_decay_immediate(consolidator):
    """Test decay calculation immediately after access."""
    # Record access
    consolidator.record_access("memory_0", importance=1.0)

    # Compute decay immediately (should be ~1.0)
    importance = consolidator.compute_importance_decay("memory_0")

    assert importance >= 0.99  # Very recent access
    assert importance <= 1.0

    print(f"✅ Immediate decay: {importance:.4f} (expected ~1.0)")


def test_importance_decay_over_time(consolidator):
    """Test exponential decay over simulated time."""
    # Record access
    consolidator.record_access("memory_0", importance=1.0)
    stats = consolidator.access_stats["memory_0"]

    # Simulate 1 day in the past
    stats.last_access = datetime.now() - timedelta(days=1)

    # Compute decay
    importance = consolidator.compute_importance_decay("memory_0")
    expected = 1.0 * (0.95 ** 1)  # decay_rate^days

    assert abs(importance - expected) < 0.01
    print(f"✅ 1-day decay: {importance:.4f} (expected {expected:.4f})")

    # Simulate 7 days
    stats.last_access = datetime.now() - timedelta(days=7)
    importance = consolidator.compute_importance_decay("memory_0")
    expected = 1.0 * (0.95 ** 7)

    assert abs(importance - expected) < 0.01
    print(f"✅ 7-day decay: {importance:.4f} (expected {expected:.4f})")

    # Simulate 30 days
    stats.last_access = datetime.now() - timedelta(days=30)
    importance = consolidator.compute_importance_decay("memory_0")
    expected = 1.0 * (0.95 ** 30)

    assert abs(importance - expected) < 0.01
    print(f"✅ 30-day decay: {importance:.4f} (expected {expected:.4f})")


def test_decay_below_threshold(consolidator):
    """Test decay falling below archive threshold."""
    # Record access
    consolidator.record_access("memory_0", importance=1.0)
    stats = consolidator.access_stats["memory_0"]

    # Simulate 50 days (should decay below 0.1 threshold)
    stats.last_access = datetime.now() - timedelta(days=50)
    importance = consolidator.compute_importance_decay("memory_0")

    assert importance < consolidator.config.archive_threshold
    print(f"✅ Below threshold decay: {importance:.4f} < {consolidator.config.archive_threshold}")


# ============================================================================
# Test Promotion to Long-Term Storage
# ============================================================================

@pytest.mark.asyncio
async def test_promote_to_long_term_success(consolidator):
    """Test successful promotion of frequently accessed memory."""
    memory_id = "memory_0"

    # Record frequent accesses (meet threshold)
    for _ in range(5):
        consolidator.record_access(memory_id, importance=1.0)

    # Promote
    promoted = await consolidator.promote_to_long_term(memory_id)

    assert promoted is True
    assert consolidator.access_stats[memory_id].promoted_to_long_term is True
    assert consolidator.total_memories_promoted == 1

    # Check promoted memory exists in AGENT scope
    agent_memories = consolidator.stream_manager.get_all_memories(
        scopes=[MemoryScope.AGENT]
    )

    promoted_memory = next(
        (m for m in agent_memories if m.metadata.get("promoted_from") == memory_id),
        None
    )

    assert promoted_memory is not None
    assert promoted_memory.metadata["access_count"] == 5

    print(f"✅ Promotion successful: {memory_id} → AGENT scope (accesses: 5)")


@pytest.mark.asyncio
async def test_promote_insufficient_accesses(consolidator):
    """Test promotion fails with insufficient accesses."""
    memory_id = "memory_0"

    # Record few accesses (below threshold)
    for _ in range(2):
        consolidator.record_access(memory_id, importance=1.0)

    # Try to promote
    promoted = await consolidator.promote_to_long_term(memory_id)

    assert promoted is False
    assert consolidator.access_stats[memory_id].promoted_to_long_term is False

    print(f"✅ Promotion blocked: {memory_id} (accesses: 2 < threshold: 3)")


@pytest.mark.asyncio
async def test_promote_already_promoted(consolidator):
    """Test promotion fails if already promoted."""
    memory_id = "memory_0"

    # Record accesses and promote
    for _ in range(5):
        consolidator.record_access(memory_id)

    promoted1 = await consolidator.promote_to_long_term(memory_id)
    assert promoted1 is True

    # Try to promote again
    promoted2 = await consolidator.promote_to_long_term(memory_id)
    assert promoted2 is False

    print(f"✅ Duplicate promotion blocked: {memory_id}")


# ============================================================================
# Test Manual Decay
# ============================================================================

@pytest.mark.asyncio
async def test_manual_decay(consolidator):
    """Test manual decay operation."""
    memory_id = "memory_0"

    # Record access
    consolidator.record_access(memory_id, importance=1.0)

    # Apply manual decay
    await consolidator.decay_memory(memory_id, decay_factor=0.5)

    stats = consolidator.access_stats[memory_id]
    assert abs(stats.base_importance - 0.5) < 0.01
    assert consolidator.total_memories_decayed == 1

    print(f"✅ Manual decay: {memory_id} importance → {stats.base_importance:.2f}")


# ============================================================================
# Test Archival
# ============================================================================

@pytest.mark.asyncio
async def test_archive_contradicted(consolidator):
    """Test archiving contradicted memory."""
    memory_id = "memory_0"

    # Record access
    consolidator.record_access(memory_id, importance=1.0)

    # Archive
    await consolidator.archive_contradicted(memory_id, reason="Contradicted by newer fact")

    # Check stats
    stats = consolidator.access_stats[memory_id]
    assert stats.archived is True
    assert stats.archive_reason == "Contradicted by newer fact"
    assert stats.current_importance == 0.0
    assert consolidator.total_memories_archived == 1

    # Check archived memory exists in USER scope
    user_memories = consolidator.stream_manager.get_all_memories(
        scopes=[MemoryScope.USER]
    )

    archived_memory = next(
        (m for m in user_memories if m.metadata.get("archived_from") == memory_id),
        None
    )

    assert archived_memory is not None
    assert archived_memory.metadata["archived"] is True
    assert archived_memory.metadata["archive_reason"] == "Contradicted by newer fact"

    print(f"✅ Archival successful: {memory_id} → USER scope (permanent)")


@pytest.mark.asyncio
async def test_archive_duplicate_blocked(consolidator):
    """Test archiving fails if already archived."""
    memory_id = "memory_0"

    # Archive once
    consolidator.record_access(memory_id)
    await consolidator.archive_contradicted(memory_id, reason="Test")

    archives_before = consolidator.total_memories_archived

    # Try to archive again
    await consolidator.archive_contradicted(memory_id, reason="Test 2")

    # Should not increase count
    assert consolidator.total_memories_archived == archives_before

    print(f"✅ Duplicate archival blocked: {memory_id}")


# ============================================================================
# Test Idle Detection
# ============================================================================

def test_is_idle_never_queried(consolidator):
    """Test idle detection when never queried."""
    # Never recorded query
    assert consolidator.is_idle() is True

    print("✅ Idle detection: Never queried = idle")


def test_is_idle_recent_query(consolidator):
    """Test idle detection with recent query."""
    # Record recent access
    consolidator.record_access("memory_0")

    # Should not be idle (threshold = 1.0 hour)
    assert consolidator.is_idle() is False

    print("✅ Idle detection: Recent query = not idle")


def test_is_idle_old_query(consolidator):
    """Test idle detection with old query."""
    # Record access
    consolidator.record_access("memory_0")

    # Simulate old query (2 hours ago)
    consolidator.last_query_time = datetime.now() - timedelta(hours=2)

    # Should be idle (threshold = 1.0 hour)
    assert consolidator.is_idle() is True

    print("✅ Idle detection: Old query (2h) = idle")


# ============================================================================
# Test Full Consolidation Cycle
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_during_idle_not_idle(consolidator):
    """Test consolidation skipped when not idle."""
    # Record recent access
    consolidator.record_access("memory_0")

    # Try to consolidate
    result = await consolidator.consolidate_during_idle()

    assert result["consolidated"] is False
    assert result["reason"] == "not_idle"

    print("✅ Consolidation skipped: System not idle")


@pytest.mark.asyncio
async def test_consolidate_during_idle_success(consolidator):
    """Test full consolidation cycle when idle."""
    # Create test scenario:
    # - Memory 0: Frequently accessed (should promote)
    # - Memory 1: Old memory (should decay)
    # - Memory 2: Very old memory (should archive)

    # Frequently accessed memory (will promote)
    for _ in range(5):
        consolidator.record_access("memory_0", importance=1.0)

    # Old memory (will decay but not archive)
    consolidator.record_access("memory_1", importance=1.0)
    stats1 = consolidator.access_stats["memory_1"]
    stats1.last_access = datetime.now() - timedelta(days=7)

    # Very old memory (will archive)
    consolidator.record_access("memory_2", importance=1.0)
    stats2 = consolidator.access_stats["memory_2"]
    stats2.last_access = datetime.now() - timedelta(days=50)

    # Make system idle
    consolidator.last_query_time = datetime.now() - timedelta(hours=2)

    # Run consolidation
    result = await consolidator.consolidate_during_idle()

    # Check results
    assert result["consolidated"] is True
    assert result["total_memories"] > 0
    assert result["promoted_count"] >= 1  # memory_0 promoted
    assert result["archived_count"] >= 1  # memory_2 archived
    assert result["consolidation_time_ms"] > 0

    # Verify promotion
    assert consolidator.access_stats["memory_0"].promoted_to_long_term is True

    # Verify archival
    assert consolidator.access_stats["memory_2"].archived is True

    print(f"✅ Full consolidation cycle:")
    print(f"   - Total memories: {result['total_memories']}")
    print(f"   - Promoted: {result['promoted_count']}")
    print(f"   - Archived: {result['archived_count']}")
    print(f"   - Decayed: {result['decayed_count']}")
    print(f"   - Time: {result['consolidation_time_ms']:.1f}ms")


# ============================================================================
# Test Statistics
# ============================================================================

def test_get_consolidation_statistics(consolidator):
    """Test statistics collection."""
    # Record some activity
    for i in range(5):
        consolidator.record_access(f"memory_{i}")

    # Get stats
    stats = consolidator.get_consolidation_statistics()

    # Check structure
    assert "total_consolidations" in stats
    assert "total_memories_tracked" in stats
    assert "total_accesses" in stats
    assert "avg_importance" in stats
    assert "total_promoted" in stats
    assert "total_archived" in stats
    assert "config" in stats

    # Check values
    assert stats["total_memories_tracked"] == 5
    assert stats["total_accesses"] == 5
    assert stats["is_idle"] in [True, False]

    print(f"✅ Statistics collection:")
    print(f"   - Memories tracked: {stats['total_memories_tracked']}")
    print(f"   - Total accesses: {stats['total_accesses']}")
    print(f"   - Avg importance: {stats['avg_importance']:.2f}")
    print(f"   - Is idle: {stats['is_idle']}")


# ============================================================================
# Test Background Task
# ============================================================================

@pytest.mark.asyncio
async def test_background_consolidation_start_stop(consolidator):
    """Test starting and stopping background consolidation."""
    # Start background task
    await consolidator.start_background_consolidation()

    assert consolidator._running is True
    assert consolidator._consolidation_task is not None

    # Wait a bit
    await asyncio.sleep(0.1)

    # Stop background task
    await consolidator.stop_background_consolidation()

    assert consolidator._running is False

    print("✅ Background task lifecycle: Start → Stop")


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_integration(stream_manager, knowledge_graph):
    """Test complete integration scenario."""
    print("\n" + "="*60)
    print("FULL INTEGRATION TEST: Human-Like Memory Consolidation")
    print("="*60)

    # Create consolidator
    config = ConsolidationConfig(
        idle_threshold_hours=0.0,  # Always idle for test
        promotion_threshold_accesses=3,
        archive_threshold=0.1
    )
    consolidator = SleepBasedConsolidation(stream_manager, knowledge_graph, config)

    # Simulate user interaction session
    print("\n1. User Session - Recording memory accesses:")

    # Frequently accessed memories (will promote)
    for i in [0, 1]:
        for _ in range(5):
            consolidator.record_access(f"memory_{i}", importance=1.0)
        print(f"   - memory_{i}: 5 accesses (frequent)")

    # Occasionally accessed memory (will keep)
    for _ in range(2):
        consolidator.record_access("memory_2", importance=1.0)
    print(f"   - memory_2: 2 accesses (occasional)")

    # Old memory (will decay)
    consolidator.record_access("memory_3", importance=1.0)
    stats3 = consolidator.access_stats["memory_3"]
    stats3.last_access = datetime.now() - timedelta(days=10)
    print(f"   - memory_3: 1 access, 10 days ago (old)")

    # Very old memory (will archive)
    consolidator.record_access("memory_4", importance=1.0)
    stats4 = consolidator.access_stats["memory_4"]
    stats4.last_access = datetime.now() - timedelta(days=60)
    print(f"   - memory_4: 1 access, 60 days ago (very old)")

    # Get initial stats
    initial_stats = consolidator.get_consolidation_statistics()
    print(f"\n2. Initial State:")
    print(f"   - Total memories tracked: {initial_stats['total_memories_tracked']}")
    print(f"   - Total accesses: {initial_stats['total_accesses']}")
    print(f"   - Avg importance: {initial_stats['avg_importance']:.2f}")

    # Run consolidation
    print(f"\n3. Running sleep-based consolidation...")
    result = await consolidator.consolidate_during_idle()

    print(f"\n4. Consolidation Results:")
    print(f"   - Consolidated: {result['consolidated']}")
    print(f"   - Total memories: {result['total_memories']}")
    print(f"   - Promoted to long-term: {result['promoted_count']}")
    print(f"   - Archived (low importance): {result['archived_count']}")
    print(f"   - Decayed: {result['decayed_count']}")
    print(f"   - Time: {result['consolidation_time_ms']:.1f}ms")

    # Get final stats
    final_stats = consolidator.get_consolidation_statistics()
    print(f"\n5. Final State:")
    print(f"   - Total consolidations: {final_stats['total_consolidations']}")
    print(f"   - Memories promoted: {final_stats['total_promoted']}")
    print(f"   - Memories archived: {final_stats['total_archived']}")
    print(f"   - Memories decayed: {final_stats['total_decayed']}")

    # Verify outcomes
    assert result["promoted_count"] >= 2  # memory_0, memory_1
    assert result["archived_count"] >= 1  # memory_4
    assert final_stats["total_promoted"] >= 2
    assert final_stats["total_archived"] >= 1

    print("\n✅ Integration test passed!")
    print("="*60)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("Running Sleep-Based Memory Consolidation Tests\n")

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
