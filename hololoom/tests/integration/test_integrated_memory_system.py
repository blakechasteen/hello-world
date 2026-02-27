"""
Integration tests for complete memory system (Weeks 1-4).

Tests full pipeline:
- Multi-level memory (Week 1)
- Agent operations (Week 2)
- LLM consolidation (Week 3)
- Hybrid retrieval (Week 4)
"""

import pytest
import asyncio
from datetime import datetime

from HoloLoom.memory.integrated_memory_system import (
    IntegratedMemorySystem,
    IntegratedMemoryConfig,
    create_integrated_memory_system,
    create_production_memory_system
)
from HoloLoom.memory.lifecycle_manager import MemoryScope
from HoloLoom.memory.consolidation import ConsolidationStrategy


# ============================================================================
# Test: Basic Integration
# ============================================================================

@pytest.mark.asyncio
async def test_integrated_system_creation():
    """Test creating integrated memory system."""
    system = create_integrated_memory_system()

    assert system.stream_manager is not None
    assert system.agent_tools is not None
    assert system.consolidator is not None
    assert system.hybrid_retriever is not None


@pytest.mark.asyncio
async def test_integrated_system_lifecycle():
    """Test integrated system lifecycle management."""
    async with create_integrated_memory_system() as system:
        # System should work within context
        result = await system.store("Test memory")
        assert result.success is True

    # System should be closed after context


# ============================================================================
# Test: Week 1+2 Integration (Memory + Agent Tools)
# ============================================================================

@pytest.mark.asyncio
async def test_store_and_simple_search():
    """Test storing and searching memories (Weeks 1+2)."""
    system = create_integrated_memory_system()

    # Store memories
    await system.store(
        "Thompson Sampling is a Bayesian approach",
        importance=0.9,
        entities=["ThompsonSampling", "Bayesian"]
    )

    await system.store(
        "Epsilon-greedy is simple but effective",
        importance=0.7,
        entities=["EpsilonGreedy"]
    )

    # Simple search (Week 2 agent tools)
    result = await system.search_simple("Thompson", limit=5)

    assert result.total_found >= 1
    assert len(result.memories) >= 1


@pytest.mark.asyncio
async def test_store_with_scoping():
    """Test memory storage with different scopes (Week 1)."""
    system = create_integrated_memory_system()

    # Store in USER scope
    user_result = await system.store(
        "User prefers dark mode",
        scope=MemoryScope.USER,
        importance=0.9
    )

    assert user_result.scope == MemoryScope.USER

    # Store in SESSION scope
    session_result = await system.store(
        "Current session started",
        metadata={"session_only": True},
        importance=0.5
    )

    assert session_result.success is True


@pytest.mark.asyncio
async def test_update_and_delete():
    """Test updating and deleting memories (Week 2)."""
    system = create_integrated_memory_system()

    # Store
    store_result = await system.store("Original content")
    memory_id = store_result.memory_id

    # Update
    update_result = await system.update(memory_id, "Updated content")
    assert update_result.success is True
    assert update_result.old_version_archived is True

    # Delete
    delete_result = await system.delete(update_result.new_version_id)
    assert delete_result.success is True
    assert delete_result.archived is True


# ============================================================================
# Test: Week 3 Integration (Consolidation)
# ============================================================================

@pytest.mark.asyncio
async def test_manual_consolidation():
    """Test manual consolidation (Week 3)."""
    system = create_integrated_memory_system()

    # Store episodic memories
    for i in range(5):
        await system.store(
            f"Episode {i}: Discussion about topic {i % 3}",
            metadata={"session_only": True},
            importance=0.6
        )

    # Consolidate
    result = await system.consolidate(
        strategy=ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours=24
    )

    assert result.input_episodes >= 5
    assert result.output_facts > 0


@pytest.mark.asyncio
async def test_background_consolidation_lifecycle():
    """Test background consolidation start/stop (Week 3)."""
    config = IntegratedMemoryConfig(enable_consolidation=True)
    system = IntegratedMemorySystem(config=config)

    # Start
    await system.start_consolidation()
    assert system.consolidator._running is True

    # Stop
    await system.stop_consolidation()
    assert system.consolidator._running is False


# ============================================================================
# Test: Week 4 Integration (Hybrid Retrieval)
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_retrieval():
    """Test hybrid retrieval (Week 4)."""
    system = create_integrated_memory_system()

    # Store diverse memories
    await system.store(
        "Thompson Sampling balances exploration and exploitation",
        importance=0.9,
        entities=["ThompsonSampling", "Exploration", "Exploitation"]
    )

    await system.store(
        "Bayesian methods use prior distributions",
        importance=0.8,
        entities=["Bayesian"]
    )

    await system.store(
        "Multi-armed bandits solve sequential decision problems",
        importance=0.7,
        entities=["MultiArmedBandit"]
    )

    # Hybrid retrieval
    result = await system.retrieve(
        query="Thompson Sampling exploration",
        limit=5
    )

    assert len(result.memories) > 0
    assert result.retrieval_time_ms >= 0
    assert "methods" in result.metadata


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_filters():
    """Test hybrid retrieval with scope/importance filters (Weeks 1+4)."""
    system = create_integrated_memory_system()

    # Store with different scopes and importance
    await system.store(
        "High importance USER fact",
        scope=MemoryScope.USER,
        importance=0.9
    )

    await system.store(
        "Low importance SESSION fact",
        metadata={"session_only": True},
        importance=0.3
    )

    # Retrieve only high-importance USER scope
    result = await system.retrieve(
        query="fact",
        scopes=[MemoryScope.USER],
        min_importance=0.7,
        limit=10
    )

    # Should only find high-importance USER memory
    assert len(result.memories) >= 1
    for memory in result.memories:
        assert memory.metadata.get("importance", 0.0) >= 0.7


# ============================================================================
# Test: Full Pipeline (Weeks 1+2+3+4)
# ============================================================================

@pytest.mark.asyncio
async def test_full_memory_pipeline():
    """Test complete memory pipeline (all 4 weeks)."""
    system = create_integrated_memory_system()

    # 1. Store episodic memories (Week 2)
    for i in range(3):
        await system.store(
            f"Thompson Sampling detail {i}",
            importance=0.8,
            entities=["ThompsonSampling"],
            metadata={"session_only": True}
        )

    # 2. Consolidate episodes → facts (Week 3)
    consolidation_result = await system.consolidate(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    assert consolidation_result.output_facts > 0

    # 3. Hybrid retrieval finds both episodes and facts (Week 4)
    retrieval_result = await system.retrieve(
        query="Thompson Sampling",
        limit=10
    )

    assert len(retrieval_result.memories) > 0

    # 4. Statistics show all systems working (Weeks 1-4)
    stats = system.get_statistics()

    assert "streams" in stats
    assert "agent_tools" in stats
    assert "consolidation" in stats
    assert "knowledge_graph" in stats


@pytest.mark.asyncio
async def test_production_system_creation():
    """Test production system factory (all weeks configured)."""
    # Note: Uses rule-based LLM (no API key needed for test)
    system = create_production_memory_system(
        llm_provider=None,  # None = rule-based
        llm_model=None
    )

    assert system.config.enable_consolidation is True
    assert system.config.enable_semantic_search is True
    assert system.config.enable_bm25_search is True
    assert system.config.enable_graph_search is True


# ============================================================================
# Test: Statistics and Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_comprehensive_statistics():
    """Test comprehensive statistics across all weeks."""
    system = create_integrated_memory_system()

    # Add some data
    await system.store("Test memory 1", importance=0.8)
    await system.store("Test memory 2", importance=0.7)

    # Get stats
    stats = system.get_statistics()

    # Week 1: Stream stats
    assert "memories_by_scope" in stats["streams"]

    # Week 2: Agent tools stats
    assert "archived_memories" in stats["agent_tools"]

    # Week 3: Consolidation stats
    assert "total_consolidations" in stats["consolidation"]
    assert "llm_usage" in stats["consolidation"]

    # Week 4: Knowledge graph stats
    assert "nodes" in stats["knowledge_graph"]
    assert "edges" in stats["knowledge_graph"]

    # Config
    assert "config" in stats
    assert "retrieval_methods" in stats["config"]


# ============================================================================
# Test: Error Handling and Resilience
# ============================================================================

@pytest.mark.asyncio
async def test_retrieval_with_no_memories():
    """Test hybrid retrieval with empty memory set."""
    system = create_integrated_memory_system()

    # Retrieve from empty system
    result = await system.retrieve("test query", limit=5)

    # Should return empty result, not crash
    assert result.memories == []
    assert result.scores == []


@pytest.mark.asyncio
async def test_consolidation_with_no_episodes():
    """Test consolidation with no recent episodes."""
    system = create_integrated_memory_system()

    # Consolidate with no episodes
    result = await system.consolidate()

    # Should return empty result, not crash
    assert result.input_episodes == 0
    assert result.output_facts == 0


@pytest.mark.asyncio
async def test_update_nonexistent_memory():
    """Test updating memory that doesn't exist."""
    system = create_integrated_memory_system()

    result = await system.update("nonexistent_id", "new content")

    assert result.success is False


@pytest.mark.asyncio
async def test_delete_nonexistent_memory():
    """Test deleting memory that doesn't exist."""
    system = create_integrated_memory_system()

    result = await system.delete("nonexistent_id")

    assert result.success is False


# ============================================================================
# Test: Configuration Variations
# ============================================================================

@pytest.mark.asyncio
async def test_minimal_configuration():
    """Test system with minimal configuration (no LLM, no consolidation)."""
    config = IntegratedMemoryConfig(
        llm_provider=None,
        enable_consolidation=False,
        enable_semantic_search=False,  # BM25 + graph only
        enable_bm25_search=True,
        enable_graph_search=True
    )

    system = IntegratedMemorySystem(config=config)

    # Should still work with limited features
    await system.store("Test", importance=0.5)
    result = await system.retrieve("Test", limit=5)

    assert isinstance(result.memories, list)


@pytest.mark.asyncio
async def test_bm25_only_configuration():
    """Test system with BM25 retrieval only."""
    config = IntegratedMemoryConfig(
        enable_semantic_search=False,
        enable_bm25_search=True,
        enable_graph_search=False
    )

    system = IntegratedMemorySystem(config=config)

    await system.store("keyword matching test", importance=0.7)

    result = await system.retrieve("keyword", limit=5)

    # Should use BM25 only
    assert "bm25" in result.metadata.get("methods", [])
