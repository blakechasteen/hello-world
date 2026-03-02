"""
Unit tests for Background Memory Consolidation (Week 2 implementation).

Tests:
- Background consolidation loop
- Fact extraction from episodes
- Entity extraction and KG integration
- Summarization strategies
- Deduplication
- Episode pruning

Based on LangMem research: "Background thread converts episodic → semantic memories"
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from hololoom.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationStrategy,
    ConsolidationResult,
    LLMConsolidator
)
from hololoom.memory.lifecycle_manager import (
    ContextStreamManager,
    MemoryScope
)
from hololoom.memory.graph import KG
from hololoom.protocols.types import MemoryShard


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
def consolidator(stream_manager, knowledge_graph):
    """Create MemoryConsolidator with manager and KG."""
    return MemoryConsolidator(
        stream_manager=stream_manager,
        knowledge_graph=knowledge_graph,
        llm_provider=None,  # Use rule-based fallback
        consolidation_interval_minutes=60,
        prune_consolidated_episodes=False
    )


@pytest.fixture
async def sample_episodes(stream_manager):
    """Create sample episodic memories in SESSION scope."""
    episodes = []

    # Use recent timestamps (minutes instead of hours to avoid expiration)
    for i in range(5):
        memory = MemoryShard(
            id=f"episode_{i}",
            text=f"Episode {i}: User discussed topic {i % 3}",
            entities=[f"Topic{i % 3}", "User"],
            metadata={
                "timestamp": (datetime.now() - timedelta(minutes=i * 10)).isoformat(),
                "type": "episode"
            }
        )
        await stream_manager.add_to_stream("session_state", memory)
        episodes.append(memory)

    return episodes


# ============================================================================
# Test: LLMConsolidator (Rule-based Fallback)
# ============================================================================

@pytest.mark.asyncio
async def test_llm_extract_facts_fallback():
    """Test rule-based fact extraction (fallback when no LLM)."""
    llm = LLMConsolidator(llm_provider=None)

    episodes = [
        MemoryShard(
            id="ep1",
            text="Python is a high-level programming language. It supports multiple paradigms.",
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="ep2",
            text="TypeScript is a superset of JavaScript. It adds static typing.",
            metadata={"timestamp": datetime.now().isoformat()}
        )
    ]

    facts = await llm.extract_facts(episodes)

    assert len(facts) > 0
    assert any("Python" in fact for fact in facts)


@pytest.mark.asyncio
async def test_llm_extract_entities_fallback():
    """Test rule-based entity extraction (fallback)."""
    llm = LLMConsolidator(llm_provider=None)

    episodes = [
        MemoryShard(
            id="ep1",
            text="Alice works with Bob on Project X",
            entities=["Alice", "Bob", "ProjectX"],
            metadata={"timestamp": datetime.now().isoformat()}
        )
    ]

    edges = await llm.extract_entities(episodes)

    assert len(edges) > 0
    # Should create edges between consecutive entities
    assert any(src == "Alice" and dst == "Bob" for src, dst, _ in edges)


@pytest.mark.asyncio
async def test_llm_deduplicate_fallback():
    """Test rule-based deduplication (exact match)."""
    llm = LLMConsolidator(llm_provider=None)

    episodes = [
        MemoryShard(id="ep1", text="Duplicate text", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="ep2", text="Duplicate text", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="ep3", text="Unique text", metadata={"timestamp": datetime.now().isoformat()})
    ]

    unique = await llm.deduplicate(episodes)

    assert len(unique) == 2  # Only 2 unique texts


# ============================================================================
# Test: Fact Extraction Consolidation
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_facts_basic(consolidator, sample_episodes):
    """Test basic fact extraction consolidation."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours=24
    )

    assert result.strategy == ConsolidationStrategy.FACT_EXTRACTION
    assert result.input_episodes == len(sample_episodes)
    assert result.output_facts > 0
    assert len(result.facts_stored) > 0


@pytest.mark.asyncio
async def test_consolidate_facts_stored_in_agent_scope(consolidator, sample_episodes):
    """Test that extracted facts are stored in AGENT scope."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    # Check that facts were routed to AGENT scope (project_facts stream)
    agent_memories = consolidator.stream_manager.get_all_memories(
        scopes=[MemoryScope.AGENT]
    )

    semantic_facts = [
        m for m in agent_memories
        if m.metadata.get("type") == "semantic_fact"
    ]

    assert len(semantic_facts) > 0


@pytest.mark.asyncio
async def test_consolidate_facts_importance(consolidator, sample_episodes):
    """Test that extracted facts have high importance score."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    # Get facts
    agent_memories = consolidator.stream_manager.get_all_memories(
        scopes=[MemoryScope.AGENT]
    )

    semantic_facts = [
        m for m in agent_memories
        if m.metadata.get("type") == "semantic_fact"
    ]

    # Facts should have importance >= 0.8
    for fact in semantic_facts:
        assert fact.metadata.get("importance", 0.0) >= 0.8


# ============================================================================
# Test: Entity Extraction Consolidation
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_entities_basic(consolidator, sample_episodes):
    """Test entity extraction consolidation."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.ENTITY_EXTRACTION
    )

    assert result.strategy == ConsolidationStrategy.ENTITY_EXTRACTION
    assert result.input_episodes == len(sample_episodes)
    assert result.output_facts > 0  # Edges created


@pytest.mark.asyncio
async def test_consolidate_entities_adds_to_kg(consolidator, knowledge_graph, sample_episodes):
    """Test that entity extraction adds edges to KG."""
    before_edge_count = len(list(knowledge_graph.G.edges()))

    await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.ENTITY_EXTRACTION
    )

    after_edge_count = len(list(knowledge_graph.G.edges()))

    assert after_edge_count > before_edge_count


# ============================================================================
# Test: Summarization Consolidation
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_summarization_basic(consolidator, sample_episodes):
    """Test summarization consolidation."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.SUMMARIZATION
    )

    assert result.strategy == ConsolidationStrategy.SUMMARIZATION
    assert result.input_episodes == len(sample_episodes)
    assert result.output_facts == 1  # One summary created
    assert len(result.facts_stored) == 1


@pytest.mark.asyncio
async def test_consolidate_summarization_content(consolidator, sample_episodes):
    """Test that summary contains metadata about episodes."""
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.SUMMARIZATION
    )

    # Get summary
    all_memories = consolidator.stream_manager.get_all_memories()
    summaries = [
        m for m in all_memories
        if m.metadata.get("type") == "summary"
    ]

    assert len(summaries) > 0
    summary = summaries[0]
    assert summary.metadata.get("episodes_summarized") == len(sample_episodes)


# ============================================================================
# Test: Deduplication Consolidation
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_deduplication_basic(stream_manager):
    """Test deduplication consolidation."""
    consolidator = MemoryConsolidator(
        stream_manager=stream_manager,
        llm_provider=None
    )

    # Add duplicate episodes
    for i in range(3):
        memory = MemoryShard(
            id=f"dup_{i}",
            text="Duplicate episode",  # Same text
            metadata={"timestamp": datetime.now().isoformat()}
        )
        await stream_manager.add_to_stream("session_state", memory)

    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.DEDUPLICATION
    )

    assert result.strategy == ConsolidationStrategy.DEDUPLICATION
    assert result.input_episodes == 3
    assert result.output_facts == 1  # Only 1 unique
    assert result.metadata.get("duplicates_removed") == 2


# ============================================================================
# Test: Episode Pruning
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_with_pruning(stream_manager, sample_episodes):
    """Test that consolidation can prune consolidated episodes."""
    consolidator = MemoryConsolidator(
        stream_manager=stream_manager,
        prune_consolidated_episodes=True  # Enable pruning
    )

    before_count = len(stream_manager.get_stream("session_state").memories)

    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    after_count = len(stream_manager.get_stream("session_state").memories)

    assert result.episodes_pruned > 0
    assert after_count < before_count


@pytest.mark.asyncio
async def test_consolidate_without_pruning(stream_manager, sample_episodes):
    """Test that consolidation preserves episodes when pruning disabled."""
    consolidator = MemoryConsolidator(
        stream_manager=stream_manager,
        prune_consolidated_episodes=False  # Disable pruning
    )

    before_count = len(stream_manager.get_stream("session_state").memories)

    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    after_count = len(stream_manager.get_stream("session_state").memories)

    assert result.episodes_pruned == 0
    assert after_count == before_count


# ============================================================================
# Test: Lookback Time Window
# ============================================================================

@pytest.mark.asyncio
async def test_consolidate_lookback_window(stream_manager):
    """Test that consolidation only processes recent episodes."""
    consolidator = MemoryConsolidator(stream_manager=stream_manager)

    # Add old episode (2 hours ago - outside 1 hour lookback)
    old_memory = MemoryShard(
        id="old_ep",
        text="Old episode from 2 hours ago",
        metadata={"timestamp": (datetime.now() - timedelta(minutes=120)).isoformat()}
    )
    await stream_manager.add_to_stream("session_state", old_memory)

    # Add recent episode (30 minutes ago - within 1 hour lookback)
    recent_memory = MemoryShard(
        id="recent_ep",
        text="Recent episode from 30 minutes ago",
        metadata={"timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()}
    )
    await stream_manager.add_to_stream("session_state", recent_memory)

    # Consolidate with 1-hour lookback
    result = await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION,
        lookback_hours=1
    )

    # Should only process recent episode
    assert result.input_episodes == 1


@pytest.mark.asyncio
async def test_consolidate_no_recent_episodes(stream_manager):
    """Test consolidation when no recent episodes exist."""
    consolidator = MemoryConsolidator(stream_manager=stream_manager)

    # No episodes added

    result = await consolidator.consolidate_recent_episodes()

    assert result.input_episodes == 0
    assert result.output_facts == 0
    assert len(result.facts_stored) == 0


# ============================================================================
# Test: Background Loop
# ============================================================================

@pytest.mark.asyncio
async def test_background_consolidation_start_stop(consolidator):
    """Test starting and stopping background consolidation loop."""
    # Start
    await consolidator.start_background_consolidation()
    assert consolidator._running is True
    assert consolidator._consolidation_task is not None

    # Stop
    await consolidator.stop_background_consolidation()
    assert consolidator._running is False


@pytest.mark.asyncio
async def test_background_consolidation_runs_periodically(stream_manager):
    """Test that background consolidation runs on schedule."""
    # Use short interval for testing (1 second)
    consolidator = MemoryConsolidator(
        stream_manager=stream_manager,
        consolidation_interval_minutes=0.017  # ~1 second
    )

    # Add episode
    memory = MemoryShard(
        id="bg_test",
        text="Background test episode",
        metadata={"timestamp": datetime.now().isoformat()}
    )
    await stream_manager.add_to_stream("session_state", memory)

    # Start background task
    await consolidator.start_background_consolidation()

    # Wait for one consolidation cycle
    await asyncio.sleep(1.5)

    # Stop
    await consolidator.stop_background_consolidation()

    # Check that consolidation ran
    assert consolidator.total_consolidations > 0


# ============================================================================
# Test: Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_consolidation_statistics(consolidator, sample_episodes):
    """Test that consolidation tracks statistics."""
    # Run consolidation
    await consolidator.consolidate_recent_episodes(
        strategy=ConsolidationStrategy.FACT_EXTRACTION
    )

    stats = consolidator.get_statistics()

    assert stats["total_consolidations"] == 1
    assert stats["total_facts_extracted"] > 0
    assert "consolidation_interval_minutes" in stats
    assert "llm_provider" in stats


@pytest.mark.asyncio
async def test_consolidation_statistics_cumulative(consolidator, sample_episodes):
    """Test that statistics accumulate across multiple consolidations."""
    # Run twice
    await consolidator.consolidate_recent_episodes()
    await consolidator.consolidate_recent_episodes()

    stats = consolidator.get_statistics()

    assert stats["total_consolidations"] == 2
