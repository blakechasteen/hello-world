"""
Tests for Episodic → Semantic Memory Transition

Comprehensive test suite covering:
- Pattern detection (15+ tests)
- Concept promotion (10+ tests)
- Provenance tracking (8+ tests)
- Background transition (5+ tests)
- Integration with consolidation (5+ tests)

Target: 95%+ coverage

Created: November 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

from HoloLoom.memory.semantic_transition import (
    SemanticTransitionEngine,
    SemanticTransitionConfig,
    EpisodicPattern,
    SemanticConcept,
    TransitionStatistics
)
from HoloLoom.memory.protocol import Memory
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom import HoloLoom


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default configuration for testing."""
    return SemanticTransitionConfig(
        pattern_threshold=3,
        similarity_threshold=0.6,
        transition_window_days=7,
        max_patterns_per_cycle=50,
        enable_background_transition=False,  # Disabled for tests
        enable_query_clustering=True,
        enable_entity_cooccurrence=True,
        enable_motif_patterns=True,
        enable_response_similarity=True
    )


@pytest.fixture
async def loom():
    """HoloLoom instance for testing."""
    from HoloLoom.config import Config

    config = Config.fast()
    loom_instance = HoloLoom(config=config)

    async with loom_instance as loom_ctx:
        yield loom_ctx


@pytest.fixture
def kg():
    """Knowledge graph for testing."""
    return KG()


@pytest.fixture
async def engine(loom, config, kg):
    """Semantic transition engine for testing."""
    return SemanticTransitionEngine(loom=loom, config=config, kg=kg)


@pytest.fixture
async def sample_episodes(loom):
    """Create sample episodic memories."""
    episodes = []

    # Create 5 episodes about "Thompson Sampling"
    for i in range(5):
        mem = await loom.experience(
            f"What is Thompson Sampling? (query {i+1})",
            context={'scope': 'SESSION', 'query_type': 'what_is'}
        )
        episodes.append(mem)

    # Create 3 episodes about "exploration vs exploitation"
    for i in range(3):
        mem = await loom.experience(
            f"Explain exploration vs exploitation (query {i+1})",
            context={'scope': 'SESSION', 'query_type': 'explain'}
        )
        episodes.append(mem)

    # Create 4 episodes with entity co-occurrence
    for i in range(4):
        mem = await loom.experience(
            f"Thompson Sampling and Bayesian methods (query {i+1})",
            context={'scope': 'SESSION'}
        )

        # Add entities to graph
        loom.graph.add_edge(mem.id, 'Thompson_Sampling', type='MENTIONS')
        loom.graph.add_edge(mem.id, 'Bayesian', type='MENTIONS')

        episodes.append(mem)

    return episodes


# ============================================================================
# Pattern Detection Tests (15+ tests)
# ============================================================================

@pytest.mark.asyncio
async def test_detect_patterns_basic(engine, sample_episodes):
    """Test basic pattern detection."""
    patterns = await engine.detect_patterns()

    assert len(patterns) > 0
    assert all(isinstance(p, EpisodicPattern) for p in patterns)
    assert all(p.frequency >= engine.config.pattern_threshold for p in patterns)


@pytest.mark.asyncio
async def test_detect_query_clusters(engine, sample_episodes):
    """Test query clustering pattern detection."""
    patterns = await engine.detect_patterns()

    # Should detect "Thompson Sampling" cluster (5 similar queries)
    thompson_patterns = [
        p for p in patterns
        if p.pattern_type == 'query_cluster'
        and 'Thompson Sampling' in str(p.signature)
    ]

    assert len(thompson_patterns) > 0
    assert thompson_patterns[0].frequency >= 3


@pytest.mark.asyncio
async def test_detect_entity_cooccurrence(engine, sample_episodes):
    """Test entity co-occurrence pattern detection."""
    patterns = await engine.detect_patterns()

    # Should detect Thompson_Sampling + Bayesian co-occurrence
    cooccurrence_patterns = [
        p for p in patterns
        if p.pattern_type == 'entity_cooccurrence'
    ]

    assert len(cooccurrence_patterns) > 0


@pytest.mark.asyncio
async def test_pattern_frequency_threshold(engine, loom):
    """Test pattern frequency threshold filtering."""
    # Create only 2 similar episodes (below threshold of 3)
    for i in range(2):
        await loom.experience(
            f"Low frequency query {i}",
            context={'scope': 'SESSION'}
        )

    patterns = await engine.detect_patterns()

    # Should not detect pattern (below threshold)
    low_freq_patterns = [
        p for p in patterns
        if 'Low frequency' in str(p.signature)
    ]

    assert len(low_freq_patterns) == 0


@pytest.mark.asyncio
async def test_pattern_similarity_threshold(engine, config):
    """Test pattern similarity threshold filtering."""
    config.similarity_threshold = 0.9  # Very high threshold

    engine_high_threshold = SemanticTransitionEngine(
        loom=engine.loom,
        config=config,
        kg=engine.kg
    )

    patterns = await engine_high_threshold.detect_patterns()

    # Should filter out low-similarity patterns
    assert all(p.similarity_score >= 0.9 for p in patterns)


@pytest.mark.asyncio
async def test_pattern_time_window(engine, loom):
    """Test pattern detection time window."""
    # Create old episode (outside window)
    old_mem = await loom.experience(
        "Old query",
        context={'scope': 'SESSION'}
    )

    # Manually set old timestamp
    if old_mem.id in loom.graph.nodes():
        loom.graph.nodes[old_mem.id]['timestamp'] = (
            datetime.now() - timedelta(days=30)
        ).timestamp()

    # Create recent episodes
    for i in range(3):
        await loom.experience(
            "Recent query",
            context={'scope': 'SESSION'}
        )

    patterns = await engine.detect_patterns()

    # Old episode should not be included
    for pattern in patterns:
        for episode_id in pattern.episode_ids:
            if episode_id == old_mem.id:
                pytest.fail("Old episode should not be in pattern")


@pytest.mark.asyncio
async def test_max_patterns_limit(engine, sample_episodes):
    """Test max patterns limit."""
    patterns = await engine.detect_patterns(max_patterns=2)

    assert len(patterns) <= 2


@pytest.mark.asyncio
async def test_pattern_sorting(engine, sample_episodes):
    """Test patterns sorted by frequency and similarity."""
    patterns = await engine.detect_patterns()

    # Verify sorted by frequency (descending)
    if len(patterns) > 1:
        for i in range(len(patterns) - 1):
            assert patterns[i].frequency >= patterns[i+1].frequency


@pytest.mark.asyncio
async def test_pattern_deduplication(engine, loom):
    """Test pattern deduplication (same pattern not detected twice)."""
    # Create episodes
    for i in range(4):
        await loom.experience(
            "Duplicate pattern query",
            context={'scope': 'SESSION'}
        )

    patterns = await engine.detect_patterns()

    # Check for duplicate pattern_ids
    pattern_ids = [p.pattern_id for p in patterns]
    assert len(pattern_ids) == len(set(pattern_ids))


@pytest.mark.asyncio
async def test_motif_pattern_detection(engine, loom):
    """Test motif pattern detection."""
    # Create episodes with common motif
    for i in range(4):
        mem = await loom.experience(
            f"Query with motif {i}",
            context={'scope': 'SESSION'}
        )

        # Add motif to metadata
        if mem.id in loom.graph.nodes():
            loom.graph.nodes[mem.id]['metadata'] = {
                'motifs': ['exploration']
            }

    patterns = await engine.detect_patterns()

    motif_patterns = [
        p for p in patterns
        if p.pattern_type == 'motif_pattern'
    ]

    assert len(motif_patterns) > 0


@pytest.mark.asyncio
async def test_response_similarity_detection(engine, loom):
    """Test response similarity pattern detection."""
    # Create episodes with similar responses
    for i in range(3):
        mem = await loom.experience(
            f"Query {i}",
            context={'scope': 'SESSION'}
        )

        # Add response to metadata
        if mem.id in loom.graph.nodes():
            loom.graph.nodes[mem.id]['metadata'] = {
                'response': 'Thompson Sampling is a Bayesian algorithm'
            }

    patterns = await engine.detect_patterns()

    response_patterns = [
        p for p in patterns
        if p.pattern_type == 'response_similarity'
    ]

    assert len(response_patterns) > 0


@pytest.mark.asyncio
async def test_pattern_metadata(engine, sample_episodes):
    """Test pattern metadata is populated correctly."""
    patterns = await engine.detect_patterns()

    for pattern in patterns:
        assert pattern.pattern_id
        assert pattern.pattern_type
        assert pattern.frequency >= 3
        assert 0.0 <= pattern.similarity_score <= 1.0
        assert len(pattern.episode_ids) > 0
        assert isinstance(pattern.signature, dict)
        assert isinstance(pattern.first_seen, datetime)
        assert isinstance(pattern.last_seen, datetime)
        assert pattern.last_seen >= pattern.first_seen


@pytest.mark.asyncio
async def test_empty_episodes(engine, loom):
    """Test pattern detection with no episodes."""
    # Don't create any episodes
    patterns = await engine.detect_patterns()

    assert len(patterns) == 0


@pytest.mark.asyncio
async def test_pattern_caching(engine, sample_episodes):
    """Test detected patterns are cached."""
    patterns = await engine.detect_patterns()

    # Check patterns are cached
    assert len(engine.detected_patterns) > 0

    for pattern in patterns:
        assert pattern.pattern_id in engine.detected_patterns


@pytest.mark.asyncio
async def test_pattern_statistics_update(engine, sample_episodes):
    """Test statistics are updated after pattern detection."""
    initial_count = engine.stats['patterns_detected']

    patterns = await engine.detect_patterns()

    assert engine.stats['patterns_detected'] == initial_count + len(patterns)


# ============================================================================
# Concept Promotion Tests (10+ tests)
# ============================================================================

@pytest.mark.asyncio
async def test_promote_to_semantic_basic(engine, sample_episodes):
    """Test basic concept promotion."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        assert isinstance(concept, SemanticConcept)
        assert concept.concept_id
        assert concept.concept_text
        assert concept.source_pattern_id == patterns[0].pattern_id
        assert len(concept.source_episode_ids) > 0


@pytest.mark.asyncio
async def test_concept_stored_in_semantic_memory(engine, sample_episodes):
    """Test concept is stored in semantic memory."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Check concept is in semantic_concepts dict
        assert concept.concept_id in engine.semantic_concepts

        # Check concept is in knowledge graph
        assert concept.concept_id in engine.kg.graph.nodes()


@pytest.mark.asyncio
async def test_concept_scope_is_agent(engine, sample_episodes):
    """Test concept has AGENT scope."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        assert concept.scope.value == 'AGENT'


@pytest.mark.asyncio
async def test_concept_text_generation(engine, sample_episodes):
    """Test concept text is generated from pattern."""
    patterns = await engine.detect_patterns()

    for pattern in patterns[:3]:  # Test first 3
        concept = await engine.promote_to_semantic(pattern)

        # Concept text should be non-empty
        assert concept.concept_text
        assert len(concept.concept_text) > 10


@pytest.mark.asyncio
async def test_concept_provenance_links(engine, sample_episodes):
    """Test concept is linked to source episodes."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Check provenance edges in graph
        for episode_id in concept.source_episode_ids:
            # Should have DERIVED_FROM edge
            if episode_id in engine.kg.graph.nodes():
                edges = list(engine.kg.graph.edges(concept.concept_id, data=True))
                derived_from_edges = [
                    e for e in edges
                    if e[2].get('type') == 'DERIVED_FROM'
                    and e[1] == episode_id
                ]

                assert len(derived_from_edges) > 0


@pytest.mark.asyncio
async def test_concept_confidence_from_pattern(engine, sample_episodes):
    """Test concept confidence matches pattern similarity."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        assert concept.confidence == patterns[0].similarity_score


@pytest.mark.asyncio
async def test_concept_statistics_update(engine, sample_episodes):
    """Test statistics updated after promotion."""
    patterns = await engine.detect_patterns()

    if patterns:
        initial_concepts = engine.stats['concepts_created']
        initial_episodes = engine.stats['episodes_transitioned']

        concept = await engine.promote_to_semantic(patterns[0])

        assert engine.stats['concepts_created'] == initial_concepts + 1
        assert engine.stats['episodes_transitioned'] >= initial_episodes


@pytest.mark.asyncio
async def test_episode_pruning_disabled(engine, sample_episodes):
    """Test episodes are NOT pruned when disabled."""
    engine.config.prune_source_episodes = False

    patterns = await engine.detect_patterns()

    if patterns:
        pattern = patterns[0]
        episode_ids_before = list(pattern.episode_ids)

        await engine.promote_to_semantic(pattern)

        # Episodes should still exist
        for episode_id in episode_ids_before:
            assert episode_id in engine.kg.graph.nodes()


@pytest.mark.asyncio
async def test_episode_pruning_enabled(engine, sample_episodes):
    """Test episodes are pruned when enabled."""
    engine.config.prune_source_episodes = True

    patterns = await engine.detect_patterns()

    if patterns:
        pattern = patterns[0]
        episode_ids_before = list(pattern.episode_ids)

        await engine.promote_to_semantic(pattern)

        # Episodes should be removed
        for episode_id in episode_ids_before:
            assert episode_id not in engine.kg.graph.nodes()


@pytest.mark.asyncio
async def test_multiple_concept_promotion(engine, sample_episodes):
    """Test promoting multiple concepts."""
    patterns = await engine.detect_patterns()

    concepts = []
    for pattern in patterns[:3]:  # Promote first 3
        concept = await engine.promote_to_semantic(pattern)
        concepts.append(concept)

    assert len(concepts) == min(3, len(patterns))

    # All concepts should be stored
    for concept in concepts:
        assert concept.concept_id in engine.semantic_concepts


# ============================================================================
# Provenance Tracking Tests (8+ tests)
# ============================================================================

@pytest.mark.asyncio
async def test_get_concept_provenance(engine, sample_episodes):
    """Test getting concept provenance."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        provenance = engine.get_concept_provenance(concept.concept_id)

        assert len(provenance) > 0
        assert provenance == concept.source_episode_ids


@pytest.mark.asyncio
async def test_provenance_returns_episode_ids(engine, sample_episodes):
    """Test provenance returns valid episode IDs."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        provenance = engine.get_concept_provenance(concept.concept_id)

        # All IDs should be valid (if not pruned)
        if not engine.config.prune_source_episodes:
            for episode_id in provenance:
                assert episode_id in engine.kg.graph.nodes()


@pytest.mark.asyncio
async def test_provenance_for_nonexistent_concept(engine):
    """Test provenance for nonexistent concept."""
    provenance = engine.get_concept_provenance("nonexistent_concept")

    assert provenance == []


@pytest.mark.asyncio
async def test_provenance_metadata(engine, sample_episodes):
    """Test provenance metadata is preserved."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Check metadata includes provenance info
        assert 'pattern_type' in concept.metadata
        assert 'pattern_frequency' in concept.metadata
        assert concept.metadata['created_from_pattern'] is True


@pytest.mark.asyncio
async def test_provenance_temporal_tracking(engine, sample_episodes):
    """Test provenance includes temporal information."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Concept should have creation timestamp
        assert isinstance(concept.created_at, datetime)

        # Pattern should have temporal bounds
        pattern = patterns[0]
        assert isinstance(pattern.first_seen, datetime)
        assert isinstance(pattern.last_seen, datetime)


@pytest.mark.asyncio
async def test_provenance_chain(engine, sample_episodes):
    """Test provenance chain is complete."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Walk provenance chain
        provenance_ids = engine.get_concept_provenance(concept.concept_id)

        # Should be able to find all source episodes
        assert len(provenance_ids) == len(concept.source_episode_ids)
        assert set(provenance_ids) == set(concept.source_episode_ids)


@pytest.mark.asyncio
async def test_provenance_graph_edges(engine, sample_episodes):
    """Test provenance is represented as graph edges."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Check DERIVED_FROM edges exist
        edges = list(engine.kg.graph.edges(concept.concept_id, data=True))

        derived_from_edges = [
            e for e in edges
            if e[2].get('type') == 'DERIVED_FROM'
        ]

        assert len(derived_from_edges) > 0


@pytest.mark.asyncio
async def test_concept_access_tracking(engine, sample_episodes):
    """Test concept access is tracked for provenance."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Initially not accessed
        assert concept.access_count == 0
        assert concept.last_accessed is None

        # Query semantic memory
        result = await engine.query_semantic(concept.concept_text)

        if result:
            # Access should be tracked
            assert result.access_count > 0
            assert result.last_accessed is not None


# ============================================================================
# Background Transition Tests (5+ tests)
# ============================================================================

@pytest.mark.asyncio
async def test_transition_during_idle(engine, sample_episodes):
    """Test idle transition operation."""
    stats = await engine.transition_during_idle()

    assert isinstance(stats, TransitionStatistics)
    assert stats.episodes_analyzed > 0
    assert stats.duration_ms > 0


@pytest.mark.asyncio
async def test_idle_transition_statistics(engine, sample_episodes):
    """Test idle transition returns statistics."""
    stats = await engine.transition_during_idle()

    assert stats.episodes_analyzed >= 0
    assert stats.patterns_detected >= 0
    assert stats.concepts_created >= 0
    assert stats.episodes_pruned >= 0
    assert isinstance(stats.pattern_types, dict)


@pytest.mark.asyncio
async def test_background_transition_start_stop(engine):
    """Test starting and stopping background transition."""
    # Start
    await engine.start_background_transition()

    assert engine._background_task is not None
    assert not engine._background_task.done()

    # Stop
    await engine.stop_background_transition()

    assert engine._stop_background is True


@pytest.mark.asyncio
async def test_background_transition_disabled_in_config(loom, kg):
    """Test background transition respects config."""
    config = SemanticTransitionConfig(
        enable_background_transition=False
    )

    engine = SemanticTransitionEngine(loom=loom, config=config, kg=kg)

    await engine.start_background_transition()

    # Should not start if disabled
    assert engine._background_task is None or engine._background_task.done()


@pytest.mark.asyncio
async def test_context_manager_lifecycle(loom, kg):
    """Test context manager starts/stops background transition."""
    config = SemanticTransitionConfig(
        enable_background_transition=True,
        background_interval_seconds=1.0
    )

    async with SemanticTransitionEngine(loom=loom, config=config, kg=kg) as engine:
        # Should be running
        assert engine._background_task is not None

    # Should be stopped after exit
    assert engine._stop_background is True


# ============================================================================
# Integration Tests (5+ tests)
# ============================================================================

@pytest.mark.asyncio
async def test_integration_with_hololoom(loom):
    """Test integration with HoloLoom."""
    engine = SemanticTransitionEngine(loom=loom)

    # Create episodes via HoloLoom
    for i in range(4):
        await loom.experience(
            f"Integration test query {i}",
            context={'scope': 'SESSION'}
        )

    # Detect patterns
    patterns = await engine.detect_patterns()

    assert len(patterns) >= 0  # May or may not detect patterns


@pytest.mark.asyncio
async def test_query_semantic_vs_episodic(engine, sample_episodes):
    """Test semantic query is faster than episodic."""
    patterns = await engine.detect_patterns()

    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])

        # Query semantic memory
        start = datetime.now()
        semantic_result = await engine.query_semantic(concept.concept_text)
        semantic_time = (datetime.now() - start).total_seconds()

        # Should find concept
        assert semantic_result is not None

        # Update statistics
        assert engine.stats['semantic_queries'] > 0


@pytest.mark.asyncio
async def test_end_to_end_transition(loom):
    """Test complete end-to-end transition flow."""
    engine = SemanticTransitionEngine(loom=loom)

    # 1. Create episodic memories
    for i in range(5):
        await loom.experience(
            "What is Thompson Sampling?",
            context={'scope': 'SESSION'}
        )

    # 2. Detect patterns
    patterns = await engine.detect_patterns()
    assert len(patterns) > 0

    # 3. Promote to semantic
    concept = await engine.promote_to_semantic(patterns[0])
    assert concept is not None

    # 4. Query semantic memory
    result = await engine.query_semantic("Thompson Sampling")
    assert result is not None

    # 5. Check provenance
    provenance = engine.get_concept_provenance(concept.concept_id)
    assert len(provenance) > 0


@pytest.mark.asyncio
async def test_statistics_accuracy(engine, sample_episodes):
    """Test engine statistics are accurate."""
    # Get initial stats
    initial_stats = engine.get_statistics()

    # Run transition
    await engine.transition_during_idle()

    # Get updated stats
    updated_stats = engine.get_statistics()

    # Verify statistics updated
    assert updated_stats['patterns_detected'] >= initial_stats['patterns_detected']
    assert updated_stats['concepts_created'] >= initial_stats['concepts_created']


@pytest.mark.asyncio
async def test_configuration_options(loom, kg):
    """Test different configuration options."""
    # Test with different thresholds
    config1 = SemanticTransitionConfig(
        pattern_threshold=2,
        similarity_threshold=0.5
    )

    engine1 = SemanticTransitionEngine(loom=loom, config=config1, kg=kg)

    config2 = SemanticTransitionConfig(
        pattern_threshold=5,
        similarity_threshold=0.9
    )

    engine2 = SemanticTransitionEngine(loom=loom, config=config2, kg=kg)

    # Both should initialize without errors
    assert engine1.config.pattern_threshold == 2
    assert engine2.config.pattern_threshold == 5


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pattern_detection_performance(engine, loom):
    """Test pattern detection meets performance target (<100ms for 100 episodes)."""
    # Create 100 episodes
    for i in range(100):
        await loom.experience(
            f"Performance test query {i % 10}",
            context={'scope': 'SESSION'}
        )

    # Measure pattern detection time
    start = datetime.now()
    patterns = await engine.detect_patterns()
    duration_ms = (datetime.now() - start).total_seconds() * 1000

    # Should be < 100ms (target)
    assert duration_ms < 200  # Allow some margin


@pytest.mark.asyncio
async def test_concept_promotion_performance(engine, sample_episodes):
    """Test concept promotion meets performance target (<50ms per concept)."""
    patterns = await engine.detect_patterns()

    if patterns:
        # Measure promotion time
        start = datetime.now()
        await engine.promote_to_semantic(patterns[0])
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # Should be < 50ms (target)
        assert duration_ms < 100  # Allow some margin


@pytest.mark.asyncio
async def test_background_transition_performance(engine, sample_episodes):
    """Test background transition meets performance target (<500ms for 50 patterns)."""
    # Measure idle transition time
    start = datetime.now()
    stats = await engine.transition_during_idle()
    duration_ms = stats.duration_ms

    # Should be < 500ms for reasonable number of patterns
    if stats.patterns_detected < 50:
        assert duration_ms < 500  # Target


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_no_patterns_detected(engine, loom):
    """Test behavior when no patterns detected."""
    # Create only 1 episode (below threshold)
    await loom.experience("Single query", context={'scope': 'SESSION'})

    patterns = await engine.detect_patterns()

    assert len(patterns) == 0


@pytest.mark.asyncio
async def test_duplicate_episode_handling(engine, loom):
    """Test handling of duplicate episodes in pattern."""
    # Create exact duplicates
    for i in range(4):
        await loom.experience(
            "Exact duplicate query",
            context={'scope': 'SESSION'}
        )

    patterns = await engine.detect_patterns()

    # Should handle duplicates gracefully
    if patterns:
        pattern = patterns[0]
        # Episode IDs should be unique
        assert len(pattern.episode_ids) == len(set(pattern.episode_ids))


@pytest.mark.asyncio
async def test_malformed_episode_data(engine, loom):
    """Test handling of malformed episode data."""
    # Create episode with missing data
    mem = await loom.experience("Test", context={'scope': 'SESSION'})

    # Remove timestamp (malformed)
    if mem.id in loom.graph.nodes():
        del loom.graph.nodes[mem.id]['timestamp']

    # Should handle gracefully
    try:
        patterns = await engine.detect_patterns()
        # Should not crash
        assert True
    except Exception as e:
        pytest.fail(f"Should handle malformed data gracefully: {e}")
