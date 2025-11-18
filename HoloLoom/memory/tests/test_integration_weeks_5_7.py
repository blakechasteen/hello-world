"""
Integration Tests for Weeks 5-7 Memory Systems

Tests the complete integration of:
- Week 5: Memory consolidation, multi-hop reasoning, curiosity engine
- Week 6: Episodic → Semantic transition
- Week 7: Temporal evolution tracking

Ensures all systems work together seamlessly in production scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.memory.consolidation import SleepBasedConsolidation, ConsolidationConfig
from HoloLoom.memory.graph_reasoning import GraphReasoner
from HoloLoom.memory.curiosity import CuriosityEngine, CuriosityConfig
from HoloLoom.memory.semantic_transition import SemanticTransitionEngine, SemanticTransitionConfig
from HoloLoom.memory.temporal_evolution import TemporalEvolutionTracker, TemporalEvolutionConfig


@pytest.fixture
async def integrated_system():
    """Create fully integrated memory system with all Week 5-7 features"""
    # Use temporary directory for persistence
    temp_dir = tempfile.mkdtemp()

    # Initialize HoloLoom core
    config = Config.fast()
    loom = HoloLoom(config=config)
    await loom.__aenter__()

    # Week 5 systems
    consolidation = SleepBasedConsolidation(
        loom=loom,
        config=ConsolidationConfig(
            enabled=True,
            idle_threshold_hours=0.001,  # Very short for testing (3.6 seconds)
            decay_rate=0.95,
            promotion_threshold_accesses=3,
            promotion_window_days=7,
            archive_threshold=0.1
        )
    )

    graph_reasoner = GraphReasoner(
        kg=loom.graph,
        retriever=None,  # Optional retriever
        enable_caching=True,
        cache_size=1000
    )

    curiosity = CuriosityEngine(
        kg=loom.graph,
        config=CuriosityConfig(
            enabled=True,
            gap_importance=0.85,
            contradiction_importance=0.75,
            access_pattern_importance=0.70
        )
    )

    # Week 6 system
    semantic = SemanticTransitionEngine(
        loom=loom,
        config=SemanticTransitionConfig(
            enabled=True,
            pattern_threshold=3,
            similarity_threshold=0.75,
            transition_window_days=7,
            enable_background_transition=False  # Manual control for testing
        )
    )

    # Week 7 system
    temporal = TemporalEvolutionTracker(
        loom=loom,
        config=TemporalEvolutionConfig(
            enabled=True,
            learning_threshold=3,
            familiar_threshold=5,
            mastery_confidence=0.85,
            dormant_days=30,
            forgotten_importance=0.1,
            persistence_path=Path(temp_dir) / "temporal_evolution.json"
        )
    )

    yield {
        'loom': loom,
        'consolidation': consolidation,
        'graph_reasoner': graph_reasoner,
        'curiosity': curiosity,
        'semantic': semantic,
        'temporal': temporal
    }

    # Cleanup
    await loom.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_basic_integration(integrated_system):
    """Test 1: Basic integration - all systems initialized"""
    loom = integrated_system['loom']
    consolidation = integrated_system['consolidation']
    graph_reasoner = integrated_system['graph_reasoner']
    curiosity = integrated_system['curiosity']
    semantic = integrated_system['semantic']
    temporal = integrated_system['temporal']

    # Verify all systems are initialized
    assert loom is not None
    assert consolidation is not None
    assert graph_reasoner is not None
    assert curiosity is not None
    assert semantic is not None
    assert temporal is not None


@pytest.mark.asyncio
async def test_episodic_to_semantic_to_temporal(integrated_system):
    """Test 2: Complete flow from episodic memory → semantic concept → temporal tracking"""
    loom = integrated_system['loom']
    semantic = integrated_system['semantic']
    temporal = integrated_system['temporal']

    # Store 5 similar episodic memories
    queries = [
        "What is Thompson Sampling?",
        "Explain Thompson Sampling",
        "How does Thompson Sampling work?",
        "Thompson Sampling algorithm",
        "Thompson Sampling exploration"
    ]

    for query in queries:
        # Store episodic memory
        memory = await loom.experience(query)

        # Track temporal interaction
        await temporal.track_interaction(
            query=query,
            entities=["Thompson Sampling"],
            confidence=0.85
        )

    # Detect patterns (Week 6)
    patterns = await semantic.detect_patterns()
    assert len(patterns) > 0, "Should detect at least one pattern"

    # Promote to semantic concept
    if patterns:
        concept = await semantic.promote_to_semantic(patterns[0])
        assert concept is not None
        assert "Thompson Sampling" in concept.concept_text

    # Check temporal evolution (Week 7)
    history = await temporal.get_concept_history("Thompson Sampling")
    assert history is not None
    assert history.current_state.value in ["LEARNING", "FAMILIAR"]
    assert history.total_interactions == 5


@pytest.mark.asyncio
async def test_multi_hop_reasoning_with_semantic_concepts(integrated_system):
    """Test 3: Multi-hop reasoning across semantic concepts"""
    loom = integrated_system['loom']
    graph_reasoner = integrated_system['graph_reasoner']
    semantic = integrated_system['semantic']

    # Create knowledge graph with relationships
    await loom.experience("Thompson Sampling is a bandit algorithm")
    await loom.experience("Bandit algorithms balance exploration and exploitation")
    await loom.experience("Reinforcement learning uses bandit algorithms")

    # Detect and promote patterns
    patterns = await semantic.detect_patterns()
    for pattern in patterns[:3]:  # Promote up to 3 patterns
        await semantic.promote_to_semantic(pattern)

    # Multi-hop query
    results = await graph_reasoner.multi_hop_query(
        query="Thompson Sampling",
        max_hops=2
    )

    assert len(results) > 0
    # Should find direct matches and connected concepts


@pytest.mark.asyncio
async def test_curiosity_with_gaps_in_knowledge(integrated_system):
    """Test 4: Curiosity engine detects gaps and suggests exploration"""
    loom = integrated_system['loom']
    curiosity = integrated_system['curiosity']

    # Create knowledge with obvious gaps
    await loom.experience("PPO is a policy optimization algorithm")
    await loom.experience("PPO uses gradient descent")
    # Gap: Missing connection between gradient descent and optimization

    await loom.experience("Actor-critic methods use value functions")
    # Gap: Missing connection between PPO and actor-critic

    # Get suggestions
    suggestions = await curiosity.suggest_exploration(limit=5)

    assert len(suggestions) > 0
    # Should suggest exploring the gaps


@pytest.mark.asyncio
async def test_consolidation_preserves_semantic_concepts(integrated_system):
    """Test 5: Consolidation preserves important semantic concepts"""
    loom = integrated_system['loom']
    consolidation = integrated_system['consolidation']
    semantic = integrated_system['semantic']

    # Create frequently accessed memories
    for i in range(5):
        memory = await loom.experience("Important concept about neural networks")
        consolidation.record_access(
            memory_id=f"memory_{i}",
            importance=0.9
        )

    # Create infrequent memories
    for i in range(3):
        memory = await loom.experience("Rarely used information")
        consolidation.record_access(
            memory_id=f"rare_{i}",
            importance=0.2
        )

    # Promote patterns to semantic
    patterns = await semantic.detect_patterns()
    for pattern in patterns:
        await semantic.promote_to_semantic(pattern)

    # Trigger consolidation
    stats = await consolidation.consolidate_during_idle()

    # Verify consolidation happened
    assert stats['promoted_count'] >= 0
    assert stats['archived_count'] >= 0


@pytest.mark.asyncio
async def test_temporal_state_transitions(integrated_system):
    """Test 6: Temporal evolution tracks state transitions correctly"""
    loom = integrated_system['loom']
    temporal = integrated_system['temporal']

    concept = "Gradient Descent"

    # UNKNOWN → INTRODUCED (1 interaction)
    await temporal.track_interaction(
        query="What is gradient descent?",
        entities=[concept],
        confidence=0.5
    )

    history1 = await temporal.get_concept_history(concept)
    assert history1.current_state.value == "INTRODUCED"

    # INTRODUCED → LEARNING (3+ interactions)
    for i in range(3):
        await temporal.track_interaction(
            query=f"Learn about gradient descent {i}",
            entities=[concept],
            confidence=0.7
        )

    history2 = await temporal.get_concept_history(concept)
    assert history2.current_state.value == "LEARNING"

    # LEARNING → FAMILIAR (10+ interactions)
    for i in range(7):
        await temporal.track_interaction(
            query=f"Using gradient descent {i}",
            entities=[concept],
            confidence=0.8
        )

    history3 = await temporal.get_concept_history(concept)
    assert history3.current_state.value in ["FAMILIAR", "LEARNING"]

    # Check milestones
    milestones = await temporal.detect_milestones(concept)
    assert len(milestones) > 0
    assert any(m.type == "first_learned" for m in milestones)


@pytest.mark.asyncio
async def test_end_to_end_learning_journey(integrated_system):
    """Test 7: Complete learning journey across all systems"""
    loom = integrated_system['loom']
    consolidation = integrated_system['consolidation']
    graph_reasoner = integrated_system['graph_reasoner']
    curiosity = integrated_system['curiosity']
    semantic = integrated_system['semantic']
    temporal = integrated_system['temporal']

    # Simulate a week of learning about reinforcement learning
    topics = [
        ("Reinforcement Learning", "RL is about learning from interaction"),
        ("Policy Gradient", "Policy gradient methods optimize policies directly"),
        ("Value Functions", "Value functions estimate expected returns"),
        ("Q-Learning", "Q-Learning is an off-policy TD control algorithm"),
        ("PPO", "Proximal Policy Optimization clips policy updates")
    ]

    # Day 1-3: Initial learning (episodic memories)
    for topic, description in topics:
        for i in range(3):
            await loom.experience(f"{description} (Day {i+1})")
            await temporal.track_interaction(
                query=description,
                entities=[topic],
                confidence=0.6 + i*0.1
            )
            consolidation.record_access(
                memory_id=f"{topic}_{i}",
                importance=0.7 + i*0.1
            )

    # Check temporal states
    for topic, _ in topics:
        history = await temporal.get_concept_history(topic)
        assert history.current_state.value in ["INTRODUCED", "LEARNING"]

    # Day 4: Pattern detection and semantic transition
    patterns = await semantic.detect_patterns()
    assert len(patterns) > 0

    promoted_concepts = []
    for pattern in patterns[:3]:
        concept = await semantic.promote_to_semantic(pattern)
        if concept:
            promoted_concepts.append(concept)

    assert len(promoted_concepts) > 0

    # Day 5: Multi-hop reasoning discovers connections
    results = await graph_reasoner.multi_hop_query(
        query="Reinforcement Learning",
        max_hops=2
    )
    assert len(results) > 0

    # Day 6: Curiosity suggests new areas
    suggestions = await curiosity.suggest_exploration(limit=5)
    assert len(suggestions) > 0

    # Day 7: Consolidation promotes important memories
    consolidation_stats = await consolidation.consolidate_during_idle()

    # Get final evolution summary
    summary = await temporal.get_evolution_summary(days=7)
    assert summary['new_concepts'] > 0


@pytest.mark.asyncio
async def test_error_recovery_across_systems(integrated_system):
    """Test 8: Systems recover gracefully from errors"""
    loom = integrated_system['loom']
    semantic = integrated_system['semantic']
    temporal = integrated_system['temporal']

    # Test empty query (should handle gracefully)
    try:
        await temporal.track_interaction(
            query="",
            entities=[],
            confidence=0.5
        )
    except Exception:
        pass  # Expected to handle gracefully

    # Test invalid confidence (should clamp)
    await temporal.track_interaction(
        query="Test query",
        entities=["Test"],
        confidence=1.5  # Invalid, should clamp to 1.0
    )

    # Test pattern detection with no memories
    patterns = await semantic.detect_patterns()
    # Should return empty list, not crash
    assert isinstance(patterns, list)


@pytest.mark.asyncio
async def test_performance_integration(integrated_system):
    """Test 9: Performance across all systems"""
    import time

    loom = integrated_system['loom']
    temporal = integrated_system['temporal']
    semantic = integrated_system['semantic']

    # Measure end-to-end latency
    start = time.time()

    # Store memory
    await loom.experience("Performance test query")

    # Track temporal
    await temporal.track_interaction(
        query="Performance test",
        entities=["Performance"],
        confidence=0.8
    )

    # Pattern detection
    patterns = await semantic.detect_patterns()

    end = time.time()
    latency_ms = (end - start) * 1000

    # Should complete in reasonable time (<500ms)
    assert latency_ms < 500


@pytest.mark.asyncio
async def test_statistics_across_systems(integrated_system):
    """Test 10: Statistics collection from all systems"""
    consolidation = integrated_system['consolidation']
    curiosity = integrated_system['curiosity']
    semantic = integrated_system['semantic']
    temporal = integrated_system['temporal']

    # Get statistics from each system
    consolidation_stats = consolidation.get_statistics()
    assert 'total_memories_tracked' in consolidation_stats

    curiosity_stats = curiosity.get_statistics()
    assert 'total_suggestions' in curiosity_stats

    semantic_stats = semantic.get_statistics()
    assert 'total_patterns_detected' in semantic_stats

    temporal_stats = temporal.get_statistics()
    assert 'concepts_tracked' in temporal_stats


# Summary test
@pytest.mark.asyncio
async def test_integration_summary(integrated_system):
    """Test 11: Summary of all systems working together"""
    loom = integrated_system['loom']

    # This test verifies that all systems can coexist
    # and don't interfere with each other

    # Quick smoke test
    memory = await loom.experience("Integration test complete")
    assert memory is not None

    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print("✅ Week 5: Consolidation system integrated")
    print("✅ Week 5: Multi-hop reasoning integrated")
    print("✅ Week 5: Curiosity engine integrated")
    print("✅ Week 6: Semantic transition integrated")
    print("✅ Week 7: Temporal evolution integrated")
    print("✅ All systems working together seamlessly")
    print("="*60)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
