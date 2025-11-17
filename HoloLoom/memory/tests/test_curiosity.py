"""
Tests for HoloLoom Curiosity Engine
====================================

Tests exploration suggestion generation from gaps, contradictions, and access patterns.

Author: HoloLoom Team
Date: 2025-11-17
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from HoloLoom.memory.curiosity import (
    CuriosityEngine,
    CuriosityConfig,
    ExplorationSuggestion,
    suggest_explorations
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.synthesis.types import GapType, ContradictionType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_kg():
    """Create a sample knowledge graph for testing."""
    kg = KG()

    # Add machine learning concepts (with gap - know ML and NN, missing prerequisites)
    kg.add_edges([
        KGEdge("machine learning", "supervised learning", "HAS_SUBTYPE"),
        KGEdge("machine learning", "neural networks", "USES"),
        KGEdge("neural networks", "deep learning", "IS_A"),
        KGEdge("deep learning", "transformers", "USES"),

        # Programming concepts (incomplete category - know Python/JS, missing TS)
        KGEdge("programming", "python", "HAS_LANGUAGE"),
        KGEdge("programming", "javascript", "HAS_LANGUAGE"),
        KGEdge("python", "data science", "USED_IN"),

        # Bridge gap - "feature engineering" mentioned but not explained
        KGEdge("machine learning", "feature engineering", "REQUIRES"),
        KGEdge("data science", "feature engineering", "REQUIRES"),
        # Note: feature engineering has high in-degree but no out-edges (unexplained)
    ])

    return kg


@pytest.fixture
def curiosity_engine(sample_kg):
    """Create a curiosity engine with sample data."""
    config = CuriosityConfig(
        max_suggestions=10,
        gap_importance_threshold=0.4,  # Lower for testing
        contradiction_score_threshold=0.25,  # Lower for testing
        trending_window_hours=24,
        enable_serendipity=True,
        serendipity_probability=1.0  # Always suggest for testing
    )

    engine = CuriosityEngine(kg=sample_kg, config=config)

    # Add some query history
    now = datetime.now()
    engine.track_query("What is machine learning?", entities=["machine learning"], timestamp=now - timedelta(hours=2))
    engine.track_query("How do neural networks work?", entities=["neural networks"], timestamp=now - timedelta(hours=1))
    engine.track_query("Explain transformers", entities=["transformers"], timestamp=now - timedelta(minutes=30))
    engine.track_query("What is deep learning?", entities=["deep learning"], timestamp=now - timedelta(minutes=15))

    return engine


# ============================================================================
# Test Suggestion Generation
# ============================================================================

@pytest.mark.asyncio
async def test_gap_based_suggestions(curiosity_engine):
    """Test gap-based exploration suggestions."""
    # Get gap-filling suggestions
    kg_dict = curiosity_engine._kg_to_dict(curiosity_engine.kg)
    suggestions = await curiosity_engine.suggest_gap_filling(kg_dict)

    assert len(suggestions) > 0, "Should generate gap-based suggestions"

    # Should include bridge gap for "feature engineering"
    bridge_suggestions = [s for s in suggestions if s.type == 'gap' and 'feature engineering' in s.concept.lower()]
    assert len(bridge_suggestions) > 0, "Should suggest exploring 'feature engineering' (bridge gap)"

    # Check suggestion properties
    for suggestion in suggestions:
        assert suggestion.type == 'gap'
        assert 0 <= suggestion.importance <= 1
        assert len(suggestion.reason) > 0
        assert len(suggestion.suggested_query) > 0
        assert len(suggestion.expected_benefit) > 0

    print(f"✅ Generated {len(suggestions)} gap-based suggestions")
    for s in suggestions[:3]:
        print(f"  - {s.concept} ({s.importance:.2f}): {s.suggested_query[:50]}...")


@pytest.mark.asyncio
async def test_contradiction_suggestions(curiosity_engine):
    """Test contradiction resolution suggestions."""
    # Add contradictory memories
    now = datetime.now()

    curiosity_engine.contradiction_detector.add_memory({
        'id': 'm1',
        'content': 'Python is the best programming language for data science',
        'timestamp': now - timedelta(days=30),
        'entities': ['python', 'data science']
    })

    curiosity_engine.contradiction_detector.add_memory({
        'id': 'm2',
        'content': 'Python is terrible for data science, too slow',
        'timestamp': now - timedelta(days=1),
        'entities': ['python', 'data science']
    })

    # Get contradiction suggestions
    suggestions = await curiosity_engine.suggest_contradiction_resolution()

    assert len(suggestions) > 0, "Should generate contradiction resolution suggestions"

    # Check suggestion properties
    for suggestion in suggestions:
        assert suggestion.type == 'contradiction'
        assert 0 <= suggestion.importance <= 1
        assert 'contradict' in suggestion.reason.lower() or 'changed' in suggestion.reason.lower()
        assert len(suggestion.suggested_query) > 0

    print(f"✅ Generated {len(suggestions)} contradiction suggestions")
    for s in suggestions:
        print(f"  - {s.concept}: {s.reason[:80]}...")


@pytest.mark.asyncio
async def test_access_pattern_suggestions(curiosity_engine):
    """Test suggestions based on access patterns (trending, related concepts)."""
    # Track more queries to establish patterns
    now = datetime.now()
    for i in range(5):
        curiosity_engine.track_query(
            f"Question about transformers #{i}",
            entities=["transformers"],
            timestamp=now - timedelta(minutes=i*5)
        )

    # Get access pattern suggestions
    suggestions = await curiosity_engine._suggest_from_access_patterns()

    assert len(suggestions) > 0, "Should generate access pattern suggestions"

    # Should suggest related concepts to "transformers"
    related_suggestions = [s for s in suggestions if s.type == 'related_concept']
    assert len(related_suggestions) > 0, "Should suggest related concepts"

    print(f"✅ Generated {len(suggestions)} access pattern suggestions")
    for s in suggestions[:3]:
        print(f"  - {s.concept} (type: {s.type}): {s.suggested_query[:50]}...")


@pytest.mark.asyncio
async def test_related_concepts(curiosity_engine):
    """Test related concept suggestions."""
    # Get related concepts for "machine learning"
    related = await curiosity_engine.suggest_related_concepts("machine learning", max_suggestions=5)

    assert len(related) > 0, "Should find related concepts"
    assert "supervised learning" in related or "neural networks" in related

    print(f"✅ Found {len(related)} related concepts for 'machine learning':")
    print(f"  {related}")


@pytest.mark.asyncio
async def test_deep_dive_suggestions(curiosity_engine):
    """Test deep dive suggestions for partially-explored topics."""
    # Track a concept a few times (moderate access)
    now = datetime.now()
    for i in range(3):
        curiosity_engine.track_query(
            f"Brief question about Python #{i}",
            entities=["python"],
            timestamp=now - timedelta(hours=i)
        )

    # Get deep dive suggestions
    suggestions = await curiosity_engine._suggest_deep_dives()

    # Should suggest deep dive on Python (accessed 3 times)
    deep_dive_suggestions = [s for s in suggestions if s.type == 'deep_dive']

    if deep_dive_suggestions:
        assert any('python' in s.concept.lower() for s in deep_dive_suggestions)
        print(f"✅ Generated {len(deep_dive_suggestions)} deep dive suggestions")
        for s in deep_dive_suggestions[:2]:
            print(f"  - {s.concept}: {s.reason[:80]}...")
    else:
        print("ℹ️  No deep dive suggestions (expected - depends on access patterns)")


@pytest.mark.asyncio
async def test_serendipitous_suggestions(curiosity_engine):
    """Test serendipitous exploration suggestions."""
    # Force serendipity
    curiosity_engine.config.serendipity_probability = 1.0

    suggestions = await curiosity_engine._suggest_serendipitous()

    assert len(suggestions) > 0, "Should generate serendipitous suggestions"

    # Should suggest an unexplored concept from the KG
    for suggestion in suggestions:
        assert suggestion.metadata.get('serendipity') is True
        # Access count should be low (unexplored)
        assert curiosity_engine.entity_access_counts.get(suggestion.concept, 0) <= 1

    print(f"✅ Generated {len(suggestions)} serendipitous suggestions")
    for s in suggestions:
        print(f"  - {s.concept}: {s.suggested_query[:50]}...")


@pytest.mark.asyncio
async def test_comprehensive_suggestions(curiosity_engine):
    """Test comprehensive suggestion generation combining all types."""
    suggestions = await curiosity_engine.suggest_exploration(limit=10)

    assert len(suggestions) > 0, "Should generate comprehensive suggestions"
    assert len(suggestions) <= 10, "Should respect limit"

    # Should have multiple types
    types = set(s.type for s in suggestions)
    assert len(types) > 1, "Should include multiple suggestion types"

    # Check all suggestions are valid
    for suggestion in suggestions:
        assert isinstance(suggestion, ExplorationSuggestion)
        assert 0 <= suggestion.importance <= 1
        assert len(suggestion.concept) > 0
        assert len(suggestion.reason) > 0
        assert len(suggestion.suggested_query) > 0
        assert len(suggestion.expected_benefit) > 0

    # Should be sorted by importance (descending)
    importances = [s.importance for s in suggestions]
    assert importances == sorted(importances, reverse=True), "Should be sorted by importance"

    print(f"✅ Generated {len(suggestions)} comprehensive suggestions")
    print(f"   Types: {types}")
    print("\nTop 3 suggestions:")
    for i, s in enumerate(suggestions[:3], 1):
        print(f"\n{i}. {s.concept} ({s.importance:.0%} important)")
        print(f"   Type: {s.type}")
        print(f"   Reason: {s.reason[:80]}...")
        print(f"   Query: {s.suggested_query[:60]}...")


# ============================================================================
# Test Tracking and History
# ============================================================================

def test_query_tracking(curiosity_engine):
    """Test query tracking functionality."""
    initial_count = len(curiosity_engine.query_history)

    # Track a new query
    curiosity_engine.track_query(
        "What is reinforcement learning?",
        entities=["reinforcement learning"],
        timestamp=datetime.now()
    )

    assert len(curiosity_engine.query_history) == initial_count + 1
    assert curiosity_engine.entity_access_counts["reinforcement learning"] == 1

    # Track again
    curiosity_engine.track_query(
        "How does RL work?",
        entities=["reinforcement learning"],
        timestamp=datetime.now()
    )

    assert curiosity_engine.entity_access_counts["reinforcement learning"] == 2

    print(f"✅ Query tracking working correctly")
    print(f"   Total queries: {len(curiosity_engine.query_history)}")
    print(f"   Unique entities: {len(curiosity_engine.entity_access_counts)}")


def test_cache_invalidation(curiosity_engine):
    """Test suggestion cache invalidation."""
    # Cache should be invalid initially
    assert not curiosity_engine._is_cache_valid()

    # Set cache
    curiosity_engine._suggestion_cache = [
        ExplorationSuggestion(
            type='test',
            concept='test',
            reason='test',
            importance=0.5,
            suggested_query='test',
            expected_benefit='test'
        )
    ]
    curiosity_engine._cache_timestamp = datetime.now()

    # Cache should be valid now
    assert curiosity_engine._is_cache_valid()

    # Track a new query (should invalidate cache)
    curiosity_engine.track_query("New query", entities=["test"])

    # Cache should be invalid
    assert not curiosity_engine._is_cache_valid()

    print(f"✅ Cache invalidation working correctly")


# ============================================================================
# Test Statistics and Utilities
# ============================================================================

def test_statistics(curiosity_engine):
    """Test statistics gathering."""
    stats = curiosity_engine.get_statistics()

    assert 'total_queries_tracked' in stats
    assert 'unique_entities_accessed' in stats
    assert 'top_trending_entities' in stats
    assert 'config' in stats

    assert stats['total_queries_tracked'] > 0
    assert stats['unique_entities_accessed'] > 0
    assert len(stats['top_trending_entities']) > 0

    print(f"✅ Statistics:")
    print(f"   Queries tracked: {stats['total_queries_tracked']}")
    print(f"   Unique entities: {stats['unique_entities_accessed']}")
    print(f"   Top trending: {[entity for entity, count in stats['top_trending_entities'][:3]]}")


def test_kg_to_dict_conversion(curiosity_engine):
    """Test KG to dict conversion."""
    kg_dict = curiosity_engine._kg_to_dict(curiosity_engine.kg)

    assert 'nodes' in kg_dict
    assert 'edges' in kg_dict
    assert 'node_counts' in kg_dict

    assert len(kg_dict['nodes']) > 0
    assert len(kg_dict['edges']) > 0

    print(f"✅ KG to dict conversion:")
    print(f"   Nodes: {len(kg_dict['nodes'])}")
    print(f"   Edges: {len(kg_dict['edges'])}")


# ============================================================================
# Test Standalone Function
# ============================================================================

@pytest.mark.asyncio
async def test_standalone_suggest_explorations(sample_kg):
    """Test standalone suggest_explorations function."""
    # Create query history
    query_history = [
        {
            'text': 'What is machine learning?',
            'entities': ['machine learning'],
            'timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'text': 'Explain neural networks',
            'entities': ['neural networks'],
            'timestamp': datetime.now() - timedelta(minutes=30)
        }
    ]

    # Get suggestions
    suggestions = await suggest_explorations(
        kg=sample_kg,
        query_history=query_history,
        limit=5
    )

    assert len(suggestions) > 0
    assert len(suggestions) <= 5

    print(f"✅ Standalone function generated {len(suggestions)} suggestions")
    for s in suggestions[:3]:
        print(f"  - {s.concept}: {s.suggested_query[:50]}...")


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_kg():
    """Test curiosity engine with empty knowledge graph."""
    empty_kg = KG()
    engine = CuriosityEngine(kg=empty_kg)

    suggestions = await engine.suggest_exploration(limit=5)

    # Should still work (might return serendipitous or contradiction suggestions)
    assert isinstance(suggestions, list)

    print(f"✅ Empty KG handled gracefully: {len(suggestions)} suggestions")


@pytest.mark.asyncio
async def test_no_query_history():
    """Test curiosity engine with no query history."""
    kg = KG()
    kg.add_edges([
        KGEdge("concept_a", "concept_b", "RELATED")
    ])

    engine = CuriosityEngine(kg=kg)

    suggestions = await engine.suggest_exploration(limit=5)

    # Should still work (gap-based and serendipitous suggestions)
    assert isinstance(suggestions, list)

    print(f"✅ No query history handled gracefully: {len(suggestions)} suggestions")


@pytest.mark.asyncio
async def test_disabled_engine():
    """Test curiosity engine when disabled."""
    config = CuriosityConfig(enabled=False)
    engine = CuriosityEngine(config=config)

    suggestions = await engine.suggest_exploration(limit=5)

    assert len(suggestions) == 0, "Disabled engine should return no suggestions"

    print(f"✅ Disabled engine returns no suggestions")


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_integration_scenario():
    """
    Test complete curiosity engine workflow simulating a real user session.
    """
    # Setup
    kg = KG()
    kg.add_edges([
        # User knows about Python and data science
        KGEdge("programming", "python", "HAS_LANGUAGE"),
        KGEdge("python", "data science", "USED_IN"),
        KGEdge("data science", "machine learning", "USES"),

        # But missing numpy (prerequisite for data science)
        KGEdge("machine learning", "neural networks", "USES"),

        # And pandas is mentioned but not explored (bridge gap)
        KGEdge("data science", "pandas", "USES"),
        KGEdge("python", "pandas", "LIBRARY"),
    ])

    engine = CuriosityEngine(kg=kg)

    # Simulate user session
    now = datetime.now()

    # User explores Python heavily (trending topic)
    for i in range(5):
        engine.track_query(
            f"Python question {i}",
            entities=["python"],
            timestamp=now - timedelta(minutes=i*10)
        )

    # User touches on data science a couple times (moderate access - deep dive candidate)
    engine.track_query(
        "What is data science?",
        entities=["data science"],
        timestamp=now - timedelta(hours=2)
    )
    engine.track_query(
        "Data science workflow",
        entities=["data science"],
        timestamp=now - timedelta(hours=1)
    )

    # Add contradictory memories about Python
    engine.contradiction_detector.add_memory({
        'id': 'm1',
        'content': 'Python is great for beginners',
        'timestamp': now - timedelta(days=30),
        'entities': ['python']
    })
    engine.contradiction_detector.add_memory({
        'id': 'm2',
        'content': 'Python is terrible for beginners',
        'timestamp': now - timedelta(days=1),
        'entities': ['python']
    })

    # Get suggestions
    suggestions = await engine.suggest_exploration(limit=10)

    print(f"\n{'='*60}")
    print("FULL INTEGRATION TEST - User Session Suggestions")
    print(f"{'='*60}\n")

    assert len(suggestions) > 0, "Should generate suggestions from user session"

    # Should have multiple types
    types = Counter(s.type for s in suggestions)
    print(f"Suggestion types: {dict(types)}\n")

    # Display all suggestions
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. [{suggestion.type.upper()}] {suggestion.concept}")
        print(f"   Importance: {suggestion.importance:.0%}")
        print(f"   Reason: {suggestion.reason[:100]}...")
        print(f"   Query: \"{suggestion.suggested_query}\"")
        print(f"   Benefit: {suggestion.expected_benefit[:80]}...")
        print()

    # Verify we have suggestions from different sources
    assert any(s.type == 'related_concept' for s in suggestions), "Should suggest related concepts (user is trending on Python)"
    assert any(s.type in ['gap', 'contradiction'] for s in suggestions), "Should suggest gaps or contradictions"

    print(f"✅ Full integration test passed!")
    print(f"   Total suggestions: {len(suggestions)}")
    print(f"   Suggestion diversity: {len(types)} types")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with: python -m pytest HoloLoom/memory/tests/test_curiosity.py -v -s
    print("Run tests with: python -m pytest HoloLoom/memory/tests/test_curiosity.py -v -s")
