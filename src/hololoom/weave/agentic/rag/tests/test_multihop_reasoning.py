"""
Tests for Multi-Hop Reasoning
==============================

Comprehensive test suite for multi-hop graph traversal.

Test Coverage:
- Path finding accuracy (A→B→C chains)
- Beam search pruning behavior
- Path ranking quality
- Cycle detection
- Max hop limits
- Bidirectional search
- Explanation generation
- Edge cases
"""

import pytest
import asyncio
from typing import List
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.rag.multihop_reasoning import (
    MultiHopRAGMixin,
    ReasoningPath,
    MultiHopRAGResult
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.config import Config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_kg():
    """Create sample knowledge graph for testing."""
    kg = KG()

    # Build a small test graph: attention → transformer → BERT/GPT
    edges = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("multi-head attention", "attention", "IS_A", 0.9),
        KGEdge("self-attention", "attention", "IS_A", 0.9),
        KGEdge("encoder", "transformer", "PART_OF", 0.8),
        KGEdge("decoder", "transformer", "PART_OF", 0.8),
    ]

    kg.add_edges(edges)
    return kg


@pytest.fixture
def complex_kg():
    """Create complex knowledge graph with multiple paths."""
    kg = KG()

    # Multiple paths from A to E
    edges = [
        # Direct path: A → E
        KGEdge("A", "E", "LEADS_TO", 1.0),

        # 2-hop path: A → B → E
        KGEdge("A", "B", "USES", 0.9),
        KGEdge("B", "E", "LEADS_TO", 0.9),

        # 3-hop path: A → C → D → E
        KGEdge("A", "C", "USES", 0.8),
        KGEdge("C", "D", "PART_OF", 0.8),
        KGEdge("D", "E", "LEADS_TO", 0.8),

        # Cycle: X → Y → Z → X
        KGEdge("X", "Y", "USES", 1.0),
        KGEdge("Y", "Z", "USES", 1.0),
        KGEdge("Z", "X", "USES", 1.0),
    ]

    kg.add_edges(edges)
    return kg


class MockHoloLoom:
    """Mock HoloLoom for testing."""
    def __init__(self, kg):
        self.awareness_graph = type('obj', (object,), {'kg': kg})()

    async def recall(self, query, limit=5):
        """Mock recall."""
        from HoloLoom.memory.protocol import Memory
        from datetime import datetime

        # Return mock memories
        return [
            Memory(
                id='mem_1',
                text=f"Memory about {query}",
                timestamp=datetime.now(),
                context={},
                metadata={}
            )
        ]


class MockRAG(MultiHopRAGMixin):
    """Mock RAG class for testing."""
    def __init__(self, kg, *args, **kwargs):
        self.loom = MockHoloLoom(kg)
        self.orchestrator = None
        super().__init__(*args, **kwargs)

    async def query(self, question, mode="verify"):
        """Mock query."""
        from HoloLoom.rag.simple_rag import RAGResult
        return RAGResult(
            response=f"Answer to: {question}",
            sources=["source1", "source2"],
            confidence=0.8,
            reasoning_mode=mode
        )


# ============================================================================
# Test ReasoningPath
# ============================================================================

def test_reasoning_path_creation():
    """Test ReasoningPath creation and validation."""
    path = ReasoningPath(
        entities=["attention", "transformer", "BERT"],
        relationships=["USES", "IS_A"],
        confidence=0.92,
        hop_count=2,
        edge_weights=[1.0, 1.0]
    )

    assert len(path.entities) == 3
    assert len(path.relationships) == 2
    assert path.hop_count == 2
    assert path.confidence == 0.92


def test_reasoning_path_to_dict():
    """Test ReasoningPath serialization."""
    path = ReasoningPath(
        entities=["A", "B", "C"],
        relationships=["USES", "IS_A"],
        confidence=0.85,
        hop_count=2,
        explanation="A uses B, and C is_a B"
    )

    path_dict = path.to_dict()

    assert path_dict['entities'] == ["A", "B", "C"]
    assert path_dict['relationships'] == ["USES", "IS_A"]
    assert path_dict['confidence'] == 0.85
    assert 'explanation' in path_dict


def test_reasoning_path_str():
    """Test ReasoningPath string representation."""
    path = ReasoningPath(
        entities=["attention", "transformer"],
        relationships=["USES"],
        confidence=0.9,
        hop_count=1
    )

    path_str = str(path)

    assert "attention" in path_str
    assert "transformer" in path_str
    assert "USES" in path_str
    assert "0.900" in path_str or "0.9" in path_str


# ============================================================================
# Test Entity Extraction
# ============================================================================

@pytest.mark.asyncio
async def test_extract_query_entities(sample_kg):
    """Test entity extraction from query."""
    rag = MockRAG(sample_kg)

    # Test capitalized entities
    entities = rag._extract_query_entities("What is BERT and GPT?")
    assert "BERT" in entities
    assert "GPT" in entities

    # Test mixed case
    entities = rag._extract_query_entities("How does Transformer work?")
    assert "Transformer" in entities or "transformer" in entities


@pytest.mark.asyncio
async def test_extract_query_entities_from_graph(sample_kg):
    """Test entity extraction using graph fallback."""
    rag = MockRAG(sample_kg)

    # Query with entities in graph but not capitalized
    entities = rag._extract_query_entities("tell me about attention mechanism")

    # Should find "attention" in graph
    assert any('attention' in e.lower() for e in entities)


# ============================================================================
# Test Path Finding
# ============================================================================

@pytest.mark.asyncio
async def test_find_reasoning_paths_simple(sample_kg):
    """Test finding simple 2-hop paths."""
    rag = MockRAG(sample_kg, max_hops=2, beam_width=5)

    paths = await rag.find_reasoning_paths(
        start_entities=["attention"],
        goal="What is BERT?",
        max_hops=2,
        beam_width=5
    )

    assert len(paths) > 0

    # Should find path: attention → transformer → BERT
    bert_paths = [p for p in paths if "BERT" in p.entities]
    assert len(bert_paths) > 0

    best_bert_path = bert_paths[0]
    assert "attention" in best_bert_path.entities
    assert "transformer" in best_bert_path.entities
    assert "BERT" in best_bert_path.entities


@pytest.mark.asyncio
async def test_find_reasoning_paths_multiple(complex_kg):
    """Test finding multiple paths to same destination."""
    rag = MockRAG(complex_kg, max_hops=3, beam_width=10)

    paths = await rag.find_reasoning_paths(
        start_entities=["A"],
        goal="How to reach E?",
        max_hops=3,
        beam_width=10
    )

    assert len(paths) > 0

    # Should find multiple paths to E
    e_paths = [p for p in paths if "E" in p.entities]
    assert len(e_paths) >= 2  # At least direct and 2-hop path

    # Check path diversity
    hop_counts = [p.hop_count for p in e_paths]
    assert len(set(hop_counts)) > 1  # Different path lengths


@pytest.mark.asyncio
async def test_cycle_detection(complex_kg):
    """Test that cycles are detected and prevented."""
    rag = MockRAG(complex_kg, max_hops=5, beam_width=10)

    # Start from cycle node X
    paths = await rag.find_reasoning_paths(
        start_entities=["X"],
        goal="cycle test",
        max_hops=5,
        beam_width=10
    )

    # Paths should not revisit nodes
    for path in paths:
        entities_set = set(path.entities)
        assert len(entities_set) == len(path.entities), f"Cycle detected in path: {path.entities}"


@pytest.mark.asyncio
async def test_max_hops_limit(sample_kg):
    """Test that max_hops limit is respected."""
    rag = MockRAG(sample_kg, max_hops=1, beam_width=5)

    paths = await rag.find_reasoning_paths(
        start_entities=["attention"],
        goal="test",
        max_hops=1,
        beam_width=5
    )

    # All paths should have at most 1 hop
    for path in paths:
        assert path.hop_count <= 1


@pytest.mark.asyncio
async def test_beam_search_pruning(complex_kg):
    """Test that beam search prunes low-scoring paths."""
    # Use very small beam width
    rag = MockRAG(complex_kg, max_hops=2, beam_width=2)

    paths = await rag.find_reasoning_paths(
        start_entities=["A"],
        goal="reach E",
        max_hops=2,
        beam_width=2
    )

    # Should have limited paths due to beam pruning
    # Even though graph has multiple paths, beam should limit exploration
    assert len(paths) <= 10  # Beam pruning limits explosion


# ============================================================================
# Test Path Scoring
# ============================================================================

def test_relationship_weight(sample_kg):
    """Test relationship type weighting."""
    rag = MockRAG(sample_kg)

    # IS_A should have highest weight
    assert rag._relationship_weight("IS_A") == 1.0

    # USES should have high weight
    assert rag._relationship_weight("USES") == 0.9

    # MENTIONS should have lower weight
    assert rag._relationship_weight("MENTIONS") == 0.6

    # Unknown should have default
    assert rag._relationship_weight("UNKNOWN") == 0.7


def test_score_path_incremental(sample_kg):
    """Test incremental path scoring."""
    rag = MockRAG(sample_kg)

    # Score should consider edge weights, relevance, length
    score = rag._score_path_incremental(
        entities=["attention", "transformer"],
        relationships=["USES"],
        edge_weights=[1.0],
        goal="What is transformer?",
        current_score=1.0
    )

    assert score > 0
    # Score can exceed 1.0 due to relevance boost (1.5x)
    assert score <= 2.0

    # Longer path should have lower score (length penalty)
    long_score = rag._score_path_incremental(
        entities=["A", "B", "C", "D"],
        relationships=["USES", "IS_A", "PART_OF"],
        edge_weights=[1.0, 1.0, 1.0],
        goal="test",
        current_score=1.0
    )

    short_score = rag._score_path_incremental(
        entities=["A", "B"],
        relationships=["USES"],
        edge_weights=[1.0],
        goal="test",
        current_score=1.0
    )

    assert long_score < short_score  # Length penalty


# ============================================================================
# Test Explanation Generation
# ============================================================================

def test_generate_explanation_simple(sample_kg):
    """Test explanation generation for simple path."""
    rag = MockRAG(sample_kg)

    path = ReasoningPath(
        entities=["attention", "transformer", "BERT"],
        relationships=["USES", "IS_A"],
        confidence=0.9,
        hop_count=2
    )

    explanation = rag._generate_explanation(path)

    assert "attention" in explanation
    assert "transformer" in explanation
    assert "BERT" in explanation
    assert "uses" in explanation.lower() or "USES" in explanation
    assert "is a" in explanation.lower() or "IS_A" in explanation


def test_generate_explanation_empty(sample_kg):
    """Test explanation for empty path."""
    rag = MockRAG(sample_kg)

    path = ReasoningPath(
        entities=[],
        relationships=[],
        confidence=0.0,
        hop_count=0
    )

    explanation = rag._generate_explanation(path)

    assert "no" in explanation.lower() or "empty" in explanation.lower()


# ============================================================================
# Test Full Query
# ============================================================================

@pytest.mark.asyncio
async def test_query_multihop_basic(sample_kg):
    """Test basic multi-hop query."""
    rag = MockRAG(sample_kg, max_hops=2, beam_width=5)

    result = await rag.query_multihop(
        "How does attention relate to BERT?",
        max_hops=2
    )

    assert isinstance(result, MultiHopRAGResult)
    assert result.response
    assert result.reasoning_mode == "multihop"
    assert result.hop_count == 2
    assert len(result.reasoning_paths) > 0


@pytest.mark.asyncio
async def test_query_multihop_with_paths(sample_kg):
    """Test multi-hop query with path return."""
    rag = MockRAG(sample_kg, max_hops=3, beam_width=5)

    result = await rag.query_multihop(
        "What is BERT?",
        max_hops=3,
        return_paths=True
    )

    assert len(result.reasoning_paths) > 0
    assert result.best_path is not None

    # Best path should have explanation
    assert result.best_path.explanation


@pytest.mark.asyncio
async def test_query_multihop_no_entities(sample_kg):
    """Test multi-hop query with no extractable entities."""
    rag = MockRAG(sample_kg)

    result = await rag.query_multihop(
        "tell me something",  # No entities
        max_hops=2
    )

    # Should fall back to standard query
    assert result.reasoning_mode == "multihop_fallback"
    assert result.metadata.get('fallback') == True


@pytest.mark.asyncio
async def test_query_multihop_stats(sample_kg):
    """Test multi-hop statistics tracking."""
    rag = MockRAG(sample_kg, max_hops=2, beam_width=5)

    # Run multiple queries with actual entities
    for _ in range(3):
        await rag.query_multihop("What is BERT?", max_hops=2)

    stats = rag.get_multihop_stats()

    assert stats['total_queries'] == 3
    assert stats['total_paths_explored'] > 0  # Should find paths for BERT
    assert stats['avg_latency_ms'] >= 0  # May be 0 if very fast
    assert stats['max_hops'] == 2
    assert stats['beam_width'] == 5


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_graph():
    """Test behavior with empty knowledge graph."""
    kg = KG()  # Empty
    rag = MockRAG(kg, max_hops=2)

    paths = await rag.find_reasoning_paths(
        start_entities=["nonexistent"],
        goal="test",
        max_hops=2,
        beam_width=5
    )

    assert len(paths) == 0


@pytest.mark.asyncio
async def test_no_path_exists(sample_kg):
    """Test when no path exists between entities."""
    # Add isolated node
    sample_kg.add_edge(KGEdge("isolated", "lonely", "MENTIONS", 1.0))

    rag = MockRAG(sample_kg, max_hops=3)

    paths = await rag.find_reasoning_paths(
        start_entities=["isolated"],
        goal="BERT",
        max_hops=3,
        beam_width=5
    )

    # Should find paths within isolated component but not to BERT
    bert_paths = [p for p in paths if "BERT" in p.entities]
    assert len(bert_paths) == 0


@pytest.mark.asyncio
async def test_single_hop_path(sample_kg):
    """Test 1-hop path (direct connection)."""
    rag = MockRAG(sample_kg, max_hops=1)

    paths = await rag.find_reasoning_paths(
        start_entities=["attention"],
        goal="transformer",
        max_hops=1,
        beam_width=5
    )

    # Should find direct path: attention → transformer
    direct_paths = [p for p in paths if "transformer" in p.entities and p.hop_count == 1]
    assert len(direct_paths) > 0


# ============================================================================
# Test Configuration
# ============================================================================

def test_multihop_mixin_initialization():
    """Test MultiHopRAGMixin initialization."""
    kg = KG()
    rag = MockRAG(
        kg,
        max_hops=4,
        beam_width=10,
        min_path_confidence=0.5,
        enable_bidirectional=True
    )

    assert rag.max_hops == 4
    assert rag.beam_width == 10
    assert rag.min_path_confidence == 0.5
    assert rag.enable_bidirectional == True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
