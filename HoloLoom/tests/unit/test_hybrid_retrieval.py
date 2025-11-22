"""
Unit tests for Hybrid Retrieval System (Week 4).

Tests:
- Semantic search (with and without sentence-transformers)
- BM25 keyword search
- Graph traversal
- Reciprocal Rank Fusion
- Hybrid retrieval (all methods combined)

Based on Research:
- LangMem: "Hybrid retrieval combines semantic + keyword + graph"
- Graphiti: "Multi-hop traversal enriches context"
- Mem0: "Rank fusion gives best of all worlds"
"""

import pytest
import asyncio
from datetime import datetime

from HoloLoom.memory.hybrid_retrieval import (
    SemanticRetriever,
    BM25Retriever,
    GraphRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
    create_hybrid_retriever
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.protocols.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_memories():
    """Create sample memories for retrieval testing."""
    return [
        MemoryShard(
            id="m1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits",
            entities=["ThompsonSampling", "Bayesian", "MultiArmedBandit"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="m2",
            text="Epsilon-greedy is a simple exploration strategy",
            entities=["EpsilonGreedy", "Exploration"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="m3",
            text="UCB (Upper Confidence Bound) balances exploration and exploitation",
            entities=["UCB", "Exploration", "Exploitation"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="m4",
            text="Bayesian optimization uses Gaussian processes",
            entities=["Bayesian", "GaussianProcess"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="m5",
            text="HoloLoom uses Thompson Sampling for tool selection",
            entities=["HoloLoom", "ThompsonSampling"],
            metadata={"timestamp": datetime.now().isoformat()}
        )
    ]


@pytest.fixture
def knowledge_graph():
    """Create knowledge graph with sample entities."""
    kg = KG()

    # Add edges
    edges = [
        KGEdge("ThompsonSampling", "Bayesian", "IS_A", 1.0),
        KGEdge("ThompsonSampling", "MultiArmedBandit", "SOLVES", 1.0),
        KGEdge("EpsilonGreedy", "Exploration", "USES", 1.0),
        KGEdge("UCB", "Exploration", "USES", 1.0),
        KGEdge("Bayesian", "GaussianProcess", "USES", 1.0),
        KGEdge("HoloLoom", "ThompsonSampling", "USES", 1.0)
    ]

    kg.add_edges(edges)

    return kg


# ============================================================================
# Test: BM25 Retriever
# ============================================================================

@pytest.mark.asyncio
async def test_bm25_basic_retrieval(sample_memories):
    """Test basic BM25 keyword search."""
    retriever = BM25Retriever()

    results = await retriever.retrieve(
        query="Thompson Sampling Bayesian",
        memories=sample_memories,
        limit=3
    )

    # Should find memories with matching keywords
    assert len(results) > 0
    assert len(results) <= 3

    # Top result should contain query terms
    top_memory, top_score = results[0]
    assert "Thompson" in top_memory.text or "Bayesian" in top_memory.text
    assert top_score > 0


@pytest.mark.asyncio
async def test_bm25_ranking(sample_memories):
    """Test BM25 ranking quality."""
    retriever = BM25Retriever()

    results = await retriever.retrieve(
        query="Thompson Sampling",
        memories=sample_memories,
        limit=5
    )

    # m1 and m5 contain "Thompson Sampling", should rank higher
    top_ids = [mem.id for mem, _ in results[:2]]
    assert "m1" in top_ids or "m5" in top_ids


@pytest.mark.asyncio
async def test_bm25_no_matches(sample_memories):
    """Test BM25 with query that has no matches."""
    retriever = BM25Retriever()

    results = await retriever.retrieve(
        query="quantum computing blockchain",
        memories=sample_memories,
        limit=5
    )

    # Should return results (even with 0 scores)
    assert isinstance(results, list)


def test_bm25_tokenization():
    """Test BM25 tokenizer."""
    retriever = BM25Retriever()

    tokens = retriever.tokenize("Thompson-Sampling is a Bayesian approach!")

    assert "thompson" in tokens
    assert "sampling" in tokens
    assert "bayesian" in tokens
    assert "approach" in tokens


def test_bm25_idf():
    """Test IDF calculation."""
    retriever = BM25Retriever()

    # Set up simple corpus
    retriever.num_docs = 5
    retriever.doc_freqs = {
        "common": 5,  # Appears in all docs
        "rare": 1     # Appears in one doc
    }

    # IDF should be higher for rare terms
    common_idf = retriever.idf("common")
    rare_idf = retriever.idf("rare")

    assert rare_idf > common_idf
    assert common_idf < 0  # log((5-5+0.5)/(5+0.5)) is negative


# ============================================================================
# Test: Graph Retriever
# ============================================================================

@pytest.mark.asyncio
async def test_graph_basic_retrieval(sample_memories, knowledge_graph):
    """Test basic graph traversal retrieval."""
    retriever = GraphRetriever(kg=knowledge_graph, max_hops=2)

    results = await retriever.retrieve(
        query="ThompsonSampling",
        memories=sample_memories,
        limit=5
    )

    # Should find memories with related entities
    assert len(results) > 0

    # Should include memory with ThompsonSampling
    mem_ids = [mem.id for mem, _ in results]
    assert "m1" in mem_ids or "m5" in mem_ids


@pytest.mark.asyncio
async def test_graph_multi_hop_traversal(knowledge_graph):
    """Test multi-hop graph traversal."""
    retriever = GraphRetriever(kg=knowledge_graph, max_hops=2)

    # Traverse from ThompsonSampling
    reachable = retriever.traverse("ThompsonSampling", max_hops=2)

    # Should reach direct neighbors
    assert "Bayesian" in reachable
    assert "MultiArmedBandit" in reachable

    # Should reach 2-hop neighbors
    assert "GaussianProcess" in reachable  # ThompsonSampling → Bayesian → GaussianProcess

    # Score should decay with distance
    assert reachable["ThompsonSampling"] == 1.0  # Start node
    assert reachable["Bayesian"] < 1.0  # 1 hop
    assert reachable["GaussianProcess"] < reachable["Bayesian"]  # 2 hops


@pytest.mark.asyncio
async def test_graph_no_entities(sample_memories, knowledge_graph):
    """Test graph retrieval when query has no entities."""
    retriever = GraphRetriever(kg=knowledge_graph)

    results = await retriever.retrieve(
        query="some random text without entities",
        memories=sample_memories,
        limit=5
    )

    # Should return empty list (no entities found)
    assert results == []


def test_graph_entity_extraction(knowledge_graph):
    """Test entity extraction from query."""
    retriever = GraphRetriever(kg=knowledge_graph)

    # Test with entities in graph
    entities = retriever.extract_entities("ThompsonSampling and Bayesian")

    assert "ThompsonSampling" in entities or "Bayesian" in entities


# ============================================================================
# Test: Semantic Retriever (Fallback Mode)
# ============================================================================

@pytest.mark.asyncio
async def test_semantic_fallback_retrieval(sample_memories):
    """Test semantic retriever with fallback (no sentence-transformers)."""
    # Force fallback by creating retriever without model
    retriever = SemanticRetriever(enable_fallback=True)
    retriever.available = False  # Simulate unavailable

    results = await retriever.retrieve(
        query="Thompson Sampling exploration",
        memories=sample_memories,
        limit=3
    )

    # Should use keyword fallback
    assert len(results) > 0
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_semantic_no_fallback():
    """Test semantic retriever fails gracefully without fallback."""
    retriever = SemanticRetriever(enable_fallback=False)
    retriever.available = False  # Simulate unavailable

    results = await retriever.retrieve(
        query="test query",
        memories=[],
        limit=5
    )

    # Should return empty list
    assert results == []


# ============================================================================
# Test: Reciprocal Rank Fusion
# ============================================================================

def test_rrf_basic():
    """Test basic RRF fusion."""
    # Create two rankings
    m1 = MemoryShard(id="m1", text="Test 1", metadata={"timestamp": datetime.now().isoformat()})
    m2 = MemoryShard(id="m2", text="Test 2", metadata={"timestamp": datetime.now().isoformat()})
    m3 = MemoryShard(id="m3", text="Test 3", metadata={"timestamp": datetime.now().isoformat()})

    ranking1 = [(m1, 0.9), (m2, 0.5), (m3, 0.1)]
    ranking2 = [(m2, 0.8), (m3, 0.6), (m1, 0.2)]

    fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)

    # Should combine rankings
    assert len(fused) == 3

    # m2 appears high in both rankings, should rank high in fusion
    fused_ids = [mem.id for mem, _ in fused]
    assert fused_ids[0] == "m2" or fused_ids[0] == "m1"


def test_rrf_single_ranking():
    """Test RRF with single ranking (passthrough)."""
    m1 = MemoryShard(id="m1", text="Test 1", metadata={"timestamp": datetime.now().isoformat()})
    m2 = MemoryShard(id="m2", text="Test 2", metadata={"timestamp": datetime.now().isoformat()})

    ranking = [(m1, 0.9), (m2, 0.5)]

    fused = reciprocal_rank_fusion([ranking], k=60)

    # Should preserve order
    assert len(fused) == 2
    assert fused[0][0].id == "m1"


def test_rrf_empty_rankings():
    """Test RRF with empty rankings."""
    fused = reciprocal_rank_fusion([], k=60)

    assert fused == []


# ============================================================================
# Test: Hybrid Retriever
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_basic_retrieval(sample_memories, knowledge_graph):
    """Test basic hybrid retrieval."""
    retriever = HybridRetriever(
        kg=knowledge_graph,
        enable_semantic=False,  # Disable to avoid sentence-transformers dependency
        enable_bm25=True,
        enable_graph=True
    )

    result = await retriever.retrieve(
        query="ThompsonSampling Bayesian",
        memories=sample_memories,
        limit=3
    )

    assert len(result.memories) > 0
    assert len(result.memories) <= 3
    assert len(result.scores) == len(result.memories)
    assert result.retrieval_time_ms >= 0
    assert result.total_candidates == len(sample_memories)


@pytest.mark.asyncio
async def test_hybrid_bm25_only(sample_memories, knowledge_graph):
    """Test hybrid retrieval with BM25 only."""
    retriever = HybridRetriever(
        kg=knowledge_graph,
        enable_semantic=False,
        enable_bm25=True,
        enable_graph=False
    )

    result = await retriever.retrieve(
        query="Thompson Sampling",
        memories=sample_memories,
        limit=5
    )

    assert len(result.memories) > 0
    assert "bm25" in result.metadata["methods"]


@pytest.mark.asyncio
async def test_hybrid_graph_only(sample_memories, knowledge_graph):
    """Test hybrid retrieval with graph only."""
    retriever = HybridRetriever(
        kg=knowledge_graph,
        enable_semantic=False,
        enable_bm25=False,
        enable_graph=True
    )

    result = await retriever.retrieve(
        query="ThompsonSampling",
        memories=sample_memories,
        limit=5
    )

    # May return empty if no graph matches
    assert isinstance(result.memories, list)


@pytest.mark.asyncio
async def test_hybrid_all_methods(sample_memories, knowledge_graph):
    """Test hybrid retrieval with all methods."""
    retriever = HybridRetriever(
        kg=knowledge_graph,
        enable_semantic=False,  # Disable to avoid dependency
        enable_bm25=True,
        enable_graph=True
    )

    result = await retriever.retrieve(
        query="Thompson Sampling exploration",
        memories=sample_memories,
        limit=3
    )

    # Should combine multiple methods
    assert len(result.memories) > 0
    assert len(result.metadata["methods"]) >= 1


@pytest.mark.asyncio
async def test_hybrid_empty_memories(knowledge_graph):
    """Test hybrid retrieval with no memories."""
    retriever = HybridRetriever(kg=knowledge_graph)

    result = await retriever.retrieve(
        query="test query",
        memories=[],
        limit=5
    )

    assert result.memories == []
    assert result.scores == []


# ============================================================================
# Test: Factory Functions
# ============================================================================

def test_create_hybrid_retriever(knowledge_graph):
    """Test hybrid retriever factory."""
    retriever = create_hybrid_retriever(
        kg=knowledge_graph,
        enable_all=True
    )

    assert retriever.semantic_retriever is not None
    assert retriever.bm25_retriever is not None
    assert retriever.graph_retriever is not None


def test_create_hybrid_retriever_minimal(knowledge_graph):
    """Test hybrid retriever with minimal config."""
    retriever = create_hybrid_retriever(
        kg=knowledge_graph,
        enable_all=False
    )

    # Should still create retriever
    assert retriever is not None
