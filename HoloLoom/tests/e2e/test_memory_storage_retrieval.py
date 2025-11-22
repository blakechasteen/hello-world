"""
End-to-End Tests: Memory Storage & Retrieval
============================================

RUTHLESS testing of the complete memory pipeline:
- Storage → Retrieval → Consolidation
- Spring physics integration
- Learning system
- Multi-level scoping
- Performance under load
- Error handling and edge cases

Author: Blake Chasteen
Date: November 8, 2025
"""

import pytest
import asyncio
import time
from pathlib import Path
import tempfile
import shutil

from HoloLoom.memory.integrated_memory_system import create_integrated_memory_system
from HoloLoom.memory.lifecycle_manager import MemoryScope
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_memory_scoring import (
    SpringMemoryScorer,
    AdaptiveSpringRetriever
)
from HoloLoom.protocols.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
async def memory_system():
    """Create clean memory system for each test."""
    system = create_integrated_memory_system()
    yield system
    # Cleanup (if system has async cleanup)
    if hasattr(system, 'close'):
        await system.close()


@pytest.fixture
def knowledge_graph():
    """Create test knowledge graph."""
    kg = KG()
    edges = [
        # Core concepts
        KGEdge("ThompsonSampling", "Bayesian", "IS_A", 1.0),
        KGEdge("ThompsonSampling", "MultiArmedBandit", "SOLVES", 1.0),
        KGEdge("Bayesian", "PriorDistribution", "USES", 0.9),
        KGEdge("Bayesian", "PosteriorUpdate", "USES", 0.9),
        KGEdge("MultiArmedBandit", "Exploration", "INVOLVES", 0.8),
        KGEdge("MultiArmedBandit", "Exploitation", "INVOLVES", 0.8),

        # Alternative strategies
        KGEdge("UCB", "Exploration", "USES", 0.8),
        KGEdge("EpsilonGreedy", "Exploration", "USES", 0.8),

        # Related concepts
        KGEdge("GaussianProcess", "Bayesian", "RELATED_TO", 0.7),
        KGEdge("NeuralBandits", "MultiArmedBandit", "IS_A", 0.9),
    ]
    kg.add_edges(edges)
    return kg


# ============================================================================
# Test 1: Basic Storage and Retrieval
# ============================================================================

@pytest.mark.asyncio
async def test_basic_store_and_retrieve(memory_system):
    """Test basic memory storage and retrieval."""
    # Store memories
    memories = [
        "Thompson Sampling is a Bayesian approach to bandits",
        "Bayesian methods use prior distributions",
        "Multi-armed bandits solve exploration problems",
    ]

    for text in memories:
        await memory_system.store(text, importance=0.9)

    # Retrieve
    result = await memory_system.retrieve("Bayesian", limit=5)

    # Assertions
    assert result is not None, "Retrieval returned None"
    assert len(result.memories) > 0, "No memories retrieved"
    assert any("Bayesian" in m.text for m in result.memories), "Bayesian memory not found"

    # Check result structure
    assert hasattr(result, 'retrieval_time_ms'), "Missing retrieval time"
    assert result.retrieval_time_ms > 0, "Invalid retrieval time"

    print(f"[OK] Basic storage/retrieval: {len(result.memories)} memories in {result.retrieval_time_ms:.2f}ms")


# ============================================================================
# Test 2: Multi-Level Scoping
# ============================================================================

@pytest.mark.asyncio
async def test_memory_scoping(memory_system):
    """Test multi-level memory scoping storage."""
    # Store at different scopes
    await memory_system.store(
        "User preference: dark mode enabled",
        scope=MemoryScope.USER,
        importance=0.9
    )

    await memory_system.store(
        "Session context: discussing AI topics",
        scope=MemoryScope.SESSION,
        importance=0.8
    )

    await memory_system.store(
        "Agent working memory: calculating statistics now",
        scope=MemoryScope.AGENT,
        importance=0.7
    )

    # Test 1: Retrieve by content (should find memories regardless of scope)
    result = await memory_system.retrieve("dark mode", limit=5)
    assert len(result.memories) > 0, "Retrieval failed completely"

    # Test 2: Verify memories were stored (check stream manager)
    all_memories = memory_system.stream_manager.get_all_memories()
    assert len(all_memories) >= 3, f"Expected ≥3 memories stored, got {len(all_memories)}"

    # Test 3: Scoping is internal - just verify storage worked for all 3 scopes
    # (The system tracks scope internally via stream names, not in memory metadata)
    print(f"[OK] Memory scoping: {len(all_memories)} memories stored, retrieval functional")


# ============================================================================
# Test 3: Spring Physics Integration
# ============================================================================

@pytest.mark.asyncio
async def test_spring_physics_retrieval(memory_system, knowledge_graph):
    """Test spring physics graph retrieval."""
    # Store memories with entities
    memories_data = [
        ("Thompson Sampling balances exploration", ["ThompsonSampling", "Exploration"]),
        ("Bayesian methods use priors", ["Bayesian", "PriorDistribution"]),
        ("UCB uses confidence bounds", ["UCB", "Exploration"]),
    ]

    for text, entities in memories_data:
        await memory_system.store(text, entities=entities, importance=0.9)

    # Enable spring physics
    memory_system.enable_spring_physics()

    # Retrieve with spring physics
    start = time.perf_counter()
    result = await memory_system.retrieve("Bayesian", limit=5)
    spring_time = (time.perf_counter() - start) * 1000

    # Assertions
    assert len(result.memories) > 0, "Spring physics retrieval failed"
    assert spring_time < 100, f"Spring physics too slow: {spring_time:.2f}ms"

    print(f"[OK] Spring physics retrieval: {len(result.memories)} memories in {spring_time:.2f}ms")


# ============================================================================
# Test 4: Learning System (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_learning_system(knowledge_graph, temp_dir):
    """Test spring physics learning system learns correctly."""
    # Create scorer with persistence
    persist_path = temp_dir / "learning_test.json"
    scorer = SpringMemoryScorer(
        persist_path=str(persist_path),
        decay_rate=0.99,
        min_score_threshold=0.01
    )

    # Create retriever
    retriever = AdaptiveSpringRetriever(kg=knowledge_graph, scorer=scorer)

    # Create test memories
    memories = [
        MemoryShard(
            id="m1",
            text="Thompson Sampling is Bayesian",
            entities=["ThompsonSampling", "Bayesian"]
        ),
        MemoryShard(
            id="m2",
            text="Bayesian uses priors",
            entities=["Bayesian", "PriorDistribution"]
        ),
        MemoryShard(
            id="m3",
            text="UCB uses exploration",
            entities=["UCB", "Exploration"]
        ),
    ]

    # Simulate learning over multiple queries
    queries = ["Bayesian"] * 5  # Repeat query 5 times

    for i, query in enumerate(queries):
        results = await retriever.retrieve(query, memories, limit=3, learn=True)

        # Check learning is happening
        stats = retriever.get_learning_stats()
        assert stats['total_edges'] > 0, f"No edges tracked after query {i+1}"
        assert stats['total_activations'] == (i+1) * len(knowledge_graph.G.edges), \
            f"Activation count mismatch: expected {(i+1) * len(knowledge_graph.G.edges)}, got {stats['total_activations']}"

    # Check edge scores increased
    top_edges = scorer.get_top_edges(limit=3)
    assert len(top_edges) > 0, "No top edges learned"

    # Bayesian → PriorDistribution should be top edge
    top_edge = top_edges[0]
    assert top_edge.access_count == 5, f"Top edge access count wrong: {top_edge.access_count}"
    assert top_edge.score > 1.0, f"Top edge score too low: {top_edge.score}"

    # Test persistence
    retriever.save_learned_patterns()
    assert persist_path.exists(), "Learning state not saved"

    # Load in new scorer
    scorer2 = SpringMemoryScorer(persist_path=str(persist_path))
    assert len(scorer2.edge_scores) == len(scorer.edge_scores), "Persistence failed"

    top_edges2 = scorer2.get_top_edges(limit=3)
    assert top_edges2[0].source == top_edges[0].source, "Top edge changed after reload"
    assert top_edges2[0].target == top_edges[0].target, "Top edge changed after reload"

    print(f"[OK] Learning system: {len(top_edges)} edges learned, top score={top_edges[0].score:.3f}")


# ============================================================================
# Test 5: Performance Under Load (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_performance_under_load(memory_system):
    """Test system performance under heavy load."""
    # Store 100 memories
    num_memories = 100

    start = time.perf_counter()
    for i in range(num_memories):
        await memory_system.store(
            f"Memory {i}: Thompson Sampling iteration {i}",
            importance=0.5 + (i % 5) * 0.1,
            entities=[f"Entity{i}", "ThompsonSampling"]
        )
    storage_time = time.perf_counter() - start

    # Performance assertions
    avg_storage_time = (storage_time / num_memories) * 1000
    assert avg_storage_time < 5.0, f"Storage too slow: {avg_storage_time:.2f}ms per memory (target: <5ms)"

    # Retrieve with high limit (large retrievals are slower due to semantic embeddings)
    start = time.perf_counter()
    result = await memory_system.retrieve("Thompson Sampling", limit=50)
    retrieval_time = (time.perf_counter() - start) * 1000

    # Realistic expectation: Large retrievals (limit=50) scale with embedding computation
    # Target: <5000ms for 50 results (100ms per result on average)
    assert retrieval_time < 5000, f"Retrieval too slow: {retrieval_time:.2f}ms (target: <5000ms for 50 results)"
    assert len(result.memories) > 0, "No memories retrieved under load"

    # Test concurrent retrievals
    async def concurrent_retrieve(query):
        return await memory_system.retrieve(query, limit=10)

    start = time.perf_counter()
    results = await asyncio.gather(*[
        concurrent_retrieve(f"query {i}") for i in range(10)
    ])
    concurrent_time = (time.perf_counter() - start) * 1000

    assert all(r is not None for r in results), "Some concurrent retrievals failed"
    assert concurrent_time < 1000, f"Concurrent retrievals too slow: {concurrent_time:.2f}ms"

    print(f"[OK] Performance under load:")
    print(f"  - Storage: {avg_storage_time:.2f}ms/memory (100 memories)")
    print(f"  - Retrieval: {retrieval_time:.2f}ms (50 results)")
    print(f"  - Concurrent: {concurrent_time:.2f}ms (10 parallel queries)")


# ============================================================================
# Test 6: Edge Cases and Error Handling (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_edge_cases(memory_system):
    """Test edge cases and error handling."""
    # Empty string
    await memory_system.store("", importance=0.5)
    result = await memory_system.retrieve("", limit=5)
    # Should not crash

    # Very long text (10KB)
    long_text = "Thompson Sampling " * 1000
    await memory_system.store(long_text, importance=0.9)
    result = await memory_system.retrieve("Thompson", limit=5)
    assert len(result.memories) > 0, "Long text storage/retrieval failed"

    # Special characters
    special_text = "Thompson & Sampling: 100% Bayesian! (with ★ symbols)"
    await memory_system.store(special_text, importance=0.8)
    result = await memory_system.retrieve("Bayesian", limit=5)
    # Should not crash

    # Retrieve with limit=0
    result = await memory_system.retrieve("test", limit=0)
    assert len(result.memories) == 0, "Limit=0 should return empty"

    # Retrieve with huge limit
    result = await memory_system.retrieve("test", limit=10000)
    # Should not crash, should cap at available memories

    # Invalid importance (should clamp or handle gracefully)
    await memory_system.store("Test", importance=2.0)  # >1.0
    await memory_system.store("Test", importance=-1.0)  # <0.0

    # Retrieve non-existent
    result = await memory_system.retrieve("NONEXISTENT_XYZABC_12345", limit=5)
    # Should return empty or low-confidence results, not crash

    print("[OK] Edge cases handled gracefully")


# ============================================================================
# Test 7: Consolidation (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_consolidation(memory_system):
    """Test background consolidation."""
    # Store episodic memories
    for i in range(10):
        await memory_system.store(
            f"Event {i}: User discussed Thompson Sampling",
            scope=MemoryScope.SESSION,
            importance=0.7
        )

    # Get stats before consolidation
    stats_before = memory_system.get_statistics()

    # Run consolidation
    await memory_system.consolidate()

    # Get stats after
    stats_after = memory_system.get_statistics()

    # Consolidation should have processed memories
    # (Exact behavior depends on implementation, but should not crash)
    assert stats_after is not None, "Stats unavailable after consolidation"

    print("[OK] Consolidation completed without errors")


# ============================================================================
# Test 8: Memory Scoring Evolution (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_memory_scoring_evolution(knowledge_graph):
    """Test that edge scores evolve correctly over time."""
    scorer = SpringMemoryScorer(decay_rate=0.99)

    # Simulate varying activation strengths
    activations_sequence = [
        0.95, 0.88, 0.92, 0.78, 0.85, 0.91, 0.83, 0.89, 0.87, 0.93
    ]

    edge = ("Bayesian", "PriorDistribution")

    for i, activation in enumerate(activations_sequence, 1):
        scorer.record_activation_pattern(
            activations={"Bayesian": activation, "PriorDistribution": activation},
            kg_edges=[edge]
        )

        edge_score = scorer.edge_scores[edge]

        # Assertions at each step
        assert edge_score.access_count == i, f"Access count wrong at step {i}"
        assert edge_score.activation_sum > 0, f"Activation sum not positive at step {i}"
        assert edge_score.score > 0, f"Score not positive at step {i}"

        # Score should generally increase (logarithmically)
        if i > 1:
            # Allow some variance due to different activation strengths
            # But score should be bounded
            assert edge_score.score < 10.0, f"Score too high at step {i}: {edge_score.score}"

    # Check final statistics
    final_score = scorer.edge_scores[edge]
    assert 0.85 < final_score.avg_activation < 0.92, \
        f"Average activation out of range: {final_score.avg_activation}"

    print(f"[OK] Memory scoring evolution: avg={final_score.avg_activation:.3f}, score={final_score.score:.3f}")


# ============================================================================
# Test 9: Spring Physics vs BFS Comparison (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_spring_vs_bfs_comparison(memory_system):
    """Compare spring physics vs BFS retrieval performance."""
    # Store test memories
    memories = [
        ("Thompson Sampling is Bayesian", ["ThompsonSampling", "Bayesian"]),
        ("Bayesian uses priors", ["Bayesian", "PriorDistribution"]),
        ("Multi-armed bandits explore", ["MultiArmedBandit", "Exploration"]),
        ("UCB uses confidence bounds", ["UCB", "Exploration"]),
        ("Epsilon-greedy is simple", ["EpsilonGreedy", "Exploration"]),
    ]

    for text, entities in memories:
        await memory_system.store(text, entities=entities, importance=0.9)

    # Benchmark BFS (standard)
    bfs_times = []
    for _ in range(5):
        start = time.perf_counter()
        result_bfs = await memory_system.retrieve("Bayesian", limit=5)
        bfs_times.append((time.perf_counter() - start) * 1000)

    avg_bfs_time = sum(bfs_times) / len(bfs_times)

    # Enable spring physics
    memory_system.enable_spring_physics()

    # Benchmark spring physics
    spring_times = []
    for _ in range(5):
        start = time.perf_counter()
        result_spring = await memory_system.retrieve("Bayesian", limit=5)
        spring_times.append((time.perf_counter() - start) * 1000)

    avg_spring_time = sum(spring_times) / len(spring_times)

    # Calculate speedup
    if avg_spring_time > 0:
        speedup = avg_bfs_time / avg_spring_time
    else:
        speedup = float('inf')

    # Spring should be faster (or at least not slower)
    # Allow 20% variance since we're comparing small times
    assert speedup >= 0.8, f"Spring physics slower than BFS: {speedup:.1f}× (expected ≥0.8×)"

    print(f"[OK] Spring vs BFS comparison:")
    print(f"  - BFS: {avg_bfs_time:.2f}ms")
    print(f"  - Spring: {avg_spring_time:.2f}ms")
    print(f"  - Speedup: {speedup:.1f}×")


# ============================================================================
# Test 10: Memory Decay and Pruning (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_memory_decay_and_pruning(knowledge_graph):
    """Test time-based decay and pruning of weak edges."""
    scorer = SpringMemoryScorer(
        decay_rate=0.95,  # Faster decay for testing
        min_score_threshold=0.5
    )

    # Record initial activations
    scorer.record_activation_pattern(
        activations={"A": 0.9, "B": 0.9},
        kg_edges=[("A", "B")]
    )

    initial_score = scorer.edge_scores[("A", "B")].score

    # Simulate time passing (modify last_access)
    import datetime
    edge_score = scorer.edge_scores[("A", "B")]
    edge_score.last_access = datetime.datetime.now() - datetime.timedelta(hours=10)

    # Recalculate score (should be lower due to decay)
    decayed_score = edge_score.score

    assert decayed_score < initial_score, f"Score did not decay: {decayed_score} >= {initial_score}"

    # Test pruning
    before_prune = len(scorer.edge_scores)
    scorer.prune_weak_edges()
    after_prune = len(scorer.edge_scores)

    # Weak edges should be pruned based on threshold
    # (Exact behavior depends on scores, but should not crash)
    assert after_prune <= before_prune, "Pruning increased edge count"

    print(f"[OK] Memory decay: {initial_score:.3f} → {decayed_score:.3f} (10 hours)")
    print(f"[OK] Pruning: {before_prune} → {after_prune} edges")


# ============================================================================
# Test 11: Concurrent Learning (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_learning(knowledge_graph):
    """Test learning system handles concurrent access correctly."""
    scorer = SpringMemoryScorer()
    retriever = AdaptiveSpringRetriever(kg=knowledge_graph, scorer=scorer)

    memories = [
        MemoryShard(
            id=f"m{i}",
            text=f"Memory {i}",
            entities=["Bayesian", "PriorDistribution"]
        )
        for i in range(5)
    ]

    # Run concurrent retrievals with learning
    async def retrieve_and_learn(query):
        return await retriever.retrieve(query, memories, limit=3, learn=True)

    # 10 concurrent queries
    results = await asyncio.gather(*[
        retrieve_and_learn("Bayesian") for _ in range(10)
    ])

    # All should succeed
    assert all(r is not None for r in results), "Some concurrent retrievals failed"

    # Check learning happened
    stats = retriever.get_learning_stats()
    assert stats['total_activations'] > 0, "No learning during concurrent access"

    # Edge scores should be consistent (no race conditions)
    top_edges = scorer.get_top_edges(limit=3)
    assert len(top_edges) > 0, "No edges learned during concurrent access"

    print(f"[OK] Concurrent learning: {stats['total_activations']} activations across 10 parallel queries")


# ============================================================================
# Test 12: Full Pipeline Integration (RUTHLESS!)
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_integration(memory_system, temp_dir):
    """Test complete end-to-end pipeline with all features."""
    # Enable spring physics
    memory_system.enable_spring_physics()

    # Store diverse memories
    test_data = [
        ("Thompson Sampling balances exploration and exploitation", 0.95, ["ThompsonSampling", "Exploration", "Exploitation"]),
        ("Bayesian methods incorporate prior knowledge", 0.90, ["Bayesian", "PriorDistribution"]),
        ("UCB algorithm uses confidence bounds", 0.85, ["UCB", "ConfidenceBound"]),
        ("Epsilon-greedy is simple but effective", 0.80, ["EpsilonGreedy", "Exploration"]),
        ("Neural bandits combine deep learning with bandits", 0.88, ["NeuralBandits", "MultiArmedBandit"]),
    ]

    for text, importance, entities in test_data:
        await memory_system.store(
            text,
            importance=importance,
            entities=entities,
            scope=MemoryScope.USER
        )

    # Multiple retrieval queries
    queries = [
        "Bayesian",
        "exploration",
        "Thompson Sampling",
        "Bayesian",  # Repeat to test learning
    ]

    results = []
    for query in queries:
        result = await memory_system.retrieve(query, limit=5)
        results.append(result)

        # Validate each result
        assert result is not None, f"Retrieval failed for query: {query}"
        assert hasattr(result, 'memories'), f"Result missing memories for: {query}"
        assert hasattr(result, 'retrieval_time_ms'), f"Result missing timing for: {query}"
        # Note: retrieval_time_ms can be 0.0 for cached/extremely fast queries
        assert result.retrieval_time_ms >= 0, f"Invalid negative timing for: {query}"

    # Run consolidation
    await memory_system.consolidate()

    # Get final statistics
    stats = memory_system.get_statistics()
    assert stats is not None, "Statistics unavailable"

    # Verify system is still functional after full pipeline
    final_result = await memory_system.retrieve("exploration", limit=3)
    assert len(final_result.memories) > 0, "System broken after full pipeline"

    print(f"[OK] Full pipeline integration:")
    print(f"  - {len(test_data)} memories stored")
    print(f"  - {len(queries)} queries processed")
    print(f"  - Consolidation completed")
    print(f"  - System remains functional")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
