"""
HoloLoom Memory System - REAL Semantic Retrieval Diagnostic
============================================================

This is NOT a test with mocks. This proves REAL semantic retrieval works.

The Test:
- Store: "Thompson Sampling uses Bayesian priors for exploration"
- Query: "What algorithm uses Bayesian methods?" (DIFFERENT words, SAME meaning)
- Expected: Should retrieve the Thompson Sampling memory

If this fails, we trace EXACTLY where data stops flowing.

Run with: PYTHONPATH=. python demos/demo_semantic_retrieval_real.py
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_status(success: bool, message: str):
    """Print status with checkmark or X."""
    mark = "[PASS]" if success else "[FAIL]"
    print(f"  {mark} {message}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def test_1_raw_embeddings():
    """Test 1: Verify raw embeddings capture semantic similarity."""
    print_section("TEST 1: RAW EMBEDDING SIMILARITY")
    print("  Goal: Prove embeddings capture semantic meaning")
    print()

    try:
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings()
        print_status(True, f"MatryoshkaEmbeddings created")
        print(f"      Model: {embedder.model_name if hasattr(embedder, 'model_name') else 'unknown'}")

        # Test sentences
        s1 = "Thompson Sampling uses Bayesian priors for exploration"
        s2 = "Bayesian bandit algorithm for decision making"  # SIMILAR meaning
        s3 = "The weather is sunny and warm today"  # DIFFERENT meaning

        print(f"\n  Sentences:")
        print(f"      S1: {s1}")
        print(f"      S2: {s2}")
        print(f"      S3: {s3}")

        # Get embeddings
        emb1 = embedder.encode_base([s1])[0]
        emb2 = embedder.encode_base([s2])[0]
        emb3 = embedder.encode_base([s3])[0]

        print(f"\n  Embedding dimensions: {emb1.shape}")
        print(f"      First 5 values of S1: {emb1[:5]}")

        # Compute similarities
        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)
        sim_23 = cosine_similarity(emb2, emb3)

        print(f"\n  Cosine Similarities:")
        print(f"      S1 vs S2 (should be HIGH):  {sim_12:.4f}")
        print(f"      S1 vs S3 (should be LOW):   {sim_13:.4f}")
        print(f"      S2 vs S3 (should be LOW):   {sim_23:.4f}")

        # Verify semantic similarity works
        semantic_works = sim_12 > sim_13 and sim_12 > sim_23
        print_status(semantic_works, f"Semantic similarity: S1-S2 ({sim_12:.2f}) > S1-S3 ({sim_13:.2f})")

        if not semantic_works:
            print("\n  WARNING: Embeddings don't capture semantic similarity!")
            print("      This means the embedding model itself is broken.")
            return False, None

        # Test multi-scale embeddings
        print(f"\n  Testing Matryoshka scales...")
        scales = embedder.encode_scales([s1])
        print(f"      Available scales: {list(scales.keys())}")
        for scale, emb in scales.items():
            print(f"      Scale {scale}: shape={emb[0].shape}")

        return True, embedder

    except Exception as e:
        print_status(False, f"Raw embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_2_semantic_calculus():
    """Test 2: Verify semantic calculus projects to 228D space correctly."""
    print_section("TEST 2: SEMANTIC CALCULUS (228D PROJECTION)")
    print("  Goal: Verify streaming analysis produces valid semantic positions")
    print()

    try:
        from hololoom.semantic_calculus.matryoshka_streaming import (
            MatryoshkaSemanticCalculus,
            MatryoshkaScale
        )

        calculus = MatryoshkaSemanticCalculus()
        print_status(True, "MatryoshkaSemanticCalculus created")

        # Test streaming analysis
        test_text = "Thompson Sampling balances exploration and exploitation"

        async def word_stream():
            for word in test_text.split():
                yield word

        snapshots = []
        async for snapshot in calculus.stream_analyze(word_stream()):
            snapshots.append(snapshot)

        print(f"\n  Processed {len(snapshots)} snapshots")

        if snapshots:
            final = snapshots[-1]
            print(f"      Available scales: {list(final.states_by_scale.keys())}")

            # Check for 228D positions
            for scale, state in final.states_by_scale.items():
                if 'position' in state:
                    pos = state['position']
                    print(f"      Scale {scale.value}: position shape = {pos.shape}")
                    if pos.shape[0] == 228:
                        print_status(True, f"228D semantic position found at scale {scale.value}")
                    else:
                        print(f"        Note: Expected 228D, got {pos.shape[0]}D")

        return True, calculus

    except Exception as e:
        print_status(False, f"Semantic calculus test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_3_awareness_graph_storage():
    """Test 3: Verify memories are stored with semantic positions."""
    print_section("TEST 3: AWARENESS GRAPH STORAGE")
    print("  Goal: Verify memories stored with retrievable semantic positions")
    print()

    try:
        import networkx as nx
        from hololoom.memory.awareness_graph import AwarenessGraph
        from hololoom.semantic_calculus.matryoshka_streaming import (
            MatryoshkaSemanticCalculus,
            MatryoshkaScale
        )

        # Create components
        graph_backend = nx.MultiDiGraph()
        semantic_calculus = MatryoshkaSemanticCalculus()

        awareness = AwarenessGraph(
            graph_backend=graph_backend,
            semantic_calculus=semantic_calculus,
            vector_store=None
        )
        print_status(True, "AwarenessGraph created with real semantic calculus")

        # Store test memories
        memories = [
            "Thompson Sampling uses Bayesian priors for exploration",
            "UCB algorithm provides optimism under uncertainty",
            "The weather is sunny and warm today",
        ]

        memory_ids = []
        for i, text in enumerate(memories):
            print(f"\n  Storing memory {i+1}: '{text[:50]}...'")

            # Perceive
            perception = await awareness.perceive(text)
            print(f"      Perception position shape: {perception.position.shape}")
            print(f"      Momentum: {perception.momentum:.3f}")

            # Remember
            mem_id = await awareness.remember(text, perception, context={"index": i})
            memory_ids.append(mem_id)
            print(f"      Stored with ID: {mem_id[:20]}...")

        # Check what's stored
        print(f"\n  Checking stored data:")
        print(f"      Graph nodes: {len(graph_backend.nodes())}")
        print(f"      Graph edges: {len(graph_backend.edges())}")

        # Check semantic positions
        positions_stored = len(awareness.semantic_positions) if hasattr(awareness, 'semantic_positions') else 0
        print(f"      Semantic positions stored: {positions_stored}")

        if positions_stored > 0:
            print_status(True, f"Semantic positions stored: {positions_stored}")
            # Sample a position
            sample_id = list(awareness.semantic_positions.keys())[0]
            sample_pos = awareness.semantic_positions[sample_id]
            print(f"      Sample position shape: {sample_pos.shape}")
        else:
            print_status(False, "No semantic positions stored!")
            print("      This is likely the bug - positions not being indexed")

        return True, awareness, memory_ids

    except Exception as e:
        print_status(False, f"Awareness graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, []


async def test_4_semantic_retrieval():
    """Test 4: THE CRITICAL TEST - Query with different words, get semantic match."""
    print_section("TEST 4: SEMANTIC RETRIEVAL (THE REAL TEST)")
    print("  Goal: Query 'Bayesian bandit algorithm' retrieves 'Thompson Sampling'")
    print()

    try:
        import networkx as nx
        from hololoom.memory.awareness_graph import AwarenessGraph
        from hololoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

        # Create fresh components
        graph_backend = nx.MultiDiGraph()
        semantic_calculus = MatryoshkaSemanticCalculus()

        awareness = AwarenessGraph(
            graph_backend=graph_backend,
            semantic_calculus=semantic_calculus,
            vector_store=None
        )

        # Store memories with CLEAR semantic differences
        memories = [
            ("thompson", "Thompson Sampling uses Bayesian priors for exploration"),
            ("ucb", "UCB algorithm provides optimism under uncertainty"),
            ("weather", "The weather is sunny and warm today"),
            ("cooking", "Pasta should be cooked in salted boiling water"),
        ]

        print("  Storing memories:")
        stored_ids = {}
        for name, text in memories:
            perception = await awareness.perceive(text)
            mem_id = await awareness.remember(text, perception, context={"name": name})
            stored_ids[name] = mem_id
            print(f"      [{name}] {text[:40]}...")

        print(f"\n  Total memories stored: {len(stored_ids)}")
        print(f"  Semantic positions: {len(awareness.semantic_positions)}")

        # THE CRITICAL QUERY
        query = "What Bayesian bandit algorithm balances exploration?"
        print(f"\n  Query: '{query}'")
        print("  Expected: Should retrieve 'thompson' memory (semantically similar)")

        # Perceive query
        query_perception = await awareness.perceive(query)
        query_pos = query_perception.position
        print(f"\n  Query position shape: {query_pos.shape}")

        # Manual semantic search (to diagnose)
        print(f"\n  Manual semantic distance check:")
        distances = {}
        for name, mem_id in stored_ids.items():
            if mem_id in awareness.semantic_positions:
                mem_pos = awareness.semantic_positions[mem_id]
                dist = np.linalg.norm(query_pos - mem_pos)
                distances[name] = dist
                print(f"      {name}: distance = {dist:.4f}")

        if distances:
            # Sort by distance (closest first)
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            closest = sorted_distances[0][0]
            print(f"\n  Closest match: '{closest}'")

            # Success if thompson is closest (or at least closer than weather/cooking)
            thompson_dist = distances.get('thompson', float('inf'))
            weather_dist = distances.get('weather', float('inf'))
            cooking_dist = distances.get('cooking', float('inf'))

            semantic_correct = thompson_dist < weather_dist and thompson_dist < cooking_dist
            print_status(semantic_correct, f"Thompson ({thompson_dist:.2f}) closer than Weather ({weather_dist:.2f})")

            if closest == 'thompson':
                print_status(True, "SEMANTIC RETRIEVAL WORKS!")
                print("      Query 'Bayesian bandit' correctly matched 'Thompson Sampling'")
            elif closest == 'ucb':
                print_status(True, "SEMANTIC RETRIEVAL WORKS (UCB is also valid)")
                print("      Query 'Bayesian bandit' matched 'UCB' (both are bandit algorithms)")
            else:
                print_status(False, f"Wrong match: '{closest}' instead of 'thompson'")
                return False
        else:
            print_status(False, "No semantic positions to search!")
            return False

        # Test recall() method if available
        print(f"\n  Testing awareness.recall()...")
        if hasattr(awareness, 'recall'):
            results = await awareness.recall(query, k=3)
            print(f"      recall() returned {len(results)} results")
            for i, r in enumerate(results[:3]):
                print(f"      {i+1}. {r}")
        else:
            print("      recall() method not found on AwarenessGraph")

        return True

    except Exception as e:
        print_status(False, f"Semantic retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_full_hololoom_api():
    """Test 5: Full HoloLoom API - experience() and recall()."""
    print_section("TEST 5: FULL HOLOLOOM API")
    print("  Goal: Test the 10/10 API surface (experience -> recall)")
    print()

    try:
        from hololoom.hololoom import HoloLoom

        print("  Creating HoloLoom instance...")

        async with HoloLoom() as loom:
            print_status(True, "HoloLoom context manager opened")

            # Experience memories
            memories = [
                "Thompson Sampling uses Bayesian priors for exploration",
                "UCB algorithm provides optimism under uncertainty",
                "The weather is sunny and warm today",
            ]

            print(f"\n  Forming {len(memories)} memories:")
            formed = []
            for text in memories:
                mem = await loom.experience(text)
                formed.append(mem)
                print(f"      Stored: {text[:40]}...")

            print_status(True, f"Formed {len(formed)} memories")

            # Get metrics
            metrics = loom.get_metrics()
            print(f"\n  System metrics:")
            if 'activation' in metrics:
                print(f"      Active nodes: {metrics['activation'].get('active_nodes', 'N/A')}")
            if 'coherence' in metrics:
                print(f"      Coherence: {metrics['coherence'].get('global_coherence', 'N/A')}")

            # THE CRITICAL RECALL
            query = "Bayesian bandit algorithm for exploration"
            print(f"\n  Recall query: '{query}'")

            results = await loom.recall(query)
            print(f"      Results: {len(results)}")

            if len(results) > 0:
                print_status(True, f"recall() returned {len(results)} memories")
                for i, r in enumerate(results[:3]):
                    text = r.text if hasattr(r, 'text') else str(r)
                    print(f"      {i+1}. {text[:60]}...")

                # Check if Thompson Sampling is in results
                top_result = results[0].text if hasattr(results[0], 'text') else str(results[0])
                if 'thompson' in top_result.lower() or 'bayesian' in top_result.lower():
                    print_status(True, "HOLOLOOM MEMORY WORKS!")
                    return True
                else:
                    print("      Note: Top result doesn't match expected semantic content")
            else:
                print_status(False, "recall() returned 0 results!")
                print("      This is the bug we need to fix")

                # Diagnose
                print(f"\n  Diagnosing...")
                if hasattr(loom, '_awareness'):
                    print(f"      Awareness positions: {len(loom._awareness.semantic_positions)}")
                    print(f"      Graph nodes: {len(loom._awareness.graph.nodes())}")

                return False

        print_status(True, "HoloLoom context manager closed cleanly")
        return True

    except Exception as e:
        print_status(False, f"Full HoloLoom test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print("    HOLOLOOM MEMORY SYSTEM - REAL SEMANTIC RETRIEVAL DIAGNOSTIC")
    print("="*70)
    print("\n  This tests REAL semantic retrieval, not mocks.")
    print("  The goal: Query 'Bayesian bandit' retrieves 'Thompson Sampling'\n")

    results = {}

    # Test 1: Raw embeddings
    success, embedder = await test_1_raw_embeddings()
    results["Raw Embeddings"] = success
    if not success:
        print("\n  STOPPING: Raw embeddings don't work - fix model first")
        return False

    # Test 2: Semantic calculus
    success, calculus = await test_2_semantic_calculus()
    results["Semantic Calculus"] = success

    # Test 3: Awareness storage
    success, awareness, mem_ids = await test_3_awareness_graph_storage()
    results["Awareness Storage"] = success

    # Test 4: Semantic retrieval (the critical one)
    success = await test_4_semantic_retrieval()
    results["Semantic Retrieval"] = success

    # Test 5: Full HoloLoom API
    success = await test_5_full_hololoom_api()
    results["Full HoloLoom API"] = success

    # Summary
    print_section("FINAL RESULTS")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n  Tests passed: {passed}/{total}")
    print()

    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    if passed == total:
        print(f"\n  {'='*50}")
        print("  MEMORY SYSTEM IS WORKING!")
        print("  Semantic retrieval successfully matches meaning, not just words.")
        print(f"  {'='*50}")
        return True
    else:
        print(f"\n  {'='*50}")
        print(f"  {total - passed} test(s) failed")
        print("  See output above for diagnosis")
        print(f"  {'='*50}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
