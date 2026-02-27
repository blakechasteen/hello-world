"""
Complete System Test - HoloLoom Integration Tests
=================================================
Tests all major components and their interactions.

Test Levels:
1. Smoke tests - Do modules load?
2. Component tests - Do they work individually?
3. Integration tests - Do they work together?
4. Performance tests - Are they fast enough?
"""

import sys
import os
# Add repository root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import asyncio
import time
import numpy as np
from typing import List, Dict


# ============================================================================
# Level 1: Smoke Tests
# ============================================================================

def test_smoke_imports():
    """Test that all major modules can be imported."""
    print("\n" + "="*80)
    print("LEVEL 1: SMOKE TESTS - Module Imports")
    print("="*80)

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        print("[OK] WeavingOrchestrator")
    except Exception as e:
        print(f"[FAIL] WeavingOrchestrator: {e}")
        return False

    try:
        from HoloLoom.convergence.mcts_engine import MCTSConvergenceEngine
        print("[OK] MCTSConvergenceEngine")
    except Exception as e:
        print(f"[FAIL] MCTSConvergenceEngine: {e}")
        return False

    try:
        from HoloLoom.embedding.matryoshka_gate import MatryoshkaGate
        print("[OK] MatryoshkaGate")
    except Exception as e:
        print(f"[FAIL] MatryoshkaGate: {e}")
        return False

    try:
        from HoloLoom.synthesis_bridge import SynthesisBridge
        print("[OK] SynthesisBridge")
    except Exception as e:
        print(f"[FAIL] SynthesisBridge: {e}")
        return False

    try:
        from HoloLoom.unified_api import HoloLoom
        print("[OK] HoloLoom (Unified API)")
    except Exception as e:
        print(f"[FAIL] HoloLoom: {e}")
        return False

    try:
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
        print("[OK] MatryoshkaEmbeddings")
    except Exception as e:
        print(f"[FAIL] MatryoshkaEmbeddings: {e}")
        return False

    print("\n[PASS] All imports successful!")
    return True


# ============================================================================
# Level 2: Component Tests
# ============================================================================

def test_mcts_engine():
    """Test MCTS Flux Capacitor standalone."""
    print("\n" + "="*80)
    print("LEVEL 2A: MCTS Flux Capacitor")
    print("="*80)

    from HoloLoom.convergence.mcts_engine import MCTSConvergenceEngine

    tools = ["search", "summarize", "extract"]
    engine = MCTSConvergenceEngine(tools=tools, n_simulations=10)

    # Make decision
    result = engine.collapse()

    print(f"Tool selected: {result.tool}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Visit counts: {result.visit_counts}")
    print(f"UCB1 scores: {[f'{s:.3f}' for s in result.ucb1_scores]}")

    # Validate
    assert result.tool in tools, f"Invalid tool: {result.tool}"
    assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
    assert len(result.visit_counts) == len(tools), "Wrong visit count length"
    assert sum(result.visit_counts) == 10, f"Visit counts don't sum to simulations: {sum(result.visit_counts)}"

    print("\n[PASS] MCTS Flux Capacitor working!")
    return True


def test_matryoshka_gating():
    """Test matryoshka gating standalone."""
    print("\n" + "="*80)
    print("LEVEL 2B: Matryoshka Gating")
    print("="*80)

    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    from HoloLoom.embedding.matryoshka_gate import MatryoshkaGate, GateConfig

    # Create embedder
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Create gate
    config = GateConfig(
        scales=[96, 192, 384],
        thresholds=[0.6, 0.75, 0.85]
    )
    gate = MatryoshkaGate(embedder, config)

    # Test data
    query = "machine learning algorithms"
    candidates = [
        "supervised learning with neural networks",
        "the weather is sunny today",
        "deep learning architectures",
        "cooking pasta recipes",
        "reinforcement learning agents"
    ]

    # Run gating
    final_indices, gate_results = gate.gate(query, candidates, final_k=3)

    print(f"Input candidates: {len(candidates)}")
    print(f"Output candidates: {len(final_indices)}")
    print(f"Gating stages: {len(gate_results)}")

    for i, result in enumerate(gate_results):
        print(f"  Stage {i+1} ({result.scale}d): {result.candidates_in} -> {result.candidates_out}")

    # Validate
    assert len(final_indices) <= 3, f"Too many results: {len(final_indices)}"
    assert len(gate_results) == 3, f"Wrong number of stages: {len(gate_results)}"
    assert gate_results[0].candidates_in == 5, "First stage should start with all 5"

    print("\n[PASS] Matryoshka gating working!")
    return True


async def test_synthesis_bridge():
    """Test synthesis bridge standalone."""
    print("\n" + "="*80)
    print("LEVEL 2C: Synthesis Bridge")
    print("="*80)

    from HoloLoom.synthesis_bridge import SynthesisBridge
    from HoloLoom.loom.command import PatternCard, PatternSpec

    # Create bridge
    bridge = SynthesisBridge(enable_enrichment=True)

    # Create pattern spec (simplified)
    pattern_spec = PatternSpec(
        name="fast",
        card=PatternCard.FAST,
        scales=[96, 192],
        fusion_weights={96: 0.3, 192: 0.7},
        quality_target=0.7,
        speed_priority=0.8
    )

    # Test query
    query_text = "What is Thompson Sampling and how does it work?"
    dot_plasma = {"motifs": ["Thompson", "Sampling"]}
    context_shards = []

    # Synthesize
    result = await bridge.synthesize(
        query_text=query_text,
        dot_plasma=dot_plasma,
        context_shards=context_shards,
        pattern_spec=pattern_spec
    )

    print(f"Entities found: {len(result.key_entities)}")
    print(f"Entities: {result.key_entities}")
    print(f"Topics: {result.topics}")
    print(f"Reasoning type: {result.reasoning_type}")
    print(f"Confidence: {result.confidence:.2f}")

    # Validate
    assert len(result.key_entities) > 0, "Should extract entities"
    assert result.reasoning_type in ["question", "explanation", "analysis", "other"], f"Invalid reasoning type: {result.reasoning_type}"

    print("\n[PASS] Synthesis bridge working!")
    return True


# ============================================================================
# Level 3: Integration Tests
# ============================================================================

async def test_weaving_cycle():
    """Test complete weaving orchestrator."""
    print("\n" + "="*80)
    print("LEVEL 3A: Complete Weaving Cycle")
    print("="*80)

    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config

    # Create orchestrator with MCTS
    weaver = WeavingOrchestrator(
        config=Config.fast(),
        default_pattern="fast",
        use_mcts=True,
        mcts_simulations=20  # Small for testing
    )

    # Test query
    query = "Explain the weaving metaphor in HoloLoom"

    # Execute weaving cycle
    spacetime = await weaver.weave(query)

    print(f"Query: {query}")
    print(f"Tool selected: {spacetime.tool_used}")
    print(f"Confidence: {spacetime.confidence:.1%}")
    print(f"Duration: {spacetime.trace.duration_ms:.0f}ms")
    print(f"Pattern: {spacetime.trace.pattern_card}")
    print(f"Motifs: {len(spacetime.trace.motifs_detected)}")

    if hasattr(spacetime.trace, 'synthesis_result'):
        synth = spacetime.trace.synthesis_result
        print(f"Entities: {synth.get('entities', [])}")
        print(f"Reasoning: {synth.get('reasoning_type', 'unknown')}")

    # Validate
    assert spacetime.tool_used is not None, "No tool selected"
    assert 0 <= spacetime.confidence <= 1, f"Invalid confidence: {spacetime.confidence}"
    assert spacetime.trace.duration_ms > 0, "Zero duration"
    assert spacetime.trace.pattern_card in ["bare", "fast", "fused"], f"Invalid pattern: {spacetime.trace.pattern_card}"

    # Get statistics
    stats = weaver.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total weavings: {stats['total_weavings']}")
    print(f"  Pattern usage: {stats['pattern_usage']}")

    if 'mcts_stats' in stats:
        print(f"  MCTS simulations: {stats['mcts_stats']['flux_stats']['total_simulations']}")

    weaver.stop()

    print("\n[PASS] Weaving cycle working!")
    return True


async def test_unified_api():
    """Test unified API."""
    print("\n" + "="*80)
    print("LEVEL 3B: Unified API")
    print("="*80)

    from HoloLoom.unified_api import HoloLoom

    # Create HoloLoom
    loom = await HoloLoom.create(
        pattern="fast",
        memory_backend="simple",
        enable_synthesis=True
    )

    # Test query
    result = await loom.query("What is MCTS?")

    print(f"Query result: {result.response[:100]}...")
    print(f"Confidence: {result.confidence:.1%}")

    # Validate
    assert result.response is not None, "No response"
    assert 0 <= result.confidence <= 1, "Invalid confidence"

    # Test chat
    response = await loom.chat("Tell me more")

    print(f"Chat response: {response[:100]}...")

    # Validate
    assert response is not None, "No chat response"
    assert len(loom.conversation_history) > 0, "No conversation history"

    # Get stats
    stats = loom.get_stats()
    print(f"\nStats:")
    print(f"  Queries: {stats['query_count']}")
    print(f"  Chats: {stats['chat_count']}")

    print("\n[PASS] Unified API working!")
    return True


# ============================================================================
# Level 4: Performance Tests
# ============================================================================

async def test_performance():
    """Test performance of weaving cycle."""
    print("\n" + "="*80)
    print("LEVEL 4: Performance Tests")
    print("="*80)

    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config

    # Create orchestrator
    weaver = WeavingOrchestrator(
        config=Config.fast(),
        use_mcts=True,
        mcts_simulations=20
    )

    # Test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does backpropagation work?",
        "What is gradient descent?",
        "Describe convolutional layers"
    ]

    # Benchmark
    start = time.time()
    durations = []

    for query in queries:
        query_start = time.time()
        spacetime = await weaver.weave(query)
        query_duration = (time.time() - query_start) * 1000  # ms
        durations.append(query_duration)

    total_duration = time.time() - start

    print(f"Queries: {len(queries)}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"Average: {np.mean(durations):.1f}ms per query")
    print(f"Median: {np.median(durations):.1f}ms")
    print(f"Min: {np.min(durations):.1f}ms")
    print(f"Max: {np.max(durations):.1f}ms")

    # Validate
    avg_duration = np.mean(durations)
    assert avg_duration < 100, f"Too slow: {avg_duration:.1f}ms avg (should be <100ms)"

    weaver.stop()

    print("\n[PASS] Performance acceptable!")
    return True


# ============================================================================
# Test Runner
# ============================================================================

async def run_all_tests():
    """Run all test levels."""
    print("="*80)
    print("HOLOLOOM COMPLETE SYSTEM TEST SUITE")
    print("="*80)
    print("Testing: MCTS + Matryoshka + Weaving + Synthesis + Unified API")
    print()

    results = {}

    # Level 1: Smoke tests
    try:
        results["smoke_imports"] = test_smoke_imports()
    except Exception as e:
        print(f"[FAIL] Smoke test failed: {e}")
        results["smoke_imports"] = False

    if not results["smoke_imports"]:
        print("\n[FAIL] Smoke tests failed! Cannot continue.")
        return results

    # Level 2: Component tests
    try:
        results["mcts"] = test_mcts_engine()
    except Exception as e:
        print(f"[FAIL] MCTS test failed: {e}")
        results["mcts"] = False

    try:
        results["gating"] = test_matryoshka_gating()
    except Exception as e:
        print(f"[FAIL] Gating test failed: {e}")
        results["gating"] = False

    try:
        results["synthesis"] = await test_synthesis_bridge()
    except Exception as e:
        print(f"[FAIL] Synthesis test failed: {e}")
        results["synthesis"] = False

    # Level 3: Integration tests
    try:
        results["weaving"] = await test_weaving_cycle()
    except Exception as e:
        print(f"[FAIL] Weaving test failed: {e}")
        import traceback
        traceback.print_exc()
        results["weaving"] = False

    try:
        results["unified_api"] = await test_unified_api()
    except Exception as e:
        print(f"[FAIL] Unified API test failed: {e}")
        import traceback
        traceback.print_exc()
        results["unified_api"] = False

    # Level 4: Performance tests
    try:
        results["performance"] = await test_performance()
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        results["performance"] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS] PASS" if passed_test else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n[WARN]  {total - passed} test(s) failed")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')

    # Run tests
    results = asyncio.run(run_all_tests())

    # Exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
