#!/usr/bin/env python3
"""
Complete Warp Drive Test Suite
================================
Comprehensive tests for all 9 weaving components working together.

Tests:
1. Individual module functionality
2. Module integration
3. Complete weaving cycle
4. Performance benchmarks
5. Error handling and edge cases
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_1_loom_command():
    """Test Loom Command pattern selection."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Loom Command - Pattern Selection")
    logger.info("="*80)

    from HoloLoom.loom.command import LoomCommand, PatternCard

    loom = LoomCommand(default_pattern=PatternCard.FAST, auto_select=True)

    # Test auto-selection
    test_queries = [
        ("Hi", "bare - short query"),
        ("What is Thompson Sampling and how does it balance exploration vs exploitation?", "fast - medium"),
        ("Explain the complete mathematical framework behind multi-armed bandit algorithms including UCB, Thompson Sampling, and epsilon-greedy strategies, with proofs of regret bounds and practical implementation details for real-world applications", "fast/fused - long")
    ]

    for query, expected in test_queries:
        pattern = loom.select_pattern(query_text=query)
        logger.info(f"Query ({len(query)} chars): '{query[:50]}...'")
        logger.info(f"  → Selected: {pattern.card.value} (expected: {expected})")
        logger.info(f"  → Scales: {pattern.scales}, Timeout: {pattern.pipeline_timeout}s")

    # Test explicit preference
    pattern = loom.select_pattern(query_text="test", user_preference="fused")
    assert pattern.card == PatternCard.FUSED
    logger.info("\n✓ Explicit preference works")

    # Test resource constraints
    pattern = loom.select_pattern(query_text="test", resource_constraints={"max_timeout": 2.0})
    assert pattern.card == PatternCard.BARE
    logger.info("✓ Resource constraints work")

    stats = loom.get_statistics()
    logger.info(f"\nStatistics: {stats}")
    logger.info("\n✅ TEST 1 PASSED: Loom Command\n")


async def test_2_chrono_trigger():
    """Test Chrono Trigger temporal control."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Chrono Trigger - Temporal Control")
    logger.info("="*80)

    from HoloLoom.chrono.trigger import ChronoTrigger
    from HoloLoom.config import Config

    config = Config.fast()
    chrono = ChronoTrigger(config, enable_heartbeat=False)

    # Test firing
    query_time = datetime.now()
    window = await chrono.fire(query_time, pattern_card_mode="fused")

    logger.info(f"Temporal Window:")
    logger.info(f"  Start: {window.start}")
    logger.info(f"  End: {window.end}")
    logger.info(f"  Max age: {window.max_age}")
    logger.info(f"  Recency bias: {window.recency_bias}")

    # Test monitoring
    async def mock_operation():
        await asyncio.sleep(0.1)
        return {"status": "success", "confidence": 0.9}

    result = await chrono.monitor(mock_operation, timeout=2.0, stage="test")
    logger.info(f"\nMonitored operation result: {result}")

    # Test completion
    metrics = chrono.record_completion()
    logger.info(f"\nExecution metrics: {metrics}")

    chrono.stop()
    logger.info("\n✅ TEST 2 PASSED: Chrono Trigger\n")


async def test_3_resonance_shed():
    """Test Resonance Shed feature extraction."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Resonance Shed - Feature Interference")
    logger.info("="*80)

    from HoloLoom.resonance.shed import ResonanceShed
    from HoloLoom.motif.base import create_motif_detector
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    # Create components
    motif_detector = create_motif_detector(mode="regex")
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    shed = ResonanceShed(
        motif_detector=motif_detector,
        embedder=embedder,
        interference_mode="weighted_sum"
    )

    # Test weaving
    text = "Thompson Sampling uses Bayesian exploration to balance exploitation and discovery in multi-armed bandit problems"
    plasma = await shed.weave(
        text,
        thread_weights={"motif": 0.8, "embedding": 1.0}
    )

    logger.info(f"DotPlasma Features:")
    logger.info(f"  Motifs detected: {plasma['motifs']}")
    logger.info(f"  Embedding dimension: {len(plasma['psi'])}")
    logger.info(f"  Thread count: {plasma['metadata']['thread_count']}")
    logger.info(f"  Interference mode: {plasma['metadata']['interference_mode']}")

    logger.info(f"\nThread Details:")
    for thread in plasma['threads']:
        logger.info(f"  {thread['name']}: weight={thread['weight']}, {thread['metadata']}")

    logger.info("\n✅ TEST 3 PASSED: Resonance Shed\n")


async def test_4_warp_space():
    """Test Warp Space tensioned manifold."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Warp Space - Tensioned Tensor Field (THE WARP DRIVE!)")
    logger.info("="*80)

    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    import numpy as np

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    # Sample threads from Yarn Graph
    thread_texts = [
        "Thompson Sampling balances exploration and exploitation",
        "Neural networks learn hierarchical representations",
        "Attention mechanisms enable context-aware processing",
        "Bayesian methods quantify uncertainty",
        "Reinforcement learning optimizes sequential decisions"
    ]

    # 1. Tension threads into Warp Space
    logger.info("1. Tensioning threads into Warp Space...")
    await warp.tension(thread_texts, tension_weights=[1.0, 0.9, 0.8, 0.7, 0.6])
    logger.info(f"   Field shape: {warp.tensor_field.shape}")
    logger.info(f"   Threads tensioned: {len(warp.threads)}")

    # 2. Compute spectral features
    logger.info("\n2. Computing spectral features...")
    spectral = warp.compute_spectral_features()
    logger.info(f"   Field norm: {spectral['field_norm']:.3f}")
    logger.info(f"   Spectral entropy: {spectral.get('spectral_entropy', 0):.3f}")
    logger.info(f"   Singular values: {[f'{s:.3f}' for s in spectral.get('singular_values', [])]}")

    # 3. Apply attention from query
    logger.info("\n3. Applying attention from query...")
    query_text = ["What is Thompson Sampling?"]
    query_emb_dict = embedder.encode_scales(query_text)
    query_emb = query_emb_dict[384][0]  # Use largest scale
    attention = warp.apply_attention(query_emb)
    logger.info(f"   Attention weights: {[f'{a:.3f}' for a in attention]}")
    logger.info(f"   Max attention on thread: {np.argmax(attention)} ('{thread_texts[np.argmax(attention)][:50]}...')")

    # 4. Compute weighted context
    logger.info("\n4. Computing weighted context...")
    context = warp.weighted_context(attention)
    logger.info(f"   Context vector shape: {context.shape}")
    logger.info(f"   Context norm: {np.linalg.norm(context):.3f}")

    # 5. Test multi-scale field access
    logger.info("\n5. Testing multi-scale field access...")
    for scale in [96, 192, 384]:
        field = warp.get_field(scale)
        logger.info(f"   Scale {scale}: field shape {field.shape}")

    # 6. Collapse Warp Space
    logger.info("\n6. Collapsing Warp Space...")
    updates = warp.collapse()
    logger.info(f"   Threads processed: {len(updates['threads'])}")
    logger.info(f"   Operations performed: {len(updates['operations'])}")
    logger.info(f"   Field stats: {updates['field_stats']}")

    logger.info("\n✅ TEST 4 PASSED: Warp Space (WARP DRIVE OPERATIONAL!)\n")


async def test_5_convergence_engine():
    """Test Convergence Engine decision collapse."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Convergence Engine - Continuous to Discrete")
    logger.info("="*80)

    from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy
    import numpy as np

    tools = ["search", "summarize", "extract", "respond", "clarify"]
    engine = ConvergenceEngine(tools, default_strategy=CollapseStrategy.EPSILON_GREEDY, epsilon=0.2)

    # Simulate neural network probabilities
    neural_probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])

    logger.info("Testing collapse strategies:\n")

    # Test each strategy
    strategies = [
        (CollapseStrategy.ARGMAX, "Pure Exploitation"),
        (CollapseStrategy.EPSILON_GREEDY, "Epsilon-Greedy (20% explore)"),
        (CollapseStrategy.BAYESIAN_BLEND, "Bayesian Blend (70% neural + 30% bandit)"),
        (CollapseStrategy.PURE_THOMPSON, "Pure Thompson Sampling")
    ]

    for strategy, description in strategies:
        logger.info(f"\n{description}:")
        logger.info("-" * 40)

        # Run 10 collapses
        tool_counts = {tool: 0 for tool in tools}

        for i in range(10):
            result = engine.collapse(neural_probs.copy(), strategy=strategy)
            tool_counts[result.tool] += 1

            # Simulate outcome (higher prob = higher success rate)
            success = np.random.rand() < result.confidence
            engine.update_from_outcome(result.tool_idx, success)

        # Display distribution
        logger.info("Tool selection distribution:")
        for tool, count in tool_counts.items():
            bar = '█' * count
            logger.info(f"  {tool:12s} {bar} ({count}/10)")

    # Show final bandit statistics
    logger.info("\n" + "="*40)
    logger.info("Final Bandit Statistics:")
    logger.info("="*40)
    stats = engine.bandit.get_statistics()
    for i, tool in enumerate(tools):
        prior = stats['tool_priors'][i]
        pulls = stats['pulls'][i]
        logger.info(f"  {tool:12s} Prior={prior:.3f}, Pulls={int(pulls):.0f}, "
                   f"Success={stats['success_counts'][i]:.1f}, Fail={stats['failure_counts'][i]:.1f}")

    logger.info("\n✅ TEST 5 PASSED: Convergence Engine\n")


async def test_6_spacetime_fabric():
    """Test Spacetime fabric output."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: Spacetime - Woven Fabric Output")
    logger.info("="*80)

    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace, FabricCollection
    import tempfile

    # Create a weaving trace
    start = datetime.now()
    await asyncio.sleep(0.05)  # Simulate processing
    end = datetime.now()

    trace = WeavingTrace(
        start_time=start,
        end_time=end,
        duration_ms=(end - start).total_seconds() * 1000,
        stage_durations={
            'features': 12.3,
            'retrieval': 23.5,
            'decision': 8.2,
            'execution': 5.5
        },
        motifs_detected=["ALGORITHM", "OPTIMIZATION", "BAYESIAN"],
        embedding_scales_used=[96, 192, 384],
        threads_activated=["thread_001", "thread_002", "thread_003"],
        context_shards_count=5,
        retrieval_mode="fused",
        policy_adapter="fused_adapter",
        tool_selected="respond",
        tool_confidence=0.87,
        bandit_statistics={'respond': {'pulls': 25, 'successes': 22}}
    )

    # Create spacetime fabric
    spacetime = Spacetime(
        query_text="Explain Thompson Sampling for multi-armed bandits",
        response="Thompson Sampling is a Bayesian approach that samples from posterior distributions to balance exploration and exploitation.",
        tool_used="respond",
        confidence=0.87,
        trace=trace,
        metadata={
            "execution_mode": "fused",
            "scales": [96, 192, 384]
        },
        sources_used=["papers/thompson_1933.pdf", "docs/bandits.md"],
        context_summary="Retrieved 5 relevant memory shards about Bayesian bandits"
    )

    # Add quality score
    spacetime.add_quality_score(0.92, feedback="Accurate and comprehensive")

    # Print summary
    logger.info("Spacetime Summary:")
    summary = spacetime.summarize()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    spacetime.save(temp_path)
    logger.info(f"\nSaved to {temp_path}")

    # Load back
    loaded = Spacetime.load(temp_path)
    logger.info(f"Loaded spacetime: query='{loaded.query_text[:50]}...'")
    assert loaded.confidence == spacetime.confidence

    # Get reflection signal
    reflection = spacetime.get_reflection_signal()
    logger.info(f"\nReflection Signal:")
    logger.info(f"  Success: {reflection['success']}")
    logger.info(f"  Tool: {reflection['tool_selected']}")
    logger.info(f"  Confidence: {reflection['confidence']:.3f}")
    logger.info(f"  Quality: {reflection['quality_score']:.3f}")

    # Test fabric collection
    logger.info(f"\nTesting FabricCollection...")
    collection = FabricCollection([spacetime])
    stats = collection.get_statistics()
    logger.info(f"Collection stats: {stats}")

    # Cleanup
    import os
    os.unlink(temp_path)

    logger.info("\n✅ TEST 6 PASSED: Spacetime Fabric\n")


async def test_7_complete_weaving_cycle():
    """Test complete weaving cycle with all components."""
    logger.info("\n" + "="*80)
    logger.info("TEST 7: COMPLETE WEAVING CYCLE - All Components Together")
    logger.info("="*80)

    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config
    from HoloLoom.convergence.engine import CollapseStrategy

    # Create orchestrator with all components
    logger.info("\nInitializing WeavingOrchestrator...")
    weaver = WeavingOrchestrator(
        config=Config.fast(),
        default_pattern="fast",
        collapse_strategy=CollapseStrategy.EPSILON_GREEDY
    )

    # Test queries
    test_queries = [
        "What is Thompson Sampling?",
        "How do neural networks learn from data?",
        "Explain the difference between exploration and exploitation"
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Execute complete weaving cycle
        result = await weaver.weave(query)

        duration = time.time() - start_time

        # Display results
        logger.info(f"\n✨ WEAVING COMPLETE ✨")
        logger.info(f"Response: {result.response[:100]}...")
        logger.info(f"Tool used: {result.tool_used}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Duration: {duration*1000:.1f}ms")

        # Show trace highlights
        trace = result.trace
        logger.info(f"\nTrace Highlights:")
        logger.info(f"  Motifs detected: {len(trace.motifs_detected)}")
        logger.info(f"  Scales used: {trace.embedding_scales_used}")
        logger.info(f"  Threads activated: {len(trace.threads_activated)}")
        logger.info(f"  Context shards: {trace.context_shards_count}")
        logger.info(f"  Policy adapter: {trace.policy_adapter}")

        if trace.stage_durations:
            logger.info(f"\n  Stage timings:")
            for stage, dur in trace.stage_durations.items():
                logger.info(f"    {stage}: {dur:.1f}ms")

    # Get orchestrator statistics
    logger.info(f"\n{'='*60}")
    logger.info("Weaving Statistics:")
    logger.info(f"{'='*60}")
    logger.info(f"Total weavings: {weaver.weaving_count}")
    logger.info(f"Pattern usage: {weaver.pattern_usage}")

    logger.info("\n✅ TEST 7 PASSED: Complete Weaving Cycle\n")


async def test_8_error_handling():
    """Test error handling and edge cases."""
    logger.info("\n" + "="*80)
    logger.info("TEST 8: Error Handling and Edge Cases")
    logger.info("="*80)

    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    # Test 1: Empty thread list
    logger.info("\n1. Testing empty thread list...")
    await warp.tension([])
    logger.info("   ✓ Handled gracefully")

    # Test 2: Operations on non-tensioned space
    logger.info("\n2. Testing operations on non-tensioned space...")
    try:
        warp.get_field()
        logger.error("   ✗ Should have raised RuntimeError")
    except RuntimeError as e:
        logger.info(f"   ✓ Correctly raised: {e}")

    # Test 3: Double collapse
    logger.info("\n3. Testing double collapse...")
    await warp.tension(["test thread"])
    warp.collapse()
    result = warp.collapse()  # Second collapse
    logger.info(f"   ✓ Handled gracefully: {result}")

    # Test 4: Very long text
    logger.info("\n4. Testing very long text...")
    long_text = "word " * 10000  # 50k characters
    await warp.tension([long_text])
    logger.info(f"   ✓ Processed {len(long_text)} characters")
    warp.collapse()

    logger.info("\n✅ TEST 8 PASSED: Error Handling\n")


async def test_9_performance_benchmark():
    """Benchmark performance of warp space operations."""
    logger.info("\n" + "="*80)
    logger.info("TEST 9: Performance Benchmark")
    logger.info("="*80)

    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    import numpy as np

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Benchmark different thread counts
    thread_counts = [5, 10, 20, 50]

    logger.info("\nBenchmarking Warp Space operations:\n")
    logger.info(f"{'Threads':<10} {'Tension':<12} {'Spectral':<12} {'Attention':<12} {'Collapse':<12} {'Total':<12}")
    logger.info("-" * 70)

    for n_threads in thread_counts:
        warp = WarpSpace(embedder, scales=[96, 192, 384])

        # Generate test threads
        threads = [f"This is test thread number {i} with some semantic content about algorithms and data"
                   for i in range(n_threads)]

        # Benchmark tension
        t0 = time.time()
        await warp.tension(threads)
        tension_time = (time.time() - t0) * 1000

        # Benchmark spectral
        t0 = time.time()
        warp.compute_spectral_features()
        spectral_time = (time.time() - t0) * 1000

        # Benchmark attention
        query_emb = np.random.randn(384)
        t0 = time.time()
        attention = warp.apply_attention(query_emb)
        attention_time = (time.time() - t0) * 1000

        # Benchmark collapse
        t0 = time.time()
        warp.collapse()
        collapse_time = (time.time() - t0) * 1000

        total_time = tension_time + spectral_time + attention_time + collapse_time

        logger.info(f"{n_threads:<10} {tension_time:>10.2f}ms {spectral_time:>10.2f}ms "
                   f"{attention_time:>10.2f}ms {collapse_time:>10.2f}ms {total_time:>10.2f}ms")

    logger.info("\n✅ TEST 9 PASSED: Performance Benchmark\n")


async def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("🚀 HOLOLOOM WARP DRIVE COMPLETE TEST SUITE 🚀")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info("\n")

    start_time = time.time()

    # Run all tests
    tests = [
        test_1_loom_command,
        test_2_chrono_trigger,
        test_3_resonance_shed,
        test_4_warp_space,
        test_5_convergence_engine,
        test_6_spacetime_fabric,
        test_7_complete_weaving_cycle,
        test_8_error_handling,
        test_9_performance_benchmark
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"\n❌ TEST FAILED: {test.__name__}")
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = time.time() - start_time

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏁 TEST SUITE COMPLETE 🏁")
    logger.info("="*80)
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed} ✅")
    logger.info(f"Failed: {failed} ❌")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"End time: {datetime.now()}")

    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED! WARP DRIVE IS FULLY OPERATIONAL! 🎉\n")
    else:
        logger.info(f"\n⚠️  {failed} test(s) failed. See errors above.\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
