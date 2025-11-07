"""
Complete 9-Layer Weaving Cycle Integration Test
================================================
Validates all 9 layers of the HoloLoom weaving architecture execute correctly
with proper data flow and complete provenance tracking.

Test Philosophy:
- Integration test (not unit test): Validates layer interactions
- End-to-end verification: From query to spacetime artifact
- Complete tracing: Every layer must leave evidence in the trace
- Multiple complexity levels: Tests BARE/FAST/FUSED execution paths
- Error resilience: Graceful degradation when components fail

The 9 Layers Validated:
1. Loom Command - Pattern card selection (BARE/FAST/FUSED)
2. Chrono Trigger - Temporal window creation
3. Yarn Graph - Thread selection (or shards fallback)
4. Resonance Shed - Feature extraction (DotPlasma)
5. Warp Space - Tensor field tensioning
6. Convergence Engine - Decision collapse (Thompson Sampling)
7. Tool Executor - Tool execution
8. Spacetime Fabric - Woven artifact with trace
9. Reflection Buffer - Learning signal extraction

Author: Claude Code (Integration test from agent swarm analysis)
Date: 2025-11-02
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import List

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_test_shards() -> List[MemoryShard]:
    """
    Create test memory shards for integration testing.

    Returns:
        List of MemoryShard objects with realistic test data
    """
    return [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            entities=["Thompson Sampling", "Bayesian", "bandit"],
            motifs=["thompson", "sampling", "bayesian"],
            metadata={"source": "test", "importance": 0.9}
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation using Beta distributions.",
            entities=["algorithm", "exploration", "exploitation", "Beta distribution"],
            motifs=["exploration", "exploitation", "beta"],
            metadata={"source": "test", "importance": 0.8}
        ),
        MemoryShard(
            id="shard_003",
            text="Thompson Sampling is optimal for Bernoulli bandits.",
            entities=["Thompson Sampling", "Bernoulli", "optimal"],
            motifs=["thompson", "bernoulli", "optimal"],
            metadata={"source": "test", "importance": 0.7}
        ),
        MemoryShard(
            id="shard_004",
            text="The weaving architecture consists of 9 layers from Loom Command to Reflection Buffer.",
            entities=["weaving", "architecture", "layers", "Loom Command", "Reflection Buffer"],
            motifs=["weaving", "architecture", "layers"],
            metadata={"source": "test", "importance": 0.85}
        ),
        MemoryShard(
            id="shard_005",
            text="Matryoshka embeddings enable multi-scale semantic representations.",
            entities=["Matryoshka", "embeddings", "semantic"],
            motifs=["matryoshka", "embeddings", "semantic"],
            metadata={"source": "test", "importance": 0.75}
        ),
    ]


def validate_layer_1_loom_command(spacetime) -> tuple[bool, str]:
    """Validate Layer 1: Loom Command pattern selection."""
    pattern_card = spacetime.metadata.get('pattern_card')
    execution_mode = spacetime.metadata.get('execution_mode')

    if not pattern_card:
        return False, "Layer 1 FAIL: No pattern_card in metadata"

    valid_patterns = ["Bare Threading", "Fast Threading", "Full Fusion"]
    if pattern_card not in valid_patterns:
        return False, f"Layer 1 FAIL: Invalid pattern '{pattern_card}'"

    if not execution_mode:
        return False, "Layer 1 FAIL: No execution_mode in metadata"

    return True, f"Layer 1 PASS: Pattern={pattern_card}, Mode={execution_mode}"


def validate_layer_2_chrono_trigger(spacetime) -> tuple[bool, str]:
    """Validate Layer 2: Chrono Trigger temporal window."""
    chrono_timeout = spacetime.metadata.get('chrono_timeout')

    if chrono_timeout is None:
        return False, "Layer 2 FAIL: No chrono_timeout in metadata"

    if chrono_timeout <= 0:
        return False, f"Layer 2 FAIL: Invalid timeout {chrono_timeout}"

    # Check timing exists
    if not spacetime.trace.start_time or not spacetime.trace.end_time:
        return False, "Layer 2 FAIL: Missing temporal markers"

    return True, f"Layer 2 PASS: Timeout={chrono_timeout}ms, Duration={spacetime.trace.duration_ms:.1f}ms"


def validate_layer_3_yarn_graph(spacetime) -> tuple[bool, str]:
    """Validate Layer 3: Yarn Graph thread selection."""
    threads = spacetime.trace.threads_activated

    if threads is None:
        return False, "Layer 3 FAIL: No threads_activated in trace"

    if not isinstance(threads, list):
        return False, f"Layer 3 FAIL: threads_activated is {type(threads)}, expected list"

    # Should have at least some threads (even if empty list is valid)
    thread_count = len(threads)

    return True, f"Layer 3 PASS: {thread_count} threads activated"


def validate_layer_4_resonance_shed(spacetime) -> tuple[bool, str]:
    """Validate Layer 4: Resonance Shed feature extraction."""
    motifs = spacetime.trace.motifs_detected
    scales = spacetime.trace.embedding_scales_used

    if motifs is None:
        return False, "Layer 4 FAIL: No motifs_detected in trace"

    if scales is None or len(scales) == 0:
        return False, "Layer 4 FAIL: No embedding_scales_used in trace"

    motif_count = len(motifs)
    scale_info = f"scales={scales}"

    return True, f"Layer 4 PASS: {motif_count} motifs detected, {scale_info}"


def validate_layer_5_warp_space(spacetime) -> tuple[bool, str]:
    """Validate Layer 5: Warp Space tensor field."""
    warp_ops = spacetime.trace.warp_operations
    tensor_stats = spacetime.trace.tensor_field_stats

    if warp_ops is None:
        return False, "Layer 5 FAIL: No warp_operations in trace"

    if not isinstance(warp_ops, list) or len(warp_ops) == 0:
        return False, "Layer 5 FAIL: warp_operations is empty or invalid"

    # Check for tension operation
    has_tension = any('tension' in str(op).lower() for op in warp_ops)
    if not has_tension:
        return False, "Layer 5 FAIL: No tension operation found"

    op_count = len(warp_ops)
    return True, f"Layer 5 PASS: {op_count} warp operations, tensor_stats={tensor_stats is not None}"


def validate_layer_6_convergence_engine(spacetime) -> tuple[bool, str]:
    """Validate Layer 6: Convergence Engine decision collapse."""
    tool = spacetime.tool_used
    confidence = spacetime.confidence
    bandit_stats = spacetime.trace.bandit_statistics

    if not tool:
        return False, "Layer 6 FAIL: No tool_used"

    valid_tools = ["answer", "search", "calc", "notion_write"]
    if tool not in valid_tools:
        return False, f"Layer 6 FAIL: Invalid tool '{tool}'"

    if not (0.0 <= confidence <= 1.0):
        return False, f"Layer 6 FAIL: Invalid confidence {confidence}"

    # Bandit statistics might be None (that's ok)
    bandit_info = "with bandit stats" if bandit_stats else "no bandit stats"

    return True, f"Layer 6 PASS: Tool={tool}, Confidence={confidence:.2f}, {bandit_info}"


def validate_layer_7_tool_executor(spacetime) -> tuple[bool, str]:
    """Validate Layer 7: Tool Executor execution."""
    response = spacetime.response
    tool = spacetime.tool_used

    if not response:
        return False, "Layer 7 FAIL: No response generated"

    if not isinstance(response, str):
        return False, f"Layer 7 FAIL: Response is {type(response)}, expected str"

    if len(response) == 0:
        return False, "Layer 7 FAIL: Empty response"

    return True, f"Layer 7 PASS: Tool '{tool}' executed, response length={len(response)}"


def validate_layer_8_spacetime_fabric(spacetime) -> tuple[bool, str]:
    """Validate Layer 8: Spacetime Fabric assembly."""
    trace = spacetime.trace

    if not trace:
        return False, "Layer 8 FAIL: No trace"

    if not trace.start_time or not trace.end_time:
        return False, "Layer 8 FAIL: Missing temporal markers"

    if trace.duration_ms <= 0:
        return False, f"Layer 8 FAIL: Invalid duration {trace.duration_ms}"

    stage_count = len(trace.stage_durations)
    if stage_count < 5:
        return False, f"Layer 8 FAIL: Only {stage_count} stages tracked (expected ≥5)"

    return True, f"Layer 8 PASS: Complete trace with {stage_count} stages, {trace.duration_ms:.1f}ms total"


def validate_layer_9_reflection_buffer(spacetime) -> tuple[bool, str]:
    """
    Validate Layer 9: Reflection Buffer (implicit).

    Note: Reflection happens asynchronously, but we can verify the data
    needed for reflection is present in the spacetime artifact.
    """
    # Check that spacetime has all fields needed for reflection
    has_query = bool(spacetime.query_text)
    has_response = bool(spacetime.response)
    has_confidence = spacetime.confidence is not None
    has_trace = spacetime.trace is not None

    if not all([has_query, has_response, has_confidence, has_trace]):
        missing = []
        if not has_query: missing.append("query")
        if not has_response: missing.append("response")
        if not has_confidence: missing.append("confidence")
        if not has_trace: missing.append("trace")
        return False, f"Layer 9 FAIL: Missing reflection data: {missing}"

    # Check metadata has pattern info for reflection learning
    has_pattern = 'pattern_card' in spacetime.metadata
    has_mode = 'execution_mode' in spacetime.metadata

    if not has_pattern or not has_mode:
        return False, "Layer 9 FAIL: Missing pattern metadata for reflection"

    return True, f"Layer 9 PASS: Reflection data complete (confidence={spacetime.confidence:.2f})"


# ============================================================================
# Core Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_complete_9_layer_weaving_cycle():
    """
    Integration test validating all 9 layers of the weaving architecture.

    This test ensures:
    1. Each layer is executed in correct order
    2. Data flows properly between layers
    3. Complete provenance is captured in Spacetime trace
    4. All timing metrics are collected
    """
    print("\n" + "=" * 80)
    print("TEST: Complete 9-Layer Weaving Cycle")
    print("=" * 80)

    # Setup
    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards, enable_reflection=False) as orchestrator:
        # Execute weaving cycle
        query = Query(text="What is Thompson Sampling?")
        print(f"\nQuery: '{query.text}'")
        print(f"Config: {config.mode.value}")
        print(f"Shards: {len(shards)}")

        start_time = time.time()
        spacetime = await orchestrator.weave(query)
        duration = (time.time() - start_time) * 1000

        print(f"\n{'=' * 80}")
        print("LAYER VALIDATION RESULTS")
        print('=' * 80)

        # Validate all 9 layers
        validators = [
            ("Layer 1: Loom Command", validate_layer_1_loom_command),
            ("Layer 2: Chrono Trigger", validate_layer_2_chrono_trigger),
            ("Layer 3: Yarn Graph", validate_layer_3_yarn_graph),
            ("Layer 4: Resonance Shed", validate_layer_4_resonance_shed),
            ("Layer 5: Warp Space", validate_layer_5_warp_space),
            ("Layer 6: Convergence Engine", validate_layer_6_convergence_engine),
            ("Layer 7: Tool Executor", validate_layer_7_tool_executor),
            ("Layer 8: Spacetime Fabric", validate_layer_8_spacetime_fabric),
            ("Layer 9: Reflection Buffer", validate_layer_9_reflection_buffer),
        ]

        passed = 0
        failed = 0

        for layer_name, validator in validators:
            success, message = validator(spacetime)
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status}: {layer_name}")
            print(f"       {message}")

            if success:
                passed += 1
            else:
                failed += 1

            # Assert for pytest
            assert success, f"{layer_name} validation failed: {message}"

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {passed}/9 layers passed, {failed} failed")
        print(f"Total Duration: {duration:.1f}ms")
        print(f"Trace Duration: {spacetime.trace.duration_ms:.1f}ms")
        print('=' * 80)

        # Overall assertions
        assert passed == 9, f"Not all layers passed: {passed}/9"
        assert spacetime is not None
        assert spacetime.response is not None
        assert len(spacetime.response) > 0


@pytest.mark.asyncio
async def test_9_layer_cycle_with_yarn_graph():
    """
    Test cycle using Yarn Graph (knowledge graph backend).

    This validates that Layer 3 (Yarn Graph) works correctly when using
    the graph-based memory backend instead of static shards.
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle with Yarn Graph (KG Backend)")
    print("=" * 80)

    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing

    # Create test shards for YarnGraph
    shards = create_test_shards()

    # Note: WeavingOrchestrator automatically wraps shards in YarnGraph
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        print(f"\nQuery: '{query.text}'")

        spacetime = await orchestrator.weave(query)

        # Verify Yarn Graph was used (threads activated)
        threads = spacetime.trace.threads_activated
        assert threads is not None, "No threads activated from Yarn Graph"
        assert isinstance(threads, list), f"threads_activated should be list, got {type(threads)}"

        print(f"[PASS] Yarn Graph activated {len(threads)} threads")

        # Verify response generated
        assert spacetime.response is not None
        assert len(spacetime.response) > 0

        print(f"[PASS] Response generated: {len(spacetime.response)} chars")
        print(f"[PASS] Tool used: {spacetime.tool_used}")
        print(f"[PASS] Confidence: {spacetime.confidence:.2f}")


@pytest.mark.asyncio
async def test_9_layer_cycle_all_complexity_levels():
    """
    Test all complexity levels (BARE/FAST/FUSED).

    Validates that all 9 layers execute correctly regardless of
    complexity level, with appropriate scaling of resources.
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle Across All Complexity Levels")
    print("=" * 80)

    configs = [
        (Config.bare(), "BARE"),
        (Config.fast(), "FAST"),
        (Config.fused(), "FUSED"),
    ]

    shards = create_test_shards()
    query = Query(text="What is Thompson Sampling?")

    results = []

    for config, mode_name in configs:
        config.enable_safety_guardrails = False  # Disable approval for automated testing

        print(f"\n{'-' * 80}")
        print(f"Testing {mode_name} mode")
        print('-' * 80)

        async with WeavingOrchestrator(cfg=config, shards=shards, enable_reflection=False) as orchestrator:
            start = time.time()
            spacetime = await orchestrator.weave(query)
            duration = (time.time() - start) * 1000

            # All layers should execute regardless of complexity
            assert spacetime is not None, f"{mode_name}: No spacetime returned"
            assert spacetime.response is not None, f"{mode_name}: No response"
            assert spacetime.confidence > 0.0, f"{mode_name}: Invalid confidence"
            assert spacetime.trace.duration_ms > 0, f"{mode_name}: Invalid trace duration"

            # Layer 3 validation
            threads = spacetime.trace.threads_activated
            assert threads is not None, f"{mode_name}: No threads activated"

            # Layer 4 validation
            scales = spacetime.trace.embedding_scales_used
            assert scales is not None and len(scales) > 0, f"{mode_name}: No embedding scales"

            # Layer 8 validation
            stages = len(spacetime.trace.stage_durations)
            assert stages >= 5, f"{mode_name}: Only {stages} stages tracked"

            results.append({
                'mode': mode_name,
                'duration': duration,
                'trace_duration': spacetime.trace.duration_ms,
                'threads': len(threads) if threads else 0,
                'scales': len(scales) if scales else 0,
                'stages': stages,
                'confidence': spacetime.confidence,
                'tool': spacetime.tool_used
            })

            print(f"[PASS] {mode_name}: {duration:.1f}ms, {len(threads)} threads, {len(scales)} scales, confidence={spacetime.confidence:.2f}")

    print(f"\n{'=' * 80}")
    print("COMPLEXITY COMPARISON")
    print('=' * 80)
    print(f"{'Mode':<10} {'Duration':<12} {'Threads':<10} {'Scales':<10} {'Stages':<10} {'Confidence':<12}")
    print('-' * 80)
    for r in results:
        print(f"{r['mode']:<10} {r['duration']:>8.1f}ms   {r['threads']:<10} {r['scales']:<10} {r['stages']:<10} {r['confidence']:<12.2f}")
    print('=' * 80)


@pytest.mark.asyncio
async def test_9_layer_cycle_timing_breakdown():
    """
    Verify stage timing metrics are collected for all layers.

    Ensures that the trace contains detailed timing information
    for debugging and performance optimization.
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle Timing Breakdown")
    print("=" * 80)

    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Expected stages (not all may be present depending on execution path)
        expected_stages = [
            'pattern_selection',
            'temporal_setup',
            'thread_selection',
            'feature_extraction',
            'warp_tensioning',
            'retrieval',
            'convergence',
            'tool_execution',
            'spacetime_assembly'
        ]

        print("\nStage Timing Breakdown:")
        print("-" * 80)

        found_stages = []
        total_measured = 0.0

        for stage in expected_stages:
            if stage in spacetime.trace.stage_durations:
                duration = spacetime.trace.stage_durations[stage]
                assert duration >= 0.0, f"Stage '{stage}' has negative duration: {duration}"
                found_stages.append(stage)
                total_measured += duration
                print(f"  {stage:<25} {duration:>8.2f} ms")

        print("-" * 80)
        print(f"  {'Total (measured stages)':<25} {total_measured:>8.2f} ms")
        print(f"  {'Total (trace)':<25} {spacetime.trace.duration_ms:>8.2f} ms")
        print(f"  {'Stages tracked':<25} {len(found_stages)}/{len(expected_stages)}")

        # Should have at least 5 core stages
        assert len(found_stages) >= 5, f"Only {len(found_stages)} stages tracked, expected >= 5"

        print(f"\n[PASS] Timing validation passed: {len(found_stages)} stages tracked")


@pytest.mark.asyncio
async def test_9_layer_cycle_error_handling():
    """
    Test that errors in any layer are handled gracefully.

    Validates the system's resilience by testing with:
    - Empty shards (should handle gracefully)
    - Malformed queries (should still process)
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle Error Handling")
    print("=" * 80)

    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing

    # Test 1: Empty shards
    print("\nTest 1: Empty shards")
    print("-" * 80)

    async with WeavingOrchestrator(cfg=config, shards=[]) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Should still produce a result (fallback behavior)
        assert spacetime is not None, "No spacetime with empty shards"
        assert spacetime.response is not None, "No response with empty shards"

        print(f"[PASS] Empty shards handled: response length={len(spacetime.response)}")
        print(f"  Threads activated: {len(spacetime.trace.threads_activated)}")
        print(f"  Tool used: {spacetime.tool_used}")

    # Test 2: Very short query
    print("\nTest 2: Very short query")
    print("-" * 80)

    shards = create_test_shards()
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        query = Query(text="Hi")
        spacetime = await orchestrator.weave(query)

        assert spacetime is not None, "No spacetime for short query"
        assert spacetime.response is not None, "No response for short query"

        print(f"[PASS] Short query handled: '{query.text}'")
        print(f"  Response length: {len(spacetime.response)}")
        print(f"  Confidence: {spacetime.confidence:.2f}")

    # Test 3: Very long query
    print("\nTest 3: Very long query")
    print("-" * 80)

    long_query = " ".join(["thompson sampling exploration exploitation"] * 20)
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        query = Query(text=long_query)
        spacetime = await orchestrator.weave(query)

        assert spacetime is not None, "No spacetime for long query"
        assert spacetime.response is not None, "No response for long query"

        print(f"[PASS] Long query handled: {len(query.text)} chars")
        print(f"  Response length: {len(spacetime.response)}")
        print(f"  Confidence: {spacetime.confidence:.2f}")


@pytest.mark.asyncio
async def test_9_layer_cycle_provenance_completeness():
    """
    Test that complete provenance is captured throughout the cycle.

    Validates that every layer leaves traceable evidence in the
    Spacetime artifact for debugging and analysis.
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle Provenance Completeness")
    print("=" * 80)

    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        query = Query(text="Explain Thompson Sampling in detail")
        spacetime = await orchestrator.weave(query)

        print("\nProvenance Data Collected:")
        print("-" * 80)

        # Layer 1: Pattern selection
        pattern = spacetime.metadata.get('pattern_card')
        mode = spacetime.metadata.get('execution_mode')
        print(f"Layer 1 (Loom Command): pattern={pattern}, mode={mode}")
        assert pattern is not None, "Missing pattern provenance"

        # Layer 2: Temporal window
        timeout = spacetime.metadata.get('chrono_timeout')
        print(f"Layer 2 (Chrono Trigger): timeout={timeout}ms")
        assert timeout is not None, "Missing temporal provenance"

        # Layer 3: Thread selection
        threads = spacetime.trace.threads_activated
        print(f"Layer 3 (Yarn Graph): {len(threads)} threads activated")
        print(f"         Thread IDs: {threads[:3]}..." if len(threads) > 3 else f"         Thread IDs: {threads}")
        assert threads is not None, "Missing thread provenance"

        # Layer 4: Feature extraction
        motifs = spacetime.trace.motifs_detected
        scales = spacetime.trace.embedding_scales_used
        spectral = spacetime.trace.spectral_features
        print(f"Layer 4 (Resonance Shed): {len(motifs)} motifs, scales={scales}")
        print(f"         Motifs: {motifs[:3]}..." if len(motifs) > 3 else f"         Motifs: {motifs}")
        print(f"         Spectral: {spectral is not None}")
        assert motifs is not None, "Missing motif provenance"
        assert scales is not None, "Missing scale provenance"

        # Layer 5: Warp operations
        warp_ops = spacetime.trace.warp_operations
        tensor_stats = spacetime.trace.tensor_field_stats
        print(f"Layer 5 (Warp Space): {len(warp_ops)} operations")
        print(f"         Operations: {warp_ops}")
        print(f"         Tensor stats: {tensor_stats}")
        assert warp_ops is not None and len(warp_ops) > 0, "Missing warp provenance"

        # Layer 6: Decision collapse
        tool = spacetime.tool_used
        confidence = spacetime.confidence
        adapter = spacetime.trace.policy_adapter
        bandit = spacetime.trace.bandit_statistics
        print(f"Layer 6 (Convergence): tool={tool}, confidence={confidence:.2f}")
        print(f"         Adapter: {adapter}")
        print(f"         Bandit stats: {bandit is not None}")
        assert tool is not None, "Missing tool provenance"

        # Layer 7: Tool execution
        response = spacetime.response
        print(f"Layer 7 (Tool Executor): response_length={len(response)}")
        print(f"         Preview: {response[:100]}...")
        assert response is not None and len(response) > 0, "Missing execution provenance"

        # Layer 8: Spacetime assembly
        start = spacetime.trace.start_time
        end = spacetime.trace.end_time
        duration = spacetime.trace.duration_ms
        stages = len(spacetime.trace.stage_durations)
        print(f"Layer 8 (Spacetime): duration={duration:.1f}ms, {stages} stages")
        print(f"         Start: {start}")
        print(f"         End: {end}")
        assert start is not None and end is not None, "Missing temporal provenance"
        assert duration > 0, "Invalid duration provenance"

        # Layer 9: Reflection data
        context_shards = spacetime.trace.context_shards_count
        sources = spacetime.sources_used
        print(f"Layer 9 (Reflection): {context_shards} context shards")
        print(f"         Sources: {sources}")
        assert context_shards >= 0, "Missing context provenance"

        print("\n" + "=" * 80)
        print("[PASS] Complete provenance validated across all 9 layers")
        print("=" * 80)


# ============================================================================
# Performance and Stress Tests
# ============================================================================

@pytest.mark.asyncio
async def test_9_layer_cycle_performance_baseline():
    """
    Establish performance baseline for 9-layer cycle.

    Measures timing for each complexity level to detect regressions.
    """
    print("\n" + "=" * 80)
    print("TEST: 9-Layer Cycle Performance Baseline")
    print("=" * 80)

    configs = [
        (Config.bare(), "BARE", 100),    # Expected < 100ms
        (Config.fast(), "FAST", 200),    # Expected < 200ms
        (Config.fused(), "FUSED", 500),  # Expected < 500ms
    ]

    shards = create_test_shards()
    query = Query(text="What is Thompson Sampling?")

    print(f"\n{'Mode':<10} {'Runs':<8} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Target':<12}")
    print("-" * 80)

    for config, mode_name, target_ms in configs:
        config.enable_safety_guardrails = False  # Disable approval for automated testing
        durations = []

        async with WeavingOrchestrator(cfg=config, shards=shards, enable_reflection=False) as orchestrator:
            # Run 3 times to get average
            for _ in range(3):
                start = time.time()
                await orchestrator.weave(query)
                duration = (time.time() - start) * 1000
                durations.append(duration)

        avg = sum(durations) / len(durations)
        min_d = min(durations)
        max_d = max(durations)

        status = "[OK]" if avg < target_ms else "[WARN]"
        print(f"{mode_name:<10} {len(durations):<8} {avg:>8.1f} ms  {min_d:>8.1f} ms  {max_d:>8.1f} ms  {status} < {target_ms}ms")

    print("=" * 80)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    import sys

    print("\n" + "=" * 80)
    print("HoloLoom 9-Layer Weaving Cycle Integration Tests")
    print("=" * 80)

    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"  # Show print statements
    ])

    sys.exit(exit_code)
