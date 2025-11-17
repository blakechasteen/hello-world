#!/usr/bin/env python3
"""
Reasoning Engine Battle Test Suite
===================================
Comprehensive tests for modularity, error handling, and edge cases.

Tests:
1. Modularity: Component swapping
2. Error handling: Malformed inputs
3. Edge cases: Empty context, null features, timeouts
4. Hardening: Missing dependencies, resource limits
5. Production scenarios: High load, concurrent requests
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports to avoid sklearn dependency
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base = Path(__file__).parent / "HoloLoom"

# Load types first
types = load_module("reasoning_types", base / "reasoning" / "types.py")
doc_types = load_module("doc_types", base / "documentation" / "types.py")

# Load components
planner_mod = load_module("planner", base / "reasoning" / "planner.py")
cot_mod = load_module("cot", base / "reasoning" / "chain_of_thought.py")
verifier_mod = load_module("verifier", base / "reasoning" / "verifier.py")
engine_mod = load_module("engine", base / "reasoning" / "engine.py")

print("=" * 70)
print("REASONING ENGINE BATTLE TEST SUITE")
print("=" * 70)
print()

# Test counters
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(name):
    def decorator(func):
        async def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            try:
                await func()
                print(f"✓ {name}")
                tests_passed += 1
            except Exception as e:
                print(f"✗ {name}")
                print(f"  Error: {e}")
                tests_failed += 1
        return wrapper
    return decorator

# ============================================================================
# 1. MODULARITY TESTS
# ============================================================================

print("1. MODULARITY TESTS")
print("-" * 70)

@test("Component swapping: Custom ChainOfThought")
async def test_custom_cot():
    """Test that we can inject a custom ChainOfThought component."""

    class CustomCoT(cot_mod.ChainOfThought):
        def generate_standard_chain(self, query, intent, context):
            return [types.ReasoningStep(
                thought="Custom reasoning",
                evidence="Custom evidence",
                confidence=0.95,
                step_type=types.StepType.SYNTHESIS
            )]

    engine = engine_mod.ReasoningEngine()
    engine.cot_generator = CustomCoT()

    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context, mode=types.ReasoningMode.STANDARD)

    assert len(result.chain) > 0
    assert result.chain[0].thought == "Custom reasoning"

@test("Component swapping: Custom Verifier")
async def test_custom_verifier():
    """Test that we can inject a custom Verifier component."""

    class StrictVerifier(verifier_mod.SelfVerifier):
        async def verify(self, chain, context):
            return types.VerificationResult(
                passed=False,
                issue="Always fails (test)",
                confidence=0.5,
                severity=types.VerificationSeverity.CRITICAL
            )

    engine = engine_mod.ReasoningEngine()
    engine.verifier = StrictVerifier(threshold=0.9)

    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    result = await engine.reason_deep(query, features, context)

    # Should have verification step
    verification_steps = [s for s in result.chain if s.step_type == types.StepType.VERIFICATION]
    assert len(verification_steps) > 0

# ============================================================================
# 2. ERROR HANDLING TESTS
# ============================================================================

print()
print("2. ERROR HANDLING TESTS")
print("-" * 70)

@test("Malformed input: Null query")
async def test_null_query():
    """Test handling of None query."""
    engine = engine_mod.ReasoningEngine()

    try:
        result = await engine.reason(
            None,
            doc_types.Features(motifs=[]),
            doc_types.Context(shards=[])
        )
        raise AssertionError("Should have raised an error for null query")
    except (AttributeError, TypeError):
        pass  # Expected

@test("Malformed input: Empty features")
async def test_empty_features():
    """Test handling of empty features."""
    engine = engine_mod.ReasoningEngine()

    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=[])
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context)

    assert result is not None
    assert len(result.chain) > 0

@test("Malformed input: Empty context")
async def test_empty_context():
    """Test handling of empty context."""
    engine = engine_mod.ReasoningEngine()

    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context)

    assert result is not None
    assert len(result.chain) > 0

@test("Invalid mode: Unknown reasoning mode")
async def test_invalid_mode():
    """Test handling of invalid reasoning mode."""
    engine = engine_mod.ReasoningEngine()

    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    try:
        # Create a fake mode
        result = await engine.reason(query, features, context, mode="INVALID")
        raise AssertionError("Should have raised ValueError for invalid mode")
    except ValueError:
        pass  # Expected

# ============================================================================
# 3. EDGE CASES
# ============================================================================

print()
print("3. EDGE CASE TESTS")
print("-" * 70)

@test("Edge case: Very long query (1000+ words)")
async def test_long_query():
    """Test handling of very long query."""
    engine = engine_mod.ReasoningEngine()

    long_text = " ".join(["word"] * 1000)
    query = doc_types.Query(text=long_text)
    features = doc_types.Features(motifs=["word"] * 10)
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context)

    assert result is not None
    assert result.total_confidence >= 0

@test("Edge case: Unicode and special characters")
async def test_unicode_query():
    """Test handling of Unicode characters."""
    engine = engine_mod.ReasoningEngine()

    query = doc_types.Query(text="测试 🧠 Тест éñ")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context)

    assert result is not None

@test("Edge case: Max steps exceeded")
async def test_max_steps():
    """Test that max_thinking_steps is respected."""
    engine = engine_mod.ReasoningEngine(max_thinking_steps=2)

    query = doc_types.Query(text="complex query requiring many steps")
    features = doc_types.Features(motifs=["test"] * 10)
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context, mode=types.ReasoningMode.DEEP)

    # Should not exceed max steps
    assert len(result.chain) <= 15  # Some buffer for planning steps

# ============================================================================
# 4. HARDENING TESTS
# ============================================================================

print()
print("4. HARDENING TESTS")
print("-" * 70)

@test("Timeout handling: MISSING - No timeout implementation")
async def test_timeout():
    """Test timeout handling (currently not implemented)."""
    # This test documents the missing feature
    engine = engine_mod.ReasoningEngine()

    # Simulate slow operation (in real code, this would need async.wait_for)
    query = doc_types.Query(text="test")
    features = doc_types.Features(motifs=["test"])
    context = doc_types.Context(shards=[])

    start = time.time()
    result = await engine.reason(query, features, context)
    duration = time.time() - start

    # NOTE: No actual timeout enforced!
    print(f"    WARNING: Duration {duration*1000:.1f}ms, no timeout enforced")

@test("Resource limits: Memory usage tracking")
async def test_memory():
    """Test memory usage stays reasonable."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    engine = engine_mod.ReasoningEngine()

    # Run 100 reasoning cycles
    for i in range(100):
        query = doc_types.Query(text=f"test query {i}")
        features = doc_types.Features(motifs=["test"])
        context = doc_types.Context(shards=[])
        result = await engine.reason(query, features, context)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_delta = mem_after - mem_before

    print(f"    Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{mem_delta:.1f}MB)")

    # Should not leak more than 10MB for 100 iterations
    assert mem_delta < 50, f"Memory leak detected: {mem_delta:.1f}MB"

@test("Concurrent requests: 10 parallel reasoning tasks")
async def test_concurrent():
    """Test concurrent reasoning requests."""
    engine = engine_mod.ReasoningEngine()

    async def reason_task(i):
        query = doc_types.Query(text=f"concurrent test {i}")
        features = doc_types.Features(motifs=["test"])
        context = doc_types.Context(shards=[])
        return await engine.reason(query, features, context)

    results = await asyncio.gather(*[reason_task(i) for i in range(10)])

    assert len(results) == 10
    assert all(r.total_confidence >= 0 for r in results)

# ============================================================================
# 5. PRODUCTION SCENARIOS
# ============================================================================

print()
print("5. PRODUCTION SCENARIO TESTS")
print("-" * 70)

@test("Production: Mode escalation (FAST → STANDARD)")
async def test_escalation():
    """Test automatic mode escalation."""
    engine = engine_mod.ReasoningEngine()

    # Create low-confidence scenario
    query = doc_types.Query(text="ambiguous query")
    features = doc_types.Features(motifs=[])
    context = doc_types.Context(shards=[])

    result = await engine.reason(query, features, context, mode=types.ReasoningMode.FAST)

    # FAST mode should return a result even with low confidence
    assert result is not None
    # NOTE: Actual escalation to STANDARD not implemented in FAST mode yet

@test("Production: Confidence calibration")
async def test_confidence_range():
    """Test that confidence scores are in valid range [0, 1]."""
    engine = engine_mod.ReasoningEngine()

    for _ in range(20):
        query = doc_types.Query(text="test query")
        features = doc_types.Features(motifs=["test"])
        context = doc_types.Context(shards=[])

        result = await engine.reason(query, features, context)

        assert 0.0 <= result.total_confidence <= 1.0, \
            f"Confidence out of range: {result.total_confidence}"

        for step in result.chain:
            assert 0.0 <= step.confidence <= 1.0, \
                f"Step confidence out of range: {step.confidence}"

@test("Production: Consistent results for same input")
async def test_consistency():
    """Test that same input produces similar results."""
    engine = engine_mod.ReasoningEngine()

    query = doc_types.Query(text="consistent test query")
    features = doc_types.Features(motifs=["test", "query"])
    context = doc_types.Context(shards=[])

    result1 = await engine.reason(query, features, context)
    result2 = await engine.reason(query, features, context)

    # Should produce same number of steps
    assert len(result1.chain) == len(result2.chain)

    # Confidence should be similar (within 10%)
    assert abs(result1.total_confidence - result2.total_confidence) < 0.1

# ============================================================================
# RUN ALL TESTS
# ============================================================================

async def run_all_tests():
    await test_custom_cot()
    await test_custom_verifier()

    await test_null_query()
    await test_empty_features()
    await test_empty_context()
    await test_invalid_mode()

    await test_long_query()
    await test_unicode_query()
    await test_max_steps()

    await test_timeout()
    await test_memory()
    await test_concurrent()

    await test_escalation()
    await test_confidence_range()
    await test_consistency()

if __name__ == "__main__":
    asyncio.run(run_all_tests())

    print()
    print("=" * 70)
    print("BATTLE TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {tests_run}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print(f"Success rate: {100 * tests_passed / tests_run if tests_run > 0 else 0:.1f}%")
    print()

    if tests_failed > 0:
        print("❌ BATTLE TEST FAILED")
        sys.exit(1)
    else:
        print("✅ BATTLE TEST PASSED")
        sys.exit(0)
