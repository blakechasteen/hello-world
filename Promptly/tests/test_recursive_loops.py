#!/usr/bin/env python3
"""
Test script for recursive loops system
Tests stopping conditions, scratchpad, and Hofstadter loops
"""

import sys
import os

# Add promptly to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from recursive_loops import (
    RecursiveEngine, LoopConfig, LoopType,
    refine_iteratively, think_recursively
)


# Global counter for tracking iterations
_iteration_counter = 0

def mock_executor_improving(prompt: str) -> str:
    """Mock executor that simulates improving quality over iterations"""
    global _iteration_counter

    # Detect if this is a critique call or refinement call
    is_critique = "critique" in prompt.lower() or "evaluate your output" in prompt.lower()

    if not is_critique:
        # This is a refinement request - increment counter
        _iteration_counter += 1
        iteration = _iteration_counter
        return f"""THOUGHT: This is iteration {iteration}
ACTION: Improving the output
IMPROVED: Better output at iteration {iteration}"""
    else:
        # This is a critique request - use current iteration
        iteration = _iteration_counter

        # Simulate improvement over iterations
        quality_scores = {
            1: 6,  # 0.6
            2: 7,  # 0.7
            3: 9,  # 0.9 - should trigger quality threshold
            4: 9,  # 0.9
        }

        score = quality_scores.get(iteration, 7)
        return f"""SCORE: {score}
STRENGTHS: Getting better with iteration {iteration}
WEAKNESSES: Still some room for improvement
SUGGESTIONS: Keep refining"""


def mock_executor_plateau(prompt: str) -> str:
    """Mock executor that plateaus (no improvement)"""
    if "critique" in prompt.lower() or "score:" in prompt.lower():
        return """SCORE: 7
STRENGTHS: Decent quality
WEAKNESSES: Not much changing
SUGGESTIONS: Minor tweaks"""
    else:
        return """THOUGHT: Making small changes
ACTION: Tweaking output
IMPROVED: Slightly modified output"""


def mock_executor_hofstadter(prompt: str) -> str:
    """Mock executor for Hofstadter strange loops"""
    if "Level 1" in prompt:
        return """OUTPUT: Direct answer to the question"""
    elif "Level 2" in prompt:
        return """REFLECTION: I notice Level 1 gave a surface answer
META_INSIGHT: We need to think about the thinking itself
OUTPUT: Meta-level perspective on the question"""
    elif "Level 3" in prompt:
        return """REFLECTION: Level 2 started thinking about thinking
META_INSIGHT: This creates a strange loop - thinking about thinking about thinking
OUTPUT: Self-referential insight about the recursive nature"""
    else:
        return """OUTPUT: Deep synthesis across all meta-levels"""


def test_quality_threshold_stop():
    """Test that loop stops when quality threshold is reached"""
    global _iteration_counter
    _iteration_counter = 0  # Reset counter

    print("\n" + "="*60)
    print("TEST 1: Quality Threshold Stopping")
    print("="*60)

    engine = RecursiveEngine(mock_executor_improving)
    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=10,  # High max
        quality_threshold=0.85,  # Should stop at iteration 3 (score 0.9)
        min_improvement=0.01,
        enable_scratchpad=True
    )

    result = engine.execute_refine_loop(
        task="Test task",
        initial_output="Initial output",
        config=config
    )

    print(f"\n- Success: {result.success}")
    print(f"- Iterations: {result.iterations}")
    print(f"- Stop Reason: {result.stop_reason}")
    print(f"- Quality History: {result.improvement_history}")

    assert result.iterations == 3, f"Expected 3 iterations, got {result.iterations}"
    assert "Quality threshold" in result.stop_reason, f"Wrong stop reason: {result.stop_reason}"
    assert result.scratchpad is not None, "Scratchpad should be enabled"
    assert len(result.scratchpad.entries) == 3, "Scratchpad should have 3 entries"

    print("\n[PASSED] Stops at quality threshold")
    print(f"\nScratchpad Preview:")
    if result.scratchpad.entries:
        entry = result.scratchpad.entries[-1]
        print(f"  Final thought: {entry.thought}")
        print(f"  Final score: {entry.score}")


def test_no_improvement_stop():
    """Test that loop stops when improvement plateaus"""
    print("\n" + "="*60)
    print("TEST 2: No Improvement Stopping")
    print("="*60)

    engine = RecursiveEngine(mock_executor_plateau)
    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=10,
        quality_threshold=0.95,  # High threshold (won't reach)
        min_improvement=0.05,  # Should detect plateau
        enable_scratchpad=True
    )

    result = engine.execute_refine_loop(
        task="Test task",
        initial_output="Initial output",
        config=config
    )

    print(f"\n- Success: {result.success}")
    print(f"- Iterations: {result.iterations}")
    print(f"- Stop Reason: {result.stop_reason}")
    print(f"- Quality History: {result.improvement_history}")

    assert result.iterations >= 2, "Should run at least 2 iterations to detect plateau"
    assert "No significant improvement" in result.stop_reason, f"Wrong stop reason: {result.stop_reason}"

    print("\n[PASS] PASSED: Stops when no improvement detected")


def test_max_iterations_stop():
    """Test that loop respects max iterations"""
    print("\n" + "="*60)
    print("TEST 3: Max Iterations Stopping")
    print("="*60)

    # Use a mock that improves slightly each time (enough to bypass min_improvement check)
    call_count = [0]
    def mock_slight_improve(prompt: str) -> str:
        if "critique" in prompt.lower() or "evaluate your output" in prompt.lower():
            call_count[0] += 1
            # Scores: 6, 6.5, 7 (each above 0.001 threshold but below 0.99 quality threshold)
            score = 6 + (call_count[0] - 1) * 0.5
            return f"SCORE: {score}\nSTRENGTHS: Okay\nWEAKNESSES: Some\nSUGGESTIONS: Improve"
        return "THOUGHT: Thinking\nACTION: Trying\nIMPROVED: Output version"

    engine = RecursiveEngine(mock_slight_improve)
    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=3,  # Low max
        quality_threshold=0.99,  # Won't reach
        min_improvement=0.001,  # Very small - won't trigger easily
        enable_scratchpad=False
    )

    result = engine.execute_refine_loop(
        task="Test task",
        initial_output="Initial output",
        config=config
    )

    print(f"\n- Success: {result.success}")
    print(f"- Iterations: {result.iterations}")
    print(f"- Stop Reason: {result.stop_reason}")

    assert result.iterations == 3, f"Expected exactly 3 iterations, got {result.iterations}"
    assert "Max iterations" in result.stop_reason, f"Wrong stop reason: {result.stop_reason}"
    assert result.scratchpad is None, "Scratchpad should be disabled"

    print("\n[PASS] PASSED: Respects max iterations limit")


def test_hofstadter_loop():
    """Test Hofstadter strange loops"""
    print("\n" + "="*60)
    print("TEST 4: Hofstadter Strange Loops")
    print("="*60)

    engine = RecursiveEngine(mock_executor_hofstadter)
    config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=3,
        enable_scratchpad=True
    )

    result = engine.execute_hofstadter_loop(
        task="What is consciousness?",
        config=config
    )

    print(f"\n- Success: {result.success}")
    print(f"- Meta-Levels: {result.iterations}")
    print(f"- Stop Reason: {result.stop_reason}")
    print(f"- Metadata levels: {len(result.metadata.get('levels', []))}")

    assert result.iterations == 3, f"Expected 3 meta-levels, got {result.iterations}"
    assert "Strange loop complete" in result.stop_reason
    assert 'levels' in result.metadata, "Should have levels in metadata"

    print("\n[PASS] PASSED: Hofstadter loops execute all meta-levels")

    # Show the meta-level progression
    print(f"\nMeta-Level Progression:")
    for i, level_output in enumerate(result.metadata['levels'], 1):
        print(f"  Level {i}: {level_output[:80]}...")


def test_convenience_functions():
    """Test convenience wrapper functions"""
    global _iteration_counter
    _iteration_counter = 0  # Reset counter

    print("\n" + "="*60)
    print("TEST 5: Convenience Functions")
    print("="*60)

    # Test refine_iteratively (returns string, not LoopResult)
    output1 = refine_iteratively(
        executor=mock_executor_improving,
        task="Test task",
        initial_output="Initial",
        max_iterations=3
    )

    print(f"\n- refine_iteratively returned: {output1[:50]}...")

    # Reset for next test
    _iteration_counter = 0

    # Test think_recursively (returns string, not LoopResult)
    output2 = think_recursively(
        executor=mock_executor_hofstadter,
        task="Deep question",
        levels=3
    )

    print(f"\n- think_recursively returned: {output2[:50]}...")

    assert isinstance(output1, str) and isinstance(output2, str)
    assert len(output1) > 0 and len(output2) > 0
    print("\n[PASS] PASSED: Convenience functions work correctly")


def test_scratchpad_tracking():
    """Test that scratchpad properly tracks reasoning"""
    global _iteration_counter
    _iteration_counter = 0  # Reset counter

    print("\n" + "="*60)
    print("TEST 6: Scratchpad Tracking")
    print("="*60)

    engine = RecursiveEngine(mock_executor_improving)
    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=3,
        enable_scratchpad=True
    )

    result = engine.execute_refine_loop(
        task="Test task",
        initial_output="Initial",
        config=config
    )

    scratchpad = result.scratchpad
    assert scratchpad is not None, "Scratchpad should exist"
    assert len(scratchpad.entries) == result.iterations, "Should have entry per iteration"

    print(f"\n- Scratchpad entries: {len(scratchpad.entries)}")

    for entry in scratchpad.entries:
        assert entry.thought, "Each entry should have thought"
        assert entry.action, "Each entry should have action"
        assert entry.observation, "Each entry should have observation"
        assert entry.score is not None, "Each entry should have score"
        print(f"\n  Iteration {entry.iteration}:")
        print(f"    Thought: {entry.thought[:50]}...")
        print(f"    Score: {entry.score}")

    # Test history formatting
    history = scratchpad.get_history()
    assert "Iteration 1" in history, "History should contain iteration markers"
    assert "Thought:" in history, "History should contain thoughts"

    print("\n[PASS] PASSED: Scratchpad tracks all reasoning steps")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RECURSIVE LOOPS TEST SUITE")
    print("="*60)

    tests = [
        ("Quality Threshold Stop", test_quality_threshold_stop),
        ("No Improvement Stop", test_no_improvement_stop),
        ("Max Iterations Stop", test_max_iterations_stop),
        ("Hofstadter Loops", test_hofstadter_loop),
        ("Convenience Functions", test_convenience_functions),
        ("Scratchpad Tracking", test_scratchpad_tracking),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] FAILED: {name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[FAIL] ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"[PASS] Passed: {passed}/{len(tests)}")
    print(f"[FAIL] Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\nSUCCESS: ALL TESTS PASSED! Recursive loops working correctly.")
        return 0
    else:
        print(f"\n[WARN]  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
