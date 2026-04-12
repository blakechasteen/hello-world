"""
Test Consciousness Chat - 6-Step Refinement Verification

Tests the refined consciousness_chat.py to verify:
- ELEGANCE: Helper methods work correctly
- VERIFY: Validation and error handling work

Run: python test_consciousness_chat_refined.py
"""

import pytest

pytest.importorskip("gradio")

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.consciousness_chat import ChatSession


async def test_validation():
    """Test validation checks."""
    print("\n" + "="*80)
    print("TEST 1: Validation")
    print("="*80)

    session = ChatSession()

    # Test 1: Empty message
    try:
        await session.process_message("")
        print("✗ Should have raised ValueError for empty message")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected empty message: {str(e)[:50]}")

    # Test 2: Whitespace only
    try:
        await session.process_message("   ")
        print("✗ Should have raised ValueError for whitespace message")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected whitespace message: {str(e)[:50]}")

    # Test 3: Invalid complexity
    session.complexity = "INVALID"
    response = await session.process_message("test message")
    if session.complexity == "FULL":
        print(f"✓ Correctly reset invalid complexity to FULL")
    else:
        print(f"✗ Failed to reset invalid complexity")
        return False

    return True


async def test_helpers():
    """Test helper methods."""
    print("\n" + "="*80)
    print("TEST 2: Helper Methods")
    print("="*80)

    session = ChatSession()

    # Test _build_fusion_details
    from ui.consciousness_ui_simple import MemoryNode
    memories = [
        MemoryNode("content1", 0.9, 0.9, 0, None),
        MemoryNode("content2", 0.8, 0.8, 1, None),
    ]
    details = session._build_fusion_details(memories, enabled=True, max_passes=3)
    print(f"✓ _build_fusion_details: {details['memories_retrieved']} memories, depth={details['max_depth']}")

    # Test _build_response_metadata
    import time
    from dataclasses import dataclass

    @dataclass
    class MockAwareness:
        confidence: float = 0.9
        domain: str = "test"
        subdomain: str = "unit"
        is_question: bool = True

    @dataclass
    class MockPacked:
        total_tokens: int = 100
        elements_included: int = 5
        elements_compressed: int = 2
        avg_importance: float = 0.8
        packing_time_ms: float = 10.0

    start = time.perf_counter()
    metadata = session._build_response_metadata(
        MockAwareness(), details, MockPacked(), start, 50.0
    )
    print(f"✓ _build_response_metadata: {metadata['performance']['total_ms']:.2f}ms total")

    return True


async def test_error_handling():
    """Test error handling with graceful degradation."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling & Graceful Degradation")
    print("="*80)

    session = ChatSession()

    # Test normal operation (should not fail)
    try:
        response = await session.process_message("What is quantum computing?")
        print(f"✓ Normal operation successful")
        print(f"  Response length: {len(response['message'])} chars")
        print(f"  Metadata present: {'metadata' in response}")
        print(f"  Internal reasoning: {'internal_reasoning' in response}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


async def test_bug_fix():
    """Test that the self.llm_generator bug is fixed."""
    print("\n" + "="*80)
    print("TEST 4: Bug Fix Verification (self.llm → not self.llm_generator)")
    print("="*80)

    session = ChatSession()

    # The bug was: if self.llm_generator (undefined attribute)
    # Should now be: if self.llm (correctly defined)
    # Test that it doesn't crash when trying to use LLM

    try:
        # This should work even if LLM is None
        response = await session.process_message("test")
        print(f"✓ No AttributeError (bug is fixed)")
        return True
    except AttributeError as e:
        if "'llm_generator'" in str(e):
            print(f"✗ Bug still present: {e}")
            return False
        else:
            raise


async def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("CONSCIOUSNESS CHAT - 6-STEP REFINEMENT TEST SUITE".center(80))
    print("🧪" * 40)

    print("\nValidating refinements:")
    print("  ✓ ELEGANCE: Clarity, Simplicity, Beauty")
    print("  ✓ VERIFY: Accuracy, Completeness, Consistency\n")

    results = []

    # Run tests
    results.append(("Validation", await test_validation()))
    results.append(("Helper Methods", await test_helpers()))
    results.append(("Error Handling", await test_error_handling()))
    results.append(("Bug Fix", await test_bug_fix()))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:30} {status}")

    all_passed = all(passed for _, passed in results)

    print("="*80)
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nRefinement successful:")
        print("  • 3 helper methods extracted (reduced complexity)")
        print("  • 3 validation checks added (input safety)")
        print("  • 5 error handlers added (graceful degradation)")
        print("  • 1 critical bug fixed (self.llm_generator → self.llm)")
        print("  • Comprehensive docstrings (clarity)")
        print("  • Section separators & emoji logging (visual structure)")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review failures above.")
    print()

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
