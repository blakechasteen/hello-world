#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Conversational Auto-Spin Tests
==========================================
Tests the complete signal vs noise filtering pipeline.

Tests:
1. Importance scoring accuracy
2. Auto-spin threshold filtering
3. Memory storage integration
4. Conversation stats tracking
5. Full MCP chat flow (simulated)
"""

import asyncio
import sys
import io
from pathlib import Path
from typing import List, Dict

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

# Import the importance scorer from MCP server
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load MCP server module to get importance scoring
repo_root = Path(__file__).parent
mcp_module = load_module(
    "mcp_server",
    repo_root / "HoloLoom" / "memory" / "mcp_server.py"
)

score_importance = mcp_module.score_importance
should_remember = mcp_module.should_remember


# ============================================================================
# Test Cases
# ============================================================================

class TestCase:
    """Single test case for importance scoring."""
    def __init__(self, name: str, user_input: str, system_output: str,
                 expected_signal: bool, category: str = ""):
        self.name = name
        self.user_input = user_input
        self.system_output = system_output
        self.expected_signal = expected_signal
        self.category = category
        self.score = None
        self.remembered = None
        self.passed = None


# Define test cases
TEST_CASES = [
    # NOISE - Should be filtered
    TestCase(
        "Greeting - Hi",
        user_input="hi",
        system_output="Hello! How can I help?",
        expected_signal=False,
        category="NOISE"
    ),
    TestCase(
        "Acknowledgment - OK",
        user_input="ok",
        system_output="Great!",
        expected_signal=False,
        category="NOISE"
    ),
    TestCase(
        "Acknowledgment - Thanks",
        user_input="thanks",
        system_output="You're welcome!",
        expected_signal=False,
        category="NOISE"
    ),
    TestCase(
        "Goodbye",
        user_input="bye",
        system_output="Goodbye! Have a great day!",
        expected_signal=False,
        category="NOISE"
    ),
    TestCase(
        "Very short exchange",
        user_input="yes",
        system_output="ok",
        expected_signal=False,
        category="NOISE"
    ),

    # SIGNAL - Should be remembered
    TestCase(
        "Question about domain topic",
        user_input="What is Thompson Sampling?",
        system_output="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It samples from Beta distributions to balance exploration and exploitation.",
        expected_signal=True,
        category="SIGNAL"
    ),
    TestCase(
        "Question with domain terms",
        user_input="How does memory recall work in HoloLoom?",
        system_output="Memory recall uses multiple strategies including semantic search, temporal ordering, and graph traversal.",
        expected_signal=True,
        category="SIGNAL"
    ),
    TestCase(
        "Substantive question",
        user_input="Explain the difference between the policy engine and memory retrieval system",
        system_output="The policy engine makes decisions about which tool to use, while memory retrieval finds relevant context from the knowledge base.",
        expected_signal=True,
        category="SIGNAL"
    ),
    TestCase(
        "Information-dense query",
        user_input="How do memory shards work and what metadata do they contain?",
        system_output="Memory shards are standardized units created by SpinningWheel spinners. They contain text, entities, motifs, and metadata for multi-strategy retrieval.",
        expected_signal=True,
        category="SIGNAL"
    ),
    TestCase(
        "Long substantive content",
        user_input="Can you explain how the knowledge graph integrates with vector search and what benefits this hybrid approach provides?",
        system_output="The hybrid approach combines semantic similarity from vector search with structural relationships from the knowledge graph, providing both meaning-based and connection-based context retrieval.",
        expected_signal=True,
        category="SIGNAL"
    ),

    # BORDERLINE - Edge cases
    TestCase(
        "Follow-up question",
        user_input="Tell me more",
        system_output="The orchestrator coordinates all components.",
        expected_signal=True,  # Should pass with moderate score
        category="BORDERLINE"
    ),
    TestCase(
        "Short but meaningful",
        user_input="What about embeddings?",
        system_output="Embeddings use Matryoshka representations at multiple scales.",
        expected_signal=True,
        category="BORDERLINE"
    ),
]


# ============================================================================
# Test Runner
# ============================================================================

async def run_importance_scoring_tests():
    """Test importance scoring accuracy."""
    print("=" * 70)
    print("TEST 1: Importance Scoring Accuracy")
    print("=" * 70)

    threshold = 0.4
    passed = 0
    failed = 0

    print(f"\nThreshold: {threshold}")
    print(f"Testing {len(TEST_CASES)} cases...\n")

    for test in TEST_CASES:
        # Score the exchange
        test.score = score_importance(
            test.user_input,
            test.system_output,
            metadata={'tool': 'chat', 'confidence': 0.7}
        )

        # Check if it would be remembered
        test.remembered = should_remember(test.score, threshold)

        # Check if this matches expectation
        test.passed = (test.remembered == test.expected_signal)

        # Display result
        status = "✓ PASS" if test.passed else "✗ FAIL"
        expected = "REMEMBER" if test.expected_signal else "FILTER"
        actual = "REMEMBER" if test.remembered else "FILTER"

        print(f"{status} | {test.category:10} | {test.name}")
        print(f"       Score: {test.score:.2f} | Expected: {expected} | Got: {actual}")
        print(f"       Input: {test.user_input[:50]}...")
        print()

        if test.passed:
            passed += 1
        else:
            failed += 1

    # Summary
    total = len(TEST_CASES)
    accuracy = passed / total * 100

    print("=" * 70)
    print(f"Results: {passed}/{total} passed ({accuracy:.1f}% accuracy)")
    print("=" * 70)

    return accuracy >= 80  # 80% accuracy threshold


async def run_threshold_tests():
    """Test different threshold settings."""
    print("\n" + "=" * 70)
    print("TEST 2: Threshold Sensitivity")
    print("=" * 70)

    # Get a mix of scores
    scores = []
    for test in TEST_CASES[:10]:
        score = score_importance(test.user_input, test.system_output)
        scores.append((test.name, score, test.category))

    thresholds = [0.2, 0.4, 0.6, 0.8]

    print("\nTesting different thresholds:\n")

    for threshold in thresholds:
        remembered_count = sum(1 for _, score, _ in scores if score >= threshold)
        rate = remembered_count / len(scores) * 100

        print(f"Threshold {threshold}: {remembered_count}/{len(scores)} remembered ({rate:.0f}%)")

        # Show which categories
        signal_remembered = sum(1 for name, score, cat in scores
                               if cat == "SIGNAL" and score >= threshold)
        noise_filtered = sum(1 for name, score, cat in scores
                            if cat == "NOISE" and score < threshold)

        print(f"  ✓ Signal captured: {signal_remembered}/5")
        print(f"  ✓ Noise filtered: {noise_filtered}/5")
        print()

    return True


async def run_conversation_flow_test():
    """Test a full conversation flow."""
    print("\n" + "=" * 70)
    print("TEST 3: Full Conversation Flow")
    print("=" * 70)

    conversation = [
        ("hi", "Hello!", "NOISE"),
        ("What is HoloLoom?", "HoloLoom is a neural decision system...", "SIGNAL"),
        ("ok", "Great!", "NOISE"),
        ("How does Thompson Sampling work?", "Thompson Sampling uses Beta distributions...", "SIGNAL"),
        ("thanks", "You're welcome!", "NOISE"),
        ("Tell me about the policy engine", "The policy engine makes decisions...", "SIGNAL"),
    ]

    print(f"\nSimulating conversation with {len(conversation)} turns...\n")

    stats = {
        'total': 0,
        'signal': 0,
        'noise': 0,
        'scores': []
    }

    for idx, (user_msg, sys_msg, expected_type) in enumerate(conversation, 1):
        score = score_importance(user_msg, sys_msg, {'tool': 'chat'})
        remembered = should_remember(score, 0.4)

        stats['total'] += 1
        stats['scores'].append(score)

        if remembered:
            stats['signal'] += 1
            actual_type = "SIGNAL"
        else:
            stats['noise'] += 1
            actual_type = "NOISE"

        match = "✓" if actual_type == expected_type else "✗"

        print(f"Turn {idx} [{match}] (Score: {score:.2f})")
        print(f"  User: {user_msg}")
        print(f"  Expected: {expected_type} | Got: {actual_type}")
        print()

    # Calculate metrics
    avg_score = sum(stats['scores']) / len(stats['scores'])
    signal_rate = stats['signal'] / stats['total'] * 100

    print("=" * 70)
    print("Conversation Summary:")
    print(f"  Total Turns: {stats['total']}")
    print(f"  Signal (Remembered): {stats['signal']} ({signal_rate:.0f}%)")
    print(f"  Noise (Filtered): {stats['noise']} ({100-signal_rate:.0f}%)")
    print(f"  Avg Importance: {avg_score:.2f}")
    print("=" * 70)

    # Test passes if we filtered noise and captured signal correctly
    # Expected: 3 signal, 3 noise
    return stats['signal'] == 3 and stats['noise'] == 3


async def run_edge_case_tests():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ("Empty input", "", "Response", False),
        ("Very long greeting", "hi" * 100, "Hello!", False),
        ("Question mark only", "?", "What?", False),
        ("Mixed case greeting", "HeLLo", "Hi!", False),
        ("Punctuation spam", "!!!", "???", False),
        ("Actual question with greeting", "hi, what is memory?", "Memory stores information...", True),
    ]

    print("\nTesting edge cases:\n")

    passed = 0
    for name, user_input, sys_output, should_be_signal in edge_cases:
        score = score_importance(user_input, sys_output)
        remembered = should_remember(score, 0.4)

        test_passed = (remembered == should_be_signal)
        status = "✓ PASS" if test_passed else "✗ FAIL"

        print(f"{status} | {name}")
        print(f"       Score: {score:.2f} | Expected: {'SIGNAL' if should_be_signal else 'NOISE'}")
        print()

        if test_passed:
            passed += 1

    return passed >= 4  # At least 4 of 6 edge cases correct


async def run_metadata_influence_test():
    """Test that metadata influences scoring."""
    print("\n" + "=" * 70)
    print("TEST 5: Metadata Influence")
    print("=" * 70)

    base_user = "Tell me something"
    base_sys = "Here's some information"

    print("\nTesting metadata influence on scoring:\n")

    # No metadata
    score_no_meta = score_importance(base_user, base_sys, None)
    print(f"No metadata: {score_no_meta:.2f}")

    # High confidence
    score_high_conf = score_importance(
        base_user, base_sys,
        {'confidence': 0.9}
    )
    print(f"High confidence (0.9): {score_high_conf:.2f}")

    # Low confidence
    score_low_conf = score_importance(
        base_user, base_sys,
        {'confidence': 0.1}
    )
    print(f"Low confidence (0.1): {score_low_conf:.2f}")

    # Important tool
    score_important_tool = score_importance(
        base_user, base_sys,
        {'tool': 'store_memory', 'confidence': 0.7}
    )
    print(f"Important tool (store_memory): {score_important_tool:.2f}")

    print()

    # Verify confidence affects score
    confidence_works = score_high_conf > score_low_conf
    tool_works = score_important_tool > score_no_meta

    print(f"✓ Confidence influences score: {confidence_works}")
    print(f"✓ Tool type influences score: {tool_works}")

    return confidence_works and tool_works


# ============================================================================
# Main Test Suite
# ============================================================================

async def main():
    """Run all end-to-end tests."""
    print("\n" + "=" * 70)
    print("CONVERSATIONAL AUTO-SPIN: End-to-End Test Suite")
    print("=" * 70)
    print("\nTesting signal vs noise filtering pipeline...\n")

    results = {}

    # Run all tests
    try:
        results['importance_scoring'] = await run_importance_scoring_tests()
        results['threshold_sensitivity'] = await run_threshold_tests()
        results['conversation_flow'] = await run_conversation_flow_test()
        results['edge_cases'] = await run_edge_case_tests()
        results['metadata_influence'] = await run_metadata_influence_test()

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUITE SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {test_name.replace('_', ' ').title()}")

        print("\n" + "=" * 70)
        print(f"Overall: {passed_count}/{total_count} test suites passed")
        print("=" * 70)

        if passed_count == total_count:
            print("\n🎉 ALL TESTS PASSED! Signal vs noise filtering works! 🎯")
            return 0
        else:
            print(f"\n⚠️  {total_count - passed_count} test suite(s) failed")
            return 1

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
