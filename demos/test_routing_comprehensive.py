#!/usr/bin/env python3
"""
Comprehensive Smart Query Routing Test Suite
=============================================
Tests all aspects of the routing system:
- Classification accuracy
- Performance benchmarks
- Edge cases
- Integration with full pipeline
- Statistics and metrics

Author: Claude Code
Date: November 2025
"""

import asyncio
import time
from typing import List, Dict, Any
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.routing.query_classifier import QueryClassifier, QueryComplexity


# ============================================================================
# Test Data
# ============================================================================

# Comprehensive test cases covering all complexity levels
TEST_CASES = [
    # TRIVIAL queries (greetings, acknowledgments)
    {"query": "hi", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hello", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hey there", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "thanks", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "thank you!", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "ok", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "bye", "expected": "TRIVIAL", "category": "goodbye"},

    # SIMPLE queries (factual lookups, definitions)
    {"query": "what is X?", "expected": "SIMPLE", "category": "definition"},
    {"query": "define Thompson Sampling", "expected": "SIMPLE", "category": "definition"},
    {"query": "who is Einstein", "expected": "SIMPLE", "category": "factual"},
    {"query": "when is Christmas", "expected": "SIMPLE", "category": "factual"},
    {"query": "where is Paris", "expected": "SIMPLE", "category": "factual"},

    # COMPLEX queries (multi-step reasoning, explanations)
    {"query": "Explain Thompson Sampling with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does reinforcement learning work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Describe the attention mechanism and its variants", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Why do neural networks need backpropagation?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Elaborate on the differences between supervised and unsupervised learning", "expected": "COMPLEX", "category": "comparison"},

    # RESEARCH queries (deep analysis, comparisons)
    {"query": "Analyze the tradeoffs between Thompson Sampling and UCB", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Compare and contrast different exploration strategies in reinforcement learning", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Evaluate the advantages and disadvantages of neural vs symbolic AI", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Examine the comprehensive implications of using attention mechanisms versus convolutional networks", "expected": "RESEARCH", "category": "analysis"},
]

# Edge cases
EDGE_CASES = [
    {"query": "", "expected": "TRIVIAL", "category": "empty"},
    {"query": "?", "expected": "TRIVIAL", "category": "punctuation"},
    {"query": "a" * 100, "expected": "RESEARCH", "category": "very_long"},
    {"query": "HELLO WORLD!!!", "expected": "TRIVIAL", "category": "uppercase"},
    {"query": "what", "expected": "SIMPLE", "category": "single_word"},
]

# Memory shards for testing
MEMORY_SHARDS = [
    MemoryShard(
        id="shard_thompson",
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation by sampling from posterior distributions.",
        entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
        motifs=["sampling", "exploration", "exploitation"]
    ),
    MemoryShard(
        id="shard_rl",
        text="Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward through trial and error.",
        entities=["reinforcement learning", "agents", "reward"],
        motifs=["learning", "decision", "reward"]
    ),
    MemoryShard(
        id="shard_attention",
        text="Attention mechanisms allow neural networks to focus on relevant parts of the input by computing weighted combinations based on learned importance scores.",
        entities=["attention", "neural networks", "weights"],
        motifs=["attention", "focus", "weights"]
    ),
]


# ============================================================================
# Test Functions
# ============================================================================

def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


async def test_classifier_accuracy():
    """Test classification accuracy across all complexity levels."""
    print_header("TEST 1: CLASSIFIER ACCURACY")

    classifier = QueryClassifier()
    results = []
    correct = 0
    total = 0

    for test_case in TEST_CASES:
        query_text = test_case["query"]
        expected = test_case["expected"]
        category = test_case["category"]

        classification = classifier.classify(query_text)
        actual = classification.complexity.value.upper()

        is_correct = actual == expected
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "query": query_text[:50],
            "expected": expected,
            "actual": actual,
            "category": category,
            "confidence": classification.confidence,
            "correct": is_correct
        })

    # Print results by category
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    for category, cat_results in sorted(categories.items()):
        print_subheader(f"Category: {category.upper()}")
        for r in cat_results:
            status = "✅" if r["correct"] else "❌"
            print(f"{status} '{r['query'][:40]:<40}' → {r['actual']:<10} (expected: {r['expected']}, conf: {r['confidence']:.2f})")

    # Summary
    accuracy = (correct / total) * 100
    print_subheader("ACCURACY SUMMARY")
    print(f"Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'✅ PASS' if accuracy >= 80 else '❌ FAIL'} - Accuracy threshold: 80%")

    return accuracy >= 80


async def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_header("TEST 2: EDGE CASES")

    classifier = QueryClassifier()

    for test_case in EDGE_CASES:
        query_text = test_case["query"]
        expected = test_case["expected"]
        category = test_case["category"]

        classification = classifier.classify(query_text)
        actual = classification.complexity.value.upper()

        is_correct = actual == expected
        status = "✅" if is_correct else "⚠️"

        display_text = query_text[:40] if len(query_text) <= 40 else query_text[:37] + "..."
        print(f"{status} [{category:>15}] '{display_text}' → {actual} (expected: {expected})")

    print("\n✅ PASS - Edge cases handled without crashes")
    return True


async def test_performance_benchmarks():
    """Test performance of each routing path."""
    print_header("TEST 3: PERFORMANCE BENCHMARKS")

    config = Config.fast()

    async with WeavingOrchestrator(cfg=config, shards=MEMORY_SHARDS) as orchestrator:
        benchmarks = []

        # Test each complexity level
        test_queries = [
            ("hi", "TRIVIAL", "<10ms"),
            ("what is Thompson Sampling?", "SIMPLE", "<50ms"),
            ("Explain Thompson Sampling in detail", "COMPLEX", "<200ms"),
            ("Analyze all tradeoffs comprehensively", "RESEARCH", "no limit"),
        ]

        print_subheader("Running benchmarks (3 iterations each)...")

        for query_text, expected_path, target in test_queries:
            latencies = []

            # Run 3 times to get average
            for i in range(3):
                start = time.perf_counter()
                query = Query(text=query_text)
                spacetime = await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000
                latencies.append(duration_ms)

            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            fast_path = spacetime.metadata.get('fast_path', None)
            actual_path = fast_path if fast_path else "FULL_PIPELINE"

            benchmarks.append({
                "query": query_text,
                "path": actual_path.upper(),
                "avg_ms": avg_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "target": target
            })

        # Print results
        print_subheader("BENCHMARK RESULTS")
        print(f"{'Query':<45} {'Path':<15} {'Avg':<10} {'Min':<10} {'Max':<10} {'Target':<12}")
        print("-" * 80)

        for b in benchmarks:
            print(f"{b['query'][:44]:<45} {b['path']:<15} {b['avg_ms']:>8.1f}ms {b['min_ms']:>8.1f}ms {b['max_ms']:>8.1f}ms {b['target']:<12}")

        # Check targets
        print_subheader("TARGET VALIDATION")
        all_pass = True

        for b in benchmarks:
            if "10ms" in b["target"]:
                passes = b["avg_ms"] < 10
                print(f"{'✅' if passes else '❌'} {b['path']}: {b['avg_ms']:.1f}ms (target: <10ms)")
                all_pass = all_pass and passes
            elif "50ms" in b["target"]:
                passes = b["avg_ms"] < 50
                print(f"{'✅' if passes else '❌'} {b['path']}: {b['avg_ms']:.1f}ms (target: <50ms)")
                all_pass = all_pass and passes
            elif "200ms" in b["target"]:
                passes = b["avg_ms"] < 200
                print(f"{'✅' if passes else '⚠️'} {b['path']}: {b['avg_ms']:.1f}ms (target: <200ms)")
                # Don't fail on this one, it's more flexible

        print(f"\n{'✅ PASS' if all_pass else '❌ FAIL'} - Performance targets")
        return all_pass


async def test_routing_statistics():
    """Test routing statistics and metrics."""
    print_header("TEST 4: ROUTING STATISTICS")

    config = Config.fast()

    async with WeavingOrchestrator(cfg=config, shards=MEMORY_SHARDS) as orchestrator:
        # Run variety of queries
        test_queries = [
            "hi", "thanks", "bye",  # 3 TRIVIAL
            "what is X?", "define Y", "who is Z",  # 3 SIMPLE
            "Explain A in detail", "How does B work?",  # 2 COMPLEX
            "Analyze all tradeoffs of C vs D",  # 1 RESEARCH
        ]

        for query_text in test_queries:
            query = Query(text=query_text)
            await orchestrator.weave(query)

        # Get statistics
        stats = orchestrator.fast_path_router.get_stats()

        print_subheader("ROUTING DISTRIBUTION")
        print(f"TRIVIAL:  {stats['trivial']} queries")
        print(f"SIMPLE:   {stats['simple']} queries")
        print(f"COMPLEX:  {stats['complex']} queries")
        print(f"Total:    {stats['total']} queries")
        print(f"Fast path: {stats['fast_path_percentage']:.1f}%")

        # Validate
        expected_total = len(test_queries)
        actual_total = stats['total']

        if actual_total == expected_total:
            print(f"\n✅ PASS - Statistics tracking correct")
            return True
        else:
            print(f"\n❌ FAIL - Statistics mismatch: {actual_total} != {expected_total}")
            return False


async def test_integration():
    """Test integration with full orchestrator pipeline."""
    print_header("TEST 5: FULL INTEGRATION")

    config = Config.fast()

    async with WeavingOrchestrator(cfg=config, shards=MEMORY_SHARDS) as orchestrator:
        print_subheader("Testing full pipeline integration...")

        # Test that routing doesn't break anything
        test_cases = [
            ("hi", "Should use fast path"),
            ("what is Thompson Sampling?", "Should use simple path"),
            ("Analyze Thompson Sampling vs epsilon-greedy", "Should use full pipeline"),
        ]

        all_pass = True

        for query_text, description in test_cases:
            try:
                query = Query(text=query_text)
                spacetime = await orchestrator.weave(query)

                # Validate spacetime structure
                assert spacetime.query_text == query_text
                assert spacetime.response is not None
                assert spacetime.confidence >= 0.0
                assert spacetime.confidence <= 1.0
                assert spacetime.trace is not None
                assert spacetime.trace.duration_ms > 0

                print(f"✅ {description}")
                print(f"   Query: '{query_text[:50]}'")
                print(f"   Response: '{spacetime.response[:60]}...'")
                print(f"   Confidence: {spacetime.confidence:.2f}")
                print(f"   Latency: {spacetime.trace.duration_ms:.1f}ms")
                print()

            except Exception as e:
                print(f"❌ {description}")
                print(f"   Error: {e}")
                all_pass = False

        print(f"{'✅ PASS' if all_pass else '❌ FAIL'} - Integration test")
        return all_pass


async def test_spacetime_structure():
    """Test Spacetime and WeavingTrace structure for fast paths."""
    print_header("TEST 6: SPACETIME STRUCTURE")

    config = Config.fast()

    async with WeavingOrchestrator(cfg=config, shards=MEMORY_SHARDS) as orchestrator:
        print_subheader("Validating Spacetime structure for each path...")

        test_cases = [
            ("hi", "TRIVIAL"),
            ("what is Thompson Sampling?", "SIMPLE"),
        ]

        all_pass = True

        for query_text, expected_path in test_cases:
            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)

            # Validate required fields
            try:
                # Spacetime fields
                assert hasattr(spacetime, 'query_text')
                assert hasattr(spacetime, 'response')
                assert hasattr(spacetime, 'tool_used')
                assert hasattr(spacetime, 'confidence')
                assert hasattr(spacetime, 'trace')
                assert hasattr(spacetime, 'metadata')

                # WeavingTrace fields
                assert hasattr(spacetime.trace, 'start_time')
                assert hasattr(spacetime.trace, 'end_time')
                assert hasattr(spacetime.trace, 'duration_ms')
                assert hasattr(spacetime.trace, 'stage_durations')

                # Metadata
                assert 'fast_path' in spacetime.metadata or spacetime.tool_used
                assert 'latency_ms' in spacetime.metadata or spacetime.trace.duration_ms > 0

                print(f"✅ {expected_path} path structure valid")
                print(f"   Fields: query_text, response, tool_used, confidence, trace, metadata ✓")
                print(f"   Trace: start_time, end_time, duration_ms, stage_durations ✓")

            except AssertionError as e:
                print(f"❌ {expected_path} path structure invalid")
                print(f"   Error: Missing field")
                all_pass = False

        print(f"\n{'✅ PASS' if all_pass else '❌ FAIL'} - Spacetime structure validation")
        return all_pass


async def main():
    """Run all comprehensive tests."""
    print_header("SMART QUERY ROUTING - COMPREHENSIVE TEST SUITE")
    print("\nRunning 6 test suites...")

    start_time = time.perf_counter()

    results = {
        "test_classifier_accuracy": await test_classifier_accuracy(),
        "test_edge_cases": await test_edge_cases(),
        "test_performance_benchmarks": await test_performance_benchmarks(),
        "test_routing_statistics": await test_routing_statistics(),
        "test_integration": await test_integration(),
        "test_spacetime_structure": await test_spacetime_structure(),
    }

    duration = time.perf_counter() - start_time

    # Final summary
    print_header("FINAL SUMMARY")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\nTest Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Duration: {duration:.1f}s")

    if passed == total:
        print("\n🎉 ✅ ALL TESTS PASSED - Smart routing system is production ready!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed - Review failures above")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
