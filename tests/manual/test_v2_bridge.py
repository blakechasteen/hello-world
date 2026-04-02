"""
Smoke test for v2_bridge.py — CognitiveBridge.

Verifies:
  1. Bridge initializes with empty memory
  2. store_and_index creates graph nodes + entity edges
  3. enhanced_recall returns enriched results with confidence + KS flags
  4. format_enhanced_result produces readable output
  5. CBR starts accumulating cases across queries
  6. Multiple queries show cross-query learning
"""

import asyncio
import sys

from hololoom.memory.v2_bridge import (
    CognitiveBridge,
    EnhancedRecallResult,
    format_enhanced_result,
)


class MockMemory:
    """Minimal mock that returns empty recalls."""
    def recall(self, query="", limit=5, **kw):
        return []


passed = 0
failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        print(f"  [+] {label}")
        passed += 1
    else:
        print(f"  [-] FAIL: {label}")
        failed += 1


async def test_bridge_init():
    """Bridge initializes with empty memory."""
    print("\n" + "=" * 72)
    print("  I. Bridge Initialization")
    print("=" * 72)

    bridge = CognitiveBridge(MockMemory())
    await bridge.initialize()

    check(bridge._initialized, "Bridge is initialized")
    check(bridge.navigator is not None, "Navigator created")
    check(bridge.confidence_estimator is not None, "ConfidenceEstimator created")
    check(len(bridge.bus._items) == 0, "Bus starts empty")

    report = bridge.report()
    check(report["initialized"], "Report shows initialized")
    check(report["graph_nodes"] == 0, "Report shows 0 nodes")

    return bridge


async def test_store_and_index(bridge):
    """store_and_index creates graph nodes with entity edges."""
    print("\n" + "=" * 72)
    print("  II. Store and Index")
    print("=" * 72)

    await bridge.store_and_index(
        "Thompson Sampling explores by sampling from posterior distributions",
        {"importance": 0.8}, ["ml"], ""
    )
    await bridge.store_and_index(
        "Epsilon-greedy explores with probability epsilon",
        {"importance": 0.7}, ["ml"], ""
    )
    await bridge.store_and_index(
        "Both Thompson and Epsilon are multi-armed bandit algorithms",
        {"importance": 0.8}, ["ml"], ""
    )
    await bridge.store_and_index(
        "UCB1 uses upper confidence bounds for exploration",
        {"importance": 0.6}, ["ml"], ""
    )
    await bridge.store_and_index(
        "Bayesian optimization extends Thompson Sampling to continuous spaces",
        {"importance": 0.7}, ["ml"], ""
    )

    report = bridge.report()
    check(report["graph_nodes"] > 5, f"Graph has nodes ({report['graph_nodes']})")
    check(report["graph_edges"] > 0, f"Graph has edges ({report['graph_edges']})")
    check(len(bridge.bus._aliases) > 0, f"Aliases registered ({len(bridge.bus._aliases)})")


async def test_enhanced_recall(bridge):
    """enhanced_recall returns enriched results."""
    print("\n" + "=" * 72)
    print("  III. Enhanced Recall")
    print("=" * 72)

    result = await bridge.enhanced_recall("Thompson Sampling vs epsilon-greedy", limit=5)

    check(isinstance(result, EnhancedRecallResult), "Returns EnhancedRecallResult")
    check(len(result.memories) > 0, f"Has memories ({len(result.memories)})")
    check(0 <= result.confidence <= 1, f"Confidence in range ({result.confidence:.2f})")
    check(result.decision in ("sufficient", "verify", "flag"), f"Valid decision ({result.decision})")
    check(result.seed_count >= 0, f"Has seed count ({result.seed_count})")
    check(isinstance(result.ppr_converged, bool), f"PPR converged flag ({result.ppr_converged})")
    check(result.node_count > 0, f"Node count > 0 ({result.node_count})")
    check(result.elapsed > 0, f"Elapsed > 0 ({result.elapsed:.4f}s)")
    check(len(result.context_text) > 0, "Context text not empty")

    # Check memory structure
    if result.memories:
        mem = result.memories[0]
        check("id" in mem, "Memory has id")
        check("text" in mem, "Memory has text")
        check("score" in mem, "Memory has score")
        check("memory_type" in mem, "Memory has memory_type")

    # TMS justifications (may or may not fire depending on graph structure)
    check(isinstance(result.justifications, list), "Justifications is a list")
    check(isinstance(result.retractions, list), "Retractions is a list")
    check(isinstance(result.cbr_hints, list), "CBR hints is a list")

    return result


async def test_format_output(result):
    """format_enhanced_result produces readable output."""
    print("\n" + "=" * 72)
    print("  IV. Format Output")
    print("=" * 72)

    output = format_enhanced_result(result, limit=5)
    check(isinstance(output, str), "Output is string")
    check(len(output) > 50, f"Output has content ({len(output)} chars)")
    check("v2 Cognitive Analysis" in output, "Contains cognitive analysis section")
    check("Confidence:" in output, "Contains confidence")
    check("Seeds:" in output, "Contains seed info")

    print("\n  --- Sample Output ---")
    for line in output.split("\n")[:15]:
        print(f"  | {line}")
    print("  --- End Sample ---")


async def test_cross_query_learning(bridge):
    """Multiple queries show CBR accumulation."""
    print("\n" + "=" * 72)
    print("  V. Cross-Query Learning (CBR)")
    print("=" * 72)

    # Run several queries to build CBR cases
    queries = [
        "What is Thompson Sampling?",
        "How does epsilon-greedy work?",
        "Compare bandit algorithms",
        "UCB1 exploration strategy",
        "Bayesian optimization",
    ]

    for q in queries:
        await bridge.enhanced_recall(q, limit=3)

    # Check CBR state
    cbr = None
    for src in bridge.sources._sources:
        if src.name == "cbr":
            cbr = src
            break

    if cbr:
        check(cbr.case_count > 0, f"CBR has cases ({cbr.case_count})")
        cbr_report = cbr.report()
        check(cbr_report["total_cases"] > 0, f"CBR report shows cases ({cbr_report['total_cases']})")
        check(cbr_report["total_retrievals"] >= 0, f"CBR total retrievals ({cbr_report['total_retrievals']})")
    else:
        check(False, "CBR source not found in registry")

    # Check ChunkCompiler state
    cc = None
    for src in bridge.sources._sources:
        if src.name == "chunk_compiler":
            cc = src
            break

    if cc:
        cc_report = cc.report()
        check(cc_report["experience_buffer_size"] > 0, f"ChunkCompiler has experience ({cc_report['experience_buffer_size']})")
    else:
        check(False, "ChunkCompiler source not found in registry")


async def test_empty_recall(bridge):
    """Recall on query with no matching seeds."""
    print("\n" + "=" * 72)
    print("  VI. Edge Case: No Seeds")
    print("=" * 72)

    result = await bridge.enhanced_recall("xyzzy foobar nonsense", limit=5)
    check(isinstance(result, EnhancedRecallResult), "Returns result even with no seeds")
    check(result.confidence >= 0, f"Confidence >= 0 ({result.confidence:.2f})")


async def main():
    bridge = await test_bridge_init()
    await test_store_and_index(bridge)
    result = await test_enhanced_recall(bridge)
    await test_format_output(result)
    await test_cross_query_learning(bridge)
    await test_empty_recall(bridge)

    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
