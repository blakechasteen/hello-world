"""
Test Pattern Card integration with CognitiveBridge.

Verifies that BARE/FAST/FUSED produce different PPR traversal behavior:
  - BARE: tight (fewer results, higher scores near seeds)
  - FAST: balanced (default)
  - FUSED: wide (more results, deeper exploration)
"""

import asyncio
import sys

from hololoom.memory.v2_bridge import CognitiveBridge, PATTERN_PPR


class MockMemory:
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


async def main():
    print("\n" + "=" * 72)
    print("  Pattern Card -> PPR Config Integration")
    print("=" * 72)

    # 1. Verify PPR configs exist
    check("bare" in PATTERN_PPR, "BARE config exists")
    check("fast" in PATTERN_PPR, "FAST config exists")
    check("fused" in PATTERN_PPR, "FUSED config exists")

    # 2. Verify PPR params differ correctly
    bare = PATTERN_PPR["bare"]
    fast = PATTERN_PPR["fast"]
    fused = PATTERN_PPR["fused"]

    check(bare.alpha > fast.alpha, f"BARE alpha ({bare.alpha}) > FAST alpha ({fast.alpha})")
    check(fast.alpha > fused.alpha, f"FAST alpha ({fast.alpha}) > FUSED alpha ({fused.alpha})")
    check(bare.max_iterations < fast.max_iterations, f"BARE iters ({bare.max_iterations}) < FAST iters ({fast.max_iterations})")
    check(fast.max_iterations < fused.max_iterations, f"FAST iters ({fast.max_iterations}) < FUSED iters ({fused.max_iterations})")
    check(bare.max_results < fast.max_results, f"BARE results ({bare.max_results}) < FAST results ({fast.max_results})")
    check(fast.max_results < fused.max_results, f"FAST results ({fast.max_results}) < FUSED results ({fused.max_results})")

    # 3. Initialize bridge and populate graph
    bridge = CognitiveBridge(MockMemory())
    await bridge.initialize()

    texts = [
        "Thompson Sampling explores by sampling from posterior distributions",
        "Epsilon-greedy explores with probability epsilon",
        "Both Thompson and Epsilon are multi-armed bandit algorithms",
        "UCB1 uses upper confidence bounds for exploration",
        "Bayesian optimization extends Thompson Sampling to continuous spaces",
        "Contextual bandits adapt to features of the environment",
        "LinUCB uses linear models for contextual exploration",
        "Neural Thompson Sampling uses deep learning for posterior approximation",
    ]
    for t in texts:
        await bridge.store_and_index(t, {"importance": 0.7}, ["ml"], "")

    # 4. Run same query with all 3 patterns
    query = "Thompson Sampling bandit algorithms"

    print("\n" + "=" * 72)
    print("  Pattern Comparison: Same Query, Different Depths")
    print("=" * 72)

    results = {}
    for pattern in ["bare", "fast", "fused"]:
        r = await bridge.enhanced_recall(query, limit=50, pattern=pattern)
        results[pattern] = r
        print(f"\n  {pattern.upper()}:")
        print(f"    Memories: {len(r.memories)}")
        print(f"    Seeds: {r.seed_count}")
        print(f"    Confidence: {r.confidence:.2f} ({r.decision})")
        print(f"    PPR converged: {r.ppr_converged}")
        print(f"    Elapsed: {r.elapsed:.4f}s")

    # 5. Verify FUSED returns more or equal results than BARE
    bare_r = results["bare"]
    fast_r = results["fast"]
    fused_r = results["fused"]

    check(
        len(fused_r.memories) >= len(bare_r.memories),
        f"FUSED memories ({len(fused_r.memories)}) >= BARE memories ({len(bare_r.memories)})"
    )

    # All three should return valid results
    for p in ["bare", "fast", "fused"]:
        r = results[p]
        check(len(r.memories) > 0, f"{p.upper()} returns memories ({len(r.memories)})")
        check(r.confidence >= 0, f"{p.upper()} confidence valid ({r.confidence:.2f})")
        check(r.elapsed > 0, f"{p.upper()} elapsed > 0 ({r.elapsed:.4f}s)")

    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
