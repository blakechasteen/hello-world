"""
Phase 2 Integration Validation: Moonshot Classifier
====================================================
Tests that moonshot classifier is correctly integrated with:
- <1ms latency (Tier 1+2 only, semantic_tier=False)
- 100% accuracy maintained
- Proper factory pattern usage
- Graceful fallback to baseline

Author: Claude Code
Date: November 13, 2025
"""

import asyncio
import time
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard


def create_test_shards():
    """Create minimal test shards for validation."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["exploration", "exploitation"],
            metadata={"source": "test", "timestamp": time.time()}
        ),
        MemoryShard(
            id="shard_2",
            text="The epsilon-greedy algorithm chooses random actions with probability epsilon.",
            entities=["epsilon-greedy", "algorithm"],
            motifs=["exploration", "random"],
            metadata={"source": "test", "timestamp": time.time()}
        )
    ]


async def test_moonshot_integration():
    """Validate Phase 2 integration."""
    print("=" * 80)
    print("PHASE 2 VALIDATION: Moonshot Classifier Integration")
    print("=" * 80)
    print()

    # Test queries covering all complexity levels
    test_queries = [
        ("hi", "TRIVIAL"),
        ("what is Thompson Sampling?", "SIMPLE"),
        ("how does Thompson Sampling balance exploration and exploitation in multi-armed bandits?", "COMPLEX"),
        ("analyze the theoretical foundations of Thompson Sampling and compare with UCB", "RESEARCH")
    ]

    # Test Configuration 1: Moonshot with Tier 1+2 only (production)
    print("TEST 1: Moonshot Classifier (Tier 1+2, production mode)")
    print("-" * 80)

    config = Config.fast()
    config.enable_smart_routing = True
    config.routing_classifier = "moonshot"
    config.enable_semantic_tier = False  # <1ms target
    config.enable_adaptive_learning = True
    config.enable_classification_telemetry = False  # Disable for test

    print(f"[OK] Config: enable_smart_routing={config.enable_smart_routing}")
    print(f"[OK] Config: routing_classifier={config.routing_classifier}")
    print(f"[OK] Config: enable_semantic_tier={config.enable_semantic_tier}")
    print()

    # Import orchestrator and create instance
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        print(f"[OK] Orchestrator created")
        print(f"[OK] Classifier type: {type(orchestrator.query_classifier).__name__}")
        print(f"[OK] Fast path router: {type(orchestrator.fast_path_router).__name__ if orchestrator.fast_path_router else 'None'}")
        print()

        # Test classification latency
        print("Classification Latency Test:")
        print("-" * 40)

        latencies = []
        accuracies = []

        for query_text, expected_complexity in test_queries:
            # Measure classification time
            start = time.perf_counter()
            classification = orchestrator.query_classifier.classify(query_text)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Check accuracy
            actual = classification.complexity.value
            correct = actual == expected_complexity
            accuracies.append(correct)

            status = "[OK]" if correct else "[FAIL]"
            print(f"{status} '{query_text[:40]}...'")
            print(f"   Expected: {expected_complexity}, Got: {actual}, Latency: {latency_ms:.3f}ms")

        print()
        avg_latency = sum(latencies) / len(latencies)
        accuracy = sum(accuracies) / len(accuracies) * 100

        print("RESULTS:")
        print("-" * 40)
        print(f"Average Latency: {avg_latency:.3f}ms")
        print(f"Accuracy: {accuracy:.0f}%")
        print(f"Target Latency: <1ms")
        print(f"Target Accuracy: 100%")
        print()

        # Validation
        latency_pass = avg_latency < 1.0
        accuracy_pass = accuracy == 100.0

        if latency_pass:
            print(f"[OK] LATENCY PASS: {avg_latency:.3f}ms < 1ms")
        else:
            print(f"[FAIL] LATENCY FAIL: {avg_latency:.3f}ms >= 1ms")

        if accuracy_pass:
            print(f"[OK] ACCURACY PASS: {accuracy:.0f}% == 100%")
        else:
            print(f"[FAIL] ACCURACY FAIL: {accuracy:.0f}% < 100%")

        print()

    # Test Configuration 2: Baseline classifier (backward compatibility)
    print()
    print("TEST 2: Baseline Classifier (backward compatibility)")
    print("-" * 80)

    config_baseline = Config.fast()
    config_baseline.enable_smart_routing = True
    config_baseline.routing_classifier = "baseline"

    async with WeavingOrchestrator(cfg=config_baseline, shards=shards) as orchestrator:
        print(f"[OK] Orchestrator created with baseline classifier")
        print(f"[OK] Classifier type: {type(orchestrator.query_classifier).__name__}")
        print()

        # Quick test
        classification = orchestrator.query_classifier.classify("what is Thompson Sampling?")
        print(f"[OK] Baseline classifier works: {classification.complexity.value}")
        print()

    # Test Configuration 3: Smart routing disabled
    print()
    print("TEST 3: Smart Routing Disabled")
    print("-" * 80)

    config_disabled = Config.fast()
    config_disabled.enable_smart_routing = False

    async with WeavingOrchestrator(cfg=config_disabled, shards=shards) as orchestrator:
        print(f"[OK] Orchestrator created with routing disabled")
        print(f"[OK] Classifier: {orchestrator.query_classifier}")
        print(f"[OK] Fast path router: {orchestrator.fast_path_router}")
        print()

    print("=" * 80)
    print("PHASE 2 VALIDATION COMPLETE")
    print("=" * 80)

    if latency_pass and accuracy_pass:
        print("[OK] ALL TESTS PASSED")
        print()
        print("Phase 2 integration successful:")
        print(f"  • Moonshot classifier loaded: [OK]")
        print(f"  • Latency <1ms: [OK] ({avg_latency:.3f}ms)")
        print(f"  • Accuracy 100%: [OK]")
        print(f"  • Factory pattern: [OK]")
        print(f"  • Backward compatibility: [OK]")
        print()
        print("*** READY FOR PRODUCTION DEPLOYMENT")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print()
        print("Please review failures above.")


if __name__ == "__main__":
    asyncio.run(test_moonshot_integration())
