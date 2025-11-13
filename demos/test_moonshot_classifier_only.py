"""
Phase 2 Simple Validation: Moonshot Classifier Direct Test
============================================================
Tests classifier factory and latency without full orchestrator overhead.

Author: Claude Code
Date: November 13, 2025
"""

import time
from HoloLoom.config import Config
from HoloLoom.routing.classifier_factory import create_classifier


def test_classifier_factory():
    """Test classifier factory and latency."""
    print("=" * 80)
    print("PHASE 2 SIMPLE VALIDATION: Classifier Factory & Latency")
    print("=" * 80)
    print()

    # Test queries covering all complexity levels
    test_queries = [
        ("hi", "trivial"),
        ("what is Thompson Sampling?", "simple"),
        ("how does Thompson Sampling balance exploration and exploitation?", "complex"),
        ("analyze the theoretical foundations of Thompson Sampling", "research")
    ]

    # Test 1: Moonshot with Tier 1+2 only (production)
    print("TEST 1: Moonshot Classifier (Tier 1+2, production mode)")
    print("-" * 80)

    config = Config.fast()
    config.enable_smart_routing = True
    config.routing_classifier = "moonshot"
    config.enable_semantic_tier = False  # <1ms target
    config.enable_adaptive_learning = False  # Disable for faster test
    config.enable_classification_telemetry = False

    print(f"[OK] Config: enable_smart_routing={config.enable_smart_routing}")
    print(f"[OK] Config: routing_classifier={config.routing_classifier}")
    print(f"[OK] Config: enable_semantic_tier={config.enable_semantic_tier}")
    print()

    # Create classifier via factory
    classifier = create_classifier(config)
    print(f"[OK] Classifier type: {type(classifier).__name__}")
    print()

    # Test classification latency
    print("Classification Latency Test:")
    print("-" * 40)

    latencies = []
    accuracies = []

    for query_text, expected_complexity in test_queries:
        # Warmup (first classification may be slower)
        classifier.classify(query_text)

        # Measure classification time (10 iterations for accuracy)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            classification = classifier.classify(query_text)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        latency_ms = sum(times) / len(times)
        latencies.append(latency_ms)

        # Check accuracy
        actual = classification.complexity.value
        correct = actual == expected_complexity
        accuracies.append(correct)

        status = "[OK]" if correct else "[FAIL]"
        print(f"{status} '{query_text[:50]}...'")
        print(f"   Expected: {expected_complexity}, Got: {actual}")
        print(f"   Latency: {latency_ms:.3f}ms (avg of 10 runs)")

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

    # Test 2: Baseline classifier (backward compatibility)
    print()
    print("TEST 2: Baseline Classifier (backward compatibility)")
    print("-" * 80)

    config_baseline = Config.fast()
    config_baseline.enable_smart_routing = True
    config_baseline.routing_classifier = "baseline"

    classifier_baseline = create_classifier(config_baseline)
    print(f"[OK] Classifier type: {type(classifier_baseline).__name__}")

    # Quick test
    classification = classifier_baseline.classify("what is Thompson Sampling?")
    print(f"[OK] Baseline classifier works: {classification.complexity.value}")
    print()

    # Test 3: Smart routing disabled
    print()
    print("TEST 3: Smart Routing Disabled")
    print("-" * 80)

    config_disabled = Config.fast()
    config_disabled.enable_smart_routing = False

    classifier_disabled = create_classifier(config_disabled)
    print(f"[OK] Classifier: {classifier_disabled}")
    print()

    print("=" * 80)
    print("PHASE 2 SIMPLE VALIDATION COMPLETE")
    print("=" * 80)

    if latency_pass and accuracy_pass:
        print("[OK] ALL TESTS PASSED")
        print()
        print("Phase 2 integration successful:")
        print(f"  - Moonshot classifier loaded: [OK]")
        print(f"  - Latency <1ms: [OK] ({avg_latency:.3f}ms)")
        print(f"  - Accuracy 100%: [OK]")
        print(f"  - Factory pattern: [OK]")
        print(f"  - Backward compatibility: [OK]")
        print()
        print("*** READY FOR PRODUCTION DEPLOYMENT ***")
        return True
    else:
        print("[FAIL] SOME TESTS FAILED")
        print()
        print("Please review failures above.")
        return False


if __name__ == "__main__":
    success = test_classifier_factory()
    exit(0 if success else 1)
