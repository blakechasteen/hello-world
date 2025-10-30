#!/usr/bin/env python3
"""
Test Pattern Card System
=========================

Tests for the PatternCard loader and configuration system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.loom.card_loader import PatternCard


def test_card_loading():
    """Test loading built-in cards."""
    print("=" * 70)
    print("TEST 1: Card Loading")
    print("=" * 70)

    # Load bare card
    print("\n[1/3] Loading BARE card...")
    bare = PatternCard.load("bare")
    print(f"  ✓ Loaded: {bare.display_name}")
    print(f"    Semantic calculus: {bare.math_capabilities.semantic_calculus.get('enabled', False)}")
    print(f"    Embedding scales: {bare.math_capabilities.spectral_embedding.get('scales', [])}")
    print(f"    Tools: {len(bare.tools_config.enabled)} enabled")

    # Load fast card
    print("\n[2/3] Loading FAST card...")
    fast = PatternCard.load("fast")
    print(f"  ✓ Loaded: {fast.display_name}")
    print(f"    Semantic calculus: {fast.math_capabilities.semantic_calculus.get('enabled', False)}")
    print(f"    Dimensions: {fast.math_capabilities.semantic_calculus.get('config', {}).get('dimensions', 0)}")
    print(f"    Embedding scales: {fast.math_capabilities.spectral_embedding.get('scales', [])}")
    print(f"    Tools: {len(fast.tools_config.enabled)} enabled")

    # Load fused card
    print("\n[3/3] Loading FUSED card...")
    fused = PatternCard.load("fused")
    print(f"  ✓ Loaded: {fused.display_name}")
    print(f"    Extends: {fused.extends}")
    print(f"    Semantic dimensions: {fused.math_capabilities.semantic_calculus.get('config', {}).get('dimensions', 0)}")
    print(f"    Embedding scales: {fused.math_capabilities.spectral_embedding.get('scales', [])}")
    print(f"    Tools: {len(fused.tools_config.enabled)} enabled")
    print(f"    Memory backend: {fused.memory_config.backend}")

    print("\n[PASS] All cards loaded successfully!")
    return True


def test_card_inheritance():
    """Test card inheritance (extends)."""
    print("\n" + "=" * 70)
    print("TEST 2: Card Inheritance")
    print("=" * 70)

    # Load fused (extends fast)
    print("\n[1/2] Testing FUSED extends FAST...")
    fast = PatternCard.load("fast")
    fused = PatternCard.load("fused")

    # Check inheritance
    print("  Checking inherited values...")

    # Should inherit motif patterns from fast
    fast_patterns = fast.math_capabilities.motif_detection.get('patterns', [])
    fused_patterns = fused.math_capabilities.motif_detection.get('patterns', [])
    print(f"    Fast patterns: {len(fast_patterns)}")
    print(f"    Fused patterns: {len(fused_patterns)}")

    # Should override dimensions
    fast_dims = fast.math_capabilities.semantic_calculus.get('config', {}).get('dimensions', 0)
    fused_dims = fused.math_capabilities.semantic_calculus.get('config', {}).get('dimensions', 0)
    print(f"    Fast dimensions: {fast_dims}")
    print(f"    Fused dimensions: {fused_dims}")
    assert fused_dims > fast_dims, "Fused should have more dimensions!"

    # Should override backend
    fast_backend = fast.memory_config.backend
    fused_backend = fused.memory_config.backend
    print(f"    Fast backend: {fast_backend}")
    print(f"    Fused backend: {fused_backend}")
    assert fused_backend != fast_backend, "Fused should override backend!"

    print("\n[PASS] Inheritance working correctly!")
    return True


def test_semantic_config_conversion():
    """Test converting card to SemanticCalculusConfig."""
    print("\n" + "=" * 70)
    print("TEST 3: SemanticCalculusConfig Conversion")
    print("=" * 70)

    cards = ["bare", "fast", "fused"]

    for card_name in cards:
        print(f"\n[{cards.index(card_name) + 1}/{len(cards)}] Testing {card_name.upper()} card...")
        card = PatternCard.load(card_name)

        # Try to convert to SemanticCalculusConfig
        sem_config = card.math_capabilities.to_semantic_config()

        if sem_config is None:
            print(f"  ✓ Semantic calculus disabled (expected for {card_name})")
        else:
            print(f"  ✓ SemanticCalculusConfig created:")
            print(f"    Dimensions: {sem_config.dimensions}")
            print(f"    Cache enabled: {sem_config.enable_cache}")
            print(f"    Cache size: {sem_config.cache_size}")
            print(f"    Compute trajectory: {sem_config.compute_trajectory}")
            print(f"    Compute ethics: {sem_config.compute_ethics}")
            print(f"    Ethical framework: {sem_config.ethical_framework}")

    print("\n[PASS] Config conversion working!")
    return True


def test_runtime_overrides():
    """Test runtime overrides."""
    print("\n" + "=" * 70)
    print("TEST 4: Runtime Overrides")
    print("=" * 70)

    print("\n[1/2] Loading FAST with overrides...")
    overrides = {
        'math': {
            'semantic_calculus': {
                'config': {
                    'dimensions': 64  # Override to 64
                }
            }
        }
    }

    fast = PatternCard.load("fast", overrides=overrides)

    dims = fast.math_capabilities.semantic_calculus.get('config', {}).get('dimensions', 0)
    print(f"  Dimensions after override: {dims}")
    assert dims == 64, f"Override failed! Expected 64, got {dims}"
    print("  ✓ Override applied successfully")

    print("\n[2/2] Loading FUSED with overrides...")
    overrides2 = {
        'performance': {
            'target_latency_ms': 5000
        }
    }

    fused = PatternCard.load("fused", overrides=overrides2)

    latency = fused.performance_profile.target_latency_ms
    print(f"  Target latency after override: {latency}ms")
    assert latency == 5000, f"Override failed! Expected 5000, got {latency}"
    print("  ✓ Override applied successfully")

    print("\n[PASS] Runtime overrides working!")
    return True


def test_tools_config():
    """Test tools configuration."""
    print("\n" + "=" * 70)
    print("TEST 5: Tools Configuration")
    print("=" * 70)

    cards = ["bare", "fast", "fused"]

    for card_name in cards:
        print(f"\n[{cards.index(card_name) + 1}/{len(cards)}] Testing {card_name.upper()} tools...")
        card = PatternCard.load(card_name)

        print(f"  Enabled tools: {', '.join(card.tools_config.enabled)}")
        print(f"  Disabled tools: {', '.join(card.tools_config.disabled) if card.tools_config.disabled else 'none'}")

        # Test tool checking
        if "summarize" in card.tools_config.enabled:
            print(f"  ✓ 'summarize' is enabled")
            assert card.tools_config.is_tool_enabled("summarize")

        if "deep_research" in card.tools_config.disabled:
            print(f"  ✓ 'deep_research' is disabled")
            assert not card.tools_config.is_tool_enabled("deep_research")

    print("\n[PASS] Tools configuration working!")
    return True


def test_card_dict_roundtrip():
    """Test converting card to dict and back."""
    print("\n" + "=" * 70)
    print("TEST 6: Dict Roundtrip")
    print("=" * 70)

    print("\n[1/2] Converting card to dict...")
    card = PatternCard.load("fast")
    card_dict = card.to_dict()

    print(f"  ✓ Converted to dict ({len(card_dict)} keys)")
    print(f"    Keys: {', '.join(card_dict.keys())}")

    print("\n[2/2] Reconstructing from dict...")
    card2 = PatternCard.from_dict(card_dict)

    print(f"  ✓ Reconstructed card: {card2.display_name}")
    assert card2.name == card.name
    assert card2.version == card.version

    print("\n[PASS] Dict roundtrip working!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("PATTERN CARD SYSTEM TESTS")
    print("=" * 70)

    tests = [
        test_card_loading,
        test_card_inheritance,
        test_semantic_config_conversion,
        test_runtime_overrides,
        test_tools_config,
        test_card_dict_roundtrip,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, result))
        except Exception as e:
            print(f"\n[FAIL] {test_fn.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
