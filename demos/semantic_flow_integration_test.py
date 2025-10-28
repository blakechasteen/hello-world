"""
Semantic Flow Integration Test

Tests the end-to-end integration of semantic calculus with the full HoloLoom pipeline:
- Pattern card selection (SEMANTIC_FLOW)
- ResonanceShed with semantic thread
- DotPlasma with semantic features
- WeavingShuttle full cycle
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.loom.command import PatternCard


def create_test_shards():
    """Create simple test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is an algorithm for multi-armed bandits",
        ),
        MemoryShard(
            id="shard_2",
            text="The algorithm balances exploration and exploitation",
        ),
    ]


async def test_semantic_flow_pattern():
    """Test SEMANTIC_FLOW pattern card integration."""
    print("\n" + "=" * 70)
    print("TEST: Semantic Flow Pattern Integration")
    print("=" * 70)

    # Create config and test data
    config = Config.fast()
    shards = create_test_shards()

    # Create shuttle
    print("\n[1/4] Creating WeavingShuttle with SEMANTIC_FLOW pattern...")
    shuttle = WeavingShuttle(
        cfg=config,
        shards=shards,
        pattern_preference=PatternCard.SEMANTIC_FLOW,
        enable_reflection=False
    )

    # Create test query
    query = Query(text="How does Thompson Sampling balance exploration and exploitation?")
    print(f"\n[2/4] Query: '{query.text}'")

    # Weave with semantic flow
    print("\n[3/4] Weaving with SEMANTIC_FLOW pattern...")
    try:
        spacetime = await shuttle.weave(query)

        # Verify semantic flow features are present
        print("\n[4/4] Verifying semantic flow features...")
        
        # Check if DotPlasma has semantic_flow
        if hasattr(spacetime, 'trace') and spacetime.trace:
            plasma = spacetime.trace.dot_plasma
            
            if plasma and 'semantic_flow' in plasma:
                semantic_features = plasma['semantic_flow']
                print("\n  SUCCESS: Semantic flow features found!")
                print(f"  - Number of words: {semantic_features['n_states']}")
                print(f"  - Avg velocity: {semantic_features['avg_velocity']:.4f}")
                print(f"  - Avg acceleration: {semantic_features['avg_acceleration']:.4f}")
                print(f"  - Total distance: {semantic_features['total_distance']:.4f}")
                
                # Verify thread info
                threads = plasma.get('threads', [])
                semantic_thread = next((t for t in threads if t['name'] == 'semantic_flow'), None)
                if semantic_thread:
                    print(f"\n  Semantic thread metadata:")
                    print(f"  - Weight: {semantic_thread['weight']}")
                    print(f"  - n_words: {semantic_thread['metadata']['n_words']}")
                    print(f"  - avg_speed: {semantic_thread['metadata']['avg_speed']:.4f}")
                
                return True
            else:
                print("\n  FAILED: No semantic_flow in DotPlasma")
                print(f"  Available keys: {list(plasma.keys())}")
                return False
        else:
            print("\n  FAILED: No trace found in spacetime")
            return False

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pattern_card_selection():
    """Test that pattern card correctly enables semantic calculus."""
    print("\n" + "=" * 70)
    print("TEST: Pattern Card Selection")
    print("=" * 70)

    from HoloLoom.loom.command import LoomCommand, SEMANTIC_FLOW_PATTERN

    loom = LoomCommand()

    # Test explicit selection
    print("\n[1/2] Testing explicit SEMANTIC_FLOW selection...")
    pattern = loom.select_pattern(
        query_text="test query",
        user_preference="semantic_flow"
    )

    print(f"  Pattern: {pattern.name}")
    print(f"  Semantic flow enabled: {pattern.enable_semantic_flow}")
    print(f"  Semantic dimensions: {pattern.semantic_dimensions}")
    print(f"  Semantic trajectory: {pattern.semantic_trajectory}")
    print(f"  Semantic ethics: {pattern.semantic_ethics}")

    if pattern.enable_semantic_flow:
        print("\n  SUCCESS: SEMANTIC_FLOW pattern correctly configured!")
        return True
    else:
        print("\n  FAILED: semantic_flow not enabled")
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "   SEMANTIC FLOW INTEGRATION TEST SUITE".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")

    results = {}

    # Test 1: Pattern card selection
    results['pattern_selection'] = await test_pattern_card_selection()

    # Test 2: End-to-end integration
    results['end_to_end'] = await test_semantic_flow_pattern()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
