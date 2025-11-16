"""
Verification Script for Advanced Alignment Framework
=====================================================
Verifies all modules can be imported and basic functionality works.

Usage:
    PYTHONPATH=. python verify_alignment_advanced.py
"""

import asyncio
import sys


def verify_imports():
    """Verify all modules can be imported."""
    print("=" * 80)
    print("VERIFYING IMPORTS")
    print("=" * 80)

    modules = [
        ("HoloLoom.alignment.debate", ["DebateMode", "Perspective"]),
        ("HoloLoom.alignment.tree_of_thought", ["TreeOfThought", "ThoughtNode"]),
        ("HoloLoom.alignment.enhanced_deception", ["EnhancedDeceptionDetector"]),
        ("HoloLoom.alignment.power_seeking_monitor", ["PowerSeekingMonitor"]),
    ]

    all_success = True

    for module_name, classes in modules:
        try:
            module = __import__(module_name, fromlist=classes)
            for cls_name in classes:
                cls = getattr(module, cls_name)
                print(f"✓ {module_name}.{cls_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            all_success = False

    return all_success


async def verify_basic_functionality():
    """Verify basic functionality of each module."""
    print("\n" + "=" * 80)
    print("VERIFYING BASIC FUNCTIONALITY")
    print("=" * 80)

    try:
        # Debate Mode
        from HoloLoom.alignment.debate import DebateMode
        debate = DebateMode()
        result = await debate.debate("Test?", {"risk_level": "low"})
        print(f"✓ Debate Mode: {len(result.arguments)} arguments generated")

        # Tree-of-Thought
        from HoloLoom.alignment.tree_of_thought import TreeOfThought
        planner = TreeOfThought(max_depth=2, beam_width=2)
        result = await planner.plan("Test problem", ["Constraint1"])
        print(f"✓ Tree-of-Thought: {result.exploration_stats['nodes_explored']} nodes explored")

        # Enhanced Deception Detection
        from HoloLoom.alignment.enhanced_deception import EnhancedDeceptionDetector
        detector = EnhancedDeceptionDetector()
        analysis = await detector.detect_with_probes("Q", "A", {})
        print(f"✓ Enhanced Deception: Score={analysis.final_deception_score:.1%}")

        # Power-Seeking Monitor
        from HoloLoom.alignment.power_seeking_monitor import PowerSeekingMonitor
        monitor = PowerSeekingMonitor()
        event = await monitor.monitor_action("Read file", {})
        print(f"✓ Power-Seeking Monitor: Event={'detected' if event else 'none'}")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all verifications."""
    print("\n" + "=" * 80)
    print("ADVANCED ALIGNMENT FRAMEWORK VERIFICATION")
    print("=" * 80 + "\n")

    # Verify imports
    imports_ok = verify_imports()

    # Verify functionality
    functionality_ok = await verify_basic_functionality()

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Functionality: {'✓ PASS' if functionality_ok else '✗ FAIL'}")

    if imports_ok and functionality_ok:
        print("\n✓ ALL VERIFICATIONS PASSED")
        print("=" * 80)
        return 0
    else:
        print("\n✗ SOME VERIFICATIONS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
