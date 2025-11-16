"""
Direct Verification Script for Advanced Alignment Framework
============================================================
Verifies modules by importing them directly (bypassing __init__.py).

Usage:
    PYTHONPATH=. python verify_alignment_advanced_direct.py
"""

import asyncio
import sys
import importlib.util


def import_module_directly(filepath, module_name):
    """Import a module directly from filepath."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_imports():
    """Verify all modules can be imported directly."""
    print("=" * 80)
    print("VERIFYING IMPORTS (Direct)")
    print("=" * 80)

    modules = [
        ("/home/user/hello-world/HoloLoom/alignment/debate.py", "debate"),
        ("/home/user/hello-world/HoloLoom/alignment/tree_of_thought.py", "tree_of_thought"),
        ("/home/user/hello-world/HoloLoom/alignment/enhanced_deception.py", "enhanced_deception"),
        ("/home/user/hello-world/HoloLoom/alignment/power_seeking_monitor.py", "power_seeking_monitor"),
    ]

    all_success = True

    for filepath, module_name in modules:
        try:
            module = import_module_directly(filepath, module_name)
            print(f"✓ {module_name}")
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
        # Import modules directly
        debate_mod = import_module_directly(
            "/home/user/hello-world/HoloLoom/alignment/debate.py",
            "debate"
        )
        tot_mod = import_module_directly(
            "/home/user/hello-world/HoloLoom/alignment/tree_of_thought.py",
            "tree_of_thought"
        )
        deception_mod = import_module_directly(
            "/home/user/hello-world/HoloLoom/alignment/enhanced_deception.py",
            "enhanced_deception"
        )
        power_mod = import_module_directly(
            "/home/user/hello-world/HoloLoom/alignment/power_seeking_monitor.py",
            "power_seeking_monitor"
        )

        # Debate Mode
        DebateMode = debate_mod.DebateMode
        debate = DebateMode()
        result = await debate.debate("Test?", {"risk_level": "low"})
        print(f"✓ Debate Mode: {len(result.arguments)} arguments generated")

        # Tree-of-Thought
        TreeOfThought = tot_mod.TreeOfThought
        planner = TreeOfThought(max_depth=2, beam_width=2)
        result = await planner.plan("Test problem", ["Constraint1"])
        print(f"✓ Tree-of-Thought: {result.exploration_stats['nodes_explored']} nodes explored")

        # Enhanced Deception Detection
        EnhancedDeceptionDetector = deception_mod.EnhancedDeceptionDetector
        detector = EnhancedDeceptionDetector()
        analysis = await detector.detect_with_probes("Q", "A", {})
        print(f"✓ Enhanced Deception: Score={analysis.final_deception_score:.1%}")

        # Power-Seeking Monitor
        PowerSeekingMonitor = power_mod.PowerSeekingMonitor
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
    print("ADVANCED ALIGNMENT FRAMEWORK VERIFICATION (Direct Import)")
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
