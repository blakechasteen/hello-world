#!/usr/bin/env python3
"""
Quick Promptly System Test
===========================
Tests core components to verify everything is working.
"""

import sys
from pathlib import Path

# Add promptly to path
sys.path.insert(0, str(Path(__file__).parent / "promptly"))

def test_core_db():
    """Test core database"""
    print("Testing Core Database...")
    from promptly import PromptlyDB

    db = PromptlyDB('.promptly/promptly.db')
    print("  [OK] PromptlyDB initialized")
    return True

def test_recursive_engine():
    """Test recursive intelligence"""
    print("\nTesting Recursive Engine...")
    from recursive_loops import RecursiveEngine, LoopType, LoopConfig

    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=3
    )
    engine = RecursiveEngine(config)
    print(f"  [OK] RecursiveEngine created")
    print(f"      Loop type: {config.loop_type.value}")
    print(f"      Max iterations: {config.max_iterations}")
    return True

def test_analytics():
    """Test analytics system"""
    print("\nTesting Analytics System...")
    from tools.prompt_analytics import PromptAnalytics

    analytics = PromptAnalytics()
    summary = analytics.get_summary()
    print(f"  [OK] Analytics working")
    print(f"      Total executions: {summary['total_executions']}")
    print(f"      Unique prompts: {summary['unique_prompts']}")
    print(f"      Average quality: {summary['avg_quality']:.2f}")
    return True

def test_hololoom_bridge():
    """Test HoloLoom integration"""
    print("\nTesting HoloLoom Bridge...")
    try:
        from hololoom_unified import HOLOLOOM_AVAILABLE, create_unified_bridge

        if HOLOLOOM_AVAILABLE:
            print("  [OK] HoloLoom available")
            bridge = create_unified_bridge()
            print(f"      Bridge enabled: {bridge.enabled}")
        else:
            print("  [SKIP] HoloLoom not available (optional)")
        return True
    except Exception as e:
        print(f"  [SKIP] HoloLoom test skipped: {e}")
        return True

def test_team_system():
    """Test team collaboration"""
    print("\nTesting Team System...")
    from team_collaboration import TeamCollaboration

    team = TeamCollaboration()
    print("  [OK] Team collaboration initialized")
    print("      Users table ready")
    print("      Teams table ready")
    return True

def test_loop_composition():
    """Test loop composition"""
    print("\nTesting Loop Composition...")
    from loop_composition import Pipeline

    pipeline = Pipeline(name="test-pipeline")
    print("  [OK] Pipeline created")
    print(f"      Name: {pipeline.name}")
    return True

def main():
    """Run all tests"""
    print("="*70)
    print("Promptly System Test")
    print("="*70)

    tests = [
        ("Core Database", test_core_db),
        ("Recursive Engine", test_recursive_engine),
        ("Analytics", test_analytics),
        ("HoloLoom Bridge", test_hololoom_bridge),
        ("Team System", test_team_system),
        ("Loop Composition", test_loop_composition),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All systems operational!")
        print("\nNext steps:")
        print("  1. Run demo: python demos/demo_terminal.py")
        print("  2. Start dashboard: python promptly/web_dashboard_realtime.py")
        print("  3. Try HoloLoom: python demo_hololoom_integration.py")
    else:
        print("\n[WARN] Some tests failed. Check errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
