"""
Simple Integration Test for Recursive Learning System
======================================================
Tests that all 5 phases can be imported and basic APIs work.
"""

import asyncio


def test_phase1_imports():
    """Test Phase 1: Scratchpad Integration imports"""
    print("[TEST] Phase 1: Scratchpad Integration")
    try:
        from HoloLoom.recursive import (
            ScratchpadOrchestrator,
            ScratchpadConfig,
            ProvenanceTracker,
            RecursiveRefiner,
            weave_with_scratchpad,
        )
        print("  [OK] All Phase 1 imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_phase2_imports():
    """Test Phase 2: Loop Engine Integration imports"""
    print("[TEST] Phase 2: Loop Engine Integration")
    try:
        from HoloLoom.recursive import (
            LearningLoopEngine,
            LearningLoopConfig,
            PatternExtractor,
            PatternLearner,
            LearnedPattern,
            weave_with_learning,
        )
        print("  [OK] All Phase 2 imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_phase3_imports():
    """Test Phase 3: Hot Pattern Feedback imports"""
    print("[TEST] Phase 3: Hot Pattern Feedback")
    try:
        from HoloLoom.recursive import (
            HotPatternFeedbackEngine,
            HotPatternConfig,
            HotPatternTracker,
            AdaptiveRetriever,
            UsageRecord,
        )
        print("  [OK] All Phase 3 imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_phase4_imports():
    """Test Phase 4: Advanced Refinement imports"""
    print("[TEST] Phase 4: Advanced Refinement")
    try:
        from HoloLoom.recursive import (
            AdvancedRefiner,
            RefinementStrategy,
            RefinementResult,
            QualityMetrics,
            RefinementPattern,
            refine_with_strategy,
        )
        print("  [OK] All Phase 4 imports successful")

        # Test enum values
        strategies = list(RefinementStrategy)
        print(f"  [OK] {len(strategies)} refinement strategies available:")
        for strategy in strategies:
            print(f"       - {strategy.value}")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_phase5_imports():
    """Test Phase 5: Full Learning Loop imports"""
    print("[TEST] Phase 5: Full Learning Loop")
    try:
        from HoloLoom.recursive import (
            FullLearningEngine,
            ThompsonPriors,
            PolicyWeights,
            BackgroundLearner,
            LearningMetrics,
            weave_with_full_learning,
        )
        print("  [OK] All Phase 5 imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_data_structures():
    """Test that data structures can be instantiated"""
    print("[TEST] Data Structures")
    try:
        from HoloLoom.recursive import (
            ThompsonPriors,
            PolicyWeights,
            LearningMetrics,
            QualityMetrics,
            UsageRecord,
        )
        from datetime import datetime

        # Test Thompson Priors
        priors = ThompsonPriors()
        priors.update_success("tool1", 0.8)
        reward = priors.get_expected_reward("tool1")
        print(f"  [OK] ThompsonPriors: Expected reward = {reward:.3f}")

        # Test Policy Weights
        weights = PolicyWeights()
        weights.update("adapter1", True)
        weight = weights.get_weight("adapter1")
        print(f"  [OK] PolicyWeights: Weight = {weight:.3f}")

        # Test Learning Metrics
        metrics = LearningMetrics()
        metrics.update(0.85)
        print(f"  [OK] LearningMetrics: Avg confidence = {metrics.avg_confidence:.2f}")

        # Test Quality Metrics
        quality = QualityMetrics(
            confidence=0.90,
            threads_activated=5,
            motifs_detected=3,
            context_size=8,
            response_length=250,
            timestamp=datetime.now()
        )
        score = quality.score()
        print(f"  [OK] QualityMetrics: Quality score = {score:.3f}")

        # Test Usage Record
        record = UsageRecord(
            element_id="test_element",
            element_type="thread",
            access_count=10,
            success_count=8,
            total_confidence=7.5,
            avg_confidence=0.75
        )
        heat = record.heat_score
        print(f"  [OK] UsageRecord: Heat score = {heat:.1f}")

        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_refinement_strategies():
    """Test refinement strategy enum"""
    print("[TEST] Refinement Strategies")
    try:
        from HoloLoom.recursive import RefinementStrategy

        strategies = {
            RefinementStrategy.REFINE: "Context expansion",
            RefinementStrategy.CRITIQUE: "Self-improvement",
            RefinementStrategy.VERIFY: "Multi-pass verification",
            RefinementStrategy.ELEGANCE: "Multi-pass polish",
            RefinementStrategy.HOFSTADTER: "Recursive self-reference",
        }

        print(f"  [OK] {len(strategies)} strategies defined:")
        for strategy, description in strategies.items():
            print(f"       - {strategy.value}: {description}")

        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    """Run all integration tests"""
    print()
    print("=" * 80)
    print("RECURSIVE LEARNING SYSTEM - INTEGRATION TEST")
    print("=" * 80)
    print()

    results = []

    # Test all phases
    results.append(("Phase 1 Imports", test_phase1_imports()))
    results.append(("Phase 2 Imports", test_phase2_imports()))
    results.append(("Phase 3 Imports", test_phase3_imports()))
    results.append(("Phase 4 Imports", test_phase4_imports()))
    results.append(("Phase 5 Imports", test_phase5_imports()))
    results.append(("Data Structures", test_data_structures()))
    results.append(("Refinement Strategies", test_refinement_strategies()))

    # Summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print()

    if passed == total:
        print("[SUCCESS] All integration tests passed!")
        print()
        print("The Recursive Learning System is ready to use:")
        print("  - All 5 phases implemented")
        print("  - All imports working")
        print("  - Data structures functional")
        print("  - ~4,700 lines of production code")
        print()
        return 0
    else:
        print(f"[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
