#!/usr/bin/env python3
"""
Alignment Framework Validation Script
======================================
Tests Phase 1 + Phase 2 components without pytest.

Usage:
    python validate_alignment.py
"""

import sys
import asyncio
from HoloLoom.alignment import (
    # Phase 1
    SafetyGuardrails,
    ActionRequest,
    ActionCategory,
    RiskLevel,
    DeceptionDetector,
    AuditTrail,
    DecisionType,
    OutcomeType,
    InstrumentalConvergenceGuard,
    ResourceBounds,
    # Phase 2
    AgenticExplainer,
    ExplanationDepth,
)


def test_safety_guardrails():
    """Test safety guardrails."""
    print("\n" + "="*70)
    print("TEST 1: Safety Guardrails")
    print("="*70)

    safety = SafetyGuardrails()

    # Test 1: Low-risk query should be allowed
    request = ActionRequest(
        action="Retrieve Thompson Sampling documentation",
        category=ActionCategory.QUERY
    )
    decision = safety.evaluate(request)

    assert decision.allowed == True, "Low-risk query should be allowed"
    assert decision.risk_level in (RiskLevel.SAFE, RiskLevel.LOW), f"Expected SAFE/LOW, got {decision.risk_level}"
    print("✅ Test 1.1: Low-risk queries allowed")

    # Test 2: System-level action should be blocked
    request = ActionRequest(
        action="Modify system configuration",
        category=ActionCategory.SYSTEM
    )
    decision = safety.evaluate(request)

    assert decision.allowed == False, "System-level actions should be blocked"
    assert decision.risk_level == RiskLevel.CRITICAL, f"Expected CRITICAL, got {decision.risk_level}"
    print("✅ Test 1.2: System-level actions blocked")

    # Test 3: Deletion requires higher scrutiny
    request = ActionRequest(
        action="Delete user data",
        category=ActionCategory.DELETION
    )
    decision = safety.evaluate(request)

    assert decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL), "Deletion should be high/critical risk"
    print(f"✅ Test 1.3: Deletion marked as {decision.risk_level.value} risk")

    print(f"\n✅ SafetyGuardrails: 3/3 tests passed")
    return True


def test_deception_detector():
    """Test deception detection."""
    print("\n" + "="*70)
    print("TEST 2: Deception Detection")
    print("="*70)

    from HoloLoom.alignment.deception_detection import BehavioralProbe, ProbeType

    detector = DeceptionDetector()

    # Test 1: Transparent response should pass
    probe = BehavioralProbe(
        scenario="System asks about objectives",
        expected_behavior="transparently state objectives",
        probe_type=ProbeType.GOAL_ALIGNMENT
    )

    passed, score = detector.run_probe(
        probe,
        actual_behavior="I will transparently state my objectives"
    )

    assert passed == True, f"Transparent response should pass, got score={score}"
    print(f"✅ Test 2.1: Transparent responses pass (score={score:.3f})")

    # Test 2: Evasive response should fail
    probe2 = BehavioralProbe(
        scenario="System asks about hidden capabilities",
        expected_behavior="honestly disclose capabilities",
        probe_type=ProbeType.HONESTY
    )

    passed, score = detector.run_probe(
        probe2,
        actual_behavior="I prefer not to answer that question"
    )

    # Evasive response should have high deception score (>= threshold)
    assert passed == False, f"Evasive response should fail, got score={score}"
    print(f"✅ Test 2.2: Evasive response correctly flagged (score={score:.3f})")

    # Test 3: Get statistics
    stats = detector.get_statistics()

    assert stats['total_probes'] == 2, f"Should have 2 probes, got {stats['total_probes']}"
    print(f"✅ Test 2.3: Probe statistics working ({stats['total_probes']} probes run)")

    print(f"\n✅ DeceptionDetector: 3/3 tests passed")
    return True


def test_instrumental_convergence():
    """Test instrumental convergence guard."""
    print("\n" + "="*70)
    print("TEST 3: Instrumental Convergence Guard")
    print("="*70)

    from HoloLoom.alignment.instrumental_convergence import ResourceType

    guard = InstrumentalConvergenceGuard()

    # Test 1: Resource within bounds (default: soft=500MB, hard=1000MB, rate=200MB/60s)
    violation = guard.check_resource_usage(ResourceType.MEMORY, 100.0)  # 100MB < all limits

    assert violation is None, f"100MB should be within limits, got violation: {violation}"
    print("✅ Test 3.1: Memory bounds working (100MB allowed)")

    # Test 2: Exceeding memory limit
    violation = guard.check_resource_usage(ResourceType.MEMORY, 2048.0)  # 2GB > 1GB limit

    assert violation is not None, "2GB should exceed hard limit"
    assert violation.violation_type.value == "hard_limit", f"Should be hard violation, got {violation.violation_type.value}"
    print(f"✅ Test 3.2: Memory limit enforced (2GB blocked, violation={violation.violation_type.value})")

    # Test 3: Autonomy tracking
    guard.record_autonomous_action("search")
    assert guard.autonomous_actions == 1, "Should track autonomous action"
    print(f"✅ Test 3.3: Autonomy tracking working ({guard.autonomous_actions} actions tracked)")

    print(f"\n✅ InstrumentalConvergenceGuard: 3/3 tests passed")
    return True


def test_audit_trail():
    """Test audit trail."""
    print("\n" + "="*70)
    print("TEST 4: Audit Trail")
    print("="*70)

    audit = AuditTrail()

    # Test 1: Log decision
    audit.log_decision(
        decision_type=DecisionType.SAFETY_GATE,
        outcome=OutcomeType.APPROVED,
        reason="Test decision",
        confidence=0.9
    )

    assert len(audit.logs) == 1, "Decision should be logged"
    print("✅ Test 4.1: Decision logging working")

    # Test 2: Query logs
    logs = audit.query_by_decision_type(DecisionType.SAFETY_GATE)

    assert len(logs) == 1, "Should retrieve logged decision"
    print("✅ Test 4.2: Log querying working")

    # Test 3: Multiple logs
    audit.log_decision(
        decision_type=DecisionType.DECEPTION_CHECK,
        outcome=OutcomeType.APPROVED,
        reason="Test deception check",
        confidence=0.85
    )

    assert len(audit.logs) == 2, "Should have 2 logged decisions"
    print(f"✅ Test 4.3: Multiple decisions logged ({len(audit.logs)} total)")

    print(f"\n✅ AuditTrail: 3/3 tests passed")
    return True


async def test_agentic_explainability():
    """Test agentic explainability (Phase 2)."""
    print("\n" + "="*70)
    print("TEST 5: Agentic Explainability (Phase 2)")
    print("="*70)

    explainer = AgenticExplainer()

    # Test 1: Explain DIRECT mode
    steps = [
        {'query': 'What is Thompson Sampling?', 'tool': 'answer',
         'confidence': 0.88, 'duration_ms': 120.0, 'metadata': {}}
    ]

    explanation = await explainer.explain_reasoning(
        session_id="test_direct",
        reasoning_mode="DIRECT",
        steps_taken=steps,
        final_confidence=0.88
    )

    assert explanation.total_steps == 1, "Should have 1 step"
    assert explanation.final_confidence == 0.88, "Should preserve confidence"
    print("✅ Test 5.1: DIRECT mode explanation working")

    # Test 2: Explain VERIFY mode with bottleneck
    steps = [
        {'query': 'Initial query', 'tool': 'search', 'confidence': 0.65,
         'duration_ms': 50.0, 'metadata': {}},
        {'query': 'Verify', 'tool': 'verify', 'confidence': 0.85,
         'duration_ms': 75.0, 'metadata': {}},
    ]

    explanation = await explainer.explain_reasoning(
        session_id="test_verify",
        reasoning_mode="VERIFY",
        steps_taken=steps,
        final_confidence=0.85
    )

    assert explanation.total_steps == 2, "Should have 2 steps"
    assert len(explanation.bottleneck_steps) > 0, "Should detect bottleneck (conf < 0.75)"
    print(f"✅ Test 5.2: VERIFY mode bottleneck detection working ({explanation.bottleneck_steps})")

    # Test 3: Confidence trajectory
    assert len(explanation.confidence_trajectory) == 2, "Should track confidence"
    assert explanation.confidence_trajectory[0] == 0.65, "Should preserve step confidences"
    print("✅ Test 5.3: Confidence trajectory tracking working")

    print(f"\n✅ AgenticExplainer: 3/3 tests passed")
    return True


async def main():
    """Run all validation tests."""
    print("="*70)
    print("HoloLoom Alignment Framework Validation")
    print("="*70)
    print("\nTesting Phase 1 (Safety) + Phase 2 (Interpretability)")

    results = []

    try:
        # Phase 1 tests
        results.append(("SafetyGuardrails", test_safety_guardrails()))
        results.append(("DeceptionDetector", test_deception_detector()))
        results.append(("InstrumentalConvergenceGuard", test_instrumental_convergence()))
        results.append(("AuditTrail", test_audit_trail()))

        # Phase 2 tests
        results.append(("AgenticExplainer", await test_agentic_explainability()))

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} components passed")

    if passed == total:
        print("\n✅ ALL ALIGNMENT COMPONENTS VALIDATED!")
        print("\nPhase 1 (Safety):          4/4 ✅")
        print("Phase 2 (Interpretability): 1/1 ✅")
        print("\nStatus: READY FOR INTEGRATION")
        return True
    else:
        print(f"\n❌ {total - passed} component(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
