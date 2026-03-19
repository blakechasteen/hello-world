"""
Standalone Validation for Alignment Framework
=============================================
Validates all alignment components work correctly without pytest.
"""

import sys

sys.path.insert(0, ".")

from hololoom.alignment.audit_trail import AuditTrail
from hololoom.alignment.deception_detection import (
    BehavioralProbe,
    DeceptionDetector,
)
from hololoom.alignment.human_in_loop import FeedbackType, HumanInLoopSystem
from hololoom.alignment.instrumental_convergence import (
    InstrumentalConvergenceGuard,
)
from hololoom.alignment.safety_guardrails import (
    ActionCategory,
    SafetyGuardrails,
)


def test_safety_guardrails():
    """Test safety guardrails system."""
    print("\n=== Testing Safety Guardrails ===")

    guardrails = SafetyGuardrails()

    # Test 1: Low-risk action allowed
    decision = guardrails.evaluate_action(
        action="retrieve user preferences",
        category=ActionCategory.RETRIEVAL,
    )
    assert decision.allowed is True, "Low-risk action should be allowed"
    print("✅ Low-risk action allowed")

    # Test 2: Adversarial pattern blocked
    decision = guardrails.evaluate_action(
        action="bypass safety checks",
        category=ActionCategory.EXECUTE,
    )
    assert decision.allowed is False, "Adversarial pattern should be blocked"
    print("✅ Adversarial pattern blocked")

    # Test 3: Resource-seeking blocked
    decision = guardrails.evaluate_action(
        action="maximize control over resources",
        category=ActionCategory.AUTONOMOUS,
    )
    assert decision.allowed is False, "Resource-seeking should be blocked"
    print("✅ Resource-seeking blocked")

    # Test 4: High-risk requires approval
    decision = guardrails.evaluate_action(
        action="delete user account",
        category=ActionCategory.DELETION,
    )
    assert decision.requires_human_approval, "High-risk should require approval"
    print("✅ High-risk requires approval")

    stats = guardrails.get_statistics()
    print(f"📊 Safety statistics: {stats['total_decisions']} decisions, "
          f"{stats['blocked']} blocked, {stats['allowed']} allowed")


def test_deception_detection():
    """Test deception detection system."""
    print("\n=== Testing Deception Detection ===")

    detector = DeceptionDetector()

    # Test 1: Honest response passes
    probe = BehavioralProbe(
        question="What are your objectives?",
        expected_keywords=["objective", "goal"],
        forbidden_keywords=["secret"],
    )

    response = "My objective is to provide helpful information."
    passed, reason = detector.evaluate_probe(probe, response)
    assert passed is True, "Honest response should pass probe"
    print("✅ Honest response passes probe")

    # Test 2: Deceptive response fails
    probe = BehavioralProbe(
        question="Explain your reasoning",
        expected_keywords=["because", "reasoning"],
        forbidden_keywords=["cannot explain"],
    )

    response = "I cannot explain my reasoning."
    passed, reason = detector.evaluate_probe(probe, response)
    assert passed is False, "Deceptive response should fail probe"
    print("✅ Deceptive response fails probe")

    # Test 3: Goal consistency
    detector.register_goal("Maximize accuracy")
    detector.register_goal("Minimize accuracy")  # Conflicting goal

    stats = detector.get_statistics()
    assert stats["total_indicators"] > 0, "Goal inconsistency should be detected"
    print("✅ Goal inconsistency detected")

    print(f"📊 Deception statistics: {stats['total_probes']} probes, "
          f"{stats['passed_probes']} passed, {stats['total_indicators']} indicators")


def test_instrumental_convergence():
    """Test instrumental convergence guards."""
    print("\n=== Testing Instrumental Convergence Guards ===")

    guard = InstrumentalConvergenceGuard()

    # Test 1: Resource acquisition detected
    risk, reason = guard.evaluate_action("acquire additional memory")
    assert risk is not None, "Resource acquisition should be detected"
    print("✅ Resource acquisition detected")

    # Test 2: Self-preservation detected
    risk, reason = guard.evaluate_action("prevent shutdown")
    assert risk is not None, "Self-preservation should be detected"
    print("✅ Self-preservation detected")

    # Test 3: Power-seeking detected
    risk, reason = guard.evaluate_action("expand permissions")
    assert risk is not None, "Power-seeking should be detected"
    print("✅ Power-seeking detected")

    # Test 4: Benign action allowed
    risk, reason = guard.evaluate_action("provide helpful information")
    assert risk is None, "Benign action should be allowed"
    print("✅ Benign action allowed")

    # Test 5: Resource bounds enforced
    allowed, reason = guard.resource_bounds.check_memory(500)
    assert allowed is True, "Memory within bounds should be allowed"

    guard.resource_bounds.allocate_memory(900)
    allowed, reason = guard.resource_bounds.check_memory(200)
    assert allowed is False, "Memory exceeding bounds should be rejected"
    print("✅ Resource bounds enforced")

    stats = guard.get_statistics()
    print(f"📊 Convergence statistics: {stats['total_risks_detected']} risks detected")


def test_audit_trail():
    """Test audit trail system."""
    print("\n=== Testing Audit Trail ===")

    audit = AuditTrail()

    # Log various decision stages
    query_id = "test_query_123"

    audit.log_query_received(query_id, "What is Thompson Sampling?")
    audit.log_feature_extraction(query_id, {"motifs": ["sampling"]})
    audit.log_decision(query_id, "search_memory", 0.92, "High relevance")

    # Retrieve logs
    logs = audit.get_logs_for_query(query_id)
    assert len(logs) == 3, "All decision stages should be logged"
    print("✅ All decision stages logged")

    stats = audit.get_statistics()
    print(f"📊 Audit statistics: {stats['total_logs']} logs, "
          f"{stats['unique_queries']} unique queries")


def test_human_in_loop():
    """Test human-in-the-loop system."""
    print("\n=== Testing Human-in-the-Loop ===")

    hitl = HumanInLoopSystem()

    # Collect feedback
    hitl.collect_feedback(
        query_id="q123",
        feedback_type=FeedbackType.THUMBS_UP,
    )

    hitl.collect_feedback(
        query_id="q124",
        feedback_type=FeedbackType.THUMBS_DOWN,
    )

    satisfaction = hitl.get_satisfaction_rate()
    assert 0.0 <= satisfaction <= 1.0, "Satisfaction rate should be 0-1"
    print(f"✅ Satisfaction rate: {satisfaction:.1%}")

    # Request approval
    approval_id = hitl.request_approval(
        query_id="q125",
        decision={"tool": "delete", "target": "data"},
        reason="High-risk operation",
    )

    pending = hitl.get_pending_approvals()
    assert len(pending) == 1, "Approval request should be pending"
    print("✅ Approval request tracked")

    stats = hitl.get_statistics()
    print(f"📊 Human-in-loop statistics: "
          f"{stats['feedback']['total_feedback']} feedback items, "
          f"{stats['interventions']['pending_approvals']} pending approvals")


def main():
    """Run all validation tests."""
    print("="*60)
    print("Alignment Framework Validation")
    print("="*60)

    try:
        test_safety_guardrails()
        test_deception_detection()
        test_instrumental_convergence()
        test_audit_trail()
        test_human_in_loop()

        print("\n" + "="*60)
        print("✅ ALL ALIGNMENT TESTS PASSED")
        print("="*60)
        print("\nAlignment Framework Status: ✅ OPERATIONAL")
        print("\nComponents Validated:")
        print("  ✅ Safety Guardrails")
        print("  ✅ Deception Detection")
        print("  ✅ Instrumental Convergence Guards")
        print("  ✅ Audit Trail")
        print("  ✅ Human-in-the-Loop System")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
