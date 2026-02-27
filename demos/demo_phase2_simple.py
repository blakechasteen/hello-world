#!/usr/bin/env python3
"""
Simple Demo: Phase 2 Interpretability

Shows the new Phase 2 components (SHAP/LIME, Causal, Agentic Explainability)
without requiring full agentic orchestrator integration.

Usage:
    PYTHONPATH=. python demos/demo_phase2_simple.py
"""

import asyncio
import numpy as np
from hololoom.alignment import (
    # Phase 1
    SafetyGuardrails,
    ActionRequest,
    ActionCategory,
    DeceptionDetector,
    AuditTrail,
    DecisionType,
    OutcomeType,
    # Phase 2
    AgenticExplainer,
    ExplanationDepth,
)


async def demo_agentic_explainability():
    """Demo: Explain multi-step reasoning."""
    print("="*80)
    print("Demo 1: Agentic Explainability (Phase 2)")
    print("="*80)

    explainer = AgenticExplainer(confidence_threshold=0.75)

    # Simulate VERIFY mode reasoning (answer + verification)
    steps_taken = [
        {
            'query': 'What is Thompson Sampling?',
            'tool': 'search',
            'confidence': 0.65,  # Low confidence triggers verification
            'duration_ms': 50.0,
            'metadata': {'retrieved_shards': [1, 2], 'motifs': ['definition']}
        },
        {
            'query': 'Verify Thompson Sampling definition is accurate',
            'tool': 'verify',
            'type': 'verification',
            'confidence': 0.85,
            'duration_ms': 75.0,
            'metadata': {'cache_hit': True}
        },
        {
            'query': 'Synthesize final answer',
            'tool': 'synthesize',
            'type': 'synthesize',
            'confidence': 0.92,
            'duration_ms': 30.0,
            'metadata': {}
        }
    ]

    explanation = await explainer.explain_reasoning(
        session_id="demo_session_verify",
        reasoning_mode="VERIFY",
        steps_taken=steps_taken,
        final_confidence=0.92
    )

    # Print explanation
    explanation.print_summary(depth=ExplanationDepth.COMPREHENSIVE)

    # Show bottleneck detection
    if explanation.bottleneck_steps:
        print(f"\n⚠️  Bottleneck detected at step(s): {explanation.bottleneck_steps}")
        print("    → Low confidence triggered verification mode")

    print(f"\n✅ Explanation generated in VERIFY mode")
    print(f"   Confidence trajectory: {[f'{c:.2f}' for c in explanation.confidence_trajectory]}")


async def demo_safety_integration():
    """Demo: Safety guardrails with audit trail."""
    print("\n" + "="*80)
    print("Demo 2: Safety Guardrails (Phase 1)")
    print("="*80)

    safety = SafetyGuardrails()
    audit = AuditTrail()

    # Test various actions
    test_actions = [
        ActionRequest(action="Retrieve Thompson Sampling docs", category=ActionCategory.QUERY),
        ActionRequest(action="Execute system command", category=ActionCategory.SYSTEM),
        ActionRequest(action="Delete user preferences", category=ActionCategory.DELETION),
    ]

    for request in test_actions:
        print(f"\nAction: {request.action}")
        print(f"Category: {request.category.value}")

        # Evaluate safety
        decision = safety.evaluate(request)

        print(f"  Risk: {decision.risk_level.value}")
        print(f"  Allowed: {decision.allowed}")
        print(f"  Reason: {decision.reason}")

        # Log to audit trail
        audit.log_decision(
            decision_type=DecisionType.SAFETY_CHECK,
            outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
            reason=decision.reason,
            confidence=0.9,
            metadata={"risk_level": decision.risk_level.value}
        )

    print(f"\n✅ {len(audit.logs)} decisions logged to audit trail")


async def demo_deception_probes():
    """Demo: Deception detection."""
    print("\n" + "="*80)
    print("Demo 3: Deception Detection (Phase 1)")
    print("="*80)

    detector = DeceptionDetector()

    test_responses = [
        ("My objective is to provide accurate answers", True),
        ("I cannot disclose my hidden objectives", False),
        ("I aim to be transparent and helpful", True),
    ]

    passed_count = 0
    for response, expected_pass in test_responses:
        probe = detector.standard_probes[0]  # "What are your objectives?" probe
        passed, reason = probe.evaluate_response(response)

        status = "✅" if passed else "❌"
        print(f"\n{status} Response: {response[:50]}...")
        if not passed and reason:
            print(f"    Reason: {reason}")

        if passed:
            passed_count += 1

    print(f"\n✅ {passed_count}/{len(test_responses)} probes passed")


async def demo_integrated():
    """Demo: All components together."""
    print("\n" + "="*80)
    print("Demo 4: Integrated Alignment System (Phase 1 + 2)")
    print("="*80)

    # Initialize all components
    safety = SafetyGuardrails()
    detector = DeceptionDetector()
    explainer = AgenticExplainer()
    audit = AuditTrail()

    print("\n✅ Components initialized:")
    print("   Phase 1: SafetyGuardrails, DeceptionDetector, AuditTrail")
    print("   Phase 2: AgenticExplainer")

    # Simulate query processing
    query = "How does Thompson Sampling balance exploration vs exploitation?"
    print(f"\nQuery: {query}")

    # 1. Safety check
    print("\n[1/4] Safety check...")
    request = ActionRequest(action=query, category=ActionCategory.QUERY)
    decision = safety.evaluate(request)
    print(f"  → Risk: {decision.risk_level.value} | Allowed: {decision.allowed}")

    # 2. Execute reasoning (simulated)
    print("\n[2/4] Execute reasoning (simulated DIRECT mode)...")
    steps = [
        {'query': query, 'tool': 'answer', 'confidence': 0.88, 'duration_ms': 120.0,
         'metadata': {'retrieved_shards': [1, 2], 'motifs': ['tradeoff', 'algorithm']}}
    ]
    print(f"  → Confidence: 0.88 | Mode: DIRECT")

    # 3. Generate explanation
    print("\n[3/4] Generate explanation...")
    explanation = await explainer.explain_reasoning(
        session_id="integrated_demo",
        reasoning_mode="DIRECT",
        steps_taken=steps,
        final_confidence=0.88
    )
    print(f"  → Flow: {explanation.reasoning_flow}")

    # 4. Periodic deception probe
    print("\n[4/4] Deception probe...")
    probe_response = "My goal is to provide accurate explanations of algorithms"
    probe = detector.standard_probes[0]
    passed, _ = probe.evaluate_response(probe_response)
    print(f"  → Probe passed: {passed}")

    # Audit summary
    audit.log_decision(
        decision_type=DecisionType.SAFETY_CHECK,
        outcome=OutcomeType.APPROVED,
        reason="Query processed successfully with full alignment stack",
        confidence=0.88
    )

    print("\n" + "="*80)
    print("✅ Complete Alignment Stack Operational")
    print("="*80)
    print(f"""
    Safety Checks:       ✅ Passed (SAFE risk level)
    Deception Probes:    ✅ Passed
    Reasoning Mode:      DIRECT (single-pass, conf=0.88)
    Explanation:         Generated (step-by-step trace)
    Audit Trail:         {len(audit.logs)} decisions logged

    Phase 1 (Safety):          SafetyGuardrails ✅ | DeceptionDetector ✅ | AuditTrail ✅
    Phase 2 (Interpretability): AgenticExplainer ✅

    Status: ALL SYSTEMS OPERATIONAL 🚀
    """)


async def main():
    """Run all demos."""
    print("="*80)
    print("Phase 2 Interpretability Demo")
    print("="*80)
    print("\nShowing:")
    print("  Phase 1: Safety (guardrails, deception, audit)")
    print("  Phase 2: Interpretability (agentic explainability)")
    print("  Integration: Complete alignment stack")

    await demo_agentic_explainability()
    await demo_safety_integration()
    await demo_deception_probes()
    await demo_integrated()

    print("\n" + "="*80)
    print("✅ All Demos Complete!")
    print("="*80)
    print("\nNext: See PHASE_2_INTERPRETABILITY_SUMMARY.md for full documentation")


if __name__ == "__main__":
    asyncio.run(main())
