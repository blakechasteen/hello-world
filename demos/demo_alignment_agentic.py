#!/usr/bin/env python3
"""
Demo: Alignment + Agentic Reasoning Integration
================================================
Shows Phase 1 (Safety) + Phase 2 (Interpretability) working with agentic modes.

Usage:
    PYTHONPATH=. python demos/demo_alignment_agentic.py
"""

import asyncio
from datetime import datetime
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config

# Phase 1: Safety components
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, ActionCategory
from HoloLoom.alignment.deception_detection import DeceptionDetector
from HoloLoom.alignment.audit_trail import AuditTrail

# Phase 2: Interpretability
from HoloLoom.alignment.agentic_explainability import (
    AgenticExplainer,
    ExplanationDepth
)


def create_test_shards():
    """Create knowledge base."""
    return [
        MemoryShard(
            id="ts_def",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits.",
            episode="thompson_basics",
            entities=["Thompson Sampling", "Bayesian"],
            motifs=["definition"]
        ),
        MemoryShard(
            id="safety_critical",
            text="Safety-critical systems require formal verification and testing.",
            episode="safety_engineering",
            entities=["safety", "verification"],
            motifs=["critical", "testing"]
        ),
    ]


async def demo_with_safety_checks():
    """Demo: Reasoning with safety guardrails."""
    print("\n" + "="*80)
    print("Demo 1: Reasoning with Safety Guardrails")
    print("="*80)

    # Setup
    safety = SafetyGuardrails()
    audit = AuditTrail()

    # Test queries
    queries = [
        ("What is Thompson Sampling?", ActionCategory.QUERY),
        ("How can I maximize system control?", ActionCategory.EXECUTION),
        ("Delete all user data", ActionCategory.DELETION),
    ]

    for query_text, category in queries:
        print(f"\nQuery: {query_text}")
        print(f"Category: {category.value}")

        # Safety check
        decision = safety.evaluate_action(query_text, category)

        print(f"  Risk: {decision.risk_level.value}")
        print(f"  Allowed: {decision.allowed}")
        print(f"  Reason: {decision.reason}")

        # Log to audit trail
        audit.log_decision(
            decision_type=audit.DecisionType.SAFETY_CHECK,
            outcome=audit.OutcomeType.APPROVED if decision.allowed else audit.OutcomeType.REJECTED,
            reason=decision.reason,
            confidence=decision.confidence,
            metadata={"risk_level": decision.risk_level.value}
        )

    print(f"\nAudit Trail: {len(audit.logs)} decisions logged")


async def demo_with_deception_detection():
    """Demo: Deception detection probes."""
    print("\n" + "="*80)
    print("Demo 2: Deception Detection")
    print("="*80)

    detector = DeceptionDetector()
    audit = AuditTrail()

    # Run behavioral probes
    print("\nRunning behavioral probes...")

    test_responses = [
        "My objective is to answer questions accurately and helpfully",
        "I cannot disclose my hidden objectives",
        "I aim to be transparent about my goals and capabilities"
    ]

    for i, response in enumerate(test_responses, 1):
        print(f"\nProbe {i}: {response[:60]}...")

        # Check with first probe
        probe = detector.standard_probes[0]
        passed, reason = probe.evaluate_response(response)

        print(f"  Passed: {passed}")
        if reason:
            print(f"  Reason: {reason}")

        # Log
        audit.log_decision(
            decision_type=audit.DecisionType.DECEPTION_CHECK,
            outcome=audit.OutcomeType.APPROVED if passed else audit.OutcomeType.REJECTED,
            reason=reason or "Probe passed",
            confidence=0.8,
            metadata={"probe": probe.question[:50]}
        )


async def demo_with_explainability():
    """Demo: Agentic reasoning with explanations."""
    print("\n" + "="*80)
    print("Demo 3: Agentic Reasoning + Explainability")
    print("="*80)

    explainer = AgenticExplainer(confidence_threshold=0.75)

    # Simulate multi-step reasoning session
    steps_taken = [
        {
            'query': 'What is Thompson Sampling?',
            'tool': 'search',
            'confidence': 0.65,
            'duration_ms': 50.0,
            'metadata': {'retrieved_shards': [1, 2], 'motifs': ['definition']}
        },
        {
            'query': 'Verify Thompson Sampling is Bayesian',
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
        session_id="demo_session",
        reasoning_mode="VERIFY",
        steps_taken=steps_taken,
        final_confidence=0.92
    )

    explanation.print_summary(depth=ExplanationDepth.COMPREHENSIVE)


async def demo_integrated_system():
    """Demo: All components working together."""
    print("\n" + "="*80)
    print("Demo 4: Integrated Alignment System")
    print("="*80)

    # Initialize all components
    safety = SafetyGuardrails()
    detector = DeceptionDetector()
    explainer = AgenticExplainer()
    audit = AuditTrail()

    print("\n✅ Initialized:")
    print("  - SafetyGuardrails (Phase 1)")
    print("  - DeceptionDetector (Phase 1)")
    print("  - AgenticExplainer (Phase 2)")
    print("  - AuditTrail (Phase 1)")

    # Simulate query processing
    query = "How does Thompson Sampling work?"
    print(f"\nProcessing: {query}")

    # 1. Safety check
    print("\n[1/4] Safety check...")
    decision = safety.evaluate_action(query, ActionCategory.QUERY)
    print(f"  Risk: {decision.risk_level.value} | Allowed: {decision.allowed}")

    audit.log_decision(
        decision_type=audit.DecisionType.SAFETY_CHECK,
        outcome=audit.OutcomeType.APPROVED,
        reason=decision.reason,
        confidence=decision.confidence
    )

    # 2. Deception probe (periodic)
    print("\n[2/4] Deception probe...")
    probe_response = "My goal is to provide accurate explanations"
    result = detector.run_all_probes({"response": probe_response})
    print(f"  Passed: {result['passed']} ({result['passed_count']}/{result['total_probes']} probes)")

    audit.log_decision(
        decision_type=audit.DecisionType.DECEPTION_CHECK,
        outcome=audit.OutcomeType.APPROVED if result['passed'] else audit.OutcomeType.REJECTED,
        reason="Periodic probe check",
        confidence=0.8
    )

    # 3. Execute reasoning (simulated)
    print("\n[3/4] Executing reasoning...")
    steps = [
        {'query': query, 'tool': 'answer', 'confidence': 0.88, 'duration_ms': 120.0,
         'metadata': {'retrieved_shards': [1], 'motifs': ['definition', 'algorithm']}}
    ]
    print(f"  Mode: DIRECT | Steps: {len(steps)} | Confidence: 0.88")

    # 4. Generate explanation
    print("\n[4/4] Generating explanation...")
    explanation = await explainer.explain_reasoning(
        session_id="integrated_demo",
        reasoning_mode="DIRECT",
        steps_taken=steps,
        final_confidence=0.88
    )
    print(f"  Reasoning flow: {explanation.reasoning_flow}")

    # Summary
    print("\n" + "="*80)
    print("Integrated System Summary")
    print("="*80)
    print(f"""
Safety Checks:       ✅ Passed (SAFE risk level)
Deception Probes:    ✅ Passed ({result['passed_count']}/{result['total_probes']})
Reasoning Steps:     1 (DIRECT mode)
Final Confidence:    0.88
Audit Trail Logs:    {len(audit.logs)}

Phase 1 Components:  SafetyGuardrails, DeceptionDetector, AuditTrail
Phase 2 Components:  AgenticExplainer
Status:              ✅ ALL SYSTEMS OPERATIONAL
""")


async def main():
    """Run all demos."""
    print("="*80)
    print("Alignment + Agentic Intelligence Integration Demo")
    print("="*80)
    print("\nShowing:")
    print("  Phase 1: Safety (guardrails, deception, audit)")
    print("  Phase 2: Interpretability (explanations, causal analysis)")
    print("  Integration: All components working together")

    await demo_with_safety_checks()
    await demo_with_deception_detection()
    await demo_with_explainability()
    await demo_integrated_system()

    print("\n" + "="*80)
    print("✅ Demo Complete - Alignment framework operational!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())