#!/usr/bin/env python3
"""
xTerminator Moonshot Demo - Phase 2
====================================

🐷 "Some Pig Built a Department!" 🐷

Demonstrates Phase 2 moonshot features:
1. Department Protocol implementation (6 core methods)
2. Cross-department collaboration (QA ↔ other departments)
3. Confidence negotiation (trust between departments)
4. DS-STAR verification loops (verify → refine → re-verify)
5. Institutional memory (learned patterns)
6. Health monitoring (status, alerts, degradation)

The Demo:
- Mock "MasterWeaver" department requests QA scan
- QA Department processes request with confidence negotiation
- Verification loop refines low-confidence responses
- Health monitoring detects department status
- Institutional memory shows learned patterns

Run:
    python xterminator/demo_moonshot_phase2.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xterminator.qa_department import QADepartment
from xterminator.department_protocol import (
    DepartmentRequest,
    DepartmentResponse,
    RequestType,
    ResponseStatus,
    negotiate_confidence
)
from xterminator.autofix_policy import AutofixPolicy


def print_banner(text: str):
    """Print a pretty banner"""
    border = "=" * 70
    print(f"\n{border}")
    print(f"{text:^70}")
    print(f"{border}\n")


def print_request(request: DepartmentRequest):
    """Print a department request"""
    print(f"  Request ID: {request.request_id}")
    print(f"  Type: {request.request_type.value}")
    print(f"  From: {request.requesting_department}")
    print(f"  Priority: {request.priority}")
    print(f"  Payload: {list(request.payload.keys())}")


def print_response(response: DepartmentResponse):
    """Print a department response"""
    # Status icon
    status_icons = {
        'success': '✅',
        'partial': '⚠️',
        'failure': '❌',
        'pending': '⏳',
        'requires_review': '👀',
        'requires_refinement': '🔄'
    }
    icon = status_icons.get(response.status.value, '?')

    print(f"  {icon} Status: {response.status.value}")
    print(f"  From: {response.responding_department}")
    print(f"  Confidence: {response.confidence:.2f}")
    print(f"  Duration: {response.duration_ms:.0f}ms")

    if response.metadata.get('negotiated_confidence'):
        print(f"  Negotiated Confidence: {response.metadata['negotiated_confidence']:.2f}")

    if response.error:
        print(f"  Error: {response.error}")

    if response.payload:
        print(f"  Payload: {list(response.payload.keys())}")


async def demo_scenario_1_basic_request():
    """
    Scenario 1: Basic Department Request

    MasterWeaver requests QA to get statistics
    """
    print_banner("🐷 SCENARIO 1: Basic Department Request 🐷")
    print("MasterWeaver department requests QA statistics...\n")

    # Create QA department
    qa_dept = QADepartment(
        department_name="Quality Assurance",
        policy=AutofixPolicy.balanced(),
        enable_feedback=True
    )

    # Create request
    request = DepartmentRequest(
        request_id="req_001",
        request_type=RequestType.GET_STATISTICS,
        requesting_department="MasterWeaver",
        priority=5,
        payload={}
    )

    print("Request:")
    print_request(request)

    # Execute
    response = await qa_dept.execute(request)

    print("\nResponse:")
    print_response(response)


async def demo_scenario_2_confidence_negotiation():
    """
    Scenario 2: Confidence Negotiation

    Two departments negotiate confidence in their outputs
    """
    print_banner("🐷 SCENARIO 2: Confidence Negotiation 🐷")
    print("Departments negotiate trust in each other's outputs...\n")

    # Simulate two departments with different confidence levels
    master_weaver_confidence = 0.85  # MasterWeaver thinks it's pretty good
    qa_confidence = 0.78             # QA found some issues

    print(f"MasterWeaver Confidence: {master_weaver_confidence:.2f}")
    print(f"QA Confidence: {qa_confidence:.2f}\n")

    # Negotiate confidence
    strategies = ['weighted_average', 'minimum', 'maximum', 'trust_weighted']

    for strategy in strategies:
        negotiated = negotiate_confidence(
            requesting_dept="MasterWeaver",
            responding_dept="Quality Assurance",
            request_conf=master_weaver_confidence,
            response_conf=qa_confidence,
            strategy=strategy,
            historical_accuracy=0.90  # QA has been 90% accurate historically
        )

        print(f"  Strategy: {strategy:20} → Negotiated: {negotiated:.2f}")

    print("\nInterpretation:")
    print("  - weighted_average: Balanced (default)")
    print("  - minimum: Pessimistic (lower bound)")
    print("  - maximum: Optimistic (upper bound)")
    print("  - trust_weighted: Trust QA's historical accuracy")


async def demo_scenario_3_verification_loop():
    """
    Scenario 3: DS-STAR Verification Loop

    QA verifies a response, finds issues, refines, re-verifies
    """
    print_banner("🐷 SCENARIO 3: DS-STAR Verification Loop 🐷")
    print("Verify → Refine → Re-verify cycle...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())

    # Initial request
    request = DepartmentRequest(
        request_id="req_002",
        request_type=RequestType.CLASSIFY_ISSUE,
        requesting_department="Infrastructure",
        payload={'issue': 'example'}
    )

    # Execute (initial response)
    print("Step 1: Initial Response")
    response = await qa_dept.execute(request)
    print_response(response)

    # Verify
    print("\nStep 2: Verification")
    verification = await qa_dept.verify(response)
    print(f"  Verified: {verification.verified}")
    print(f"  Original Confidence: {verification.original_confidence:.2f}")
    print(f"  Verified Confidence: {verification.verified_confidence:.2f}")
    print(f"  Confidence Delta: {verification.confidence_delta:+.2f}")
    print(f"  Issues Found: {len(verification.issues_found)}")
    for issue in verification.issues_found:
        print(f"    - {issue}")

    # Refine (if needed)
    if verification.requires_refinement:
        print("\nStep 3: Refinement")
        refined_response = await qa_dept.refine(request, response, verification)
        print_response(refined_response)
        print(f"  Refinement Iteration: {refined_response.metadata.get('refinement_iteration', 0)}")

        # Re-verify
        print("\nStep 4: Re-verification")
        re_verification = await qa_dept.verify(refined_response)
        print(f"  Verified: {re_verification.verified}")
        print(f"  Verified Confidence: {re_verification.verified_confidence:.2f}")
    else:
        print("\n✓ No refinement needed - verification passed!")


async def demo_scenario_4_institutional_memory():
    """
    Scenario 4: Institutional Memory

    Query learned patterns from QA department
    """
    print_banner("🐷 SCENARIO 4: Institutional Memory 🐷")
    print("Querying learned patterns from QA department...\n")

    qa_dept = QADepartment(enable_feedback=True)

    # Query different pattern types
    pattern_types = [
        'successful_strategies',
        'failed_patterns',
        'confidence_calibration',
        'performance_trends'
    ]

    for pattern_type in pattern_types:
        print(f"Pattern Type: {pattern_type}")
        memory = await qa_dept.get_institutional_memory(pattern_type)

        if 'error' in memory:
            print(f"  {memory['error']}\n")
        else:
            print(f"  Data available: {list(memory.keys())[:5]}...\n")


async def demo_scenario_5_health_monitoring():
    """
    Scenario 5: Health Monitoring

    Check QA department health status
    """
    print_banner("🐷 SCENARIO 5: Health Monitoring 🐷")
    print("Checking QA department health...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced(), enable_feedback=True)

    # Simulate some requests
    for i in range(10):
        request = DepartmentRequest(
            request_id=f"health_req_{i}",
            request_type=RequestType.GET_STATISTICS,
            requesting_department="Orchestration",
            payload={}
        )
        await qa_dept.execute(request)

    # Check health
    health = await qa_dept.health_check()

    # Status icon
    status_icons = {
        'healthy': '✅',
        'degraded': '⚠️',
        'unhealthy': '❌'
    }
    icon = status_icons.get(health['status'], '?')

    print(f"Status: {icon} {health['status'].upper()}")
    print(f"Uptime: {health['uptime_seconds']:.1f}s")
    print(f"Total Requests: {health['total_requests']}")
    print(f"Success Rate: {health['success_rate']:.1%}")
    print(f"Error Rate: {health['error_rate']:.1%}")
    print(f"Avg Latency: {health['avg_latency_ms']:.0f}ms")
    print(f"Confidence Accuracy: {health['confidence_accuracy']:.1%}")
    print(f"Degradation Detected: {health['degradation_detected']}")
    print(f"Policy: {health['policy_profile']} ({health['policy_domain']})")

    if health['alerts']:
        print(f"\nAlerts:")
        for alert in health['alerts']:
            print(f"  ⚠️  {alert}")
    else:
        print("\n✓ No alerts")


async def demo_scenario_6_cross_department_workflow():
    """
    Scenario 6: Cross-Department Workflow

    Complete workflow: MasterWeaver → QA → Infrastructure
    """
    print_banner("🐷 SCENARIO 6: Cross-Department Workflow 🐷")
    print("Simulating cross-department collaboration...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())

    # Step 1: MasterWeaver requests scan
    print("Step 1: MasterWeaver → QA")
    print("  MasterWeaver: 'Please scan my entity extraction code'\n")

    scan_request = DepartmentRequest(
        request_id="workflow_001",
        request_type=RequestType.SCAN_CODE,
        requesting_department="MasterWeaver",
        priority=3,
        payload={
            'code': 'def extract_entities(text): ...',
            'file_path': 'entity_extractor.py'
        },
        context={'requesting_confidence': 0.75}  # MasterWeaver is 75% confident
    )

    scan_response = await qa_dept.execute(scan_request)
    print("  QA Response:")
    print_response(scan_response)

    # Step 2: QA proposes fix
    print("\nStep 2: QA Proposes Fix")
    print("  QA: 'Found 0 issues, code looks good!'\n")

    # Step 3: Infrastructure stores result
    print("Step 3: Infrastructure Stores Result")
    print("  Infrastructure: 'Storing QA report in Neo4j...'\n")

    print("✓ Cross-department workflow complete!")


async def main():
    """Run all demo scenarios"""
    print("\n")
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)
    print("        xTerminator Moonshot Demo - Phase 2".center(70))
    print("         \"From Tool to Department!\"".center(70))
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)

    await demo_scenario_1_basic_request()
    await demo_scenario_2_confidence_negotiation()
    await demo_scenario_3_verification_loop()
    await demo_scenario_4_institutional_memory()
    await demo_scenario_5_health_monitoring()
    await demo_scenario_6_cross_department_workflow()

    print_banner("🐷 Phase 2 Complete! 🐷")
    print("Department Protocol Features:")
    print("  ✓ execute() - Process requests from other departments")
    print("  ✓ verify() - Verify responses for quality")
    print("  ✓ refine() - Refine low-confidence responses")
    print("  ✓ update_strategy() - Learn from outcomes")
    print("  ✓ get_institutional_memory() - Query learned patterns")
    print("  ✓ health_check() - Monitor department health")
    print("\nCross-Department Integration:")
    print("  ✓ Confidence negotiation (weighted, min, max, trust-weighted)")
    print("  ✓ DS-STAR verification loops (verify → refine → re-verify)")
    print("  ✓ Institutional memory (successful strategies, failed patterns)")
    print("  ✓ Health monitoring (status, alerts, degradation detection)")
    print("\nNext steps:")
    print("  - Phase 3 (Weeks 5-7): Orchestration Integration")
    print("  - Phase 4 (Weeks 8-10): Thompson Sampling (adaptive strategy)")
    print("  - Phase 5 (Weeks 11-18): Advanced Features (marketplace, semantic)")
    print("\n(*)<  OINK OINK OINK!")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
