#!/usr/bin/env python3
"""
xTerminator Moonshot Demo - Phase 3
====================================

🐷 "Some Pig Built an Orchestration!" 🐷

Demonstrates Phase 3 moonshot features:
1. Cross-department scanning (QA scans MasterWeaver, Infrastructure, etc.)
2. Quality gates (block operations if quality fails)
3. Error escalation (route issues to humans)
4. Parallel department scanning
5. Orchestration metrics and monitoring

The Demo:
- Mock multiple departments (MasterWeaver, Infrastructure, Verification)
- Orchestration Bridge scans all outputs
- Quality gates enforce standards
- Escalation system routes issues
- Metrics show overall system quality

Run:
    python xterminator/demo_moonshot_phase3.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xterminator.qa_department import QADepartment
from xterminator.orchestration_bridge import (
    OrchestrationBridge,
    QualityGateConfig,
    EscalationLevel,
    ScanResult
)
from xterminator.autofix_policy import AutofixPolicy


def print_banner(text: str):
    """Print a pretty banner"""
    border = "=" * 70
    print(f"\n{border}")
    print(f"{text:^70}")
    print(f"{border}\n")


def print_scan_result(result: ScanResult):
    """Print a scan result"""
    gate_icons = {
        'passed': '✅',
        'warning': '⚠️',
        'failed': '❌',
        'blocked': '🚫'
    }
    icon = gate_icons.get(result.quality_gate_result.value, '?')

    print(f"  Scan: {result.department_scanned} / {result.file_path}")
    print(f"  Issues: {result.issues_found} found "
          f"({result.issues_auto_fixed} fixed, {result.issues_need_review} review, {result.issues_blocked} blocked)")
    print(f"  Quality: {result.quality_score:.2f} | Confidence: {result.confidence:.2f}")
    print(f"  {icon} Gate: {result.quality_gate_result.value} - {result.quality_gate_reason}")

    if result.escalation_level != EscalationLevel.NONE:
        escalation_icons = {
            'notify': '📢',
            'review': '👀',
            'block': '🚫',
            'critical': '🚨'
        }
        esc_icon = escalation_icons.get(result.escalation_level.value, '⚠️')
        print(f"  {esc_icon} Escalation: {result.escalation_level.value} - {result.escalation_reason}")

    print(f"  Duration: {result.scan_duration_ms:.0f}ms\n")


# Mock department outputs (simulated code from various departments)
MOCK_OUTPUTS = {
    'MasterWeaver': {
        'entity_extractor.py': '''
def extract_entities(text):
    """Extract entities from text"""
    entities = []
    # TODO: implement entity extraction
    return entities
''',
        'relationship_mapper.py': '''
def map_relationships(entities):
    """Map relationships between entities"""
    # Missing error handling
    db = connect_database()
    result = db.query(entities)
    return result
'''
    },
    'Infrastructure': {
        'database_migration.py': '''
def migrate_schema(version):
    """Migrate database schema"""
    if version == "v2":
        # Hard-coded connection string (security issue)
        conn = psycopg2.connect("postgresql://user:pass@localhost/db")
        conn.execute("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
    return True
''',
        'cache_manager.py': '''
def clear_cache():
    """Clear application cache"""
    import os
    # Dangerous: removes all files
    os.system("rm -rf /tmp/cache/*")
    return "Cache cleared"
'''
    },
    'Verification': {
        'confidence_scorer.py': '''
def calculate_confidence(data):
    """Calculate confidence score"""
    # Missing input validation
    score = sum(data) / len(data)
    return score
''',
        'result_validator.py': '''
def validate_result(result):
    """Validate result quality"""
    # Good code - no issues
    if result is None:
        raise ValueError("Result cannot be None")
    if not hasattr(result, 'confidence'):
        raise ValueError("Result missing confidence field")
    if result.confidence < 0 or result.confidence > 1:
        raise ValueError(f"Invalid confidence: {result.confidence}")
    return True
'''
    }
}


async def demo_scenario_1_single_scan():
    """
    Scenario 1: Single Department Scan

    Orchestration scans output from MasterWeaver
    """
    print_banner("🐷 SCENARIO 1: Single Department Scan 🐷")
    print("Orchestration scans MasterWeaver entity extractor...\n")

    # Create QA department and bridge
    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Scan MasterWeaver output
    result = await bridge.scan_department_output(
        department_name="MasterWeaver",
        output_code=MOCK_OUTPUTS['MasterWeaver']['entity_extractor.py'],
        file_path="entity_extractor.py"
    )

    print_scan_result(result)


async def demo_scenario_2_quality_gates():
    """
    Scenario 2: Quality Gates

    Different departments with different quality standards
    """
    print_banner("🐷 SCENARIO 2: Quality Gates 🐷")
    print("Different quality standards for different departments...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Register quality gates
    bridge.register_quality_gate(
        "Infrastructure",
        QualityGateConfig.strict("Infrastructure")  # Strict for infra (healthcare-grade)
    )
    bridge.register_quality_gate(
        "MasterWeaver",
        QualityGateConfig.balanced("MasterWeaver")  # Balanced for feature work
    )
    bridge.register_quality_gate(
        "Verification",
        QualityGateConfig.relaxed("Verification")  # Relaxed for verification
    )

    # Scan with strict gate (Infrastructure)
    print("Scanning Infrastructure (STRICT gate):")
    result = await bridge.scan_department_output(
        department_name="Infrastructure",
        output_code=MOCK_OUTPUTS['Infrastructure']['database_migration.py'],
        file_path="database_migration.py"
    )
    print_scan_result(result)

    # Scan with balanced gate (MasterWeaver)
    print("Scanning MasterWeaver (BALANCED gate):")
    result = await bridge.scan_department_output(
        department_name="MasterWeaver",
        output_code=MOCK_OUTPUTS['MasterWeaver']['entity_extractor.py'],
        file_path="entity_extractor.py"
    )
    print_scan_result(result)

    # Scan with relaxed gate (Verification)
    print("Scanning Verification (RELAXED gate):")
    result = await bridge.scan_department_output(
        department_name="Verification",
        output_code=MOCK_OUTPUTS['Verification']['result_validator.py'],
        file_path="result_validator.py"
    )
    print_scan_result(result)


async def demo_scenario_3_error_escalation():
    """
    Scenario 3: Error Escalation

    Critical issues get escalated to humans
    """
    print_banner("🐷 SCENARIO 3: Error Escalation 🐷")
    print("Critical issues trigger escalation...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Register escalation handlers
    escalation_log = []

    async def handle_critical(scan_result: ScanResult):
        """Handle critical escalation"""
        escalation_log.append({
            'level': 'CRITICAL',
            'department': scan_result.department_scanned,
            'file': scan_result.file_path,
            'reason': scan_result.escalation_reason
        })
        print(f"  🚨 CRITICAL ALERT: {scan_result.department_scanned}/{scan_result.file_path}")
        print(f"     Reason: {scan_result.escalation_reason}")
        print(f"     Notifying ops team...\n")

    async def handle_review(scan_result: ScanResult):
        """Handle review escalation"""
        escalation_log.append({
            'level': 'REVIEW',
            'department': scan_result.department_scanned,
            'file': scan_result.file_path,
            'reason': scan_result.escalation_reason
        })
        print(f"  👀 REVIEW REQUIRED: {scan_result.department_scanned}/{scan_result.file_path}")
        print(f"     Reason: {scan_result.escalation_reason}")
        print(f"     Adding to review queue...\n")

    bridge.register_escalation_handler(EscalationLevel.CRITICAL, handle_critical)
    bridge.register_escalation_handler(EscalationLevel.REVIEW, handle_review)

    # Scan dangerous code (cache_manager.py has os.system call)
    print("Scanning dangerous code (os.system):")
    result = await bridge.scan_department_output(
        department_name="Infrastructure",
        output_code=MOCK_OUTPUTS['Infrastructure']['cache_manager.py'],
        file_path="cache_manager.py"
    )
    print_scan_result(result)

    # Scan code needing review
    print("Scanning code needing review:")
    result = await bridge.scan_department_output(
        department_name="MasterWeaver",
        output_code=MOCK_OUTPUTS['MasterWeaver']['relationship_mapper.py'],
        file_path="relationship_mapper.py"
    )
    print_scan_result(result)

    # Show escalation log
    print("Escalation Log:")
    for entry in escalation_log:
        print(f"  [{entry['level']}] {entry['department']}/{entry['file']}: {entry['reason']}")


async def demo_scenario_4_parallel_scanning():
    """
    Scenario 4: Parallel Department Scanning

    Scan multiple departments at once
    """
    print_banner("🐷 SCENARIO 4: Parallel Department Scanning 🐷")
    print("Scanning all departments in parallel...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Prepare scan requests
    scan_requests = []
    for dept_name, files in MOCK_OUTPUTS.items():
        for file_path, code in files.items():
            scan_requests.append({
                'department_name': dept_name,
                'output_code': code,
                'file_path': file_path,
                'metadata': {'parallel_scan': True}
            })

    # Scan all in parallel
    print(f"Scanning {len(scan_requests)} files from {len(MOCK_OUTPUTS)} departments...\n")

    results = await bridge.scan_multiple_departments(scan_requests)

    # Summary
    passed = sum(1 for r in results if r.quality_gate_passed)
    failed = sum(1 for r in results if not r.quality_gate_passed)

    print(f"Results:")
    print(f"  Total Scans: {len(results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Pass Rate: {passed/len(results):.1%}\n")

    # Show details for failed scans
    if failed > 0:
        print("Failed Scans:")
        for result in results:
            if not result.quality_gate_passed:
                print(f"  ❌ {result.department_scanned}/{result.file_path}")
                print(f"     {result.quality_gate_reason}\n")


async def demo_scenario_5_orchestration_metrics():
    """
    Scenario 5: Orchestration Metrics

    View overall system quality metrics
    """
    print_banner("🐷 SCENARIO 5: Orchestration Metrics 🐷")
    print("Viewing overall system quality...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Scan multiple files
    for dept_name, files in MOCK_OUTPUTS.items():
        for file_path, code in files.items():
            await bridge.scan_department_output(
                department_name=dept_name,
                output_code=code,
                file_path=file_path
            )

    # Get metrics
    metrics = bridge.get_metrics()

    print("Orchestration Metrics:")
    print(f"  Total Scans: {metrics['total_scans']}")
    print(f"  Passed: {metrics['passed_scans']}")
    print(f"  Failed: {metrics['failed_scans']}")
    print(f"  Escalated: {metrics['escalated_scans']}")
    print(f"  Pass Rate: {metrics['pass_rate']:.1%}")
    print(f"  Escalation Rate: {metrics['escalation_rate']:.1%}\n")

    # QA department health
    health = await qa_dept.health_check()
    print("QA Department Health:")
    print(f"  Status: {health['status']}")
    print(f"  Success Rate: {health['success_rate']:.1%}")
    print(f"  Avg Latency: {health['avg_latency_ms']:.0f}ms")
    print(f"  Confidence Accuracy: {health['confidence_accuracy']:.1%}")


async def demo_scenario_6_workflow():
    """
    Scenario 6: Complete Orchestration Workflow

    End-to-end workflow showing the full integration
    """
    print_banner("🐷 SCENARIO 6: Complete Orchestration Workflow 🐷")
    print("Simulating complete orchestration cycle...\n")

    qa_dept = QADepartment(policy=AutofixPolicy.balanced())
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Step 1: MasterWeaver produces output
    print("Step 1: MasterWeaver produces entity extractor")
    print("  MasterWeaver: 'I've built an entity extraction module'\n")

    # Step 2: Orchestration requests QA scan
    print("Step 2: Orchestration requests QA scan")
    print("  Orchestration: 'QA, please scan MasterWeaver output'\n")

    result = await bridge.scan_department_output(
        department_name="MasterWeaver",
        output_code=MOCK_OUTPUTS['MasterWeaver']['entity_extractor.py'],
        file_path="entity_extractor.py"
    )

    # Step 3: QA responds with findings
    print("Step 3: QA responds with findings")
    print(f"  QA: 'Scan complete - {result.issues_found} issues found'")
    print(f"      Quality score: {result.quality_score:.2f}")
    print(f"      Confidence: {result.confidence:.2f}")
    print(f"      Gate status: {result.quality_gate_result.value}\n")

    # Step 4: Orchestration decides
    print("Step 4: Orchestration decides")
    if result.quality_gate_passed:
        print("  Orchestration: '✓ Quality gate passed - proceeding to deployment'\n")

        # Step 5: Infrastructure deploys
        print("Step 5: Infrastructure deploys")
        print("  Infrastructure: 'Deploying MasterWeaver entity extractor'")
        print("  Infrastructure: 'Deployment successful ✓'")
    else:
        print(f"  Orchestration: '✗ Quality gate failed - blocking deployment'\n")

        # Step 5: Escalation
        print("Step 5: Escalation")
        print(f"  Orchestration: 'Escalating to human review ({result.escalation_level.value})'")
        print(f"  Orchestration: 'Reason: {result.escalation_reason}'")

    print("\n✓ Workflow complete!")


async def main():
    """Run all demo scenarios"""
    print("\n")
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)
    print("        xTerminator Moonshot Demo - Phase 3".center(70))
    print("      \"From Department to Orchestration!\"".center(70))
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)

    await demo_scenario_1_single_scan()
    await demo_scenario_2_quality_gates()
    await demo_scenario_3_error_escalation()
    await demo_scenario_4_parallel_scanning()
    await demo_scenario_5_orchestration_metrics()
    await demo_scenario_6_workflow()

    print_banner("🐷 Phase 3 Complete! 🐷")
    print("Orchestration Integration Features:")
    print("  ✓ Cross-department scanning (scan any department's output)")
    print("  ✓ Quality gates (strict/balanced/relaxed)")
    print("  ✓ Error escalation (notify/review/block/critical)")
    print("  ✓ Parallel scanning (scan multiple departments at once)")
    print("  ✓ Orchestration metrics (pass rate, escalation rate)")
    print("  ✓ Complete workflows (MasterWeaver → QA → Infrastructure)")
    print("\nIntegration Points:")
    print("  ✓ MasterWeaver outputs → QA scanning")
    print("  ✓ Infrastructure migrations → QA validation")
    print("  ✓ Verification results → QA quality check")
    print("  ✓ All departments → centralized quality enforcement")
    print("\nNext steps:")
    print("  - Phase 4 (Weeks 8-10): Thompson Sampling (adaptive learning)")
    print("  - Phase 5 (Weeks 11-18): Advanced features (marketplace, semantic)")
    print("\n(*)<  OINK OINK OINK!")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
