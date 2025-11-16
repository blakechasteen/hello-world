"""
Compliance Monitoring Demo

Demonstrates automated compliance monitoring across SOC2, GDPR, and ISO 27001.

**Created: 2025-11-16**

Usage:
    PYTHONPATH=. python demos/demo_compliance_monitoring.py
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from HoloLoom.security.compliance import (
    ComplianceMonitor,
    ComplianceFramework,
    SOC2Monitor,
    GDPRMonitor,
    ISO27001Monitor,
    DataSubjectRight,
    EvidenceCollector,
    ComplianceReporter,
    generate_compliance_report,
)


async def demo_compliance_monitoring():
    """Demonstrate comprehensive compliance monitoring"""

    print("=" * 80)
    print("HoloLoom Compliance Monitoring Demo")
    print("=" * 80)
    print()

    # 1. Core Compliance Monitor
    print("1. Core Compliance Monitor")
    print("-" * 80)

    monitor = ComplianceMonitor(
        frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
        evidence_dir=Path("./demo_evidence"),
        auto_collect=True
    )

    print(f"✓ Monitoring {len(monitor.frameworks)} frameworks")
    print(f"  - {', '.join(f.value for f in monitor.frameworks)}")
    print()

    # Run automated checks
    print("Running automated compliance checks...")
    await monitor._run_automated_checks()
    print("✓ Automated checks completed")
    print()

    # Check SOC2 compliance
    soc2_status = await monitor.check_compliance(ComplianceFramework.SOC2)
    print(f"SOC2 Compliance Score: {soc2_status.overall_score:.1%}")
    print(f"  - Controls Implemented: {soc2_status.controls_implemented}/{soc2_status.controls_total}")
    print(f"  - Controls Verified: {soc2_status.controls_verified}/{soc2_status.controls_total}")
    print(f"  - Audit Ready: {'YES ✓' if soc2_status.audit_ready else 'NO ✗'}")
    print(f"  - Open Gaps: {len(soc2_status.gaps)}")
    print()

    # 2. SOC2 Type II Monitor
    print("2. SOC2 Type II Monitor")
    print("-" * 80)

    soc2_monitor = SOC2Monitor(
        evidence_dir=Path("./demo_soc2_evidence")
    )

    print(f"✓ {len(soc2_monitor.controls)} SOC2 controls defined")
    print()

    # Sample control check
    print("Sample Control: CC6.1 (Password Policy)")
    cc61 = soc2_monitor.controls["CC6.1"]
    print(f"  - Title: {cc61.title}")
    print(f"  - Implemented: {cc61.implemented}")
    print(f"  - Automated: {cc61.automated}")
    print(f"  - Effective: {soc2_monitor.check_control_effectiveness('CC6.1')}")
    print()

    # Collect evidence
    print("Collecting evidence for audit period...")
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    evidence = await soc2_monitor.collect_evidence(start_date, end_date)
    print(f"✓ Collected {sum(len(v) for v in evidence.values())} evidence items")
    print()

    # Readiness report
    readiness = soc2_monitor.generate_readiness_report()
    print(f"SOC2 Readiness Score: {readiness['readiness_score']:.1%}")
    print(f"  - Audit Ready: {'YES ✓' if readiness['audit_ready'] else 'NO ✗'}")
    print(f"  - Gaps: {len(readiness['gaps'])}")
    if readiness['recommendations']:
        print(f"  - Recommendations:")
        for rec in readiness['recommendations']:
            print(f"    • {rec}")
    print()

    # 3. GDPR Monitor
    print("3. GDPR Compliance Monitor")
    print("-" * 80)

    gdpr_monitor = GDPRMonitor(
        data_controller="HoloLoom Inc.",
        dpo_email="dpo@hololoom.ai"
    )

    print(f"✓ {len(gdpr_monitor.processing_activities)} processing activities documented")
    print()

    # Conduct DPIA
    print("Conducting Data Protection Impact Assessment (DPIA)...")
    dpia = gdpr_monitor.conduct_dpia(
        processing_description="AI model training on user interactions",
        purposes=["Service improvement", "Feature development"],
        risks=["Re-identification", "Bias amplification"],
        assessor="Security Team"
    )

    print(f"✓ DPIA ID: {dpia.assessment_id}")
    print(f"  - Risks Identified: {len(dpia.risks_identified)}")
    print(f"  - Mitigations: {len(dpia.risk_mitigation)}")
    print(f"  - Conclusion: {dpia.conclusion.upper()}")
    print()

    # Handle data subject request
    print("Processing Data Subject Request...")
    request = gdpr_monitor.receive_request(
        request_type=DataSubjectRight.ACCESS,
        email="user@example.com"
    )

    print(f"✓ Request ID: {request.request_id}")
    print(f"  - Type: {request.request_type.value}")
    print(f"  - Due Date: {request.due_date.date()}")
    print()

    data_package = gdpr_monitor.process_access_request(request)
    print(f"✓ Access request processed")
    print(f"  - Status: {request.status}")
    print(f"  - Data Categories: {len(data_package['personal_data'])}")
    print()

    # GDPR compliance check
    gdpr_status = gdpr_monitor.check_compliance()
    print(f"GDPR Compliance Score: {gdpr_status['compliance_score']:.1%}")
    print(f"  - Compliant: {'YES ✓' if gdpr_status['compliant'] else 'NO ✗'}")
    print(f"  - Pending DSRs: {gdpr_status['data_subject_requests']['pending']}")
    print()

    # 4. ISO 27001 Monitor
    print("4. ISO 27001 Compliance Monitor")
    print("-" * 80)

    iso_monitor = ISO27001Monitor(
        organization="HoloLoom Inc."
    )

    print(f"✓ {len(iso_monitor.controls)} ISO 27001 controls implemented")
    print()

    # Risk assessment
    print("Conducting Risk Assessment...")
    risks = iso_monitor.conduct_risk_assessment()

    print(f"✓ {len(risks)} risks identified")
    for risk in risks[:3]:  # Show top 3
        print(f"  - {risk['risk_id']}: {risk['threat']}")
        print(f"    Impact: {risk['impact'].upper()}, Likelihood: {risk['likelihood'].upper()}")
        print(f"    Residual Risk: {risk['residual_risk'].upper()}")
    print()

    # ISMS status
    isms_status = iso_monitor.get_isms_status()
    print(f"ISMS Status:")
    print(f"  - Controls: {isms_status.controls_implemented}/{isms_status.controls_total}")
    print(f"  - Certification: {isms_status.certification_status.upper()}")
    if isms_status.certification_date:
        print(f"  - Certified: {isms_status.certification_date.date()}")
        print(f"  - Expiry: {isms_status.certificate_expiry.date()}")
    print()

    # 5. Evidence Collection
    print("5. Automated Evidence Collection")
    print("-" * 80)

    evidence_collector = EvidenceCollector()

    print("Collecting audit evidence package...")
    package = await evidence_collector.collect_audit_package(
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now()
    )

    total_evidence = sum(len(v) for v in package.values())
    print(f"✓ Collected {total_evidence} evidence items")
    print(f"  - Access Logs: {len(package['access_logs'])}")
    print(f"  - Change Logs: {len(package['change_logs'])}")
    print(f"  - Training Records: {len(package['training_records'])}")
    print()

    # 6. Compliance Reporting
    print("6. Compliance Reporting")
    print("-" * 80)

    reporter = ComplianceReporter(
        output_dir=Path("./demo_compliance_reports")
    )

    # Monthly report
    print("Generating Monthly Compliance Report...")
    monthly_report = await reporter.generate_monthly_report()

    print(f"✓ Report generated: {monthly_report['report_month']}")
    print(f"  - Overall Status: {monthly_report['overall_status']['compliance_posture'].upper()}")
    print(f"  - Audit Ready: {'YES ✓' if monthly_report['overall_status']['audit_ready'] else 'NO ✗'}")
    print(f"  - Critical Gaps: {monthly_report['overall_status']['critical_gaps']}")
    print()

    print("Framework Scores:")
    print(f"  - SOC2: {monthly_report['soc2']['score']:.1%}")
    print(f"  - GDPR: {monthly_report['gdpr']['score']:.1%}")
    print(f"  - ISO 27001: {monthly_report['iso27001']['controls_implemented']}/{monthly_report['iso27001']['controls_total']}")
    print()

    # Board report
    print("Generating Quarterly Board Report...")
    board_report = reporter.generate_board_report()

    print(f"✓ Board report generated: {board_report['quarter']}")
    print(f"  - Overall Score: {board_report['executive_summary']['overall_compliance_score']:.1%}")
    print(f"  - Critical Findings: {board_report['executive_summary']['critical_findings']}")
    print(f"  - High Findings: {board_report['executive_summary']['high_findings']}")
    print()

    # Audit package
    print("Generating Audit Package...")
    audit_package = await reporter.generate_audit_package(
        framework=ComplianceFramework.SOC2
    )

    print(f"✓ Audit package generated")
    print(f"  - Framework: {audit_package['framework'].upper()}")
    print(f"  - Period: {audit_package['audit_period']['start'][:10]} to {audit_package['audit_period']['end'][:10]}")
    print(f"  - Audit Ready: {'YES ✓' if audit_package['audit_ready'] else 'NO ✗'}")
    print(f"  - Evidence Items: {sum(audit_package['evidence'].values())}")
    print()

    # 7. Compliance Score
    print("7. Overall Compliance Score")
    print("-" * 80)

    for framework in [ComplianceFramework.SOC2, ComplianceFramework.GDPR, ComplianceFramework.ISO27001]:
        score = reporter.calculate_compliance_score(framework)
        print(f"{framework.value.upper()}: {score.score:.1%} (Grade: {score.grade})")
        print(f"  - Audit Ready: {'YES ✓' if score.audit_ready else 'NO ✗'}")

    print()

    # Summary
    print("=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print()
    print("✓ Compliance monitoring across 3 frameworks")
    print("✓ Automated control testing and evidence collection")
    print("✓ GDPR DPIA and data subject request handling")
    print("✓ ISO 27001 risk assessment and ISMS management")
    print("✓ Automated reporting (monthly, quarterly, audit packages)")
    print("✓ Audit-ready evidence packages generated")
    print()
    print("Reports saved to: ./demo_compliance_reports/")
    print("Evidence saved to: ./demo_evidence/")
    print()


if __name__ == "__main__":
    asyncio.run(demo_compliance_monitoring())
