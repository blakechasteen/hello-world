#!/usr/bin/env python3
"""
Compliance Report Generator CLI

Command-line tool for generating compliance reports and audit packages.

**Created: 2025-11-16**

Usage:
    # Generate monthly report
    python scripts/generate_compliance_report.py --type monthly --output ./reports

    # Generate board report
    python scripts/generate_compliance_report.py --type board --output ./reports

    # Generate SOC2 audit package
    python scripts/generate_compliance_report.py --type audit --framework soc2 --output ./audit

    # Generate GDPR audit package
    python scripts/generate_compliance_report.py --type audit --framework gdpr --output ./audit

    # Check compliance status
    python scripts/generate_compliance_report.py --type status --framework soc2
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.security.compliance import (
    ComplianceReporter,
    ComplianceFramework,
    generate_compliance_report,
    generate_audit_package,
)


def print_banner():
    """Print CLI banner"""
    print("=" * 80)
    print("HoloLoom Compliance Report Generator")
    print("=" * 80)
    print()


def print_status(framework_name: str, score: float, audit_ready: bool):
    """Print compliance status"""
    grade = "A+" if score >= 0.97 else "A" if score >= 0.93 else "B" if score >= 0.85 else "C"

    status_icon = "✓" if audit_ready else "✗"
    status_text = "AUDIT READY" if audit_ready else "NOT READY"

    print(f"{framework_name}:")
    print(f"  Score: {score:.1%} (Grade: {grade})")
    print(f"  Status: {status_text} {status_icon}")
    print()


async def generate_monthly(output_dir: Path):
    """Generate monthly compliance report"""
    print("Generating monthly compliance report...")
    print()

    reporter = ComplianceReporter(output_dir=output_dir)
    report = await reporter.generate_monthly_report()

    print(f"✓ Monthly report generated")
    print(f"  Period: {report['report_month']}")
    print(f"  Output: {output_dir}")
    print()

    # Display summary
    print("Compliance Summary:")
    print_status("SOC2", report['soc2']['score'], report['soc2']['audit_ready'])
    print_status("GDPR", report['gdpr']['score'], report['gdpr']['compliant'])

    iso_score = report['iso27001']['controls_implemented'] / report['iso27001']['controls_total']
    print_status("ISO 27001", iso_score, report['iso27001']['certification_status'] == "certified")

    print("Key Metrics:")
    print(f"  MFA Adoption: {report['metrics']['mfa_adoption']:.1%}")
    print(f"  Backup Success: {report['metrics']['backup_success_rate']:.1%}")
    print(f"  Patching SLA: {'MET ✓' if report['metrics']['patching_sla_met'] else 'MISSED ✗'}")
    print()

    return report


async def generate_board(output_dir: Path):
    """Generate quarterly board report"""
    print("Generating quarterly board report...")
    print()

    reporter = ComplianceReporter(output_dir=output_dir)
    report = reporter.generate_board_report()

    print(f"✓ Board report generated")
    print(f"  Quarter: {report['quarter']}")
    print(f"  Output: {output_dir}")
    print()

    # Display executive summary
    print("Executive Summary:")
    print(f"  Overall Score: {report['executive_summary']['overall_compliance_score']:.1%}")
    print(f"  Audit Ready: {'YES ✓' if report['executive_summary']['audit_ready'] else 'NO ✗'}")
    print(f"  Critical Findings: {report['executive_summary']['critical_findings']}")
    print(f"  High Findings: {report['executive_summary']['high_findings']}")
    print()

    print("Action Items:")
    for item in report['action_items']:
        print(f"  • {item}")
    print()

    return report


async def generate_audit(framework: str, output_dir: Path, start_date: datetime, end_date: datetime):
    """Generate audit package"""
    print(f"Generating {framework.upper()} audit package...")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print()

    package = await generate_audit_package(
        framework=framework,
        start_date=start_date,
        end_date=end_date
    )

    print(f"✓ Audit package generated")
    print(f"  Framework: {package['framework'].upper()}")
    print(f"  Output: {output_dir}")
    print()

    print("Package Contents:")
    print(f"  Audit Ready: {'YES ✓' if package['audit_ready'] else 'NO ✗'}")
    print(f"  Evidence Items: {sum(package['evidence'].values())}")
    print(f"    - Access Logs: {package['evidence'].get('access_logs', 0)}")
    print(f"    - Change Logs: {package['evidence'].get('change_logs', 0)}")
    print(f"    - Training Records: {package['evidence'].get('training_records', 0)}")
    print()

    # Save package summary
    summary_path = output_dir / f"audit_package_{framework}_{datetime.now().date()}.json"
    with open(summary_path, 'w') as f:
        json.dump(package, f, indent=2)

    print(f"Package summary saved: {summary_path}")
    print()

    return package


async def check_status(framework: str):
    """Check compliance status"""
    print(f"Checking {framework.upper()} compliance status...")
    print()

    reporter = ComplianceReporter()

    if framework == "soc2":
        report = reporter.soc2_monitor.generate_readiness_report()
        score = report['readiness_score']
        audit_ready = report['audit_ready']

        print_status(framework.upper(), score, audit_ready)

        print(f"Controls:")
        print(f"  Total: {report['controls']['total']}")
        print(f"  Implemented: {report['controls']['implemented']}")
        print(f"  Tested: {report['controls']['tested']}")
        print(f"  Effective: {report['controls']['effective']}")
        print()

        if report['gaps']:
            print(f"Gaps ({len(report['gaps'])}):")
            for gap in report['gaps'][:5]:  # Show top 5
                print(f"  • {gap}")
            print()

    elif framework == "gdpr":
        status = reporter.gdpr_monitor.check_compliance()
        score = status['compliance_score']
        compliant = status['compliant']

        print_status(framework.upper(), score, compliant)

        print("GDPR Checks:")
        for article, check in status['checks'].items():
            icon = "✓" if check['compliant'] else "✗"
            print(f"  {icon} {article}: {check['details']}")
        print()

    elif framework == "iso27001":
        isms = reporter.iso27001_monitor.get_isms_status()
        score = isms.controls_implemented / isms.controls_total

        print_status(framework.upper(), score, isms.certification_status == "certified")

        print("ISMS Status:")
        print(f"  Controls: {isms.controls_implemented}/{isms.controls_total}")
        print(f"  Certification: {isms.certification_status.upper()}")
        if isms.last_risk_assessment:
            print(f"  Last Risk Assessment: {isms.risk_assessment_date.date()}")
        print()

    else:
        print(f"Unknown framework: {framework}")
        print("Supported frameworks: soc2, gdpr, iso27001")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate compliance reports and audit packages"
    )

    parser.add_argument(
        "--type",
        choices=["monthly", "board", "audit", "status"],
        required=True,
        help="Report type to generate"
    )

    parser.add_argument(
        "--framework",
        choices=["soc2", "gdpr", "iso27001"],
        help="Framework (required for audit and status types)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./compliance_reports"),
        help="Output directory (default: ./compliance_reports)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Audit period start date (YYYY-MM-DD, default: 365 days ago)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Audit period end date (YYYY-MM-DD, default: today)"
    )

    args = parser.parse_args()

    # Validate framework for audit and status types
    if args.type in ["audit", "status"] and not args.framework:
        parser.error(f"--framework is required for {args.type} type")

    # Parse dates
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    else:
        start_date = datetime.now() - timedelta(days=365)

    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)
    else:
        end_date = datetime.now()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Print banner
    print_banner()

    # Run appropriate command
    if args.type == "monthly":
        asyncio.run(generate_monthly(args.output))

    elif args.type == "board":
        asyncio.run(generate_board(args.output))

    elif args.type == "audit":
        asyncio.run(generate_audit(args.framework, args.output, start_date, end_date))

    elif args.type == "status":
        asyncio.run(check_status(args.framework))

    print("Done!")


if __name__ == "__main__":
    main()
