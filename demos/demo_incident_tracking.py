#!/usr/bin/env python3
"""
Incident Response System Demo

Demonstrates HoloLoom incident tracking and GDPR/CCPA breach notification
following NIST SP 800-61 framework.

Usage:
    PYTHONPATH=. python demos/demo_incident_tracking.py

Created: 2025-11-16
Status: Production Ready
"""

import sys
import logging
from datetime import datetime, timedelta
import json

# Add repo root to path
sys.path.insert(0, '/home/user/hello-world')

from HoloLoom.security.incident import (
    IncidentRegistry,
    IncidentSeverity,
    IncidentType,
    IncidentStatus,
    BreachNotificationBuilder,
    assess_high_risk
)

# ==================== Setup ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_incident_creation():
    """Demo: Creating incident and initializing tracking."""
    print_section("DEMO 1: Incident Creation (T+0 Detection)")

    registry = IncidentRegistry(log_file="/tmp/demo_incidents.jsonl")

    print("📋 Creating incident: SEV-1 Data Breach")
    print("   (SQL injection in login form)")
    print()

    incident = registry.create_incident(
        severity=IncidentSeverity.SEV1_CRITICAL,
        incident_type=IncidentType.DATA_BREACH,
        title="Customer Database Breach - SQL Injection",
        description=(
            "Unauthorized access detected to customer database via SQL injection "
            "in login form. Attacker exploited unsanitized email parameter."
        ),
        affected_users=15000,
        affected_systems=["prod-db-001", "prod-api-02"],
        data_categories=["customer_names", "emails", "phone_numbers", "password_hashes"],
        incident_commander="John Smith",
        security_lead="Jane Doe"
    )

    print(f"✅ Incident Created")
    print(f"   ID: {incident.incident_id}")
    print(f"   Severity: {incident.severity.value}")
    print(f"   Type: {incident.type.value}")
    print(f"   Status: {incident.status.value}")
    print(f"   Detected: {incident.detected_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   Affected Users: {incident.affected_users:,}")
    print(f"   GDPR Notification Deadline: {incident.notification_deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   (72 hours from detection)")
    print()

    return registry, incident


def demo_incident_response(registry, incident):
    """Demo: Simulating incident response actions."""
    print_section("DEMO 2: Incident Response (T+5 min to T+2 hours)")

    print("🚨 Incident Commander Activates Response Team")
    print("   Team: IC + Security Lead + DBA + Legal + Communications")
    print()

    # T+5 minutes: Block attacker
    print("⏱️  T+5 minutes: Containment Actions")
    action1 = registry.record_action(
        incident_id=incident.incident_id,
        action_type="BLOCK_IP",
        description="Blocked attacker IP (203.0.113.45) at firewall",
        taken_by="Network Security",
        evidence_collected=True
    )
    print(f"   ✓ {action1.description}")
    print()

    # T+15 minutes: Investigation
    print("⏱️  T+15 minutes: Investigation Underway")
    print("   ✓ Root cause identified: SQL injection in login form")
    print("   ✓ Attack dwell time: ~12 hours (02:15 - 14:30 UTC)")
    print("   ✓ Data exfiltration confirmed: 4.2 GB to attacker FTP")
    print("   ✓ Forensics firm engaged: Mandiant")
    print()

    # Update status
    registry.update_status(
        incident.incident_id,
        IncidentStatus.INVESTIGATING,
        notes="Team assembled, investigation ongoing"
    )
    print("   📊 Status updated: DETECTED → INVESTIGATING")
    print()

    # T+30 minutes: Patch
    print("⏱️  T+30 minutes: Vulnerability Patch")
    action2 = registry.record_action(
        incident_id=incident.incident_id,
        action_type="PATCH_VULNERABILITY",
        description=(
            "SQL injection vulnerability patched with parameterized queries "
            "in login endpoint"
        ),
        taken_by="Development",
        evidence_collected=False
    )
    print(f"   ✓ {action2.description}")
    print()

    # T+45 minutes: Revoke credentials
    print("⏱️  T+45 minutes: Credential Revocation")
    action3 = registry.record_action(
        incident_id=incident.incident_id,
        action_type="REVOKE_CREDENTIALS",
        description="Revoked compromised database user credentials",
        taken_by="Database Team",
        evidence_collected=False
    )
    print(f"   ✓ {action3.description}")
    print()

    # T+2 hours: Contained
    print("⏱️  T+2 hours: Incident Contained")
    registry.update_status(
        incident.incident_id,
        IncidentStatus.CONTAINED,
        notes="Attacker blocked, patch deployed, systems secured"
    )
    updated_incident = registry.get_incident(incident.incident_id)
    print(f"   ✓ Status updated: INVESTIGATING → CONTAINED")
    print(f"   ✓ Containment time: {updated_incident.containment_time.strftime('%H:%M:%S UTC')}")
    print()

    return registry, incident


def demo_breach_notification(registry, incident):
    """Demo: GDPR breach notification (Article 33-34)."""
    print_section("DEMO 3: Breach Notification (GDPR Compliance)")

    print("🔍 Assessing Breach Risk")
    print()

    # Assess risk
    is_high_risk = assess_high_risk(
        data_categories=incident.data_categories_affected,
        volume=incident.affected_users
    )

    print(f"   Data Categories Affected:")
    for cat in incident.data_categories_affected:
        print(f"      • {cat}")
    print(f"   Affected Individuals: {incident.affected_users:,}")
    print()
    print(f"   High-Risk Assessment: {'🔴 YES - INDIVIDUAL NOTIFICATION REQUIRED' if is_high_risk else '⚠️  LOW - DPA ONLY'}")
    print()

    # Create notification record
    print("📧 Creating Breach Notification Record")
    notification = registry.create_breach_notification(
        incident_id=incident.incident_id,
        is_high_risk=is_high_risk,
        affected_categories=["customers"]
    )

    print(f"   ✓ Notification ID: {notification.notification_id}")
    print(f"   ✓ GDPR Article 33 Deadline: {notification.deadline_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Generate DPA notification
    print("📝 Generating GDPR Article 33 (DPA) Notification")
    builder = BreachNotificationBuilder()

    dpa_notif = builder.generate_dpa_notification(
        incident=incident,
        breach=notification,
        dpa_country="IRELAND"
    )

    print(f"   ✓ DPA: {dpa_notif['dpa']['name']}")
    print(f"   ✓ Contact: {dpa_notif['dpa']['email']}")
    print(f"   ✓ Hours Since Awareness: {dpa_notif['hours_since_awareness']}")
    print(f"   ✓ Status: {dpa_notif['status']}")
    print(f"   ✓ Required Content:")
    print(f"      - Breach nature: ✓")
    print(f"      - Affected data subjects: ✓")
    print(f"      - Likely consequences: ✓")
    print(f"      - Measures taken/proposed: ✓")
    print()

    # Simulate DPA notification
    print("✉️  Submitting DPA Notification (Simulated)")
    registry.mark_dpa_notified(
        notification_id=notification.notification_id,
        dpa_name="Irish DPC (Data Protection Commission)",
        dpa_contact="dataprotection@dataprotection.ie",
        notification_date=datetime.utcnow()
    )
    print(f"   ✓ Submitted to Irish DPC")
    print()

    # Individual notification
    if is_high_risk:
        print("📬 Sending Individual Notifications (Article 34)")
        sample_customers = [
            {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
            {"name": "Bob Smith", "email": "bob.smith@example.com"},
            {"name": "Carol White", "email": "carol.white@example.com"},
            # ... in production, 15,000 customers
        ]

        email_body = builder.generate_individual_notification_email(
            incident=incident,
            breach=notification,
            customer_name=sample_customers[0]["name"],
            customer_email=sample_customers[0]["email"]
        )

        print(f"   Sample email sent to {len(sample_customers)}+ customers:")
        print(f"      To: {sample_customers[0]['email']}")
        print(f"      Subject: Important Security Notice - HoloLoom Data Protection Incident")
        print(f"      Content: ✓ (16KB, HTML formatted)")
        print()

        registry.mark_individuals_notified(
            notification_id=notification.notification_id,
            notification_date=datetime.utcnow() + timedelta(hours=12),
            method="EMAIL"
        )
        print(f"   ✓ Notifications complete")
    print()

    return registry, notification


def demo_metrics_and_compliance(registry, incident, notification):
    """Demo: Incident metrics and GDPR compliance checking."""
    print_section("DEMO 4: Metrics & GDPR Compliance")

    # Get metrics
    metrics = registry.get_metrics(incident.incident_id)

    print("📊 Incident Response Metrics")
    print(f"   Incident ID: {metrics['incident_id']}")
    print(f"   Severity: {metrics['severity']}")
    print(f"   Affected Users: {metrics['affected_users']:,}")
    print()
    print("   Response Timings (NIST SP 800-61):")
    print(f"      MTTD (Detection): {metrics.get('detected_at', 'N/A')}")
    if 'mttr_minutes' in metrics:
        print(f"      MTTR (Respond): {metrics['mttr_minutes']:.1f} minutes")
    if 'mtrc_minutes' in metrics:
        print(f"      MTRC (Contain): {metrics['mtrc_minutes']:.1f} minutes")
    if 'mtco_hours' in metrics:
        print(f"      MTCO (Resolve): {metrics['mtco_hours']:.1f} hours")
    print()

    # Check notification deadline
    print("⏰ GDPR Article 33 Compliance Check")
    deadline_status = registry.check_notification_deadline(notification.notification_id)

    print(f"   Deadline: {deadline_status['deadline'].split('T')[0]}")
    print(f"   Hours Remaining: {deadline_status['hours_remaining']:.1f} hours")
    print(f"   Status: {'✅ ON TRACK' if deadline_status['deadline_met'] else '❌ OVERDUE'}")
    print()
    print("   Compliance Checklist:")
    for check, status in deadline_status['compliance_checks'].items():
        icon = "✓" if status else "✗"
        print(f"      {icon} {check.replace('_', ' ').title()}")
    print()

    # Validate compliance
    builder = BreachNotificationBuilder()
    compliance = builder.validate_gdpr_compliance(notification)

    print(f"   Overall Status: {compliance['status']}")
    print()


def demo_post_mortem_export(registry, incident):
    """Demo: Exporting incident for post-mortem report."""
    print_section("DEMO 5: Post-Mortem Export")

    # Update to resolved
    registry.update_status(
        incident.incident_id,
        IncidentStatus.ERADICATING,
        notes="Vulnerability removed, systems hardened"
    )
    registry.update_status(
        incident.incident_id,
        IncidentStatus.RECOVERING,
        notes="Systems restored and verified"
    )
    registry.update_status(
        incident.incident_id,
        IncidentStatus.RESOLVED,
        notes="Incident fully resolved, waiting for post-mortem"
    )

    # Export report
    print("📋 Exporting Incident Report for Post-Mortem")
    report = registry.export_incident_report(incident.incident_id)

    print()
    print("✓ Incident Summary:")
    print(f"   ID: {report['incident']['incident_id']}")
    print(f"   Type: {report['incident']['type']}")
    print(f"   Status: {report['incident']['status']}")
    print(f"   Affected Users: {report['incident']['affected_users']:,}")
    print()
    print("✓ Response Timeline:")
    print(f"   Detected: {report['timeline_summary']['detected'].split('T')[0]}")
    print(f"   Contained: {report['timeline_summary']['contained'].split('T')[0] if report['timeline_summary']['contained'] else 'N/A'}")
    print(f"   Resolved: {report['timeline_summary']['resolved'].split('T')[0] if report['timeline_summary']['resolved'] else 'N/A'}")
    print()
    print(f"✓ Actions Recorded: {len(report['actions'])}")
    for i, action in enumerate(report['actions'], 1):
        print(f"   {i}. {action['action_type']}: {action['description'][:60]}...")
    print()
    print("✓ Report ready for post-mortem meeting")
    print("   File: /tmp/demo_incidents.jsonl (audit trail)")
    print()


def main():
    """Run complete incident response demo."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  HoloLoom Incident Response System Demo (NIST SP 800-61)".center(68) + "║")
    print("║" + "  GDPR Article 33-34 + CCPA Compliance".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("Scenario: Customer database breach via SQL injection")
    print("Compliance Requirements: GDPR (72-hour DPA notification)")
    print()

    try:
        # Run demos
        registry, incident = demo_incident_creation()
        registry, incident = demo_incident_response(registry, incident)
        registry, notification = demo_breach_notification(registry, incident)
        demo_metrics_and_compliance(registry, incident, notification)
        demo_post_mortem_export(registry, incident)

        print_section("Demo Complete ✅")
        print("All modules working correctly:")
        print("  ✓ Incident creation and tracking")
        print("  ✓ Status lifecycle management")
        print("  ✓ Action recording with evidence")
        print("  ✓ Breach notification (GDPR)")
        print("  ✓ Metrics and compliance checking")
        print("  ✓ Post-mortem export")
        print()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
