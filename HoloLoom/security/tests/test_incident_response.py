"""
Tests for Incident Response System

Tests core incident tracking and GDPR/CCPA compliance features.

Created: 2025-11-16
"""

import pytest
from datetime import datetime, timedelta
from HoloLoom.security.incident import (
    IncidentRegistry,
    IncidentSeverity,
    IncidentType,
    IncidentStatus,
    NotificationStatus,
    BreachNotificationBuilder,
    get_notification_deadline,
    assess_high_risk
)


class TestIncidentCreation:
    """Test incident creation and initialization."""

    def test_create_incident(self):
        """Test creating a new incident."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test Breach",
            description="Test description",
            affected_users=1000,
            affected_systems=["db-001", "api-01"],
            data_categories=["names", "emails"],
            incident_commander="Test IC",
            security_lead="Test SL"
        )

        assert incident is not None
        assert incident.incident_id.startswith("INC-2025")
        assert incident.severity == IncidentSeverity.SEV1_CRITICAL
        assert incident.type == IncidentType.DATA_BREACH
        assert incident.status == IncidentStatus.DETECTED
        assert incident.affected_users == 1000
        assert len(incident.affected_systems) == 2

    def test_incident_notification_deadline(self):
        """Test GDPR 72-hour notification deadline calculation."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        # Deadline should be 72 hours from creation
        time_diff = (incident.notification_deadline - incident.detected_time).total_seconds()
        assert time_diff == 72 * 3600  # Exactly 72 hours

    def test_retrieve_incident(self):
        """Test retrieving incident by ID."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident1 = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        retrieved = registry.get_incident(incident1.incident_id)
        assert retrieved is not None
        assert retrieved.incident_id == incident1.incident_id


class TestIncidentStatusTransitions:
    """Test incident status lifecycle."""

    def test_status_progression(self):
        """Test incident status transitions."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        # DETECTED → INVESTIGATING
        assert registry.update_status(incident.incident_id, IncidentStatus.INVESTIGATING)
        assert registry.get_incident(incident.incident_id).status == IncidentStatus.INVESTIGATING

        # INVESTIGATING → CONTAINED
        assert registry.update_status(incident.incident_id, IncidentStatus.CONTAINED)
        updated = registry.get_incident(incident.incident_id)
        assert updated.status == IncidentStatus.CONTAINED
        assert updated.containment_time is not None

        # CONTAINED → RESOLVED
        assert registry.update_status(incident.incident_id, IncidentStatus.RESOLVED)
        updated = registry.get_incident(incident.incident_id)
        assert updated.status == IncidentStatus.RESOLVED
        assert updated.resolution_time is not None

    def test_invalid_incident_update(self):
        """Test updating non-existent incident."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        result = registry.update_status("INVALID-ID", IncidentStatus.INVESTIGATING)
        assert result is False


class TestIncidentActions:
    """Test recording incident response actions."""

    def test_record_action(self):
        """Test recording an incident action."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        action = registry.record_action(
            incident_id=incident.incident_id,
            action_type="BLOCK_IP",
            description="Blocked attacker IP",
            taken_by="Test User",
            evidence_collected=True
        )

        assert action is not None
        assert action.action_id.startswith("ACT-")
        assert action.action_type == "BLOCK_IP"
        assert action.evidence_collected is True

    def test_multiple_actions(self):
        """Test recording multiple actions."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        # Record multiple actions
        actions_recorded = []
        for i in range(5):
            action = registry.record_action(
                incident_id=incident.incident_id,
                action_type=f"ACTION_{i}",
                description=f"Test action {i}",
                taken_by="Test User"
            )
            actions_recorded.append(action)

        assert len(actions_recorded) == 5
        assert len(registry.actions[incident.incident_id]) == 5


class TestBreachNotification:
    """Test breach notification functionality."""

    def test_create_breach_notification(self):
        """Test creating a breach notification."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test Breach",
            description="Test",
            affected_users=15000,
            data_categories=["names", "emails", "phone_numbers"]
        )

        notification = registry.create_breach_notification(
            incident_id=incident.incident_id,
            is_high_risk=True,
            affected_categories=["customers"]
        )

        assert notification is not None
        assert notification.notification_id.startswith("BRN-")
        assert notification.incident_id == incident.incident_id
        assert notification.is_high_risk is True
        assert notification.requires_individual_notification is True

    def test_mark_dpa_notified(self):
        """Test marking DPA notification as complete."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        notification = registry.create_breach_notification(
            incident_id=incident.incident_id,
            is_high_risk=True
        )

        # Mark DPA notified
        result = registry.mark_dpa_notified(
            notification_id=notification.notification_id,
            dpa_name="Irish DPC",
            dpa_contact="dataprotection@dataprotection.ie"
        )

        assert result is True
        updated_notification = registry.notifications[notification.notification_id]
        assert updated_notification.dpa_notification_date is not None
        assert updated_notification.dpa_name == "Irish DPC"

    def test_mark_individuals_notified(self):
        """Test marking individuals as notified."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test",
            affected_users=1000
        )

        notification = registry.create_breach_notification(
            incident_id=incident.incident_id,
            is_high_risk=True
        )

        # Mark individuals notified
        result = registry.mark_individuals_notified(
            notification_id=notification.notification_id,
            method="EMAIL"
        )

        assert result is True
        updated_notification = registry.notifications[notification.notification_id]
        assert updated_notification.individual_notification_date is not None


class TestMetrics:
    """Test incident response metrics (MTTD, MTTR, MTRC)."""

    def test_calculate_metrics(self):
        """Test calculating incident metrics."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        # Simulate response actions
        registry.record_action(
            incident_id=incident.incident_id,
            action_type="BLOCK_IP",
            description="Block attacker",
            taken_by="User1"
        )

        # Update status to contained
        registry.update_status(incident.incident_id, IncidentStatus.INVESTIGATING)
        registry.update_status(incident.incident_id, IncidentStatus.CONTAINED)

        # Get metrics
        metrics = registry.get_metrics(incident.incident_id)

        assert metrics["incident_id"] == incident.incident_id
        assert "detected_at" in metrics
        assert "mttr_minutes" in metrics
        assert "mtrc_minutes" in metrics
        assert metrics["mtrc_minutes"] >= 0  # Containment time non-negative


class TestNotificationDeadline:
    """Test GDPR notification deadline functionality."""

    def test_notification_deadline_calculation(self):
        """Test calculating GDPR 72-hour deadline."""
        breach_time = datetime.utcnow()
        deadline = get_notification_deadline(breach_time)

        diff = (deadline - breach_time).total_seconds() / 3600
        assert diff == 72

    def test_check_deadline_status(self):
        """Test checking deadline compliance."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test"
        )

        notification = registry.create_breach_notification(
            incident_id=incident.incident_id,
            is_high_risk=True
        )

        status = registry.check_notification_deadline(notification.notification_id)

        assert status["deadline_met"] is True
        assert status["hours_remaining"] > 0
        assert status["hours_remaining"] < 72.5


class TestHighRiskAssessment:
    """Test high-risk breach assessment."""

    def test_high_risk_sensitive_data(self):
        """Test high-risk determination for sensitive data."""
        assert assess_high_risk(["health_data"]) is True
        assert assess_high_risk(["biometric_data"]) is True
        assert assess_high_risk(["ssn"]) is True
        assert assess_high_risk(["payment_card_data"]) is True

    def test_high_risk_volume(self):
        """Test high-risk determination for large volume."""
        assert assess_high_risk(["email"], volume=2000) is True
        assert assess_high_risk(["name"], volume=10000) is True

    def test_low_risk(self):
        """Test low-risk determination."""
        assert assess_high_risk(["public_profile"]) is False
        assert assess_high_risk(["name"], volume=100) is False
        assert assess_high_risk(["email"], volume=500) is False


class TestBreachNotificationBuilder:
    """Test breach notification generation."""

    def test_generate_dpa_notification(self):
        """Test generating GDPR DPA notification."""
        from HoloLoom.security.incident import IncidentMetadata

        builder = BreachNotificationBuilder()

        incident = IncidentMetadata(
            incident_id="TEST-001",
            severity=IncidentSeverity.SEV1_CRITICAL,
            type=IncidentType.DATA_BREACH,
            status=IncidentStatus.CONTAINED,
            title="Test",
            description="Test",
            detected_time=datetime.utcnow(),
            data_categories_affected=["names", "emails"]
        )

        notification = type('obj', (object,), {
            'affected_individual_categories': ['customers'],
            'affected_count': 1000,
            'likely_consequences': ['phishing'],
            'measures_taken': ['patched'],
            'awareness_datetime': datetime.utcnow()
        })()

        dpa_notif = builder.generate_dpa_notification(incident, notification)

        assert dpa_notif["type"] == "GDPR_ARTICLE_33"
        assert "dpa" in dpa_notif
        assert "content" in dpa_notif
        assert dpa_notif["deadline_met"] is True

    def test_generate_individual_notification_email(self):
        """Test generating individual notification email."""
        from HoloLoom.security.incident import IncidentMetadata

        builder = BreachNotificationBuilder()

        incident = IncidentMetadata(
            incident_id="TEST-001",
            severity=IncidentSeverity.SEV1_CRITICAL,
            type=IncidentType.DATA_BREACH,
            status=IncidentStatus.CONTAINED,
            title="Test",
            description="Test",
            detected_time=datetime.utcnow(),
            data_categories_affected=["names", "emails"]
        )

        notification = type('obj', (object,), {})()

        email_body = builder.generate_individual_notification_email(
            incident,
            notification,
            customer_name="John Doe",
            customer_email="john@example.com"
        )

        assert "John Doe" in email_body
        assert "Important Security Notice" in email_body
        assert incident.incident_id in email_body

    def test_validate_gdpr_compliance(self):
        """Test GDPR compliance validation."""
        from HoloLoom.security.incident import BreachNotification

        builder = BreachNotificationBuilder()

        notification = BreachNotification(
            notification_id="BRN-001",
            incident_id="INC-001",
            breach_nature="SQL injection",
            affected_individual_categories=["customers"],
            affected_count=1000,
            likely_consequences=["phishing"],
            measures_taken=["patched"],
            dpa_name="Test DPA"
        )

        compliance = builder.validate_gdpr_compliance(notification)

        assert "article_33_deadline" in compliance
        assert "deadline_met" in compliance
        assert "compliance_checks" in compliance
        assert "status" in compliance


class TestExportReporting:
    """Test incident export and reporting."""

    def test_export_incident_report(self):
        """Test exporting incident for post-mortem report."""
        registry = IncidentRegistry(log_file="/tmp/test_incidents.jsonl")

        incident = registry.create_incident(
            severity=IncidentSeverity.SEV1_CRITICAL,
            incident_type=IncidentType.DATA_BREACH,
            title="Test",
            description="Test",
            affected_users=1000
        )

        # Record actions
        registry.record_action(
            incident_id=incident.incident_id,
            action_type="BLOCK_IP",
            description="Test",
            taken_by="User1"
        )

        # Update status
        registry.update_status(incident.incident_id, IncidentStatus.INVESTIGATING)
        registry.update_status(incident.incident_id, IncidentStatus.CONTAINED)
        registry.update_status(incident.incident_id, IncidentStatus.RESOLVED)

        # Export report
        report = registry.export_incident_report(incident.incident_id)

        assert "incident" in report
        assert "actions" in report
        assert "metrics" in report
        assert "timeline_summary" in report
        assert len(report["actions"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
