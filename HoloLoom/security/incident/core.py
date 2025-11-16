"""
Incident Response Tracking System

Core incident tracking module for HoloLoom security pipeline Phase 4.
Implements NIST SP 800-61 incident response framework with GDPR/CCPA compliance.

Created: 2025-11-16
Status: Production Ready
"""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field
import hashlib
import uuid

# ==================== Constants ====================

class IncidentSeverity(Enum):
    """Incident severity levels per NIST classification."""
    SEV1_CRITICAL = "SEV-1"
    SEV2_HIGH = "SEV-2"
    SEV3_MEDIUM = "SEV-3"
    SEV4_LOW = "SEV-4"


class IncidentType(Enum):
    """Common incident types."""
    DATA_BREACH = "DATA_BREACH"
    MALWARE = "MALWARE"
    DDOS = "DDOS"
    RANSOMWARE = "RANSOMWARE"
    INJECTION_ATTACK = "INJECTION_ATTACK"
    ACCESS_CONTROL = "ACCESS_CONTROL"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    INSIDER_THREAT = "INSIDER_THREAT"
    MISCONFIGURATION = "MISCONFIGURATION"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    DETECTED = "DETECTED"           # T+0: Initial alert
    INVESTIGATING = "INVESTIGATING" # T+5-30 min: Team assembled
    CONTAINED = "CONTAINED"         # T+1-4 hours: Threat stopped
    ERADICATING = "ERADICATING"    # T+4 hours: Removing threat
    RECOVERING = "RECOVERING"       # T+12+ hours: Restoring systems
    RESOLVED = "RESOLVED"           # Complete
    CLOSED = "CLOSED"               # Post-mortem done


class NotificationStatus(Enum):
    """Breach notification status (GDPR Article 33-34)."""
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    NOTIFIED_DPA = "NOTIFIED_DPA"      # GDPR Article 33
    NOTIFIED_INDIVIDUALS = "NOTIFIED"   # GDPR Article 34
    NOTIFIED_AG = "NOTIFIED_AG"         # State Attorney General
    COMPLETE = "COMPLETE"


# ==================== Data Classes ====================

@dataclass
class IncidentMetadata:
    """Core incident information."""
    incident_id: str
    severity: IncidentSeverity
    type: IncidentType
    status: IncidentStatus
    title: str
    description: str

    # Timeline
    detected_time: datetime
    containment_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None

    # Impact
    affected_users: int = 0
    affected_systems: List[str] = field(default_factory=list)
    data_categories_affected: List[str] = field(default_factory=list)

    # GDPR/CCPA
    breach_notification_required: bool = False
    notification_status: NotificationStatus = NotificationStatus.NOT_REQUIRED
    notification_deadline: Optional[datetime] = None

    # Team
    incident_commander: Optional[str] = None
    security_lead: Optional[str] = None
    other_responders: List[str] = field(default_factory=list)

    # Audit trail
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    creator: str = "SYSTEM"


@dataclass
class BreachNotification:
    """GDPR/CCPA breach notification details."""
    notification_id: str
    incident_id: str

    # GDPR Article 33 (DPA notification)
    dpa_name: str = ""
    dpa_contact: str = ""
    dpa_notification_date: Optional[datetime] = None
    dpa_notification_method: str = "PORTAL"  # PORTAL, EMAIL, PHONE

    # GDPR Article 34 (Individual notification)
    individual_notification_date: Optional[datetime] = None
    individual_notification_method: str = "EMAIL"  # EMAIL, IN_APP, MAIL, MEDIA

    # CCPA & State notification
    state_ag_notifications: Dict[str, datetime] = field(default_factory=dict)

    # Content
    breach_nature: str = ""
    affected_individual_categories: List[str] = field(default_factory=list)
    affected_count: int = 0
    likely_consequences: List[str] = field(default_factory=list)
    measures_taken: List[str] = field(default_factory=list)

    # Timeline (GDPR)
    awareness_datetime: datetime = field(default_factory=datetime.utcnow)
    deadline_datetime: datetime = field(default_factory=lambda:
        datetime.utcnow() + timedelta(hours=72))

    # Status tracking
    is_high_risk: bool = False
    requires_individual_notification: bool = False
    all_notifications_complete: bool = False


@dataclass
class IncidentAction:
    """Action taken during incident response."""
    action_id: str
    incident_id: str
    timestamp: datetime
    action_type: str  # BLOCK_IP, REVOKE_CREDENTIALS, PATCH, etc
    description: str
    taken_by: str
    evidence_collected: bool = False
    evidence_hash: Optional[str] = None  # SHA-256 of evidence
    reversible: bool = False
    reverse_action: Optional[str] = None

    # Status
    status: str = "COMPLETED"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    error_message: Optional[str] = None


# ==================== Incident Registry ====================

class IncidentRegistry:
    """Central registry for all incidents and notifications."""

    def __init__(self, log_file: str = "incidents.jsonl"):
        """
        Initialize incident registry.

        Args:
            log_file: Path to JSONL file for audit trail
        """
        self.incidents: Dict[str, IncidentMetadata] = {}
        self.notifications: Dict[str, BreachNotification] = {}
        self.actions: Dict[str, List[IncidentAction]] = {}
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)

    def create_incident(
        self,
        severity: IncidentSeverity,
        incident_type: IncidentType,
        title: str,
        description: str,
        affected_users: int = 0,
        affected_systems: List[str] = None,
        data_categories: List[str] = None,
        incident_commander: str = "",
        security_lead: str = ""
    ) -> IncidentMetadata:
        """
        Create new incident (T+0: Detection phase).

        Args:
            severity: Incident severity (SEV-1/2/3/4)
            incident_type: Type of incident
            title: Short title
            description: Detailed description
            affected_users: Number of affected users (0 if unknown)
            affected_systems: List of affected systems
            data_categories: List of data categories affected (GDPR classification)
            incident_commander: Name of incident commander
            security_lead: Name of security lead

        Returns:
            Created IncidentMetadata object
        """
        incident_id = f"INC-2025-{incident_type.name[:3]}-{uuid.uuid4().hex[:8].upper()}"

        # Calculate GDPR notification deadline (72 hours from now)
        now = datetime.utcnow()
        notification_deadline = now + timedelta(hours=72)

        incident = IncidentMetadata(
            incident_id=incident_id,
            severity=severity,
            type=incident_type,
            status=IncidentStatus.DETECTED,
            title=title,
            description=description,
            detected_time=now,
            affected_users=affected_users or 0,
            affected_systems=affected_systems or [],
            data_categories_affected=data_categories or [],
            incident_commander=incident_commander,
            security_lead=security_lead,
            notification_deadline=notification_deadline
        )

        self.incidents[incident_id] = incident
        self.actions[incident_id] = []

        self._log_audit_entry({
            "action": "INCIDENT_CREATED",
            "incident_id": incident_id,
            "severity": severity.value,
            "type": incident_type.value,
            "timestamp": now.isoformat()
        })

        self.logger.info(f"Incident created: {incident_id} ({severity.value})")
        return incident

    def get_incident(self, incident_id: str) -> Optional[IncidentMetadata]:
        """Retrieve incident by ID."""
        return self.incidents.get(incident_id)

    def update_status(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        notes: str = ""
    ) -> bool:
        """
        Update incident status through response lifecycle.

        Timeline expectations:
        - DETECTED → INVESTIGATING: 5-15 minutes
        - INVESTIGATING → CONTAINED: 30 min - 4 hours
        - CONTAINED → ERADICATING: 1-4 hours
        - ERADICATING → RECOVERING: 2-12 hours
        - RECOVERING → RESOLVED: 4-24 hours
        - RESOLVED → CLOSED: After post-mortem (3 days+)
        """
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        incident.last_updated = datetime.utcnow()

        # Track containment time
        if new_status == IncidentStatus.CONTAINED and not incident.containment_time:
            incident.containment_time = datetime.utcnow()

        # Track resolution time
        if new_status == IncidentStatus.RESOLVED and not incident.resolution_time:
            incident.resolution_time = datetime.utcnow()

        self._log_audit_entry({
            "action": "STATUS_UPDATED",
            "incident_id": incident_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat()
        })

        self.logger.info(f"Incident {incident_id}: {old_status.value} → {new_status.value}")
        return True

    def record_action(
        self,
        incident_id: str,
        action_type: str,
        description: str,
        taken_by: str,
        evidence_collected: bool = False,
        reversible: bool = False
    ) -> IncidentAction:
        """
        Record action taken during incident response.

        Args:
            incident_id: Incident ID
            action_type: Type of action (BLOCK_IP, PATCH, etc)
            description: Detailed description
            taken_by: Person/system that took action
            evidence_collected: Whether evidence was preserved
            reversible: Whether action can be reversed

        Returns:
            Recorded IncidentAction
        """
        if incident_id not in self.actions:
            return None

        action = IncidentAction(
            action_id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
            incident_id=incident_id,
            timestamp=datetime.utcnow(),
            action_type=action_type,
            description=description,
            taken_by=taken_by,
            evidence_collected=evidence_collected,
            reversible=reversible
        )

        self.actions[incident_id].append(action)

        self._log_audit_entry({
            "action": "ACTION_RECORDED",
            "incident_id": incident_id,
            "action_type": action_type,
            "taken_by": taken_by,
            "evidence_collected": evidence_collected,
            "timestamp": action.timestamp.isoformat()
        })

        self.logger.info(f"Action recorded: {action.action_id} ({action_type})")
        return action

    def create_breach_notification(
        self,
        incident_id: str,
        is_high_risk: bool = False,
        affected_categories: List[str] = None
    ) -> BreachNotification:
        """
        Create breach notification record (GDPR Article 33-34 compliance).

        Args:
            incident_id: Incident ID
            is_high_risk: Is this a high-risk breach requiring individual notification
            affected_categories: Data categories affected (for notification content)

        Returns:
            BreachNotification object
        """
        if incident_id not in self.incidents:
            return None

        incident = self.incidents[incident_id]
        now = datetime.utcnow()
        deadline = now + timedelta(hours=72)

        notification = BreachNotification(
            notification_id=f"BRN-{uuid.uuid4().hex[:8].upper()}",
            incident_id=incident_id,
            is_high_risk=is_high_risk,
            requires_individual_notification=is_high_risk,
            awareness_datetime=incident.detected_time,
            deadline_datetime=deadline,
            affected_individual_categories=affected_categories or [],
            affected_count=incident.affected_users
        )

        self.notifications[notification.notification_id] = notification
        incident.breach_notification_required = True
        incident.notification_status = NotificationStatus.PENDING

        self._log_audit_entry({
            "action": "BREACH_NOTIFICATION_CREATED",
            "incident_id": incident_id,
            "notification_id": notification.notification_id,
            "is_high_risk": is_high_risk,
            "deadline": deadline.isoformat(),
            "timestamp": now.isoformat()
        })

        self.logger.info(f"Breach notification created: {notification.notification_id}")
        return notification

    def mark_dpa_notified(
        self,
        notification_id: str,
        dpa_name: str,
        dpa_contact: str,
        notification_date: datetime = None
    ) -> bool:
        """
        Mark DPA notification complete (GDPR Article 33).

        Args:
            notification_id: Breach notification ID
            dpa_name: Data Protection Authority name
            dpa_contact: DPA contact email/phone
            notification_date: When notification was sent

        Returns:
            True if successful
        """
        if notification_id not in self.notifications:
            return False

        notification = self.notifications[notification_id]
        notification.dpa_name = dpa_name
        notification.dpa_contact = dpa_contact
        notification.dpa_notification_date = notification_date or datetime.utcnow()

        # Update incident status
        incident = self.incidents.get(notification.incident_id)
        if incident:
            incident.notification_status = NotificationStatus.NOTIFIED_DPA

        self._log_audit_entry({
            "action": "DPA_NOTIFIED",
            "notification_id": notification_id,
            "incident_id": notification.incident_id,
            "dpa_name": dpa_name,
            "timestamp": notification.dpa_notification_date.isoformat()
        })

        self.logger.info(f"DPA notified: {dpa_name}")
        return True

    def mark_individuals_notified(
        self,
        notification_id: str,
        notification_date: datetime = None,
        method: str = "EMAIL"
    ) -> bool:
        """
        Mark individuals notified (GDPR Article 34).

        Args:
            notification_id: Breach notification ID
            notification_date: When individuals were notified
            method: Notification method (EMAIL, IN_APP, MAIL, MEDIA)

        Returns:
            True if successful
        """
        if notification_id not in self.notifications:
            return False

        notification = self.notifications[notification_id]
        notification.individual_notification_date = notification_date or datetime.utcnow()
        notification.individual_notification_method = method

        # Update incident status
        incident = self.incidents.get(notification.incident_id)
        if incident:
            incident.notification_status = NotificationStatus.NOTIFIED_INDIVIDUALS

        self._log_audit_entry({
            "action": "INDIVIDUALS_NOTIFIED",
            "notification_id": notification_id,
            "incident_id": notification.incident_id,
            "method": method,
            "count": notification.affected_count,
            "timestamp": notification.individual_notification_date.isoformat()
        })

        self.logger.info(f"Individuals notified: {notification.affected_count} via {method}")
        return True

    def get_metrics(self, incident_id: str) -> Dict:
        """Calculate incident response metrics (MTTD, MTTR, MTRC)."""
        if incident_id not in self.incidents:
            return {}

        incident = self.incidents[incident_id]
        metrics = {
            "incident_id": incident_id,
            "severity": incident.severity.value,
            "type": incident.type.value,
            "status": incident.status.value,
            "affected_users": incident.affected_users
        }

        # MTTD: Mean Time to Detect (from incident start to detection)
        # Note: In real scenarios, this would require knowing when attack actually started
        # For now, using detection_time as reference
        metrics["detected_at"] = incident.detected_time.isoformat()

        # MTTR: Mean Time to Respond (from detection to first response action)
        if self.actions.get(incident_id):
            first_action = self.actions[incident_id][0]
            response_time = (first_action.timestamp - incident.detected_time).total_seconds() / 60
            metrics["mttr_minutes"] = response_time

        # MTRC: Mean Time to Contain (from detection to containment)
        if incident.containment_time:
            containment_time = (incident.containment_time - incident.detected_time).total_seconds() / 60
            metrics["mtrc_minutes"] = containment_time

        # MTCO: Mean Time to Resolve (from detection to resolution)
        if incident.resolution_time:
            resolution_time = (incident.resolution_time - incident.detected_time).total_seconds() / 3600
            metrics["mtco_hours"] = resolution_time

        return metrics

    def check_notification_deadline(self, notification_id: str) -> Dict:
        """Check GDPR 72-hour notification deadline status."""
        if notification_id not in self.notifications:
            return {}

        notification = self.notifications[notification_id]
        now = datetime.utcnow()
        time_remaining = notification.deadline_datetime - now

        return {
            "notification_id": notification_id,
            "deadline": notification.deadline_datetime.isoformat(),
            "hours_remaining": time_remaining.total_seconds() / 3600,
            "deadline_met": now < notification.deadline_datetime,
            "dpa_notified": notification.dpa_notification_date is not None,
            "individuals_notified": notification.individual_notification_date is not None
        }

    def export_incident_report(self, incident_id: str) -> Dict:
        """Export incident data for post-mortem report."""
        if incident_id not in self.incidents:
            return {}

        incident = self.incidents[incident_id]
        actions = self.actions.get(incident_id, [])

        return {
            "incident": asdict(incident),
            "actions": [asdict(a) for a in actions],
            "metrics": self.get_metrics(incident_id),
            "timeline_summary": {
                "detected": incident.detected_time.isoformat(),
                "contained": incident.containment_time.isoformat() if incident.containment_time else None,
                "resolved": incident.resolution_time.isoformat() if incident.resolution_time else None
            }
        }

    def _log_audit_entry(self, entry: Dict):
        """Log entry to audit trail (immutable log)."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ==================== Utilities ====================

def get_notification_deadline(breach_datetime: datetime) -> datetime:
    """
    Calculate GDPR Article 33 notification deadline.

    Per GDPR: "without undue delay and, where feasible, not later than 72 hours
    after having become aware of a personal data breach"

    Args:
        breach_datetime: When organization became aware of breach

    Returns:
        Deadline datetime (72 hours later)
    """
    return breach_datetime + timedelta(hours=72)


def assess_high_risk(data_categories: List[str], volume: int = 0) -> bool:
    """
    Assess if breach is "high risk" requiring individual notification.

    Per GDPR: Notify individuals if there is "high risk to rights and freedoms".

    High-risk factors:
    - Sensitive personal data (health, biometric, racial origin, etc)
    - Large volume of individuals
    - Financial data
    - Government ID numbers
    """
    high_risk_categories = {
        "health_data",
        "biometric_data",
        "racial_origin",
        "political_opinions",
        "religious_beliefs",
        "genetic_data",
        "ssn",
        "payment_card_data",
        "bank_accounts",
        "government_id",
        "password"
    }

    # Check for sensitive data categories
    has_sensitive = any(cat.lower() in high_risk_categories for cat in data_categories)

    # Large volume (>1000) increases risk
    large_volume = volume > 1000

    return has_sensitive or large_volume


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    registry = IncidentRegistry()

    # Create incident
    incident = registry.create_incident(
        severity=IncidentSeverity.SEV1_CRITICAL,
        incident_type=IncidentType.DATA_BREACH,
        title="Customer Database Breach",
        description="SQL injection vulnerability in login form",
        affected_users=15000,
        affected_systems=["prod-db-001", "prod-api-02"],
        data_categories=["names", "emails", "phone_numbers"],
        incident_commander="John Smith",
        security_lead="Jane Doe"
    )

    print(f"Created incident: {incident.incident_id}")
    print(f"Severity: {incident.severity.value}")
    print(f"Notification deadline: {incident.notification_deadline}")
