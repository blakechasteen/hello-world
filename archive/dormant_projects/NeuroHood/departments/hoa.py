"""
HOA Department

Handles:
- Noise complaints
- Property violations
- Neighbor disputes
- HOA rule enforcement
- Violation reviews with causal reasoning

Uses causal reasoning to:
- Fairly attribute blame for violations
- Understand root causes of complaints
- Suggest effective resolutions
- Track violation history and patterns
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from HoloLoom.departments.base import BaseDepartment
from HoloLoom.departments.protocol import DepartmentRequest, DepartmentResponse

from ..causal.blame import BlameAttributionSystem, BlameReport, quick_blame_attribution
from ..causal.interventions import NeuroHoodInterventionEngine
from ..causal.relationship_scm import RelationshipSCM, extract_relationship_data

logger = logging.getLogger(__name__)


@dataclass
class Complaint:
    """HOA complaint."""
    complaint_id: str
    complainant: str
    subject: str  # Who the complaint is about
    type: str  # "noise", "property", "parking", "behavior", etc.
    description: str
    status: str = "open"  # "open", "under_review", "resolved", "dismissed"
    timestamp: float = 0.0
    blame_report: Optional[BlameReport] = None
    resolution: Optional[str] = None


@dataclass
class Violation:
    """HOA violation record."""
    violation_id: str
    resident: str
    type: str
    description: str
    severity: str = "minor"  # "minor", "moderate", "severe"
    fine_amount: float = 0.0
    status: str = "pending"  # "pending", "paid", "appealed", "waived"
    timestamp: float = 0.0


class HOADepartment(BaseDepartment):
    """
    HOA Department for complaint management and rule enforcement.

    Uses causal reasoning to:
    - Fairly attribute blame in neighbor disputes
    - Understand root causes of complaints
    - Suggest effective interventions
    - Track violation patterns over time
    """

    def __init__(self, scm: Optional[RelationshipSCM] = None):
        """
        Initialize HOA Department.

        Args:
            scm: Optional trained RelationshipSCM for causal reasoning
        """
        super().__init__(name="HOA", version="1.0.0")
        self.scm = scm
        self.blame_system = BlameAttributionSystem(scm) if scm else None
        self.intervention_engine = NeuroHoodInterventionEngine(scm) if scm else None

        # HOA records
        self.complaints: List[Complaint] = []
        self.violations: List[Violation] = []

    async def process(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Process HOA request.

        Supported request types:
        - "file_complaint": File a new complaint
        - "review_complaint": Review complaint with blame attribution
        - "issue_violation": Issue a violation notice
        - "appeal_violation": Appeal a violation
        - "get_statistics": Get department statistics
        - "get_resident_history": Get a resident's violation history
        """
        request_type = request.request_type

        if request_type == "file_complaint":
            return await self._file_complaint(request)
        elif request_type == "review_complaint":
            return await self._review_complaint(request)
        elif request_type == "issue_violation":
            return await self._issue_violation(request)
        elif request_type == "appeal_violation":
            return await self._appeal_violation(request)
        elif request_type == "get_statistics":
            return await self._get_statistics(request)
        elif request_type == "get_resident_history":
            return await self._get_resident_history(request)
        else:
            return DepartmentResponse(
                success=False,
                error=f"Unknown request type: {request_type}"
            )

    async def _file_complaint(self, request: DepartmentRequest) -> DepartmentResponse:
        """File a new HOA complaint."""
        data = request.data

        complainant = data.get("complainant")
        subject = data.get("subject")
        complaint_type = data.get("type")
        description = data.get("description")

        if not all([complainant, subject, complaint_type, description]):
            return DepartmentResponse(
                success=False,
                error="Missing required fields"
            )

        complaint = Complaint(
            complaint_id=f"HOA-{len(self.complaints)+1:04d}",
            complainant=complainant,
            subject=subject,
            type=complaint_type,
            description=description,
            timestamp=data.get("timestamp", 0.0)
        )

        self.complaints.append(complaint)

        logger.info(f"Filed complaint: {complaint.complaint_id} ({complainant} → {subject})")

        return DepartmentResponse(
            success=True,
            data={
                "complaint_id": complaint.complaint_id,
                "message": f"Complaint filed: {complaint.complaint_id}"
            }
        )

    async def _review_complaint(self, request: DepartmentRequest) -> DepartmentResponse:
        """Review complaint with causal blame attribution."""
        data = request.data
        complaint_id = data.get("complaint_id")
        state = data.get("state")  # NeuroHoodState

        if not complaint_id:
            return DepartmentResponse(
                success=False,
                error="Missing complaint_id"
            )

        complaint = next((c for c in self.complaints if c.complaint_id == complaint_id), None)
        if not complaint:
            return DepartmentResponse(
                success=False,
                error=f"Complaint not found: {complaint_id}"
            )

        if not self.scm or not state:
            return DepartmentResponse(
                success=False,
                error="Causal reasoning requires SCM and state data"
            )

        # Find relationship between complainant and subject
        relationship_key = self._get_relationship_key(
            complaint.complainant,
            complaint.subject,
            state
        )

        if not relationship_key:
            return DepartmentResponse(
                success=False,
                error=f"No relationship found between {complaint.complainant} and {complaint.subject}"
            )

        relationship = state.relationships[relationship_key]
        factual_data = extract_relationship_data(
            relationship,
            state.residents,
            timestamp=complaint.timestamp
        )

        # Compute blame attribution
        blame_report = quick_blame_attribution(
            self.scm,
            factual_data,
            resident_a_name=complaint.complainant,
            resident_b_name=complaint.subject,
            conflict_description=complaint.description
        )

        complaint.blame_report = blame_report
        complaint.status = "under_review"

        logger.info(f"Reviewed complaint {complaint_id}: {blame_report.primary_cause}")

        return DepartmentResponse(
            success=True,
            data={
                "complaint_id": complaint_id,
                "blame_report": blame_report.format(),
                "primary_cause": blame_report.primary_cause,
                "total_blame_explained": blame_report.total_blame_explained
            }
        )

    async def _issue_violation(self, request: DepartmentRequest) -> DepartmentResponse:
        """Issue a violation notice."""
        data = request.data

        resident = data.get("resident")
        violation_type = data.get("type")
        description = data.get("description")
        severity = data.get("severity", "minor")

        if not all([resident, violation_type, description]):
            return DepartmentResponse(
                success=False,
                error="Missing required fields"
            )

        # Calculate fine based on severity
        fine_schedule = {
            "minor": 50.0,
            "moderate": 150.0,
            "severe": 500.0
        }
        fine_amount = fine_schedule.get(severity, 50.0)

        violation = Violation(
            violation_id=f"VIO-{len(self.violations)+1:04d}",
            resident=resident,
            type=violation_type,
            description=description,
            severity=severity,
            fine_amount=fine_amount,
            timestamp=data.get("timestamp", 0.0)
        )

        self.violations.append(violation)

        logger.info(f"Issued violation: {violation.violation_id} to {resident} (${fine_amount})")

        return DepartmentResponse(
            success=True,
            data={
                "violation_id": violation.violation_id,
                "fine_amount": fine_amount,
                "message": f"Violation issued: {violation.violation_id}"
            }
        )

    async def _appeal_violation(self, request: DepartmentRequest) -> DepartmentResponse:
        """Appeal a violation."""
        violation_id = request.data.get("violation_id")
        reason = request.data.get("reason", "")

        violation = next((v for v in self.violations if v.violation_id == violation_id), None)
        if not violation:
            return DepartmentResponse(
                success=False,
                error=f"Violation not found: {violation_id}"
            )

        if violation.status != "pending":
            return DepartmentResponse(
                success=False,
                error=f"Violation {violation_id} cannot be appealed (status: {violation.status})"
            )

        violation.status = "appealed"

        logger.info(f"Violation {violation_id} appealed: {reason}")

        return DepartmentResponse(
            success=True,
            data={
                "violation_id": violation_id,
                "status": "appealed",
                "message": "Appeal received, under review"
            }
        )

    async def _get_statistics(self, request: DepartmentRequest) -> DepartmentResponse:
        """Get HOA department statistics."""
        total_complaints = len(self.complaints)
        total_violations = len(self.violations)

        # Complaint breakdown
        complaints_by_status = {}
        complaints_by_type = {}
        for c in self.complaints:
            complaints_by_status[c.status] = complaints_by_status.get(c.status, 0) + 1
            complaints_by_type[c.type] = complaints_by_type.get(c.type, 0) + 1

        # Violation breakdown
        violations_by_status = {}
        violations_by_severity = {}
        total_fines = 0.0
        for v in self.violations:
            violations_by_status[v.status] = violations_by_status.get(v.status, 0) + 1
            violations_by_severity[v.severity] = violations_by_severity.get(v.severity, 0) + 1
            if v.status == "paid":
                total_fines += v.fine_amount

        resolution_rate = (
            complaints_by_status.get("resolved", 0) / total_complaints
            if total_complaints > 0 else 0.0
        )

        return DepartmentResponse(
            success=True,
            data={
                "total_complaints": total_complaints,
                "total_violations": total_violations,
                "complaints_by_status": complaints_by_status,
                "complaints_by_type": complaints_by_type,
                "violations_by_status": violations_by_status,
                "violations_by_severity": violations_by_severity,
                "resolution_rate": resolution_rate,
                "total_fines_collected": total_fines
            }
        )

    async def _get_resident_history(self, request: DepartmentRequest) -> DepartmentResponse:
        """Get a resident's complaint and violation history."""
        resident = request.data.get("resident")

        if not resident:
            return DepartmentResponse(
                success=False,
                error="Missing resident name"
            )

        # Find complaints involving this resident
        complaints_filed = [c for c in self.complaints if c.complainant == resident]
        complaints_against = [c for c in self.complaints if c.subject == resident]

        # Find violations for this resident
        violations = [v for v in self.violations if v.resident == resident]

        # Calculate total fines
        total_fines = sum(v.fine_amount for v in violations if v.status in ["pending", "paid"])
        fines_paid = sum(v.fine_amount for v in violations if v.status == "paid")

        return DepartmentResponse(
            success=True,
            data={
                "resident": resident,
                "complaints_filed": len(complaints_filed),
                "complaints_against": len(complaints_against),
                "total_violations": len(violations),
                "violations_by_severity": {
                    "minor": len([v for v in violations if v.severity == "minor"]),
                    "moderate": len([v for v in violations if v.severity == "moderate"]),
                    "severe": len([v for v in violations if v.severity == "severe"])
                },
                "total_fines": total_fines,
                "fines_paid": fines_paid,
                "fines_outstanding": total_fines - fines_paid
            }
        )

    def _get_relationship_key(
        self,
        resident_a: str,
        resident_b: str,
        state: Any
    ) -> Optional[str]:
        """Get relationship key between two residents."""
        for key, rel in state.relationships.items():
            if (rel.resident_a == resident_a and rel.resident_b == resident_b) or \
               (rel.resident_a == resident_b and rel.resident_b == resident_a):
                return key
        return None

    def resolve_complaint(
        self,
        complaint_id: str,
        resolution: str
    ) -> bool:
        """
        Resolve a complaint.

        Args:
            complaint_id: Complaint ID
            resolution: Resolution description

        Returns:
            True if resolved, False if not found
        """
        complaint = next((c for c in self.complaints if c.complaint_id == complaint_id), None)
        if not complaint:
            return False

        complaint.status = "resolved"
        complaint.resolution = resolution

        logger.info(f"Resolved complaint {complaint_id}: {resolution}")

        return True

    def dismiss_complaint(
        self,
        complaint_id: str,
        reason: str
    ) -> bool:
        """
        Dismiss a complaint.

        Args:
            complaint_id: Complaint ID
            reason: Dismissal reason

        Returns:
            True if dismissed, False if not found
        """
        complaint = next((c for c in self.complaints if c.complaint_id == complaint_id), None)
        if not complaint:
            return False

        complaint.status = "dismissed"
        complaint.resolution = f"Dismissed: {reason}"

        logger.info(f"Dismissed complaint {complaint_id}: {reason}")

        return True
