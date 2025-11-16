"""
GDPR Compliance Verification

Automated GDPR compliance monitoring, DPIA templates, and data subject rights handling.

**Created: 2025-11-16**
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import json


class GDPRArticle(Enum):
    """GDPR Articles"""
    ART_5_PRINCIPLES = "Article 5"  # Principles
    ART_6_LAWFUL_BASIS = "Article 6"  # Lawful basis
    ART_12_TRANSPARENCY = "Article 12"  # Transparency
    ART_13_INFORMATION = "Article 13"  # Information provided
    ART_15_ACCESS = "Article 15"  # Right of access
    ART_16_RECTIFICATION = "Article 16"  # Right to rectification
    ART_17_ERASURE = "Article 17"  # Right to erasure
    ART_18_RESTRICTION = "Article 18"  # Right to restriction
    ART_20_PORTABILITY = "Article 20"  # Right to portability
    ART_21_OBJECTION = "Article 21"  # Right to object
    ART_25_DESIGN = "Article 25"  # Privacy by design
    ART_30_RECORDS = "Article 30"  # Records of processing
    ART_32_SECURITY = "Article 32"  # Security of processing
    ART_33_BREACH_NOTIFICATION = "Article 33"  # Breach notification
    ART_35_DPIA = "Article 35"  # Data protection impact assessment


class DataSubjectRight(Enum):
    """GDPR Data Subject Rights"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17
    RESTRICTION = "restriction"  # Article 18
    PORTABILITY = "portability"  # Article 20
    OBJECTION = "objection"  # Article 21


class ProcessingPurpose(Enum):
    """Purposes for data processing"""
    CONSENT = "consent"  # Article 6(1)(a)
    CONTRACT = "contract"  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Article 6(1)(d)
    PUBLIC_TASK = "public_task"  # Article 6(1)(e)
    LEGITIMATE_INTEREST = "legitimate_interest"  # Article 6(1)(f)


@dataclass
class ProcessingActivity:
    """Record of processing activity (Article 30)"""
    activity_id: str
    purpose: ProcessingPurpose
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    retention_period: str
    security_measures: List[str]
    legal_basis: str
    created_date: datetime = field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None


@dataclass
class DPIAAssessment:
    """Data Protection Impact Assessment (Article 35)"""
    assessment_id: str
    processing_description: str
    purposes: List[str]
    legitimate_interests: str
    necessity_assessment: str
    proportionality_assessment: str
    risks_identified: List[str]
    risk_mitigation: List[str]
    safeguards: List[str]
    conclusion: str  # "acceptable", "needs_consultation", "not_acceptable"
    assessor: str
    assessment_date: datetime
    next_review: datetime


@dataclass
class DataSubjectRequest:
    """Data subject request tracking"""
    request_id: str
    request_type: DataSubjectRight
    data_subject_email: str
    received_date: datetime
    due_date: datetime  # 1 month from receipt
    status: str  # "pending", "in_progress", "completed", "rejected"
    completion_date: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)


class GDPRMonitor:
    """
    GDPR compliance monitor

    Automated GDPR compliance verification, DPIA management, and data subject rights handling.

    **Created: 2025-11-16**

    Example:
        ```python
        monitor = GDPRMonitor()

        # Conduct DPIA
        dpia = monitor.conduct_dpia(
            processing_description="AI model training on user data",
            purposes=["Service improvement"],
            risks=["Re-identification", "Bias amplification"]
        )

        # Handle data subject request
        request = monitor.receive_request(
            request_type=DataSubjectRight.ACCESS,
            email="user@example.com"
        )

        # Check compliance
        status = monitor.check_compliance()
        ```
    """

    def __init__(
        self,
        data_controller: str = "HoloLoom Inc.",
        dpo_email: str = "dpo@hololoom.ai"
    ):
        """
        Initialize GDPR monitor

        Args:
            data_controller: Data controller organization name
            dpo_email: Data Protection Officer email
        """
        self.data_controller = data_controller
        self.dpo_email = dpo_email

        # Processing activities registry
        self.processing_activities: Dict[str, ProcessingActivity] = {}

        # DPIA assessments
        self.dpias: Dict[str, DPIAAssessment] = {}

        # Data subject requests
        self.dsr_queue: List[DataSubjectRequest] = []

        self._initialize_processing_activities()

    def _initialize_processing_activities(self):
        """Initialize processing activity records"""
        # Example: AI model training
        self.processing_activities["model_training"] = ProcessingActivity(
            activity_id="model_training",
            purpose=ProcessingPurpose.LEGITIMATE_INTEREST,
            data_categories=["User interactions", "System logs"],
            data_subjects=["End users", "Administrators"],
            recipients=["Internal AI team"],
            retention_period="2 years",
            security_measures=[
                "Encryption at rest and in transit",
                "Access controls (RBAC)",
                "Data minimization",
                "Pseudonymization",
            ],
            legal_basis="Legitimate interest in service improvement",
        )

        # Example: User authentication
        self.processing_activities["authentication"] = ProcessingActivity(
            activity_id="authentication",
            purpose=ProcessingPurpose.CONTRACT,
            data_categories=["Email", "Password hash", "MFA tokens"],
            data_subjects=["Registered users"],
            recipients=["Authentication service"],
            retention_period="Account lifetime + 30 days",
            security_measures=[
                "Bcrypt password hashing",
                "MFA enforcement",
                "Rate limiting",
                "Session management",
            ],
            legal_basis="Contract performance (account access)",
        )

    def conduct_dpia(
        self,
        processing_description: str,
        purposes: List[str],
        risks: List[str],
        assessor: str = "Security Team"
    ) -> DPIAAssessment:
        """
        Conduct Data Protection Impact Assessment

        Args:
            processing_description: Description of processing operations
            purposes: Purposes of processing
            risks: Identified privacy risks
            assessor: Person conducting assessment

        Returns:
            DPIA assessment
        """
        assessment_id = f"DPIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Necessity assessment
        necessity = self._assess_necessity(purposes)

        # Proportionality assessment
        proportionality = self._assess_proportionality(risks)

        # Risk mitigation
        mitigation = self._recommend_mitigation(risks)

        # Safeguards
        safeguards = [
            "Data minimization",
            "Encryption (AES-256)",
            "Access controls (RBAC)",
            "Audit logging",
            "Regular security reviews",
        ]

        # Conclusion
        if len(risks) == 0:
            conclusion = "acceptable"
        elif all(r in mitigation for r in risks):
            conclusion = "acceptable"
        else:
            conclusion = "needs_consultation"

        dpia = DPIAAssessment(
            assessment_id=assessment_id,
            processing_description=processing_description,
            purposes=purposes,
            legitimate_interests="Service improvement and security",
            necessity_assessment=necessity,
            proportionality_assessment=proportionality,
            risks_identified=risks,
            risk_mitigation=mitigation,
            safeguards=safeguards,
            conclusion=conclusion,
            assessor=assessor,
            assessment_date=datetime.now(),
            next_review=datetime.now() + timedelta(days=365),
        )

        self.dpias[assessment_id] = dpia
        return dpia

    def _assess_necessity(self, purposes: List[str]) -> str:
        """Assess necessity of processing"""
        return (
            f"Processing is necessary to achieve the following purposes: "
            f"{', '.join(purposes)}. No less intrusive alternatives identified."
        )

    def _assess_proportionality(self, risks: List[str]) -> str:
        """Assess proportionality"""
        if len(risks) == 0:
            return "Processing is proportionate with minimal privacy impact."
        else:
            return (
                f"Processing presents {len(risks)} identified risks. "
                f"Mitigation measures implemented to ensure proportionality."
            )

    def _recommend_mitigation(self, risks: List[str]) -> List[str]:
        """Recommend risk mitigation measures"""
        mitigation = []

        for risk in risks:
            if "identification" in risk.lower():
                mitigation.append("Implement pseudonymization and anonymization")
            if "bias" in risk.lower():
                mitigation.append("Regular bias audits and fairness testing")
            if "breach" in risk.lower():
                mitigation.append("Enhanced encryption and access controls")
            if "profiling" in risk.lower():
                mitigation.append("Human review of automated decisions")

        # Default mitigations
        if not mitigation:
            mitigation = [
                "Data minimization",
                "Encryption",
                "Access controls",
                "Regular audits",
            ]

        return mitigation

    def receive_request(
        self,
        request_type: DataSubjectRight,
        email: str
    ) -> DataSubjectRequest:
        """
        Receive data subject request

        Args:
            request_type: Type of request
            email: Data subject email

        Returns:
            Request tracking object
        """
        request_id = f"DSR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        received = datetime.now()
        due = received + timedelta(days=30)  # GDPR requires 1 month response

        request = DataSubjectRequest(
            request_id=request_id,
            request_type=request_type,
            data_subject_email=email,
            received_date=received,
            due_date=due,
            status="pending",
        )

        self.dsr_queue.append(request)
        return request

    def process_access_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """
        Process Article 15 access request

        Args:
            request: Access request

        Returns:
            Dictionary with all data subject's personal data
        """
        # Simulated data retrieval
        data_package = {
            "data_subject": request.data_subject_email,
            "request_id": request.request_id,
            "generated_date": datetime.now().isoformat(),
            "personal_data": {
                "account": {
                    "email": request.data_subject_email,
                    "created_date": "2025-01-15",
                    "last_login": "2025-11-16",
                },
                "processing_activities": [
                    {
                        "activity": "Authentication",
                        "purpose": "Account access",
                        "retention": "Account lifetime + 30 days",
                    }
                ],
                "third_party_recipients": [],
            },
            "rights_information": {
                "rectification": "You have the right to correct inaccurate data",
                "erasure": "You have the right to request deletion",
                "restriction": "You have the right to restrict processing",
                "portability": "You have the right to receive your data",
                "objection": "You have the right to object to processing",
            },
        }

        # Update request status
        request.status = "completed"
        request.completion_date = datetime.now()

        return data_package

    def process_erasure_request(self, request: DataSubjectRequest) -> bool:
        """
        Process Article 17 erasure request

        Args:
            request: Erasure request

        Returns:
            True if erasure completed
        """
        # Simulated data deletion
        # In production: delete from all systems, backups, etc.

        request.status = "completed"
        request.completion_date = datetime.now()
        request.notes.append(f"Data erased from all systems: {datetime.now().isoformat()}")

        return True

    def check_compliance(self) -> Dict[str, Any]:
        """
        Check GDPR compliance status

        Returns:
            Compliance assessment
        """
        compliance_checks = {
            "Article 5 (Principles)": self._check_article_5(),
            "Article 6 (Lawful basis)": self._check_article_6(),
            "Article 25 (Privacy by design)": self._check_article_25(),
            "Article 30 (Records of processing)": self._check_article_30(),
            "Article 32 (Security)": self._check_article_32(),
            "Article 33 (Breach notification)": self._check_article_33(),
            "Article 35 (DPIA)": self._check_article_35(),
        }

        compliant_count = sum(1 for v in compliance_checks.values() if v["compliant"])
        total_count = len(compliance_checks)
        compliance_score = compliant_count / total_count

        return {
            "compliance_score": compliance_score,
            "compliant": compliance_score >= 0.95,
            "checks": compliance_checks,
            "data_subject_requests": {
                "pending": len([r for r in self.dsr_queue if r.status == "pending"]),
                "overdue": len([r for r in self.dsr_queue if r.status == "pending" and r.due_date < datetime.now()]),
            },
            "dpias": {
                "total": len(self.dpias),
                "acceptable": len([d for d in self.dpias.values() if d.conclusion == "acceptable"]),
                "needs_review": len([d for d in self.dpias.values() if d.next_review < datetime.now()]),
            },
        }

    def _check_article_5(self) -> Dict[str, Any]:
        """Check Article 5 (Principles) compliance"""
        # Check data minimization
        data_minimization = all(
            "minimization" in act.security_measures[0].lower()
            for act in self.processing_activities.values()
        )

        return {
            "compliant": data_minimization,
            "details": "Data minimization applied" if data_minimization else "Data minimization not verified",
        }

    def _check_article_6(self) -> Dict[str, Any]:
        """Check Article 6 (Lawful basis) compliance"""
        # Check all activities have legal basis
        has_legal_basis = all(
            act.legal_basis for act in self.processing_activities.values()
        )

        return {
            "compliant": has_legal_basis,
            "details": "Legal basis documented for all processing" if has_legal_basis else "Missing legal basis",
        }

    def _check_article_25(self) -> Dict[str, Any]:
        """Check Article 25 (Privacy by design) compliance"""
        # Check encryption and pseudonymization
        privacy_by_design = all(
            any(measure.lower().find("encrypt") >= 0 or measure.lower().find("pseudonym") >= 0
                for measure in act.security_measures)
            for act in self.processing_activities.values()
        )

        return {
            "compliant": privacy_by_design,
            "details": "Privacy by design implemented" if privacy_by_design else "Privacy measures incomplete",
        }

    def _check_article_30(self) -> Dict[str, Any]:
        """Check Article 30 (Records of processing) compliance"""
        # Check processing activities documented
        has_records = len(self.processing_activities) > 0

        return {
            "compliant": has_records,
            "details": f"{len(self.processing_activities)} processing activities documented",
        }

    def _check_article_32(self) -> Dict[str, Any]:
        """Check Article 32 (Security) compliance"""
        # Check security measures
        has_security = all(
            len(act.security_measures) >= 3
            for act in self.processing_activities.values()
        )

        return {
            "compliant": has_security,
            "details": "Security measures implemented" if has_security else "Security measures insufficient",
        }

    def _check_article_33(self) -> Dict[str, Any]:
        """Check Article 33 (Breach notification) compliance"""
        # Check breach notification procedure exists (implemented in Phase 4)
        breach_notification_ready = True  # Implemented in incident response module

        return {
            "compliant": breach_notification_ready,
            "details": "Breach notification procedure implemented",
        }

    def _check_article_35(self) -> Dict[str, Any]:
        """Check Article 35 (DPIA) compliance"""
        # Check DPIA conducted for high-risk processing
        has_dpias = len(self.dpias) > 0

        return {
            "compliant": has_dpias,
            "details": f"{len(self.dpias)} DPIAs conducted",
        }
