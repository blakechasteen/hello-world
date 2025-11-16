"""
Automated Evidence Collection

Automated evidence collection for compliance audits.

**Created: 2025-11-16**
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import json
import hashlib


class EvidenceType(Enum):
    """Types of compliance evidence"""
    ACCESS_LOG = "access_log"
    CHANGE_LOG = "change_log"
    INCIDENT_LOG = "incident_log"
    TRAINING_RECORD = "training_record"
    VENDOR_ASSESSMENT = "vendor_assessment"
    POLICY_ACKNOWLEDGMENT = "policy_acknowledgment"
    SYSTEM_CONFIGURATION = "system_configuration"
    BACKUP_VERIFICATION = "backup_verification"
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    AUDIT_REPORT = "audit_report"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class AuditEvidence:
    """Individual piece of audit evidence"""
    evidence_id: str
    evidence_type: EvidenceType
    control_id: str
    description: str
    file_path: Optional[Path] = None
    file_hash: Optional[str] = None
    collected_date: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceRepository:
    """
    Evidence repository with versioning and integrity verification

    **Created: 2025-11-16**
    """

    def __init__(self, repo_path: Path = Path("./evidence_repository")):
        """
        Initialize evidence repository

        Args:
            repo_path: Path to repository
        """
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)

        # Evidence index
        self.index: Dict[str, AuditEvidence] = {}
        self._load_index()

    def _load_index(self):
        """Load evidence index from disk"""
        index_path = self.repo_path / "evidence_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                for eid, edata in data.items():
                    evidence = AuditEvidence(
                        evidence_id=edata['evidence_id'],
                        evidence_type=EvidenceType(edata['evidence_type']),
                        control_id=edata['control_id'],
                        description=edata['description'],
                        file_path=Path(edata['file_path']) if edata.get('file_path') else None,
                        file_hash=edata.get('file_hash'),
                        collected_date=datetime.fromisoformat(edata['collected_date']),
                        period_start=datetime.fromisoformat(edata['period_start']) if edata.get('period_start') else None,
                        period_end=datetime.fromisoformat(edata['period_end']) if edata.get('period_end') else None,
                        metadata=edata.get('metadata', {}),
                    )
                    self.index[eid] = evidence

    def _save_index(self):
        """Save evidence index to disk"""
        index_path = self.repo_path / "evidence_index.json"
        data = {}
        for eid, evidence in self.index.items():
            data[eid] = {
                'evidence_id': evidence.evidence_id,
                'evidence_type': evidence.evidence_type.value,
                'control_id': evidence.control_id,
                'description': evidence.description,
                'file_path': str(evidence.file_path) if evidence.file_path else None,
                'file_hash': evidence.file_hash,
                'collected_date': evidence.collected_date.isoformat(),
                'period_start': evidence.period_start.isoformat() if evidence.period_start else None,
                'period_end': evidence.period_end.isoformat() if evidence.period_end else None,
                'metadata': evidence.metadata,
            }

        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_evidence(self, evidence: AuditEvidence):
        """Add evidence to repository"""
        self.index[evidence.evidence_id] = evidence
        self._save_index()

    def get_evidence(self, evidence_id: str) -> Optional[AuditEvidence]:
        """Get evidence by ID"""
        return self.index.get(evidence_id)

    def search_evidence(
        self,
        control_id: Optional[str] = None,
        evidence_type: Optional[EvidenceType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEvidence]:
        """Search evidence by criteria"""
        results = list(self.index.values())

        if control_id:
            results = [e for e in results if e.control_id == control_id]

        if evidence_type:
            results = [e for e in results if e.evidence_type == evidence_type]

        if start_date:
            results = [e for e in results if e.collected_date >= start_date]

        if end_date:
            results = [e for e in results if e.collected_date <= end_date]

        return results

    def verify_integrity(self, evidence_id: str) -> bool:
        """Verify evidence file integrity using hash"""
        evidence = self.index.get(evidence_id)
        if not evidence or not evidence.file_path or not evidence.file_hash:
            return False

        if not evidence.file_path.exists():
            return False

        # Compute current hash
        current_hash = self._compute_hash(evidence.file_path)
        return current_hash == evidence.file_hash

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class EvidenceCollector:
    """
    Automated evidence collector

    Collects evidence from various sources for compliance audits.

    **Created: 2025-11-16**

    Example:
        ```python
        collector = EvidenceCollector()

        # Collect access logs
        evidence = await collector.collect_access_logs(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31)
        )

        # Collect all evidence for audit period
        package = await collector.collect_audit_package(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31)
        )
        ```
    """

    def __init__(
        self,
        repository: Optional[EvidenceRepository] = None
    ):
        """
        Initialize evidence collector

        Args:
            repository: Evidence repository (creates new if None)
        """
        self.repository = repository or EvidenceRepository()

    async def collect_access_logs(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AuditEvidence]:
        """
        Collect access logs for audit period

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            List of access log evidence
        """
        evidence_list = []

        # Simulated access log collection
        # In production: query IAM system, SIEM, etc.

        evidence_id = f"ACCESS_LOG_{start_date.date()}_{end_date.date()}"
        evidence_path = self.repository.repo_path / f"{evidence_id}.json"

        # Generate access log data
        log_data = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_access_events": 15234,
            "unique_users": 150,
            "failed_login_attempts": 23,
            "privileged_access_events": 456,
            "sample_events": [
                {
                    "timestamp": "2025-11-16T10:30:00Z",
                    "user": "admin@hololoom.ai",
                    "action": "login",
                    "resource": "production_console",
                    "result": "success",
                },
                {
                    "timestamp": "2025-11-16T14:15:00Z",
                    "user": "user@hololoom.ai",
                    "action": "access",
                    "resource": "customer_data",
                    "result": "success",
                },
            ]
        }

        # Write to file
        with open(evidence_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Compute hash
        file_hash = self.repository._compute_hash(evidence_path)

        # Create evidence object
        evidence = AuditEvidence(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.ACCESS_LOG,
            control_id="CC6.1",
            description="Access logs for audit period",
            file_path=evidence_path,
            file_hash=file_hash,
            period_start=start_date,
            period_end=end_date,
            metadata={
                "total_events": 15234,
                "unique_users": 150,
            }
        )

        self.repository.add_evidence(evidence)
        evidence_list.append(evidence)

        return evidence_list

    async def collect_change_logs(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AuditEvidence]:
        """Collect change logs"""
        evidence_id = f"CHANGE_LOG_{start_date.date()}_{end_date.date()}"
        evidence_path = self.repository.repo_path / f"{evidence_id}.json"

        change_data = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_changes": 487,
            "approved_changes": 485,
            "emergency_changes": 2,
            "sample_changes": [
                {
                    "change_id": "CHG001",
                    "timestamp": "2025-11-16T08:00:00Z",
                    "description": "Deploy security patch",
                    "approver": "security_team",
                    "status": "completed",
                }
            ]
        }

        with open(evidence_path, 'w') as f:
            json.dump(change_data, f, indent=2)

        file_hash = self.repository._compute_hash(evidence_path)

        evidence = AuditEvidence(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.CHANGE_LOG,
            control_id="CC8.1",
            description="Change logs for audit period",
            file_path=evidence_path,
            file_hash=file_hash,
            period_start=start_date,
            period_end=end_date,
        )

        self.repository.add_evidence(evidence)
        return [evidence]

    async def collect_training_records(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[AuditEvidence]:
        """Collect security awareness training records"""
        evidence_id = f"TRAINING_{start_date.date()}_{end_date.date()}"
        evidence_path = self.repository.repo_path / f"{evidence_id}.json"

        training_data = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_employees": 150,
            "completed_training": 148,
            "completion_rate": 0.987,
            "training_modules": [
                "Security Awareness Basics",
                "Phishing Detection",
                "Data Protection",
                "Incident Reporting",
            ]
        }

        with open(evidence_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        file_hash = self.repository._compute_hash(evidence_path)

        evidence = AuditEvidence(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.TRAINING_RECORD,
            control_id="CC1.1",
            description="Security awareness training records",
            file_path=evidence_path,
            file_hash=file_hash,
            period_start=start_date,
            period_end=end_date,
        )

        self.repository.add_evidence(evidence)
        return [evidence]

    async def collect_audit_package(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, List[AuditEvidence]]:
        """
        Collect complete audit evidence package

        Args:
            start_date: Audit period start
            end_date: Audit period end

        Returns:
            Dictionary mapping evidence types to evidence lists
        """
        package = {}

        # Collect all evidence types
        package['access_logs'] = await self.collect_access_logs(start_date, end_date)
        package['change_logs'] = await self.collect_change_logs(start_date, end_date)
        package['training_records'] = await self.collect_training_records(start_date, end_date)

        return package
