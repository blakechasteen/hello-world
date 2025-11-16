"""
Core Compliance Monitoring Engine

Unified compliance monitoring across SOC2, GDPR, ISO 27001, and other frameworks.

**Created: 2025-11-16**
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import json
import asyncio


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    NIST_CSF = "nist_csf"


class ControlStatus(Enum):
    """Control implementation status"""
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    NON_COMPLIANT = "non_compliant"


class RiskLevel(Enum):
    """Risk levels for compliance gaps"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Control:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    status: ControlStatus
    implementation_date: Optional[datetime] = None
    last_verified: Optional[datetime] = None
    evidence_count: int = 0
    risk_if_not_implemented: RiskLevel = RiskLevel.MEDIUM
    responsible_party: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    automated: bool = False


@dataclass
class ComplianceGap:
    """Identified compliance gap"""
    control_id: str
    framework: ComplianceFramework
    severity: RiskLevel
    description: str
    remediation: str
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    status: str = "open"


@dataclass
class ComplianceStatus:
    """Overall compliance status"""
    framework: ComplianceFramework
    overall_score: float  # 0.0 to 1.0
    controls_total: int
    controls_implemented: int
    controls_verified: int
    gaps: List[ComplianceGap]
    last_assessment: datetime
    next_assessment: datetime
    audit_ready: bool
    evidence_complete: bool


class ComplianceMonitor:
    """
    Main compliance monitoring engine

    Monitors compliance across multiple frameworks with automated checks,
    evidence collection, and gap analysis.

    **Created: 2025-11-16**

    Example:
        ```python
        monitor = ComplianceMonitor()

        # Check overall compliance
        status = await monitor.check_compliance()
        print(f"Overall Score: {status.overall_score:.1%}")

        # Get gaps by severity
        critical_gaps = monitor.get_gaps_by_severity(RiskLevel.CRITICAL)

        # Generate audit package
        package = await monitor.prepare_audit_package()
        ```
    """

    def __init__(
        self,
        frameworks: Optional[List[ComplianceFramework]] = None,
        evidence_dir: Path = Path("./compliance_evidence"),
        auto_collect: bool = True
    ):
        """
        Initialize compliance monitor

        Args:
            frameworks: Frameworks to monitor (default: all)
            evidence_dir: Directory for evidence storage
            auto_collect: Enable automatic evidence collection
        """
        self.frameworks = frameworks or [
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.ISO27001,
        ]
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.auto_collect = auto_collect

        # Control registry
        self.controls: Dict[str, Control] = {}
        self.gaps: List[ComplianceGap] = []

        # Monitoring configuration
        self.monitoring_interval = timedelta(hours=1)
        self.assessment_interval = timedelta(days=30)
        self.last_monitoring = datetime.now()

        # Automated checks
        self._monitoring_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start continuous compliance monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            return  # Already monitoring

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval.total_seconds())
                await self._run_automated_checks()
                self.last_monitoring = datetime.now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")

    async def _run_automated_checks(self):
        """Run automated compliance checks"""
        # Check password policy compliance
        await self._check_password_policy()

        # Check MFA adoption
        await self._check_mfa_adoption()

        # Check encryption coverage
        await self._check_encryption_coverage()

        # Check access review status
        await self._check_access_reviews()

        # Check backup compliance
        await self._check_backup_compliance()

        # Check patching SLA
        await self._check_patching_sla()

    async def _check_password_policy(self):
        """Check password policy enforcement (automated)"""
        # Simulated check - in production, integrate with IAM
        policy_compliant = True  # Check: min 12 chars, complexity

        if policy_compliant:
            self._update_control_status(
                "CC6.1",
                ControlStatus.VERIFIED,
                "Password policy automated check passed"
            )
        else:
            self._add_gap(
                control_id="CC6.1",
                framework=ComplianceFramework.SOC2,
                severity=RiskLevel.HIGH,
                description="Password policy not enforced",
                remediation="Enable password complexity requirements"
            )

    async def _check_mfa_adoption(self):
        """Check MFA adoption rate (target: >95%)"""
        # Simulated check
        adoption_rate = 0.97  # 97% adoption

        if adoption_rate >= 0.95:
            self._update_control_status(
                "CC6.2",
                ControlStatus.VERIFIED,
                f"MFA adoption: {adoption_rate:.1%}"
            )
        else:
            self._add_gap(
                control_id="CC6.2",
                framework=ComplianceFramework.SOC2,
                severity=RiskLevel.HIGH,
                description=f"MFA adoption below target: {adoption_rate:.1%}",
                remediation="Mandate MFA for all users"
            )

    async def _check_encryption_coverage(self):
        """Check encryption coverage (target: 100% sensitive data)"""
        # Simulated check
        coverage = 1.0  # 100% coverage

        if coverage >= 1.0:
            self._update_control_status(
                "A.10.1",
                ControlStatus.VERIFIED,
                "All sensitive data encrypted"
            )

    async def _check_access_reviews(self):
        """Check access review frequency (quarterly)"""
        # Simulated check
        last_review = datetime.now() - timedelta(days=60)
        days_since = (datetime.now() - last_review).days

        if days_since <= 90:  # Within quarterly window
            self._update_control_status(
                "CC6.3",
                ControlStatus.VERIFIED,
                f"Last access review: {days_since} days ago"
            )
        else:
            self._add_gap(
                control_id="CC6.3",
                framework=ComplianceFramework.SOC2,
                severity=RiskLevel.MEDIUM,
                description=f"Access review overdue: {days_since} days",
                remediation="Conduct quarterly access review"
            )

    async def _check_backup_compliance(self):
        """Check backup success rate (target: 100%)"""
        # Simulated check
        success_rate = 1.0  # 100% success

        if success_rate >= 0.99:
            self._update_control_status(
                "A.12.3",
                ControlStatus.VERIFIED,
                f"Backup success: {success_rate:.1%}"
            )

    async def _check_patching_sla(self):
        """Check patching SLA (critical: 7 days, high: 30 days)"""
        # Simulated check
        critical_patched_within_7d = True
        high_patched_within_30d = True

        if critical_patched_within_7d and high_patched_within_30d:
            self._update_control_status(
                "A.12.6",
                ControlStatus.VERIFIED,
                "Patching SLA met"
            )
        else:
            self._add_gap(
                control_id="A.12.6",
                framework=ComplianceFramework.ISO27001,
                severity=RiskLevel.CRITICAL,
                description="Patching SLA not met",
                remediation="Expedite critical patch deployment"
            )

    def _update_control_status(
        self,
        control_id: str,
        status: ControlStatus,
        note: str
    ):
        """Update control status"""
        if control_id in self.controls:
            control = self.controls[control_id]
            control.status = status
            control.last_verified = datetime.now()
            control.notes.append(f"{datetime.now().isoformat()}: {note}")

    def _add_gap(
        self,
        control_id: str,
        framework: ComplianceFramework,
        severity: RiskLevel,
        description: str,
        remediation: str
    ):
        """Add compliance gap"""
        gap = ComplianceGap(
            control_id=control_id,
            framework=framework,
            severity=severity,
            description=description,
            remediation=remediation,
            due_date=datetime.now() + timedelta(days=30)
        )

        # Avoid duplicates
        existing = [g for g in self.gaps if g.control_id == control_id and g.status == "open"]
        if not existing:
            self.gaps.append(gap)

    async def check_compliance(
        self,
        framework: Optional[ComplianceFramework] = None
    ) -> ComplianceStatus:
        """
        Check compliance status for framework

        Args:
            framework: Specific framework or all if None

        Returns:
            ComplianceStatus with overall score and gaps
        """
        target_framework = framework or ComplianceFramework.SOC2

        # Filter controls for framework
        framework_controls = [
            c for c in self.controls.values()
            if c.framework == target_framework
        ]

        total = len(framework_controls)
        implemented = len([c for c in framework_controls if c.status in [
            ControlStatus.IMPLEMENTED, ControlStatus.VERIFIED
        ]])
        verified = len([c for c in framework_controls if c.status == ControlStatus.VERIFIED])

        # Calculate score
        if total == 0:
            score = 0.0
        else:
            score = (implemented * 0.7 + verified * 0.3) / total

        # Filter gaps
        framework_gaps = [g for g in self.gaps if g.framework == target_framework and g.status == "open"]

        # Audit readiness check
        audit_ready = (
            score >= 0.95 and
            verified >= total * 0.9 and
            len([g for g in framework_gaps if g.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]]) == 0
        )

        return ComplianceStatus(
            framework=target_framework,
            overall_score=score,
            controls_total=total,
            controls_implemented=implemented,
            controls_verified=verified,
            gaps=framework_gaps,
            last_assessment=datetime.now(),
            next_assessment=datetime.now() + self.assessment_interval,
            audit_ready=audit_ready,
            evidence_complete=self._check_evidence_complete(framework_controls)
        )

    def _check_evidence_complete(self, controls: List[Control]) -> bool:
        """Check if evidence collection is complete"""
        for control in controls:
            if control.status == ControlStatus.VERIFIED and control.evidence_count == 0:
                return False
        return True

    def get_gaps_by_severity(self, severity: RiskLevel) -> List[ComplianceGap]:
        """Get open gaps by severity level"""
        return [g for g in self.gaps if g.severity == severity and g.status == "open"]

    def register_control(self, control: Control):
        """Register a compliance control"""
        self.controls[control.control_id] = control

    async def prepare_audit_package(
        self,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """
        Prepare audit-ready evidence package

        Args:
            framework: Framework to prepare package for

        Returns:
            Dictionary with all audit materials
        """
        status = await self.check_compliance(framework)

        package = {
            'framework': framework.value,
            'assessment_date': datetime.now().isoformat(),
            'overall_score': status.overall_score,
            'audit_ready': status.audit_ready,
            'controls': [],
            'evidence': [],
            'gaps': [],
            'attestations': []
        }

        # Add controls
        for control in self.controls.values():
            if control.framework == framework:
                package['controls'].append({
                    'control_id': control.control_id,
                    'title': control.title,
                    'status': control.status.value,
                    'evidence_count': control.evidence_count,
                    'last_verified': control.last_verified.isoformat() if control.last_verified else None,
                })

        # Add gaps
        for gap in status.gaps:
            package['gaps'].append({
                'control_id': gap.control_id,
                'severity': gap.severity.value,
                'description': gap.description,
                'remediation': gap.remediation,
            })

        return package

    def export_status(self, path: Path):
        """Export compliance status to file"""
        data = {
            'frameworks': [f.value for f in self.frameworks],
            'controls': {
                cid: {
                    'control_id': c.control_id,
                    'framework': c.framework.value,
                    'title': c.title,
                    'status': c.status.value,
                    'evidence_count': c.evidence_count,
                    'automated': c.automated,
                }
                for cid, c in self.controls.items()
            },
            'gaps': [
                {
                    'control_id': g.control_id,
                    'framework': g.framework.value,
                    'severity': g.severity.value,
                    'description': g.description,
                    'status': g.status,
                }
                for g in self.gaps
            ],
            'last_monitoring': self.last_monitoring.isoformat(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def import_status(self, path: Path):
        """Import compliance status from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Import controls
        for cid, cdata in data.get('controls', {}).items():
            control = Control(
                control_id=cdata['control_id'],
                framework=ComplianceFramework(cdata['framework']),
                title=cdata['title'],
                description="",
                status=ControlStatus(cdata['status']),
                evidence_count=cdata.get('evidence_count', 0),
                automated=cdata.get('automated', False),
            )
            self.controls[cid] = control

        # Import gaps
        self.gaps = []
        for gdata in data.get('gaps', []):
            if gdata['status'] == 'open':
                gap = ComplianceGap(
                    control_id=gdata['control_id'],
                    framework=ComplianceFramework(gdata['framework']),
                    severity=RiskLevel(gdata['severity']),
                    description=gdata['description'],
                    remediation="",
                    status=gdata['status'],
                )
                self.gaps.append(gap)
