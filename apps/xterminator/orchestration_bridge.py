"""
Orchestration Bridge
====================

🐷 Phase 3: Cross-Department Quality Integration! 🐷

Connects QA Department with the HoloLoom Orchestration layer, enabling:
- Cross-department scanning (QA scans outputs from all departments)
- Quality gates (block operations if QA fails)
- Error escalation (route high-risk issues to humans)
- Automated quality enforcement

The Bridge:
- Orchestration → QA: "Scan this MasterWeaver output"
- QA → Orchestration: "Found 3 issues (2 auto-fixed, 1 needs review)"
- Orchestration → Human: "Review required for 1 issue"

Integration Points:
- MasterWeaver outputs → QA scanning
- Infrastructure migrations → QA validation
- Any department → QA quality check
- High-risk issues → Human escalation

Moonshot Integration:
- Enables cross-department quality feedback (Idea #1)
- Provides confidence authority for all departments (Idea #4)
- Prepares for live monitoring (Idea #7)

Usage:
    # Create orchestration bridge
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Scan output from another department
    result = await bridge.scan_department_output(
        department_name="MasterWeaver",
        output_code=entity_extractor_code,
        file_path="entity_extractor.py"
    )

    # Check if quality gate passes
    if result.quality_gate_passed:
        print("✓ Quality gate passed - proceed")
    else:
        print("✗ Quality gate failed - blocked")
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from .qa_department import QADepartment
from .department_protocol import (
    DepartmentRequest,
    DepartmentResponse,
    RequestType,
    ResponseStatus
)
from .autofix_policy import FixDecision


class EscalationLevel(Enum):
    """Escalation level for quality issues"""
    NONE = "none"              # No escalation needed
    NOTIFY = "notify"          # Notify but don't block
    REVIEW = "review"          # Human review required
    BLOCK = "block"            # Block operation entirely
    CRITICAL = "critical"      # Critical issue, alert immediately


class QualityGateResult(Enum):
    """Result of quality gate check"""
    PASSED = "passed"          # All checks passed
    WARNING = "warning"        # Warnings but can proceed
    FAILED = "failed"          # Failed, should not proceed
    BLOCKED = "blocked"        # Blocked by policy


@dataclass
class ScanResult:
    """
    Result of scanning a department's output.

    Contains QA findings, quality gate status, and escalation recommendations.
    """
    scan_id: str
    department_scanned: str
    file_path: str
    timestamp: float = field(default_factory=time.time)

    # Scan findings
    issues_found: int = 0
    issues_auto_fixed: int = 0
    issues_need_review: int = 0
    issues_blocked: int = 0

    # Quality assessment
    quality_score: float = 0.0  # 0.0-1.0
    confidence: float = 0.0     # 0.0-1.0

    # Quality gate
    quality_gate_result: QualityGateResult = QualityGateResult.PASSED
    quality_gate_reason: str = ""
    quality_gate_passed: bool = True

    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.NONE
    escalation_reason: str = ""
    escalation_details: Dict[str, Any] = field(default_factory=dict)

    # Performance
    scan_duration_ms: float = 0.0

    # Raw QA response
    qa_response: Optional[DepartmentResponse] = None

    def summary(self) -> str:
        """Human-readable summary"""
        gate_icons = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌',
            'blocked': '🚫'
        }
        icon = gate_icons.get(self.quality_gate_result.value, '?')

        lines = [
            f"Scan {self.scan_id}:",
            f"  Department: {self.department_scanned}",
            f"  File: {self.file_path}",
            f"  Issues: {self.issues_found} found",
            f"    - {self.issues_auto_fixed} auto-fixed",
            f"    - {self.issues_need_review} need review",
            f"    - {self.issues_blocked} blocked",
            f"  Quality Score: {self.quality_score:.2f}",
            f"  Confidence: {self.confidence:.2f}",
            f"  {icon} Quality Gate: {self.quality_gate_result.value}",
        ]

        if self.escalation_level != EscalationLevel.NONE:
            lines.append(f"  ⚠️  Escalation: {self.escalation_level.value} - {self.escalation_reason}")

        lines.append(f"  Duration: {self.scan_duration_ms:.0f}ms")

        return "\n".join(lines)


@dataclass
class QualityGateConfig:
    """
    Configuration for quality gate thresholds.

    Different departments can have different quality standards.
    """
    department_name: str
    min_quality_score: float = 0.70      # Minimum quality score to pass
    min_confidence: float = 0.75         # Minimum confidence to pass
    max_issues_allowed: int = 10         # Maximum issues before failing
    max_critical_issues: int = 0         # Maximum critical issues (usually 0)
    block_on_failure: bool = True        # Block operation if gate fails?
    require_human_review: bool = False   # Always require human review?

    @classmethod
    def strict(cls, department_name: str):
        """Strict quality gate (healthcare, finance)"""
        return cls(
            department_name=department_name,
            min_quality_score=0.95,
            min_confidence=0.90,
            max_issues_allowed=0,
            max_critical_issues=0,
            block_on_failure=True,
            require_human_review=True
        )

    @classmethod
    def balanced(cls, department_name: str):
        """Balanced quality gate (default)"""
        return cls(
            department_name=department_name,
            min_quality_score=0.75,
            min_confidence=0.80,
            max_issues_allowed=5,
            max_critical_issues=0,
            block_on_failure=True,
            require_human_review=False
        )

    @classmethod
    def relaxed(cls, department_name: str):
        """Relaxed quality gate (beekeeping, internal)"""
        return cls(
            department_name=department_name,
            min_quality_score=0.60,
            min_confidence=0.70,
            max_issues_allowed=20,
            max_critical_issues=1,
            block_on_failure=False,
            require_human_review=False
        )


class OrchestrationBridge:
    """
    Bridge between Orchestration and QA Department.

    Enables cross-department quality scanning and enforcement.
    """

    def __init__(
        self,
        qa_department: QADepartment,
        default_quality_gate: Optional[QualityGateConfig] = None,
        enable_auto_escalation: bool = True
    ):
        """
        Initialize OrchestrationBridge.

        Args:
            qa_department: QA Department instance
            default_quality_gate: Default quality gate config
            enable_auto_escalation: Auto-escalate high-risk issues?
        """
        self.qa_dept = qa_department
        self.default_gate = default_quality_gate or QualityGateConfig.balanced("default")
        self.enable_auto_escalation = enable_auto_escalation

        # Per-department quality gates
        self.quality_gates: Dict[str, QualityGateConfig] = {}

        # Escalation handlers
        self.escalation_handlers: Dict[EscalationLevel, List[Callable]] = {
            level: [] for level in EscalationLevel
        }

        # Metrics
        self.total_scans = 0
        self.passed_scans = 0
        self.failed_scans = 0
        self.escalated_scans = 0

    def register_quality_gate(self, department_name: str, config: QualityGateConfig):
        """
        Register quality gate for a department.

        Args:
            department_name: Department name
            config: Quality gate configuration
        """
        self.quality_gates[department_name] = config

    def register_escalation_handler(
        self,
        level: EscalationLevel,
        handler: Callable[[ScanResult], None]
    ):
        """
        Register handler for escalation events.

        Args:
            level: Escalation level
            handler: Async function to call on escalation
        """
        self.escalation_handlers[level].append(handler)

    async def scan_department_output(
        self,
        department_name: str,
        output_code: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScanResult:
        """
        Scan output from a department for quality issues.

        Args:
            department_name: Name of department whose output is being scanned
            output_code: Code output to scan
            file_path: File path
            metadata: Additional metadata

        Returns:
            ScanResult with findings and quality gate status
        """
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]
        self.total_scans += 1

        # Create QA request
        request = DepartmentRequest(
            request_id=f"scan_{scan_id}",
            request_type=RequestType.SCAN_CODE,
            requesting_department="Orchestration",
            priority=3,
            payload={
                'code': output_code,
                'file_path': file_path,
                'source_department': department_name
            },
            context=metadata or {}
        )

        # Execute scan
        response = await self.qa_dept.execute(request)

        # Parse results (placeholder - in real implementation would parse actual issues)
        issues_found = response.payload.get('issues_found', 0)
        issues_auto_fixed = response.payload.get('issues_auto_fixed', 0)
        issues_need_review = response.payload.get('issues_need_review', 0)
        issues_blocked = response.payload.get('issues_blocked', 0)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            issues_found, issues_auto_fixed, issues_need_review, issues_blocked
        )

        # Get quality gate config
        gate_config = self.quality_gates.get(department_name, self.default_gate)

        # Check quality gate
        gate_result, gate_reason, gate_passed = self._check_quality_gate(
            quality_score=quality_score,
            confidence=response.confidence,
            issues_found=issues_found,
            issues_blocked=issues_blocked,
            config=gate_config
        )

        # Determine escalation
        escalation_level, escalation_reason, escalation_details = self._determine_escalation(
            gate_result=gate_result,
            issues_blocked=issues_blocked,
            issues_need_review=issues_need_review,
            confidence=response.confidence,
            config=gate_config
        )

        # Create scan result
        scan_result = ScanResult(
            scan_id=scan_id,
            department_scanned=department_name,
            file_path=file_path,
            issues_found=issues_found,
            issues_auto_fixed=issues_auto_fixed,
            issues_need_review=issues_need_review,
            issues_blocked=issues_blocked,
            quality_score=quality_score,
            confidence=response.confidence,
            quality_gate_result=gate_result,
            quality_gate_reason=gate_reason,
            quality_gate_passed=gate_passed,
            escalation_level=escalation_level,
            escalation_reason=escalation_reason,
            escalation_details=escalation_details,
            scan_duration_ms=(time.time() - start_time) * 1000,
            qa_response=response
        )

        # Update metrics
        if gate_passed:
            self.passed_scans += 1
        else:
            self.failed_scans += 1

        if escalation_level != EscalationLevel.NONE:
            self.escalated_scans += 1
            # Trigger escalation handlers
            await self._trigger_escalation(scan_result)

        return scan_result

    async def scan_multiple_departments(
        self,
        scan_requests: List[Dict[str, Any]]
    ) -> List[ScanResult]:
        """
        Scan outputs from multiple departments in parallel.

        Args:
            scan_requests: List of scan request dicts with keys:
                - department_name
                - output_code
                - file_path
                - metadata (optional)

        Returns:
            List of ScanResult objects
        """
        tasks = [
            self.scan_department_output(
                department_name=req['department_name'],
                output_code=req['output_code'],
                file_path=req['file_path'],
                metadata=req.get('metadata')
            )
            for req in scan_requests
        ]

        return await asyncio.gather(*tasks)

    def _calculate_quality_score(
        self,
        issues_found: int,
        issues_auto_fixed: int,
        issues_need_review: int,
        issues_blocked: int
    ) -> float:
        """
        Calculate quality score (0.0-1.0) based on issues found.

        Formula:
        - Start with 1.0
        - Deduct for each issue type
        - More severe issues deduct more
        """
        score = 1.0

        # Deductions
        score -= issues_auto_fixed * 0.05   # -5% per auto-fixed issue
        score -= issues_need_review * 0.10  # -10% per review issue
        score -= issues_blocked * 0.20      # -20% per blocked issue

        return max(0.0, score)

    def _check_quality_gate(
        self,
        quality_score: float,
        confidence: float,
        issues_found: int,
        issues_blocked: int,
        config: QualityGateConfig
    ) -> tuple[QualityGateResult, str, bool]:
        """
        Check if quality gate passes.

        Returns:
            (QualityGateResult, reason, passed)
        """
        # Check critical issues
        if issues_blocked > config.max_critical_issues:
            return (
                QualityGateResult.BLOCKED,
                f"Critical issues ({issues_blocked}) exceed limit ({config.max_critical_issues})",
                False
            )

        # Check total issues
        if issues_found > config.max_issues_allowed:
            if config.block_on_failure:
                return (
                    QualityGateResult.FAILED,
                    f"Issues ({issues_found}) exceed limit ({config.max_issues_allowed})",
                    False
                )
            else:
                return (
                    QualityGateResult.WARNING,
                    f"Issues ({issues_found}) exceed limit ({config.max_issues_allowed}) but not blocking",
                    True
                )

        # Check quality score
        if quality_score < config.min_quality_score:
            if config.block_on_failure:
                return (
                    QualityGateResult.FAILED,
                    f"Quality score ({quality_score:.2f}) below minimum ({config.min_quality_score})",
                    False
                )
            else:
                return (
                    QualityGateResult.WARNING,
                    f"Quality score ({quality_score:.2f}) below minimum ({config.min_quality_score}) but not blocking",
                    True
                )

        # Check confidence
        if confidence < config.min_confidence:
            return (
                QualityGateResult.WARNING,
                f"Confidence ({confidence:.2f}) below minimum ({config.min_confidence})",
                True
            )

        # All checks passed
        return (
            QualityGateResult.PASSED,
            "All quality checks passed",
            True
        )

    def _determine_escalation(
        self,
        gate_result: QualityGateResult,
        issues_blocked: int,
        issues_need_review: int,
        confidence: float,
        config: QualityGateConfig
    ) -> tuple[EscalationLevel, str, Dict[str, Any]]:
        """
        Determine escalation level based on scan results.

        Returns:
            (EscalationLevel, reason, details)
        """
        details = {
            'gate_result': gate_result.value,
            'issues_blocked': issues_blocked,
            'issues_need_review': issues_need_review,
            'confidence': confidence
        }

        # Critical: blocked issues
        if issues_blocked > 0:
            return (
                EscalationLevel.CRITICAL,
                f"{issues_blocked} critical issues require immediate attention",
                details
            )

        # Block: quality gate failed with blocking enabled
        if gate_result == QualityGateResult.BLOCKED:
            return (
                EscalationLevel.BLOCK,
                "Quality gate blocked - operation cannot proceed",
                details
            )

        # Review: issues need human review
        if issues_need_review > 0:
            return (
                EscalationLevel.REVIEW,
                f"{issues_need_review} issues need human review",
                details
            )

        # Review: policy requires human review
        if config.require_human_review:
            return (
                EscalationLevel.REVIEW,
                "Policy requires human review",
                details
            )

        # Notify: quality gate warning
        if gate_result == QualityGateResult.WARNING:
            return (
                EscalationLevel.NOTIFY,
                "Quality warning detected",
                details
            )

        # No escalation needed
        return (EscalationLevel.NONE, "", details)

    async def _trigger_escalation(self, scan_result: ScanResult):
        """Trigger escalation handlers"""
        handlers = self.escalation_handlers.get(scan_result.escalation_level, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(scan_result)
                else:
                    handler(scan_result)
            except Exception as e:
                print(f"Warning: Escalation handler failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestration bridge metrics"""
        return {
            'total_scans': self.total_scans,
            'passed_scans': self.passed_scans,
            'failed_scans': self.failed_scans,
            'escalated_scans': self.escalated_scans,
            'pass_rate': self.passed_scans / self.total_scans if self.total_scans > 0 else 0.0,
            'escalation_rate': self.escalated_scans / self.total_scans if self.total_scans > 0 else 0.0
        }
