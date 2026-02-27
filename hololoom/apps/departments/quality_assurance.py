"""
Quality Assurance Department - DATAPIG Integration

"Make it so... but validate everything first!" - Captain Picard (adapted)

Unified code + data quality assurance using Trough + DATAPIG + xTerminator.
"""

from typing import Dict, List, Optional, Any
import logging

from .base import BaseDepartment
from .protocol import (
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceMetadata,
    ConfidenceLevel
)

# DATAPIG imports
try:
    from hololoom.datapig import DataPigDetector, Severity as DataPigSeverity
    from hololoom.datapig.config import create_config
    DATAPIG_AVAILABLE = True
except ImportError:
    DATAPIG_AVAILABLE = False
    DataPigDetector = None

# Trough imports
try:
    from trough.datapig_integration import UnifiedDetector
    TROUGH_AVAILABLE = True
except ImportError:
    TROUGH_AVAILABLE = False
    UnifiedDetector = None

# xTerminator imports
try:
    from xterminator.datapig_fixer import fix_dataset
    XTERMINATOR_AVAILABLE = True
except ImportError:
    XTERMINATOR_AVAILABLE = False


logger = logging.getLogger(__name__)


class QualityAssuranceDepartment(BaseDepartment):
    """
    Quality Assurance Department

    Provides unified code + data quality validation using:
    - DATAPIG: Data quality detection
    - Trough: Code quality detection
    - xTerminator: Auto-fixing

    Supported Tasks:
    - validate_data: Data quality validation
    - fix_data: Auto-fix data quality issues
    - validate_code: Code quality validation (via Trough)
    - unified_scan: Both code + data quality
    """

    def __init__(self):
        super().__init__(
            name="quality_assurance",
            domain="qa",
            version="1.0.0",
            supported_tasks=[
                "validate_data",
                "fix_data",
                "validate_code",
                "unified_scan",
                "warp_validation"
            ],
            confidence_range=(0.70, 0.95)
        )

        # Initialize detectors
        if DATAPIG_AVAILABLE:
            self.datapig = DataPigDetector(enable_verbose=False)
        else:
            self.datapig = None
            logger.warning("DATAPIG not available - data quality validation disabled")

        if TROUGH_AVAILABLE:
            self.trough = UnifiedDetector(enable_datapig=DATAPIG_AVAILABLE)
        else:
            self.trough = None
            logger.warning("Trough not available - code quality validation disabled")

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute QA task

        Supported actions:
        - validate_data: Run DATAPIG on dataset
        - fix_data: Auto-fix data quality issues
        - validate_code: Run Trough on code file
        - unified_scan: Both code + data quality
        - warp_validation: Quick CAPTAIN-level check
        """
        action = request.parameters.get("action")

        if action == "validate_data":
            return await self._validate_data(request)
        elif action == "fix_data":
            return await self._fix_data(request)
        elif action == "validate_code":
            return await self._validate_code(request)
        elif action == "unified_scan":
            return await self._unified_scan(request)
        elif action == "warp_validation":
            return await self._warp_validation(request)
        else:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": f"Unknown action: {action}"},
                confidence=ConfidenceMetadata.from_score(0.0, justification=["Unknown action"])
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify QA response quality

        QA responses are verified by checking:
        - No critical issues (CAPTAIN-level)
        - Confidence >= 0.70
        """
        if "error" in response.result:
            return VerificationResult(
                sufficient=False,
                confidence_valid=False,
                reasoning_sound=False,
                issues=["Error in QA execution"]
            )

        # Check for critical issues
        critical_count = response.result.get("by_severity", {}).get("CAPTAIN", 0)
        sufficient = critical_count == 0

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=response.confidence.score >= 0.70,
            reasoning_sound=True,
            issues=[] if sufficient else [f"{critical_count} critical issues detected"]
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Refine QA response

        For QA, refinement means re-running with stricter settings
        """
        # Re-execute with strict preset
        request = DepartmentRequest(
            task_id=response.task_id,
            task_type=response.task_id.split("_")[0],  # Extract action from task_id
            parameters={**response.result.get("original_params", {}), "preset": "strict"}
        )

        return await self.execute(request)

    # ========================================================================
    # Task Implementations
    # ========================================================================

    async def _validate_data(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Validate data quality with DATAPIG

        Parameters:
            - data: List[Dict] - Dataset to validate
            - severity_threshold: str - CAPTAIN/COMMANDER/LIEUTENANT/ENSIGN
            - preset: str - Configuration preset
        """
        if not DATAPIG_AVAILABLE:
            return self._error_response(request, "DATAPIG not available")

        data = request.parameters.get("data", [])
        severity_threshold = request.parameters.get("severity_threshold", "COMMANDER")
        preset = request.parameters.get("preset", "default")

        # Create detector with config
        config = create_config(preset)
        self.datapig = DataPigDetector(enable_verbose=False)

        # Analyze
        all_issues = self.datapig.analyze_dataset(data)

        # Filter by severity
        threshold_map = {"CAPTAIN": 4, "COMMANDER": 3, "LIEUTENANT": 2, "ENSIGN": 1}
        threshold_val = threshold_map.get(severity_threshold, 2)
        filtered_issues = [
            i for i in all_issues
            if threshold_map.get(i.severity.value, 0) >= threshold_val
        ]

        # Count by severity
        by_severity = {
            severity.value: len([i for i in filtered_issues if i.severity.value == severity.value])
            for severity in DataPigSeverity
        }

        # Check if safe
        captain_count = by_severity.get("CAPTAIN", 0)
        safe = captain_count == 0

        # Compute confidence (higher confidence = fewer issues)
        confidence_score = 1.0 - (len(filtered_issues) / max(len(data), 1))
        confidence_score = max(0.5, min(1.0, confidence_score))  # Clamp to [0.5, 1.0]

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "status": "all_clear" if safe else "red_alert",
                "safe_for_production": safe,
                "total_issues": len(filtered_issues),
                "by_severity": by_severity,
                "issues": [self._serialize_issue(i) for i in filtered_issues[:20]],
                "original_params": request.parameters
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[
                    f"Detected {len(filtered_issues)} issues",
                    f"Safe for production: {safe}"
                ]
            )
        )

    async def _fix_data(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Auto-fix data quality issues with xTerminator

        Parameters:
            - data: List[Dict] - Dataset to fix
            - issues: Optional[List[Dict]] - Pre-detected issues (if None, detect first)
            - validate: bool - Enable validation after fixes
        """
        if not DATAPIG_AVAILABLE or not XTERMINATOR_AVAILABLE:
            return self._error_response(request, "DATAPIG or xTerminator not available")

        data = request.parameters.get("data", [])
        issues_dicts = request.parameters.get("issues")
        validate = request.parameters.get("validate", True)

        # Detect issues if not provided
        if issues_dicts is None:
            all_issues = self.datapig.analyze_dataset(data)
        else:
            # Convert from dict format
            all_issues = [self._dict_to_issue(i) for i in issues_dicts]

        # Fix
        result = fix_dataset(data, all_issues, validate=validate)

        confidence_score = 0.9 if result["success"] else 0.3

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "status": "fixed" if result["success"] else "failed",
                "fixed_data": result["fixed_data"] if result["success"] else None,
                "fixes_applied": result["fixes_applied"],
                "validation_passed": result["validation_passed"],
                "validation_errors": result["validation_errors"]
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[
                    f"Applied {len(result['fixes_applied'])} fixes",
                    f"Validation: {'passed' if result['validation_passed'] else 'failed'}"
                ]
            )
        )

    async def _validate_code(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Validate code quality with Trough

        Parameters:
            - file_path: str - Path to code file
        """
        if not TROUGH_AVAILABLE:
            return self._error_response(request, "Trough not available")

        file_path = request.parameters.get("file_path")
        if not file_path:
            return self._error_response(request, "file_path required")

        # Run Trough detection
        issues = self.trough.detect_all(file_path)

        confidence_score = 1.0 - (len(issues) / 100.0)  # Rough heuristic
        confidence_score = max(0.5, min(1.0, confidence_score))

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "status": "clean" if not issues else "issues_found",
                "total_issues": len(issues),
                "issues": [self._serialize_trough_issue(i) for i in issues[:20]]
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[f"Detected {len(issues)} code quality issues"]
            )
        )

    async def _unified_scan(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Unified code + data quality scan

        Parameters:
            - file_path: str - Path to file (code or data)
        """
        if not TROUGH_AVAILABLE:
            return self._error_response(request, "Trough not available")

        file_path = request.parameters.get("file_path")
        if not file_path:
            return self._error_response(request, "file_path required")

        # Run unified detection
        from trough.datapig_integration import analyze_file_unified
        result = analyze_file_unified(file_path)

        confidence_score = 1.0 - (result["summary"]["total_issues"] / 100.0)
        confidence_score = max(0.5, min(1.0, confidence_score))

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[
                    f"File type: {result['file_type']}",
                    f"Total issues: {result['summary']['total_issues']}"
                ]
            )
        )

    async def _warp_validation(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Quick warp validation (CAPTAIN-level only)

        Parameters:
            - data: List[Dict] - Dataset to validate
        """
        if not DATAPIG_AVAILABLE:
            return self._error_response(request, "DATAPIG not available")

        data = request.parameters.get("data", [])

        # Quick check: only CAPTAIN issues
        all_issues = self.datapig.analyze_dataset(data)
        captain_issues = [i for i in all_issues if i.severity == DataPigSeverity.CAPTAIN]

        safe_for_warp = len(captain_issues) == 0

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "safe_for_warp": safe_for_warp,
                "critical_issues": len(captain_issues),
                "total_issues": len(all_issues),
                "message": "*** All systems nominal. Engaging warp drive! ***" if safe_for_warp
                          else f"*** RED ALERT! {len(captain_issues)} critical issues detected! ***"
            },
            confidence=ConfidenceMetadata.from_score(
                1.0 if safe_for_warp else 0.3,
                justification=[f"Warp safe: {safe_for_warp}"]
            )
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _error_response(self, request: DepartmentRequest, error: str) -> DepartmentResponse:
        """Create error response"""
        return DepartmentResponse(
            task_id=request.task_id,
            result={"error": error},
            confidence=ConfidenceMetadata.from_score(0.0, justification=[error])
        )

    def _serialize_issue(self, issue) -> Dict[str, Any]:
        """Serialize DATAPIG issue to dict"""
        return {
            "type": issue.issue_type.value,
            "severity": issue.severity.value,
            "message": issue.message,
            "location": issue.location,
            "details": issue.details
        }

    def _serialize_trough_issue(self, issue) -> Dict[str, Any]:
        """Serialize Trough issue to dict"""
        return {
            "category": issue.category.value if hasattr(issue.category, 'value') else str(issue.category),
            "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
            "message": issue.message,
            "line_number": issue.line_number,
            "suggestion": issue.suggestion
        }

    def _dict_to_issue(self, issue_dict: Dict):
        """Convert dict back to DataQualityIssue"""
        from hololoom.datapig.detector import DataQualityIssue, IssueType, Severity

        return DataQualityIssue(
            issue_type=IssueType[issue_dict["type"]],
            severity=Severity[issue_dict["severity"]],
            message=issue_dict["message"],
            location=issue_dict["location"],
            details=issue_dict.get("details", {}),
            stardate=0.0
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_qa_department() -> QualityAssuranceDepartment:
    """Create Quality Assurance department instance"""
    return QualityAssuranceDepartment()
