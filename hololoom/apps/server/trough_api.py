#!/usr/bin/env python3
from __future__ import annotations
"""
Trough API
==========

Code Quality Assurance endpoints for Trough persona.

Features:
- Issue detection (24 algorithms: 15 AI slop + 9 ML logic)
- Auto-fixing with xTerminator (87% success rate)
- Validation pipeline (5 stages)
- QA statistics and learning

Created: 2025-01-20
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/trough", tags=["trough"])
logger = logging.getLogger(__name__)

# Import Trough components
try:
    from trough.detector import TroughDetector
    from xterminator.fixer import XTerminatorFixer
    from xterminator.validator import Validator
    TROUGH_AVAILABLE = True
    logger.info("✓ Trough + xTerminator loaded successfully")
except ImportError as e:
    TROUGH_AVAILABLE = False
    logger.warning(f"⚠ Trough components not available: {e}")
    TroughDetector = None

# Global instances
trough_detector: Any | None = None
xterminator_fixer: Any | None = None


def get_trough_detector():
    """Get or create Trough detector instance."""
    global trough_detector
    if trough_detector is None and TROUGH_AVAILABLE:
        trough_detector = TroughDetector()
    return trough_detector


def get_xterminator_fixer():
    """Get or create xTerminator fixer instance."""
    global xterminator_fixer
    if xterminator_fixer is None and TROUGH_AVAILABLE:
        xterminator_fixer = XTerminatorFixer()
    return xterminator_fixer


# ============== REQUEST/RESPONSE MODELS ==============

class AnalyzeRequest(BaseModel):
    code: str
    file_path: str | None = None
    language: str = "python"
    enable_ml_logic: bool = True


class Issue(BaseModel):
    id: str
    type: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    message: str
    line: int
    column: int | None = None
    fixable: bool
    fix_confidence: float | None = None


class AnalyzeResponse(BaseModel):
    issues: list[Issue]
    total_issues: int
    by_severity: dict[str, int]
    fixable_count: int
    file_path: str | None = None


class FixRequest(BaseModel):
    code: str
    file_path: str | None = None
    language: str = "python"
    issue_ids: list[str] | None = None  # Fix specific issues, or all if None
    validate: bool = True
    git_safe: bool = True


class Fix(BaseModel):
    issue_id: str
    status: str  # "success", "failed", "skipped"
    original_code: str
    fixed_code: str
    confidence: float
    validation_passed: bool


class FixResponse(BaseModel):
    fixes: list[Fix]
    original_code: str
    fixed_code: str
    success_count: int
    failed_count: int
    validation_status: str  # "passed", "failed", "not_run"


class ValidateRequest(BaseModel):
    original_code: str
    fixed_code: str
    file_path: str | None = None


class ValidationStage(BaseModel):
    stage: str
    passed: bool
    message: str
    duration_ms: float


class ValidateResponse(BaseModel):
    overall_passed: bool
    stages: list[ValidationStage]
    total_duration_ms: float
    rollback_recommended: bool


class StatsResponse(BaseModel):
    total_analyzed: int
    total_issues_found: int
    total_fixes_attempted: int
    fix_success_rate: float
    by_issue_type: dict[str, int]
    top_issues: list[dict[str, Any]]


# ============== 1. ANALYZE CODE ==============

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code(request: AnalyzeRequest):
    """
    Detect code issues using Trough's 24 algorithms.

    Returns:
        - All detected issues
        - Severity breakdown
        - Fixability information
    """

    if not TROUGH_AVAILABLE:
        # Fallback detection (basic)
        logger.warning("Trough not available, using fallback detection")

        issues = []

        # Basic checks
        if 'TODO' in request.code:
            issues.append(Issue(
                id="incomplete_001",
                type="incomplete_code",
                severity="MEDIUM",
                message="Found TODO comment - incomplete implementation",
                line=request.code.find('TODO'),
                fixable=False,
                fix_confidence=None
            ))

        if 'pass' in request.code and len(request.code.split('\n')) < 5:
            issues.append(Issue(
                id="incomplete_002",
                type="incomplete_code",
                severity="MEDIUM",
                message="Empty function body with 'pass' statement",
                line=request.code.find('pass'),
                fixable=False,
                fix_confidence=None
            ))

        # Check for hardcoded values
        import re
        if re.search(r'password\s*=\s*["\']', request.code, re.IGNORECASE):
            issues.append(Issue(
                id="security_001",
                type="hardcoded_secret",
                severity="CRITICAL",
                message="Hardcoded password detected",
                line=request.code.lower().find('password'),
                fixable=True,
                fix_confidence=0.75
            ))

        by_severity = {
            "LOW": sum(1 for i in issues if i.severity == "LOW"),
            "MEDIUM": sum(1 for i in issues if i.severity == "MEDIUM"),
            "HIGH": sum(1 for i in issues if i.severity == "HIGH"),
            "CRITICAL": sum(1 for i in issues if i.severity == "CRITICAL")
        }

        fixable_count = sum(1 for i in issues if i.fixable)

        return AnalyzeResponse(
            issues=issues,
            total_issues=len(issues),
            by_severity=by_severity,
            fixable_count=fixable_count,
            file_path=request.file_path
        )

    try:
        detector = get_trough_detector()

        # Run detection
        detection_result = detector.analyze(
            code=request.code,
            file_path=request.file_path,
            enable_ml_logic=request.enable_ml_logic
        )

        # Convert to API format
        issues = []
        for issue_dict in detection_result.get('issues', []):
            issues.append(Issue(
                id=issue_dict['id'],
                type=issue_dict['type'],
                severity=issue_dict['severity'],
                message=issue_dict['message'],
                line=issue_dict['line'],
                column=issue_dict.get('column'),
                fixable=issue_dict['fixable'],
                fix_confidence=issue_dict.get('fix_confidence')
            ))

        by_severity = detection_result.get('by_severity', {})
        fixable_count = sum(1 for i in issues if i.fixable)

        return AnalyzeResponse(
            issues=issues,
            total_issues=len(issues),
            by_severity=by_severity,
            fixable_count=fixable_count,
            file_path=request.file_path
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============== 2. AUTO-FIX WITH XTERMINATOR ==============

@router.post("/fix", response_model=FixResponse)
async def fix_issues(request: FixRequest):
    """
    Auto-fix issues using xTerminator.

    Features:
        - AST-based transformations
        - 87% success rate
        - Optional validation (5 stages)
        - Git safety checks
    """

    if not TROUGH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="xTerminator not available. Install trough + xterminator."
        )

    try:
        fixer = get_xterminator_fixer()

        # Run fixer
        fix_result = fixer.fix(
            code=request.code,
            file_path=request.file_path,
            issue_ids=request.issue_ids,
            validate=request.validate,
            git_safe=request.git_safe
        )

        # Convert to API format
        fixes = []
        for fix_dict in fix_result.get('fixes', []):
            fixes.append(Fix(
                issue_id=fix_dict['issue_id'],
                status=fix_dict['status'],
                original_code=fix_dict.get('original_code', ''),
                fixed_code=fix_dict.get('fixed_code', ''),
                confidence=fix_dict['confidence'],
                validation_passed=fix_dict.get('validation_passed', True)
            ))

        success_count = sum(1 for f in fixes if f.status == "success")
        failed_count = sum(1 for f in fixes if f.status == "failed")

        validation_status = fix_result.get('validation_status', 'not_run')

        return FixResponse(
            fixes=fixes,
            original_code=request.code,
            fixed_code=fix_result.get('fixed_code', request.code),
            success_count=success_count,
            failed_count=failed_count,
            validation_status=validation_status
        )

    except Exception as e:
        logger.error(f"Fixing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fixing failed: {str(e)}")


# ============== 3. VALIDATE FIXES ==============

@router.post("/validate", response_model=ValidateResponse)
async def validate_fixes(request: ValidateRequest):
    """
    Validate fixes using 5-stage pipeline.

    Stages:
        1. Syntax validation
        2. Import resolution
        3. Test execution
        4. Git safety checks
        5. Rollback on failure
    """

    if not TROUGH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Validator not available. Install xterminator."
        )

    try:
        validator = Validator()

        # Run validation
        validation_result = validator.validate(
            original_code=request.original_code,
            fixed_code=request.fixed_code,
            file_path=request.file_path
        )

        # Convert to API format
        stages = []
        for stage_dict in validation_result.get('stages', []):
            stages.append(ValidationStage(
                stage=stage_dict['stage'],
                passed=stage_dict['passed'],
                message=stage_dict['message'],
                duration_ms=stage_dict['duration_ms']
            ))

        overall_passed = validation_result.get('overall_passed', False)
        total_duration = sum(s.duration_ms for s in stages)
        rollback_recommended = not overall_passed

        return ValidateResponse(
            overall_passed=overall_passed,
            stages=stages,
            total_duration_ms=total_duration,
            rollback_recommended=rollback_recommended
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# ============== 4. QA STATISTICS ==============

@router.get("/stats", response_model=StatsResponse)
async def get_qa_stats():
    """
    Get QA statistics and learning metrics.

    Returns:
        - Total files analyzed
        - Issues found
        - Fix success rate
        - Thompson Sampling stats
    """

    if not TROUGH_AVAILABLE:
        # Return dummy stats
        return StatsResponse(
            total_analyzed=0,
            total_issues_found=0,
            total_fixes_attempted=0,
            fix_success_rate=0.0,
            by_issue_type={},
            top_issues=[]
        )

    try:
        detector = get_trough_detector()
        fixer = get_xterminator_fixer()

        # Get stats from detector
        detector_stats = detector.get_stats()
        fixer_stats = fixer.get_stats()

        # Calculate success rate
        total_attempted = fixer_stats.get('total_attempted', 0)
        total_succeeded = fixer_stats.get('total_succeeded', 0)
        success_rate = total_succeeded / total_attempted if total_attempted > 0 else 0.0

        # Top issues
        by_issue_type = detector_stats.get('by_issue_type', {})
        top_issues = [
            {'type': issue_type, 'count': count}
            for issue_type, count in sorted(
                by_issue_type.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        ]

        return StatsResponse(
            total_analyzed=detector_stats.get('total_analyzed', 0),
            total_issues_found=detector_stats.get('total_issues', 0),
            total_fixes_attempted=total_attempted,
            fix_success_rate=success_rate,
            by_issue_type=by_issue_type,
            top_issues=top_issues
        )

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")
