#!/usr/bin/env python3
"""
Verification Department - Cross-department fact-checking and validation.

Provides verification services across departments:
- Cross-check facts between departments
- Validate confidence scores
- Detect inconsistencies
- Generate verification reports

Design Philosophy:
"Trust, but verify."

Supported Tasks:
- "verify_response": Verify a single department response
- "cross_check": Cross-check across multiple departments
- "validate_confidence": Validate confidence calibration
- "detect_inconsistencies": Find conflicts between departments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..base import BaseDepartment
from ..protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class CrossCheckResult:
    """Result of cross-checking across departments."""
    consistent: bool
    confidence_agreement: float
    conflicting_claims: list[dict[str, Any]]
    agreement_level: str  # "full", "partial", "conflict"
    evidence: dict[str, Any]


@dataclass
class InconsistencyReport:
    """Report of detected inconsistencies."""
    inconsistency_type: str
    severity: str  # "low", "medium", "high", "critical"
    departments_involved: list[str]
    details: str
    recommended_action: str


# ============================================================================
# Verification Department
# ============================================================================

class VerificationDepartment(BaseDepartment):
    """
    Verification Department - Cross-department validation.

    Provides verification services:
    1. Single-department response verification
    2. Cross-department fact-checking
    3. Confidence score validation
    4. Inconsistency detection

    Example:

        async with VerificationDepartment(registry=registry) as dept:
            # Verify single response
            request = DepartmentRequest(
                task_id="verify_001",
                task_type="verify_response",
                parameters={
                    "department": "context",
                    "response": context_response
                }
            )

            result = await dept.execute(request)

            # Cross-check across departments
            cross_check_request = DepartmentRequest(
                task_id="cross_001",
                task_type="cross_check",
                parameters={
                    "query": "What is Thompson Sampling?",
                    "departments": ["context", "masterweaver"]
                }
            )

            cross_result = await dept.execute(cross_check_request)
    """

    def __init__(
        self,
        registry: Any | None = None,
        confidence_threshold: float = 0.75,
        consistency_threshold: float = 0.80,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize Verification Department.

        Args:
            registry: Department registry for cross-department checks
            confidence_threshold: Minimum acceptable confidence
            consistency_threshold: Minimum consistency for cross-checks
            dept_config: Department configuration
        """
        super().__init__(
            name="verification",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "verify_response",
                "cross_check",
                "validate_confidence",
                "detect_inconsistencies"
            ],
            confidence_range=(0.75, 0.99),
            config=dept_config or DepartmentConfig(name="verification", domain="verification")
        )

        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.consistency_threshold = consistency_threshold

        # Verification history
        self._verification_history: list[dict[str, Any]] = []

        # Inconsistency tracking
        self._detected_inconsistencies: list[InconsistencyReport] = []

        logger.info("Verification Department initialized")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize verification department."""
        if self._initialized:
            return

        # Load verification history if persisted
        # (Future: Load from disk)

        await super().initialize()

    async def close(self) -> None:
        """Cleanup resources."""
        # Save verification history
        # (Future: Persist to disk)

        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """Execute a verification task."""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        try:
            if request.task_type == "verify_response":
                result, confidence = await self._verify_response(request)
            elif request.task_type == "cross_check":
                result, confidence = await self._cross_check(request)
            elif request.task_type == "validate_confidence":
                result, confidence = await self._validate_confidence(request)
            elif request.task_type == "detect_inconsistencies":
                result, confidence = await self._detect_inconsistencies(request)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score
            }

            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            self._record_request(
                success=True,
                latency_ms=response.execution_time_ms,
                confidence=confidence.score
            )

            return response

        except Exception as e:
            self._record_error(e)

            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """Verify verification operation quality."""
        confidence_valid = response.confidence.score >= 0.75
        has_error = "error" in response.result
        operation_success = not has_error

        sufficient = confidence_valid and operation_success

        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {
                "retry": True,
                "reason": "Low confidence or error in verification"
            }

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=operation_success,
            alternative_paths=["retry"] if not sufficient else [],
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.5,
            escalation_reason="Very low confidence" if response.confidence.score < 0.5 else None,
            verifier_confidence=0.95
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Refine verification operation."""
        # For verification, refinement means re-checking with stricter criteria
        return await self.execute(request)

    # ========================================================================
    # Task Handlers
    # ========================================================================

    async def _verify_response(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Verify a single department response."""
        response_data = request.parameters.get("response")

        if not response_data:
            raise ValueError("Missing required parameter: 'response'")

        # Extract response details
        confidence_score = response_data.get("confidence", {}).get("score", 0.0)
        result = response_data.get("result", {})

        # Verification checks
        checks = {
            "confidence_acceptable": confidence_score >= self.confidence_threshold,
            "has_result": bool(result),
            "no_errors": "error" not in result,
            "confidence_calibrated": 0.0 <= confidence_score <= 1.0
        }

        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks)
        verification_score = passed_checks / total_checks

        confidence = ConfidenceMetadata.from_score(
            verification_score,
            justification=[f"Passed {passed_checks}/{total_checks} checks"],
            uncertainty_sources=[
                k for k, v in checks.items() if not v
            ]
        )

        result_data = {
            "checks": checks,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "verification_score": verification_score,
            "verified": verification_score >= 0.75
        }

        # Record verification
        self._verification_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": response_data,
            "result": result_data
        })

        return result_data, confidence

    async def _cross_check(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Cross-check across multiple departments."""
        query = request.parameters.get("query")
        departments = request.parameters.get("departments", [])

        if not query:
            raise ValueError("Missing required parameter: 'query'")

        if len(departments) < 2:
            raise ValueError("Need at least 2 departments for cross-checking")

        if not self.registry:
            raise RuntimeError("Registry required for cross-checking")

        # Query each department
        responses = {}
        for dept_name in departments:
            try:
                dept_request = DepartmentRequest(
                    task_id=f"cross_check_{dept_name}",
                    task_type="weave_response",  # Default task
                    parameters={"query": query}
                )

                response = await self.registry.route_request(
                    dept_request,
                    department_name=dept_name
                )
                responses[dept_name] = response

            except Exception as e:
                logger.warning(f"Failed to query {dept_name}: {e}")
                responses[dept_name] = None

        # Analyze agreement
        valid_responses = {k: v for k, v in responses.items() if v is not None}

        if len(valid_responses) < 2:
            confidence = ConfidenceMetadata.from_score(
                0.3,
                justification=[],
                uncertainty_sources=["Too few valid responses for cross-check"]
            )

            return {
                "responses": responses,
                "agreement": "insufficient_data",
                "confidence_agreement": 0.0
            }, confidence

        # Calculate confidence agreement
        confidences = [r.confidence.score for r in valid_responses.values()]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        agreement_level = "full"
        if confidence_variance > 0.1:
            agreement_level = "partial"
        if confidence_variance > 0.25:
            agreement_level = "conflict"

        # Detect conflicts
        conflicting_claims = []
        # (Simplified: Would analyze response content for conflicts)

        cross_check_result = CrossCheckResult(
            consistent=agreement_level == "full",
            confidence_agreement=1.0 - confidence_variance,
            conflicting_claims=conflicting_claims,
            agreement_level=agreement_level,
            evidence={dept: r.result for dept, r in valid_responses.items()}
        )

        confidence = ConfidenceMetadata.from_score(
            cross_check_result.confidence_agreement,
            justification=[f"Agreement level: {agreement_level}"],
            uncertainty_sources=[] if cross_check_result.consistent else ["Detected conflicts"]
        )

        result_data = {
            "responses": {k: {"confidence": v.confidence.score if v else 0.0} for k, v in responses.items()},
            "agreement_level": agreement_level,
            "confidence_agreement": cross_check_result.confidence_agreement,
            "consistent": cross_check_result.consistent,
            "conflicting_claims": conflicting_claims
        }

        return result_data, confidence

    async def _validate_confidence(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Validate confidence calibration."""
        responses = request.parameters.get("responses", [])

        if not responses:
            raise ValueError("Missing required parameter: 'responses'")

        # Analyze confidence scores
        confidences = [r.get("confidence", {}).get("score", 0.0) for r in responses]

        if not confidences:
            confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=["No confidence scores found"]
            )

            return {"valid": False, "reason": "No confidence scores"}, confidence

        # Calculate statistics
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        # Validation checks
        well_calibrated = variance < 0.15
        reasonable_range = 0.3 <= avg_confidence <= 0.95
        no_extreme_values = all(0.0 <= c <= 1.0 for c in confidences)

        validation_score = sum([well_calibrated, reasonable_range, no_extreme_values]) / 3.0

        confidence = ConfidenceMetadata.from_score(
            validation_score,
            justification=[f"Confidence scores well-calibrated: {well_calibrated}"],
            uncertainty_sources=[] if well_calibrated else ["High variance in confidence scores"]
        )

        result_data = {
            "valid": validation_score >= 0.75,
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "variance": variance,
            "well_calibrated": well_calibrated
        }

        return result_data, confidence

    async def _detect_inconsistencies(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Detect inconsistencies across departments."""
        responses = request.parameters.get("responses", {})

        if len(responses) < 2:
            raise ValueError("Need at least 2 responses to detect inconsistencies")

        inconsistencies = []

        # Check for confidence disagreement
        confidences = {dept: r.get("confidence", {}).get("score", 0.0) for dept, r in responses.items()}
        conf_values = list(confidences.values())

        if len(conf_values) >= 2:
            conf_range = max(conf_values) - min(conf_values)

            if conf_range > 0.3:
                inconsistencies.append(InconsistencyReport(
                    inconsistency_type="confidence_disagreement",
                    severity="medium",
                    departments_involved=list(responses.keys()),
                    details=f"Confidence range: {conf_range:.2f}",
                    recommended_action="Review responses for conflicts"
                ))

        # Store detected inconsistencies
        self._detected_inconsistencies.extend(inconsistencies)

        confidence = ConfidenceMetadata.from_score(
            0.85 if len(inconsistencies) == 0 else 0.65,
            justification=[f"Detected {len(inconsistencies)} inconsistencies"],
            uncertainty_sources=[]
        )

        result_data = {
            "inconsistencies": [
                {
                    "type": inc.inconsistency_type,
                    "severity": inc.severity,
                    "departments": inc.departments_involved,
                    "details": inc.details,
                    "action": inc.recommended_action
                }
                for inc in inconsistencies
            ],
            "inconsistency_count": len(inconsistencies),
            "severity_breakdown": {
                "low": sum(1 for inc in inconsistencies if inc.severity == "low"),
                "medium": sum(1 for inc in inconsistencies if inc.severity == "medium"),
                "high": sum(1 for inc in inconsistencies if inc.severity == "high")
            }
        }

        return result_data, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_reasoning(
        self,
        result: dict[str, Any],
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "task_type": request.task_type,
            "verification_history_count": len(self._verification_history),
            "inconsistencies_detected": len(self._detected_inconsistencies)
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_verification": result
        }

    def get_verification_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent verification history."""
        return self._verification_history[-limit:]

    def get_inconsistencies(self, severity: str | None = None) -> list[InconsistencyReport]:
        """Get detected inconsistencies."""
        if severity:
            return [inc for inc in self._detected_inconsistencies if inc.severity == severity]
        return self._detected_inconsistencies
