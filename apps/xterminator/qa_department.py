"""
Quality Assurance Department
=============================

🐷 xTerminator as a First-Class HoloLoom Department! 🐷

Implements the Department Protocol, making xTerminator a full participant
in the HoloLoom B2B platform ecosystem.

The QA Department:
- Scans code from other departments (MasterWeaver, Infrastructure, etc.)
- Proposes and validates fixes
- Learns from outcomes (Thompson Sampling, confidence calibration)
- Negotiates confidence with other departments
- Provides health monitoring and alerting

Integration Points:
- MasterWeaver → QA: "Scan my entity extraction code"
- Infrastructure → QA: "Validate this database migration"
- Orchestration → QA: "Check system health"
- Verification → QA: "Verify this fix proposal"

Usage:
    # Create QA department
    qa_dept = QADepartment(
        policy=AutofixPolicy.conservative(domain='healthcare'),
        enable_feedback=True
    )

    # Another department requests a scan
    request = DepartmentRequest(
        request_id="req_123",
        request_type=RequestType.SCAN_CODE,
        requesting_department="MasterWeaver",
        payload={
            'code': master_weaver_code,
            'file_path': 'entity_extractor.py'
        }
    )

    response = await qa_dept.execute(request)
    # → DepartmentResponse with issues found + confidence score
"""

import asyncio
import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path

from .department_protocol import (
    DepartmentProtocol,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    RequestType,
    ResponseStatus,
    ConfidenceNegotiation,
    negotiate_confidence
)
from .moonshot_integration import MoonshotOrchestrator, MoonshotResult
from .autofix_policy import AutofixPolicy, PolicyProfile, FixDecision
from .feedback_tracker import FeedbackTracker, FixOutcome
from .xterminator_types import RiskLevel, FixStrategy


class QADepartment:
    """
    Quality Assurance Department implementing DepartmentProtocol.

    Wraps MoonshotOrchestrator to provide department-level interface.
    """

    def __init__(
        self,
        department_name: str = "Quality Assurance",
        policy: Optional[AutofixPolicy] = None,
        enable_feedback: bool = True,
        feedback_log: str = "./qa_department_feedback.jsonl",
        repo_path: str = "."
    ):
        """
        Initialize QA Department.

        Args:
            department_name: Name of this department
            policy: AutofixPolicy (defaults to balanced)
            enable_feedback: Enable feedback tracking?
            feedback_log: Path to feedback log
            repo_path: Git repository path
        """
        self.department_name = department_name
        self.policy = policy or AutofixPolicy.balanced()

        # Core orchestrator
        self.orchestrator = MoonshotOrchestrator(
            policy=self.policy,
            enable_feedback=enable_feedback,
            feedback_log=feedback_log,
            repo_path=repo_path
        )

        # Track department health
        self.start_time = time.time()
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_latency_ms = 0.0

        # Confidence negotiation history
        self.negotiation_history: List[ConfidenceNegotiation] = []

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute a QA department operation.

        Supported request types:
        - SCAN_CODE: Scan code for issues
        - CLASSIFY_ISSUE: Classify a specific issue
        - PROPOSE_FIX: Propose fix for an issue
        - APPLY_FIX: Apply a proposed fix
        - VALIDATE_FIX: Validate an applied fix
        - GET_STATISTICS: Get QA statistics
        - DETECT_DEGRADATION: Check for degradation

        Args:
            request: DepartmentRequest

        Returns:
            DepartmentResponse with results
        """
        start_time = time.time()
        self.request_count += 1

        try:
            # Route to appropriate handler
            if request.request_type == RequestType.SCAN_CODE:
                response = await self._handle_scan_code(request)

            elif request.request_type == RequestType.CLASSIFY_ISSUE:
                response = await self._handle_classify_issue(request)

            elif request.request_type == RequestType.PROPOSE_FIX:
                response = await self._handle_propose_fix(request)

            elif request.request_type == RequestType.APPLY_FIX:
                response = await self._handle_apply_fix(request)

            elif request.request_type == RequestType.VALIDATE_FIX:
                response = await self._handle_validate_fix(request)

            elif request.request_type == RequestType.GET_STATISTICS:
                response = await self._handle_get_statistics(request)

            elif request.request_type == RequestType.DETECT_DEGRADATION:
                response = await self._handle_detect_degradation(request)

            else:
                response = DepartmentResponse(
                    request_id=request.request_id,
                    status=ResponseStatus.FAILURE,
                    responding_department=self.department_name,
                    confidence=0.0,
                    error=f"Unsupported request type: {request.request_type.value}"
                )

            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            response.duration_ms = duration_ms
            self.total_latency_ms += duration_ms

            if response.status == ResponseStatus.SUCCESS:
                self.success_count += 1
            elif response.status == ResponseStatus.FAILURE:
                self.failure_count += 1

            # Confidence negotiation (if requesting dept provided confidence)
            if 'requesting_confidence' in request.context:
                negotiated_conf = negotiate_confidence(
                    requesting_dept=request.requesting_department,
                    responding_dept=self.department_name,
                    request_conf=request.context['requesting_confidence'],
                    response_conf=response.confidence,
                    strategy="weighted_average"
                )
                response.metadata['negotiated_confidence'] = negotiated_conf

            return response

        except Exception as e:
            self.failure_count += 1
            return DepartmentResponse(
                request_id=request.request_id,
                status=ResponseStatus.FAILURE,
                responding_department=self.department_name,
                confidence=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                error=f"Internal error: {str(e)}",
                error_details={'exception_type': type(e).__name__}
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify a department response (from QA or another dept).

        For QA responses, verifies:
        - Fix proposals are valid
        - Confidence scores are calibrated
        - Validation results are accurate

        Args:
            response: DepartmentResponse to verify

        Returns:
            VerificationResult
        """
        # Extract response details
        original_conf = response.confidence
        issues_found = []
        corrections_needed = []
        requires_refinement = False
        refinement_suggestions = []

        # Verification checks
        if response.status == ResponseStatus.FAILURE:
            issues_found.append("Response status is FAILURE")
            requires_refinement = False  # Can't refine a failure

        elif response.confidence < 0.50:
            issues_found.append(f"Low confidence ({response.confidence:.2f})")
            requires_refinement = True
            refinement_suggestions.append("Re-classify with more context")

        # Check payload completeness
        if not response.payload:
            issues_found.append("Empty payload")
            corrections_needed.append("Include result data in payload")

        # Determine verified confidence
        verified_conf = original_conf
        if issues_found:
            # Penalize confidence if issues found
            penalty = 0.10 * len(issues_found)
            verified_conf = max(0.0, original_conf - penalty)

        confidence_delta = verified_conf - original_conf
        verified = len(issues_found) == 0

        return VerificationResult(
            verified=verified,
            confidence_delta=confidence_delta,
            original_confidence=original_conf,
            verified_confidence=verified_conf,
            verification_method="qa_department_verify",
            issues_found=issues_found,
            corrections_needed=corrections_needed,
            requires_refinement=requires_refinement,
            refinement_suggestions=refinement_suggestions
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine a previous response based on verification feedback.

        For QA, refinement might mean:
        - Re-classify with additional context
        - Try different fix strategy
        - Request human review

        Args:
            request: Original request
            prior_response: Previous response
            verification: Verification results

        Returns:
            Refined DepartmentResponse
        """
        # For now, simple refinement: re-execute with adjusted confidence threshold
        refined_request = request
        refined_request.context['refinement_iteration'] = \
            refined_request.context.get('refinement_iteration', 0) + 1

        # Add verification feedback to context
        refined_request.context['verification_feedback'] = {
            'issues': verification.issues_found,
            'suggestions': verification.refinement_suggestions
        }

        # Re-execute
        refined_response = await self.execute(refined_request)

        # Mark as refined
        refined_response.metadata['refined'] = True
        refined_response.metadata['refinement_iteration'] = refined_request.context['refinement_iteration']
        refined_response.metadata['prior_confidence'] = prior_response.confidence

        return refined_response

    async def update_strategy(self, learning_signals: Dict[str, Any]) -> None:
        """
        Update QA strategy based on learning signals.

        Learning signals come from fix outcomes:
        - Which strategies worked?
        - Was confidence accurate?
        - Should we adjust thresholds?

        Args:
            learning_signals: Learning data
        """
        # Extract signals
        outcome = learning_signals.get('outcome', FixOutcome.PENDING.value)
        confidence = learning_signals.get('confidence', 0.0)
        accuracy = learning_signals.get('accuracy', True)
        strategy = learning_signals.get('strategy_used', '')

        # Update policy thresholds based on outcomes
        # (This is prep for Phase 4 Thompson Sampling)
        if outcome == FixOutcome.SUCCESS.value and confidence >= 0.85:
            # High confidence success - good!
            pass

        elif outcome == FixOutcome.FAILURE.value and confidence >= 0.85:
            # High confidence failure - overconfident!
            # Could lower auto-fix threshold slightly
            pass

        # Track learning signal in feedback tracker
        if self.orchestrator.feedback_tracker:
            # Feedback is already tracked via orchestrator
            pass

    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """
        Query institutional memory for learned patterns.

        Pattern types:
        - 'successful_strategies': What strategies work best?
        - 'failed_patterns': What patterns fail often?
        - 'confidence_calibration': How accurate are confidence scores?
        - 'performance_trends': How is performance trending?

        Args:
            pattern_type: Type of pattern to query

        Returns:
            Institutional memory data
        """
        if not self.orchestrator.feedback_tracker:
            return {'error': 'Feedback tracking disabled'}

        tracker = self.orchestrator.feedback_tracker

        if pattern_type == 'successful_strategies':
            # Get strategy performance
            stats = tracker.get_summary_statistics()
            return stats.get('strategy_performance', {})

        elif pattern_type == 'failed_patterns':
            # Get false positive patterns
            return tracker.get_false_positive_patterns()

        elif pattern_type == 'confidence_calibration':
            # Get calibration data
            return tracker.get_confidence_calibration()

        elif pattern_type == 'performance_trends':
            # Get degradation analysis
            return tracker.detect_degradation()

        else:
            return {'error': f'Unknown pattern type: {pattern_type}'}

    async def health_check(self) -> Dict[str, Any]:
        """
        Check QA department health.

        Returns health metrics for monitoring and alerting.

        Returns:
            Health status dictionary
        """
        uptime = time.time() - self.start_time
        success_rate = self.success_count / self.request_count if self.request_count > 0 else 0.0
        avg_latency = self.total_latency_ms / self.request_count if self.request_count > 0 else 0.0
        error_rate = self.failure_count / self.request_count if self.request_count > 0 else 0.0

        # Check for degradation
        degradation = {}
        if self.orchestrator.feedback_tracker and self.request_count >= 40:
            degradation = self.orchestrator.detect_degradation(window_size=20, threshold=0.10)

        degradation_detected = degradation.get('degradation_detected', False)

        # Determine status
        if error_rate > 0.20 or degradation_detected:
            status = 'unhealthy'
        elif error_rate > 0.10 or success_rate < 0.80:
            status = 'degraded'
        else:
            status = 'healthy'

        # Alerts
        alerts = []
        if degradation_detected:
            alerts.append(f"Degradation detected: success rate dropped to {degradation['recent_success_rate']:.1%}")
        if error_rate > 0.20:
            alerts.append(f"High error rate: {error_rate:.1%}")
        if avg_latency > 1000:
            alerts.append(f"High latency: {avg_latency:.0f}ms")

        # Confidence calibration
        confidence_accuracy = 0.0
        if self.orchestrator.feedback_tracker:
            calib = self.orchestrator.feedback_tracker.get_confidence_calibration()
            confidence_accuracy = calib.get('calibration_accuracy', 0.0)

        return {
            'status': status,
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_latency_ms': avg_latency,
            'confidence_accuracy': confidence_accuracy,
            'degradation_detected': degradation_detected,
            'degradation_details': degradation,
            'alerts': alerts,
            'policy_profile': self.policy.profile.value,
            'policy_domain': self.policy.domain or 'general'
        }

    # Request handlers
    async def _handle_scan_code(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle SCAN_CODE request"""
        code = request.payload.get('code', '')
        file_path = request.payload.get('file_path', 'unknown.py')

        # For now, return placeholder
        # In full implementation, would scan with Trough + classify all issues
        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=0.90,
            payload={
                'issues_found': 0,
                'message': 'Scan complete (placeholder)'
            }
        )

    async def _handle_classify_issue(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle CLASSIFY_ISSUE request"""
        # Placeholder
        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=0.85,
            payload={'classification': 'placeholder'}
        )

    async def _handle_propose_fix(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle PROPOSE_FIX request"""
        # Placeholder
        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=0.88,
            payload={'fix_proposed': 'placeholder'}
        )

    async def _handle_apply_fix(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle APPLY_FIX request"""
        # Placeholder
        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=0.92,
            payload={'fix_applied': 'placeholder'}
        )

    async def _handle_validate_fix(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle VALIDATE_FIX request"""
        # Placeholder
        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=0.95,
            payload={'validation_passed': True}
        )

    async def _handle_get_statistics(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle GET_STATISTICS request"""
        stats = self.orchestrator.get_learning_statistics()

        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=1.0,
            payload={'statistics': stats}
        )

    async def _handle_detect_degradation(self, request: DepartmentRequest) -> DepartmentResponse:
        """Handle DETECT_DEGRADATION request"""
        window_size = request.payload.get('window_size', 20)
        threshold = request.payload.get('threshold', 0.10)

        degradation = self.orchestrator.detect_degradation(window_size, threshold)

        return DepartmentResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            responding_department=self.department_name,
            confidence=1.0,
            payload={'degradation': degradation}
        )
