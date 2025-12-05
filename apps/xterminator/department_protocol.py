"""
Department Protocol
===================

🐷 Phase 2: xTerminator as a HoloLoom Department 🐷

Implements the standard Department Protocol that enables xTerminator to integrate
with the HoloLoom B2B platform as a first-class Quality Assurance Department.

The Protocol:
- execute(request) -> DepartmentResponse
- verify(response) -> VerificationResult
- refine(request, prior, verification) -> DepartmentResponse
- update_strategy(learning_signals) -> None
- get_institutional_memory(pattern_type) -> Dict
- health_check() -> Dict

This enables:
- Cross-department collaboration (MasterWeaver → QA → Infrastructure)
- Confidence negotiation (departments trust each other's scores)
- DS-STAR verification loops (verify → refine → re-verify)
- Institutional learning (departments improve together)
- Live monitoring (health checks, degradation alerts)

Moonshot Integration:
- Enables cross-department quality feedback (Idea #1)
- Provides confidence authority (Idea #4)
- Enables live monitoring (Idea #7)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum
import time


class RequestType(Enum):
    """Types of requests a department can handle"""
    SCAN_CODE = "scan_code"              # Scan code for quality issues
    CLASSIFY_ISSUE = "classify_issue"    # Classify a specific issue
    PROPOSE_FIX = "propose_fix"          # Propose a fix for an issue
    APPLY_FIX = "apply_fix"              # Apply a proposed fix
    VALIDATE_FIX = "validate_fix"        # Validate an applied fix
    ROLLBACK_FIX = "rollback_fix"        # Rollback a bad fix
    GET_STATISTICS = "get_statistics"    # Get QA statistics
    DETECT_DEGRADATION = "detect_degradation"  # Check for degradation


class ResponseStatus(Enum):
    """Status of a department response"""
    SUCCESS = "success"          # Request completed successfully
    PARTIAL = "partial"          # Request partially completed
    FAILURE = "failure"          # Request failed
    PENDING = "pending"          # Request pending (async)
    REQUIRES_REVIEW = "requires_review"  # Needs human review
    REQUIRES_REFINEMENT = "requires_refinement"  # Needs refinement


@dataclass
class DepartmentRequest:
    """
    Standard request format for department operations.

    All departments accept DepartmentRequest and return DepartmentResponse.
    """
    request_id: str
    request_type: RequestType
    requesting_department: str  # Which department is asking?
    priority: int = 5           # 1 (highest) to 10 (lowest)
    timeout_ms: float = 5000.0  # Max time to process

    # Request payload (varies by request_type)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DepartmentResponse:
    """
    Standard response format from department operations.

    Contains results, confidence score, and metadata.
    """
    request_id: str
    status: ResponseStatus
    responding_department: str
    confidence: float = 0.0     # 0.0-1.0 confidence in this response

    # Response payload (varies by request_type)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error details (if status = FAILURE)
    error: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """
    Result of verifying a department's response.

    Used in DS-STAR verification loops.
    """
    verified: bool              # Did verification pass?
    confidence_delta: float     # Change in confidence (-1.0 to +1.0)
    original_confidence: float
    verified_confidence: float

    # Verification details
    verification_method: str    # How was it verified?
    issues_found: List[str] = field(default_factory=list)
    corrections_needed: List[str] = field(default_factory=list)

    # Should we refine?
    requires_refinement: bool = False
    refinement_suggestions: List[str] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceNegotiation:
    """
    Confidence negotiation between departments.

    When departments collaborate, they negotiate trust in each other's outputs.
    """
    requesting_department: str
    responding_department: str
    request_confidence: float    # How confident is requester in their request?
    response_confidence: float   # How confident is responder in their response?
    negotiated_confidence: float # Final agreed confidence

    # Negotiation strategy
    strategy: str = "weighted_average"  # How to combine confidences
    weights: Dict[str, float] = field(default_factory=dict)

    # Trust tracking
    historical_accuracy: float = 0.85  # How often has this dept been right?
    trust_factor: float = 1.0          # Multiplier based on history

    def calculate_negotiated_confidence(self) -> float:
        """
        Calculate final confidence based on negotiation strategy.

        Strategies:
        - weighted_average: Weighted average of request + response confidences
        - minimum: Take the minimum (pessimistic)
        - maximum: Take the maximum (optimistic)
        - trust_weighted: Weight by historical accuracy
        """
        if self.strategy == "minimum":
            return min(self.request_confidence, self.response_confidence)

        elif self.strategy == "maximum":
            return max(self.request_confidence, self.response_confidence)

        elif self.strategy == "trust_weighted":
            # Weight response by historical accuracy
            weighted_response = self.response_confidence * self.historical_accuracy
            return (self.request_confidence + weighted_response) / 2.0

        else:  # weighted_average (default)
            req_weight = self.weights.get('request', 0.3)
            resp_weight = self.weights.get('response', 0.7)
            return (self.request_confidence * req_weight +
                    self.response_confidence * resp_weight)


class DepartmentProtocol(Protocol):
    """
    Protocol that all HoloLoom departments must implement.

    This enables cross-department collaboration, confidence negotiation,
    and institutional learning.
    """

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute a department operation.

        Args:
            request: DepartmentRequest with operation details

        Returns:
            DepartmentResponse with results and confidence score
        """
        ...

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify a response from this or another department.

        Used in DS-STAR verification loops:
        1. Department produces response
        2. Verification checks quality
        3. If issues found, refine() is called
        4. Loop until verified or max iterations

        Args:
            response: DepartmentResponse to verify

        Returns:
            VerificationResult with pass/fail and refinement suggestions
        """
        ...

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine a previous response based on verification feedback.

        This is called when verify() indicates refinement is needed.

        Args:
            request: Original request
            prior_response: Previous response that failed verification
            verification: Verification results with suggestions

        Returns:
            Refined DepartmentResponse
        """
        ...

    async def update_strategy(self, learning_signals: Dict[str, Any]) -> None:
        """
        Update department strategy based on learning signals.

        This enables institutional learning - the department improves
        based on outcomes from execute/verify/refine cycles.

        Args:
            learning_signals: Dictionary with learning data
                - 'outcome': SUCCESS/FAILURE/ROLLBACK
                - 'confidence': Original confidence score
                - 'accuracy': Was confidence accurate?
                - 'strategy_used': Which strategy was used
                - 'category': Issue category
                - etc.
        """
        ...

    async def get_institutional_memory(self, pattern_type: str) -> Dict[str, Any]:
        """
        Query institutional memory for learned patterns.

        Departments remember what works and what doesn't.

        Args:
            pattern_type: Type of pattern to query
                - 'successful_strategies': What strategies work best?
                - 'failed_patterns': What patterns fail often?
                - 'confidence_calibration': How accurate are confidence scores?
                - 'performance_trends': How is performance trending?

        Returns:
            Dictionary with institutional memory data
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Check department health status.

        Used for live monitoring and alerting.

        Returns:
            Dictionary with health metrics:
                - 'status': 'healthy' | 'degraded' | 'unhealthy'
                - 'uptime': Uptime in seconds
                - 'success_rate': Recent success rate (0.0-1.0)
                - 'avg_latency_ms': Average response time
                - 'error_rate': Recent error rate
                - 'confidence_accuracy': Confidence calibration accuracy
                - 'degradation_detected': True if degradation detected
                - 'alerts': List of active alerts
        """
        ...


# Convenience functions for confidence negotiation
def negotiate_confidence(
    requesting_dept: str,
    responding_dept: str,
    request_conf: float,
    response_conf: float,
    strategy: str = "weighted_average",
    historical_accuracy: float = 0.85
) -> float:
    """
    Negotiate final confidence between departments.

    Args:
        requesting_dept: Name of requesting department
        responding_dept: Name of responding department
        request_conf: Requesting department's confidence
        response_conf: Responding department's confidence
        strategy: Negotiation strategy
        historical_accuracy: Historical accuracy of responding dept

    Returns:
        Negotiated confidence score
    """
    negotiation = ConfidenceNegotiation(
        requesting_department=requesting_dept,
        responding_department=responding_dept,
        request_confidence=request_conf,
        response_confidence=response_conf,
        negotiated_confidence=0.0,
        strategy=strategy,
        historical_accuracy=historical_accuracy
    )

    negotiation.negotiated_confidence = negotiation.calculate_negotiated_confidence()
    return negotiation.negotiated_confidence
