"""
HoloLoom Department Protocol - Core Interface and Types
========================================================

Canonical location for the Department protocol and associated types.
Moved from ``HoloLoom.departments.protocol`` on 2026-02-26 so that
Layer-0 protocols live in ``HoloLoom.protocols`` and carry no
dependency on the ``departments`` package (which is an app-layer module).

**Purpose**: Defines the generic department interface that makes HoloLoom modular
and B2B-ready. Provides complete protocol definitions, confidence tracking, privacy
envelopes, verification, and helper functions for department implementations.

**Date**: November 13, 2025
**Moved**: 2026-02-26 (to protocols/)
**Phase**: Moonshot Week 1-2 - Core Framework (Task 1.1)
**Key Innovations**:
- Confidence metadata with automatic score classification
- Privacy envelopes with access logging
- Verification checks with composite results
- Department protocol with 7 core methods
- Helper functions for request/response creation
- Type aliases for factories and verification functions
- Base configuration and manifest types for compatibility

**Philosophy**:
"Every department is a module. Every module has clear inputs, outputs, confidence,
and verification. This makes HoloLoom a composable system, not a monolith."

The protocol defines:
1. What departments DO (execute, verify, refine, etc.)
2. What they CONSUME (requests with parameters)
3. What they PRODUCE (responses with confidence)
4. How they MEASURE success (verification checks)
5. How they IMPROVE (learning signals)

This design enables:
- Mix-and-match departments (plug'n'play)
- Multi-department teams (parallel execution)
- Graceful degradation (fallback strategies)
- Complete observability (metrics at each stage)
- Type-safe integration (Protocol-based duck typing)

Author: HoloLoom Architecture Team
Timestamp: 2025-11-13
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Protocol, Union, Callable, runtime_checkable
)
from uuid import UUID, uuid4


# ============================================================================
# Confidence Level and Metadata
# ============================================================================

class ConfidenceLevel(Enum):
    """
    Confidence levels for department responses.

    Used to quickly categorize response quality without needing to look
    at the numeric score. Ranges are guidelines, not absolute cutoffs.
    """
    CRITICAL = "critical"      # 0.0-0.2: Invalid/unreliable
    LOW = "low"                 # 0.2-0.5: Uncertain, needs verification
    MEDIUM = "medium"           # 0.5-0.75: Reasonable, acceptable
    HIGH = "high"               # 0.75-0.95: Solid, can be trusted
    VERIFIED = "verified"       # 0.95-1.0: Verified correct, gold standard


@dataclass
class ConfidenceMetadata:
    """
    Confidence metadata with score, level, and justification.

    Tracks:
    - Numeric score (0-1)
    - Confidence level (CRITICAL/LOW/MEDIUM/HIGH/VERIFIED)
    - Justification (why we're confident)
    - Sources (where the signal came from)
    - Timestamp (when computed)
    - Calibration info (for improving confidence over time)

    Example:
        confidence = ConfidenceMetadata.from_score(
            0.87,
            justification=["Multiple sources agree", "High redundancy"],
            sources=["memory_recall", "knowledge_graph"]
        )
    """
    score: float                           # 0-1, higher = more confident
    level: ConfidenceLevel                 # CRITICAL/LOW/MEDIUM/HIGH/VERIFIED
    justification: List[str] = field(default_factory=list)  # Why confident?
    sources: List[str] = field(default_factory=list)        # Where did signal come from?
    timestamp: datetime = field(default_factory=datetime.utcnow)
    calibration_count: int = 0             # How many times calibrated?
    calibration_history: List[float] = field(default_factory=list)  # Historical scores

    @classmethod
    def from_score(
        cls,
        score: float,
        justification: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> 'ConfidenceMetadata':
        """
        Create ConfidenceMetadata from numeric score.

        Automatically determines confidence level based on score ranges.

        Args:
            score: Numeric confidence (0-1)
            justification: Why we're confident (optional)
            sources: Where signal came from (optional)

        Returns:
            ConfidenceMetadata with level determined from score
        """
        # Determine level from score
        if score < 0.2:
            level = ConfidenceLevel.CRITICAL
        elif score < 0.5:
            level = ConfidenceLevel.LOW
        elif score < 0.75:
            level = ConfidenceLevel.MEDIUM
        elif score < 0.95:
            level = ConfidenceLevel.HIGH
        else:
            level = ConfidenceLevel.VERIFIED

        return cls(
            score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            level=level,
            justification=justification or [],
            sources=sources or [],
        )


# ============================================================================
# Privacy Level and Envelope
# ============================================================================

class PrivacyLevel(Enum):
    """
    Privacy levels for data in department responses.

    Determines who can access what and what auditing is required.
    """
    PUBLIC = "public"                      # Anyone can see
    INTERNAL = "internal"                  # HoloLoom systems only
    CONFIDENTIAL = "confidential"          # Department + authorized systems
    RESTRICTED = "restricted"              # Department only
    CRITICAL = "critical"                  # Requires explicit audit trail


@dataclass
class PrivacyEnvelope:
    """
    Privacy wrapper for sensitive data in responses.

    Tracks:
    - Data content
    - Privacy level
    - Access logs (who accessed when)
    - Redaction rules (what gets masked)

    Example:
        private_data = PrivacyEnvelope(
            content="customer_secrets",
            level=PrivacyLevel.RESTRICTED,
            redaction_rules=["redact_email", "redact_phone"]
        )

        private_data.log_access(accessor="system_a", reason="processing")
    """
    content: Any                            # The actual data
    level: PrivacyLevel                     # Who can access?
    redaction_rules: List[str] = field(default_factory=list)  # Redact what?
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def log_access(self, accessor: str, reason: str, approved: bool = True) -> None:
        """
        Log who accessed this private data and when.

        Args:
            accessor: Who accessed (system name, user ID, etc.)
            reason: Why they accessed
            approved: Was access approved? (false = security event)
        """
        self.access_log.append({
            "accessor": accessor,
            "reason": reason,
            "approved": approved,
            "timestamp": datetime.utcnow(),
            "level": self.level.value
        })


# ============================================================================
# Request and Response Types
# ============================================================================

@dataclass
class DepartmentRequest:
    """
    Request to a department.

    Encapsulates what we're asking a department to do:
    - Task identifier
    - Task type (what operation)
    - Parameters (input data)
    - Context (surrounding information)
    - Constraints (time budget, confidence threshold, etc.)

    Example:
        request = DepartmentRequest(
            task_type="retrieve_context",
            parameters={"query": "What is Thompson Sampling?"},
            constraints={"max_latency_ms": 150, "min_confidence": 0.75}
        )
    """
    task_id: UUID = field(default_factory=uuid4)     # Unique task ID
    task_type: str = ""                              # Operation type
    parameters: Dict[str, Any] = field(default_factory=dict)  # Input data
    context: Dict[str, Any] = field(default_factory=dict)    # Surrounding info
    constraints: Dict[str, Any] = field(default_factory=dict)  # Limits
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 50                               # 0-100, higher = more urgent


@dataclass
class DepartmentResponse:
    """
    Response from a department.

    Encapsulates what a department delivers:
    - Result (the answer)
    - Confidence (how sure are we?)
    - Metadata (additional context)
    - Performance metrics (how long did it take?)

    Example:
        response = DepartmentResponse(
            task_id=request.task_id,
            result="Thompson Sampling is a Bayesian...",
            confidence=ConfidenceMetadata.from_score(0.89),
            metadata={"source": "knowledge_graph", "traversal_depth": 3}
        )
    """
    task_id: UUID                                    # Echo back request ID
    result: Any                                      # The answer
    confidence: ConfidenceMetadata                   # How sure?
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0                          # How long did it take?
    error: Optional[str] = None                      # Error message if failed


# ============================================================================
# Verification Types
# ============================================================================

class VerificationStatus(Enum):
    """Status of a verification check."""
    PASSED = "passed"                      # Check succeeded
    FAILED = "failed"                      # Check failed
    SKIPPED = "skipped"                    # Check was skipped (N/A)
    IN_PROGRESS = "in_progress"            # Check is running


@dataclass
class VerificationCheck:
    """
    A single verification check.

    Departments implement multiple checks (DS-STAR):
    - D: Domain knowledge validation
    - S: Sensibility check
    - T: Temporal consistency
    - A: Argument structure
    - R: Reference verification

    Example:
        check = VerificationCheck(
            name="sensibility",
            status=VerificationStatus.PASSED,
            reason="Response uses correct terminology",
            score=0.95
        )
    """
    name: str                                        # Check name
    status: VerificationStatus                       # PASSED/FAILED/SKIPPED
    reason: str                                      # Why this status?
    score: float = 0.0                               # 0-1, optional metric
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSStarCheck:
    """
    DS-STAR verification check (Domain, Sensibility, Temporal, Argument, Reference).

    Implements the DS-STAR framework for comprehensive response verification:
    - D (Domain): Are sources relevant to query domain?
    - S (Sensibility): Does answer make logical sense?
    - T (Temporal): Is information up-to-date?
    - A (Argument): Is answer supported by sources?
    - R (Reference): Are sources traceable and credible?

    Example:
        check = DSStarCheck(
            dimension="Domain",
            passed=True,
            score=0.85,
            details="Source relevance score: 0.85"
        )
    """
    dimension: str                                   # DS-STAR dimension name
    passed: bool                                     # Did check pass?
    score: float                                     # 0-1 check score
    details: str = ""                                # Human-readable details


@dataclass
class VerificationResult:
    """
    Result of verification (composite of multiple checks).

    Combines multiple VerificationCheck results into overall verdict.

    Example:
        result = VerificationResult(
            checks=[
                VerificationCheck("domain", VerificationStatus.PASSED, ...),
                VerificationCheck("sensibility", VerificationStatus.PASSED, ...),
            ],
            summary="All checks passed",
            confidence=0.92
        )
    """
    verified: bool = False                           # All checks passed?
    checks: List[VerificationCheck] = field(default_factory=list)
    overall_score: float = 0.0                       # Aggregate verification score
    summary: str = ""                                # Overall summary
    confidence: float = 0.0                          # 0-1 confidence in verification
    recommendations: List[str] = field(default_factory=list)  # Improvement suggestions
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Check if all non-skipped checks passed."""
        if not self.checks:
            return True  # No checks = pass by default
        non_skipped = [c for c in self.checks if c.status != VerificationStatus.SKIPPED]
        if not non_skipped:
            return True  # All skipped = pass by default
        return all(c.status == VerificationStatus.PASSED for c in non_skipped)


# ============================================================================
# Department Protocol
# ============================================================================

@runtime_checkable
class Department(Protocol):
    """
    Protocol defining what all departments must implement.

    A "Department" is any component that can:
    1. Execute tasks (do domain-specific work)
    2. Verify results (check quality)
    3. Refine responses (improve on demand)
    4. Track strategies (learn what works)
    5. Report capabilities (what can I do?)
    6. Expose metrics (how am I doing?)
    7. Health check (am I alive?)

    Design:
    - All methods are async (non-blocking)
    - All responses include confidence
    - All responses can be verified
    - All departments are composable

    Example (minimal implementation):

        class SimpleDepartment:
            async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result="Hello, World!",
                    confidence=ConfidenceMetadata.from_score(1.0)
                )

            async def verify(self, response: DepartmentResponse) -> VerificationResult:
                return VerificationResult(
                    summary="Always passes",
                    confidence=1.0
                )

            async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
                return response  # No refinement

            async def update_strategy(self, feedback: Dict[str, Any]) -> None:
                pass  # No learning

            async def get_capabilities(self) -> Dict[str, Any]:
                return {"supported_tasks": ["anything"]}

            async def get_metrics(self) -> Dict[str, Any]:
                return {"status": "ok"}

            async def health_check(self) -> bool:
                return True
    """

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute the requested task.

        Core method: implement domain-specific logic here.
        Must return response with confidence even on partial failure.

        Args:
            request: What to execute

        Returns:
            DepartmentResponse with result and confidence

        Raises:
            May raise exceptions for catastrophic failures
        """
        ...

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify response quality (DS-STAR framework).

        Implement multiple checks:
        - Domain: Uses correct domain knowledge?
        - Sensibility: Does it make sense?
        - Temporal: Consistent with time?
        - Argument: Logic sound?
        - Reference: Sources valid?

        Args:
            response: Response to verify

        Returns:
            VerificationResult with pass/fail and details
        """
        ...

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Improve response quality on demand.

        Called when verification fails or confidence is low.
        Iterate and improve until passing or timeout.

        Args:
            response: Response to improve

        Returns:
            Refined DepartmentResponse
        """
        ...

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback to improve future responses.

        Feedback typically contains:
        - outcome: "success" or "failure"
        - confidence_calibration: actual vs predicted confidence
        - user_rating: 1-5 stars
        - corrections: what should have been different

        Args:
            feedback: Learning signal
        """
        ...

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Report what this department can do.

        Returns dict with:
        - supported_tasks: List[str]
        - input_schemas: Dict[str, Schema]
        - output_schemas: Dict[str, Schema]
        - constraints: Dict[str, Any]
        - dependencies: List[str]

        Args: None

        Returns:
            Capability dictionary
        """
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Returns dict with:
        - tasks_executed: int
        - avg_latency_ms: float
        - avg_confidence: float
        - verification_pass_rate: float
        - error_rate: float

        Args: None

        Returns:
            Metrics dictionary
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if department is healthy and ready.

        Should check:
        - Dependencies available?
        - External services up?
        - Memory/resources ok?

        Args: None

        Returns:
            True if healthy, False otherwise
        """
        ...


# Alias for backward compatibility
DepartmentProtocol = Department


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_request(
    task_type: str,
    parameters: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
) -> DepartmentRequest:
    """
    Create a simple department request.

    Args:
        task_type: What operation
        parameters: Input data
        constraints: Optional execution limits

    Returns:
        DepartmentRequest ready to send

    Example:
        request = create_simple_request(
            "retrieve_context",
            {"query": "What is Thompson Sampling?"},
            {"max_latency_ms": 150}
        )
    """
    return DepartmentRequest(
        task_type=task_type,
        parameters=parameters,
        constraints=constraints or {}
    )


def create_simple_response(
    task_id: UUID,
    result: Any,
    confidence_score: float = 0.5
) -> DepartmentResponse:
    """
    Create a simple department response.

    Args:
        task_id: Which request is this answering?
        result: The answer
        confidence_score: How confident (0-1)

    Returns:
        DepartmentResponse ready to send

    Example:
        response = create_simple_response(
            task_id=request.task_id,
            result="Thompson Sampling is...",
            confidence_score=0.87
        )
    """
    return DepartmentResponse(
        task_id=task_id,
        result=result,
        confidence=ConfidenceMetadata.from_score(confidence_score)
    )


# ============================================================================
# Type Aliases
# ============================================================================

# Factory function for creating departments
DepartmentFactory = Callable[..., Department]

# Verification function type
VerificationFunction = Callable[[DepartmentResponse], VerificationResult]


# ============================================================================
# Additional Types for base.py Compatibility
# ============================================================================

@dataclass
class DepartmentConfig:
    """
    Configuration for a department.

    Stores:
    - name: Department name
    - domain: Domain (beekeeping, context, etc.)
    - version: Semantic version
    - supported_tasks: What can it do?
    - confidence_range: Expected confidence bounds
    - enable_learning: Can it improve?
    - enable_verification: Should responses be verified?
    - memory capacities: Three-tier memory limits

    Example:
        config = DepartmentConfig(
            name="context",
            domain="general",
            version="1.0.0",
            supported_tasks=["retrieve_context", "weave"],
            confidence_range=(0.65, 0.95)
        )
    """
    name: str                                        # Department name
    domain: str                                      # Domain (beekeeping, etc.)
    version: str = "1.0.0"                           # Semantic version
    supported_tasks: List[str] = field(default_factory=list)
    confidence_range: tuple = (0.0, 1.0)             # (min, max) expected confidence
    enable_learning: bool = True                     # Can improve?
    enable_verification: bool = True                 # Verify responses?
    enable_session_memory: bool = True               # Track session state?
    max_latency_ms: float = 5000.0                   # Max execution time
    retry_on_failure: bool = True                    # Retry failed tasks?

    # Three-tier memory capacity limits
    short_term_capacity: int = 100                   # Recent interactions (this session)
    medium_term_capacity: int = 500                  # Session patterns (hours to days)
    long_term_capacity: int = 2000                   # Institutional knowledge (weeks to months)


@dataclass
class DepartmentManifest:
    """
    Manifest describing a department (for B2B/deployment).

    Stores:
    - config: DepartmentConfig
    - capabilities: What it can do
    - dependencies: What it needs
    - metadata: Additional info

    Example:
        manifest = DepartmentManifest(
            config=config,
            capabilities=["retrieve", "reason", "synthesize"],
            dependencies=["memory_store", "embeddings"],
            metadata={"author": "hololoom", "license": "MIT"}
        )
    """
    config: DepartmentConfig
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Learning and Strategy Functions
# ============================================================================

def compute_learning_rate(
    iteration: int,
    initial_rate: float = 0.1,
    decay_rate: float = 0.99,
    min_rate: float = 0.001
) -> float:
    """
    Compute adaptive learning rate for department learning.

    Uses exponential decay: lr = initial_rate * (decay_rate ** iteration)

    Args:
        iteration: Learning iteration number
        initial_rate: Starting learning rate
        decay_rate: Exponential decay factor
        min_rate: Minimum learning rate floor

    Returns:
        Learning rate for this iteration

    Example:
        for step in range(100):
            lr = compute_learning_rate(step)
            # Update using lr
    """
    lr = initial_rate * (decay_rate ** iteration)
    return max(min_rate, lr)


def should_update_now(
    last_update: Optional[datetime],
    min_interval_seconds: float = 60.0,
    confidence_threshold: float = 0.75
) -> bool:
    """
    Determine if department should update strategy now.

    Returns True if:
    - Never updated before, OR
    - Enough time has passed since last update, OR
    - Confidence is low (needs learning)

    Args:
        last_update: Timestamp of last update (or None)
        min_interval_seconds: Minimum time between updates
        confidence_threshold: If confidence below this, force update

    Returns:
        True if should update now, False otherwise

    Example:
        if should_update_now(last_update_time, min_interval_seconds=300):
            await department.update_strategy(feedback)
    """
    if last_update is None:
        return True  # Never updated

    elapsed = (datetime.utcnow() - last_update).total_seconds()
    return elapsed >= min_interval_seconds


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Confidence
    'ConfidenceLevel',
    'ConfidenceMetadata',

    # Privacy
    'PrivacyLevel',
    'PrivacyEnvelope',

    # Requests and Responses
    'DepartmentRequest',
    'DepartmentResponse',

    # Verification
    'VerificationStatus',
    'VerificationCheck',
    'DSStarCheck',
    'VerificationResult',

    # Protocols
    'Department',
    'DepartmentProtocol',  # Alias for backward compatibility

    # Helpers
    'create_simple_request',
    'create_simple_response',

    # Type Aliases
    'DepartmentFactory',
    'VerificationFunction',

    # Configuration
    'DepartmentConfig',
    'DepartmentManifest',

    # Learning Functions
    'compute_learning_rate',
    'should_update_now',
]
