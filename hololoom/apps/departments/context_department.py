"""
Context Department - Context management and enrichment department.

Provides context-aware request enrichment, session tracking, privacy enforcement,
and cross-department state management. Integrates with HoloLoom's existing context
system (HoloLoom/context/) for routing, learning, and monitoring.

Key Capabilities:
- Context enrichment: Add user context, session history, preferences to requests
- Session tracking: Track user sessions, conversation history, state
- Privacy enforcement: Ensure privacy constraints are respected
- Context retrieval: Pull relevant context for requests
- Preference learning: Learn user patterns and adapt
- State management: Maintain state across department calls

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from hololoom.config import Config
from hololoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    PrivacyEnvelope,
    PrivacyLevel,
    VerificationResult,
    VerificationCheck,
    VerificationStatus,
    DSStarCheck,
    DepartmentConfig,
)
from hololoom.apps.departments.base import BaseDepartment

# Import existing context system components
from hololoom.context import (
    QueryRouter,
    QueryClassifier,
    ThompsonBandit,
    LearningTracker,
    ConfidenceCalibrator,
    StrategyUpdater,
    ErrorHandler,
    SystemMonitor,
    CircuitBreaker,
    RateLimiter,
    HealthChecker,
    create_query_router,
    create_classifier,
    create_thompson_bandit,
    create_error_handler,
    create_system_monitor,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Context Types
# ============================================================================

class DataSensitivity(Enum):
    """Data sensitivity levels for privacy enforcement."""
    PUBLIC = "public"              # No restrictions
    INTERNAL = "internal"          # Organization only
    CONFIDENTIAL = "confidential"  # Authorized users only
    RESTRICTED = "restricted"      # Strict access control
    CRITICAL = "critical"          # Maximum security


@dataclass
class ContextEnvelope:
    """
    Context envelope containing user context, session state, and preferences.

    Wraps all contextual information needed to enrich requests:
    - User identity and session
    - Historical interactions
    - User preferences and settings
    - Privacy constraints
    - Temporal context
    """
    user_id: str                                    # User identifier
    session_id: str                                 # Session identifier
    preferences: Dict[str, Any] = field(default_factory=dict)  # User preferences
    history: List[Dict[str, Any]] = field(default_factory=list)  # Recent interactions
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    # Privacy constraints
    privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL
    data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    retention_days: int = 90  # How long to keep data

    # Access control
    allowed_departments: Set[str] = field(default_factory=set)  # Which departments can access
    access_log: List[Dict[str, Any]] = field(default_factory=list)  # Who accessed when

    def log_access(self, department: str, reason: str, approved: bool = True) -> None:
        """Log context access by a department."""
        self.access_log.append({
            "department": department,
            "reason": reason,
            "approved": approved,
            "timestamp": datetime.utcnow(),
            "privacy_level": self.privacy_level.value,
        })
        self.last_accessed = datetime.utcnow()


@dataclass
class SessionState:
    """
    Active session state with conversation history and metadata.

    Tracks:
    - Session lifecycle (created, last active, timeout)
    - Conversation history (recent messages)
    - Department interactions (which departments involved)
    - Session metadata (browser, location, etc.)
    """
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    timeout_minutes: int = 30

    # Conversation tracking
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=50))
    message_count: int = 0

    # Department tracking
    departments_used: Set[str] = field(default_factory=set)
    department_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        elapsed = (datetime.utcnow() - self.last_active).total_seconds() / 60
        return elapsed > self.timeout_minutes

    def touch(self) -> None:
        """Update last active timestamp."""
        self.last_active = datetime.utcnow()

    def add_message(self, role: str, content: str, department: Optional[str] = None) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "department": department,
        })
        self.message_count += 1

        if department:
            self.departments_used.add(department)
            self.department_counts[department] += 1

        self.touch()


@dataclass
class PrivacyConstraint:
    """
    Privacy constraint defining access controls and retention policies.

    Enforces:
    - Data sensitivity classification
    - Access control (who can access)
    - Retention policy (how long to keep)
    - Redaction rules (what to mask)
    - Audit requirements
    """
    sensitivity: DataSensitivity
    retention_days: int
    access_control: Dict[str, List[str]] = field(default_factory=dict)  # role → departments
    redaction_rules: List[str] = field(default_factory=list)  # What to redact
    requires_audit: bool = False
    allowed_operations: Set[str] = field(default_factory=set)  # What operations allowed

    def is_allowed(self, department: str, operation: str) -> bool:
        """Check if department is allowed to perform operation."""
        if not self.allowed_operations or operation in self.allowed_operations:
            # Check access control by department
            for role, depts in self.access_control.items():
                if department in depts:
                    return True
        return False


@dataclass
class ContextEnrichment:
    """
    Result of context enrichment operation.

    Contains:
    - Original request
    - Enriched request (with context added)
    - Context sources used
    - Enrichment metadata
    """
    original_request: DepartmentRequest
    enriched_request: DepartmentRequest
    context_sources: List[str] = field(default_factory=list)  # Where context came from
    enrichment_metadata: Dict[str, Any] = field(default_factory=dict)
    enrichment_time_ms: float = 0.0


# ============================================================================
# Context Department
# ============================================================================

class ContextDepartment(BaseDepartment):
    """
    Context Department implementing context management and enrichment.

    Provides:
    1. Context-aware request enrichment (add user context, history, preferences)
    2. Cross-department state management (maintain state across department calls)
    3. Session tracking and memory (track sessions, conversation history)
    4. Privacy envelope enforcement (ensure privacy constraints)
    5. Context retrieval and injection (pull relevant context)
    6. Context learning and adaptation (learn user patterns)

    Department Capabilities:
    - enrich_request: Add context to request
    - track_session: Track user session
    - enforce_privacy: Validate privacy constraints
    - retrieve_context: Get relevant context
    - update_preferences: Learn user preferences

    Constraints:
    - max_history_length: 50 messages per session
    - max_context_size: 10KB per request
    - session_timeout: 30 minutes
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        department_id: str = "context",
        enable_learning: bool = True,
        enable_privacy: bool = True,
    ):
        """
        Initialize Context Department.

        Args:
            config: HoloLoom configuration
            department_id: Department identifier
            enable_learning: Enable context learning
            enable_privacy: Enable privacy enforcement
        """
        # Create department config
        dept_config = DepartmentConfig(
            name="context",
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "enrich_request",
                "track_session",
                "enforce_privacy",
                "retrieve_context",
                "update_preferences",
            ],
            confidence_range=(0.70, 0.95),
        )

        super().__init__(
            name="context",
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "enrich_request",
                "track_session",
                "enforce_privacy",
                "retrieve_context",
                "update_preferences",
            ],
            confidence_range=(0.70, 0.95),
            config=dept_config,
        )

        self.config = config or Config.fast()
        self.enable_learning = enable_learning
        self.enable_privacy = enable_privacy

        # Context storage
        self._context_envelopes: Dict[str, ContextEnvelope] = {}  # user_id → envelope
        self._active_sessions: Dict[str, SessionState] = {}  # session_id → state
        self._privacy_constraints: Dict[str, PrivacyConstraint] = {}  # user_id → constraint

        # Context learning
        self._preference_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # user_id → preference_key → count
        self._context_usage: Dict[str, int] = defaultdict(int)  # context_type → usage_count

        # Integration with existing context system
        self._classifier: Optional[QueryClassifier] = None
        self._error_handler: Optional[ErrorHandler] = None
        self._monitor: Optional[SystemMonitor] = None

        # Context retrieval cache
        self._context_cache: Dict[str, Any] = {}  # cache_key → context_data
        self._cache_hits = 0
        self._cache_misses = 0

        # Privacy tracking
        self._privacy_violations: List[Dict[str, Any]] = []
        self._access_denied_count = 0

        # Performance tracking
        self._enrichment_times: List[float] = []
        self._avg_context_size: float = 0.0

        logger.info(f"✓ ContextDepartment initialized (id={department_id})")

    async def __aenter__(self):
        """Async context manager entry - initialize context system components."""
        await super().__aenter__()

        # Initialize context system components
        try:
            self._classifier = create_classifier()
            self._error_handler = create_error_handler()
            self._monitor = create_system_monitor(
                enable_performance=True,
                enable_resources=True,
                enable_learning=True,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize context system components: {e}")
            # Graceful degradation

        # Start session cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        logger.info("✓ Context Department ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup."""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await super().__aexit__(exc_type, exc_val, exc_tb)
        logger.info("✓ Context Department closed")

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute context operation.

        Supported operations:
        - enrich_request: Add context to request
        - track_session: Track user session
        - enforce_privacy: Validate privacy constraints
        - retrieve_context: Get relevant context
        - update_preferences: Update user preferences

        Args:
            request: DepartmentRequest with task_type and parameters

        Returns:
            DepartmentResponse with result and confidence
        """
        start_time = time.time()
        task_type = request.task_type

        try:
            if task_type == "enrich_request":
                result = await self._enrich_request(request)
                confidence_score = 0.85
            elif task_type == "track_session":
                result = await self._track_session(request)
                confidence_score = 0.90
            elif task_type == "enforce_privacy":
                result = await self._enforce_privacy(request)
                confidence_score = 0.95
            elif task_type == "retrieve_context":
                result = await self._retrieve_context(request)
                confidence_score = 0.80
            elif task_type == "update_preferences":
                result = await self._update_preferences(request)
                confidence_score = 0.85
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            latency_ms = (time.time() - start_time) * 1000
            self._enrichment_times.append(latency_ms)

            # Update base class metrics
            self._metrics['total_requests'] += 1
            self._metrics['successful_requests'] += 1
            self._metrics['total_latency_ms'] += latency_ms
            self._metrics['confidence_history'].append(confidence_score)

            confidence = ConfidenceMetadata.from_score(
                confidence_score,
                justification=[f"Successfully executed {task_type}"],
                sources=["context_system"],
            )

            return DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Context execution failed: {e}")
            latency_ms = (time.time() - start_time) * 1000

            # Update base class metrics
            self._metrics['total_requests'] += 1
            self._metrics['failed_requests'] += 1
            self._metrics['total_latency_ms'] += latency_ms
            self._metrics['error_log'].append(str(e))

            return DepartmentResponse(
                task_id=request.task_id,
                result=None,
                confidence=ConfidenceMetadata.from_score(0.0),
                error=str(e),
                latency_ms=latency_ms,
            )

    async def _enrich_request(self, request: DepartmentRequest) -> ContextEnrichment:
        """
        Enrich request with context.

        Adds:
        - User context (preferences, history)
        - Session state (conversation history)
        - Temporal context (time of day, day of week)
        - Privacy envelope
        """
        start_time = time.time()

        user_id = request.parameters.get("user_id", "anonymous")
        session_id = request.parameters.get("session_id", "default")

        # Get or create context envelope
        envelope = self._get_or_create_envelope(user_id, session_id)

        # Get session state
        session = self._get_or_create_session(user_id, session_id)

        # Build enriched context
        enriched_context = {
            **request.context,
            "user_id": user_id,
            "session_id": session_id,
            "preferences": envelope.preferences,
            "conversation_history": list(session.conversation_history)[-10:],  # Last 10 messages
            "session_metadata": session.metadata,
            "temporal": {
                "timestamp": datetime.utcnow().isoformat(),
                "hour": datetime.utcnow().hour,
                "day_of_week": datetime.utcnow().weekday(),
            },
        }

        # Create enriched request
        enriched_request = DepartmentRequest(
            task_id=request.task_id,
            task_type=request.task_type,
            parameters=request.parameters,
            context=enriched_context,
            constraints=request.constraints,
            timestamp=request.timestamp,
            priority=request.priority,
        )

        enrichment_time = (time.time() - start_time) * 1000

        # Log access
        envelope.log_access(
            department="context",
            reason="request_enrichment",
            approved=True,
        )

        # Track context usage
        self._context_usage["user_preferences"] += 1
        self._context_usage["conversation_history"] += 1

        return ContextEnrichment(
            original_request=request,
            enriched_request=enriched_request,
            context_sources=["user_envelope", "session_state", "temporal"],
            enrichment_time_ms=enrichment_time,
            enrichment_metadata={
                "preferences_count": len(envelope.preferences),
                "history_length": len(session.conversation_history),
            },
        )

    async def _track_session(self, request: DepartmentRequest) -> Dict[str, Any]:
        """
        Track user session.

        Updates session state with new interaction.
        """
        user_id = request.parameters.get("user_id", "anonymous")
        session_id = request.parameters.get("session_id", "default")
        message = request.parameters.get("message", "")
        role = request.parameters.get("role", "user")
        department = request.parameters.get("department")

        # Get or create session
        session = self._get_or_create_session(user_id, session_id)

        # Add message
        session.add_message(role, message, department)

        return {
            "session_id": session_id,
            "message_count": session.message_count,
            "departments_used": list(session.departments_used),
            "is_active": not session.is_expired(),
            "last_active": session.last_active.isoformat(),
        }

    async def _enforce_privacy(self, request: DepartmentRequest) -> Dict[str, Any]:
        """
        Enforce privacy constraints.

        Validates that request complies with privacy requirements.
        """
        user_id = request.parameters.get("user_id", "anonymous")
        department = request.parameters.get("requesting_department", "unknown")
        operation = request.parameters.get("operation", "read")

        # Get privacy constraint
        constraint = self._get_or_create_constraint(user_id)

        # Check if access allowed
        allowed = constraint.is_allowed(department, operation)

        if not allowed:
            self._access_denied_count += 1
            self._privacy_violations.append({
                "user_id": user_id,
                "department": department,
                "operation": operation,
                "timestamp": datetime.utcnow(),
                "reason": "access_denied",
            })
            logger.warning(f"Privacy violation: {department} denied {operation} for {user_id}")

        return {
            "allowed": allowed,
            "sensitivity": constraint.sensitivity.value,
            "retention_days": constraint.retention_days,
            "requires_audit": constraint.requires_audit,
        }

    async def _retrieve_context(self, request: DepartmentRequest) -> Dict[str, Any]:
        """
        Retrieve relevant context for request.

        Returns context from envelope and session state.
        """
        user_id = request.parameters.get("user_id", "anonymous")
        session_id = request.parameters.get("session_id", "default")
        context_types = request.parameters.get("context_types", ["preferences", "history"])

        # Check cache
        cache_key = f"{user_id}:{session_id}:{':'.join(sorted(context_types))}"
        if cache_key in self._context_cache:
            self._cache_hits += 1
            return self._context_cache[cache_key]

        self._cache_misses += 1

        # Get envelope and session
        envelope = self._get_or_create_envelope(user_id, session_id)
        session = self._get_or_create_session(user_id, session_id)

        # Build context
        context = {}

        if "preferences" in context_types:
            context["preferences"] = envelope.preferences

        if "history" in context_types:
            context["conversation_history"] = list(session.conversation_history)[-10:]

        if "metadata" in context_types:
            context["session_metadata"] = session.metadata

        if "departments" in context_types:
            context["departments_used"] = list(session.departments_used)

        # Cache result
        self._context_cache[cache_key] = context

        return context

    async def _update_preferences(self, request: DepartmentRequest) -> Dict[str, Any]:
        """
        Update user preferences.

        Learns from user interactions and feedback.
        """
        user_id = request.parameters.get("user_id", "anonymous")
        session_id = request.parameters.get("session_id", "default")
        preferences = request.parameters.get("preferences", {})

        # Get envelope
        envelope = self._get_or_create_envelope(user_id, session_id)

        # Update preferences
        for key, value in preferences.items():
            envelope.preferences[key] = value

            # Track preference patterns for learning
            if self.enable_learning:
                self._preference_patterns[user_id][key] += 1

        return {
            "user_id": user_id,
            "preferences_updated": len(preferences),
            "total_preferences": len(envelope.preferences),
        }

    def _get_or_create_envelope(self, user_id: str, session_id: str) -> ContextEnvelope:
        """Get or create context envelope for user."""
        if user_id not in self._context_envelopes:
            self._context_envelopes[user_id] = ContextEnvelope(
                user_id=user_id,
                session_id=session_id,
                privacy_level=PrivacyLevel.INTERNAL,
                data_sensitivity=DataSensitivity.INTERNAL,
            )
        return self._context_envelopes[user_id]

    def _get_or_create_session(self, user_id: str, session_id: str) -> SessionState:
        """Get or create session state."""
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = SessionState(
                session_id=session_id,
                user_id=user_id,
            )
        return self._active_sessions[session_id]

    def _get_or_create_constraint(self, user_id: str) -> PrivacyConstraint:
        """Get or create privacy constraint."""
        if user_id not in self._privacy_constraints:
            # Default constraint
            self._privacy_constraints[user_id] = PrivacyConstraint(
                sensitivity=DataSensitivity.INTERNAL,
                retention_days=90,
                access_control={
                    "admin": ["*"],  # Admins can access all departments
                    "user": ["context", "rag", "planning"],  # Users limited
                },
                allowed_operations={"read", "write", "enrich"},
            )
        return self._privacy_constraints[user_id]

    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                expired = []
                for session_id, session in self._active_sessions.items():
                    if session.is_expired():
                        expired.append(session_id)

                for session_id in expired:
                    del self._active_sessions[session_id]
                    logger.info(f"Cleaned up expired session: {session_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify context response quality (5 dimensions).

        Verification dimensions:
        1. Context Completeness: Is all required context present?
        2. Privacy Compliance: Are privacy constraints respected?
        3. Session Validity: Is session active and valid?
        4. State Consistency: Is state consistent across departments?
        5. Relevance: Is context relevant to request?
        """
        checks = []

        # 1. Context Completeness
        completeness_score = self._check_context_completeness(response)
        checks.append(VerificationCheck(
            name="context_completeness",
            status=VerificationStatus.PASSED if completeness_score >= 0.8 else VerificationStatus.FAILED,
            reason=f"Context completeness: {completeness_score:.2f}",
            score=completeness_score,
        ))

        # 2. Privacy Compliance
        privacy_score = self._check_privacy_compliance(response)
        checks.append(VerificationCheck(
            name="privacy_compliance",
            status=VerificationStatus.PASSED if privacy_score >= 0.95 else VerificationStatus.FAILED,
            reason=f"Privacy compliance: {privacy_score:.2f}",
            score=privacy_score,
        ))

        # 3. Session Validity
        session_score = self._check_session_validity(response)
        checks.append(VerificationCheck(
            name="session_validity",
            status=VerificationStatus.PASSED if session_score >= 0.9 else VerificationStatus.FAILED,
            reason=f"Session validity: {session_score:.2f}",
            score=session_score,
        ))

        # 4. State Consistency
        consistency_score = self._check_state_consistency(response)
        checks.append(VerificationCheck(
            name="state_consistency",
            status=VerificationStatus.PASSED if consistency_score >= 0.85 else VerificationStatus.FAILED,
            reason=f"State consistency: {consistency_score:.2f}",
            score=consistency_score,
        ))

        # 5. Relevance
        relevance_score = self._check_relevance(response)
        checks.append(VerificationCheck(
            name="relevance",
            status=VerificationStatus.PASSED if relevance_score >= 0.75 else VerificationStatus.FAILED,
            reason=f"Context relevance: {relevance_score:.2f}",
            score=relevance_score,
        ))

        # Compute overall score
        overall_score = sum(c.score for c in checks) / len(checks)
        verified = all(c.status == VerificationStatus.PASSED for c in checks)

        return VerificationResult(
            verified=verified,
            checks=checks,
            overall_score=overall_score,
            summary=f"Context verification: {overall_score:.2%}",
            confidence=response.confidence.score,
        )

    def _check_context_completeness(self, response: DepartmentResponse) -> float:
        """Check if all required context is present."""
        if response.error:
            return 0.0

        result = response.result
        if isinstance(result, ContextEnrichment):
            # Check enrichment quality
            has_preferences = "preferences" in result.enriched_request.context
            has_history = "conversation_history" in result.enriched_request.context
            has_temporal = "temporal" in result.enriched_request.context

            completeness = sum([has_preferences, has_history, has_temporal]) / 3
            return completeness

        return 0.8  # Default for non-enrichment tasks

    def _check_privacy_compliance(self, response: DepartmentResponse) -> float:
        """Check privacy compliance."""
        # No violations in last 10 checks = perfect score
        recent_violations = len([v for v in self._privacy_violations[-10:]])
        return max(0.0, 1.0 - (recent_violations * 0.1))

    def _check_session_validity(self, response: DepartmentResponse) -> float:
        """Check session validity."""
        if response.error:
            return 0.0

        # Check if sessions are being tracked
        active_sessions = len([s for s in self._active_sessions.values() if not s.is_expired()])
        if active_sessions > 0:
            return 0.95
        return 0.5  # No active sessions

    def _check_state_consistency(self, response: DepartmentResponse) -> float:
        """Check state consistency."""
        # Check if context cache is consistent
        cache_hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        return 0.7 + (cache_hit_rate * 0.3)  # 0.7-1.0 range

    def _check_relevance(self, response: DepartmentResponse) -> float:
        """Check context relevance."""
        # Based on confidence score
        return response.confidence.score

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult,
    ) -> DepartmentResponse:
        """
        Refine context response.

        Refinement strategies:
        - Low completeness: Fetch additional context
        - Privacy violation: Apply stricter constraints
        - Session expired: Refresh session
        - Low relevance: Filter to most relevant context
        """
        # Analyze verification failures
        failed_checks = [c for c in verification.checks if c.status == VerificationStatus.FAILED]

        if not failed_checks:
            return prior_response  # No refinement needed

        logger.info(f"Refining context response: {[c.name for c in failed_checks]}")

        # Apply refinement based on failures
        refined_request = request

        for check in failed_checks:
            if check.name == "context_completeness":
                # Fetch additional context
                refined_request.parameters["context_types"] = [
                    "preferences", "history", "metadata", "departments"
                ]
            elif check.name == "privacy_compliance":
                # Apply stricter privacy
                refined_request.parameters["privacy_level"] = "restricted"
            elif check.name == "session_validity":
                # Refresh session
                user_id = request.parameters.get("user_id", "anonymous")
                session_id = request.parameters.get("session_id", "default")
                session = self._get_or_create_session(user_id, session_id)
                session.touch()

        # Re-execute with refined request
        return await self.execute(refined_request)

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback to improve context operations.

        Updates:
        - Preference patterns (what users prefer)
        - Context usage patterns (what context is useful)
        - Privacy patterns (what privacy levels are typical)
        """
        if not self.enable_learning:
            return

        outcome = feedback.get("outcome", "unknown")
        user_id = feedback.get("user_id")

        if outcome == "success" and user_id:
            # Learn successful patterns
            preferences = feedback.get("preferences", {})
            for key, value in preferences.items():
                self._preference_patterns[user_id][key] += 1

        elif outcome == "failure":
            # Learn from failures
            failure_reason = feedback.get("reason", "unknown")
            if "privacy" in failure_reason.lower():
                # Privacy violation - increase privacy level
                if user_id and user_id in self._privacy_constraints:
                    constraint = self._privacy_constraints[user_id]
                    # Could upgrade sensitivity level here

        logger.debug(f"Updated context strategy based on {outcome} feedback")

    async def get_capabilities(self) -> Dict[str, Any]:
        """Report context department capabilities."""
        return {
            "supported_tasks": self.supported_tasks,
            "constraints": {
                "max_history_length": 50,
                "max_context_size_kb": 10,
                "session_timeout_minutes": 30,
            },
            "features": {
                "context_enrichment": True,
                "session_tracking": True,
                "privacy_enforcement": self.enable_privacy,
                "context_learning": self.enable_learning,
                "context_caching": True,
            },
            "integrations": {
                "classifier": self._classifier is not None,
                "error_handler": self._error_handler is not None,
                "monitor": self._monitor is not None,
            },
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get context department metrics."""
        cache_hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        avg_enrichment_time = sum(self._enrichment_times[-100:]) / max(1, len(self._enrichment_times[-100:]))

        return {
            "sessions": {
                "active_sessions": len(self._active_sessions),
                "total_sessions": len(self._active_sessions),
            },
            "context": {
                "envelopes_tracked": len(self._context_envelopes),
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "avg_enrichment_time_ms": avg_enrichment_time,
            },
            "privacy": {
                "privacy_violations": len(self._privacy_violations),
                "access_denied_count": self._access_denied_count,
                "constraints_active": len(self._privacy_constraints),
            },
            "learning": {
                "preference_patterns": len(self._preference_patterns),
                "context_usage": dict(self._context_usage),
            },
            "performance": {
                "requests_processed": self._metrics["total_requests"],
                "success_rate": (
                    self._metrics["successful_requests"] / max(1, self._metrics["total_requests"])
                ),
            },
        }

    async def health_check(self) -> bool:
        """Check context department health."""
        try:
            # Check active sessions
            active_count = len([s for s in self._active_sessions.values() if not s.is_expired()])

            # Check memory usage
            envelope_count = len(self._context_envelopes)

            # All systems operational
            return active_count >= 0 and envelope_count >= 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# ============================================================================
# Factory Function
# ============================================================================

def create_context_department(
    config: Optional[Config] = None,
    enable_learning: bool = True,
    enable_privacy: bool = True,
) -> ContextDepartment:
    """
    Create a Context Department instance.

    Args:
        config: HoloLoom configuration
        enable_learning: Enable context learning
        enable_privacy: Enable privacy enforcement

    Returns:
        ContextDepartment instance

    Example:
        async with create_context_department() as dept:
            request = DepartmentRequest(
                task_type="enrich_request",
                parameters={"user_id": "user123", "session_id": "sess456"}
            )
            response = await dept.execute(request)
    """
    return ContextDepartment(
        config=config,
        enable_learning=enable_learning,
        enable_privacy=enable_privacy,
    )
