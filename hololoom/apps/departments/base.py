#!/usr/bin/env python3
"""
Base Department - Abstract base class providing common department functionality.

This module provides a concrete implementation of common department behaviors,
enabling rapid development of new departments. Inherit from BaseDepartment
and override only the domain-specific methods (execute, verify, refine).

Key Features:
- Three-tier memory system (short/medium/long-term)
- Automatic confidence tracking and calibration
- Session management with automatic cleanup
- Learning signal aggregation
- Health monitoring with performance metrics
- Async lifecycle management (context manager support)

Design Philosophy:
"Don't repeat yourself. Common department behaviors should be implemented once
and inherited, not copy-pasted across departments."
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from hololoom.protocols.department import (
    DepartmentConfig,
    DepartmentManifest,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class BaseDepartment:
    """
    Abstract base class for all departments.

    Provides common functionality that all departments need:
    - Memory management (3-tier system)
    - Confidence tracking and calibration
    - Session state management
    - Learning signal aggregation
    - Health monitoring
    - Lifecycle management

    Subclasses must implement:
    - execute(): Core department logic
    - verify(): Quality checks (DS-STAR)
    - refine(): Response improvement

    Example:

        class ContextDepartment(BaseDepartment):
            def __init__(self, orchestrator: WeavingOrchestrator):
                super().__init__(
                    name="context",
                    domain="general",
                    version="1.0.0",
                    supported_tasks=["retrieve_context", "weave_response"],
                    confidence_range=(0.65, 0.95)
                )
                self.orchestrator = orchestrator

            async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
                # Wrap orchestrator.weave()
                result = await self.orchestrator.weave(Query(text=request.parameters['query']))

                confidence = ConfidenceMetadata.from_score(
                    result.confidence,
                    justification=["Weaving completed successfully"]
                )

                return DepartmentResponse(
                    task_id=request.task_id,
                    result=result.response,
                    confidence=confidence
                )

            async def verify(self, response: DepartmentResponse) -> VerificationResult:
                # Simple verification: check confidence threshold
                sufficient = response.confidence.score >= 0.75
                return VerificationResult(
                    sufficient=sufficient,
                    confidence_valid=True,
                    reasoning_sound=True
                )

            async def refine(self, request, prior_response, verification) -> DepartmentResponse:
                # Context expansion for low confidence
                expanded_request = request.copy()
                expanded_request.context_preference = "comprehensive"
                return await self.execute(expanded_request)
    """

    def __init__(
        self,
        name: str,
        domain: str,
        version: str,
        supported_tasks: list[str],
        confidence_range: tuple[float, float],
        config: DepartmentConfig | None = None
    ):
        """
        Initialize base department.

        Args:
            name: Unique department identifier
            domain: Domain specialization (e.g., "general", "beekeeping")
            version: Semantic version (e.g., "1.0.0")
            supported_tasks: List of task types this department handles
            confidence_range: Typical (min, max) confidence
            config: Optional configuration (uses defaults if not provided)
        """
        # Identity
        self.name = name
        self.domain = domain
        self.version = version
        self.supported_tasks = supported_tasks
        self.confidence_range = confidence_range

        # Configuration
        self.config = config or DepartmentConfig(name=name, domain=domain)

        # Three-tier memory system
        self.short_term_memory: dict[str, Any] = {}  # Recent interactions (this session)
        self.medium_term_memory: dict[str, Any] = {}  # Session patterns (hours to days)
        self.long_term_memory: dict[str, Any] = {}  # Institutional knowledge (weeks to months)

        # Memory capacity tracking
        self._short_term_keys: deque = deque(maxlen=self.config.short_term_capacity)
        self._medium_term_keys: deque = deque(maxlen=self.config.medium_term_capacity)
        self._long_term_keys: deque = deque(maxlen=self.config.long_term_capacity)

        # Session tracking
        self._active_sessions: dict[str, datetime] = {}  # session_id → last_access
        self._session_requests: dict[str, int] = defaultdict(int)  # session_id → request_count

        # Learning signals buffer
        self._learning_signals: list[dict[str, Any]] = []
        self._last_strategy_update = datetime.now()

        # Performance metrics
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0.0,
            'confidence_history': [],  # Recent confidence scores
            'error_log': []  # Recent errors
        }

        # Confidence calibration tracking
        self._confidence_calibration: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # task_type → [(predicted_confidence, actual_quality)]

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        # Lifecycle
        self._initialized = False
        self._closed = False

        logger.info(f"Initialized {self.name} department (version {self.version})")

    # ========================================================================
    # Core Department Methods (Must Override)
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """
        Execute a task and return a response.

        MUST be overridden by subclasses to implement domain-specific logic.

        Args:
            request: Standardized request with task details

        Returns:
            DepartmentResponse with result and confidence metadata

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute() method"
        )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """
        Verify response quality using DS-STAR pattern.

        MUST be overridden by subclasses to implement domain-specific verification.

        Args:
            response: Response to verify

        Returns:
            VerificationResult with pass/fail checks

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement verify() method"
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine a response based on verification feedback.

        MUST be overridden by subclasses to implement domain-specific refinement.

        Args:
            request: Original request
            prior_response: Previous response that was insufficient
            verification: Verification results with improvement suggestions

        Returns:
            Refined DepartmentResponse

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement refine() method"
        )

    # ========================================================================
    # Learning and Strategy Updates
    # ========================================================================

    async def update_strategy(
        self,
        learning_signals: list[dict[str, Any]]
    ) -> None:
        """
        Update department strategy based on learning signals.

        This base implementation:
        1. Aggregates learning signals
        2. Updates confidence calibration
        3. Updates institutional memory (long-term patterns)

        Subclasses can override to add domain-specific learning.

        Args:
            learning_signals: List of signal dicts from recent executions
        """
        if not self.config.enable_learning:
            return

        # Aggregate signals
        self._learning_signals.extend(learning_signals)

        # Extract patterns
        for signal in learning_signals:
            task_type = signal.get('task_type')
            outcome = signal.get('outcome')
            confidence_predicted = signal.get('confidence_predicted', 0.0)
            confidence_actual = signal.get('confidence_actual', 0.0)

            # Update confidence calibration
            if task_type and confidence_predicted > 0:
                self._confidence_calibration[task_type].append(
                    (confidence_predicted, confidence_actual)
                )

            # Update institutional memory (success rates by task type)
            if task_type:
                if task_type not in self.long_term_memory:
                    self.long_term_memory[task_type] = {
                        'total': 0,
                        'successes': 0,
                        'failures': 0,
                        'avg_confidence': 0.0
                    }

                stats = self.long_term_memory[task_type]
                stats['total'] += 1

                if outcome == 'success':
                    stats['successes'] += 1
                else:
                    stats['failures'] += 1

                # Update rolling average confidence
                stats['avg_confidence'] = (
                    (stats['avg_confidence'] * (stats['total'] - 1) + confidence_actual)
                    / stats['total']
                )

        self._last_strategy_update = datetime.now()

        logger.debug(
            f"{self.name}: Updated strategy with {len(learning_signals)} signals"
        )

    # ========================================================================
    # Session Management
    # ========================================================================

    async def get_session_state(
        self,
        session_id: str
    ) -> dict[str, Any] | None:
        """
        Retrieve session-specific state.

        Returns combined short-term and medium-term memory for the session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dict with session state, or None if session not found
        """
        if session_id not in self._active_sessions:
            return None

        # Update last access time
        self._active_sessions[session_id] = datetime.now()

        # Combine short-term and medium-term memory for this session
        session_state = {
            'session_id': session_id,
            'last_access': self._active_sessions[session_id].isoformat(),
            'request_count': self._session_requests[session_id],
            'short_term': self.short_term_memory.get(session_id, {}),
            'medium_term': self.medium_term_memory.get(session_id, {})
        }

        return session_state

    async def _store_session_state(
        self,
        session_id: str,
        state: dict[str, Any]
    ) -> None:
        """Store session state in short-term memory."""
        if not self.config.enable_session_memory:
            return

        # Update session tracking
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = datetime.now()

        self._session_requests[session_id] += 1

        # Store in short-term memory
        self.short_term_memory[session_id] = state
        self._short_term_keys.append(session_id)

        # Cleanup old sessions (every 100 requests)
        if len(self._active_sessions) % 100 == 0:
            await self._cleanup_old_sessions()

    async def _cleanup_old_sessions(self) -> None:
        """Remove sessions inactive for >1 hour."""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        inactive_sessions = [
            sid for sid, last_access in self._active_sessions.items()
            if last_access < cutoff
        ]

        for session_id in inactive_sessions:
            # Move to medium-term memory before deleting
            if session_id in self.short_term_memory:
                self.medium_term_memory[session_id] = self.short_term_memory[session_id]
                self._medium_term_keys.append(session_id)
                del self.short_term_memory[session_id]

            del self._active_sessions[session_id]

        if inactive_sessions:
            logger.debug(
                f"{self.name}: Cleaned up {len(inactive_sessions)} inactive sessions"
            )

    # ========================================================================
    # Institutional Memory
    # ========================================================================

    async def get_institutional_memory(
        self,
        pattern_type: str
    ) -> dict[str, Any]:
        """
        Retrieve cross-session patterns (long-term memory).

        Supported pattern types:
        - "successful_strategies": What works well
        - "failure_modes": Common mistakes
        - "confidence_calibration": Historical accuracy
        - "performance_stats": Latency, throughput

        Args:
            pattern_type: Type of pattern to retrieve

        Returns:
            Dict with institutional patterns
        """
        if pattern_type == "successful_strategies":
            # Return task types with >80% success rate
            return {
                task_type: stats
                for task_type, stats in self.long_term_memory.items()
                if stats.get('total', 0) > 0
                and stats['successes'] / stats['total'] >= 0.8
            }

        elif pattern_type == "failure_modes":
            # Return task types with <50% success rate
            return {
                task_type: stats
                for task_type, stats in self.long_term_memory.items()
                if stats.get('total', 0) > 0
                and stats['successes'] / stats['total'] < 0.5
            }

        elif pattern_type == "confidence_calibration":
            # Return calibration data for each task type
            return {
                task_type: {
                    'count': len(pairs),
                    'avg_predicted': sum(p for p, _ in pairs) / len(pairs),
                    'avg_actual': sum(a for _, a in pairs) / len(pairs),
                    'calibration_error': abs(
                        sum(p for p, _ in pairs) / len(pairs) -
                        sum(a for _, a in pairs) / len(pairs)
                    )
                }
                for task_type, pairs in self._confidence_calibration.items()
                if pairs
            }

        elif pattern_type == "performance_stats":
            return {
                'total_requests': self._metrics['total_requests'],
                'success_rate': (
                    self._metrics['successful_requests'] / self._metrics['total_requests']
                    if self._metrics['total_requests'] > 0 else 0.0
                ),
                'avg_latency_ms': (
                    self._metrics['total_latency_ms'] / self._metrics['total_requests']
                    if self._metrics['total_requests'] > 0 else 0.0
                ),
                'recent_errors': self._metrics['error_log'][-10:]  # Last 10 errors
            }

        else:
            logger.warning(f"{self.name}: Unknown pattern_type: {pattern_type}")
            return {}

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Report department health and performance metrics.

        Returns comprehensive health information:
        - Status: "healthy" | "degraded" | "unhealthy"
        - Performance: latency, throughput, error rate
        - Memory: usage statistics
        - Learning: confidence calibration

        Returns:
            Dict with health metrics
        """
        # Compute health status
        success_rate = (
            self._metrics['successful_requests'] / self._metrics['total_requests']
            if self._metrics['total_requests'] > 0 else 1.0
        )

        if success_rate >= 0.95:
            status = "healthy"
        elif success_rate >= 0.80:
            status = "degraded"
        else:
            status = "unhealthy"

        # Compute average latency
        avg_latency = (
            self._metrics['total_latency_ms'] / self._metrics['total_requests']
            if self._metrics['total_requests'] > 0 else 0.0
        )

        return {
            'status': status,
            'name': self.name,
            'version': self.version,
            'performance': {
                'total_requests': self._metrics['total_requests'],
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'error_rate': 1.0 - success_rate
            },
            'memory': {
                'short_term_size': len(self.short_term_memory),
                'medium_term_size': len(self.medium_term_memory),
                'long_term_size': len(self.long_term_memory),
                'active_sessions': len(self._active_sessions)
            },
            'learning': {
                'last_update': self._last_strategy_update.isoformat(),
                'signals_pending': len(self._learning_signals),
                'confidence_calibration': await self.get_institutional_memory('confidence_calibration')
            }
        }

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize department (override for custom initialization)."""
        if self._initialized:
            return

        self._initialized = True
        logger.info(f"{self.name}: Department initialized")

    async def close(self) -> None:
        """Cleanup resources on shutdown."""
        if self._closed:
            return

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Flush learning signals
        if self._learning_signals:
            await self.update_strategy(self._learning_signals)
            self._learning_signals.clear()

        self._closed = True
        logger.info(f"{self.name}: Department closed")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _record_request(
        self,
        success: bool,
        latency_ms: float,
        confidence: float
    ) -> None:
        """Record request metrics."""
        self._metrics['total_requests'] += 1

        if success:
            self._metrics['successful_requests'] += 1
        else:
            self._metrics['failed_requests'] += 1

        self._metrics['total_latency_ms'] += latency_ms
        self._metrics['confidence_history'].append(confidence)

        # Keep only recent history (last 1000)
        if len(self._metrics['confidence_history']) > 1000:
            self._metrics['confidence_history'] = self._metrics['confidence_history'][-1000:]

    def _record_error(self, error: Exception) -> None:
        """Record error for health monitoring."""
        self._metrics['error_log'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': type(error).__name__
        })

        # Keep only recent errors (last 100)
        if len(self._metrics['error_log']) > 100:
            self._metrics['error_log'] = self._metrics['error_log'][-100:]

    def get_manifest(self) -> DepartmentManifest:
        """Generate department manifest for marketplace registration."""
        # Compute average performance from metrics
        avg_latency = (
            self._metrics['total_latency_ms'] / self._metrics['total_requests']
            if self._metrics['total_requests'] > 0 else 0.0
        )

        throughput_qps = (
            self._metrics['total_requests'] / 3600  # Rough estimate (requests per hour)
            if self._metrics['total_requests'] > 0 else 0.0
        )

        # Create config for manifest
        manifest_config = DepartmentConfig(
            name=self.name,
            domain=self.domain,
            version=self.version,
            supported_tasks=self.supported_tasks,
            confidence_range=self.confidence_range
        )

        return DepartmentManifest(
            config=manifest_config,
            capabilities=self.supported_tasks,
            metadata={
                "avg_latency_ms": avg_latency,
                "throughput_qps": throughput_qps
            }
        )
