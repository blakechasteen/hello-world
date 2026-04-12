#!/usr/bin/env python3
"""
Context Department - Wraps existing HoloLoom as a modular department.

This department demonstrates that HoloLoom's existing functionality (244D embeddings,
Thompson Sampling, knowledge graphs, recursive refinement) is already a complete
department - it just needs wrapping in the department protocol.

Key Features:
- Wraps WeavingOrchestrator.weave() in execute()
- Maps existing memory to three-tier system (short/medium/long-term)
- Implements DS-STAR verification (quality checks)
- Implements refinement (context expansion, recursive improvement)
- Exposes existing learning systems through update_strategy()

Design Philosophy:
"70-80% code reuse. We're not rebuilding HoloLoom - we're making it modular."
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from hololoom.config import Config
from hololoom.protocols.types import MemoryShard, Query
from hololoom.fabric.spacetime import Spacetime
from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

from .base import BaseDepartment
from .protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class ContextDepartment(BaseDepartment):
    """
    Context Department - Wraps existing HoloLoom WeavingOrchestrator.

    This department provides context retrieval and response generation by
    delegating to the existing HoloLoom weaving cycle:

    1. Execute: WeavingOrchestrator.weave(query)
    2. Verify: Check confidence, response quality, trace completeness
    3. Refine: Context expansion, recursive improvement
    4. Learn: Extract signals from spacetime trace

    Supported Tasks:
    - "retrieve_context": Find relevant knowledge from memory
    - "weave_response": Generate a complete response with context
    - "expand_context": Expand context for low-confidence results

    Example:

        # Create Context Department
        config = Config.fused()
        shards = create_memory_shards()

        async with ContextDepartment(config=config, shards=shards) as dept:
            # Execute
            request = DepartmentRequest(
                task_id="req_001",
                task_type="weave_response",
                parameters={"query": "What is Thompson Sampling?"}
            )

            response = await dept.execute(request)

            # Verify
            verification = await dept.verify(response)

            # Refine if needed
            if not verification.sufficient:
                refined = await dept.refine(request, response, verification)
    """

    def __init__(
        self,
        config: Config | None = None,
        shards: list[MemoryShard] | None = None,
        orchestrator: WeavingOrchestrator | None = None,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize Context Department.

        Args:
            config: HoloLoom config (defaults to Config.fast())
            shards: Initial memory shards (optional)
            orchestrator: Existing orchestrator to wrap (optional, created if not provided)
            dept_config: Department configuration (optional)
        """
        # Initialize base department
        super().__init__(
            name="context",
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "retrieve_context",
                "weave_response",
                "expand_context"
            ],
            confidence_range=(0.65, 0.95),
            config=dept_config or DepartmentConfig()
        )

        # Store config and shards
        self.hololoom_config = config or Config.fast()
        self.shards = shards or []

        # Create or wrap orchestrator
        self._orchestrator = orchestrator
        self._owns_orchestrator = orchestrator is None

        # Refinement tracking
        self._refinement_history: dict[str, list[dict[str, Any]]] = {}

        # Performance tracking
        self._task_performance: dict[str, dict[str, float]] = {}

        logger.info(
            f"Context Department initialized with {len(self.shards)} shards "
            f"(config: {self.hololoom_config.mode})"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize orchestrator (lazy loading)."""
        if self._initialized:
            return

        # Create orchestrator if not provided
        if self._orchestrator is None:
            self._orchestrator = WeavingOrchestrator(
                cfg=self.hololoom_config,
                shards=self.shards,
                enable_semantic_cache=False  # Disable for faster initialization (tests)
            )

            # Note: WeavingOrchestrator.__init__ is sync, but we could add
            # async initialization if needed

        await super().initialize()

    async def close(self) -> None:
        """Cleanup orchestrator resources."""
        # Close orchestrator if we own it
        if self._owns_orchestrator and self._orchestrator is not None:
            if hasattr(self._orchestrator, 'close'):
                await self._orchestrator.close()

        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """
        Execute a context-related task.

        Supported task types:
        - "retrieve_context": Find relevant knowledge (no response generation)
        - "weave_response": Full weaving cycle (context + response)
        - "expand_context": Context expansion for refinement

        Args:
            request: Standardized request with task details

        Returns:
            DepartmentResponse with spacetime result and confidence

        Raises:
            ValueError: If task_type not supported
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Validate task type
        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        # Extract query from parameters
        query_text = request.parameters.get("query")
        if not query_text:
            raise ValueError("Missing required parameter: 'query'")

        query = Query(text=query_text)

        # Execute weaving cycle
        try:
            spacetime = await self._orchestrator.weave(query)

            # Extract confidence from spacetime
            confidence = self._extract_confidence(spacetime, request)

            # Build result based on task type
            result = self._build_result(spacetime, request.task_type)

            # Extract learning signals
            learning_signals = self._extract_learning_signals(spacetime, request)

            # Build response
            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._extract_reasoning(spacetime) if request.context_preference != "minimal" else None,
                alternatives_considered=None,  # Could extract from trace
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, spacetime),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            # Record success
            latency_ms = response.execution_time_ms
            self._record_request(success=True, latency_ms=latency_ms, confidence=confidence.score)

            # Track task performance
            self._track_task_performance(request.task_type, latency_ms, confidence.score)

            return response

        except Exception as e:
            # Record error
            self._record_error(e)

            # Return error response
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
        """
        Verify response quality using DS-STAR pattern.

        Checks:
        1. Confidence validity (score matches quality)
        2. Response completeness (has required fields)
        3. Reasoning soundness (trace is coherent)

        Args:
            response: Response to verify

        Returns:
            VerificationResult with pass/fail checks and refinement suggestions
        """
        # Check 1: Confidence threshold
        confidence_threshold = 0.75  # Minimum acceptable confidence
        confidence_valid = response.confidence.score >= confidence_threshold

        # Check 2: Response completeness
        result = response.result
        has_response = isinstance(result, dict) and "response" in result
        response_not_empty = has_response and result["response"]

        # Check 3: Reasoning soundness (if available)
        reasoning_sound = True
        if response.reasoning:
            # Check that trace has required stages
            trace = response.reasoning.get("trace", {})
            required_stages = ["retrieval", "decision"]
            reasoning_sound = all(stage in trace.get("stage_durations", {}) for stage in required_stages)

        # Determine if sufficient
        sufficient = confidence_valid and response_not_empty and reasoning_sound

        # Generate refinement suggestions if insufficient
        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {}

            if not confidence_valid:
                refinement_suggestions["expand_context"] = True
                refinement_suggestions["reason"] = "Low confidence - need more context"

            if not response_not_empty:
                refinement_suggestions["regenerate"] = True
                refinement_suggestions["reason"] = "Empty response - retry with different parameters"

            if not reasoning_sound:
                refinement_suggestions["check_trace"] = True
                refinement_suggestions["reason"] = "Incomplete reasoning trace"

        # Identify alternative paths
        alternative_paths = []
        if not sufficient:
            if response.confidence.score < 0.6:
                alternative_paths.append("recursive_refinement")
            if response.confidence.score < 0.5:
                alternative_paths.append("context_expansion")
            if response.confidence.score < 0.4:
                alternative_paths.append("fallback_to_retrieval_only")

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=alternative_paths,
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.3,
            escalation_reason="Very low confidence" if response.confidence.score < 0.3 else None,
            verifier_confidence=0.9  # High confidence in verification logic
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine response based on verification feedback.

        Refinement strategies:
        1. Context expansion (more comprehensive context preference)
        2. Recursive refinement (iterate on the response)
        3. Parameter tuning (adjust execution mode, scale)

        Args:
            request: Original request
            prior_response: Previous insufficient response
            verification: Verification results with suggestions

        Returns:
            Refined DepartmentResponse with improved confidence
        """
        logger.info(
            f"Refining response for task {request.task_id} "
            f"(prior confidence: {prior_response.confidence.score:.2f})"
        )

        # Track refinement
        if request.task_id not in self._refinement_history:
            self._refinement_history[request.task_id] = []

        self._refinement_history[request.task_id].append({
            "iteration": len(self._refinement_history[request.task_id]),
            "prior_confidence": prior_response.confidence.score,
            "suggestions": verification.refinement_suggestions
        })

        # Apply refinement strategy based on suggestions
        refined_request = self._apply_refinement_strategy(
            request,
            verification.refinement_suggestions or {}
        )

        # Execute with refined parameters
        refined_response = await self.execute(refined_request)

        # Add refinement metadata
        refined_response.reasoning = refined_response.reasoning or {}
        refined_response.reasoning["refinement_history"] = self._refinement_history[request.task_id]

        return refined_response

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_confidence(
        self,
        spacetime: Spacetime,
        request: DepartmentRequest
    ) -> ConfidenceMetadata:
        """Extract confidence metadata from spacetime."""
        # Base confidence from spacetime
        score = spacetime.confidence if hasattr(spacetime, 'confidence') else 0.75

        # Extract justification from trace
        justification = []
        if spacetime.trace:
            if spacetime.trace.stage_durations:
                justification.append(
                    f"Weaving completed in {sum(spacetime.trace.stage_durations.values()):.0f}ms"
                )

        # Extract uncertainty sources
        uncertainty_sources = []
        if score < 0.8:
            uncertainty_sources.append("Moderate confidence - may benefit from refinement")
        if spacetime.response and len(spacetime.response) < 50:
            uncertainty_sources.append("Short response - limited context available")

        return ConfidenceMetadata.from_score(
            score,
            justification=justification,
            uncertainty_sources=uncertainty_sources
        )

    def _build_result(
        self,
        spacetime: Spacetime,
        task_type: str
    ) -> dict[str, Any]:
        """Build result dict based on task type."""
        if task_type == "retrieve_context":
            # Return only context, no response
            return {
                "context": spacetime.trace.threads_activated if spacetime.trace else [],
                "context_count": len(spacetime.trace.threads_activated) if spacetime.trace else 0
            }

        elif task_type == "weave_response":
            # Return full response + context
            return {
                "response": spacetime.response,
                "context": spacetime.trace.threads_activated if spacetime.trace else [],
                "query_text": spacetime.query_text
            }

        elif task_type == "expand_context":
            # Return expanded context
            return {
                "expanded_context": spacetime.trace.threads_activated if spacetime.trace else [],
                "expansion_depth": 1  # Default depth (retrieval_depth not in WeavingTrace)
            }

        else:
            # Fallback: return spacetime as-is
            return {"spacetime": spacetime}

    def _extract_reasoning(
        self,
        spacetime: Spacetime
    ) -> dict[str, Any]:
        """Extract reasoning from spacetime trace."""
        if not spacetime.trace:
            return {}

        return {
            "trace": {
                "stage_durations": spacetime.trace.stage_durations,
                "threads_activated": len(spacetime.trace.threads_activated) if spacetime.trace.threads_activated else 0,
                "context_shards_count": spacetime.trace.context_shards_count
            },
            "metadata": spacetime.metadata
        }

    def _extract_learning_signals(
        self,
        spacetime: Spacetime,
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Extract learning signals from spacetime."""
        return {
            "task_type": request.task_type,
            "outcome": "success",
            "confidence_predicted": spacetime.confidence if hasattr(spacetime, 'confidence') else 0.75,
            "latency_ms": sum(spacetime.trace.stage_durations.values()) if spacetime.trace else 0,
            "threads_used": len(spacetime.trace.threads_activated) if spacetime.trace and spacetime.trace.threads_activated else 0
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        spacetime: Spacetime
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_query": spacetime.query_text,
            "last_response": spacetime.response[:100] if spacetime.response else "",
            "last_confidence": spacetime.confidence if hasattr(spacetime, 'confidence') else 0.0,
            "interaction_count": 1  # Would increment in real session tracking
        }

    def _apply_refinement_strategy(
        self,
        request: DepartmentRequest,
        suggestions: dict[str, Any]
    ) -> DepartmentRequest:
        """Apply refinement suggestions to create a refined request."""
        # Copy request
        refined = DepartmentRequest(
            task_id=request.task_id + "_refined",
            task_type=request.task_type,
            parameters=request.parameters.copy(),
            confidence_expected=request.confidence_expected,
            context_preference=request.context_preference,
            privacy_level=request.privacy_level,
            session_id=request.session_id,
            parent_task_id=request.task_id,
            timeout_ms=request.timeout_ms
        )

        # Apply suggestions
        if suggestions.get("expand_context"):
            # Upgrade context preference
            if refined.context_preference == "minimal":
                refined.context_preference = "adaptive"
            elif refined.context_preference == "adaptive":
                refined.context_preference = "comprehensive"

        if suggestions.get("regenerate"):
            # Could adjust parameters (temperature, etc.) if supported
            pass

        return refined

    def _track_task_performance(
        self,
        task_type: str,
        latency_ms: float,
        confidence: float
    ) -> None:
        """Track performance by task type."""
        if task_type not in self._task_performance:
            self._task_performance[task_type] = {
                "count": 0,
                "total_latency": 0.0,
                "total_confidence": 0.0
            }

        stats = self._task_performance[task_type]
        stats["count"] += 1
        stats["total_latency"] += latency_ms
        stats["total_confidence"] += confidence
