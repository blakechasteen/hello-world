"""
Proto Agentic Bridge
====================

Wrapper around HoloLoom's AgenticOrchestrator for Proto.

Provides simplified interface for Proto's reasoning needs with graceful
fallback when HoloLoom components are unavailable.

Responsibilities:
    - Map Proto reasoning modes to HoloLoom modes
    - Execute agentic reasoning via HoloLoom
    - Handle errors and fallbacks
    - Track performance metrics

Status: Production ready (2025-12-02)
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import HoloLoom types
try:
    from hololoom.agentic.core import AgenticOrchestrator, ReasoningMode
    from hololoom.protocols.types import Query
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    AgenticOrchestrator = None
    ReasoningMode = None
    Query = None


logger = logging.getLogger("proto.agentic_bridge")


class ProtoReasoningMode(Enum):
    """Proto reasoning modes (maps to HoloLoom modes).

    Maps Proto's simplified reasoning modes to HoloLoom's 4 modes:
        - QUICK → DIRECT: Single-pass answer
        - CAREFUL → VERIFY: Answer + verification
        - EXPLORE → RESEARCH: Multi-query exploration
        - PLAN → PLAN_EXECUTE: Goal decomposition
    """
    QUICK = "quick"        # -> DIRECT
    CAREFUL = "careful"    # -> VERIFY
    EXPLORE = "explore"    # -> RESEARCH
    PLAN = "plan"          # -> PLAN_EXECUTE


@dataclass
class AgenticBridgeResult:
    """Result from agentic reasoning.

    Attributes:
        response: Generated response text
        confidence: Confidence score (0.0-1.0)
        steps_taken: Number of reasoning steps
        duration_ms: Execution time in milliseconds
        mode_used: Reasoning mode used
        metadata: Additional metadata
        error: Error message if execution failed
    """
    response: str = ""
    confidence: float = 0.0
    steps_taken: int = 0
    duration_ms: float = 0.0
    mode_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AgenticBridge:
    """
    Bridge to HoloLoom's AgenticOrchestrator.

    Simplifies agentic reasoning for Proto's use cases.

    Features:
        - Maps Proto modes to HoloLoom modes
        - Executes multi-query reasoning
        - Graceful fallback mode
        - Metrics tracking

    Usage:
        bridge = AgenticBridge(agentic_orchestrator)
        result = await bridge.reason(
            "Explain this code",
            mode=ProtoReasoningMode.CAREFUL
        )

    Graceful Degradation:
        When HoloLoom not available, returns basic fallback response
        with mode_used, confidence=0.5, and metadata["fallback"]=True
    """

    def __init__(self, orchestrator: Optional['AgenticOrchestrator'] = None):
        """Initialize agentic bridge.

        Args:
            orchestrator: HoloLoom AgenticOrchestrator instance (optional)
        """
        self._orchestrator = orchestrator

        if orchestrator and AGENTIC_AVAILABLE:
            logger.info("AgenticBridge initialized with HoloLoom orchestrator")
        else:
            logger.warning(
                "AgenticBridge: No orchestrator available, using fallback mode"
            )

    @property
    def is_available(self) -> bool:
        """Check if HoloLoom agentic reasoning is available.

        Returns:
            True if both orchestrator and HoloLoom agentic are available
        """
        return self._orchestrator is not None and AGENTIC_AVAILABLE

    async def reason(
        self,
        query: str,
        mode: ProtoReasoningMode = ProtoReasoningMode.CAREFUL,
        max_steps: int = 5,
        confidence_threshold: float = 0.75
    ) -> AgenticBridgeResult:
        """Execute agentic reasoning.

        Args:
            query: The question or request
            mode: Reasoning mode (QUICK/CAREFUL/EXPLORE/PLAN)
            max_steps: Maximum reasoning steps
            confidence_threshold: Target confidence threshold

        Returns:
            AgenticBridgeResult with response and metadata
        """
        start = time.time()

        if not self.is_available:
            return self._fallback_reason(query, mode)

        try:
            # Map Proto mode to HoloLoom mode
            holo_mode = self._map_mode(mode)

            # Execute via HoloLoom
            result = await self._orchestrator.reason(
                Query(text=query),
                mode=holo_mode,
                max_steps=max_steps,
                confidence_threshold=confidence_threshold
            )

            duration = (time.time() - start) * 1000

            # Extract response from result
            response_text = ""
            if hasattr(result, 'spacetime') and hasattr(result.spacetime, 'metadata'):
                response_text = result.spacetime.metadata.get("response", str(result.spacetime))
            elif hasattr(result, 'response'):
                response_text = result.response
            else:
                response_text = str(result)

            return AgenticBridgeResult(
                response=response_text,
                confidence=result.spacetime.confidence if hasattr(result, 'spacetime') else 0.8,
                steps_taken=result.total_queries if hasattr(result, 'total_queries') else 1,
                duration_ms=duration,
                mode_used=mode.value,
                metadata={
                    "hololoom_mode": holo_mode.value if hasattr(holo_mode, 'value') else str(holo_mode),
                    "verification": (
                        result.verification._asdict()
                        if hasattr(result, 'verification') and result.verification
                        else None
                    )
                }
            )

        except Exception as e:
            logger.error(f"Agentic reasoning failed: {e}", exc_info=True)
            duration = (time.time() - start) * 1000
            return AgenticBridgeResult(
                response=f"Error during reasoning: {str(e)}",
                confidence=0.0,
                duration_ms=duration,
                mode_used=mode.value,
                error=str(e)
            )

    def _map_mode(self, mode: ProtoReasoningMode) -> 'ReasoningMode':
        """Map Proto mode to HoloLoom mode.

        Args:
            mode: Proto reasoning mode

        Returns:
            Corresponding HoloLoom ReasoningMode
        """
        mapping = {
            ProtoReasoningMode.QUICK: ReasoningMode.DIRECT,
            ProtoReasoningMode.CAREFUL: ReasoningMode.VERIFY,
            ProtoReasoningMode.EXPLORE: ReasoningMode.RESEARCH,
            ProtoReasoningMode.PLAN: ReasoningMode.PLAN_EXECUTE,
        }
        return mapping.get(mode, ReasoningMode.VERIFY)

    def _fallback_reason(
        self,
        query: str,
        mode: ProtoReasoningMode
    ) -> AgenticBridgeResult:
        """Fallback when HoloLoom not available.

        Returns a basic response indicating fallback mode was used.

        Args:
            query: User query
            mode: Requested reasoning mode

        Returns:
            AgenticBridgeResult with fallback response
        """
        return AgenticBridgeResult(
            response=(
                f"[Fallback Mode] Processing request (HoloLoom unavailable):\n"
                f"Query: {query[:100]}..."
            ),
            confidence=0.5,
            steps_taken=1,
            mode_used=mode.value,
            metadata={"fallback": True}
        )
