"""
BaseLoom - Common Loom Functionality with Dreaming Mixin
==========================================================

**Purpose**: Provides base implementation for Loom protocol including
dreaming capabilities, insight integration, and Thompson Sampling priors.

**Date**: December 2025
**Phase**: Weave House Implementation

**Core Components**:
- DreamingMixin: Adds dreaming capabilities to any class
- BaseLoom: Abstract base implementing common Loom functionality

**Philosophy**:
"Looms weave independently during waking. They dream together during rest.
Math bleeds freely. Corrections always propagate. Patterns preserve diversity."

Author: HoloLoom Architecture Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import random
from uuid import uuid4

from HoloLoom.loom.protocol import (
    Loom,
    DreamInsight,
    InsightType,
    MathInsight,
    PatternInsight,
    CorrectionInsight,
    DiscoveryInsight,
)
from HoloLoom.fabric.fabric import Fabric, Tension
from HoloLoom.protocols.department import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceMetadata,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dreaming Mixin
# ============================================================================

class DreamingMixin:
    """
    Mixin adding dreaming capabilities to any class.

    Provides:
    - Dream buffer for accumulating insights
    - Dream receive for accepting insights from other looms
    - Dream share for returning accumulated insights
    - Insight integration logic

    Bleed Rates (from plan):
    - MATH: 30% (Warp Space insights are universal structure)
    - PATTERN: 20% (preserve diversity)
    - CORRECTION: 100% (errors always propagate)
    - DISCOVERY: high-confidence only (if confidence >= 0.8)

    Usage:
        class MyLoom(DreamingMixin, BaseLoom):
            pass
    """

    def __init__(self):
        """Initialize dreaming state."""
        # Insights waiting to be shared
        self._dream_buffer: List[DreamInsight] = []

        # Insights received from other looms
        self._received_insights: List[DreamInsight] = []

        # Integration settings
        self._insight_integration_rate = 0.5
        self._math_integration_rate = 0.8
        self._pattern_integration_rate = 0.3
        self._discovery_confidence_threshold = 0.7

        # Statistics
        self._insights_shared = 0
        self._insights_received = 0
        self._insights_integrated = 0

    async def dream_share(self) -> List[DreamInsight]:
        """
        Share accumulated insights for collective dreaming.

        Returns all shareable insights in the buffer, then clears it.

        Returns:
            List of insights to share with other looms
        """
        # Filter to shareable insights
        shareable = [i for i in self._dream_buffer if i.shareable]

        # Update stats
        self._insights_shared += len(shareable)

        # Clear buffer
        self._dream_buffer.clear()

        logger.debug(f"Sharing {len(shareable)} insights from {getattr(self, 'perspective', 'unknown')}")
        return shareable

    async def dream_receive(self, insight: DreamInsight) -> None:
        """
        Receive and potentially integrate an insight.

        Args:
            insight: Insight from another loom
        """
        self._received_insights.append(insight)
        self._insights_received += 1

        # Decide whether to integrate
        if await self._should_integrate(insight):
            await self._integrate_insight(insight)
            self._insights_integrated += 1

    async def _should_integrate(self, insight: DreamInsight) -> bool:
        """
        Decide whether to integrate an insight.

        Integration rules:
        - MATH: High probability (80%)
        - CORRECTION: Always integrate
        - PATTERN: Based on relevance to this loom's perspective
        - DISCOVERY: Only if high confidence

        Args:
            insight: Insight to consider

        Returns:
            True if should integrate, False otherwise
        """
        # Corrections always integrate
        if insight.insight_type == InsightType.CORRECTION:
            return True

        # Math insights integrate with high probability
        if insight.insight_type == InsightType.MATH:
            return random.random() < self._math_integration_rate

        # Patterns integrate based on relevance
        if insight.insight_type == InsightType.PATTERN:
            return await self._is_relevant_pattern(insight)

        # Discoveries integrate only if high confidence
        if insight.insight_type == InsightType.DISCOVERY:
            return insight.confidence >= self._discovery_confidence_threshold

        return False

    async def _is_relevant_pattern(self, insight: DreamInsight) -> bool:
        """
        Check if pattern is relevant to this loom's perspective.

        Override in subclasses for perspective-specific logic.

        Args:
            insight: Pattern insight to evaluate

        Returns:
            True if relevant to this loom
        """
        # Default: integrate patterns with probability
        return random.random() < self._pattern_integration_rate

    async def _integrate_insight(self, insight: DreamInsight) -> None:
        """
        Actually integrate the insight into this loom's priors.

        Dispatches to type-specific integration methods.

        Args:
            insight: Insight to integrate
        """
        if insight.insight_type == InsightType.MATH:
            await self._update_warp_priors(insight)

        elif insight.insight_type == InsightType.CORRECTION:
            await self._apply_correction(insight)

        elif insight.insight_type == InsightType.PATTERN:
            await self._update_pattern_priors(insight)

        elif insight.insight_type == InsightType.DISCOVERY:
            await self._add_discovery(insight)

        logger.debug(
            f"Integrated {insight.insight_type.value} insight from "
            f"{insight.source_loom} into {getattr(self, 'perspective', 'unknown')}"
        )

    async def _update_warp_priors(self, insight: DreamInsight) -> None:
        """
        Update Warp Space mathematical priors.

        Override in subclasses for custom Warp Space integration.

        Args:
            insight: Math insight to integrate
        """
        # Default: store in metadata for later use
        if not hasattr(self, '_warp_priors'):
            self._warp_priors = []
        self._warp_priors.append(insight.content)

    async def _apply_correction(self, insight: DreamInsight) -> None:
        """
        Apply correction to loom's knowledge.

        Override in subclasses for custom correction handling.

        Args:
            insight: Correction insight to apply
        """
        # Default: store correction for reference
        if not hasattr(self, '_corrections'):
            self._corrections = []
        self._corrections.append({
            "source": insight.source_loom,
            "content": insight.content,
            "timestamp": insight.timestamp.isoformat(),
        })

    async def _update_pattern_priors(self, insight: DreamInsight) -> None:
        """
        Update Thompson Sampling pattern priors.

        Override in subclasses for custom pattern integration.

        Args:
            insight: Pattern insight to integrate
        """
        # Default: store pattern for later use
        if not hasattr(self, '_learned_patterns'):
            self._learned_patterns = []
        self._learned_patterns.append(insight.content)

    async def _add_discovery(self, insight: DreamInsight) -> None:
        """
        Add discovery to knowledge graph.

        Override in subclasses for custom discovery handling.

        Args:
            insight: Discovery insight to add
        """
        # Default: store discovery
        if not hasattr(self, '_discoveries'):
            self._discoveries = []
        self._discoveries.append({
            "source": insight.source_loom,
            "content": insight.content,
            "confidence": insight.confidence,
            "timestamp": insight.timestamp.isoformat(),
        })

    def log_insight(self, insight: DreamInsight) -> None:
        """
        Log an insight for sharing during next dream cycle.

        Args:
            insight: Insight to share
        """
        self._dream_buffer.append(insight)
        logger.debug(
            f"Logged {insight.insight_type.value} insight for sharing "
            f"(confidence: {insight.confidence:.2f})"
        )

    def log_math_insight(self, content: Any, confidence: float = 0.8) -> None:
        """Convenience method to log a math insight."""
        self.log_insight(MathInsight(
            source_loom=getattr(self, 'perspective', 'unknown'),
            content=content,
            confidence=confidence,
        ))

    def log_pattern_insight(self, content: Any, confidence: float = 0.7) -> None:
        """Convenience method to log a pattern insight."""
        self.log_insight(PatternInsight(
            source_loom=getattr(self, 'perspective', 'unknown'),
            content=content,
            confidence=confidence,
        ))

    def log_correction_insight(self, content: Any, confidence: float = 0.9) -> None:
        """Convenience method to log a correction insight."""
        self.log_insight(CorrectionInsight(
            source_loom=getattr(self, 'perspective', 'unknown'),
            content=content,
            confidence=confidence,
        ))

    def log_discovery_insight(self, content: Any, confidence: float = 0.85) -> None:
        """Convenience method to log a discovery insight."""
        self.log_insight(DiscoveryInsight(
            source_loom=getattr(self, 'perspective', 'unknown'),
            content=content,
            confidence=confidence,
        ))

    def get_dreaming_stats(self) -> Dict[str, Any]:
        """Get dreaming statistics."""
        return {
            "insights_shared": self._insights_shared,
            "insights_received": self._insights_received,
            "insights_integrated": self._insights_integrated,
            "buffer_size": len(self._dream_buffer),
            "pending_received": len(self._received_insights),
        }


# ============================================================================
# Base Loom
# ============================================================================

class BaseLoom(DreamingMixin, ABC):
    """
    Abstract base class implementing common Loom functionality.

    Provides:
    - DreamingMixin integration
    - Default implementations for Department methods
    - Thompson Sampling priors management
    - Metrics tracking
    - Common utilities

    Subclasses must implement:
    - perspective (property): Return loom's perspective identifier
    - blind_spots (property): Return loom's blind spots
    - execute(request): Core domain logic
    - admits_ignorance(query): What I don't know about query
    - falsifiable_by(claim): What would change my mind

    Usage:
        class RecallLoom(BaseLoom):
            @property
            def perspective(self) -> str:
                return "recall"

            @property
            def blind_spots(self) -> List[str]:
                return ["novel connections", "future predictions"]

            async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
                # ... implement recall logic
                pass
    """

    def __init__(
        self,
        name: Optional[str] = None,
        enable_learning: bool = True,
        enable_verification: bool = True,
    ):
        """
        Initialize base loom.

        Args:
            name: Loom name (defaults to perspective)
            enable_learning: Enable Thompson Sampling updates
            enable_verification: Enable verification checks
        """
        # Initialize dreaming
        DreamingMixin.__init__(self)

        # Configuration
        self._name = name
        self._enable_learning = enable_learning
        self._enable_verification = enable_verification

        # Thompson Sampling priors (α, β for Beta distribution)
        self._tool_priors: Dict[str, Tuple[float, float]] = {}

        # Metrics
        self._tasks_executed = 0
        self._verification_passes = 0
        self._verification_fails = 0
        self._total_confidence = 0.0
        self._total_latency_ms = 0.0

        # Cache for common computations
        self._cache: Dict[str, Any] = {}

    # ====== Abstract Properties (must implement) ======

    @property
    @abstractmethod
    def perspective(self) -> str:
        """Return this loom's perspective identifier."""
        ...

    @property
    @abstractmethod
    def blind_spots(self) -> List[str]:
        """Return this loom's blind spots."""
        ...

    @abstractmethod
    def admits_ignorance(self, query: str) -> List[str]:
        """Return what this loom doesn't know about the query."""
        ...

    @abstractmethod
    def falsifiable_by(self, claim: str) -> List[str]:
        """Return what evidence would change this claim."""
        ...

    # ====== Abstract Methods (must implement) ======

    @abstractmethod
    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute the core loom logic."""
        ...

    # ====== Default Department Implementations ======

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Default verification implementation.

        Override for custom verification logic.

        Args:
            response: Response to verify

        Returns:
            Verification result
        """
        from HoloLoom.protocols.department import VerificationCheck, VerificationStatus

        checks = []

        # Check 1: Confidence threshold
        conf_passed = response.confidence.score >= 0.5
        checks.append(VerificationCheck(
            name="confidence_threshold",
            status=VerificationStatus.PASSED if conf_passed else VerificationStatus.FAILED,
            reason=f"Confidence: {response.confidence.score:.2f}",
            score=response.confidence.score,
        ))

        # Check 2: Result not empty
        result_valid = response.result is not None and response.result != ""
        checks.append(VerificationCheck(
            name="result_not_empty",
            status=VerificationStatus.PASSED if result_valid else VerificationStatus.FAILED,
            reason="Result is valid" if result_valid else "Result is empty",
            score=1.0 if result_valid else 0.0,
        ))

        # Overall
        overall_passed = all(c.status == VerificationStatus.PASSED for c in checks)
        overall_score = sum(c.score for c in checks) / len(checks)

        if overall_passed:
            self._verification_passes += 1
        else:
            self._verification_fails += 1

        return VerificationResult(
            verified=overall_passed,
            checks=checks,
            overall_score=overall_score,
            summary="All checks passed" if overall_passed else "Some checks failed",
            confidence=overall_score,
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Default refinement implementation.

        Override for custom refinement logic.

        Args:
            response: Response to refine

        Returns:
            Refined response (or original if refinement not needed)
        """
        # Default: return as-is if confidence is acceptable
        if response.confidence.score >= 0.7:
            return response

        # Otherwise, try to improve by re-executing with more context
        # (Subclasses can override for sophisticated refinement)
        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Update Thompson Sampling priors based on feedback.

        Args:
            feedback: Learning signal with outcome, confidence, etc.
        """
        if not self._enable_learning:
            return

        outcome = feedback.get("outcome", "unknown")
        confidence = feedback.get("confidence", 0.5)
        tool_used = feedback.get("tool_used", "answer")

        # Initialize priors if needed
        if tool_used not in self._tool_priors:
            self._tool_priors[tool_used] = (1.0, 1.0)  # Uniform prior

        alpha, beta = self._tool_priors[tool_used]

        # Update based on outcome
        if outcome == "success" or confidence >= 0.75:
            # Success: increase alpha (more successes)
            alpha += confidence
        else:
            # Failure: increase beta (more failures)
            beta += (1 - confidence)

        self._tool_priors[tool_used] = (alpha, beta)

        logger.debug(f"Updated priors for {tool_used}: α={alpha:.2f}, β={beta:.2f}")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Report loom capabilities.

        Returns:
            Capability dictionary
        """
        return {
            "name": self._name or self.perspective,
            "perspective": self.perspective,
            "blind_spots": self.blind_spots,
            "supported_tasks": ["weave", "verify", "refine"],
            "enable_learning": self._enable_learning,
            "enable_verification": self._enable_verification,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Metrics dictionary
        """
        avg_confidence = (
            self._total_confidence / self._tasks_executed
            if self._tasks_executed > 0 else 0.0
        )
        avg_latency = (
            self._total_latency_ms / self._tasks_executed
            if self._tasks_executed > 0 else 0.0
        )
        verification_rate = (
            self._verification_passes / (self._verification_passes + self._verification_fails)
            if (self._verification_passes + self._verification_fails) > 0 else 0.0
        )

        return {
            "tasks_executed": self._tasks_executed,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "verification_pass_rate": verification_rate,
            "thompson_priors": {
                tool: f"α={a:.2f}, β={b:.2f}"
                for tool, (a, b) in self._tool_priors.items()
            },
            **self.get_dreaming_stats(),
        }

    async def health_check(self) -> bool:
        """
        Check if loom is healthy.

        Returns:
            True if healthy
        """
        # Default: always healthy
        # Override for custom health checks
        return True

    # ====== Utility Methods ======

    def get_expected_reward(self, tool: str) -> float:
        """
        Get expected reward for a tool using Thompson Sampling.

        Args:
            tool: Tool name

        Returns:
            Expected reward (0-1)
        """
        if tool not in self._tool_priors:
            return 0.5  # Uniform prior

        alpha, beta = self._tool_priors[tool]
        return alpha / (alpha + beta)

    def sample_tool(self, tools: List[str]) -> str:
        """
        Sample a tool using Thompson Sampling.

        Args:
            tools: List of tool names

        Returns:
            Sampled tool name
        """
        samples = {}
        for tool in tools:
            if tool in self._tool_priors:
                alpha, beta = self._tool_priors[tool]
            else:
                alpha, beta = 1.0, 1.0

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta)
            samples[tool] = sample

        # Return tool with highest sample
        return max(samples, key=samples.get)

    def _create_fabric(
        self,
        query: str,
        response: str,
        tool_used: str,
        confidence: float,
        duration_ms: float,
        **kwargs
    ) -> Fabric:
        """
        Create a Fabric output with this loom's perspective.

        Args:
            query: Original query
            response: Generated response
            tool_used: Tool that was used
            confidence: Confidence score
            duration_ms: Processing duration
            **kwargs: Additional fabric fields

        Returns:
            Fabric with perspective and epistemic fields
        """
        from HoloLoom.fabric.spacetime import WeavingTrace

        trace = WeavingTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=duration_ms,
            policy_adapter=f"{self.perspective}_adapter",
            tool_selected=tool_used,
            tool_confidence=confidence,
        )

        return Fabric(
            query_text=query,
            response=response,
            tool_used=tool_used,
            confidence=confidence,
            trace=trace,
            perspective=self.perspective,
            blind_spots=self.blind_spots,
            epistemic_confidence=kwargs.get("epistemic_confidence", confidence * 0.9),
            falsifiable_by=kwargs.get("falsifiable_by", []),
            admits_ignorance=kwargs.get("admits_ignorance", []),
            metadata=kwargs.get("metadata", {}),
        )

    def _update_metrics(
        self,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """
        Update internal metrics after task execution.

        Args:
            confidence: Task confidence
            latency_ms: Task latency
        """
        self._tasks_executed += 1
        self._total_confidence += confidence
        self._total_latency_ms += latency_ms


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DreamingMixin',
    'BaseLoom',
]
