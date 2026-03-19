"""
Loom Protocol - A Coherent Way of Seeing
==========================================

**Purpose**: Defines the Loom protocol - a specialized Department with perspective,
epistemic awareness, and dreaming capabilities for the Multi-Loom Architecture.

**Date**: December 2025
**Phase**: Weave House Implementation

**Core Insight**: A Loom IS a Department (inherits 7 methods) but adds:
- Perspective identity (how this loom "sees" the world)
- Blind spots (what this loom cannot see)
- Epistemic awareness (knows what it doesn't know)
- Dreaming interface (collective insight sharing during consolidation)

**Philosophy**:
"Each loom is a coherent way of seeing. Different looms see different truths.
Together, they weave a fabric richer than any single perspective."

The Five Core Looms:
- RECALL (The Librarian): Dense retrieval, memory-grounded
- REASON (The Logician): Causal inference, logic-grounded
- REACH (The Artist): Lateral association, creativity-grounded
- REFLECT (The Auditor): Consistency checking, verification-grounded
- REFUSE (The Skeptic): Adversarial probing, breaking-grounded

Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# Import Department protocol
from hololoom.protocols.department import (
    Department,
)

# ============================================================================
# Dream Insight Types
# ============================================================================

class InsightType(Enum):
    """Types of insights that can be shared during collective dreaming."""

    MATH = "math"              # Mathematical patterns, tensor field insights
    PATTERN = "pattern"        # Hot patterns, tool selection patterns
    CORRECTION = "correction"  # Error corrections, reliability updates
    DISCOVERY = "discovery"    # Novel findings, new connections


@dataclass
class DreamInsight:
    """
    An insight to share during collective dreaming.

    Insights flow between looms during consolidation, enabling
    collective learning while preserving perspective diversity.

    Example:
        insight = DreamInsight(
            source_loom="recall",
            insight_type=InsightType.CORRECTION,
            content={"node_id": "xyz", "correction": "This claim was wrong"},
            confidence=0.95
        )
    """
    source_loom: str                    # Which loom generated this insight
    insight_type: InsightType           # MATH/PATTERN/CORRECTION/DISCOVERY
    content: Any                         # The actual insight content
    confidence: float                    # How confident (0-1)
    shareable: bool = True               # Can be shared with other looms?
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_loom": self.source_loom,
            "insight_type": self.insight_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "shareable": self.shareable,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DreamInsight':
        """Deserialize from dictionary."""
        return cls(
            source_loom=data["source_loom"],
            insight_type=InsightType(data["insight_type"]),
            content=data["content"],
            confidence=data["confidence"],
            shareable=data.get("shareable", True),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Specialized Insight Types
# ============================================================================

@dataclass
class MathInsight(DreamInsight):
    """
    Warp Space mathematical insights - bleed freely (30%).

    These represent fundamental mathematical patterns discovered
    in the tensor field that benefit all looms.

    Examples:
    - Tensor field patterns discovered
    - Eigenvalue distributions
    - Topological features
    - Dimensional relationships
    """

    def __init__(
        self,
        source_loom: str,
        content: Any,
        confidence: float,
        **kwargs
    ):
        super().__init__(
            source_loom=source_loom,
            insight_type=InsightType.MATH,
            content=content,
            confidence=confidence,
            **kwargs
        )


@dataclass
class PatternInsight(DreamInsight):
    """
    Hot patterns - bleed conservatively (20%).

    Preserve diversity by limiting pattern sharing.
    Each loom should maintain its own style.

    Examples:
    - Query patterns that worked
    - Tool selection patterns
    - Retrieval strategies
    """

    def __init__(
        self,
        source_loom: str,
        content: Any,
        confidence: float,
        **kwargs
    ):
        super().__init__(
            source_loom=source_loom,
            insight_type=InsightType.PATTERN,
            content=content,
            confidence=confidence,
            **kwargs
        )


@dataclass
class CorrectionInsight(DreamInsight):
    """
    Error corrections - always bleed (100%).

    Corrections propagate immediately because error fixes
    are universal - no loom wants to repeat mistakes.

    Examples:
    - "This claim was wrong"
    - "This source is unreliable"
    - "This reasoning pattern failed"
    """

    def __init__(
        self,
        source_loom: str,
        content: Any,
        confidence: float,
        **kwargs
    ):
        super().__init__(
            source_loom=source_loom,
            insight_type=InsightType.CORRECTION,
            content=content,
            confidence=confidence,
            shareable=True,  # Always shareable
            **kwargs
        )


@dataclass
class DiscoveryInsight(DreamInsight):
    """
    Novel discoveries - bleed if confident (high confidence only).

    New findings share only if validated to prevent
    propagating unverified information.

    Examples:
    - New entity relationships
    - Unexpected connections
    - Contradiction resolutions
    """

    def __init__(
        self,
        source_loom: str,
        content: Any,
        confidence: float,
        **kwargs
    ):
        # Only shareable if high confidence
        shareable = confidence >= 0.8
        super().__init__(
            source_loom=source_loom,
            insight_type=InsightType.DISCOVERY,
            content=content,
            confidence=confidence,
            shareable=shareable,
            **kwargs
        )


# ============================================================================
# Loom Protocol
# ============================================================================

@runtime_checkable
class Loom(Department, Protocol):
    """
    A Loom is a coherent way of seeing.

    Extends Department with perspective, epistemic awareness, and dreaming.

    Inherited from Department (7 methods):
    - execute(request) -> response
    - verify(response) -> verification
    - refine(response) -> refined
    - update_strategy(feedback) -> None
    - get_capabilities() -> dict
    - get_metrics() -> dict
    - health_check() -> bool

    Added by Loom:
    - perspective: str (identity)
    - blind_spots: List[str] (what I can't see)
    - admits_ignorance(query) -> List[str] (what I know I don't know)
    - falsifiable_by(claim) -> List[str] (what would change my mind)
    - dream_receive(insight) -> None (receive during dreaming)
    - dream_share() -> List[DreamInsight] (share during dreaming)

    Example (minimal implementation):

        class MyLoom:
            perspective = "my_view"
            blind_spots = ["things I cannot see"]

            def admits_ignorance(self, query: str) -> List[str]:
                return ["I don't know about..."]

            def falsifiable_by(self, claim: str) -> List[str]:
                return ["Finding contradicting evidence"]

            async def dream_receive(self, insight: DreamInsight) -> None:
                self._received_insights.append(insight)

            async def dream_share(self) -> List[DreamInsight]:
                return self._dream_buffer
    """

    # ====== Identity ======

    @property
    def perspective(self) -> str:
        """
        The perspective this loom takes.

        Returns:
            Perspective identifier (e.g., "recall", "reason", "reach", etc.)
        """
        ...

    @property
    def blind_spots(self) -> list[str]:
        """
        What this perspective cannot see.

        Every perspective has blind spots. Being explicit about
        limitations enables complementary looms to fill gaps.

        Returns:
            List of blind spot descriptions

        Example:
            ["novel connections not in memory",
             "future predictions",
             "creative leaps"]
        """
        ...

    # ====== Epistemic Awareness ======

    def admits_ignorance(self, query: str) -> list[str]:
        """
        What I know I don't know about this query.

        Epistemic humility: explicitly acknowledge uncertainty.
        This enables other looms to help and prevents overconfidence.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps

        Example:
            ["Cannot verify claims about 2024 events",
             "No data on this specific company"]
        """
        ...

    def falsifiable_by(self, claim: str) -> list[str]:
        """
        What evidence would change my mind.

        Every claim should be falsifiable. This enables
        productive disagreement and auto-exploration.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers (evidence that would change this claim)

        Example:
            ["Finding a contradicting primary source",
             "Newer data showing different trend"]
        """
        ...

    # ====== Dreaming Interface ======

    async def dream_receive(self, insight: DreamInsight) -> None:
        """
        Receive insight during collective dreaming.

        Called by DreamOrchestrator to deliver insights from
        other looms. Implementation decides whether to integrate.

        Args:
            insight: Insight from another loom
        """
        ...

    async def dream_share(self) -> list[DreamInsight]:
        """
        Share insights for collective dreaming.

        Called by DreamOrchestrator to collect shareable insights.
        Return accumulated insights, then clear the buffer.

        Returns:
            List of insights to share with other looms
        """
        ...


# ============================================================================
# Type Aliases
# ============================================================================

# Factory for creating looms
LoomFactory = type  # Returns Loom instances

# Perspective identifiers
RECALL = "recall"    # The Librarian
REASON = "reason"    # The Logician
REACH = "reach"      # The Artist
REFLECT = "reflect"  # The Auditor
REFUSE = "refuse"    # The Skeptic
COLLECTIVE = "collective" # The Unified Whole


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core Protocol
    'Loom',

    # Insight Types
    'InsightType',
    'DreamInsight',
    'MathInsight',
    'PatternInsight',
    'CorrectionInsight',
    'DiscoveryInsight',

    # Type Aliases
    'LoomFactory',

    # Perspective Constants
    'RECALL',
    'REASON',
    'REACH',
    'REFLECT',
    'REFUSE',
    'COLLECTIVE',
]
