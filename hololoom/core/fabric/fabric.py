"""
Fabric - Spacetime with Perspective and Epistemic Metadata
===========================================================

**Purpose**: Extends Spacetime to include perspective identity, epistemic
awareness, and inter-loom tension tracking for the Multi-Loom Architecture.

**Date**: December 2025
**Phase**: Weave House Implementation

**Core Insight**: Fabric is Spacetime with perspective and epistemic metadata.
It captures not just WHAT was produced and HOW, but also:
- WHO (which loom/perspective)
- WHAT I DON'T KNOW (admits_ignorance)
- WHAT WOULD CHANGE MY MIND (falsifiable_by)
- WHERE I DISAGREE (tensions_with)
- WHERE I AGREE (agreement_with)

**Philosophy**:
"The fabric remembers its weaver. Each thread carries provenance,
each pattern carries perspective, each tension carries truth."

Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import logging
from pathlib import Path

from hololoom.core.fabric.spacetime import Spacetime, WeavingTrace

logger = logging.getLogger(__name__)


# ============================================================================
# Tension - Productive Disagreement Between Looms
# ============================================================================

@dataclass
class Tension:
    """
    A productive disagreement between looms.

    Tensions are not errors - they're opportunities for deeper understanding.
    When looms disagree, we explore the disagreement rather than averaging away.

    Example:
        tension = Tension(
            loom_a="recall",
            loom_b="reason",
            claim_a="Memory shows X happened",
            claim_b="Logic suggests Y should happen",
            severity=0.7,
            exploration_query="What evidence supports X vs Y?"
        )
    """
    loom_a: str                               # First disagreeing loom
    loom_b: str                               # Second disagreeing loom
    claim_a: str                              # What loom_a says
    claim_b: str                              # What loom_b says
    severity: float                           # 0.0-1.0, how significant?
    exploration_query: Optional[str] = None   # Query to explore this tension
    resolution: Optional[str] = None          # How was it resolved (if at all)?
    is_irreducible: bool = False              # True if cannot be resolved

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "loom_a": self.loom_a,
            "loom_b": self.loom_b,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "severity": self.severity,
            "exploration_query": self.exploration_query,
            "resolution": self.resolution,
            "is_irreducible": self.is_irreducible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tension':
        """Deserialize from dictionary."""
        return cls(
            loom_a=data["loom_a"],
            loom_b=data["loom_b"],
            claim_a=data["claim_a"],
            claim_b=data["claim_b"],
            severity=data["severity"],
            exploration_query=data.get("exploration_query"),
            resolution=data.get("resolution"),
            is_irreducible=data.get("is_irreducible", False),
        )


# ============================================================================
# Resolution - Result of Tension Exploration
# ============================================================================

@dataclass
class Resolution:
    """
    Attempted resolution of a tension.

    After exploring a tension, we either:
    - Resolve it (looms agree on consensus)
    - Mark it irreducible (fundamental difference in perspective)

    Example:
        resolution = Resolution(
            resolved=True,
            consensus_claim="Evidence supports X with caveats",
            confidence=0.85,
            supporting_looms=["recall", "reflect"],
            dissenting_looms=[]
        )
    """
    resolved: bool                            # Did we resolve it?
    consensus_claim: Optional[str]            # What we agreed on (if resolved)
    confidence: float                         # How confident in resolution?
    supporting_looms: List[str] = field(default_factory=list)
    dissenting_looms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "resolved": self.resolved,
            "consensus_claim": self.consensus_claim,
            "confidence": self.confidence,
            "supporting_looms": self.supporting_looms,
            "dissenting_looms": self.dissenting_looms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resolution':
        """Deserialize from dictionary."""
        return cls(
            resolved=data["resolved"],
            consensus_claim=data.get("consensus_claim"),
            confidence=data["confidence"],
            supporting_looms=data.get("supporting_looms", []),
            dissenting_looms=data.get("dissenting_looms", []),
        )


# ============================================================================
# Fabric - Extended Spacetime
# ============================================================================

@dataclass
class Fabric(Spacetime):
    """
    Spacetime with perspective and epistemic metadata.

    Fabric extends Spacetime with:
    - Perspective identity (which loom made this)
    - Blind spots (what this perspective can't see)
    - Epistemic confidence (confidence in confidence)
    - Falsifiability (what would change my mind)
    - Admits ignorance (what I know I don't know)
    - Tensions (disagreements with other looms)
    - Agreement (alignment with other looms)

    This enables:
    - Multi-perspective synthesis
    - Explicit uncertainty tracking
    - Productive disagreement exploration
    - Collective verification

    Usage:
        fabric = Fabric(
            query_text="What is Thompson Sampling?",
            response="Thompson Sampling is...",
            tool_used="answer",
            confidence=0.87,
            trace=weaving_trace,
            # Perspective fields
            perspective="recall",
            blind_spots=["novel connections", "future predictions"],
            epistemic_confidence=0.75,
            falsifiable_by=["contradicting primary source"],
            admits_ignorance=["Cannot verify 2024 claims"]
        )
    """

    # ====== Inherited from Spacetime ======
    # query_text: str
    # response: str
    # tool_used: str
    # confidence: float
    # trace: WeavingTrace
    # metadata: Dict[str, Any]
    # context_summary: Optional[str]
    # sources_used: List[str]
    # created_at: datetime
    # quality_score: Optional[float]
    # user_feedback: Optional[str]

    # ====== Perspective Identity ======
    perspective: str = "unknown"              # Which loom made this?
    blind_spots: List[str] = field(default_factory=list)  # What I can't see

    # ====== Epistemic Awareness ======
    epistemic_confidence: float = 0.5         # Confidence in my confidence (0-1)
    falsifiable_by: List[str] = field(default_factory=list)  # What would change my mind?
    admits_ignorance: List[str] = field(default_factory=list)  # What I know I don't know

    # ====== Inter-Loom Dynamics ======
    tensions_with: Dict[str, str] = field(default_factory=dict)  # {loom: disagreement}
    agreement_with: Dict[str, float] = field(default_factory=dict)  # {loom: similarity}

    def __post_init__(self):
        """Initialize and validate fabric fields."""
        # Call parent post_init
        super().__post_init__()

        # Validate epistemic_confidence
        if not 0.0 <= self.epistemic_confidence <= 1.0:
            self.epistemic_confidence = max(0.0, min(1.0, self.epistemic_confidence))

        # Add perspective to metadata
        if 'perspective' not in self.metadata:
            self.metadata['perspective'] = self.perspective

        if 'epistemic_confidence' not in self.metadata:
            self.metadata['epistemic_confidence'] = self.epistemic_confidence

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dictionary.

        Extends Spacetime.to_dict() with perspective and epistemic fields.
        """
        # Get base dict from Spacetime
        base_dict = super().to_dict()

        # Add Fabric-specific fields
        base_dict.update({
            'perspective': self.perspective,
            'blind_spots': self.blind_spots,
            'epistemic_confidence': self.epistemic_confidence,
            'falsifiable_by': self.falsifiable_by,
            'admits_ignorance': self.admits_ignorance,
            'tensions_with': self.tensions_with,
            'agreement_with': self.agreement_with,
        })

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fabric':
        """
        Create Fabric from dictionary.

        Extends Spacetime.from_dict() with perspective and epistemic fields.
        """
        # Parse trace
        trace_data = data.get('trace', {})
        trace = WeavingTrace(
            start_time=datetime.fromisoformat(trace_data.get('start_time', datetime.now().isoformat())),
            end_time=datetime.fromisoformat(trace_data.get('end_time', datetime.now().isoformat())),
            duration_ms=trace_data.get('duration_ms', 0.0),
            stage_durations=trace_data.get('stage_durations', {}),
            motifs_detected=trace_data.get('motifs_detected', []),
            embedding_scales_used=trace_data.get('embedding_scales_used', []),
            spectral_features=trace_data.get('spectral_features'),
            threads_activated=trace_data.get('threads_activated', []),
            context_shards_count=trace_data.get('context_shards_count', 0),
            retrieval_mode=trace_data.get('retrieval_mode', 'unknown'),
            policy_adapter=trace_data.get('policy_adapter', 'unknown'),
            tool_selected=trace_data.get('tool_selected', 'unknown'),
            tool_confidence=trace_data.get('tool_confidence', 0.0),
            bandit_statistics=trace_data.get('bandit_statistics'),
            warp_operations=trace_data.get('warp_operations', []),
            tensor_field_stats=trace_data.get('tensor_field_stats'),
            errors=trace_data.get('errors', []),
            warnings=trace_data.get('warnings', [])
        )

        # Parse created_at
        created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()

        return cls(
            # Spacetime fields
            query_text=data['query_text'],
            response=data['response'],
            tool_used=data['tool_used'],
            confidence=data['confidence'],
            trace=trace,
            metadata=data.get('metadata', {}),
            context_summary=data.get('context_summary'),
            sources_used=data.get('sources_used', []),
            created_at=created_at,
            quality_score=data.get('quality_score'),
            user_feedback=data.get('user_feedback'),
            # Fabric fields
            perspective=data.get('perspective', 'unknown'),
            blind_spots=data.get('blind_spots', []),
            epistemic_confidence=data.get('epistemic_confidence', 0.5),
            falsifiable_by=data.get('falsifiable_by', []),
            admits_ignorance=data.get('admits_ignorance', []),
            tensions_with=data.get('tensions_with', {}),
            agreement_with=data.get('agreement_with', {}),
        )

    @classmethod
    def from_spacetime(
        cls,
        spacetime: Spacetime,
        perspective: str = "unknown",
        blind_spots: Optional[List[str]] = None,
        epistemic_confidence: float = 0.5,
        falsifiable_by: Optional[List[str]] = None,
        admits_ignorance: Optional[List[str]] = None
    ) -> 'Fabric':
        """
        Create Fabric from existing Spacetime.

        Upgrades a Spacetime to Fabric by adding perspective and epistemic fields.

        Args:
            spacetime: Existing Spacetime to upgrade
            perspective: Which loom made this
            blind_spots: What this perspective can't see
            epistemic_confidence: Confidence in confidence
            falsifiable_by: What would change my mind
            admits_ignorance: What I know I don't know

        Returns:
            Fabric with all fields populated
        """
        return cls(
            query_text=spacetime.query_text,
            response=spacetime.response,
            tool_used=spacetime.tool_used,
            confidence=spacetime.confidence,
            trace=spacetime.trace,
            metadata=spacetime.metadata.copy(),
            context_summary=spacetime.context_summary,
            sources_used=spacetime.sources_used.copy(),
            created_at=spacetime.created_at,
            quality_score=spacetime.quality_score,
            user_feedback=spacetime.user_feedback,
            perspective=perspective,
            blind_spots=blind_spots or [],
            epistemic_confidence=epistemic_confidence,
            falsifiable_by=falsifiable_by or [],
            admits_ignorance=admits_ignorance or [],
        )

    def has_significant_tensions(self, threshold: float = 0.3) -> bool:
        """
        Check if any tensions exceed severity threshold.

        Args:
            threshold: Minimum severity to consider significant

        Returns:
            True if any tension exceeds threshold
        """
        tensions = self.metadata.get("tensions", [])
        return any(t.get("severity", 0) > threshold for t in tensions)

    def add_tension(self, tension: Tension) -> None:
        """
        Add a tension to this fabric.

        Args:
            tension: Tension to add
        """
        self.tensions_with[tension.loom_b] = tension.claim_b

        # Also store full tension in metadata
        if "tensions" not in self.metadata:
            self.metadata["tensions"] = []
        self.metadata["tensions"].append(tension.to_dict())

    def add_agreement(self, loom_name: str, similarity: float) -> None:
        """
        Record agreement with another loom.

        Args:
            loom_name: Name of agreeing loom
            similarity: Similarity score (0-1)
        """
        self.agreement_with[loom_name] = similarity

    def summarize(self) -> Dict[str, Any]:
        """
        Get concise summary including perspective and epistemic fields.

        Returns:
            Dict with key metrics
        """
        base_summary = super().summarize()
        base_summary.update({
            'perspective': self.perspective,
            'blind_spots_count': len(self.blind_spots),
            'epistemic_confidence': f"{self.epistemic_confidence:.2f}",
            'tensions_count': len(self.tensions_with),
            'agreements_count': len(self.agreement_with),
            'admits_ignorance_count': len(self.admits_ignorance),
        })
        return base_summary

    def get_reflection_signal(self) -> Dict[str, Any]:
        """
        Extract reflection signal including epistemic metadata.

        Returns:
            Dict with reflection learning signals
        """
        base_signal = super().get_reflection_signal()
        base_signal.update({
            'perspective': self.perspective,
            'blind_spots': self.blind_spots,
            'epistemic_confidence': self.epistemic_confidence,
            'falsifiable_by': self.falsifiable_by,
            'admits_ignorance': self.admits_ignorance,
            'tensions': list(self.tensions_with.keys()),
            'agreements': self.agreement_with,
        })
        return base_signal


# ============================================================================
# FabricCollection Extension
# ============================================================================

class MultiFabricCollection:
    """
    Collection of Fabrics from multiple looms.

    Enables analysis across perspectives:
    - Find consensus and disagreement
    - Track epistemic confidence
    - Analyze perspective coverage
    """

    def __init__(self, fabrics: Optional[List[Fabric]] = None):
        """
        Initialize multi-fabric collection.

        Args:
            fabrics: Initial list of Fabric instances
        """
        self.fabrics: List[Fabric] = fabrics or []
        self.logger = logging.getLogger(__name__)

    def add(self, fabric: Fabric) -> None:
        """Add a fabric to the collection."""
        self.fabrics.append(fabric)

    def by_perspective(self, perspective: str) -> List[Fabric]:
        """Get all fabrics from a specific perspective."""
        return [f for f in self.fabrics if f.perspective == perspective]

    def perspectives_present(self) -> List[str]:
        """Get list of unique perspectives in collection."""
        return list(set(f.perspective for f in self.fabrics))

    def find_consensus(self) -> Dict[str, Any]:
        """
        Find areas of consensus across perspectives.

        Returns:
            Dict with consensus information
        """
        if len(self.fabrics) < 2:
            return {"status": "insufficient_fabrics", "count": len(self.fabrics)}

        # Find common agreement patterns
        all_agreements: Dict[str, List[float]] = {}
        for fabric in self.fabrics:
            for loom, score in fabric.agreement_with.items():
                if loom not in all_agreements:
                    all_agreements[loom] = []
                all_agreements[loom].append(score)

        # Calculate mean agreement per loom
        consensus_scores = {
            loom: sum(scores) / len(scores)
            for loom, scores in all_agreements.items()
            if scores
        }

        return {
            "status": "analyzed",
            "fabrics_count": len(self.fabrics),
            "perspectives": self.perspectives_present(),
            "consensus_scores": consensus_scores,
            "avg_epistemic_confidence": sum(f.epistemic_confidence for f in self.fabrics) / len(self.fabrics),
        }

    def find_disagreements(self) -> List[Tension]:
        """
        Find all tensions across fabrics.

        Returns:
            List of Tension objects
        """
        tensions = []
        for fabric in self.fabrics:
            fabric_tensions = fabric.metadata.get("tensions", [])
            for t in fabric_tensions:
                tensions.append(Tension.from_dict(t) if isinstance(t, dict) else t)
        return tensions

    def get_collective_blind_spots(self) -> List[str]:
        """
        Find blind spots shared by ALL perspectives.

        Only blind spots present in every fabric are truly collective.

        Returns:
            List of shared blind spots
        """
        if not self.fabrics:
            return []

        # Start with first fabric's blind spots
        common = set(self.fabrics[0].blind_spots)

        # Intersect with all other fabrics
        for fabric in self.fabrics[1:]:
            common &= set(fabric.blind_spots)

        return list(common)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core Classes
    'Fabric',
    'Tension',
    'Resolution',
    'MultiFabricCollection',
]
