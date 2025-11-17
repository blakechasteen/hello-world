"""
DreamEngine Types
=================

Data structures for memory synthesis system.

Author: HoloLoom Team
Date: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# Synthesis Types
# ============================================================================

class SynthesisType(Enum):
    """Types of synthetic memories."""
    PATTERN = "pattern_synthesis"
    CONTRADICTION = "contradiction_detection"
    GAP = "gap_identification"
    SUMMARY = "summary"


@dataclass
class SyntheticMemory:
    """A memory created by Dream Engine synthesis."""
    content: str
    provenance: str  # e.g., "pattern_synthesis"
    source_memories: List[str]  # IDs of source memories
    confidence: float  # 0.0-1.0
    created_at: datetime = field(default_factory=datetime.now)
    synthesis_type: SynthesisType = SynthesisType.PATTERN
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pattern Synthesis Types
# ============================================================================

@dataclass
class PatternCluster:
    """A cluster of related high-confidence queries."""
    queries: List[Any]  # List of query objects
    common_entities: List[str]
    common_motifs: List[str]
    semantic_centroid: Optional[Any] = None  # Embedding centroid
    avg_confidence: float = 0.0


# ============================================================================
# Contradiction Detection Types
# ============================================================================

class ContradictionType(Enum):
    """Types of contradictions."""
    FACT_UPDATE = "fact_update"  # Later fact supersedes earlier
    PREFERENCE_REVERSAL = "preference_reversal"  # User preference changed
    BELIEF_CHANGE = "belief_change"  # Worldview evolution
    CONTEXT_DEPENDENT = "context_dependent"  # Both valid in different contexts
    LOGICAL_CONFLICT = "logical_conflict"  # Mutually exclusive


@dataclass
class Contradiction:
    """A detected contradiction between two memories."""
    memory1_id: str
    memory2_id: str
    memory1_content: str
    memory2_content: str
    score: float  # 0.0-1.0 contradiction confidence
    type: ContradictionType
    temporal_gap_days: int  # Days between memories
    explanation: str = ""  # Human-readable explanation
    detected_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Gap Identification Types
# ============================================================================

class GapType(Enum):
    """Types of knowledge gaps."""
    BRIDGE_GAP = "bridge_gap"  # Mentioned but not explained
    MISSING_PREREQUISITE = "missing_prerequisite"  # Know advanced, missing basics
    INCOMPLETE_CATEGORY = "incomplete_category"  # Some items, missing others
    STALE_KNOWLEDGE = "stale_knowledge"  # Old info, new developments exist


@dataclass
class KnowledgeGap:
    """An identified gap in knowledge graph."""
    bridge_concept: str  # The mentioned but underdeveloped concept
    missing_concepts: List[str]  # Related concepts that are missing
    importance: float  # 0.0-1.0 importance score
    gap_type: GapType
    suggested_queries: List[str]  # Queries to fill the gap
    related_cluster_id: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Synthesis Results
# ============================================================================

@dataclass
class SynthesisResults:
    """Results from a complete synthesis cycle."""
    patterns_synthesized: List[SyntheticMemory]
    contradictions_detected: List[Contradiction]
    gaps_identified: List[KnowledgeGap]
    cycle_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
DreamEngine Synthesis Cycle Complete
=====================================
Timestamp: {self.timestamp.isoformat()}
Duration: {self.cycle_duration_ms:.1f}ms

Patterns Synthesized: {len(self.patterns_synthesized)}
Contradictions Detected: {len(self.contradictions_detected)}
Knowledge Gaps Identified: {len(self.gaps_identified)}

Details:
{self._format_patterns()}
{self._format_contradictions()}
{self._format_gaps()}
        """.strip()

    def _format_patterns(self) -> str:
        if not self.patterns_synthesized:
            return "  No patterns synthesized"

        lines = ["\nPatterns:"]
        for i, pattern in enumerate(self.patterns_synthesized[:5], 1):
            lines.append(f"  {i}. {pattern.content[:80]}...")
            lines.append(f"     Sources: {len(pattern.source_memories)} memories")

        if len(self.patterns_synthesized) > 5:
            lines.append(f"  ... and {len(self.patterns_synthesized) - 5} more")

        return "\n".join(lines)

    def _format_contradictions(self) -> str:
        if not self.contradictions_detected:
            return "\n  No contradictions detected"

        lines = ["\nContradictions:"]
        for i, contradiction in enumerate(self.contradictions_detected[:5], 1):
            lines.append(f"  {i}. {contradiction.type.value}")
            lines.append(f"     Earlier: {contradiction.memory1_content[:60]}...")
            lines.append(f"     Later: {contradiction.memory2_content[:60]}...")
            lines.append(f"     Gap: {contradiction.temporal_gap_days} days")

        if len(self.contradictions_detected) > 5:
            lines.append(f"  ... and {len(self.contradictions_detected) - 5} more")

        return "\n".join(lines)

    def _format_gaps(self) -> str:
        if not self.gaps_identified:
            return "\n  No knowledge gaps identified"

        lines = ["\nKnowledge Gaps:"]
        for i, gap in enumerate(self.gaps_identified[:5], 1):
            lines.append(f"  {i}. {gap.bridge_concept}")
            lines.append(f"     Missing: {', '.join(gap.missing_concepts[:3])}")
            lines.append(f"     Importance: {gap.importance:.2f}")
            lines.append(f"     Suggested: {gap.suggested_queries[0] if gap.suggested_queries else 'N/A'}")

        if len(self.gaps_identified) > 5:
            lines.append(f"  ... and {len(self.gaps_identified) - 5} more")

        return "\n".join(lines)
