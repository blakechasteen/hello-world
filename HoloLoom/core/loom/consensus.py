"""
Loom Consensus - Perspective-Aware Synthesis with Auto-Exploration
===================================================================

**Purpose**: Synthesizes multiple Fabric outputs from different looms into
a coherent collective response, detecting tensions and auto-exploring
disagreement zones.

**Date**: December 2025
**Phase**: Weave House Implementation

**Core Insight**: Disagreement is a feature, not a bug. When looms disagree,
we explore the tension rather than averaging it away. This leads to deeper
understanding and explicit handling of irreducible uncertainty.

**Philosophy**:
"True consensus isn't agreement - it's understanding where we agree,
where we disagree, and why. The tensions ARE the interesting parts."

Author: HoloLoom Architecture Team
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from HoloLoom.core.fabric.fabric import Fabric, Tension, Resolution
from HoloLoom.core.fabric.spacetime import WeavingTrace

logger = logging.getLogger(__name__)


# ============================================================================
# Claim - Extracted from Fabric Response
# ============================================================================

@dataclass
class Claim:
    """
    A discrete claim extracted from a Fabric response.

    Used for finding agreements and tensions between looms.
    """
    text: str                      # The claim itself
    source_loom: str               # Which loom made this claim
    confidence: float              # Confidence in this claim
    supporting_evidence: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # For similarity comparison

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if not isinstance(other, Claim):
            return False
        return self.text == other.text


# ============================================================================
# Agreement - Claims where looms align
# ============================================================================

@dataclass
class Agreement:
    """
    A zone of agreement between multiple looms.

    When multiple looms make similar claims, we identify the consensus.
    """
    claim: str                     # The agreed-upon claim
    supporting_looms: List[str]    # Which looms agree
    avg_confidence: float          # Average confidence across looms
    strength: float                # How strong is the agreement (0.0-1.0)


# ============================================================================
# Loom Consensus - Main Synthesis Engine
# ============================================================================

class LoomConsensus:
    """
    Perspective-aware consensus with auto-exploration.

    Takes multiple Fabrics from different looms and synthesizes them into
    a collective Fabric, tracking where looms agree and disagree.

    Key capabilities:
    - Find agreement zones (claims multiple looms make)
    - Find tensions (significant disagreements)
    - Compute collective epistemic confidence
    - Attempt resolution of tensions
    - Mark irreducible uncertainties

    Example:
        consensus = LoomConsensus(exploration_enabled=True)

        # Collect fabrics from looms
        fabrics = [recall_fabric, reason_fabric, reach_fabric]

        # Synthesize
        collective = await consensus.synthesize(fabrics)

        # Check for tensions
        if consensus.has_significant_tensions(collective, threshold=0.3):
            print("Significant tensions detected!")
            for t in collective.metadata.get('tensions', []):
                print(f"  {t['loom_a']} vs {t['loom_b']}: {t['severity']}")
    """

    def __init__(
        self,
        exploration_enabled: bool = True,
        similarity_threshold: float = 0.85,
        tension_threshold: float = 0.3,
        n_arms: int = 5  # One per core loom
    ):
        """
        Initialize consensus engine.

        Args:
            exploration_enabled: Whether to auto-explore tensions
            similarity_threshold: Similarity needed for agreement
            tension_threshold: Severity threshold for significant tensions
            n_arms: Number of Thompson Sampling arms (looms)
        """
        self.exploration_enabled = exploration_enabled
        self.similarity_threshold = similarity_threshold
        self.tension_threshold = tension_threshold

        # Thompson Sampling priors (alpha, beta per loom)
        self._thompson_priors: Dict[str, Tuple[float, float]] = {}

    # ====== Main Synthesis ======

    async def synthesize(self, fabrics: List[Fabric]) -> Fabric:
        """
        Synthesize multiple fabrics into collective fabric.

        Process:
        1. Find agreement zones (similar claims)
        2. Find tensions (disagreements)
        3. Compute collective epistemic confidence
        4. Compute collective blind spots (intersection)
        5. Collect falsifiability from all looms
        6. Build synthesis

        Args:
            fabrics: List of Fabric objects from different looms

        Returns:
            Collective Fabric with synthesis
        """
        if not fabrics:
            return self._empty_fabric()

        if len(fabrics) == 1:
            # Single fabric - just return it with collective perspective
            return self._single_fabric_synthesis(fabrics[0])

        # 1. Extract claims from each fabric
        claims_by_loom = self._extract_claims(fabrics)

        # 2. Find agreement zones
        agreements = self._find_agreements(claims_by_loom)

        # 3. Find tensions
        tensions = self._find_tensions(fabrics, claims_by_loom)

        # 4. Compute collective epistemic confidence
        epistemic = self._compute_collective_epistemic(fabrics, tensions)

        # 5. Compute collective blind spots (intersection - only what ALL looms miss)
        collective_blind_spots = self._intersect_blind_spots(fabrics)

        # 6. Collect falsifiability from all looms
        all_falsifiers = self._collect_falsifiability(fabrics)

        # 7. Merge agreed claims into response
        merged_response = self._merge_agreed_claims(agreements, fabrics)

        # 8. Build collective fabric
        trace = self._merge_traces(fabrics)

        return Fabric(
            query_text=fabrics[0].query_text,
            response=merged_response,
            tool_used="collective",
            confidence=self._weighted_confidence(fabrics),
            trace=trace,
            # Collective perspective
            perspective="collective",
            blind_spots=collective_blind_spots,
            epistemic_confidence=epistemic,
            falsifiable_by=all_falsifiers,
            admits_ignorance=self._collect_ignorance(fabrics),
            tensions_with={},  # Collective has no external tensions
            agreement_with={},
            metadata={
                "tensions": [t.to_dict() for t in tensions],
                "agreements": [
                    {"claim": a.claim, "looms": a.supporting_looms, "strength": a.strength}
                    for a in agreements
                ],
                "agreement_ratio": len(agreements) / max(len(tensions) + len(agreements), 1),
                "per_loom_confidence": {f.perspective: f.confidence for f in fabrics},
                "per_loom_epistemic": {f.perspective: f.epistemic_confidence for f in fabrics},
                "synthesis_time": datetime.utcnow().isoformat(),
            }
        )

    # ====== Tension Detection ======

    def has_significant_tensions(self, fabric: Fabric, threshold: Optional[float] = None) -> bool:
        """
        Check if fabric has significant unresolved tensions.

        Args:
            fabric: The collective fabric to check
            threshold: Severity threshold (default: self.tension_threshold)

        Returns:
            True if any tension exceeds threshold
        """
        threshold = threshold or self.tension_threshold
        tensions = fabric.metadata.get("tensions", [])
        return any(t.get("severity", 0) > threshold for t in tensions)

    def get_tensions(self, fabric: Fabric) -> List[Tension]:
        """
        Get all tensions from a collective fabric.

        Args:
            fabric: The collective fabric

        Returns:
            List of Tension objects
        """
        return [
            Tension.from_dict(t)
            for t in fabric.metadata.get("tensions", [])
        ]

    # ====== Resolution Attempts ======

    async def attempt_resolution(
        self,
        tension: Tension,
        exploration_fabrics: List[Fabric]
    ) -> Resolution:
        """
        Try to resolve a tension with deeper exploration.

        Takes fabrics from targeted exploration of the disagreement
        and attempts to find consensus.

        Args:
            tension: The tension to resolve
            exploration_fabrics: Fabrics from exploring the specific disagreement

        Returns:
            Resolution object indicating success/failure
        """
        if not exploration_fabrics:
            return Resolution(
                resolved=False,
                consensus_claim=None,
                confidence=0.0,
                supporting_looms=[],
                dissenting_looms=[tension.loom_a, tension.loom_b]
            )

        # Extract claims from exploration
        claims = [f.response for f in exploration_fabrics]
        perspectives = [f.perspective for f in exploration_fabrics]

        # Compute pairwise similarity
        similarities = self._compute_pairwise_similarity(claims)

        if similarities > 0.8:
            # Agreement found!
            return Resolution(
                resolved=True,
                consensus_claim=self._merge_claims_text(claims),
                confidence=similarities,
                supporting_looms=perspectives,
                dissenting_looms=[]
            )
        else:
            # Still disagreement - mark irreducible
            return Resolution(
                resolved=False,
                consensus_claim=None,
                confidence=similarities,
                supporting_looms=[],
                dissenting_looms=perspectives
            )

    # ====== Internal Methods ======

    def _extract_claims(
        self,
        fabrics: List[Fabric]
    ) -> Dict[str, List[Claim]]:
        """
        Extract discrete claims from each fabric's response.

        Simple heuristic: split on sentences.
        Future: Use LLM for claim extraction.
        """
        claims_by_loom: Dict[str, List[Claim]] = {}

        for fabric in fabrics:
            claims = []
            # Simple sentence split (future: better NLP)
            sentences = fabric.response.replace("!", ".").replace("?", ".").split(".")
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

            for sentence in sentences:
                claims.append(Claim(
                    text=sentence,
                    source_loom=fabric.perspective,
                    confidence=fabric.confidence,
                ))

            claims_by_loom[fabric.perspective] = claims

        return claims_by_loom

    def _find_agreements(
        self,
        claims_by_loom: Dict[str, List[Claim]]
    ) -> List[Agreement]:
        """
        Find claims where multiple looms agree.

        Uses simple text similarity (future: semantic similarity).
        """
        agreements: List[Agreement] = []
        seen_claims: Set[str] = set()

        all_claims = []
        for loom, claims in claims_by_loom.items():
            all_claims.extend(claims)

        # Compare each pair of claims
        for i, claim_i in enumerate(all_claims):
            if claim_i.text in seen_claims:
                continue

            # Find similar claims
            similar = [claim_i]
            for j, claim_j in enumerate(all_claims):
                if i != j and claim_j.source_loom != claim_i.source_loom:
                    similarity = self._text_similarity(claim_i.text, claim_j.text)
                    if similarity >= self.similarity_threshold:
                        similar.append(claim_j)

            if len(similar) > 1:
                # Found agreement
                looms = list(set(c.source_loom for c in similar))
                avg_conf = sum(c.confidence for c in similar) / len(similar)

                agreements.append(Agreement(
                    claim=claim_i.text,
                    supporting_looms=looms,
                    avg_confidence=avg_conf,
                    strength=len(looms) / len(claims_by_loom)  # Fraction of looms agreeing
                ))

                for c in similar:
                    seen_claims.add(c.text)

        return agreements

    def _find_tensions(
        self,
        fabrics: List[Fabric],
        claims_by_loom: Dict[str, List[Claim]]
    ) -> List[Tension]:
        """
        Find significant disagreements between looms.

        Looks at:
        1. Explicit tensions_with in each fabric
        2. Low agreement_with scores
        3. Contradicting claims (future: semantic contradiction detection)
        """
        tensions: List[Tension] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # 1. Collect explicit tensions
        for fabric in fabrics:
            for other_loom, description in fabric.tensions_with.items():
                pair = tuple(sorted([fabric.perspective, other_loom]))
                if pair not in seen_pairs:
                    tensions.append(Tension(
                        loom_a=fabric.perspective,
                        loom_b=other_loom,
                        claim_a=fabric.response[:100],
                        claim_b=description,
                        severity=0.5,  # Default severity
                    ))
                    seen_pairs.add(pair)

        # 2. Check agreement_with for low scores
        for fabric in fabrics:
            for other_loom, agreement_score in fabric.agreement_with.items():
                if agreement_score < 0.3:  # Low agreement = potential tension
                    pair = tuple(sorted([fabric.perspective, other_loom]))
                    if pair not in seen_pairs:
                        # Find the other fabric
                        other_fabric = next(
                            (f for f in fabrics if f.perspective == other_loom),
                            None
                        )
                        if other_fabric:
                            tensions.append(Tension(
                                loom_a=fabric.perspective,
                                loom_b=other_loom,
                                claim_a=fabric.response[:100],
                                claim_b=other_fabric.response[:100],
                                severity=1.0 - agreement_score,
                            ))
                            seen_pairs.add(pair)

        # 3. Check for contradicting claims (simple heuristic)
        looms = list(claims_by_loom.keys())
        for i, loom_a in enumerate(looms):
            for loom_b in looms[i + 1:]:
                pair = tuple(sorted([loom_a, loom_b]))
                if pair in seen_pairs:
                    continue

                for claim_a in claims_by_loom[loom_a]:
                    for claim_b in claims_by_loom[loom_b]:
                        if self._appears_contradictory(claim_a.text, claim_b.text):
                            tensions.append(Tension(
                                loom_a=loom_a,
                                loom_b=loom_b,
                                claim_a=claim_a.text,
                                claim_b=claim_b.text,
                                severity=0.6,
                            ))
                            seen_pairs.add(pair)
                            break
                    else:
                        continue
                    break

        return tensions

    def _compute_collective_epistemic(
        self,
        fabrics: List[Fabric],
        tensions: List[Tension]
    ) -> float:
        """
        Compute collective epistemic confidence.

        Lower when:
        - There are significant tensions
        - Individual epistemic confidence is low
        - Looms admit significant ignorance

        Higher when:
        - Strong agreement across looms
        - High individual epistemic confidence
        """
        if not fabrics:
            return 0.0

        # Base: average of individual epistemic confidences
        avg_epistemic = sum(f.epistemic_confidence for f in fabrics) / len(fabrics)

        # Penalty for tensions
        if tensions:
            avg_severity = sum(t.severity for t in tensions) / len(tensions)
            tension_penalty = avg_severity * 0.3  # Up to 30% penalty
        else:
            tension_penalty = 0.0

        # Penalty for admitted ignorance
        total_ignorance = sum(len(f.admits_ignorance) for f in fabrics)
        ignorance_penalty = min(total_ignorance * 0.02, 0.2)  # Up to 20% penalty

        return max(0.0, min(1.0, avg_epistemic - tension_penalty - ignorance_penalty))

    def _intersect_blind_spots(self, fabrics: List[Fabric]) -> List[str]:
        """
        Find blind spots shared by ALL looms.

        Only what every loom cannot see remains as collective blind spot.
        Diverse perspectives should eliminate most blind spots.
        """
        if not fabrics:
            return []

        common = set(fabrics[0].blind_spots)
        for fabric in fabrics[1:]:
            common &= set(fabric.blind_spots)

        return list(common)

    def _collect_falsifiability(self, fabrics: List[Fabric]) -> List[str]:
        """
        Collect all falsifiers from all looms.

        Union of what would change each loom's mind.
        """
        all_falsifiers: Set[str] = set()
        for fabric in fabrics:
            all_falsifiers.update(fabric.falsifiable_by)
        return list(all_falsifiers)

    def _collect_ignorance(self, fabrics: List[Fabric]) -> List[str]:
        """
        Collect admitted ignorance from all looms.

        Tracks what different looms know they don't know.
        """
        all_ignorance: Set[str] = set()
        for fabric in fabrics:
            all_ignorance.update(fabric.admits_ignorance)
        return list(all_ignorance)

    def _weighted_confidence(self, fabrics: List[Fabric]) -> float:
        """
        Compute weighted average confidence.

        Weights by epistemic confidence (more certain looms count more).
        """
        if not fabrics:
            return 0.0

        total_weight = sum(f.epistemic_confidence for f in fabrics)
        if total_weight == 0:
            return sum(f.confidence for f in fabrics) / len(fabrics)

        weighted_sum = sum(
            f.confidence * f.epistemic_confidence
            for f in fabrics
        )
        return weighted_sum / total_weight

    def _merge_agreed_claims(
        self,
        agreements: List[Agreement],
        fabrics: List[Fabric]
    ) -> str:
        """
        Merge agreed-upon claims into response.

        Prioritizes claims with higher agreement strength.
        Falls back to highest-confidence fabric if no agreements.
        """
        if not agreements:
            # No agreements - use response from highest-confidence fabric
            if fabrics:
                best = max(fabrics, key=lambda f: f.confidence * f.epistemic_confidence)
                return best.response
            return ""

        # Sort by strength and take top claims
        sorted_agreements = sorted(agreements, key=lambda a: a.strength, reverse=True)
        merged = ". ".join(a.claim for a in sorted_agreements[:5])

        return merged

    def _merge_traces(self, fabrics: List[Fabric]) -> WeavingTrace:
        """
        Merge traces from multiple fabrics.

        Creates collective trace showing all loom contributions.
        """
        if not fabrics:
            return WeavingTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=0.0
            )

        # Use earliest start, latest end
        start = min(f.trace.start_time for f in fabrics)
        end = max(f.trace.end_time for f in fabrics)
        total_duration = sum(f.trace.duration_ms for f in fabrics)

        # Merge motifs
        all_motifs = []
        for f in fabrics:
            all_motifs.extend(f.trace.motifs_detected)

        return WeavingTrace(
            start_time=start,
            end_time=end,
            duration_ms=total_duration,
            motifs_detected=list(set(all_motifs)),
            stage_durations={"collective_synthesis": total_duration},
        )

    def _empty_fabric(self) -> Fabric:
        """Create empty collective fabric."""
        now = datetime.now()
        trace = WeavingTrace(
            start_time=now,
            end_time=now,
            duration_ms=0.0
        )
        return Fabric(
            query_text="",
            response="No fabrics to synthesize.",
            tool_used="collective",
            confidence=0.0,
            trace=trace,
            perspective="collective",
            epistemic_confidence=0.0,
        )

    def _single_fabric_synthesis(self, fabric: Fabric) -> Fabric:
        """
        Wrap single fabric as collective.

        Just changes perspective to "collective" and notes single source.
        """
        return Fabric(
            query_text=fabric.query_text,
            response=fabric.response,
            tool_used=fabric.tool_used,
            confidence=fabric.confidence,
            trace=fabric.trace,
            perspective="collective",
            blind_spots=fabric.blind_spots,  # Keep single loom's blind spots
            epistemic_confidence=fabric.epistemic_confidence,
            falsifiable_by=fabric.falsifiable_by,
            admits_ignorance=fabric.admits_ignorance,
            metadata={
                "single_source": fabric.perspective,
                "note": "Single perspective synthesis",
            }
        )

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute text similarity (simple word overlap).

        Future: Use embeddings for semantic similarity.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _appears_contradictory(self, claim_a: str, claim_b: str) -> bool:
        """
        Check if two claims appear contradictory.

        Simple heuristic: checks for negation patterns.
        Future: Use LLM for contradiction detection.
        """
        negation_words = {"not", "no", "never", "neither", "nor", "nothing", "none"}

        words_a = set(claim_a.lower().split())
        words_b = set(claim_b.lower().split())

        # Check if one contains negation of similar words
        a_has_negation = bool(words_a & negation_words)
        b_has_negation = bool(words_b & negation_words)

        # Simple heuristic: similar content but different negation
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

        if overlap > 0.3 and a_has_negation != b_has_negation:
            return True

        return False

    def _compute_pairwise_similarity(self, claims: List[str]) -> float:
        """
        Compute average pairwise similarity between claims.
        """
        if len(claims) < 2:
            return 1.0

        total = 0.0
        count = 0

        for i, claim_a in enumerate(claims):
            for claim_b in claims[i + 1:]:
                total += self._text_similarity(claim_a, claim_b)
                count += 1

        return total / count if count > 0 else 0.0

    def _merge_claims_text(self, claims: List[str]) -> str:
        """
        Merge multiple claims into consensus text.

        Takes common elements from all claims.
        """
        if not claims:
            return ""

        if len(claims) == 1:
            return claims[0]

        # Simple merge: take first claim and note synthesis
        return claims[0] + " [synthesized from multiple perspectives]"


# ============================================================================
# Factory Function
# ============================================================================

def create_loom_consensus(
    exploration_enabled: bool = True,
    tension_threshold: float = 0.3
) -> LoomConsensus:
    """
    Factory function for creating LoomConsensus.

    Args:
        exploration_enabled: Whether to auto-explore tensions
        tension_threshold: Severity threshold for significant tensions

    Returns:
        Configured LoomConsensus instance
    """
    return LoomConsensus(
        exploration_enabled=exploration_enabled,
        tension_threshold=tension_threshold
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core Classes
    'LoomConsensus',
    'Claim',
    'Agreement',

    # Factory
    'create_loom_consensus',
]
